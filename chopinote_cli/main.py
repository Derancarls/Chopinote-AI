"""
Chopinote-AI CLI: 钢琴谱续写命令行工具

用法:
    chopin best.pt input.musicxml
    chopin best.pt input.musicxml --temp 1.2 --top-k 40 --seed 42
    chopin best.pt input.musicxml -n 3
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

# 添加项目根目录到 sys.path（兼容 pip install -e 和直接运行）
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from chopinote_model.config import ModelConfig
from chopinote_model.model import MusicTransformer
from chopinote_model.generate import GenerationParams
from chopinote_dataset.tokenizer import REMITokenizer
from chopinote_dataset.converter import MusicXMLToREMI
from chopinote_cli.presets import Preset, get_preset, list_presets as _list_presets
from chopinote_cli.config import load_config, find_config
from chopinote_model.auto_config import (
    detect_system, suggest_inference, print_hardware_report,
)

logger = logging.getLogger(__name__)


# -- 模型加载 --------------------------------------------------

def load_model(checkpoint_path: str, device: torch.device, infer_cfg=None):
    """从 checkpoint 加载模型，返回 (model, config, step, loss).
    自动适应 checkpoint 的 vocab_size，无需手动对齐 ModelConfig。
    infer_cfg: InferenceConfig | None，自动应用最优推理设置。
    """
    if not os.path.isfile(checkpoint_path):
        print(f'  [X] checkpoint 文件不存在: {checkpoint_path}')
        sys.exit(1)

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # 重建配置（优先用 checkpoint 中的值）
    saved_config = ckpt.get('config')
    if isinstance(saved_config, dict):
        config = ModelConfig(**saved_config)
    elif saved_config is not None:
        config = saved_config
    else:
        config = ModelConfig()

    model = MusicTransformer(config)
    state_dict = ckpt['model_state_dict']
    model_state = model.state_dict()

    # 只加载 shape 完全匹配的参数
    loaded = 0
    skipped = []
    for k, v in state_dict.items():
        if k in model_state and v.shape == model_state[k].shape:
            model_state[k] = v
            loaded += 1
        else:
            skipped.append(
                f'{k}: ckpt {tuple(v.shape)} vs model '
                f'{tuple(model_state[k].shape) if k in model_state else "N/A"}'
            )

    model.load_state_dict(model_state)

    if skipped:
        print(f'      [!] 跳过了 {len(skipped)} 个 shape 不匹配的参数:')
        for s in skipped[:5]:
            print(f'         - {s}')
        if len(skipped) > 5:
            print(f'         - ... 还有 {len(skipped) - 5} 个')

    model.to(device)

    # ── 硬件自适应优化 ──
    if infer_cfg:
        # TF32 matmul
        if infer_cfg.use_tf32:
            torch.set_float32_matmul_precision('high')
        # 显存上限
        if device.type == 'cuda' and infer_cfg.memory_fraction < 1.0:
            try:
                torch.cuda.set_per_process_memory_fraction(infer_cfg.memory_fraction)
            except Exception:
                pass
        # 精度转换
        if infer_cfg.dtype == 'bf16':
            model = model.bfloat16()
        elif infer_cfg.dtype == 'fp16':
            model = model.half()
        # fp8: 不转 weights，由 FP8Linear 内部量化
    # ────────────────────────────────────────

    model.eval()

    # FP8 模式（需在 eval 后激活）
    if infer_cfg and infer_cfg.dtype == 'fp8':
        try:
            model.set_fp8_mode(True)
        except Exception:
            pass

    # torch.compile（最后一步，包裹模型）
    if infer_cfg and infer_cfg.torch_compile:
        try:
            model = torch.compile(model, mode='reduce-overhead')
        except Exception:
            pass

    step = ckpt.get('step', 0)
    loss = ckpt.get('loss', None)
    return model, config, step, loss


# -- MusicXML → seed tokens ----------------------------------

def musicxml_to_seed(file_path: str, tokenizer: REMITokenizer,
                     model_vocab_size: int) -> tuple[list[int], dict]:
    """解析 MusicXML 为种子 token 序列。
    自动将超出模型词表的 token 替换为 MASK。
    返回 (seed_tokens, metadata)。
    """
    if not os.path.isfile(file_path):
        print(f'[X] 输入文件不存在: {file_path}')
        sys.exit(1)

    conv = MusicXMLToREMI(grid_size=tokenizer.grid_size,
                          velocity_levels=tokenizer.velocity_levels)
    tokens, metadata = conv.convert(file_path, collect_metadata=True)

    if not tokens:
        print('  [!] 未能从文件中解析出有效音符，将从头生成')
        return [tokenizer.bos_token_id], metadata

    # 截断超出模型词表的 token（兼容旧 checkpoint）
    oob_count = sum(1 for t in tokens if t >= model_vocab_size)
    if oob_count:
        mask_id = tokenizer.encode_token(tokenizer.MASK)
        tokens = [t if t < model_vocab_size else mask_id for t in tokens]
        print(f'      [!] {oob_count} 个 token 超出模型词表 (vocab_size={model_vocab_size})，已替换为 MASK')

    return tokens, metadata


def _extract_intro_info(tokens: list[int], tokenizer: REMITokenizer) -> dict:
    """从 token 序列前段提取曲谱基本信息（调性、拍号、速度）。"""
    info = {}
    for tid in tokens[:40]:
        token = tokenizer.decode_token(tid)
        if token.startswith(REMITokenizer.TONIC):
            info['key'] = token[len(REMITokenizer.TONIC) + 1:-1]
        elif token.startswith(REMITokenizer.TIMESIG):
            info['time_sig'] = token[len(REMITokenizer.TIMESIG) + 1:-1]
        elif token.startswith(REMITokenizer.TEMPO):
            info['tempo'] = int(token[len(REMITokenizer.TEMPO) + 1:-1])
    return info


# -- 复杂度控制 -----------------------------------------------

def _estimate_seed_complexity(seed_tokens: torch.Tensor, tokenizer) -> float:
    """从 seed token 序列估算自然复杂度 (0-10)。"""
    events = tokenizer.detokenize(seed_tokens[0].tolist())
    bar_count = 0
    note_count = 0
    rest_count = 0
    short_durs = 0
    total_durs = 0
    for etype, evalue in events:
        if etype == tokenizer.BAR:
            bar_count += 1
        elif etype == tokenizer.NOTE_ON:
            note_count += 1
        elif etype == tokenizer.DURATION:
            total_durs += 1
            if evalue <= 4:
                short_durs += 1
        elif etype == tokenizer.REST:
            rest_count += 1

    if bar_count == 0:
        return 5.0

    density = note_count / bar_count  # notes/bar
    density_score = min(10.0, density / 4.0)

    rest_ratio = rest_count / max(1, note_count + rest_count)
    rest_score = (1.0 - rest_ratio) * 10.0

    short_ratio = short_durs / max(1, total_durs)
    dur_score = short_ratio * 10.0

    score = density_score * 0.5 + rest_score * 0.3 + dur_score * 0.2
    return max(0.0, min(10.0, score))

def _apply_complexity(logits, complexity: float, tokenizer):
    """根据复杂度值 (0-10) 修改 logits，控制音乐密度和丰富程度。
    偏置值为加法，叠加到原始 logits 上。
    """
    if abs(complexity - 5.0) < 1e-6:
        return  # 默认值不做任何修改

    vocab_dim = logits.size(-1)
    c = max(0.0, min(1.0, complexity / 10.0))

    # ── 1. Duration 偏置 ─────────────────────────────────
    # low C → 长时值（稀疏）, high C → 短时值（密集）
    dur_ids = [
        tokenizer.encode_token(f'<Duration {d}>')
        for d in range(1, tokenizer.grid_size + 1)
    ]
    # 短时值 (1-4): high C → +bias, low C → -bias
    short_bias = (c - 0.5) * 6.0
    for i in range(0, 4):
        if dur_ids[i] < vocab_dim:
            logits[0, dur_ids[i]] += short_bias
    # 长时值 (9-16): 反向
    long_bias = -short_bias
    for i in range(8, tokenizer.grid_size):
        if dur_ids[i] < vocab_dim:
            logits[0, dur_ids[i]] += long_bias

    # ── 2. Rest 偏置 ─────────────────────────────────────
    rest_id = tokenizer.encode_token('<Rest>')
    rest_bias = (0.5 - c) * 4.0  # low C → +bias（更多休止）
    if rest_id < vocab_dim:
        logits[0, rest_id] += rest_bias

    # ── 3. Velocity 偏置 ─────────────────────────────────
    vel_ids = [
        tokenizer.encode_token(f'<Velocity {v}>')
        for v in range(tokenizer.velocity_levels)
    ]
    for v, tid in enumerate(vel_ids):
        if tid >= vocab_dim:
            continue
        if v in (3, 4):  # 中间力度：low C 倾向这里
            logits[0, tid] += (0.5 - c) * 2.0
        elif v <= 1 or v >= 6:  # 极端力度：high C 倾向这里
            logits[0, tid] += (c - 0.5) * 4.0
        else:
            logits[0, tid] += (c - 0.5) * 1.5

    # ── 4. 连音/倚音/装饰音 硬开关 ───────────────────────
    if complexity < 3:
        for prefix, values in [
            ('<TupletStart', REMITokenizer.TUPLET_RATIOS),
            ('<GraceNote', ('acciaccatura', 'appoggiatura', 'grace')),
            ('<Ornament', ('trill', 'mordent', 'turn', 'tremolo')),
        ]:
            for val in values:
                tid = tokenizer.encode_token(f'{prefix} {val}>')
                if tid < vocab_dim:
                    logits[0, tid] = float('-inf')

    # ── 5. 演奏法/表情 偏置 ───────────────────────────────
    expr_bias = (c - 0.5) * 4.0  # low C → -bias, high C → +bias
    for prefix, values in [
        ('<Artic', ('staccato', 'accent', 'tenuto', 'marcato', 'pizzicato', 'fermata')),
        ('<Dynamic', ('ppp', 'pp', 'p', 'mp', 'mf', 'f', 'ff', 'fff', 'sfz', 'fp')),
        ('<Hairpin', ('cresc', 'dim')),
        ('<Slur', ('start', 'end')),
        ('<Pedal', ('start', 'end')),
    ]:
        for val in values:
            tid = tokenizer.encode_token(f'{prefix} {val}>')
            if tid < vocab_dim:
                logits[0, tid] += expr_bias


# -- 带进度条的生成 --------------------------------------------

@torch.no_grad()
def generate_with_progress(
    model: MusicTransformer,
    seed_tokens: torch.Tensor,
    tokenizer: REMITokenizer,
    max_bars: int = 32,
    max_new_tokens: int = 4096,
    temperature: float = 1.0,
    top_k: int = 20,
    effective_vocab_size: Optional[int] = None,
    lock_key: bool = True,
    lock_time: bool = True,
    lock_tempo: bool = True,
    lock_program: bool = True,
    complexity: Optional[float] = None,
    rest_penalty: float = 0.0,
    max_polyphony: int = 10,
    key_bias_strength: float = 2.0,
    prog_switch_strength: float = 1.0,
    prog_switch_interval: int = 12,
    feedback_callback=None,
    gen_params: Optional[GenerationParams] = None,
    harmony_guide: Optional[list[int]] = None,
    rollback_from_token: Optional[int] = None,
) -> tuple[torch.Tensor, dict]:
    """自回归生成，带 tqdm 进度条。

    Args:
        model: 训练好的模型
        seed_tokens: (1, T) 种子 token 序列
        tokenizer: REMI tokenizer
        max_bars: 最多生成多少个小节
        max_new_tokens: 最多生成多少 token
        temperature: 采样温度
        top_k: top-k 采样
        effective_vocab_size: 若指定，屏蔽 ≥ 此值的 logits（兼容 vocab 不匹配）
        lock_key: 禁止模型变调（屏蔽 Key token）
        lock_time: 禁止模型变拍号（屏蔽 TimeSig token）
        lock_tempo: 禁止模型变速度（屏蔽 Tempo token）
        lock_program: 只允许 seed 已有的乐器（防多轨污染）
        complexity: 音乐密度 0-10
        rest_penalty: Rest token 负偏置（越大休止越少）
        max_polyphony: 全局同粒度最大同时发音数（配合乐器级 INSTRUMENT_POLYPHONY_CAP 使用）

    Returns:
        (1, T_total) 完整生成序列, stats dict
    """
    device = seed_tokens.device
    eos_id = tokenizer.eos_token_id
    bar_id = tokenizer.bar_token_id
    start_time = time.time()

    # ── 退回重写：截断到指定位置 ─────────────────────────────
    if rollback_from_token is not None:
        truncate_pos = min(rollback_from_token, seed_tokens.size(1) - 1)
        seed_tokens = seed_tokens[:, :truncate_pos]
        logger.info("退回重写: 截断至 token %d (共 %d tokens)", truncate_pos, seed_tokens.size(1))
    # ─────────────────────────────────────────────────────────

    # ── 音高限制准备 ──────────────────────────────────────────
    from chopinote_model.generate import (
        GM_INSTRUMENT_RANGES, SUBTRACK_RANGES, KEY_TO_DIATONIC_PITCHES,
        _parse_program, _parse_subtrack, get_polyphony_cap,
    )
    from chopinote_dataset.tokenizer import key_name_to_tonic_midi
    _OFFSET = 60
    # NOTE_ON tokens 从绝对音高改为半音程（-60 .. +60）
    note_on_ids = [tokenizer.encode_token(f'<Note_ON {i - _OFFSET}>')
                   for i in range(121)]
    _prog_prefix = '<Program'
    _pos_prefix = '<Position'
    rest_id = tokenizer.encode_token('<Rest>')
    # ──────────────────────────────────────────────────────────

    # 当前 program/subtrack（从 seed 最后一条 Program 出发搜索）
    cur_program: int | None = None
    cur_subtrack: int = 0
    for tid in reversed(seed_tokens[0].tolist()):
        ts = tokenizer.decode_token(tid)
        if ts.startswith(_prog_prefix):
            cur_program = _parse_program(ts)
            cur_subtrack = _parse_subtrack(ts)
            break

    # ── Lock: 收集 seed 已有的 Program token ────────────────
    allowed_prog_tids: set[int] = set()
    if lock_program:
        for tid in seed_tokens[0].tolist():
            ts = tokenizer.decode_token(tid)
            if ts.startswith(_prog_prefix):
                allowed_prog_tids.add(tid)

    # ── 程序切换追踪 ──────────────────────────
    seed_program_pairs: list[tuple[int, int]] = []
    program_note_counts: dict[tuple[int, int], int] = {}
    for tid in allowed_prog_tids:
        ts = tokenizer.decode_token(tid)
        prog = _parse_program(ts)
        sub = _parse_subtrack(ts)
        pair = (prog, sub)
        if pair not in program_note_counts:
            program_note_counts[pair] = 0
            seed_program_pairs.append(pair)
    notes_since_last_switch: int = 0
    # ───────────────────────────────────────────
    # ─────────────────────────────────────────────────────────

    # ── 预计算所有 Program token 的 ID ──────────────────────
    all_prog_tids: list[int] = []
    for tid in range(tokenizer.vocab_size):
        ts = tokenizer.decode_token(tid)
        if ts.startswith(_prog_prefix):
            all_prog_tids.append(tid)
    # ─────────────────────────────────────────────────────────

    # ── 锁定 token ID 预计算 ──────────────────────────────
    _lock_ids = []
    if lock_key:
        _lock_ids.extend(
            tokenizer.encode_token(f'{tokenizer.TONIC} {k}>') for k in REMITokenizer.TONIC_NAMES
        )
    if lock_time:
        _lock_ids.extend(
            tokenizer.encode_token(f'<TimeSig {ts}>') for ts in REMITokenizer.TIME_SIGNATURES
        )
    if lock_tempo:
        _lock_ids.extend(
            tokenizer.encode_token(f'<Tempo {bpm}>') for bpm in range(30, 241, 10)
        )
    # ─────────────────────────────────────────────────────

    # ── 多音 count 跟踪（同 Position 内每 track 的 NOTE_ON 数） ───
    cur_position = -1
    notes_this_pos: dict[tuple[int, int], int] = {}  # (program, subtrack) -> count

    # ── 调性跟踪（从 seed 反向扫描获取当前调性） ──────
    current_key: str | None = None
    current_tonic_midi: int = 60
    for tid in reversed(seed_tokens[0].tolist()):
        ts = tokenizer.decode_token(tid)
        if ts.startswith('<Key'):
            current_key = ts[len('<Key') + 1:-1]  # 如 'C', 'Am', 'F#'
            current_tonic_midi = key_name_to_tonic_midi(current_key)
            break

    # ── seed 复杂度基线 ───────────────────────────
    seed_complexity = _estimate_seed_complexity(seed_tokens, tokenizer)
    if complexity is None:
        complexity = seed_complexity
    # 相对调整：用户选的 0-10 相对于 5，映射到 seed 的基线
    adjusted_complexity = complexity + (seed_complexity - 5.0)
    adjusted_complexity = max(0.0, min(10.0, adjusted_complexity))
    # ──────────────────────────────────────────────

    # KV cache 初始化
    kv_caches = [[None, None] for _ in range(model.config.n_layers)]

    generated = seed_tokens.clone()
    next_token = seed_tokens
    bar_count = 0
    cached_measure_ids = torch.cumsum((seed_tokens[0] == bar_id).int(), dim=0)
    seed_measure_count = cached_measure_ids[-1].item()  # seed 末位置的累计小节数

    # 种子过长时截断（只保留最后 max_seq_len 个 token）
    max_len = model.config.max_seq_len
    if generated.size(1) > max_len:
        trim = generated.size(1) - max_len
        print(f'  [!] 种子长度 {generated.size(1)} > max_seq_len {max_len}，截断前 {trim} 个 token')
        next_token = generated[:, -max_len:]
        generated = generated[:, -max_len:]
        cached_measure_ids = cached_measure_ids[-max_len:]
        seed_measure_count = cached_measure_ids[-1].item()

    from chopinote_cli.remi_grammar import patch_token_sequence

    pbar = tqdm(total=max_bars, desc='生成中', unit='bar', ncols=80)

    for step in range(max_new_tokens):
        # 首轮预填充所有 seed token，后续单 token 解码（依靠 KV cache）
        if step > 0:
            next_token = generated[:, -1:]

        logits = model.forward(next_token, kv_caches=kv_caches, measure_ids=cached_measure_ids)
        logits = logits[:, -1, :] / temperature  # (1, V)

        # ── 复杂度控制 ──────────────────────────────────
        _apply_complexity(logits, adjusted_complexity, tokenizer)
        # ──────────────────────────────────────────────────

        # ── Rest 惩罚 ─────────────────────────────────────
        if rest_penalty > 0 and rest_id < logits.size(-1):
            logits[0, rest_id] -= rest_penalty
        # ──────────────────────────────────────────────────

        # ── 词表截断 ──────────────────────────────────────
        if effective_vocab_size is not None:
            logits[0, effective_vocab_size:] = float('-inf')

        # ── 锁定屏蔽（Key/TimeSig/Tempo） ────────────────
        vocab_dim = logits.size(-1)
        for tid in _lock_ids:
            if tid < vocab_dim:
                logits[0, tid] = float('-inf')
        # ──────────────────────────────────────────────────

        # ── Program 锁定（只允许 seed 已有的乐器） ────────
        if lock_program:
            for tid in all_prog_tids:
                if tid < vocab_dim and tid not in allowed_prog_tids:
                    logits[0, tid] = float('-inf')
        # ──────────────────────────────────────────────────

        # ── 程序切换促进 ──────────────────────────
        if lock_program and prog_switch_strength > 0:
            if notes_since_last_switch >= prog_switch_interval:
                total_notes = sum(program_note_counts.values())
                n_progs = len(seed_program_pairs)
                avg_notes = total_notes / max(1, n_progs)
                max_urge = 3.0
                urge_ratio = min(1.0, (notes_since_last_switch - prog_switch_interval) / max(1, prog_switch_interval))
                urgency = urge_ratio * prog_switch_strength * max_urge

                for tid in allowed_prog_tids:
                    if tid >= vocab_dim:
                        continue
                    ts = tokenizer.decode_token(tid)
                    prog = _parse_program(ts)
                    sub = _parse_subtrack(ts)
                    pair_key = (prog, sub)
                    neglect = 0.0
                    if total_notes > 0 and avg_notes > 0:
                        this_n = program_note_counts.get(pair_key, 0)
                        neglect = max(0.0, (avg_notes - this_n) / avg_notes) * prog_switch_strength
                    is_current = (prog == cur_program and sub == cur_subtrack)
                    boost = urgency * (0.3 if is_current else 1.0) + neglect
                    if boost > 0:
                        logits[0, tid] += boost
        # ───────────────────────────────────────────

        # ── 音高限制（subtrack 感知，interval 空间） ──────────
        if cur_program is not None and cur_program < 112:
            sub_range = SUBTRACK_RANGES.get(cur_program, {}).get(cur_subtrack)
            if sub_range is not None:
                rmin, rmax = sub_range
            else:
                rmin, rmax = GM_INSTRUMENT_RANGES.get(cur_program, (0, 127))
            for i in range(121):
                abs_pitch = current_tonic_midi + (i - _OFFSET)
                if abs_pitch < rmin or abs_pitch > rmax:
                    logits[0, note_on_ids[i]] = float('-inf')
        # ──────────────────────────────────────────────────

        # ── 调性音高偏置（interval pitch class 空间） ────────
        if key_bias_strength > 0 and current_key is not None:
            diatonic_pcs = KEY_TO_DIATONIC_PITCHES.get(current_key)
            if diatonic_pcs is not None:
                tonic_pc = current_tonic_midi % 12
                for i in range(121):
                    abs_pc = (tonic_pc + (i - _OFFSET)) % 12
                    tid = note_on_ids[i]
                    if tid < vocab_dim:
                        if abs_pc in diatonic_pcs:
                            logits[0, tid] += key_bias_strength * 1.0
                        else:
                            logits[0, tid] -= key_bias_strength * 0.5
        # ───────────────────────────────────────────

        # ── 多音限制（per-track 乐器级上限） ─────────────────
        if max_polyphony > 0:
            track_key = (cur_program, cur_subtrack)
            track_count = notes_this_pos.get(track_key, 0)
            inst_cap = get_polyphony_cap(cur_program) if cur_program is not None else max_polyphony
            effective_cap = min(inst_cap, max_polyphony)
            if track_count >= effective_cap:
                for pid in note_on_ids:
                    if pid < vocab_dim:
                        logits[0, pid] = float('-inf')
        # ──────────────────────────────────────────────────

        if top_k > 0:
            k = min(top_k, logits.size(-1))
            vals, _ = torch.topk(logits, k, dim=-1)
            logits[logits < vals[:, -1:]] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

        generated = torch.cat([generated, next_token], dim=1)
        token_id = next_token.item()

        # 追踪 Program / Position / NOTE_ON
        ts = tokenizer.decode_token(token_id)
        if ts.startswith('<Key'):
            current_key = ts[len('<Key') + 1:-1]
            current_tonic_midi = key_name_to_tonic_midi(current_key)
        if ts.startswith(_prog_prefix):
            cur_program = _parse_program(ts)
            cur_subtrack = _parse_subtrack(ts)
            notes_since_last_switch = 0
        elif ts.startswith(_pos_prefix):
            cur_position = int(ts[len(_pos_prefix) + 1:-1])
            notes_this_pos.clear()
        elif ts.startswith('<Note_ON') and max_polyphony > 0:
            track_key = (cur_program, cur_subtrack)
            track_count = notes_this_pos.get(track_key, 0)
            inst_cap = get_polyphony_cap(cur_program) if cur_program is not None else max_polyphony
            effective_cap = min(inst_cap, max_polyphony)
            if track_count < effective_cap:
                notes_this_pos[track_key] = track_count + 1
            notes_since_last_switch += 1

        is_bar = (token_id == bar_id)
        new_measure = seed_measure_count + bar_count + (1 if is_bar else 0)
        cached_measure_ids = torch.cat(
            [cached_measure_ids, torch.tensor([new_measure], device=device)]
        )

        if is_bar:
            bar_count += 1
            pbar.update(1)

            # ── 和声引导（测试用）：每 bar 开头注入 chord token ──
            if harmony_guide and bar_count - 1 < len(harmony_guide):
                chord_id = harmony_guide[bar_count - 1]
                if chord_id < logits.size(-1):
                    chord_t = torch.tensor([[chord_id]], dtype=torch.long, device=device)
                    generated = torch.cat([generated, chord_t], dim=1)
                    new_measure2 = seed_measure_count + bar_count
                    cached_measure_ids = torch.cat(
                        [cached_measure_ids,
                         torch.tensor([new_measure2], device=device)]
                    )
            # ────────────────────────────────────────────────

            # ── ABC Engine B 层干预（每小节后调参） ──────
            if feedback_callback is not None and gen_params is not None:
                full_list = generated[0].tolist()
                adjustments = feedback_callback(full_list, bar_count, gen_params)
                if adjustments:
                    gen_params.apply_adjustments(adjustments)
                    temperature = gen_params.temperature
                    rest_penalty = gen_params.rest_penalty
                    key_bias_strength = gen_params.key_bias_strength
                    if 'complexity' in adjustments:
                        complexity = gen_params.complexity
                        seed_complexity = _estimate_seed_complexity(seed_tokens, tokenizer)
                        adjusted_complexity = complexity + (seed_complexity - 5.0)
                        adjusted_complexity = max(0.0, min(10.0, adjusted_complexity))
            # ─────────────────────────────────────────────

            if bar_count >= max_bars:
                break
        else:
            pbar.set_postfix(tokens=step + 1, refresh=False)

        if token_id == eos_id:
            break

    pbar.close()
    elapsed = time.time() - start_time

    # ── 生成后语法补全 ────────────────────────────────
    raw_tokens = generated[0].tolist()
    patched_tokens = patch_token_sequence(raw_tokens, tokenizer)
    if len(patched_tokens) != len(raw_tokens):
        logger.info(f'Grammar patch: {len(raw_tokens)} → {len(patched_tokens)} tokens '
                    f'(+{len(patched_tokens) - len(raw_tokens)})')

    new_tokens = step + 1
    result = torch.tensor([patched_tokens], dtype=torch.long, device=device)
    stats = {
        'bars': bar_count,
        'new_tokens': new_tokens,
        'total_tokens': result.size(1),
        'time_seconds': elapsed,
        'tokens_per_sec': new_tokens / elapsed if elapsed > 0 else 0,
    }
    return result, stats


# -- token → MusicXML 导出 ------------------------------------

def save_to_musicxml(
    token_ids: list[int],
    tokenizer: REMITokenizer,
    output_path: str,
    max_bars: int = 256,
    save_tokens: bool = False,
    export_midi: bool = True,
    fast_path: bool = False,
):
    """将 token 序列保存为 MusicXML + MIDI 文件。

    Args:
        fast_path: If True, write MusicXML directly without music21 (~100x faster).
                   Does NOT support MIDI export. Use for C evaluation review.
    """
    from chopinote_dataset.renderer import REMIToMusicXML

    renderer = REMIToMusicXML(grid_size=tokenizer.grid_size,
                               velocity_levels=tokenizer.velocity_levels)
    score = renderer.render_from_tokens(token_ids, output_path, fast_path=fast_path)

    # 保存 token 序列（用于退回重写和 DPO）
    if save_tokens and token_ids:
        tok_path = output_path.rsplit('.musicxml', 1)[0] + '.tokens'
        try:
            with open(tok_path, 'w', encoding='utf-8') as f:
                f.write(' '.join(str(t) for t in token_ids))
        except OSError as e:
            logger.warning('保存 token 文件失败: %s', e)

    # 自动导出 MIDI（快速试听反馈）— fast_path 跳过
    if export_midi and score is not None:
        midi_path = output_path.rsplit('.musicxml', 1)[0] + '.mid'
        try:
            score.write('midi', midi_path)
        except Exception as e:
            logger.warning('MIDI 导出失败: %s', e)

    # 计算小节数（fast_path 返回 None，由调用方自行计算）
    if score is None:
        return 0
    parts = list(score.parts)
    if parts:
        num_measures = len(list(parts[0].getElementsByClass('Measure')))
    else:
        num_measures = 0
    return num_measures


# -- token 文件读写 -----------------------------------------------

def load_tokens_file(tokens_path: str) -> list[int] | None:
    """从 .tokens 文件加载 token ID 列表。"""
    try:
        with open(tokens_path, encoding='utf-8') as f:
            return [int(x) for x in f.read().strip().split()]
    except (OSError, ValueError) as e:
        logger.warning('加载 token 文件失败: %s', e)
        return None


# -- 输出路径 --------------------------------------------------

def make_output_path(input_path: str, custom: Optional[str] = None,
                     suffix: str = '') -> str:
    """生成输出文件路径。为 multi-sample 添加序号后缀。"""
    if custom:
        p = Path(custom)
        if suffix:
            stem = p.stem + suffix
            p = p.with_name(stem + p.suffix)
        p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    name = f'output_chopinote_{timestamp}{suffix}.musicxml'
    return str(Path.cwd() / name)


# -- 预设选择 --------------------------------------------------

def _select_preset_interactive() -> Optional[Preset]:
    """交互式预设选择，回车跳过。"""
    presets = _list_presets()
    if not presets:
        return None

    print('  可用预设模板（回车跳过）:')
    for i, p in enumerate(presets):
        conds = []
        if p.condition_key: conds.append(p.condition_key)
        if p.condition_time: conds.append(p.condition_time)
        if p.condition_tempo: conds.append(f'{p.condition_tempo}bpm')
        cstr = f' [{", ".join(conds)}]' if conds else ''
        print(f'    [{i}] {p.label}{cstr} — {p.description}')
    raw = input('  选择预设 [回车=跳过]: ').strip()
    if raw:
        try:
            idx = int(raw)
            if 0 <= idx < len(presets):
                p = presets[idx]
                print(f'      [预设] {p.label} — {p.description}')
                return p
        except ValueError:
            pass
    return None


# -- 交互式参数收集 --------------------------------------------

def prompt_params(max_bars_cli: Optional[int] = None,
                  temperature_cli: Optional[float] = None,
                  top_k_cli: Optional[int] = None,
                  complexity_cli: Optional[int] = None,
                  key_mode_cli: Optional[bool] = None,
                  time_mode_cli: Optional[bool] = None,
                  tempo_mode_cli: Optional[bool] = None,
                  meta: Optional[dict] = None,
                  seed_bars: int = 0,
                  model_vocab_size: Optional[int] = None):
    """交互式收集生成参数。
    已通过 CLI 提供的参数不再询问，返回参数字典。
    """
    params = {}

    min_new = max(1, max(4, seed_bars * 2) - seed_bars)

    if max_bars_cli is not None:
        params['max_bars'] = max_bars_cli
    else:
        default = 32
        raw = input(f'  最大续写小节数 (至少 {min_new}, seed 共 {seed_bars} 小节) [{default}]: ').strip()
        params['max_bars'] = int(raw) if raw else default

    if params['max_bars'] < min_new:
        print(f'      [!] 种子仅 {seed_bars} 小节，续写最少 {min_new} 小节，已自动设为 {min_new}')
        params['max_bars'] = min_new

    if temperature_cli is not None:
        params['temperature'] = temperature_cli
    else:
        default = 1.0
        raw = input(f'  采样温度 (0.1~2.0, 越低越保守) [{default}]: ').strip()
        params['temperature'] = float(raw) if raw else default

    if top_k_cli is not None:
        params['top_k'] = top_k_cli
    else:
        default = 20
        raw = input(f'  Top-K (越小越集中) [{default}]: ').strip()
        params['top_k'] = int(raw) if raw else default

    if complexity_cli is not None:
        params['complexity'] = max(0, min(10, complexity_cli))
    else:
        default = 5
        raw = input(f'  音乐复杂度 (0~10, 0=最简单/10=最复杂) [{default}]: ').strip()
        params['complexity'] = int(raw) if raw else default
        params['complexity'] = max(0, min(10, params['complexity']))

    # ── 调性 / 拍号 / 速度 开关 ──────────────────────────
    if key_mode_cli is not None:
        params['lock_key'] = key_mode_cli
    else:
        raw = input(f'  锁定调性（不变调）? [Y/n]: ').strip().lower()
        params['lock_key'] = raw not in ('n', 'no')

    if time_mode_cli is not None:
        params['lock_time'] = time_mode_cli
    else:
        raw = input(f'  锁定拍号（不变拍）? [Y/n]: ').strip().lower()
        params['lock_time'] = raw not in ('n', 'no')

    if tempo_mode_cli is not None:
        params['lock_tempo'] = tempo_mode_cli
    else:
        raw = input(f'  锁定速度（不变速）? [Y/n]: ').strip().lower()
        params['lock_tempo'] = raw not in ('n', 'no')

    return params


# -- 曲谱基本信息展示 ------------------------------------------

def display_seed_info(input_path: str, all_tokens: list[int],
                      tokenizer: REMITokenizer, metadata: dict,
                      seed_tokens: list[int], model_vocab_size: int,
                      lock_key: bool = True, lock_time: bool = True,
                      lock_tempo: bool = True):
    """展示输入曲谱和种子的基本信息。"""
    bar_id = tokenizer.bar_token_id
    total_bars = sum(1 for t in all_tokens if t == bar_id)
    seed_bars = sum(1 for t in seed_tokens if t == bar_id)

    print(f'      [OK] {os.path.basename(input_path)}')
    print(f'      | 总 token 数:  {len(all_tokens)}')
    print(f'      | 种子小节:    {seed_bars} 小节 ({len(seed_tokens)} tokens)')
    print(f'      | 输入曲谱共   {total_bars} 小节')

    # 调性 / 拍号 / 速度（含锁定状态）
    info = _extract_intro_info(all_tokens, tokenizer)
    parts = []
    if 'key' in info:
        label = f'调性 {info["key"]}'
        if lock_key: label += '（锁定）'
        parts.append(label)
    if 'time_sig' in info:
        label = f'拍号 {info["time_sig"]}'
        if lock_time: label += '（锁定）'
        parts.append(label)
    if 'tempo' in info:
        label = f'速度 {info["tempo"]} BPM'
        if lock_tempo: label += '（锁定）'
        parts.append(label)
    if parts:
        print(f'      | {" | ".join(parts)}')

    if model_vocab_size != tokenizer.vocab_size:
        print(f'      | 模型词表:   {model_vocab_size}（tokenizer: {tokenizer.vocab_size}，OOB 已屏蔽）')


# -- 生成后交互循环 --------------------------------------------

def generate_once(model, tokenizer, seed_tensor, device,
                  params: dict, model_vocab_size: int,
                  generation_idx: int = 0,
                  feedback_callback=None,
                  gen_params=None) -> tuple[list[int], dict]:
    """一次生成，返回 (token_id_list, stats)。不保存文件。"""
    seed = params.get('seed')
    if seed is not None:
        torch.manual_seed(seed + generation_idx)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(seed + generation_idx)

    # ── 段落感知两阶段生成 ──
    if params.get('section_aware'):
        return generate_section_aware_once(
            model, tokenizer, seed_tensor[0].tolist(), device,
            params, model_vocab_size,
        )

    full_ids, stats = generate_with_progress(
        model, seed_tensor, tokenizer,
        max_bars=params['max_bars'],
        temperature=params['temperature'],
        top_k=params['top_k'],
        effective_vocab_size=model_vocab_size,
        lock_key=params.get('lock_key', True),
        lock_time=params.get('lock_time', True),
        lock_tempo=params.get('lock_tempo', True),
        lock_program=params.get('lock_program', True),
        complexity=params.get('complexity'),  # None = auto from seed
        rest_penalty=params.get('rest_penalty', 0.0),
        max_polyphony=params.get('max_polyphony', 10),
        key_bias_strength=params.get('key_bias_strength', 2.0),
        prog_switch_strength=params.get('prog_switch_strength', 1.0),
        prog_switch_interval=params.get('prog_switch_interval', 12),
        feedback_callback=feedback_callback,
        gen_params=gen_params,
    )
    print()

    # 展示统计
    print(f'  [生成完成]')
    print(f'    生成小节: {stats["bars"]} | 生成 token: {stats["new_tokens"]}')
    print(f'    耗时: {stats["time_seconds"]:.1f}s | 速度: {stats["tokens_per_sec"]:.0f} tok/s')

    full_list = full_ids[0].tolist()
    return full_list, stats


def interactive_retry_loop(model, tokenizer, seed_tensor, device,
                           base_params: dict, base_output_path: str,
                           max_bars_total: int, model_vocab_size: int,
                           num_samples: int = 1, do_validate: bool = False,
                           auto_save: bool = False,
                           feedback_callback=None,
                           gen_params=None):
    """主交互循环：生成 → 操作选择 → 保存/重试/变体/退出。

    当 num_samples > 1 时自动批量生成。
    auto_save=True 时单样本也直接保存退出（CLI 全参数模式）。
    """
    if num_samples > 1:
        print(f'  [批量生成 {num_samples} 个变体]')
        variant_seed = base_params.get('seed')
        for i in range(num_samples):
            print(f'\n--- 变体 {i + 1}/{num_samples} ---')
            params = dict(base_params)
            if variant_seed is not None:
                params['seed'] = variant_seed
            path = str(Path(base_output_path).with_name(
                f'{Path(base_output_path).stem}_{i + 1}.musicxml'
            ))
            full_list, _ = generate_once(
                model, tokenizer, seed_tensor, device,
                params, model_vocab_size, i,
                feedback_callback=feedback_callback,
                gen_params=gen_params,
            )
            save_to_musicxml(full_list, tokenizer, path, max_bars_total)
            print(f'    [OK] 已保存: {os.path.abspath(path)}')
            if do_validate:
                from scripts.validate_generation import validate_generated_xml
                vr = validate_generated_xml(path, tokenizer=tokenizer)
                if not vr['passed']:
                    print(f'    [!] 验证警告: {vr["summary"]}')
        print(f'\n  [批量完成] 生成了 {num_samples} 个变体')
        return

    # 单样本模式：交互循环
    params = dict(base_params)
    gen_idx = 0

    while True:
        gen_idx += 1
        gen_path = str(Path(base_output_path).with_name(
            f'{Path(base_output_path).stem}_v{gen_idx}.musicxml'
        )) if gen_idx > 1 else base_output_path

        full_list, stats = generate_once(
            model, tokenizer, seed_tensor, device,
            params, model_vocab_size, gen_idx,
            feedback_callback=feedback_callback,
            gen_params=gen_params,
        )

        # 操作选择
        if auto_save:
            save_to_musicxml(full_list, tokenizer, gen_path, max_bars_total, save_tokens=True)
            print(f'\n  [OK] 已保存: {os.path.abspath(gen_path)}')
            if do_validate:
                from scripts.validate_generation import validate_generated_xml
                vr = validate_generated_xml(gen_path, tokenizer=tokenizer)
                if vr['passed']:
                    print('      验证通过: ' + vr['summary'])
                else:
                    print('      [!] 验证: ' + vr['summary'])
            return

        print('  操作选择:')
        print('    [s] 保存并退出')
        if gen_idx > 1:
            print('    [r] 换参数重新生成')
            print('    [v] 生成另一个变体（自动换种子）')
        print('    [q] 不保存退出')
        choice = input('  请输入: ').strip().lower()

        if choice == 's':
            save_to_musicxml(full_list, tokenizer, gen_path, max_bars_total, save_tokens=True)
            print(f'\n  [OK] 已保存: {os.path.abspath(gen_path)}')
            if do_validate:
                from scripts.validate_generation import validate_generated_xml
                vr = validate_generated_xml(gen_path, tokenizer=tokenizer)
                if vr['passed']:
                    print('      验证通过: ' + vr['summary'])
                else:
                    print('      [!] 验证: ' + vr['summary'])
            return
        elif choice == 'r' and gen_idx > 1:
            print()
            params = prompt_params()
            print()
        elif choice == 'v' and gen_idx > 1:
            seed = params.get('seed')
            if seed is not None:
                params['seed'] = seed + gen_idx
            print()
        elif choice == 'q':
            print('\n  已退出，未保存')
            return
        else:
            print('  [!] 请输入 s 或 q')
            print()


# -- 评价模式 ----------------------------------------------------

def _run_evaluate(args):
    """评价模式入口：对输入乐谱做内在/一致性评价（C 层）。"""
    from chopinote_abc.scoring import evaluate_generation
    from chopinote_abc.parser import parse_musicxml

    print('=== Chopinote-AI 乐谱评价 ===')
    print()

    print(f'  输入: {args.input}')
    if args.seed_path:
        print(f'  种子: {args.seed_path}')
    print()

    # 解析乐谱 + 评估
    score_obj = parse_musicxml(args.input)
    seed_score = parse_musicxml(args.seed_path) if args.seed_path else None

    from chopinote_dataset.tokenizer import REMITokenizer
    tokenizer = REMITokenizer()

    # 从 Score 提取 token 用于评价
    # (simplified: use score-level evaluation only)
    report = evaluate_generation(
        [], tokenizer,  # tokens not available in standalone eval mode
        seed_tokens=None,
        score=score_obj,
    )

    print(f'  综合评分: {report.total_score:.4f}')
    print(f'  合法性: {"通过" if report.legality_passed else "失败"}')
    print(f'  内在分: {report.intrinsic_score:.4f}')
    print(f'  理论违规: {report.theory.get("n_violations", 0)}')

    # JSON 输出
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump({
                'total_score': report.total_score,
                'legality_passed': report.legality_passed,
                'intrinsic_score': report.intrinsic_score,
                'theory': report.theory,
            }, f, indent=2, ensure_ascii=False)
        print()
        print(f'  JSON 报告已保存: {args.output}')

    print()


# -- 段落感知生成入口 ----------------------------------------------------

def _parse_structure_tokens(structure_tokens: list[int], tokenizer) -> list[dict]:
    """将结构规划的输出 token 解析为 section_plan。"""
    section_token_map = {
        name: f'<Section {name}>' for name in tokenizer.SECTION_NAMES
    }

    plan = []
    current_section = None
    current_bars = 8
    current_key = 'C'

    for tid in structure_tokens:
        ts = tokenizer.decode_token(tid)

        # 检测 Section token
        for sec_name, sec_str in section_token_map.items():
            if ts == sec_str:
                if current_section is not None:
                    plan.append({
                        'type': current_section,
                        'bars': current_bars,
                        'key': current_key,
                    })
                current_section = sec_name
                current_bars = 8  # reset to default
                break

        if ts.startswith(tokenizer.TONIC):
            current_key = ts[len(tokenizer.TONIC) + 1:-1]
        elif ts.startswith('<Bar_'):
            try:
                current_bars = int(ts[len('<Bar_'):-1])
            except ValueError:
                pass

    # 最后一个 section
    if current_section is not None:
        plan.append({
            'type': current_section,
            'bars': current_bars,
            'key': current_key,
        })

    return plan


@torch.no_grad()
def generate_section_aware_once(
    model, tokenizer, seed_tokens, device,
    params: dict, model_vocab_size: int,
) -> tuple[list[int], dict]:
    """段落感知两阶段生成，返回 (token_id_list, stats)。"""
    import time
    from chopinote_model.generate import generate_structure_plan, section_aware_generate

    start_time = time.time()

    # Stage 1: 结构规划
    print('  [Stage 1/2] 结构规划...')
    form_constraint = {
        'form': params.get('section_form', 'sonata'),
        'total_bars': params.get('section_total_bars', 64),
    }

    structure_tokens = generate_structure_plan(
        model, tokenizer, list(seed_tokens),
        form_constraint=form_constraint,
        max_new_tokens=params.get('structure_max_tokens', 32),
        temperature=params.get('temperature', 1.0),
        top_k=params.get('top_k', 20),
    )

    # Parse structure tokens into section plan
    section_plan = _parse_structure_tokens(structure_tokens, tokenizer)
    if not section_plan:
        print('  [!] 结构规划为空，使用默认奏鸣曲式')
        section_plan = [
            {'type': 'exposition', 'bars': 16, 'key': 'C'},
            {'type': 'development', 'bars': 16, 'key': 'G'},
            {'type': 'recapitulation', 'bars': 16, 'key': 'C'},
            {'type': 'coda', 'bars': 8, 'key': 'C'},
        ]

    print(f'  结构规划: {len(section_plan)} 段')
    for i, sec in enumerate(section_plan):
        print(f'    段{i+1}: {sec["type"]} x {sec["bars"]}bars key={sec.get("key", "?")}')

    # Stage 2: 段落条件生成
    print('  [Stage 2/2] 段落条件生成...')
    full_tokens = section_aware_generate(
        model, tokenizer, list(seed_tokens),
        section_plan=section_plan,
        max_bars=params.get('max_bars', 64),
        max_new_tokens=params.get('max_new_tokens', 4096),
        temperature=params.get('temperature', 1.0),
        top_k=params.get('top_k', 20),
    )

    elapsed = time.time() - start_time
    bar_id = tokenizer.bar_token_id
    seed_bars = sum(1 for t in seed_tokens if t == bar_id)
    total_bars = sum(1 for t in full_tokens if t == bar_id)
    new_tokens = len(full_tokens) - len(seed_tokens)

    stats = {
        'bars': total_bars - seed_bars,
        'new_tokens': max(0, new_tokens),
        'total_tokens': len(full_tokens),
        'time_seconds': elapsed,
        'tokens_per_sec': max(0, new_tokens) / elapsed if elapsed > 0 else 0,
    }

    return full_tokens, stats


# -- 主入口 ----------------------------------------------------

def main():
    # `--list-presets` 早期退出，避免 argparse 要求 checkpoint/input
    if '--list-presets' in sys.argv:
        from chopinote_cli.presets import list_presets as _lp
        print('=== 可用预设 ===')
        for p in _lp():
            conds = []
            if p.condition_key: conds.append(f'key={p.condition_key}')
            if p.condition_time: conds.append(f'time={p.condition_time}')
            if p.condition_tempo: conds.append(f'tempo={p.condition_tempo}')
            if p.program is not None: conds.append(f'prog={p.program}')
            extra = f' [{", ".join(conds)}]' if conds else ''
            print(f'  {p.name:12s}  {p.label} — {p.description}{extra}')
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description='Chopinote-AI 钢琴谱续写工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            '示例:\n'
            '  chopin best.pt input.musicxml\n'
            '  chopin best.pt input.musicxml --config my_cfg.yaml\n'
            '  chopin best.pt input.musicxml --temp 1.2 --top-k 40\n'
            '  chopin best.pt input.musicxml --random-seed -n 3\n'
        ),
    )
    parser.add_argument('checkpoint', nargs='?', default=None,
                        help='模型权重文件路径 (.pt)，评价模式下可选')
    parser.add_argument('input', help='输入 MusicXML 乐谱文件路径')
    parser.add_argument('-o', '--output', default=None,
                        help='输出 MusicXML 文件路径')
    parser.add_argument('--seed-bars', type=int, default=None,
                        help='从输入曲谱末尾截取的小节数作为种子（默认: 16）')
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument('--seed', type=int, default=None,
                        help='固定随机种子（与 --random-seed 互斥）')
    seed_group.add_argument('--random-seed', action='store_true', default=None,
                        help='自动生成随机种子实现可复现')
    parser.add_argument('-n', '--num-samples', type=int, default=None,
                        help='一次生成 N 个变体（默认: 1）')
    parser.add_argument('--temp', type=float, default=None,
                        help='采样温度 (0.1~2.0)，指定后跳过交互式输入')
    parser.add_argument('--top-k', type=int, default=None,
                        help='Top-K 采样数，指定后跳过交互式输入')
    parser.add_argument('--max-bars', type=int, default=None,
                        help='最大续写小节数，指定后跳过交互式输入')
    parser.add_argument('--key-mode', choices=['lock', 'free'], default=None,
                        help='调性模式: lock=锁定不变调, free=自由变调')
    parser.add_argument('--time-mode', choices=['lock', 'free'], default=None,
                        help='拍号模式: lock=锁定不变拍, free=自由变拍')
    parser.add_argument('--tempo-mode', choices=['lock', 'free'], default=None,
                        help='速度模式: lock=锁定不变速, free=自由变速')
    parser.add_argument('--complexity', type=int, default=None,
                        help='音乐复杂度 0~10（0=最简单, 10=最复杂）')
    parser.add_argument('--validate', action='store_true',
                        help='生成后自动交叉验证 MusicXML 质量')
    parser.add_argument('--preset', type=str, default=None,
                        help='预设模板（baroque/romantic/classical 等）')
    parser.add_argument('--target-key', type=str, default=None,
                        help='目标调性（如 G, Am），在种子末尾插入 Anticipate token 引导转调')
    parser.add_argument('--condition-key', type=str, default=None,
                        help='指定调性，如 C, Am, G 等')
    parser.add_argument('--condition-time', type=str, default=None,
                        help='指定拍号，如 4/4, 3/4, 6/8 等')
    parser.add_argument('--condition-tempo', type=int, default=None,
                        help='指定速度 BPM，如 60, 120, 180 等')
    parser.add_argument('--list-presets', action='store_true',
                        help='列出所有可用预设并退出')
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径（默认自动搜索 ./chopinote_config.yaml / ~/.chopinote/config.yaml）')
    parser.add_argument('--lock-program', choices=['lock', 'free'], default=None,
                        help='乐器锁定: lock=只保留 seed 已有的乐器, free=允许自由切换')
    parser.add_argument('--rest-penalty', type=float, default=None,
                        help='Rest 惩罚（0~10，越大休止越少，默认 0）')
    parser.add_argument('--max-polyphony', type=int, default=None,
                        help='同粒度最大同时发音数（模拟手指上限，默认 10）')
    parser.add_argument('--key-bias', type=float, default=None,
                        help='调性音高偏置强度 (0~5, 越大越严格遵循调性, 默认 2.0)')
    parser.add_argument('--prog-switch-strength', type=float, default=None,
                        help='乐器切换偏置强度 (0~5, 越大切换越频繁, 默认 1.0)')
    parser.add_argument('--prog-switch-interval', type=int, default=None,
                        help='触发切换偏置的最少连续音符数 (1~128, 默认 12)')
    # ── 段落感知生成 ─────────────────────────────────────
    parser.add_argument('--section-aware', action='store_true', default=None,
                        help='启用段落感知两阶段生成（结构规划 → 细节填充）')
    parser.add_argument('--section-form', type=str, default=None,
                        choices=['sonata', 'rondo', 'aba', 'theme-variations', 'binary'],
                        help='曲式约束 (默认: sonata)')
    parser.add_argument('--section-total-bars', type=int, default=None,
                        help='结构规划目标总小节数 (默认: 64)')

    # ── 评价模式 ─────────────────────────────────────────
    parser.add_argument('--evaluate', action='store_true',
                        help='评价模式：对输入的乐谱打分')
    # ── 反馈模式（默认启用） ─────────────────────────────
    parser.add_argument('--no-feedback', action='store_true',
                        help='禁用评价反馈，纯推理模式（最快速度）')
    parser.add_argument('--feedback-level',
                        choices=['off', 'light', 'normal', 'strict'],
                        default='normal',
                        help='ABC 引擎强度: off=纯推理 light=A+C normal=A+B+C(默认) strict=A+B+C+自动重试')
    parser.add_argument('--feedback', action='store_true',
                        help='[已弃用] ABC Engine 现在默认启用，请使用 --no-feedback 关闭或 --feedback-level 调整')
    parser.add_argument('--local-weight', type=float, default=0.5,
                        help='B1 硬约束权重 (0~1, 默认 0.5)')
    parser.add_argument('--global-weight', type=float, default=0.5,
                        help='B2 调参权重 (0~1, 默认 0.5)')
    parser.add_argument('--retry-threshold', type=float, default=0.55,
                        help='C 层重试阈值 (0~1, 低于此值重试, 默认 0.55)')
    parser.add_argument('--max-retries', type=int, default=3,
                        help='C 层最大重试次数 (默认 3)')
    parser.add_argument('--seed-path', type=str, default=None,
                        help='种子乐谱文件路径（续写场景评价）')
    parser.add_argument('--alpha', type=float, default=0.3,
                        help='场景权重 0~1（1=纯内在评价，0.3=续写评价）')
    parser.add_argument('--group', type=str, default='all',
                        help='对比基准组名（all, timesig_4_4, source_musescore 等）')
    args = parser.parse_args()

    # 判断用户是否主动传了 CLI 参数（config 覆盖前计算，避免被配置默认值误判）
    has_cli_params = any([
        args.max_bars is not None, args.temp is not None,
        args.top_k is not None, args.complexity is not None,
        args.key_mode is not None, args.time_mode is not None,
        args.tempo_mode is not None, args.lock_program is not None,
        args.rest_penalty is not None, args.max_polyphony is not None,
        args.key_bias is not None, args.prog_switch_strength is not None,
        args.prog_switch_interval is not None,
        args.section_aware,
    ])

    # ── 加载配置文件 ─────────────────────────────────────
    cfg = load_config(args.config)
    cfg_path = find_config(args.config)
    # 配置文件覆盖 argparse 默认值（不影响 has_cli_params 判断）
    if args.seed_bars is None:
        args.seed_bars = cfg.seed_bars
    if args.num_samples is None:
        args.num_samples = cfg.num_samples
    if args.section_form is None:
        args.section_form = cfg.section_form
    if args.section_total_bars is None:
        args.section_total_bars = cfg.section_total_bars
    # max_bars 仅当实际配置文件存在时才覆盖（无配置文件时保持交互式询问）
    if args.max_bars is None and cfg_path:
        args.max_bars = cfg.max_bars
    # 注意: section_aware 属于 has_cli_params，不从 config 覆盖 args
    # ─────────────────────────────────────────────────────

    # ── 评价模式 ─────────────────────────────────────────
    if args.evaluate:
        _run_evaluate(args)
        return

    # 生成模式需要 checkpoint
    if not args.checkpoint:
        print('[X] 生成模式需要提供 checkpoint 路径')
        print('  用法: chopin <checkpoint.pt> <input.musicxml>')
        print('  评价: chopin --evaluate <input.musicxml>')
        sys.exit(1)

    print('=== Chopinote-AI - 钢琴谱续写工具 ===')
    print()
    if cfg_path:
        print(f'  配置文件: {cfg_path}')
        print()

    # -- 设备与硬件检测 -----------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'  设备: {device}')

    sys_profile = detect_system()
    infer_cfg = suggest_inference(sys_profile)
    print_hardware_report(sys_profile, infer_cfg)
    # 在 PyTorch 初始化线程池前设置线程数
    if infer_cfg.num_threads:
        torch.set_num_threads(infer_cfg.num_threads)
    print()

    # -- 第 1 步：加载模型 --------------------------------
    print('[1/4] 加载模型...')
    model, config, step, loss = load_model(args.checkpoint, device, infer_cfg=infer_cfg)

    # 使用 checkpoint 的 vocab_size（可能不同于当前 ModelConfig 默认值）
    model_vocab_size = config.vocab_size

    print(f'      [OK] checkpoint: {args.checkpoint}')
    print(f'      | 训练步数:   {step}')
    if loss is not None:
        print(f'      | 保存时 loss: {loss:.4f}')
    print(f'      | 词表大小:   {model_vocab_size}')
    print(f'      | 模型参数:   {sum(p.numel() for p in model.parameters()):,}')
    print()

    # -- 第 2 步：解析输入 --------------------------------
    print('[2/4] 解析输入乐谱...')
    tokenizer = REMITokenizer(grid_size=16, velocity_levels=8)
    all_tokens, metadata = musicxml_to_seed(args.input, tokenizer, model_vocab_size)

    # 移除首尾 BOS/EOS 以截取有效内容
    content_tokens = [t for t in all_tokens
                      if t not in (tokenizer.bos_token_id, tokenizer.eos_token_id)]

    # 截取末尾作为种子（保留完整小节）
    bar_id = tokenizer.bar_token_id
    bar_positions = [i for i, t in enumerate(content_tokens) if t == bar_id]
    seed_bars_count = min(args.seed_bars, len(bar_positions))
    if seed_bars_count > 0:
        cut_idx = bar_positions[-seed_bars_count]
        seed_tokens = content_tokens[cut_idx:]
    else:
        seed_tokens = content_tokens
        print(f'      [!] 输入曲谱不足 1 小节，使用全部内容作为种子')

    # ── 开关 CLI 转 bool（None=交互询问） ──────────────
    def _mode_bool(v: str | None) -> bool | None:
        return {'lock': True, 'free': False}.get(v)

    key_mode_bool = _mode_bool(args.key_mode)
    time_mode_bool = _mode_bool(args.time_mode)
    tempo_mode_bool = _mode_bool(args.tempo_mode)
    # ────────────────────────────────────────────────────

    # ── 预设模板 ────────────────────────────────────────
    preset = None
    if args.preset:
        preset = get_preset(args.preset)
        if preset is None:
            print(f'  [!] 未知预设: {args.preset}')
            print(f'  可用预设: {", ".join(p.name for p in _list_presets())}')
            sys.exit(1)
        print(f'      [预设] {preset.label} — {preset.description}')

    if preset is None and not has_cli_params:
        preset = _select_preset_interactive()
    # ────────────────────────────────────────────────────

    # 用预设值填充 CLI 默认值
    if preset:
        pa = preset.attrs()
        _temp = args.temp if args.temp is not None else pa.get('temperature')
        _top_k = args.top_k if args.top_k is not None else pa.get('top_k')
        _complexity = args.complexity if args.complexity is not None else pa.get('complexity')
        _key_mode = key_mode_bool if key_mode_bool is not None else pa.get('lock_key')
        _time_mode = time_mode_bool if time_mode_bool is not None else pa.get('lock_time')
        _tempo_mode = tempo_mode_bool if tempo_mode_bool is not None else pa.get('lock_tempo')
    else:
        if cfg_path:
            # 配置文件存在 → 使用配置值（跳过交互式询问）
            _temp = args.temp if args.temp is not None else cfg.temperature
            _top_k = args.top_k if args.top_k is not None else cfg.top_k
            _complexity = args.complexity if args.complexity is not None else cfg.complexity
            _key_mode = key_mode_bool if key_mode_bool is not None else cfg.lock_key
            _time_mode = time_mode_bool if time_mode_bool is not None else cfg.lock_time
            _tempo_mode = tempo_mode_bool if tempo_mode_bool is not None else cfg.lock_tempo
        else:
            # 无配置文件 → 保留原始交互行为
            _temp = args.temp
            _top_k = args.top_k
            _complexity = args.complexity
            _key_mode = key_mode_bool
            _time_mode = time_mode_bool
            _tempo_mode = tempo_mode_bool

    # -- 第 3 步：参数设置 --------------------------------
    print('[3/4] 设置续写参数...')
    params = prompt_params(
        max_bars_cli=args.max_bars,
        temperature_cli=_temp,
        top_k_cli=_top_k,
        complexity_cli=_complexity,
        key_mode_cli=_key_mode,
        time_mode_cli=_time_mode,
        tempo_mode_cli=_tempo_mode,
        seed_bars=seed_bars_count,
        model_vocab_size=model_vocab_size,
    )
    # ── 随机种子（优先级: --seed > --random-seed > config.random_seed > 无） ──
    if args.seed is not None:
        params['seed'] = args.seed
    elif args.random_seed if args.random_seed is not None else cfg.random_seed:
        import random
        params['seed'] = random.randint(0, 2**31 - 1)
        print(f'      [Seed] 自动随机种子: {params["seed"]}')

    params['lock_program'] = (
        _mode_bool(args.lock_program)
        if args.lock_program is not None else cfg.lock_program
    )
    params['rest_penalty'] = args.rest_penalty if args.rest_penalty is not None else cfg.rest_penalty
    params['max_polyphony'] = args.max_polyphony if args.max_polyphony is not None else cfg.max_polyphony
    params['key_bias_strength'] = args.key_bias if args.key_bias is not None else cfg.key_bias_strength
    params['prog_switch_strength'] = args.prog_switch_strength if args.prog_switch_strength is not None else cfg.prog_switch_strength
    params['prog_switch_interval'] = args.prog_switch_interval if args.prog_switch_interval is not None else cfg.prog_switch_interval
    # ── 段落感知生成参数 ──────────────────────────────────
    params['section_aware'] = cfg.section_aware if args.section_aware is None else args.section_aware
    params['section_form'] = args.section_form if args.section_form is not None else cfg.section_form
    params['section_total_bars'] = args.section_total_bars if args.section_total_bars is not None else cfg.section_total_bars
    params['structure_max_tokens'] = 64  # 结构规划最多 64 tokens
    # ─────────────────────────────────────────────────────

    print()
    # 展示曲谱信息（含锁定状态）
    display_seed_info(args.input, all_tokens, tokenizer, metadata,
                      seed_tokens, model_vocab_size,
                      lock_key=params['lock_key'],
                      lock_time=params['lock_time'],
                      lock_tempo=params['lock_tempo'])

    # ── 条件注入：控制 token 前缀 ────────────────────────
    condition_tokens = []
    condition_labels = []
    conds = {}
    if preset:
        conds.update(preset.conditions())
    if args.condition_key:
        conds['key'] = args.condition_key
    if args.condition_time:
        conds['time'] = args.condition_time
    if args.condition_tempo:
        conds['tempo'] = args.condition_tempo

    if 'key' in conds:
        condition_tokens.append(tokenizer.encode_token(f'{tokenizer.TONIC} {conds["key"]}>'))
        condition_labels.append(f'调性 {conds["key"]}')
    if 'time' in conds:
        condition_tokens.append(tokenizer.encode_token(f'<TimeSig {conds["time"]}>'))
        condition_labels.append(f'拍号 {conds["time"]}')
    if 'tempo' in conds:
        condition_tokens.append(tokenizer.encode_token(f'<Tempo {conds["tempo"]}>'))
        condition_labels.append(f'速度 {conds["tempo"]} BPM')
    if 'program' in conds:
        condition_tokens.append(tokenizer.encode_token(f'<Program {conds["program"]}>'))
        condition_labels.append(f'音色 {conds["program"]}')

    if condition_tokens:
        seed_tokens = condition_tokens + seed_tokens
        print(f'      [条件] {" | ".join(condition_labels)}')
    # ────────────────────────────────────────────────────

    # ── 目标调性（v0.3.0: 通过 Tonic token 置于 prefix）─────
    if args.target_key:
        if args.target_key in REMITokenizer.TONIC_NAMES:
            tonic_tid = tokenizer.encode_token(f'{tokenizer.TONIC} {args.target_key}>')
            if tonic_tid < model_vocab_size:
                seed_tokens.append(tonic_tid)
                print(f'      [Tonic] 目标调性 {args.target_key}')
            else:
                print(f'  [!] Tonic token 超出模型词表 (vocab_size={model_vocab_size})，忽略')
        else:
            print(f'  [!] 无效目标调性: {args.target_key}，忽略')
    # ────────────────────────────────────────────────────

    # ── 段落感知模式提示 ──────────────────────────────────
    if params.get('section_aware'):
        print(f'      [段落感知] 两阶段生成 | 曲式: {params.get("section_form")} | '
              f'目标: {params.get("section_total_bars")} 小节')
    print()

    # ── ABC Engine 生成模式（默认） ─────────────────────
    feedback_level = 'off' if args.no_feedback else args.feedback_level

    # 兼容旧 --feedback 参数
    if args.feedback:
        logger.warning("[已弃用] --feedback 现在默认启用，请使用 --no-feedback 关闭或 --feedback-level 调整")

    output_path = make_output_path(args.input, args.output)
    max_bars_total = params['max_bars'] + args.seed_bars
    seed_tensor = torch.tensor([seed_tokens], dtype=torch.long, device=device)

    if feedback_level != 'off':
        # ── ABC Engine v2: 三阶段逐段迭代生成 ──
        from chopinote_model.generate import stage3_iterative_generate
        from chopinote_abc.scoring import evaluate_generation
        from chopinote_abc.database import write_reward_log

        form = args.section_form or 'free'
        max_retries = args.max_retries if feedback_level == 'strict' else 1

        print(f'  [ABC Engine] 曲式: {form} | 目标: {params["max_bars"]} 小节 | '
              f'温度: {params["temperature"]} | top_k: {params["top_k"]}')
        print(f'               重试: {max_retries} | '
              f'模式: {"严格" if feedback_level == "strict" else "正常"}')
        print()

        # 一次调用完成全部 A→B→C 流程
        print('[ABC] 开始三阶段逐段迭代生成...')
        all_tokens, abc_report = stage3_iterative_generate(
            model, tokenizer, seed_tokens,
            max_bars=params['max_bars'],
            form=form,
            max_retries=max_retries,
            base_temperature=params.get('temperature', 1.0),
            top_k=params.get('top_k', 20),
        )

        # 保存 MusicXML
        save_to_musicxml(all_tokens, tokenizer, output_path, max_bars_total,
                         save_tokens=True)
        print(f'  [✓] 已保存: {output_path}')

        # ── C 层：进化评价 ──
        novelty = abc_report.get('novelty_score', 0.0) if abc_report else 0.0
        diversity = abc_report.get('diversity_score', 0.0) if abc_report else 0.0

        print(f'  [C 层] 评价: {os.path.basename(output_path)}')
        try:
            eval_report = evaluate_generation(
                all_tokens, tokenizer,
                seed_tokens=seed_tokens,
                musicxml_path=output_path,
                novelty_bonus=novelty,
                diversity_bonus=diversity,
                structural_fixes=abc_report.get('structural_fixes', []) if abc_report else [],
                archive_commands=abc_report.get('archive_commands', []) if abc_report else [],
            )
            print(f'      | 综合评分: {eval_report.total_score:.4f} | '
                  f'合法性: {"通过" if eval_report.legality_passed else "失败"}')
            print(f'      | 内在分: {eval_report.intrinsic_score:.4f} | '
                  f'理论违规: {eval_report.theory.get("n_violations", 0)}')

            if eval_report.structural_fixes:
                print(f'      | 结构修复建议 ({len(eval_report.structural_fixes)} 条):')
                for fix in eval_report.structural_fixes[:5]:
                    desc = getattr(fix, 'description', '') or ''
                    print(f'        - {fix.type}: {desc}')

            # 写 reward log — 统一路径 + 传 dataclass 字段
            try:
                write_reward_log(
                    '/root/autodl-tmp/chopinote/reward_log.jsonl',
                    eval_report,
                    novelty,
                    diversity,
                    seed_path=args.input,
                    musicxml_path=output_path,
                    total_score=eval_report.total_score,
                )
            except Exception:
                pass
        except Exception as e:
            print(f'      | [!] Score 级评价失败: {e}')
            import traceback
            traceback.print_exc()

    else:
        # ── 无反馈模式：直接生成（兼容旧行为）──
        print('[4/4] 开始续写（无反馈模式）...')
        interactive_retry_loop(
            model, tokenizer, seed_tensor, device,
            params, output_path, max_bars_total, model_vocab_size,
            num_samples=args.num_samples, do_validate=args.validate,
            auto_save=has_cli_params,
            feedback_callback=None,
            gen_params=None,
        )
    # ─────────────────────────────────────────────────────
    print()


if __name__ == '__main__':
    main()
