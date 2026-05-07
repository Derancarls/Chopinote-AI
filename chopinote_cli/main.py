"""
Chopinote-AI CLI: 钢琴谱续写命令行工具

用法:
    chopinote-generate best.pt input.musicxml
    chopinote-generate best.pt input.musicxml --temp 1.2 --top-k 40 --seed 42
    chopinote-generate best.pt input.musicxml -n 3
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
from chopinote_dataset.tokenizer import REMITokenizer
from chopinote_dataset.converter import MusicXMLToREMI

logger = logging.getLogger(__name__)


# -- 模型加载 --------------------------------------------------

def load_model(checkpoint_path: str, device: torch.device):
    """从 checkpoint 加载模型，返回 (model, config, step, loss).
    自动适应 checkpoint 的 vocab_size，无需手动对齐 ModelConfig。
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
    model.eval()

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
        if token.startswith(REMITokenizer.KEY):
            info['key'] = token[len(REMITokenizer.KEY) + 1:-1]
        elif token.startswith(REMITokenizer.TIMESIG):
            info['time_sig'] = token[len(REMITokenizer.TIMESIG) + 1:-1]
        elif token.startswith(REMITokenizer.TEMPO):
            info['tempo'] = int(token[len(REMITokenizer.TEMPO) + 1:-1])
    return info


# -- 复杂度控制 -----------------------------------------------

def _apply_complexity(logits, complexity: float, tokenizer):
    """根据复杂度值 (0-10) 修改 logits，控制音乐密度和丰富程度。
    偏置值为加法，叠加到原始 logits 上。
    """
    if complexity == 5.0:
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
    complexity: float = 5.0,
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

    Returns:
        (1, T_total) 完整生成序列, stats dict
    """
    device = seed_tokens.device
    eos_id = tokenizer.eos_token_id
    bar_id = tokenizer.bar_token_id
    start_time = time.time()

    # ── 音高限制准备 ──────────────────────────────────────────
    from chopinote_model.generate import GM_INSTRUMENT_RANGES, _parse_program
    note_on_ids = [tokenizer.encode_token(f'<Note_ON {p}>') for p in range(128)]
    _prog_prefix = '<Program'
    cur_program: int | None = None
    for tid in reversed(seed_tokens[0].tolist()):
        ts = tokenizer.decode_token(tid)
        if ts.startswith(_prog_prefix):
            cur_program = _parse_program(ts)
            break
    # ──────────────────────────────────────────────────────────

    # ── 锁定 token ID 预计算 ──────────────────────────────
    _lock_ids = []
    if lock_key:
        _lock_ids.extend(
            tokenizer.encode_token(f'<Key {k}>') for k in REMITokenizer.KEY_NAMES
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

    # KV cache 初始化
    kv_caches = [[None, None] for _ in range(model.config.n_layers)]

    generated = seed_tokens.clone()
    next_token = seed_tokens
    bar_count = 0

    pbar = tqdm(total=max_bars, desc='生成中', unit='bar', ncols=80)

    for step in range(max_new_tokens):
        logits = model.forward(next_token, kv_caches=kv_caches)
        logits = logits[:, -1, :] / temperature  # (1, V)

        # ── 复杂度控制 ──────────────────────────────────
        _apply_complexity(logits, complexity, tokenizer)
        # ──────────────────────────────────────────────────

        # ── 词表截断 ──────────────────────────────────────
        if effective_vocab_size is not None:
            logits[0, effective_vocab_size:] = float('-inf')

        # ── 锁定屏蔽 ──────────────────────────────────────
        vocab_dim = logits.size(-1)
        for tid in _lock_ids:
            if tid < vocab_dim:
                logits[0, tid] = float('-inf')
        # ──────────────────────────────────────────────────

        # ── 音高限制 ──────────────────────────────────────
        if cur_program is not None and cur_program < 112:
            rmin, rmax = GM_INSTRUMENT_RANGES.get(cur_program, (0, 127))
            for pitch in range(rmin):
                logits[0, note_on_ids[pitch]] = float('-inf')
            for pitch in range(rmax + 1, 128):
                logits[0, note_on_ids[pitch]] = float('-inf')
        # ──────────────────────────────────────────────────

        if top_k > 0:
            k = min(top_k, logits.size(-1))
            vals, _ = torch.topk(logits, k, dim=-1)
            logits[logits < vals[:, -1:]] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

        generated = torch.cat([generated, next_token], dim=1)
        token_id = next_token.item()

        # 追踪 Program 变化
        ts = tokenizer.decode_token(token_id)
        if ts.startswith(_prog_prefix):
            cur_program = _parse_program(ts)

        if token_id == bar_id:
            bar_count += 1
            pbar.update(1)
            if bar_count >= max_bars:
                break
        else:
            pbar.set_postfix(tokens=step + 1, refresh=False)

        if token_id == eos_id:
            break

    pbar.close()
    elapsed = time.time() - start_time
    new_tokens = step + 1
    stats = {
        'bars': bar_count,
        'new_tokens': new_tokens,
        'total_tokens': generated.size(1),
        'time_seconds': elapsed,
        'tokens_per_sec': new_tokens / elapsed if elapsed > 0 else 0,
    }
    return generated, stats


# -- token → MusicXML 导出 ------------------------------------

def save_to_musicxml(
    token_ids: list[int],
    tokenizer: REMITokenizer,
    output_path: str,
    max_bars: int = 256,
):
    """将 token 序列保存为 MusicXML 文件。"""
    from chopinote_model.generate import tokens_to_notes, notes_to_score

    notes = tokens_to_notes(token_ids, tokenizer)
    score = notes_to_score(notes, grid_size=tokenizer.grid_size, max_bars=max_bars)
    score.write('musicxml', fp=output_path)

    num_bars = max(n['bar'] for n in notes) if notes else 0
    return num_bars


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
                  generation_idx: int = 0) -> tuple[list[int], dict]:
    """一次生成，返回 (token_id_list, stats)。不保存文件。"""
    seed = params.get('seed')
    if seed is not None:
        torch.manual_seed(seed + generation_idx)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(seed + generation_idx)

    full_ids, stats = generate_with_progress(
        model, seed_tensor, tokenizer,
        max_bars=params['max_bars'],
        temperature=params['temperature'],
        top_k=params['top_k'],
        effective_vocab_size=model_vocab_size,
        lock_key=params.get('lock_key', True),
        lock_time=params.get('lock_time', True),
        lock_tempo=params.get('lock_tempo', True),
        complexity=params.get('complexity', 5.0),
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
                           num_samples: int = 1):
    """主交互循环：生成 → 操作选择 → 保存/重试/变体/退出。

    当 num_samples > 1 时自动批量生成，跳过交互。
    """
    if num_samples > 1:
        print(f'  [批量生成 {num_samples} 个变体]')
        variant_seed = base_params.get('seed')
        for i in range(num_samples):
            print(f'\n--- 变体 {i + 1}/{num_samples} ---')
            params = dict(base_params)
            if variant_seed is not None:
                params['seed'] = variant_seed + i
            path = str(Path(base_output_path).with_name(
                f'{Path(base_output_path).stem}_{i + 1}.musicxml'
            ))
            full_list, _ = generate_once(
                model, tokenizer, seed_tensor, device,
                params, model_vocab_size, i,
            )
            save_to_musicxml(full_list, tokenizer, path, max_bars_total)
            print(f'    [OK] 已保存: {os.path.abspath(path)}')
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
        )

        # 操作选择
        print('  操作选择:')
        print('    [s] 保存并退出')
        if gen_idx > 1:
            print('    [r] 换参数重新生成')
            print('    [v] 生成另一个变体（自动换种子）')
        print('    [q] 不保存退出')
        choice = input('  请输入: ').strip().lower()

        if choice == 's':
            save_to_musicxml(full_list, tokenizer, gen_path, max_bars_total)
            print(f'\n  [OK] 已保存: {os.path.abspath(gen_path)}')
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


# -- 主入口 ----------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Chopinote-AI 钢琴谱续写工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            '示例:\n'
            '  chopinote-generate best.pt input.musicxml\n'
            '  chopinote-generate best.pt input.musicxml --temp 1.2 --top-k 40\n'
            '  chopinote-generate best.pt input.musicxml --seed 42 -n 3\n'
        ),
    )
    parser.add_argument('checkpoint', help='模型权重文件路径 (.pt)')
    parser.add_argument('input', help='输入 MusicXML 乐谱文件路径')
    parser.add_argument('-o', '--output', default=None,
                        help='输出 MusicXML 文件路径')
    parser.add_argument('--seed-bars', type=int, default=16,
                        help='从输入曲谱末尾截取的小节数作为种子（默认: 16）')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子，固定后可复现生成结果')
    parser.add_argument('-n', '--num-samples', type=int, default=1,
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
    args = parser.parse_args()

    print('=== Chopinote-AI - 钢琴谱续写工具 ===')
    print()

    # -- 设备 ---------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'  设备: {device}')
    print()

    # -- 第 1 步：加载模型 --------------------------------
    print('[1/4] 加载模型...')
    model, config, step, loss = load_model(args.checkpoint, device)

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

    # -- 第 3 步：参数设置 --------------------------------
    print('[3/4] 设置续写参数...')
    params = prompt_params(
        max_bars_cli=args.max_bars,
        temperature_cli=args.temp,
        top_k_cli=args.top_k,
        complexity_cli=args.complexity,
        key_mode_cli=key_mode_bool,
        time_mode_cli=time_mode_bool,
        tempo_mode_cli=tempo_mode_bool,
        seed_bars=seed_bars_count,
        model_vocab_size=model_vocab_size,
    )
    params['seed'] = args.seed  # seed 仅从 CLI 设置

    print()
    # 展示曲谱信息（含锁定状态）
    display_seed_info(args.input, all_tokens, tokenizer, metadata,
                      seed_tokens, model_vocab_size,
                      lock_key=params['lock_key'],
                      lock_time=params['lock_time'],
                      lock_tempo=params['lock_tempo'])

    print()

    # -- 第 4 步：生成 + 交互循环 -------------------------
    print('[4/4] 开始续写...')
    seed_tensor = torch.tensor([seed_tokens], dtype=torch.long, device=device)
    output_path = make_output_path(args.input, args.output)
    max_bars_total = params['max_bars'] + args.seed_bars

    interactive_retry_loop(
        model, tokenizer, seed_tensor, device,
        params, output_path, max_bars_total, model_vocab_size,
        num_samples=args.num_samples,
    )
    print()


if __name__ == '__main__':
    main()
