"""
Chopinote-AI CLI: 钢琴谱续写命令行工具

用法:
    chopinote-generate checkpoints/best.pt input.musicxml
    chopinote-generate checkpoints/best.pt input.musicxml -o output.musicxml
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
    自动处理 vocab_size 等 shape 不匹配的参数。
    """
    if not os.path.isfile(checkpoint_path):
        print(f'  [X] checkpoint 文件不存在: {checkpoint_path}')
        sys.exit(1)

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # 重建配置
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
            skipped.append(f'{k}: ckpt {tuple(v.shape)} vs model {tuple(model_state[k].shape) if k in model_state else "N/A"}')

    model.load_state_dict(model_state)

    if skipped:
        print(f'      [!] 跳过了 {len(skipped)} 个 shape 不匹配的参数（可能为旧版 checkpoint）:')
        for s in skipped[:3]:
            print(f'         - {s}')
        if len(skipped) > 3:
            print(f'         - ... 还有 {len(skipped) - 3} 个')

    model.to(device)
    model.eval()

    step = ckpt.get('step', 0)
    loss = ckpt.get('loss', None)
    return model, config, step, loss


# -- MusicXML → seed tokens ----------------------------------

def musicxml_to_seed(file_path: str, tokenizer: REMITokenizer) -> list[int]:
    """解析 MusicXML 为种子 token 序列，返回完整 token 列表。"""
    if not os.path.isfile(file_path):
        print(f'[X] 输入文件不存在: {file_path}')
        sys.exit(1)

    conv = MusicXMLToREMI(grid_size=tokenizer.grid_size,
                          velocity_levels=tokenizer.velocity_levels)
    tokens, metadata = conv.convert(file_path, collect_metadata=True)

    if not tokens:
        print('  [!] 未能从文件中解析出有效音符，将从头生成')
        # 返回仅含 BOS 的种子
        return [tokenizer.bos_token_id]

    return tokens


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
) -> torch.Tensor:
    """自回归生成，带 tqdm 进度条。

    Args:
        model: 训练好的模型
        seed_tokens: (1, T) 种子 token 序列
        tokenizer: REMI tokenizer
        max_bars: 最多生成多少个小节
        max_new_tokens: 最多生成多少 token
        temperature: 采样温度
        top_k: top-k 采样

    Returns:
        (1, T_total) 完整生成序列
    """
    device = seed_tokens.device
    eos_id = tokenizer.eos_token_id
    bar_id = tokenizer.bar_token_id

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

    # KV cache 初始化
    kv_caches = [[None, None] for _ in range(model.config.n_layers)]

    generated = seed_tokens.clone()
    next_token = seed_tokens
    bar_count = 0

    pbar = tqdm(total=max_bars, desc='生成中', unit='bar', ncols=80)

    for step in range(max_new_tokens):
        logits = model.forward(next_token, kv_caches=kv_caches)
        logits = logits[:, -1, :] / temperature  # (1, V)

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
    return generated


# -- token → MusicXML 导出 ------------------------------------

def save_to_musicxml(
    token_ids: list[int],
    tokenizer: REMITokenizer,
    output_path: str,
    max_bars: int = 256,
):
    """将 token 序列保存为 MusicXML 文件。

    复用 generate.py 中的 tokens_to_notes + notes_to_score。
    """
    from chopinote_model.generate import tokens_to_notes, notes_to_score

    notes = tokens_to_notes(token_ids, tokenizer)
    score = notes_to_score(notes, grid_size=tokenizer.grid_size, max_bars=max_bars)
    score.write('musicxml', fp=output_path)

    # 提取小节数信息
    num_bars = max(n['bar'] for n in notes) if notes else 0
    return num_bars


# -- 交互式参数收集 --------------------------------------------

def prompt_params(meta: Optional[dict] = None, seed_length: int = 0):
    """交互式收集生成参数，返回参数字典。"""
    print()
    print('参数设置（直接回车使用默认值）')
    print('-' * 40)

    max_bars_input = input(f'  最大续写小节数 [32]: ').strip()
    max_bars = int(max_bars_input) if max_bars_input else 32

    temp_input = input(f'  采样温度 (0.1~2.0, 越低越保守) [1.0]: ').strip()
    temperature = float(temp_input) if temp_input else 1.0

    topk_input = input(f'  Top-K (越小越集中) [20]: ').strip()
    top_k = int(topk_input) if topk_input else 20

    print()
    return {
        'max_bars': max_bars,
        'temperature': temperature,
        'top_k': top_k,
    }


# -- 输出路径 --------------------------------------------------

def make_output_path(input_path: str, custom: Optional[str] = None) -> str:
    """生成输出文件路径。"""
    if custom:
        out = Path(custom)
        out.parent.mkdir(parents=True, exist_ok=True)
        return str(out)

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    out = Path.cwd() / f'output_chopinote_{timestamp}.musicxml'
    return str(out)


# -- 主入口 ----------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Chopinote-AI 钢琴谱续写工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            '示例:\n'
            '  chopinote-generate ../autodl-fs/chopinote/checkpoints/best.pt input.musicxml\n'
            '  chopinote-generate ../autodl-fs/chopinote/checkpoints/step_50000.pt input.musicxml -o out.musicxml\n'
        ),
    )
    parser.add_argument('checkpoint', help='模型权重文件路径 (.pt)')
    parser.add_argument('input', help='输入 MusicXML 乐谱文件路径')
    parser.add_argument('-o', '--output', default=None,
                        help='输出 MusicXML 文件路径（默认: ./output_chopinote_时间戳.musicxml）')
    parser.add_argument('--seed-bars', type=int, default=16,
                        help='从输入曲谱末尾截取的小节数作为种子（默认: 16）')
    args = parser.parse_args()

    print('=== Chopinote-AI - 钢琴谱续写工具 ===')
    print()

    # -- 设备 ---------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}')
    print()

    # -- 第 1 步：加载模型 --------------------------------
    print('[1/4] 加载模型...')
    model, config, step, loss = load_model(args.checkpoint, device)
    print(f'      [OK] checkpoint: {args.checkpoint}')
    print(f'      | 训练步数:   {step}')
    if loss is not None:
        print(f'      | 保存时 loss: {loss:.4f}')
    print(f'      | 词表大小:   {config.vocab_size}')
    print(f'      | 模型参数量: {sum(p.numel() for p in model.parameters()):,}')
    print()

    # -- 第 2 步：解析输入 --------------------------------
    print('[2/4] 解析输入乐谱...')
    tokenizer = REMITokenizer(grid_size=16, velocity_levels=8)
    all_tokens = musicxml_to_seed(args.input, tokenizer)
    # 移除首尾 BOS/EOS
    content_tokens = [t for t in all_tokens
                      if t not in (tokenizer.bos_token_id, tokenizer.eos_token_id)]
    print(f'      [OK] 输入: {args.input}')
    print(f'      | 总 token 数: {len(all_tokens)}')
    print(f'      | 有效 token:  {len(content_tokens)}')

    # 截取末尾作为种子（保留完整小节）
    bar_id = tokenizer.bar_token_id
    bar_positions = [i for i, t in enumerate(content_tokens) if t == bar_id]
    seed_bars_count = min(args.seed_bars, len(bar_positions))
    if seed_bars_count > 0:
        cut_idx = bar_positions[-seed_bars_count]
        seed_tokens = content_tokens[cut_idx:]
        print(f'      | 种子小节:   {seed_bars_count} 小节 ({len(seed_tokens)} tokens)')
        input_bars = len(bar_positions)
        print(f'      | 输入曲谱共 {input_bars} 小节')
    else:
        seed_tokens = content_tokens
        print(f'      [!] 输入曲谱不足 1 小节，使用全部内容作为种子')

    # 确保 token 不超出模型词表范围
    oob = [t for t in seed_tokens if t >= config.vocab_size]
    if oob:
        print(f'      [!] {len(oob)} 个 token 超出模型词表 (vocab_size={config.vocab_size})，已替换为 MASK')
        mask_id = tokenizer.encode_token(tokenizer.MASK)
        seed_tokens = [t if t < config.vocab_size else mask_id for t in seed_tokens]
    print()

    # -- 第 3 步：参数设置 --------------------------------
    print('[3/4] 设置续写参数...')
    params = prompt_params(seed_length=len(seed_tokens))
    print()

    # -- 第 4 步：生成 ------------------------------------
    print('[4/4] 开始续写...')
    seed_tensor = torch.tensor([seed_tokens], dtype=torch.long, device=device)

    full_ids = generate_with_progress(
        model, seed_tensor, tokenizer,
        max_bars=params['max_bars'],
        temperature=params['temperature'],
        top_k=params['top_k'],
    )
    print()

    # -- 输出 ---------------------------------------------
    output_path = make_output_path(args.input, args.output)
    num_bars = save_to_musicxml(
        full_ids[0].tolist(), tokenizer, output_path,
        max_bars=params['max_bars'] + args.seed_bars,
    )

    print(f'[OK] 生成完成！')
    print(f'   输出文件: {os.path.abspath(output_path)}')
    print(f'   总小节数: {num_bars}')
    print()


if __name__ == '__main__':
    main()
