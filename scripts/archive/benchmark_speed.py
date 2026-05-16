"""
训练速度 benchmark — 对比 baseline vs 优化后配置。

用法:
    # 快速对比 (baseline vs optimized)
    python scripts/benchmark_speed.py --data-dir /root/autodl-tmp/data/processed

    # 逐个优化增量对比
    python scripts/benchmark_speed.py --data-dir /root/autodl-tmp/data/processed --all
"""
import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import autocast

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

from chopinote_model.config import ModelConfig
from chopinote_model.model import MusicTransformer
from chopinote_model.dataset import TokenDataset, collate_fn

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger('benchmark')

WARMUP_STEPS = 10
BENCH_STEPS = 100


def get_vram_gb(device: torch.device) -> float:
    return torch.cuda.memory_allocated(device) / 1024**3


def run_benchmark(name: str, model: nn.Module, dataloader: DataLoader,
                  device: torch.device, overrides: dict) -> dict:
    """运行一组配置的 benchmark。"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    compile_enabled = overrides.get('compile', False)
    use_fp8 = overrides.get('use_fp8', False)
    gradient_ckpt = overrides.get('gradient_checkpointing', True)
    fused_adamw = overrides.get('fused_adamw', False)

    model.set_fp8_mode(use_fp8)
    model.set_gradient_checkpointing(gradient_ckpt)
    model.train()

    model_compiled = torch.compile(model) if compile_enabled else model

    params = list(dict.fromkeys(model_compiled.parameters()))
    optimizer = AdamW(params, lr=1.5e-4, weight_decay=0.1,
                      betas=(0.9, 0.95), fused=fused_adamw)

    step_times = []
    total_tokens = 0
    accum_steps = 16
    peak_vram = 0

    data_iter = iter(dataloader)
    total_steps = WARMUP_STEPS + BENCH_STEPS

    logger.info(f'  {name}: 预热 {WARMUP_STEPS} + 测量 {BENCH_STEPS} 步 ...')

    for step in range(total_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        mask = batch['attention_mask'].to(device)

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        with autocast('cuda', dtype=torch.bfloat16):
            logits = model_compiled(input_ids, mask)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1),
                ignore_index=-100, reduction='sum',
            )
            loss = loss / ((labels != -100).sum() + 1)
        loss = loss / accum_steps
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        if use_fp8:
            model.invalidate_fp8_caches()

        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0

        if step >= WARMUP_STEPS:
            step_times.append(elapsed)
            total_tokens += labels.numel()

        peak_vram = max(peak_vram, get_vram_gb(device))

        if step == 0 and compile_enabled:
            pass  # compilation happened during first step

    avg_step = sum(step_times) / len(step_times)
    tok_per_sec = total_tokens / sum(step_times)

    return {
        'name': name,
        'avg_step_s': avg_step,
        'tokens_per_sec': tok_per_sec,
        'vram_peak_gb': peak_vram,
    }


def format_table(results: list[dict]) -> str:
    baseline_tps = results[0]['tokens_per_sec']
    lines = []
    header = '┌' + '─' * 18 + '┬' + '─' * 10 + '┬' + '─' * 10 + '┬' + '─' * 10 + '┬' + '─' * 10 + '┐'
    lines.append(header)
    lines.append(f'│ {"Config":<16s} │ {"Step(s)":>8s} │ {"tok/s":>8s} │ {"VRAM(G)":>8s} │ {"vs Base":>8s} │')
    lines.append('├' + '─' * 18 + '┼' + '─' * 10 + '┼' + '─' * 10 + '┼' + '─' * 10 + '┼' + '─' * 10 + '┤')

    for r in results:
        speedup = r['tokens_per_sec'] / baseline_tps
        lines.append(
            f'│ {r["name"]:<16s} │ {r["avg_step_s"]:>8.3f} │ '
            f'{r["tokens_per_sec"]:>8.0f} │ {r["vram_peak_gb"]:>8.1f} │ '
            f'{speedup:>7.2f}x │'
        )
    lines.append('└' + '─' * 18 + '┴' + '─' * 10 + '┴' + '─' * 10 + '┴' + '─' * 10 + '┴' + '─' * 10 + '┘')
    return '\n'.join(lines)


def main():
    global WARMUP_STEPS, BENCH_STEPS

    parser = argparse.ArgumentParser(description='Chopinote-AI 训练速度 benchmark')
    parser.add_argument('--data-dir', type=str, default='/root/autodl-tmp/data/processed')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--steps', type=int, default=BENCH_STEPS)
    parser.add_argument('--warmup', type=int, default=WARMUP_STEPS)
    parser.add_argument('--all', action='store_true', default=False,
                        help='测试所有优化组合的增量效果')
    args = parser.parse_args()

    WARMUP_STEPS = args.warmup
    BENCH_STEPS = args.steps

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'设备: {device}')
    logger.info(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')

    data_dir = Path(args.data_dir)
    train_file = data_dir / 'train.txt'
    if not train_file.exists():
        logger.error(f'数据文件不存在: {train_file}')
        sys.exit(1)

    dataset = TokenDataset(str(train_file), data_dir=str(data_dir), max_seq_len=4096)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=False, collate_fn=collate_fn, drop_last=True,
    )

    if args.all:
        configs = [
            # (name, overrides dict)
            # overrides: compile, use_fp8, gradient_checkpointing, fused_adamw
            ('baseline',       dict(compile=False, use_fp8=False, ckpt=True,  fused=False)),
            ('+fp8',           dict(compile=False, use_fp8=True,  ckpt=True,  fused=False)),
            ('+fused_adamw',   dict(compile=False, use_fp8=False, ckpt=True,  fused=True)),
            ('-checkpointing', dict(compile=False, use_fp8=False, ckpt=False, fused=False)),
            ('+compile',       dict(compile=True,  use_fp8=False, ckpt=True,  fused=False)),
            ('all_optimized',  dict(compile=True,  use_fp8=True,  ckpt=False, fused=True)),
        ]
    else:
        configs = [
            ('baseline',  dict(compile=False, use_fp8=False, ckpt=True,  fused=False)),
            ('optimized', dict(compile=True,  use_fp8=True,  ckpt=False, fused=True)),
        ]

    results = []

    for name, overrides in configs:
        ckpt_flag = overrides.pop('ckpt')
        fused_flag = overrides.pop('fused')

        logger.info(f'{"="*60}')
        logger.info(f'Testing: {name}')
        logger.info(f'  compile={overrides.get("compile", False)} '
                    f'fp8={overrides.get("use_fp8", False)} '
                    f'checkpointing={ckpt_flag} fused_adamw={fused_flag}')

        torch.manual_seed(42)
        model_config = ModelConfig(gradient_checkpointing=ckpt_flag)
        model = MusicTransformer(model_config).to(device)
        for p in model.parameters():
            p.data = p.data.to(torch.bfloat16)

        # pass fused flag through overrides
        overrides['fused_adamw'] = fused_flag
        result = run_benchmark(name, model, dataloader, device, overrides)
        results.append(result)

        del model
        torch.cuda.empty_cache()

    print('\n' + format_table(results))

    baseline = results[0]
    for r in results[1:]:
        speedup = r['tokens_per_sec'] / baseline['tokens_per_sec']
        vram_delta = r['vram_peak_gb'] - baseline['vram_peak_gb']
        print(f'{r["name"]}: {speedup:.2f}x 加速, VRAM {vram_delta:+.1f}G')


if __name__ == '__main__':
    main()
