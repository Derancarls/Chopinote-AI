#!/usr/bin/env python3
"""批量生成+评价脚本 — 自动填充 reward_log，为 DPO 自动触发提供偏好数据。

两种用法:
  1. 独立运行: python batch_evaluate.py --checkpoint step_50000.pt --seeds seeds.txt
  2. 被 train.py 内联调用: run_batch_evaluation(model, tokenizer, ...)

流程:
  seed MusicXML × temperature → ABC Engine 生成 → 保存 .musicxml + .tokens
  → parsing Score → C evaluate_generation() → write_reward_log
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from chopinote_dataset.tokenizer import REMITokenizer
from chopinote_dataset.converter import MusicXMLToREMI

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
)
logger = logging.getLogger('batch_eval')


@dataclass
class EvalResult:
    seed_path: str
    temperature: float
    musicxml_path: str
    tokens_path: str
    total_bars: int
    total_tokens: int
    total_score: float
    novelty: float
    diversity: float
    elapsed: float
    error: str | None = None


def run_batch_evaluation(
    model,
    tokenizer,
    seeds: list[str],
    temperatures: list[float],
    samples_per_seed: int = 2,
    max_bars: int = 48,
    output_dir: str = '',
    reward_log_path: str = '',
) -> list[EvalResult]:
    """核心批量评估逻辑 — train.py 可直接调用（复用已加载的模型）。

    Args:
        model: MusicTransformer (已加载到 GPU, eval 模式)
        tokenizer: REMITokenizer
        seeds: 种子 MusicXML 路径列表
        temperatures: 温度档位列表
        samples_per_seed: 每种子的重复生成次数
        max_bars: 生成长度
        output_dir: 输出目录（MusicXML + .tokens）
        reward_log_path: reward_log 路径

    Returns:
        EvalResult 列表
    """
    from chopinote_model.generate import stage3_iterative_generate
    from chopinote_abc.scoring import evaluate_generation
    from chopinote_abc.parser import parse_musicxml
    from chopinote_abc.database import write_reward_log
    from chopinote_cli.main import save_to_musicxml

    device = next(model.parameters()).device
    converter = MusicXMLToREMI(grid_size=16, velocity_levels=8)
    os.makedirs(output_dir, exist_ok=True)
    if reward_log_path:
        os.makedirs(os.path.dirname(reward_log_path) or '.', exist_ok=True)

    results: list[EvalResult] = []
    n_total = len(seeds) * len(temperatures) * samples_per_seed
    n_done = 0

    for seed_idx, seed_path in enumerate(seeds):
        if not os.path.isfile(seed_path):
            logger.warning("Seed 不存在，跳过: %s", seed_path)
            continue

        # 转换 seed
        try:
            seed_tokens = converter.convert(seed_path)
        except Exception as e:
            logger.error("Seed 转换失败 %s: %s", seed_path, e)
            continue

        seed_bars = sum(1 for t in seed_tokens if t == tokenizer.bar_token_id)
        seed_name = os.path.splitext(os.path.basename(seed_path))[0]
        logger.info("Seed [%d/%d] %s (%d tokens, %d bars)",
                    seed_idx + 1, len(seeds), seed_name, len(seed_tokens), seed_bars)

        for temp in temperatures:
            for sample_i in range(samples_per_seed):
                n_done += 1
                ts = int(time.time())
                out_xml = os.path.join(
                    output_dir,
                    f'eval_{seed_name}_T{temp:.1f}_s{sample_i}_{ts}.musicxml')

                logger.info("[%d/%d] %s T=%.1f sample=%d → %s",
                            n_done, n_total, seed_name, temp, sample_i,
                            os.path.basename(out_xml))

                t0 = time.time()
                error_msg = None
                all_tokens = None

                try:
                    with torch.no_grad():
                        all_tokens, report = stage3_iterative_generate(
                            model, tokenizer, seed_tokens,
                            max_bars=max_bars,
                            form='free',
                            max_retries=2,
                            base_temperature=temp,
                            top_k=20,
                        )
                except Exception as e:
                    error_msg = f"生成异常: {e}"
                    logger.error(error_msg)
                    results.append(EvalResult(
                        seed_path=seed_path, temperature=temp,
                        musicxml_path='', tokens_path='',
                        total_bars=0, total_tokens=0,
                        total_score=0.0, novelty=0.0, diversity=0.0,
                        elapsed=time.time() - t0, error=error_msg,
                    ))
                    continue

                elapsed = time.time() - t0
                total_bars = all_tokens.count(tokenizer.bar_token_id) if all_tokens else 0
                gen_bars = total_bars - seed_bars

                # 保存 MusicXML + tokens
                try:
                    save_to_musicxml(all_tokens, tokenizer, out_xml,
                                     total_bars, save_tokens=True)
                except Exception as e:
                    error_msg = f"保存失败: {e}"
                    logger.error(error_msg)

                tok_path = out_xml.replace('.musicxml', '.tokens')

                # C 评分 — 读取 MusicXML 做 Score 级评价
                novelty = report.get('novelty_score', 0.0) if report else 0.0
                diversity = report.get('diversity_score', 0.0) if report else 0.0
                total_score = 0.0

                if not error_msg and os.path.isfile(out_xml):
                    try:
                        score_obj = parse_musicxml(out_xml)
                        eval_report = evaluate_generation(
                            all_tokens, tokenizer,
                            seed_tokens=seed_tokens,
                            score=score_obj,
                            novelty_bonus=novelty,
                            diversity_bonus=diversity,
                            structural_fixes=report.get('structural_fixes', []) if report else [],
                        )
                        total_score = eval_report.total_score
                    except Exception as e:
                        logger.warning("评分异常: %s", e)
                        total_score = 0.3  # fallback

                # 写入 reward_log
                if reward_log_path and all_tokens:
                    try:
                        write_reward_log(
                            output_path=reward_log_path,
                            report=report or {},
                            novelty=novelty,
                            diversity=diversity,
                            seed_path=seed_path,
                            musicxml_path=out_xml,
                            total_score=total_score,
                            seed_bars=seed_bars,
                        )
                    except Exception as e:
                        logger.warning("写 reward_log 失败: %s", e)

                logger.info("  ✓ bars=%d(gen=%d) tok=%d score=%.3f time=%.1fs",
                            total_bars, gen_bars,
                            len(all_tokens) if all_tokens else 0,
                            total_score, elapsed)

                results.append(EvalResult(
                    seed_path=seed_path, temperature=temp,
                    musicxml_path=out_xml, tokens_path=tok_path,
                    total_bars=total_bars,
                    total_tokens=len(all_tokens) if all_tokens else 0,
                    total_score=total_score, novelty=novelty,
                    diversity=diversity, elapsed=elapsed,
                ))

    # 汇总
    success = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]
    if success:
        scores = [r.total_score for r in success]
        logger.info("=" * 50)
        logger.info("批量评估完成: %d 成功, %d 失败", len(success), len(failed))
        logger.info("分数: min=%.3f max=%.3f mean=%.3f",
                    min(scores), max(scores), sum(scores) / len(scores))
        logger.info("总耗时: %.0fs", sum(r.elapsed for r in results))
        if reward_log_path:
            logger.info("Reward log: %s", reward_log_path)

    return results


# ═══════════════════════════════════════════════════════════════
#  独立 CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='批量生成 + C 评价，填充 reward_log 供 DPO 使用')
    parser.add_argument('--checkpoint', required=True,
                        help='模型 checkpoint 路径')
    parser.add_argument('--seeds', required=True,
                        help='种子列表文件（一行一个 MusicXML 路径）')
    parser.add_argument('--temperatures', default='0.9,1.0,1.1',
                        help='温度档位，逗号分隔 (default: 0.9,1.0,1.1)')
    parser.add_argument('--samples-per-seed', type=int, default=2,
                        help='每温度的重复次数 (default: 2)')
    parser.add_argument('--max-bars', type=int, default=48,
                        help='生成长度 (default: 48)')
    parser.add_argument('--output-dir', default='',
                        help='输出目录 (default: checkpoint 同级的 eval_output/)')
    parser.add_argument('--reward-log', default='',
                        help='reward_log 路径 (default: 从环境变量或默认路径)')

    args = parser.parse_args()

    # 加载模型
    from chopinote_model.config import ModelConfig
    from chopinote_model.model import MusicTransformer

    if not os.path.isfile(args.checkpoint):
        logger.error("Checkpoint 不存在: %s", args.checkpoint)
        sys.exit(1)

    logger.info("加载 checkpoint: %s", args.checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = ModelConfig()
    model = MusicTransformer(cfg)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model = model.bfloat16().to(device)
    model.eval()
    logger.info("模型: %.2fB params, step=%s loss=%.4f",
                sum(p.numel() for p in model.parameters()) / 1e9,
                ckpt.get('step', '?'), ckpt.get('loss', 0.0))

    tokenizer = REMITokenizer()

    # 读种子列表
    if not os.path.isfile(args.seeds):
        logger.error("种子列表不存在: %s", args.seeds)
        sys.exit(1)

    with open(args.seeds) as f:
        seeds = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    logger.info("种子: %d 个", len(seeds))

    temperatures = [float(x.strip()) for x in args.temperatures.split(',')]
    logger.info("温度: %s", temperatures)

    # 输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        ckpt_dir = os.path.dirname(args.checkpoint)
        output_dir = os.path.join(ckpt_dir, 'eval_output')

    # reward log
    reward_log = args.reward_log or os.environ.get(
        'CHOPINOTE_REWARD_DIR',
        '/root/autodl-tmp/chopinote/rewards')
    if not reward_log.endswith('.jsonl'):
        reward_log = os.path.join(reward_log, 'reward_log.jsonl')

    results = run_batch_evaluation(
        model, tokenizer, seeds, temperatures,
        samples_per_seed=args.samples_per_seed,
        max_bars=args.max_bars,
        output_dir=output_dir,
        reward_log_path=reward_log,
    )

    # 输出 JSON summary
    summary = {
        'checkpoint': args.checkpoint,
        'total': len(results),
        'success': len([r for r in results if r.error is None]),
        'failed': len([r for r in results if r.error is not None]),
        'results': [
            {
                'seed': r.seed_path,
                'temperature': r.temperature,
                'musicxml': r.musicxml_path,
                'bars': r.total_bars,
                'tokens': r.total_tokens,
                'score': r.total_score,
                'elapsed': r.elapsed,
                'error': r.error,
            }
            for r in results
        ],
    }
    summary_path = os.path.join(output_dir, 'eval_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary: %s", summary_path)


if __name__ == '__main__':
    main()
