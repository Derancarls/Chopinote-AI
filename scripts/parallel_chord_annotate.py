"""22 核并行和弦标注包装器（含调性覆盖统计）。"""
import argparse
import json
import logging
import multiprocessing
import os
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.chord_annotator import process_file, _key_coverage_stats


def worker(args: tuple) -> tuple:
    """Worker: (token_path_str, output_suffix) -> (name, ok, msg_or_stats)"""
    token_path, output_suffix = args
    try:
        process_file(token_path, output_suffix)
        # 调性覆盖统计
        with open(token_path) as f:
            token_ids = json.load(f)
        coverage = _key_coverage_stats(token_ids)
        return (Path(token_path).name, True, coverage)
    except Exception as e:
        return (Path(token_path).name, False, str(e))


def main():
    parser = argparse.ArgumentParser(description='22 核并行和弦标注')
    parser.add_argument('--tokens-dir', default='/root/autodl-tmp/data/processed/tokens_v3')
    parser.add_argument('--output-suffix', default='.chord.json')
    parser.add_argument('--num-workers', type=int, default=22)
    parser.add_argument('--resume', action='store_true',
                        help='跳过已有 .chord.json 的文件')
    args = parser.parse_args()

    tokens_dir = Path(args.tokens_dir)
    token_files = sorted(tokens_dir.glob('*.tokens'))
    total = len(token_files)
    logger.info(f'找到 {total} 个 token 文件')

    if args.resume:
        existing = {f.stem for f in tokens_dir.glob(f'*{args.output_suffix}')}
        todo = [str(f) for f in token_files if f.stem not in existing]
        logger.info(f'已有 {len(existing)} 个 .chord.json 文件，跳过，待处理 {len(todo)} 个')
    else:
        todo = [str(f) for f in token_files]

    if not todo:
        logger.info('全部已完成，无需处理')
        return

    n_workers = min(args.num_workers, len(todo))
    logger.info(f'启动 {n_workers} 个 worker...')

    task_args = [(f, args.output_suffix) for f in todo]
    start_time = time.time()

    ok = 0
    fail = 0
    fail_log = []

    # 调性覆盖统计汇总
    total_bars = 0
    total_keyless_bars = 0
    files_with_keyless_bars = 0

    with multiprocessing.Pool(n_workers) as pool:
        for i, (name, success, data) in enumerate(
            pool.imap_unordered(worker, task_args, chunksize=500),
            start=1
        ):
            if success:
                ok += 1
                coverage = data
                total_bars += coverage['total_bars']
                total_keyless_bars += coverage['keyless_bars']
                if coverage['keyless_bars'] > 0:
                    files_with_keyless_bars += 1
            else:
                fail += 1
                if fail <= 20:
                    fail_log.append((name, data))

            if i % 50000 == 0 or i == len(todo):
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                remaining = (len(todo) - i) / rate if rate > 0 else 0
                logger.info(
                    f'  [{i}/{len(todo)}] ✓{ok} ✗{fail}  '
                    f'{rate:.0f} files/s, ETA {remaining/60:.1f}min'
                )

    elapsed = time.time() - start_time
    logger.info(f'完成! ✓{ok} ✗{fail} / {len(todo)} 耗时 {elapsed/60:.1f}min')

    # 汇总调性覆盖报告
    if total_bars > 0:
        keyless_pct = total_keyless_bars / total_bars * 100
        logger.info(
            f'调性覆盖报告: {total_keyless_bars}/{total_bars} 小节缺 Key '
            f'({keyless_pct:.2f}%), 涉及 {files_with_keyless_bars} 个文件'
        )
        if total_keyless_bars > 0:
            logger.warning(
                '预处理或标注管线存在缺口 — 有小节缺少 <Key> token'
            )

    if fail_log:
        logger.info(f'失败文件样例 ({len(fail_log)}):')
        for name, err in fail_log[:10]:
            logger.info(f'  {name}: {err}')


if __name__ == '__main__':
    main()
