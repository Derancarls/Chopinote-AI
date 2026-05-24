#!/usr/bin/env python3
"""Batch section annotation using subprocess parallelism.
Works around multiprocessing.Pool overhead for 1.6M files.
"""
import sys, os, time, logging, subprocess, math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

TOKENS_DIR = '/root/autodl-tmp/data/processed/tokens_v3'
NUM_WORKERS = 8


def main():
    all_files = sorted(Path(TOKENS_DIR).glob('*.tokens'))
    total = len(all_files)
    logger.info(f"Token 文件总数: {total}")

    batch_size = math.ceil(total / NUM_WORKERS)
    logger.info(f"{NUM_WORKERS} workers, ~{batch_size} files/worker")

    # Remove old sec.json
    old = list(Path(TOKENS_DIR).glob('*.sec.json'))
    for f in old:
        f.unlink()
    logger.info(f"已清理 {len(old)} 个旧 sec.json")

    procs = []
    worker_script = str(Path(__file__).resolve().parent / 'annotate_sections_worker.py')

    for w in range(NUM_WORKERS):
        start = w * batch_size
        end = min(start + batch_size, total)
        if start >= total:
            break
        batch = all_files[start:end]
        file_list = f'/tmp/sec_batch_{w}.txt'
        with open(file_list, 'w') as f:
            for p in batch:
                f.write(str(p) + '\n')

        proc = subprocess.Popen([
            sys.executable, worker_script,
            '--file-list', file_list,
            '--output-dir', TOKENS_DIR,
        ])
        procs.append((proc, file_list, len(batch)))

    logger.info(f"启动了 {len(procs)} 个子进程")

    t0 = time.time()
    done_count = [0] * len(procs)
    while any(proc.poll() is None for proc, _, _ in procs):
        time.sleep(30)
        for i, (proc, _, bsize) in enumerate(procs):
            if proc.poll() is not None and done_count[i] == 0:
                done_count[i] = 1
        n_done = sum(done_count)
        sec_count = len(list(Path(TOKENS_DIR).glob('*.sec.json')))
        elapsed = time.time() - t0
        rate = sec_count / elapsed if elapsed > 0 else 0
        eta = (total - sec_count) / rate if rate > 0 else 0
        logger.info(f"  [{sec_count}/{total}] {sec_count/total*100:.1f}% "
                    f"workers_done={n_done}/{len(procs)} "
                    f"({rate:.0f}/s ETA {eta/60:.1f}min)")

    elapsed = time.time() - t0
    sec_count = len(list(Path(TOKENS_DIR).glob('*.sec.json')))
    logger.info(f"完成: {sec_count}/{total} 个段落标注 ({elapsed:.0f}s)")

    # Cleanup
    for _, fl, _ in procs:
        try:
            os.remove(fl)
        except Exception:
            pass


if __name__ == '__main__':
    main()
