"""
全量重跑 PDMX + MusicXML 预处理，最快速度。
- 有缓存（文件 hash 命中）→ 跳过
- 无缓存 → 处理并写入
"""
import sys, os, time, logging, json
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

DATA_DISK = '/root/autodl-tmp'
OUTPUT_DIR = f'{DATA_DISK}/data/processed'

# ── PDMX ──────────────────────────────────────────────────────────
PDMX_DIR = '/root/autodl-tmp/chopinote_data/data/raw/PDMX'

def _init_pdmx_worker():
    global _pdmx_proc
    from chopinote_dataset.processor import PDMXPreprocessor
    _pdmx_proc = PDMXPreprocessor(config_path='/root/Chopinote-AI/config.yaml')

def _process_one_pdmx(fpath):
    global _pdmx_proc
    try:
        r = _pdmx_proc.process_file(fpath, OUTPUT_DIR)
        return (fpath, True, None)
    except Exception as e:
        return (fpath, False, str(e))

def run_pdmx():
    logger.info("=" * 60)
    logger.info("PDMX 全量重跑")
    logger.info("=" * 60)

    # 收集所有 PDMX JSON
    files = []
    for root, dirs, fnames in os.walk(PDMX_DIR):
        dirs[:] = [d for d in dirs if d != 'metadata']
        for f in fnames:
            if f.endswith('.json') and not f.endswith('.json.hash'):
                files.append(os.path.join(root, f))

    logger.info(f"PDMX 源文件: {len(files)}")
    t0 = time.time()

    n_workers = min(16, cpu_count())
    logger.info(f"Worker: {n_workers}")

    ok = 0
    fail = 0
    with Pool(n_workers, initializer=_init_pdmx_worker) as pool:
        for i, (fpath, success, err) in enumerate(pool.imap_unordered(_process_one_pdmx, files, chunksize=200)):
            if success:
                ok += 1
            else:
                fail += 1
                if fail <= 5:
                    logger.warning(f"  FAIL: {fpath} -> {err}")
            if (i + 1) % 20000 == 0 or (i + 1) == len(files):
                logger.info(f"  PDMX: {i+1}/{len(files)} | OK={ok} FAIL={fail} ({time.time()-t0:.0f}s)")

    elapsed = time.time() - t0
    logger.info(f"PDMX 完成: {ok} OK, {fail} FAIL, {elapsed:.0f}s")

# ── MusicXML ──────────────────────────────────────────────────────
MUSICXML_DIRS = [
    '/root/autodl-tmp/asap',
    '/root/autodl-tmp/ATEPP-1.2',
    '/root/autodl-tmp/openscore_lieder',
    '/root/autodl-tmp/openscore_string_quartets',
]

def _init_musicxml_worker():
    global _mxl_proc
    from chopinote_dataset.processor import MusicXMLPreprocessor
    _mxl_proc = MusicXMLPreprocessor(config_path='/root/Chopinote-AI/config.yaml')

def _find_musicxml_files(directory):
    exts = {'.musicxml', '.mxl', '.xml'}
    files = []
    for root, dirs, fnames in os.walk(directory):
        for f in fnames:
            if Path(f).suffix.lower() in exts:
                files.append(os.path.join(root, f))
    return files

def _process_one_musicxml(fpath):
    global _mxl_proc
    try:
        r = _mxl_proc.process_file(fpath, OUTPUT_DIR)
        return (fpath, r is not None, None if r else 'quality/seq filter')
    except Exception as e:
        return (fpath, False, str(e))

def run_musicxml():
    logger.info("=" * 60)
    logger.info("MusicXML 全量重跑")
    logger.info("=" * 60)

    all_files = []
    for d in MUSICXML_DIRS:
        if os.path.isdir(d):
            fs = _find_musicxml_files(d)
            logger.info(f"  {os.path.basename(d)}: {len(fs)} 文件")
            all_files.extend(fs)
        else:
            logger.warning(f"  目录不存在，跳过: {d}")

    logger.info(f"MusicXML 总计: {len(all_files)}")

    if not all_files:
        logger.info("无 MusicXML 文件需要处理")
        return

    t0 = time.time()
    # music21 may have thread issues, use fewer workers
    n_workers = min(4, cpu_count())
    logger.info(f"Worker: {n_workers}")

    ok = 0
    fail = 0
    with Pool(n_workers, initializer=_init_musicxml_worker) as pool:
        for i, (fpath, success, err) in enumerate(pool.imap_unordered(_process_one_musicxml, all_files, chunksize=10)):
            if success:
                ok += 1
            else:
                fail += 1
                if fail <= 10:
                    logger.warning(f"  FAIL: {os.path.basename(fpath)} -> {err}")
            if (i + 1) % 200 == 0 or (i + 1) == len(all_files):
                logger.info(f"  MusicXML: {i+1}/{len(all_files)} | OK={ok} FAIL={fail} ({time.time()-t0:.0f}s)")

    elapsed = time.time() - t0
    logger.info(f"MusicXML 完成: {ok} OK, {fail} FAIL, {elapsed:.0f}s")

# ── 更新 splits ──────────────────────────────────────────────────
def regenerate_splits():
    logger.info("=" * 60)
    logger.info("重新生成 train/val/test 划分")
    logger.info("=" * 60)

    token_dir = f'{OUTPUT_DIR}/tokens'
    token_files = sorted(f for f in os.listdir(token_dir) if f.endswith('.tokens'))
    logger.info(f"总 token 文件: {len(token_files)}")

    import random
    random.seed(42)
    shuffled = [os.path.join(token_dir, f) for f in token_files]
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    splits = {
        'train.txt': shuffled[:n_train],
        'val.txt': shuffled[n_train:n_train + n_val],
        'test.txt': shuffled[n_train + n_val:],
    }

    for name, files in splits.items():
        path = os.path.join(OUTPUT_DIR, name)
        with open(path, 'w', encoding='utf-8') as f:
            for fp in files:
                f.write(fp + '\n')
        logger.info(f"  {name}: {len(files)} 文件")

    logger.info(f"划分完成: train={n_train}, val={n_val}, test={n - n_train - n_val}")

# ── Main ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    overall_start = time.time()

    # 先跑 PDMX (快)
    run_pdmx()

    # 再跑 MusicXML (慢)
    run_musicxml()

    # 最后更新划分
    regenerate_splits()

    total = time.time() - overall_start
    logger.info(f"全部完成！总耗时 {total:.0f}s ({total/60:.1f}min)")
