"""
PDMX 全量重跑（v3）
"""
import sys, os, time, logging, json, pickle
from pathlib import Path
from multiprocessing import Pool, cpu_count

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = '/root/autodl-tmp/data/processed'
PDMX_DIR = '/root/autodl-tmp/data/raw/PDMX/data'
CACHE_DIR = '/root/Chopinote-AI/data/cache'

# ── Step 1: 清空旧 PDMX token + stale cache ──────────────────────

def clean_old_pdmx():
    logger.info("清空旧 PDMX token 和缓存...")
    token_dir = f'{DATA_DIR}/tokens_v4'
    meta_dir = f'{DATA_DIR}/metadata_v4'
    os.makedirs(token_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    # 删除旧 PDMX token（通过 metadata 中 file_path 含 /PDMX/ 判断）
    n_tok = 0
    n_meta = 0
    for f in os.listdir(meta_dir):
        if not f.endswith('.meta.json'):
            continue
        mp = os.path.join(meta_dir, f)
        try:
            with open(mp) as fh:
                md = json.load(fh)
        except Exception:
            continue
        if '/PDMX/' not in md.get('file_path', ''):
            continue
        # 删 metadata
        os.remove(mp)
        n_meta += 1
        # 删对应 token
        tok_name = f.replace('.meta.json', '.tokens')
        tp = os.path.join(token_dir, tok_name)
        if os.path.exists(tp):
            os.remove(tp)
            n_tok += 1

    logger.info(f"  删除 {n_tok} 个旧 token, {n_meta} 个旧 metadata")

    # 删除 PDMX 缓存
    n_cache = 0
    if os.path.isdir(CACHE_DIR):
        for f in os.listdir(CACHE_DIR):
            fpath = os.path.join(CACHE_DIR, f)
            try:
                with open(fpath, 'rb') as fh:
                    d = pickle.load(fh)
            except Exception:
                continue
            if '/PDMX/' in d.get('original_path', ''):
                os.remove(fpath)
                n_cache += 1
    logger.info(f"  删除 {n_cache} 个 PDMX 缓存")

# ── Step 2: 并行处理 ────────────────────────────────────────────

def init_worker():
    global _proc
    from chopinote_dataset.processor import PDMXPreprocessor
    _proc = PDMXPreprocessor(config_path='/root/Chopinote-AI/config.yaml')

def process_one(fpath):
    global _proc
    try:
        r = _proc.process_file(fpath, DATA_DIR)
        if r:
            return (fpath, 'converted', r.get('num_tokens', 0))
        else:
            return (fpath, 'skipped', None)
    except Exception as e:
        return (fpath, 'failed', str(e)[:200])

def run_pdmx():
    files = []
    for root, dirs, fnames in os.walk(PDMX_DIR):
        dirs[:] = [d for d in dirs if d != 'metadata']
        for f in fnames:
            if f.endswith('.json') and not f.endswith('.hash'):
                files.append(os.path.join(root, f))

    total = len(files)
    logger.info(f"PDMX 源文件总数: {total}")
    logger.info(f"Worker: {min(16, cpu_count())}")
    t0 = time.time()

    converted = 0
    skipped = 0
    failed = 0
    n_workers = min(25, cpu_count())

    with Pool(n_workers, initializer=init_worker) as pool:
        for i, (fpath, status, info) in enumerate(pool.imap_unordered(process_one, files, chunksize=200)):
            if status == 'converted':
                converted += 1
            elif status == 'skipped':
                skipped += 1
            else:
                failed += 1
                if failed <= 5:
                    logger.warning(f"  ERROR: {Path(fpath).parent.name}/{Path(fpath).name} -> {info}")

            if (i + 1) % 50000 == 0 or (i + 1) == total:
                elapsed = time.time() - t0
                assert converted + skipped + failed == i + 1
                logger.info(
                    f"  [{i+1}/{total}] "
                    f"✓{converted} skipped↓{skipped} ✗{failed} "
                    f"({elapsed:.0f}s)"
                )

    elapsed = time.time() - t0
    assert converted + skipped + failed == total, \
        f"计数不一致: {converted}+{skipped}+{failed} != {total}"
    logger.info(
        f"PDMX 完成: 总共 {total} → ✓{converted} skipped↓{skipped} ✗{failed} "
        f"({elapsed:.0f}s)"
    )

if __name__ == '__main__':
    clean_old_pdmx()
    run_pdmx()
