"""
PDMX 全量重跑（修正版）
- 每个文件正确报告成功/失败
- 源文件一个不漏
"""
import sys, os, time, logging, json
from pathlib import Path
from multiprocessing import Pool, cpu_count

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = '/root/autodl-tmp/data/processed'
PDMX_DIR = '/root/autodl-tmp/data/raw/PDMX/data'

# ── Step 1: 清空旧的 PDMX token ──────────────────────────────
def clean_old_pdmx():
    logger.info("清空旧 PDMX token...")
    token_dir = f'{DATA_DIR}/tokens_v3'
    meta_dir = f'{DATA_DIR}/metadata_v3'
    os.makedirs(token_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    n = 0
    for f in os.listdir(token_dir):
        if f.startswith('Qm'):
            os.remove(os.path.join(token_dir, f))
            n += 1
    m = 0
    for f in os.listdir(meta_dir):
        if f.startswith('Qm'):
            os.remove(os.path.join(meta_dir, f))
            m += 1
    logger.info(f"  删除 {n} tokens, {m} metadata")

# ── Step 2: 并行处理 PDMX ────────────────────────────────────
def init_worker():
    global _proc
    from chopinote_dataset.processor import PDMXPreprocessor
    _proc = PDMXPreprocessor(config_path='/root/Chopinote-AI/config.yaml')

def process_one(fpath):
    global _proc
    try:
        r = _proc.process_file(fpath, DATA_DIR)
        if r:
            return (fpath, 'ok', None)
        else:
            return (fpath, 'skip', None)
    except Exception as e:
        return (fpath, 'error', str(e))

def run_pdmx():
    # 收集源文件
    files = []
    for root, dirs, fnames in os.walk(PDMX_DIR):
        dirs[:] = [d for d in dirs if d != 'metadata']
        for f in fnames:
            if f.endswith('.json') and not f.endswith('.hash'):
                files.append(os.path.join(root, f))

    logger.info(f"PDMX 源文件: {len(files)}")

    n_workers = min(16, cpu_count())
    logger.info(f"Worker: {n_workers}")
    t0 = time.time()

    ok = 0
    skip = 0
    err = 0
    with Pool(n_workers, initializer=init_worker) as pool:
        for i, (fpath, status, msg) in enumerate(pool.imap_unordered(process_one, files, chunksize=200)):
            if status == 'ok':
                ok += 1
            elif status == 'skip':
                skip += 1
            else:
                err += 1
                if err <= 5:
                    logger.warning(f"  ERROR: {Path(fpath).parent.name}/{Path(fpath).name} -> {msg}")
            if (i+1) % 50000 == 0 or (i+1) == len(files):
                logger.info(f"  {i+1}/{len(files)} | OK={ok} SKIP={skip} ERR={err} ({time.time()-t0:.0f}s)")

    elapsed = time.time() - t0
    logger.info(f"PDMX 完成: OK={ok} SKIP={skip} ERR={err} ({elapsed:.0f}s)")

    # 验证各目录
    logger.info("按目录统计...")
    meta_dir = f'{DATA_DIR}/metadata_v3'
    dirs = {}
    for fname in os.listdir(meta_dir):
        if not fname.startswith('Qm'):
            continue
        with open(os.path.join(meta_dir, fname)) as f:
            md = json.load(f)
        fp = md.get('file_path', '')
        for i, p in enumerate(fp.split('/')):
            if p == 'PDMX' and i+2 < len(fp.split('/')):
                l1 = fp.split('/')[i+2]
                dirs[l1] = dirs.get(l1, 0) + 1
                break
    for d in sorted(dirs):
        logger.info(f"  PDMX/{d}: {dirs[d]} tokens")

if __name__ == '__main__':
    clean_old_pdmx()
    run_pdmx()
