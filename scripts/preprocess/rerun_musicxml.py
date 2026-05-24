#!/usr/bin/env python3
"""MusicXML 全量预处理 v3。

Usage: python scripts/rerun_musicxml.py
"""
import sys, os, time, logging, json, pickle
from pathlib import Path
from multiprocessing import Pool, cpu_count

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = '/root/autodl-tmp/data/processed'
CACHE_DIR = '/root/Chopinote-AI/data/cache'
MUSICXML_DIRS = [
    '/root/autodl-tmp/data/raw/MusicXML/asap',
    '/root/autodl-tmp/data/raw/MusicXML/ATEPP-1.2',
    '/root/autodl-tmp/data/raw/MusicXML/openscore_lieder',
    '/root/autodl-tmp/data/raw/MusicXML/openscore_string_quartets',
    '/root/autodl-tmp/data/raw/MusicXML/music21_corpus',
]

# ── Step 1: 清空旧 MusicXML token + stale cache ────────────────────

def clean_old_musicxml():
    logger.info("清空旧 MusicXML token 和缓存...")
    token_dir = f'{DATA_DIR}/tokens_v3'
    meta_dir = f'{DATA_DIR}/metadata_v3'
    os.makedirs(token_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    # 删除旧 MusicXML token（通过 metadata 中 file_path 含 /MusicXML/ 判断）
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
        if '/MusicXML/' not in md.get('file_path', ''):
            continue
        os.remove(mp)
        n_meta += 1
        tok_name = f.replace('.meta.json', '.tokens')
        tp = os.path.join(token_dir, tok_name)
        if os.path.exists(tp):
            os.remove(tp)
            n_tok += 1

    logger.info(f"  删除 {n_tok} 个旧 token, {n_meta} 个旧 metadata")

    # 删除 MusicXML 缓存
    n_cache = 0
    if os.path.isdir(CACHE_DIR):
        for f in os.listdir(CACHE_DIR):
            fpath = os.path.join(CACHE_DIR, f)
            try:
                with open(fpath, 'rb') as fh:
                    d = pickle.load(fh)
            except Exception:
                continue
            if '/MusicXML/' in d.get('original_path', ''):
                os.remove(fpath)
                n_cache += 1
    logger.info(f"  删除 {n_cache} 个 MusicXML 缓存")

# ── Step 2: 并行处理 ──────────────────────────────────────────────

def find_musicxml_files(dirs):
    files = []
    for d in dirs:
        for root, _, fnames in os.walk(d):
            for fn in fnames:
                if fn.endswith(('.musicxml', '.xml', '.mxl')):
                    files.append(os.path.join(root, fn))
    return sorted(files)


def init_worker():
    global _proc
    from chopinote_dataset.processor import MusicXMLPreprocessor
    _proc = MusicXMLPreprocessor(config_path='/root/Chopinote-AI/config.yaml')


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


def run_musicxml():
    files = find_musicxml_files(MUSICXML_DIRS)
    total = len(files)
    logger.info(f"MusicXML 源文件总数: {total}")
    if not files:
        logger.warning("无 MusicXML 文件可处理")
        return

    n_workers = min(cpu_count(), 8)
    logger.info(f"Worker: {n_workers}")

    converted = 0
    skipped = 0
    failed = 0
    t0 = time.time()

    with Pool(processes=n_workers, initializer=init_worker) as pool:
        for i, (fpath, status, info) in enumerate(pool.imap_unordered(process_one, files, chunksize=4)):
            if status == 'converted':
                converted += 1
            elif status == 'skipped':
                skipped += 1
            else:
                failed += 1
                if failed <= 10:
                    logger.warning(f"  FAIL: {fpath}: {info}")

            done = i + 1
            if done % 500 == 0 or done == total:
                elapsed = time.time() - t0
                assert converted + skipped + failed == done
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                logger.info(
                    f"  [{done}/{total}] "
                    f"✓{converted} skipped↓{skipped} ✗{failed} "
                    f"({rate:.1f}/s ETA {eta/60:.0f}min)"
                )

    elapsed = time.time() - t0
    assert converted + skipped + failed == total, \
        f"计数不一致: {converted}+{skipped}+{failed} != {total}"
    logger.info(
        f"MusicXML 完成: 总共 {total} → ✓{converted} skipped↓{skipped} ✗{failed} "
        f"({elapsed:.0f}s)"
    )


if __name__ == '__main__':
    clean_old_musicxml()
    run_musicxml()
