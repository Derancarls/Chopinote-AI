#!/usr/bin/env python3
"""MusicXML 全量预处理 — 使用 MusicXMLToREMI 转换器。

Usage: python scripts/rerun_musicxml.py
"""
import sys, os, time, logging, json
from pathlib import Path
from multiprocessing import Pool, cpu_count

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = '/root/autodl-tmp/data/processed'
MUSICXML_DIRS = [
    '/root/autodl-tmp/data/raw/MusicXML/asap',
    '/root/autodl-tmp/data/raw/MusicXML/ATEPP-1.2',
    '/root/autodl-tmp/data/raw/MusicXML/openscore_lieder',
    '/root/autodl-tmp/data/raw/MusicXML/openscore_string_quartets',
    '/root/autodl-tmp/data/raw/MusicXML/music21_corpus',
]

# ── Step 1: 清空旧的 MusicXML token ────────────────────────────

def clean_old_musicxml():
    """删除旧 MusicXML token（仅删除 source_format 为 musicxml 的文件，保护 MIDI/PDMX）。"""
    import json as _json
    logger.info("清空旧 MusicXML token...")
    token_dir = f'{DATA_DIR}/tokens_v3'
    meta_dir = f'{DATA_DIR}/metadata_v3'
    os.makedirs(token_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    n = 0
    m = 0
    for f in os.listdir(meta_dir):
        if not f.endswith('.meta.json'):
            continue
        meta_path = os.path.join(meta_dir, f)
        try:
            with open(meta_path) as mf:
                md = _json.load(mf)
        except Exception:
            continue
        if md.get('source_format') != 'musicxml':
            continue
        token_name = f.replace('.meta.json', '.tokens')
        token_path = os.path.join(token_dir, token_name)
        if os.path.exists(token_path):
            os.remove(token_path)
            n += 1
        os.remove(meta_path)
        m += 1
    logger.info(f"  删除 {n} tokens, {m} metadata")

# ── Step 2: 并行处理 MusicXML ──────────────────────────────────

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
            return (fpath, 'ok', r.get('num_tokens', 0))
        else:
            return (fpath, 'skip', None)
    except Exception as e:
        return (fpath, 'error', str(e))


def run_musicxml():
    files = find_musicxml_files(MUSICXML_DIRS)
    logger.info(f"找到 {len(files)} 个 MusicXML 文件")
    if not files:
        logger.warning("无 MusicXML 文件可处理")
        return

    n_workers = min(cpu_count(), 8)
    logger.info(f"并行 workers: {n_workers}")

    ok, skip, fail = 0, 0, 0
    total_tokens = 0
    t0 = time.time()

    with Pool(processes=n_workers, initializer=init_worker) as pool:
        for fpath, status, info in pool.imap_unordered(process_one, files, chunksize=4):
            if status == 'ok':
                ok += 1
                total_tokens += info or 0
            elif status == 'skip':
                skip += 1
            else:
                fail += 1
                if fail <= 10:
                    logger.warning(f"  FAIL: {fpath}: {info}")

            elapsed = time.time() - t0
            done = ok + skip + fail
            if done % 500 == 0 or done == len(files):
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(files) - done) / rate if rate > 0 else 0
                logger.info(f"  进度: {done}/{len(files)} ok={ok} skip={skip} fail={fail} "
                          f"rate={rate:.1f}/s eta={eta/60:.0f}min")

    logger.info(f"MusicXML 完成: ok={ok} skip={skip} fail={fail} "
                f"tokens={total_tokens} time={time.time()-t0:.0f}s")


if __name__ == '__main__':
    clean_old_musicxml()
    run_musicxml()
