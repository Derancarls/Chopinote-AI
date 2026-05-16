#!/usr/bin/env python3
"""
为所有原始数据文件生成 .hash 侧车文件，避免后续预处理时重复扫描。

三种格式的数据集目录：
  - MIDI:   bread-midi-dataset, lmd_full, aria-midi-v1-pruned-ext,
            maestro-v3.0.0, musicnet_midis, POP909, giant-midi-repo
  - PDMX:   chopinote_data/data/raw/PDMX
  - MusicXML: asap, ATEPP-1.2, openscore (配置在 preprocess_musicxml.py)

用法:
    python scripts/backfill_source_hashes.py
    python scripts/backfill_source_hashes.py --workers 8
"""
import os
import sys
import hashlib
import logging
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
)
logger = logging.getLogger(__name__)

DATA_DISK = '/root/autodl-tmp'

# ── 所有源数据目录 ──────────────────────────────────────────
SOURCE_DIRS = [
    # MIDI
    '/root/autodl-tmp/bread-midi-dataset',
    '/root/autodl-tmp/lmd_full',
    '/root/autodl-tmp/aria-midi-v1-pruned-ext',
    '/root/autodl-tmp/maestro-v3.0.0',
    '/root/autodl-tmp/musicnet_midis',
    '/root/autodl-tmp/POP909',
    '/root/autodl-tmp/giant-midi-repo',
    # PDMX
    '/root/autodl-tmp/chopinote_data/data/raw/PDMX',
    # MusicXML
    '/root/autodl-tmp/asap',
    '/root/autodl-tmp/ATEPP-1.2',
]

MIDI_EXTS = {'.mid', '.midi'}
MUSICXML_EXTS = {'.musicxml', '.mxl', '.xml'}
PDMX_EXTS = {'.json'}


def find_source_files(directory: str) -> list:
    """递归查找所有可处理的源文件。"""
    files = []
    directory = os.path.abspath(directory)
    if not os.path.isdir(directory):
        logger.warning(f"目录不存在，跳过: {directory}")
        return files

    # 跳过 __MACOSX 和 .git 等
    skip_dirs = {'__MACOSX', '.git', '__pycache__'}

    for root, dirs, fnames in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for fname in fnames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in MIDI_EXTS or ext in MUSICXML_EXTS or ext in PDMX_EXTS:
                files.append(os.path.join(root, fname))

    return files


def _hash_one_file(file_path: str) -> tuple:
    """为单个文件计算 hash 并写 .hash 侧车。"""
    sidecar = file_path + '.hash'

    # 跳过已有侧车的文件
    if os.path.exists(sidecar):
        try:
            with open(sidecar) as f:
                cached = f.read().strip()
                if cached and len(cached) == 32:
                    return (file_path, cached, 'cached')
        except (OSError, ValueError):
            pass

    try:
        h = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                h.update(chunk)
        hexdigest = h.hexdigest()

        with open(sidecar, 'w') as f:
            f.write(hexdigest)

        return (file_path, hexdigest, 'new')
    except Exception as e:
        return (file_path, None, f'error:{e}')


def main():
    logger.info("=" * 60)
    logger.info("源文件 Hash 侧车回填")
    logger.info("=" * 60)

    n_workers = min(16, cpu_count())
    logger.info(f"Worker 数: {n_workers}")

    total_files = 0
    total_new = 0
    total_cached = 0
    total_errors = 0
    overall_start = time.time()

    for src_dir in SOURCE_DIRS:
        if not os.path.isdir(src_dir):
            logger.info(f"  [{os.path.basename(src_dir)}] 目录不存在，跳过")
            continue

        logger.info(f"  [{os.path.basename(src_dir)}] 扫描中 ...")
        t0 = time.time()
        files = find_source_files(src_dir)
        if not files:
            logger.info(f"  [{os.path.basename(src_dir)}] 未找到源文件")
            continue

        logger.info(f"  [{os.path.basename(src_dir)}] 找到 {len(files)} 个源文件，开始 hash ...")

        cnt_new = 0
        cnt_cached = 0
        cnt_err = 0

        with Pool(n_workers) as pool:
            for i, result in enumerate(pool.imap_unordered(_hash_one_file, files, chunksize=200)):
                path, h, status = result
                if status == 'new':
                    cnt_new += 1
                elif status == 'cached':
                    cnt_cached += 1
                else:
                    cnt_err += 1

                if (i + 1) % 20000 == 0 or (i + 1) == len(files):
                    elapsed = time.time() - t0
                    logger.info(f"    {i+1}/{len(files)} | new={cnt_new} cached={cnt_cached} err={cnt_err} ({elapsed:.1f}s)")

        elapsed = time.time() - t0
        logger.info(f"  [{os.path.basename(src_dir)}] 完成: "
                    f"{cnt_new} new, {cnt_cached} cached, {cnt_err} errors ({elapsed:.1f}s)")

        total_files += len(files)
        total_new += cnt_new
        total_cached += cnt_cached
        total_errors += cnt_err

    overall_elapsed = time.time() - overall_start
    logger.info("=" * 60)
    logger.info(f"全部完成！")
    logger.info(f"  总文件:   {total_files}")
    logger.info(f"  新建侧车: {total_new}")
    logger.info(f"  已有侧车: {total_cached}")
    logger.info(f"  错误:     {total_errors}")
    logger.info(f"  耗时:     {overall_elapsed:.1f}s ({overall_elapsed/3600:.1f}h)")


if __name__ == '__main__':
    main()
