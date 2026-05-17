#!/usr/bin/env python3
"""
Fast MIDI → REMI token preprocessing v3.

Features:
  - Per-file timeout via SIGALRM (prevents worker hangs)
  - Separate counters: converted / duplicate_skip / quality_skip / timeout / error
  - Persistent state file for cross-session recovery
  - File manifest cache: first run scans+hashes, restarts skip scan entirely
  - PID lock prevents concurrent instances

Usage:
    python scripts/run_fast_preprocess.py
"""
import os
import sys
import json
import time
import logging
import signal
import hashlib
import atexit
import random
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chopinote_dataset.fast_converter import process_midi_file_fast, compute_file_hash

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# ── Paths ──
DATA_DISK = '/root/autodl-tmp'
PROCESSED_DIR = os.path.join(DATA_DISK, 'data', 'processed')
TOKEN_DIR = os.path.join(PROCESSED_DIR, 'tokens_v2')
META_DIR = os.path.join(PROCESSED_DIR, 'metadata_v2')
MANIFEST_DIR = os.path.join(PROCESSED_DIR, 'manifests_v2')
STATE_FILE = os.path.join(PROCESSED_DIR, 'conversion_state_v2.json')
PID_LOCK_PATH = os.path.join(PROCESSED_DIR, 'preprocess.lock')
os.makedirs(TOKEN_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)
os.makedirs(MANIFEST_DIR, exist_ok=True)

# ── Quality thresholds ──
MIN_NOTES = 10
MAX_NOTES = 50000
MIN_TOKENS = 50
MAX_TOKENS = 65536
MIN_SIZE_KB = 1
MAX_SIZE_MB = 50
PER_FILE_TIMEOUT = 120

# ── MIDI datasets ──
MIDI_DIRS = [
    '/root/autodl-tmp/bread-midi-dataset',
    '/root/autodl-tmp/lmd_full',
    '/root/autodl-tmp/aria-midi-v1-pruned-ext',
    '/root/autodl-tmp/maestro-v3.0.0',
    '/root/autodl-tmp/musicnet_midis',
    '/root/autodl-tmp/POP909',
    '/root/autodl-tmp/giant-midi-repo',
]

# ═══════════════════════════════════════════════════════════════
#  PID lock
# ═══════════════════════════════════════════════════════════════

_LOCK_FD = None

def acquire_lock():
    global _LOCK_FD
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    try:
        _LOCK_FD = os.open(PID_LOCK_PATH, os.O_CREAT | os.O_WRONLY | os.O_EXCL)
    except FileExistsError:
        try:
            with open(PID_LOCK_PATH) as f:
                old_pid = int(f.read().strip())
        except (ValueError, OSError):
            old_pid = None
        if old_pid is not None:
            try:
                os.kill(old_pid, 0)
                logger.error(f"Another instance (PID {old_pid}) already running. Exiting.")
                sys.exit(1)
            except OSError:
                logger.warning(f"Removing stale lock from dead PID {old_pid}")
                os.remove(PID_LOCK_PATH)
                _LOCK_FD = os.open(PID_LOCK_PATH, os.O_CREAT | os.O_WRONLY | os.O_EXCL)
    os.write(_LOCK_FD, str(os.getpid()).encode())
    os.fsync(_LOCK_FD)

def release_lock():
    global _LOCK_FD
    if _LOCK_FD is not None:
        try:
            os.close(_LOCK_FD)
        except OSError:
            pass
        _LOCK_FD = None
    try:
        if os.path.exists(PID_LOCK_PATH):
            with open(PID_LOCK_PATH) as f:
                if f.read().strip() == str(os.getpid()):
                    os.remove(PID_LOCK_PATH)
    except OSError:
        pass

# ═══════════════════════════════════════════════════════════════
#  State persistence
# ═══════════════════════════════════════════════════════════════

def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {
        'version': 3,
        'last_updated': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'datasets': {},
        'totals': {'converted': 0, 'duplicate_skip': 0, 'quality_skip': 0,
                   'timeout': 0, 'error': 0},
        'all_done': False,
    }

def save_state(state: dict):
    state['last_updated'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    tmp = STATE_FILE + '.tmp'
    try:
        with open(tmp, 'w') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        os.replace(tmp, STATE_FILE)
    except OSError as e:
        logger.warning(f"Failed to save state: {e}")

# ═══════════════════════════════════════════════════════════════
#  Manifest: cached file→hash mapping (so restarts skip scanning)
# ═══════════════════════════════════════════════════════════════

def manifest_path(ds_name: str) -> str:
    return os.path.join(MANIFEST_DIR, f'{ds_name}_manifest.json')

def load_manifest(ds_name: str):
    """Load cached file→hash manifest. Returns dict or None."""
    path = manifest_path(ds_name)
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return None

def save_manifest(ds_name: str, manifest: dict):
    path = manifest_path(ds_name)
    tmp = path + '.tmp'
    try:
        with open(tmp, 'w') as f:
            json.dump(manifest, f, indent=2)
        os.replace(tmp, path)
    except OSError as e:
        logger.warning(f"Failed to save manifest: {e}")

# ═══════════════════════════════════════════════════════════════
#  Build processed-hash set from existing token filenames
# ═══════════════════════════════════════════════════════════════

def build_processed_hashes() -> set:
    hashes = set()
    if not os.path.isdir(TOKEN_DIR):
        return hashes
    for fname in os.listdir(TOKEN_DIR):
        if not fname.endswith('.tokens'):
            continue
        parts = fname.rsplit('_', 2)
        if len(parts) == 3:
            hashes.add(parts[1])
    return hashes

# ═══════════════════════════════════════════════════════════════
#  Worker with timeout
# ═══════════════════════════════════════════════════════════════

class FileTimeoutError(Exception):
    pass

def _alarm_handler(signum, frame):
    raise FileTimeoutError('timeout')

def process_one_file(file_path: str, output_dir: str, processed_hashes: set) -> dict:
    """Convert one MIDI file with timeout."""
    # Hash check
    try:
        fhash = compute_file_hash(file_path)[:8]
        if fhash in processed_hashes:
            return {'status': 'duplicate_skip', 'path': file_path, 'hash': fhash}
    except Exception:
        fhash = None

    # Quick size check
    try:
        fsize_kb = os.path.getsize(file_path) / 1024
    except OSError:
        return {'status': 'error', 'path': file_path, 'reason': 'unreadable'}
    if fsize_kb < MIN_SIZE_KB:
        return {'status': 'quality_skip', 'path': file_path, 'reason': f'size_lt_{MIN_SIZE_KB}KB'}
    if fsize_kb > MAX_SIZE_MB * 1024:
        return {'status': 'quality_skip', 'path': file_path, 'reason': f'size_gt_{MAX_SIZE_MB}MB'}

    # Convert with timeout
    old = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(PER_FILE_TIMEOUT)
    try:
        result = process_midi_file_fast(
            file_path, output_dir,
            min_notes=MIN_NOTES, max_notes=MAX_NOTES,
            min_tokens=MIN_TOKENS, max_tokens=MAX_TOKENS,
            min_size_kb=MIN_SIZE_KB, max_size_mb=MAX_SIZE_MB,
        )
        signal.alarm(0)
        if result:
            return {'status': 'converted', 'path': file_path,
                    'hash': fhash, 'tokens': result['num_tokens']}
        else:
            return {'status': 'quality_skip', 'path': file_path,
                    'reason': 'quality_filter', 'hash': fhash}
    except FileTimeoutError:
        return {'status': 'timeout', 'path': file_path, 'hash': fhash}
    except MemoryError:
        return {'status': 'timeout', 'path': file_path,
                'reason': 'memory', 'hash': fhash}
    except Exception as e:
        return {'status': 'error', 'path': file_path,
                'reason': str(e)[:200].replace('\n', ' '), 'hash': fhash}
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

# ═══════════════════════════════════════════════════════════════
#  Dataset processing
# ═══════════════════════════════════════════════════════════════

def find_midi_files(directory: str) -> list:
    files = []
    for root, dirs, fnames in os.walk(directory):
        dirs[:] = [d for d in dirs if d != '__MACOSX']
        for fname in fnames:
            if fname.lower().endswith(('.mid', '.midi')):
                files.append(os.path.join(root, fname))
    return files

def process_dataset(dataset_dir: str, n_workers: int, processed_hashes: set) -> dict:
    ds_name = os.path.basename(dataset_dir)

    # ── Load or build manifest ──
    manifest = load_manifest(ds_name)
    if manifest is not None:
        logger.info(f"  Loaded cached manifest ({len(manifest)} files)")
        # Build list from manifest: only files whose hashes are NOT yet processed
        new_files = [(path, h) for path, h in manifest.items()
                     if h not in processed_hashes]
        total = len(manifest)
        remaining = len(new_files)
        file_list = [path for path, _ in new_files]
        logger.info(f"  {total} total in manifest, {remaining} need conversion")
    else:
        # No manifest — first run: scan all files and hash them
        logger.info(f"  Scanning for MIDI files ...")
        midi_files = find_midi_files(dataset_dir)
        total = len(midi_files)
        logger.info(f"  Found {total} MIDI files, computing hashes ...")

        # Hash all files in parallel
        t0 = time.time()
        manifest = {}
        worker = partial(_hash_one_file)
        with Pool(n_workers) as pool:
            results = pool.imap_unordered(worker, midi_files, chunksize=200)
            for i, (path, h) in enumerate(results):
                if h:
                    manifest[path] = h
                if (i + 1) % 20000 == 0:
                    logger.info(f"    hashed {i+1}/{total}")

        elapsed = time.time() - t0
        logger.info(f"  Hashed {len(manifest)} files in {elapsed:.1f}s, saving manifest...")
        save_manifest(ds_name, manifest)

        # Now figure out what needs conversion
        file_list = [path for path, h in manifest.items()
                     if h not in processed_hashes]
        remaining = len(file_list)
        logger.info(f"  {total} total scanned, {remaining} need conversion")

    if remaining == 0:
        logger.info(f"  All files already converted, nothing to do.")
        return {'total': total, 'converted': 0, 'duplicate_skip': total,
                'quality_skip': 0, 'timeout': 0, 'error': 0}

    # ── Convert remaining files ──
    t0 = time.time()
    cnt = {'converted': 0, 'duplicate_skip': 0,
           'quality_skip': 0, 'timeout': 0, 'error': 0}

    worker = partial(process_one_file, output_dir=PROCESSED_DIR,
                     processed_hashes=processed_hashes)

    with Pool(n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(worker, file_list, chunksize=50)):
            status = result['status']
            cnt[status] += 1
            if status == 'converted' and result.get('hash'):
                processed_hashes.add(result['hash'])

            done = i + 1
            if done % 2000 == 0 or done == remaining:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed else 0
                eta = (remaining - done) / rate if rate else 0
                logger.info(
                    f"  [{ds_name}] convert {done}/{remaining} | "
                    f"✓{cnt['converted']} "
                    f"qual↓{cnt['quality_skip']} "
                    f"⏱{cnt['timeout']} "
                    f"✗{cnt['error']} | "
                    f"{rate:.0f}f/s ETA{eta:.0f}s"
                )

    elapsed = time.time() - t0
    logger.info(
        f"  [{ds_name}] DONE in {elapsed:.1f}s — "
        f"✓{cnt['converted']} converted, "
        f"{cnt['duplicate_skip']} duplicate_skip, "
        f"{cnt['quality_skip']} quality_skip, "
        f"{cnt['timeout']} timeout, "
        f"{cnt['error']} error"
    )

    # duplicate_skip here means files that existed in token dir = total - remaining
    cnt['duplicate_skip'] += total - remaining
    cnt['total'] = total
    return cnt


def _hash_one_file(file_path: str):
    """Compute hash for a single file (used for manifest building)."""
    try:
        fsize = os.path.getsize(file_path) / 1024
        if fsize < MIN_SIZE_KB:
            return (file_path, None)
    except OSError:
        return (file_path, None)
    try:
        h = compute_file_hash(file_path)[:8]
        return (file_path, h)
    except Exception:
        return (file_path, None)


# ═══════════════════════════════════════════════════════════════
#  Train/val/test splits
# ═══════════════════════════════════════════════════════════════

def regenerate_splits(token_files: list, processed_dir: str, data_disk: str) -> dict:
    all_path = os.path.join(processed_dir, 'all_files.txt')
    with open(all_path, 'w', encoding='utf-8') as f:
        for fp in token_files:
            rel = os.path.relpath(fp, os.path.join(data_disk, 'data'))
            f.write(f'data/processed/{rel}\n')

    random.seed(42)
    shuffled = list(token_files)
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
        path = os.path.join(processed_dir, name)
        with open(path, 'w', encoding='utf-8') as f:
            for fp in files:
                rel = os.path.relpath(fp, os.path.join(data_disk, 'data'))
                f.write(f'data/processed/{rel}\n')

    return {'total': n, 'train': n_train, 'val': n_val, 'test': n - n_train - n_val}


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    acquire_lock()
    atexit.register(release_lock)

    logger.info("=" * 70)
    logger.info("MIDI → REMI Preprocessor v3 (manifest-cached)")
    logger.info(f"Output: {PROCESSED_DIR}")

    # Load state
    state = load_state()
    prev_totals = state.get('totals', {})
    logger.info(f"Previous totals: {prev_totals}")

    # Build hash set from existing token filenames
    processed_hashes = build_processed_hashes()
    logger.info(f"Existing token hashes: {len(processed_hashes)}")

    n_workers = 24
    logger.info(f"Workers: {n_workers}, Timeout: {PER_FILE_TIMEOUT}s per file")

    overall_start = time.time()
    totals = {'converted': 0, 'duplicate_skip': 0, 'quality_skip': 0,
              'timeout': 0, 'error': 0}

    for dataset_dir in MIDI_DIRS:
        ds_name = os.path.basename(dataset_dir)
        if not os.path.isdir(dataset_dir):
            logger.warning(f"Directory not found: {dataset_dir}, skipping")
            continue

        # Skip if already completed per state
        ds_prev = state.get('datasets', {}).get(ds_name, {})
        if ds_prev.get('completed') and ds_prev.get('total', 0) > 0:
            logger.info(f"  [{ds_name}] already completed, skipping")
            for k in totals:
                totals[k] += ds_prev.get(k, 0)
            continue

        result = process_dataset(dataset_dir, n_workers, processed_hashes)
        for k in totals:
            totals[k] += result.get(k, 0)

        state.setdefault('datasets', {})[ds_name] = {
            'total': result.get('total', 0),
            'converted': result.get('converted', 0),
            'duplicate_skip': result.get('duplicate_skip', 0),
            'quality_skip': result.get('quality_skip', 0),
            'timeout': result.get('timeout', 0),
            'error': result.get('error', 0),
            'completed': True,
        }
        state['totals'] = dict(totals)
        save_state(state)

    # Final summary
    overall_elapsed = time.time() - overall_start
    logger.info("=" * 70)
    logger.info(f"ALL DATASETS DONE ({overall_elapsed:.1f}s / {overall_elapsed/3600:.1f}h)")
    logger.info(f"  ✓ Converted:       {totals['converted']}")
    logger.info(f"  dup↓ Duplicate:     {totals['duplicate_skip']}")
    logger.info(f"  qual↓ Quality skip: {totals['quality_skip']}")
    logger.info(f"  ⏱ Timeout:         {totals['timeout']}")
    logger.info(f"  ✗ Error:           {totals['error']}")

    # Regenerate splits
    logger.info("Regenerating dataset splits...")
    token_files = sorted(
        os.path.join(TOKEN_DIR, f) for f in os.listdir(TOKEN_DIR)
        if f.endswith('.tokens')
    )
    stats = regenerate_splits(token_files, PROCESSED_DIR, DATA_DISK)
    logger.info(f"Train: {stats['train']} | Val: {stats['val']} | Test: {stats['test']}")

    state['totals'] = dict(totals)
    state['all_done'] = True
    save_state(state)
    logger.info(f"State saved to {STATE_FILE}")
    logger.info("Done!")


if __name__ == '__main__':
    main()
