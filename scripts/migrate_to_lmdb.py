#!/usr/bin/env python3
"""Migrate Chopinote-AI data → LMDB.

Single-process with ThreadPoolExecutor for parallel file I/O.
Reliable alternative to multiprocessing.Queue-based approach.

Usage:
    python scripts/migrate_to_lmdb.py migrate \
        --tokens-dir ... --ssf-dir ... --metadata-dir ... \
        --lmdb-path ... --workers 16 --verify
"""

from __future__ import annotations

import argparse, json, os, struct, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from pathlib import Path

import msgpack

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chopinote_dataset.lmdb_store import (
    LMDBStore, VERSION, _encode_tokens, _encode_len,
)

# ═══════════════════════════════════════════════════════════════

def _load_json(path: Path):
    try:
        with open(path, 'rb') as f:
            return json.load(f)
    except Exception:
        return None


def _pack_one(file_id: str, tokens_dir: str, ssf_dir: str,
              metadata_dir: str) -> dict | None:
    """Read one file + sidecars, return packed dict or None."""
    tp = os.path.join(tokens_dir, file_id + '.tokens')
    try:
        with open(tp, 'rb') as f:
            tokens = json.load(f)
    except Exception:
        return None

    if not isinstance(tokens, list) or len(tokens) < 10:
        return None

    item = {
        'file_id': file_id,
        'tokens_raw': _encode_tokens(tokens),
        'length': len(tokens),
    }

    # .sec.json
    sec = _load_json(Path(tokens_dir) / (file_id + '.sec.json'))
    if sec is not None:
        item['sec'] = msgpack.packb(sec)

    # .ssf.json
    ssf = _load_json(Path(ssf_dir) / (file_id + '.ssf.json'))
    if ssf is not None:
        item['ssf'] = msgpack.packb(ssf)

    # .func.json
    func = _load_json(Path(tokens_dir) / (file_id + '.func.json'))
    if func is not None:
        item['func'] = msgpack.packb(func)

    # .meta.json
    meta = _load_json(Path(metadata_dir) / (file_id + '.meta.json'))
    if meta is not None:
        item['meta'] = msgpack.packb(meta)

    return item


# ═══════════════════════════════════════════════════════════════

def cmd_migrate(args):
    # ── Scan ──
    print("[Phase 1] Scanning .tokens files...", flush=True)
    path_file = '/tmp/lmdb_migrate_paths.txt'
    subprocess.run(
        ['find', args.tokens_dir, '-maxdepth', '1', '-name', '*.tokens', '-print0'],
        stdout=open(path_file, 'wb'), check=True
    )

    file_ids = []
    buf = b''
    with open(path_file, 'rb') as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                if buf:
                    fn = os.path.basename(buf.decode('utf-8'))
                    if fn.endswith('.tokens'):
                        file_ids.append(fn[:-7])
                break
            buf += chunk
            while b'\0' in buf:
                pb, buf = buf.split(b'\0', 1)
                if pb:
                    fn = os.path.basename(pb.decode('utf-8'))
                    if fn.endswith('.tokens'):
                        file_ids.append(fn[:-7])
    os.unlink(path_file)
    total = len(file_ids)
    print(f"[Phase 1] {total} files", flush=True)

    # ── Skip existing ──
    already = set()
    if os.path.exists(args.lmdb_path) and not args.force:
        print("[Phase 2] Checking already-migrated...", flush=True)
        db = LMDBStore.open(args.lmdb_path, readonly=True)
        already = set(db.iter_file_ids())
        db.close()
        print(f"[Phase 2] {len(already)} done, {total - len(already)} to go", flush=True)
        file_ids = [f for f in file_ids if f not in already]
    else:
        print("[Phase 2] Creating new LMDB...", flush=True)
        if os.path.exists(args.lmdb_path):
            import shutil; shutil.rmtree(args.lmdb_path)
        LMDBStore.create(args.lmdb_path, map_size=args.map_size).close()

    if not file_ids:
        print("Nothing to do.")
        return

    # ── Migrate: ThreadPool for I/O + main-thread LMDB writes ──
    print(f"[Phase 3] {args.workers} threads reading, main thread writing LMDB...", flush=True)

    db = LMDBStore.open(args.lmdb_path, readonly=False, map_size=args.map_size)
    stats = Counter()
    t0 = time.time()
    done = len(already)
    last_tick = done // 40000  # report every 40K successful writes

    with db.batch_write(args.batch_size) as batch:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            # Submit in chunks to control memory
            CHUNK = args.workers * 50  # Submit 800 at a time with 16 workers
            for i in range(0, len(file_ids), CHUNK):
                chunk_ids = file_ids[i:i + CHUNK]
                futures = {
                    executor.submit(_pack_one, fid, args.tokens_dir,
                                    args.ssf_dir, args.metadata_dir): fid
                    for fid in chunk_ids
                }
                for future in as_completed(futures):
                    item = future.result()
                    if item is None:
                        stats['skip'] += 1
                        continue
                    fid = item['file_id']
                    batch.put(fid, 'tokens', item['tokens_raw'])
                    batch.put(fid, 'len', item['length'])
                    for kind in ('sec', 'ssf', 'func', 'meta'):
                        if kind in item:
                            batch.put(fid, kind, item[kind])
                    stats['ok'] += 1
                    done += 1

                tick = done // 40000
                if tick > last_tick or stats['ok'] < 1600:
                    last_tick = tick
                    elapsed = time.time() - t0
                    rate = stats['ok'] / max(1, elapsed)
                    eta = (total - done) / max(1, rate)
                    print(f"  {done}/{total} ({100*done/total:.0f}%) "
                          f"| {rate:.0f} f/s | ETA {eta:.0f}s", flush=True)

    db.close()
    elapsed = time.time() - t0
    print(f"[Phase 3] Done: {stats['ok']} ok, {stats.get('skip', 0)} skipped "
          f"({elapsed:.0f}s, {stats['ok']/max(1,elapsed):.0f} f/s)", flush=True)

    # ── Verify ──
    if args.verify:
        cmd_verify_internal(args.lmdb_path, args.tokens_dir, args.sample_rate or 0.001)

    # ── Stats ──
    db_size = os.path.getsize(os.path.join(args.lmdb_path, 'data.mdb'))
    print(f"\nLMDB: {args.lmdb_path}")
    print(f"  Size: {db_size / 1024**3:.1f} GB")
    print(f"  Entries: {LMDBStore.open(args.lmdb_path, readonly=True).stats['num_entries_main']}")


def cmd_verify_internal(lmdb_path, tokens_dir, sample_rate=0.001):
    import random
    print(f"[Verify] {sample_rate*100:.1f}% sample...", flush=True)
    db = LMDBStore.open(lmdb_path, readonly=True)
    all_ids = list(db.iter_file_ids())
    n = max(100, int(len(all_ids) * sample_rate))
    sample = random.sample(all_ids, min(n, len(all_ids)))
    mismatches = 0
    for fid in sample:
        lmdb_tokens = db.get_tokens(fid)
        with open(os.path.join(tokens_dir, fid + '.tokens'), 'rb') as f:
            file_tokens = json.load(f)
        if lmdb_tokens != file_tokens:
            mismatches += 1
        if db.get_length(fid) != len(file_tokens):
            mismatches += 1
    db.close()
    print(f"[Verify] {len(sample)} checked, {mismatches} mismatches")
    return mismatches == 0


def cmd_verify(args):
    cmd_verify_internal(args.lmdb_path, args.tokens_dir, args.sample_rate or 0.001)


def cmd_build_indices(args):
    print("[Build-indices] ...", flush=True)
    db = LMDBStore.open(args.lmdb_path, readonly=False)
    count = 0
    with db.env.begin(write=True) as txn:
        for fid in db.iter_file_ids():
            cls_data = db.get_cls(fid)
            if cls_data and 'level' in cls_data:
                db.add_to_index(b'idx:level',
                              db._level_idx_key(cls_data['level'], ''),
                              fid, txn=txn)
                count += 1
    db.close()
    print(f"[Build-indices] {count} indexed")


def cmd_stats(args):
    db = LMDBStore.open(args.lmdb_path, readonly=True)
    s = db.stats
    n = sum(1 for _ in db.iter_file_ids())
    db.close()
    print(f"LMDB: {args.lmdb_path}")
    print(f"  Entries:  {s['num_entries_main']}")
    print(f"  Files:    {n}")
    print(f"  Map size: {s['map_size'] / 1024**3:.1f} GB")
    print(f"  DB file:  {os.path.getsize(os.path.join(args.lmdb_path, 'data.mdb')) / 1024**3:.1f} GB")


# ═══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description='Chopinote-AI LMDB migration')
    sub = ap.add_subparsers(dest='command')

    p = sub.add_parser('migrate')
    p.add_argument('--tokens-dir', required=True)
    p.add_argument('--ssf-dir', default=None)
    p.add_argument('--metadata-dir', default=None)
    p.add_argument('--lmdb-path', required=True)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=5000)
    p.add_argument('--map-size', type=int, default=48 * 1024**3)
    p.add_argument('--force', action='store_true')
    p.add_argument('--verify', action='store_true')
    p.add_argument('--sample-rate', type=float, default=0.001)

    p2 = sub.add_parser('verify')
    p2.add_argument('--lmdb-path', required=True)
    p2.add_argument('--tokens-dir', default='/root/autodl-tmp/data/processed/tokens_v4')
    p2.add_argument('--sample-rate', type=float, default=0.001)

    p3 = sub.add_parser('build-indices')
    p3.add_argument('--lmdb-path', required=True)

    p4 = sub.add_parser('stats')
    p4.add_argument('--lmdb-path', required=True)

    args = ap.parse_args()

    if args.command == 'migrate':
        if args.ssf_dir is None:
            args.ssf_dir = args.tokens_dir
        if args.metadata_dir is None:
            args.metadata_dir = os.path.join(
                os.path.dirname(os.path.dirname(args.tokens_dir)), 'metadata_v4')
        cmd_migrate(args)
    elif args.command == 'verify':
        cmd_verify(args)
    elif args.command == 'build-indices':
        cmd_build_indices(args)
    elif args.command == 'stats':
        cmd_stats(args)
    else:
        ap.print_help()


if __name__ == '__main__':
    main()
