#!/usr/bin/env python3
"""
Deduplicate token files produced by run_fast_preprocess.py.

Multiple watchdog instances ran simultaneously, creating duplicate .tokens
files (same hash8, different uuid8). This script keeps one file per unique
hash and removes the rest.
"""
import os
import sys
from collections import defaultdict

TOKEN_DIR = '/root/autodl-tmp/data/processed/tokens'


def main():
    if not os.path.isdir(TOKEN_DIR):
        print(f"ERROR: {TOKEN_DIR} not found")
        sys.exit(1)

    # Group by hash8
    by_hash = defaultdict(list)
    for fname in os.listdir(TOKEN_DIR):
        if not fname.endswith('.tokens'):
            continue
        # filename: {stem}_{hash8}_{uuid8}.tokens
        parts = fname.rsplit('_', 2)
        if len(parts) == 3:
            hash8 = parts[1]
            by_hash[hash8].append(os.path.join(TOKEN_DIR, fname))

    total_files = sum(len(v) for v in by_hash.values())
    unique_hashes = len(by_hash)
    duplicates = total_files - unique_hashes

    print(f"Total files:  {total_files}")
    print(f"Unique hashes: {unique_hashes}")
    print(f"Duplicates:   {duplicates}")
    print()

    if duplicates == 0:
        print("No duplicates found, nothing to do.")
        return

    # Verify files are identical before deleting
    removed = 0
    errors = 0
    for hash8, paths in by_hash.items():
        if len(paths) <= 1:
            continue

        # Sort by mtime, keep the oldest
        paths_sorted = sorted(paths, key=lambda p: os.path.getmtime(p))
        keep = paths_sorted[0]
        rest = paths_sorted[1:]

        # Verify content matches
        with open(keep, 'rb') as f:
            ref_content = f.read()

        for dup in rest:
            with open(dup, 'rb') as f:
                dup_content = f.read()
            if dup_content != ref_content:
                print(f"WARNING: {dup} content differs from {keep}, skipping delete")
                errors += 1
                continue
            os.remove(dup)
            removed += 1

    print(f"Removed:  {removed}")
    print(f"Errors:   {errors}")
    print(f"Remaining: {total_files - removed}")


if __name__ == '__main__':
    main()
