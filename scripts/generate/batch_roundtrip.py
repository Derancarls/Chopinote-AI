"""Batch round-trip test runner for file lists."""
import json, os, sys, time
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _project_root)

from scripts.roundtrip_test import evaluate_roundtrip, _report_to_dict, _aggregate, _print_aggregate

file_list = sys.argv[1]
output_dir = sys.argv[2] if len(sys.argv) > 2 else 'roundtrip_reports'

with open(file_list) as f:
    files = [line.strip() for line in f if line.strip()]

print(f"Processing {len(files)} files...")
os.makedirs(output_dir, exist_ok=True)

reports = []
errors = []
start = time.time()
for i, fpath in enumerate(files):
    name = os.path.basename(fpath)
    elapsed = time.time() - start
    eta = elapsed / (i + 1) * (len(files) - i - 1) if i > 0 else 0
    print(f"  [{i+1}/{len(files)}] {name}...", end=' ', flush=True)
    t0 = time.time()
    result = evaluate_roundtrip(fpath)
    dt = time.time() - t0
    if 'error' in result:
        print(f"ERROR ({dt:.1f}s): {result['error']}")
        errors.append({'file': fpath, 'error': result['error']})
    else:
        r = result['notes']['recall_pct']
        m = result['notes']['missed']
        print(f"recall={r:.1f}% missed={m} ({dt:.1f}s) ETA:{eta:.0f}s")
        reports.append(result)
        stem = Path(fpath).stem.replace('/', '_')
        with open(os.path.join(output_dir, f'{stem}.json'), 'w') as out_f:
            json.dump(result, out_f, indent=2)

if reports:
    agg = _aggregate(reports)
    with open(os.path.join(output_dir, '_aggregate.json'), 'w') as f:
        json.dump(agg, f, indent=2)
    print(f"\nAggregate saved: {output_dir}/_aggregate.json")
    _print_aggregate(agg)

if errors:
    with open(os.path.join(output_dir, '_errors.json'), 'w') as f:
        json.dump(errors, f, indent=2)
    print(f"\n{len(errors)} errors: saved")

elapsed_total = time.time() - start
print(f"\nTotal time: {elapsed_total:.1f}s ({elapsed_total/len(files):.1f}s/file)")
