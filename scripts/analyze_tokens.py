"""
Analyze processed token files to see what token types actually appear in the training data.

Token files are stored as JSON arrays of integers.
"""
import json
import os
import sys
from collections import Counter

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TOKENS_DIR = os.path.join(BASE, 'data', 'processed', 'tokens')

sys.path.insert(0, BASE)
from chopinote_dataset.tokenizer import REMITokenizer

tokenizer = REMITokenizer(grid_size=16, velocity_levels=8)
id_to_token = tokenizer._id_to_token

token_files = sorted(os.listdir(TOKENS_DIR))
sample_size = min(2000, len(token_files))

all_token_types = Counter()
token_value_subs = Counter()
total_tokens = 0
files_processed = 0

for fn in token_files[:sample_size]:
    fp = os.path.join(TOKENS_DIR, fn)
    try:
        with open(fp, 'r') as f:
            ids = json.load(f)
    except Exception as e:
        continue

    if not isinstance(ids, list):
        continue

    files_processed += 1
    total_tokens += len(ids)

    for tid in ids:
        token_str = id_to_token.get(tid, f'<UNKNOWN_{tid}>')

        # Extract type prefix (everything before the space)
        if ' ' in token_str:
            type_prefix = token_str.split(' ')[0] + '>'
            val_part = token_str.split(' ', 1)[1].rstrip('>')
        else:
            type_prefix = token_str
            val_part = ''

        all_token_types[type_prefix] += 1

        # Track specific values for important token types
        if type_prefix == '<Dynamic':
            token_value_subs['dynamic_' + val_part] += 1
        elif type_prefix == '<Artic':
            token_value_subs['artic_' + val_part] += 1
        elif type_prefix == '<Ornament':
            token_value_subs['ornament_' + val_part] += 1
        elif type_prefix == '<Pedal':
            token_value_subs['pedal_' + val_part] += 1
        elif type_prefix == '<Slur':
            token_value_subs['slur_' + val_part] += 1
        elif type_prefix == '<Hairpin':
            token_value_subs['hairpin_' + val_part] += 1
        elif type_prefix == '<Octave':
            token_value_subs['octave_' + val_part] += 1
        elif type_prefix == '<Arpeggio>':
            token_value_subs['arpeggio'] += 1
        elif type_prefix == '<Repeat':
            token_value_subs['repeat_' + val_part] += 1
        elif type_prefix == '<Jump':
            token_value_subs['jump_' + val_part] += 1
        elif type_prefix == '<Tempo':
            token_value_subs['tempo_' + val_part] += 1
        elif type_prefix == '<TimeSig':
            token_value_subs['timesig_' + val_part] += 1
        elif type_prefix == '<Key':
            token_value_subs['key_' + val_part] += 1
        elif type_prefix == '<TupletStart':
            token_value_subs['tupletstart_' + val_part] += 1
        elif type_prefix == '<TupletEnd>':
            token_value_subs['tupletend'] += 1
        elif type_prefix == '<GraceNote':
            token_value_subs['grace_' + val_part] += 1
        elif type_prefix == '<Program':
            token_value_subs['program_' + val_part] += 1
        elif type_prefix == '<Position':
            token_value_subs['position_' + val_part] += 1
        elif type_prefix == '<Note_ON':
            token_value_subs['pitch_' + val_part] += 1
        elif type_prefix == '<Velocity':
            token_value_subs['velocity_' + val_part] += 1
        elif type_prefix == '<Duration':
            token_value_subs['duration_' + val_part] += 1
        elif type_prefix == '<Beat':
            token_value_subs['beat_' + val_part] += 1
        elif type_prefix == '<Clef':
            token_value_subs['clef_' + val_part] += 1

print(f"Files processed: {files_processed}")
print(f"Total tokens: {total_tokens}")
print()

# Notation categories
notation_categories = [
    '<Dynamic', '<Artic', '<Ornament', '<Pedal', '<Slur',
    '<Hairpin', '<Octave', '<Arpeggio>', '<Repeat', '<Jump',
    '<TupletStart', '<TupletEnd>', '<GraceNote', '<Clef',
    '<Key', '<TimeSig', '<Tempo', '<Beat',
]

print("=== ALL TOKEN TYPE COUNTS ===")
for ttype, cnt in sorted(all_token_types.most_common()):
    marker = ' *** NOTATION ***' if ttype in notation_categories else ''
    print(f"  {ttype:30s} {cnt:>10d}{marker}")

print()
print("=== NOTATION TOKEN SUMMARY ===")
for cat in ['<Clef', '<Dynamic', '<Artic', '<Ornament', '<Pedal', '<Slur',
            '<Hairpin', '<Octave', '<Arpeggio>', '<Repeat', '<Jump',
            '<TupletStart', '<TupletEnd>', '<GraceNote', '<Key', '<TimeSig', '<Tempo', '<Beat']:
    if cat in all_token_types:
        print(f"  {cat:30s} --- FOUND ({all_token_types[cat]} occurrences)")
    else:
        print(f"  {cat:30s} --- NOT FOUND")

print()
print("=== DETAILED TOKEN VALUE BREAKDOWN ===")
for val, cnt in sorted(token_value_subs.most_common()):
    if cnt > 0:
        print(f"  {val:40s} {cnt:>10d}")

# Summary
print()
print("=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"Training token files scanned: {files_processed}")
print(f"Total tokens counted in sample: {total_tokens}")

notation_token_types = {k: v for k, v in all_token_types.items() if k in notation_categories}
total_notation = sum(notation_token_types.values())
total_not_notation = total_tokens - total_notation

print(f"Total notation tokens: {total_notation} ({total_notation/max(total_tokens,1)*100:.1f}%)")
print(f"Total non-notation tokens: {total_not_notation} ({total_not_notation/max(total_tokens,1)*100:.1f}%)")
print()

notation_found = {k: v for k, v in notation_token_types.items() if v > 0}
if not notation_found:
    print("CRITICAL FINDING: NO notation tokens found in training data.")
    print("The training dataset contains only:")
    print("  - Core tokens: Bar, Position, Program, Note_ON, Velocity, Duration")
    print("  - Special tokens: PAD, BOS, EOS, MASK")
    print()
    print("The following token types are DEFINED in vocabulary but NEVER appear in training data:")
    for cat in notation_categories:
        if cat not in notation_token_types:
            print(f"  - {cat}")
else:
    print("Notation tokens FOUND in training data:")
    for ttype, cnt in sorted(notation_found.items(), key=lambda x: -x[1]):
        print(f"  {ttype}: {cnt}")
