"""
Analyze the PDMX training dataset for musical notations/articulations/expressions.

Scans PDMX JSON files across the entire dataset and counts occurrences of
various musical notation elements. Also checks the processed .tokens files
to see what actually appears in the training data.

Usage:
    python scripts/analyze_notations.py
"""
import json
import os
import glob
from collections import defaultdict, Counter

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# === Configuration ===
PDMX_DATA = os.path.join(BASE, 'data', 'raw', 'pdmx_extracted', 'PDMX', 'data')
PDMX_CSV = os.path.join(BASE, 'data', 'raw', 'pdmx_extracted', 'PDMX', 'PDMX.csv')
TOKENS_DIR = os.path.join(BASE, 'data', 'processed', 'tokens')
ALL_FILES_LIST = os.path.join(BASE, 'data', 'processed', 'all_files.txt')
METADATA_DIR = os.path.join(BASE, 'data', 'processed', 'metadata')

# === Counters ===
counts = {
    # Notes-related
    'total_files': 0,
    'files_with_grace_notes': 0,
    'total_grace_notes': 0,
    'grace_acciaccatura': 0,
    'grace_appoggiatura': 0,
    'grace_other': 0,

    # Chords
    'files_with_chords': 0,
    'total_chords': 0,

    # Barline types
    'barlines_single': 0,
    'barlines_double': 0,
    'barlines_end': 0,
    'barlines_start_repeat': 0,
    'barlines_end_repeat': 0,
    'barlines_other': 0,

    # Top-level PDMX metadata
    'tempos_total': 0,
    'files_with_tempo': 0,
    'keysigs_total': 0,
    'timesigs_total': 0,
    'beats_total': 0,
    'files_with_beats': 0,
    'lyrics_total': 0,
    'files_with_lyrics': 0,
    'annotations_total': 0,
    'files_with_annotations': 0,

    # Track-level
    'total_tracks': 0,
    'programs': Counter(),
    'tracks_with_name': 0,
    'tracks_is_drum': 0,
    'tracks_with_lyrics': 0,
    'tracks_with_annotations': 0,
    'tracks_with_chords': 0,

    # Note data
    'total_notes': 0,
    'note_fields_found': set(),
    'notes_with_extra_fields': 0,
    'note_min_pitch': 127,
    'note_max_pitch': 0,
    'note_velocities': Counter(),

    # Barline subtypes across dataset
    'barline_subtypes_found': set(),

    # Tokens analysis
    'token_types_found': Counter(),
    'tokens_files_scanned': 0,
}

# === Helper ===
def scan_directory_recursive(directory):
    """Find all JSON files recursively."""
    files = []
    for root, dirs, filenames in os.walk(directory):
        for f in filenames:
            if f.endswith('.json'):
                files.append(os.path.join(root, f))
    return files

def analyze_note_extra_fields(note, base_fields):
    """Check if a note has any extra fields beyond the standard ones."""
    extra = set(note.keys()) - base_fields
    return extra


def scan_pdmx_files():
    """Main scan of PDMX JSON files."""
    print("=== Scanning PDMX JSON files ===")

    # First, get all PDMX files - but we need a representative sample
    # Total is very large (~508k), so sample strategically
    all_files = scan_directory_recursive(PDMX_DATA)
    print(f"Total PDMX JSON files found: {len(all_files)}")

    # Standard note fields
    base_note_fields = {'name', 'time', 'pitch', 'duration', 'velocity', 'pitch_str', 'measure', 'is_grace'}

    # Sample size: scan all files from first-level subdirs only (extensive coverage)
    # First-level subdirs each have ~35 files
    first_level_dirs = []
    for subdir in sorted(os.listdir(PDMX_DATA)):
        full = os.path.join(PDMX_DATA, subdir)
        if os.path.isdir(full):
            first_level_dirs.append(full)

    print(f"First-level subdirectories: {len(first_level_dirs)}")

    # Also sample from deeper dirs
    deeper_dirs = [os.path.join(PDMX_DATA, 'a', str(i)) for i in range(1, 20)]
    deeper_dirs.extend([os.path.join(PDMX_DATA, 'b', str(i)) for i in range(1, 20)])

    sample_dirs = first_level_dirs + deeper_dirs
    print(f"Sample directories: {len(sample_dirs)}")

    # Collect all note extra fields
    all_extra_fields = Counter()

    for subdir in sample_dirs:
        if not os.path.isdir(subdir):
            continue
        for fn in sorted(os.listdir(subdir)):
            if not fn.endswith('.json'):
                continue
            fp = os.path.join(subdir, fn)
            try:
                with open(fp, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception:
                continue

            counts['total_files'] += 1

            # --- Top-level metadata ---
            tempos = data.get('tempos', [])
            counts['tempos_total'] += len(tempos)
            if tempos:
                counts['files_with_tempo'] += 1

            keysigs = data.get('key_signatures', [])
            counts['keysigs_total'] += len(keysigs)

            timesigs = data.get('time_signatures', [])
            counts['timesigs_total'] += len(timesigs)

            beats = data.get('beats', [])
            counts['beats_total'] += len(beats)
            if beats:
                counts['files_with_beats'] += 1

            lyrics = data.get('lyrics', [])
            counts['lyrics_total'] += len(lyrics)
            if lyrics:
                counts['files_with_lyrics'] += 1

            annotations = data.get('annotations', [])
            counts['annotations_total'] += len(annotations)
            if annotations:
                counts['files_with_annotations'] += 1

            # --- Barlines ---
            for bl in data.get('barlines', []):
                subtype = bl.get('subtype', 'unknown')
                counts['barline_subtypes_found'].add(subtype)
                if subtype == 'single':
                    counts['barlines_single'] += 1
                elif subtype == 'double':
                    counts['barlines_double'] += 1
                elif subtype == 'end':
                    counts['barlines_end'] += 1
                elif subtype == 'start-repeat':
                    counts['barlines_start_repeat'] += 1
                elif subtype == 'end-repeat':
                    counts['barlines_end_repeat'] += 1
                else:
                    counts['barlines_other'] += 1

            # --- Tracks ---
            for trk in data.get('tracks', []):
                counts['total_tracks'] += 1
                prog = trk.get('program', -1)
                counts['programs'][prog] += 1
                if trk.get('name'):
                    counts['tracks_with_name'] += 1
                if trk.get('is_drum'):
                    counts['tracks_is_drum'] += 1
                if trk.get('lyrics'):
                    counts['tracks_with_lyrics'] += 1
                if trk.get('annotations'):
                    counts['tracks_with_annotations'] += 1

                # Check track-level keys beyond standard
                track_extra = set(trk.keys()) - {'name', 'program', 'is_drum', 'notes', 'chords', 'lyrics', 'annotations'}
                if track_extra:
                    print(f"  EXTRA TRACK KEYS in {fn}: {track_extra}")

                # --- Chords ---
                chords = trk.get('chords', [])
                if chords:
                    counts['files_with_chords'] += 1
                    counts['total_chords'] += len(chords)
                    # Check chord fields
                    for c in chords:
                        extra_chord_keys = set(c.keys()) - {'name', 'time', 'pitch', 'duration', 'velocity', 'pitch_str', 'measure', 'is_grace', 'notes'}
                        if extra_chord_keys:
                            print(f"  EXTRA CHORD KEYS in {fn}: {extra_chord_keys}")
                            print(f"    chord: {c}")

                # --- Notes ---
                notes = trk.get('notes', [])
                for n in notes:
                    counts['total_notes'] += 1

                    # Track note field counts
                    for k in n.keys():
                        counts['note_fields_found'].add(k)

                    # Extra fields beyond standard
                    extra = set(n.keys()) - base_note_fields
                    if extra:
                        counts['notes_with_extra_fields'] += 1
                        for e in extra:
                            all_extra_fields[e] += 1

                    # Grace notes
                    if n.get('is_grace'):
                        counts['total_grace_notes'] += 1

                    # Pitch range
                    pitch = n.get('pitch', -1)
                    if pitch >= 0:
                        if pitch < counts['note_min_pitch']:
                            counts['note_min_pitch'] = pitch
                        if pitch > counts['note_max_pitch']:
                            counts['note_max_pitch'] = pitch

                    # Velocity
                    vel = n.get('velocity', -1)
                    if vel >= 0:
                        counts['note_velocities'][vel] += 1

            # Check for grace_notes in counts
            if any(n.get('is_grace') for trk in data.get('tracks', []) for n in trk.get('notes', [])):
                counts['files_with_grace_notes'] += 1

    print(f"Files scanned: {counts['total_files']}")
    print(f"Files with grace notes: {counts['files_with_grace_notes']}")
    print(f"Total grace notes: {counts['total_grace_notes']}")
    print(f"Note extra fields found: {sorted(counts['note_fields_found'] - base_note_fields)}")
    if all_extra_fields:
        print(f"Extra field frequencies: {all_extra_fields}")
    print(f"Note pitch range: {counts['note_min_pitch']} - {counts['note_max_pitch']}")
    print(f"Files with chords: {counts['files_with_chords']}")
    print(f"Total chords: {counts['total_chords']}")
    print(f"Barline subtypes found: {sorted(counts['barline_subtypes_found'])}")
    print(f"Total notes: {counts['total_notes']}")
    print(f"Total tracks: {counts['total_tracks']}")


def scan_tokens_files():
    """Scan processed token files to see what token types appear."""
    print("\n=== Scanning processed token files ===")

    if not os.path.isdir(TOKENS_DIR):
        print(f"  Token directory not found: {TOKENS_DIR}")
        return

    token_files = sorted(os.listdir(TOKENS_DIR))[:500]  # Sample 500 files
    print(f"  Scanning {len(token_files)} token files (from {len(os.listdir(TOKENS_DIR))} total)")

    # Known token type prefixes
    token_prefixes = [
        '<PAD>', '<BOS>', '<EOS>', '<MASK>', '<Bar>', '<Position', '<Program',
        '<Note_ON', '<Velocity', '<Duration', '<Clef', '<Dynamic', '<Hairpin',
        '<Artic', '<Ornament', '<Pedal', '<Slur', '<Repeat', '<Jump', '<Tempo',
        '<TupletStart', '<TupletEnd>', '<TimeSig', '<Rest>', '<GraceNote',
        '<Key', '<Beat', '<Octave', '<Arpeggio>'
    ]

    token_type_map = {}
    for prefix in token_prefixes:
        # Get the type name (strip '<' and trailing number/val markers)
        type_name = prefix.strip('<>').split()[0].split('_')[0].split('>')[0]
        if '>' in prefix:
            type_name = prefix.strip('<>').split('>')[0]
        token_type_map[prefix] = type_name

    for fn in token_files:
        fp = os.path.join(TOKENS_DIR, fn)
        if not fn.endswith('.tokens'):
            continue
        try:
            with open(fp, 'r') as f:
                line = f.read().strip()
        except Exception:
            continue

        ids = [int(x) for x in line.split() if x.strip().isdigit()]
        counts['tokens_files_scanned'] += 1
        counts['total_tokens'] = len(ids)

    print(f"  Token files scanned: {counts['tokens_files_scanned']}")


def scan_pdmx_metadata_csv():
    """Scan the PDMX CSV for additional metadata about notation content."""
    print("\n=== Scanning PDMX CSV ===")

    if not os.path.isfile(PDMX_CSV):
        print(f"  PDMX CSV not found: {PDMX_CSV}")
        return

    import csv
    row_count = 0
    with open(PDMX_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_count += 1
            if row_count == 1:
                print(f"  CSV columns ({len(row)}):")
                for k in sorted(row.keys()):
                    print(f"    - {k}")
            if row_count >= 5:
                break
    print(f"  Total rows in CSV: scanned first {row_count}")


def scan_all_note_fields_wide():
    """Wide scan: check note objects across a large sample for ANY extra fields."""
    print("\n=== Wide scan for extra note fields ===")

    base_note_fields = {'name', 'time', 'pitch', 'duration', 'velocity', 'pitch_str', 'measure', 'is_grace'}

    # Get a broad sample: 1 file from each deeper subdirectory
    all_deeper = []
    for root, dirs, files in os.walk(PDMX_DATA):
        for f in files:
            if f.endswith('.json'):
                all_deeper.append(os.path.join(root, f))

    import random
    random.seed(42)
    sample = random.sample(all_deeper, min(5000, len(all_deeper)))
    print(f"  Sampling {len(sample)} files from {len(all_deeper)} total")

    extra_field_files = 0
    extra_fields_total = Counter()
    max_fields = 0
    max_fields_file = ''

    for fp in sample:
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue

        has_extra = False
        for trk in data.get('tracks', []):
            for n in trk.get('notes', []):
                extra = set(n.keys()) - base_note_fields
                if extra:
                    has_extra = True
                    extra_fields_total.update(extra)

        if has_extra:
            extra_field_files += 1

    print(f"  Files with extra note fields: {extra_field_files} / {len(sample)}")
    if extra_fields_total:
        print(f"  Extra fields found: {dict(extra_fields_total.most_common())}")
    else:
        print(f"  NO extra note fields found beyond: {sorted(base_note_fields)}")


def analyze_tokens_more():
    """More detailed analysis: read token content to find actual token types used."""
    print("\n=== Detailed token analysis ===")
    TOKENS_DIR = os.path.join(BASE, 'data', 'processed', 'tokens')

    # We need the tokenizer to map IDs back to names
    import sys
    sys.path.insert(0, os.path.join(BASE))
    from chopinote_dataset.tokenizer import REMITokenizer

    tokenizer = REMITokenizer(grid_size=16, velocity_levels=8)

    # Build reverse vocabulary: id -> token string
    id_to_token = {v: k for k, v in tokenizer.vocab.items()}

    token_files = sorted(os.listdir(TOKENS_DIR))[:200]

    all_token_types = Counter()
    token_value_subs = Counter()

    for fn in token_files:
        fp = os.path.join(TOKENS_DIR, fn)
        with open(fp, 'r') as f:
            line = f.read().strip()

        ids = [int(x) for x in line.split() if x.strip().isdigit()]
        for tid in ids:
            token_str = id_to_token.get(tid, f'<UNKNOWN_{tid}>')
            # Extract type prefix
            type_prefix = token_str.split(' ')[0].split('>')[0]
            if '>' in token_str and not token_str.startswith('<'):
                type_prefix = token_str
            all_token_types[type_prefix] += 1

            # Track specific token values
            if '<Dynamic' in token_str:
                val = token_str.split(' ')[1].rstrip('>') if ' ' in token_str else token_str
                token_value_subs['dynamic_' + val] += 1
            elif '<Artic' in token_str:
                val = token_str.split(' ')[1].rstrip('>') if ' ' in token_str else token_str
                token_value_subs['artic_' + val] += 1
            elif '<Ornament' in token_str:
                val = token_str.split(' ')[1].rstrip('>') if ' ' in token_str else token_str
                token_value_subs['ornament_' + val] += 1
            elif '<Pedal' in token_str:
                val = token_str.split(' ')[1].rstrip('>') if ' ' in token_str else token_str
                token_value_subs['pedal_' + val] += 1
            elif '<Slur' in token_str:
                val = token_str.split(' ')[1].rstrip('>') if ' ' in token_str else token_str
                token_value_subs['slur_' + val] += 1
            elif '<Repeat' in token_str:
                val = token_str.split(' ')[1].rstrip('>') if ' ' in token_str else token_str
                token_value_subs['repeat_' + val] += 1
            elif '<Jump' in token_str:
                val = token_str.split(' ')[1].rstrip('>') if ' ' in token_str else token_str
                token_value_subs['jump_' + val] += 1
            elif '<Tempo' in token_str:
                val = token_str.split(' ')[1].rstrip('>') if ' ' in token_str else token_str
                token_value_subs['tempo_' + val] += 1
            elif '<TimeSig' in token_str:
                val = token_str.split(' ')[1].rstrip('>') if ' ' in token_str else token_str
                token_value_subs['timesig_' + val] += 1
            elif '<Key' in token_str:
                val = token_str.split(' ')[1].rstrip('>') if ' ' in token_str else token_str
                token_value_subs['key_' + val] += 1
            elif '<TupletStart' in token_str:
                val = token_str.split(' ')[1].rstrip('>') if ' ' in token_str else token_str
                token_value_subs['tupletstart_' + val] += 1
            elif '<GraceNote' in token_str:
                val = token_str.split(' ')[1].rstrip('>') if ' ' in token_str else token_str
                token_value_subs['grace_' + val] += 1
            elif '<Program' in token_str:
                val = token_str.split(' ')[1].rstrip('>') if ' ' in token_str else token_str
                token_value_subs['program_' + val] += 1
            elif '<Position' in token_str:
                val = token_str.split(' ')[1].rstrip('>') if ' ' in token_str else token_str
                token_value_subs['position_' + val] += 1
            elif '<Note_ON' in token_str:
                val = token_str.split(' ')[1].rstrip('>') if ' ' in token_str else token_str
                token_value_subs['pitch_' + val] += 1
            elif '<Hairpin' in token_str:
                val = token_str.split(' ')[1].rstrip('>') if ' ' in token_str else token_str
                token_value_subs['hairpin_' + val] += 1
            elif '<Octave' in token_str:
                val = token_str.split(' ')[1].rstrip('>') if ' ' in token_str else token_str
                token_value_subs['octave_' + val] += 1

    print(f"  Token types found in {len(token_files)} files:")
    for ttype, cnt in sorted(all_token_types.most_common()):
        print(f"    {ttype}: {cnt}")

    print(f"\n  Detailed token values:")
    for val, cnt in sorted(token_value_subs.most_common()):
        print(f"    {val}: {cnt}")


if __name__ == '__main__':
    scan_pdmx_files()
    scan_all_note_fields_wide()
    scan_pdmx_metadata_csv()
    analyze_tokens_more()

    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)

    # Files stats
    print(f"\nFiles scanned (PDMX JSON): {counts['total_files']}")
    print(f"Total notes scanned: {counts['total_notes']}")
    print(f"Total tracks scanned: {counts['total_tracks']}")

    # Note data
    print(f"\n--- Note Data ---")
    print(f"Grace notes: {counts['total_grace_notes']}")
    print(f"Files with grace notes: {counts['files_with_grace_notes']}")
    print(f"Pitch range: {counts['note_min_pitch']} - {counts['note_max_pitch']}")

    # Chords
    print(f"\n--- Chords ---")
    print(f"Files with chord data: {counts['files_with_chords']}")
    print(f"Total chords: {counts['total_chords']}")

    # Barlines
    print(f"\n--- Barline Types ---")
    print(f"  single: {counts['barlines_single']}")
    print(f"  double: {counts['barlines_double']}")
    print(f"  end: {counts['barlines_end']}")
    print(f"  start-repeat: {counts['barlines_start_repeat']}")
    print(f"  end-repeat: {counts['barlines_end_repeat']}")
    print(f"  other: {counts['barlines_other']}")

    # Programs
    print(f"\n--- Instrument Programs (top 20) ---")
    for prog, cnt in counts['programs'].most_common(20):
        print(f"  Program {prog}: {cnt}")

    # Velocity distribution
    print(f"\n--- Velocity Distribution (top 10) ---")
    for vel, cnt in counts['note_velocities'].most_common(10):
        print(f"  Velocity {vel}: {cnt}")
