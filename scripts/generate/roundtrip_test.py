"""Round-trip fidelity test (转换同度测试).

Independent of both converter and renderer internals.
Pipeline: MusicXML → REMI tokens → MusicXML → compare.

Covers: notes, rests, structure (key/timesig/clef), dynamics, articulations,
ornaments, arpeggios, slurs, grace notes, tuplets, pedal, hairpin, octave.

Usage:
    python scripts/roundtrip_test.py input.musicxml
    python scripts/roundtrip_test.py --dir /path/to/dir --sample 100
    python scripts/roundtrip_test.py --dir /path/to/dir --all
    python scripts/roundtrip_test.py --summarize reports/roundtrip/
"""

import argparse
import glob
import json
import os
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import music21
from music21 import (
    chord, clef, converter as m21conv, dynamics, expressions, key,
    meter, note, spanner, tempo,
)

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# ── Data structures ────────────────────────────────────────────

@dataclass
class NoteInfo:
    measure: int
    part_idx: int
    midi_pitch: int
    position: int
    duration: int
    velocity: int


@dataclass
class MeasureContext:
    key_name: Optional[str] = None
    timesig: Optional[str] = None


@dataclass
class MarkingsReport:
    """Cumulative counts for performance markings."""
    dynamics: int = 0
    articulations: int = 0
    ornaments: int = 0
    arpeggios: int = 0
    slurs: int = 0
    grace_notes: int = 0
    tuplets: int = 0

    def as_dict(self) -> dict:
        return {
            'dynamics': self.dynamics,
            'articulations': self.articulations,
            'ornaments': self.ornaments,
            'arpeggios': self.arpeggios,
            'slurs': self.slurs,
            'grace_notes': self.grace_notes,
            'tuplets': self.tuplets,
        }


@dataclass
class XMLDirections:
    """Counts of XML-injected direction elements (don't survive m21 round-trip)."""
    pedals: int = 0
    hairpins: int = 0
    octaves: int = 0

    def as_dict(self) -> dict:
        return {'pedals': self.pedals, 'hairpins': self.hairpins, 'octaves': self.octaves}


@dataclass
class RoundTripReport:
    file: str = ''
    total_measures: int = 0

    # Structure
    key_errs: int = 0
    timesig_errs: int = 0
    clef_errs: int = 0

    # Notes
    orig_notes: int = 0
    rt_notes: int = 0
    note_matched: int = 0
    note_missed: int = 0
    note_extra: int = 0
    position_jitter: list = field(default_factory=list)
    duration_error: list = field(default_factory=list)
    velocity_error: list = field(default_factory=list)

    # Rests
    orig_rests: int = 0
    rt_rests: int = 0

    # Markings (music21-extractable)
    orig_markings: MarkingsReport = field(default_factory=MarkingsReport)
    rt_markings: MarkingsReport = field(default_factory=MarkingsReport)

    # XML directions (post-processing items)
    orig_directions: XMLDirections = field(default_factory=XMLDirections)
    rt_directions: XMLDirections = field(default_factory=XMLDirections)


# ── 1. Extraction from music21 Score ───────────────────────────

def _quantize_position(offset: float, quarter_per_position: float,
                        grid_size: int) -> int:
    return min(grid_size - 1, max(0, int(round(offset / quarter_per_position))))


def extract_all_from_score(score, grid_size: int = 16,
                           musicxml_path: str = '') -> dict:
    """Extract notes, structure, rests, markings, and XML directions.

    Returns a dict with all extracted data for comparison.
    """
    parts = list(score.parts)
    quarter_per_position = 4.0 / grid_size
    all_notes: list[NoteInfo] = []
    contexts: dict[int, MeasureContext] = {}
    rest_counts: dict[int, int] = {}
    markings = MarkingsReport()

    current_key_name: Optional[str] = None
    current_timesig: Optional[str] = None

    # ── Key from score ──
    try:
        kos = score.flatten().getElementsByClass(key.Key)
        if kos:
            k = kos[0]
            current_key_name = k.tonic.name + ('m' if k.mode == 'minor' else '')
        else:
            kss = score.flatten().getElementsByClass(key.KeySignature)
            if kss:
                k = kss[0].asKey()
                current_key_name = k.tonic.name + ('m' if k.mode == 'minor' else '')
    except Exception:
        for ks in score.flatten().getElementsByClass(key.KeySignature):
            try:
                current_key_name = _sharps_to_key_name(ks.sharps, 'major')
            except Exception:
                pass
            break

    for part_idx, part in enumerate(parts):
        rest_counts[part_idx] = 0
        measure_seq = 0  # sequential 0-based index, robust to pickup/anacrusis numbering

        for measure in part.getElementsByClass('Measure'):
            mn = measure_seq
            measure_seq += 1

            # Time signature
            ts = measure.timeSignature
            if ts:
                ts_str = f'{ts.numerator}/{ts.denominator}'
                if '/' in ts_str:
                    current_timesig = ts_str

            # Context snapshot
            if mn not in contexts:
                contexts[mn] = MeasureContext(
                    key_name=current_key_name,
                    timesig=current_timesig,
                )

            # ── Dynamics ──
            for d in measure.flatten().getElementsByClass(dynamics.Dynamic):
                markings.dynamics += 1

            # ── Notes, rests, and their attributes ──
            for elem in measure.flatten().notesAndRests:
                if isinstance(elem, note.Rest):
                    rest_counts[part_idx] += 1
                    continue

                pos = _quantize_position(elem.offset, quarter_per_position, grid_size)

                # Grace notes
                if hasattr(elem.duration, 'isGrace') and elem.duration.isGrace:
                    markings.grace_notes += 1
                    continue

                # Tuplets
                if elem.duration.tuplets:
                    markings.tuplets += 1

                # Articulations
                markings.articulations += len(elem.articulations)

                # Ornaments (expressions like Trill, Mordent, Turn, Tremolo)
                for exp in elem.expressions:
                    if type(exp).__name__ in ('Trill', 'Mordent', 'Turn', 'Tremolo'):
                        markings.ornaments += 1

                # Arpeggio
                for exp in elem.expressions:
                    if type(exp).__name__ == 'ArpeggioMark':
                        markings.arpeggios += 1

                # Pitch extraction
                pitches = []
                if isinstance(elem, note.Note):
                    pitches = [elem.pitch.midi]
                elif isinstance(elem, chord.Chord):
                    pitches = [n.pitch.midi for n in elem.notes]

                vel = int(elem.volume.velocity) if elem.volume.velocity else 64
                vel_level = min(7, vel // 16)
                dur = max(1, min(grid_size,
                                  int(round(elem.quarterLength / quarter_per_position))))

                for p in pitches:
                    all_notes.append(NoteInfo(
                        measure=mn,
                        part_idx=part_idx,
                        midi_pitch=p,
                        position=pos,
                        duration=dur,
                        velocity=vel_level,
                    ))

    # ── Slurs (spanners on the score) ──
    for s in score.flatten().getElementsByClass(spanner.Slur):
        markings.slurs += 1

    # ── Backfill context ──
    ck, ct = current_key_name, current_timesig
    for m in range(max(contexts.keys()) + 1 if contexts else 0):
        if m not in contexts:
            contexts[m] = MeasureContext(key_name=ck, timesig=ct)
        else:
            if contexts[m].key_name:
                ck = contexts[m].key_name
            else:
                contexts[m].key_name = ck
            if contexts[m].timesig:
                ct = contexts[m].timesig
            else:
                contexts[m].timesig = ct

    # ── XML directions (pedal, hairpin, octave) ──
    directions = XMLDirections()
    if musicxml_path and os.path.isfile(musicxml_path):
        try:
            with open(musicxml_path, 'r', encoding='utf-8') as f:
                xml_content = f.read()
            directions.pedals = len(re.findall(r'<pedal\b', xml_content))
            directions.hairpins = len(re.findall(r'<wedge\b', xml_content))
            directions.octaves = len(re.findall(r'<octave-shift\b', xml_content))
        except Exception:
            pass

    return {
        'notes': all_notes,
        'contexts': contexts,
        'rest_counts': rest_counts,
        'markings': markings,
        'directions': directions,
    }


def _sharps_to_key_name(sharps: int, mode: str = 'major') -> str:
    major_map = {-7: 'Cb', -6: 'Gb', -5: 'Db', -4: 'Ab', -3: 'Eb', -2: 'Bb', -1: 'F',
                 0: 'C', 1: 'G', 2: 'D', 3: 'A', 4: 'E', 5: 'B', 6: 'F#', 7: 'C#'}
    minor_map = {-7: 'Abm', -6: 'Ebm', -5: 'Bbm', -4: 'Fm', -3: 'Cm', -2: 'Gm', -1: 'Dm',
                 0: 'Am', 1: 'Em', 2: 'Bm', 3: 'F#m', 4: 'C#m', 5: 'G#m', 6: 'D#m', 7: 'A#m'}
    if mode == 'minor':
        return minor_map.get(sharps, 'C')
    return major_map.get(sharps, 'C')


# ── 2. Comparison ──────────────────────────────────────────────

def compare_all(orig: dict, rt: dict) -> RoundTripReport:
    """Compare original and round-trip extracted data."""
    report = RoundTripReport()

    # ── Notes ──
    orig_notes = orig['notes']
    rt_notes = rt['notes']
    report.orig_notes = len(orig_notes)
    report.rt_notes = len(rt_notes)

    # Compare notes by (measure, pitch) — part-order-independent.
    # Group all parts together per measure for pitch-based matching.
    orig_meas: dict = defaultdict(lambda: defaultdict(list))
    for ni in orig_notes:
        orig_meas[ni.measure][ni.midi_pitch].append(ni)
    rt_meas: dict = defaultdict(lambda: defaultdict(list))
    for ni in rt_notes:
        rt_meas[ni.measure][ni.midi_pitch].append(ni)

    jitters, dur_errs, vel_errs = [], [], []
    all_measures = sorted(set(list(orig_meas.keys()) + list(rt_meas.keys())))
    for m in all_measures:
        orig_pitches = orig_meas.get(m, {})
        rt_pitches = rt_meas.get(m, {})
        for pitch, o_list in orig_pitches.items():
            r_list = rt_pitches.get(pitch, [])
            for o_note in o_list:
                if r_list:
                    r_note = r_list.pop(0)
                    jitters.append(abs(o_note.position - r_note.position))
                    dur_errs.append(abs(o_note.duration - r_note.duration))
                    vel_errs.append(abs(o_note.velocity - r_note.velocity))
                    report.note_matched += 1
                else:
                    report.note_missed += 1
        for r_list in rt_pitches.values():
            report.note_extra += len(r_list)

    report.position_jitter = jitters
    report.duration_error = dur_errs
    report.velocity_error = vel_errs

    # ── Structure ──
    all_measures = set(list(orig['contexts'].keys()) + list(rt['contexts'].keys()))
    for m in sorted(all_measures):
        o = orig['contexts'].get(m, MeasureContext())
        r = rt['contexts'].get(m, MeasureContext())
        if o.key_name and r.key_name and o.key_name != r.key_name:
            report.key_errs += 1
        if o.timesig and r.timesig and o.timesig != r.timesig:
            report.timesig_errs += 1

    # ── Rests ──
    report.orig_rests = sum(orig['rest_counts'].values())
    report.rt_rests = sum(rt['rest_counts'].values())

    # ── Markings ──
    report.orig_markings = orig['markings']
    report.rt_markings = rt['markings']

    # ── XML directions ──
    report.orig_directions = orig['directions']
    report.rt_directions = rt['directions']

    return report


# ── 3. Single-file evaluation ──────────────────────────────────

def evaluate_roundtrip(musicxml_path: str, grid_size: int = 16) -> dict:
    """Single-file round-trip evaluation."""
    from chopinote_dataset.converter import MusicXMLToREMI
    from chopinote_dataset.renderer import REMIToMusicXML

    if not os.path.isfile(musicxml_path):
        return {'error': f'File not found: {musicxml_path}'}

    # ── Load original ──
    try:
        score_orig = m21conv.parse(musicxml_path)
    except Exception as e:
        return {'error': f'Parse failed: {e}'}

    # ── Extract from original ──
    orig_data = extract_all_from_score(score_orig, grid_size, musicxml_path)

    # ── Step 1: MusicXML → REMI events ──
    try:
        conv = MusicXMLToREMI(grid_size=grid_size, velocity_levels=8)
        events = conv._score_to_events(score_orig)
    except Exception as e:
        return {'error': f'Convert failed: {e}'}
    if not events:
        return {'error': 'Converter returned empty events'}

    # ── Step 2: REMI events → MusicXML ──
    try:
        renderer = REMIToMusicXML(grid_size=grid_size, velocity_levels=8)
        tmp_path = '/tmp/roundtrip_test.musicxml'
        renderer.write(events, tmp_path)
        score_rt = m21conv.parse(tmp_path)
    except Exception as e:
        return {'error': f'Render failed: {e}'}

    # ── Extract from round-trip ──
    rt_data = extract_all_from_score(score_rt, grid_size, tmp_path)

    # ── Compare ──
    report = compare_all(orig_data, rt_data)
    report.file = musicxml_path
    report.total_measures = max(len(orig_data['contexts']), len(rt_data['contexts']))

    return _report_to_dict(report)


def _report_to_dict(report: RoundTripReport) -> dict:
    """Convert RoundTripReport to JSON-serializable dict."""
    jitters = report.position_jitter
    dur_errs = report.duration_error
    vel_errs = report.velocity_error
    return {
        'file': report.file,
        'total_measures': report.total_measures,
        'structure': {
            'key_errs': report.key_errs,
            'timesig_errs': report.timesig_errs,
            'clef_errs': report.clef_errs,
        },
        'notes': {
            'orig_count': report.orig_notes,
            'rt_count': report.rt_notes,
            'matched': report.note_matched,
            'missed': report.note_missed,
            'extra': report.note_extra,
            'recall_pct': round(report.note_matched / max(report.orig_notes, 1) * 100, 2),
            'precision_pct': round(report.note_matched / max(report.rt_notes, 1) * 100, 2),
        },
        'quantization': {
            'position_jitter_mean': round(sum(jitters) / max(len(jitters), 1), 3),
            'position_jitter_max': max(jitters) if jitters else 0,
            'duration_error_mean': round(sum(dur_errs) / max(len(dur_errs), 1), 3),
            'duration_error_max': max(dur_errs) if dur_errs else 0,
            'velocity_error_mean': round(sum(vel_errs) / max(len(vel_errs), 1), 3),
            'velocity_error_max': max(vel_errs) if vel_errs else 0,
        },
        'rests': {
            'orig_count': report.orig_rests,
            'rt_count': report.rt_rests,
        },
        'markings': {
            'orig': report.orig_markings.as_dict(),
            'rt': report.rt_markings.as_dict(),
        },
        'directions': {
            'orig': report.orig_directions.as_dict(),
            'rt': report.rt_directions.as_dict(),
        },
    }


# ── 4. Batch & CLI ────────────────────────────────────────────

def evaluate_directory(input_dir: str, sample: Optional[int] = None,
                        output_dir: str = 'roundtrip_reports') -> list:
    """Batch round-trip evaluation."""
    files = sorted(glob.glob(os.path.join(input_dir, '*.musicxml')) +
                   glob.glob(os.path.join(input_dir, '*.mxl')))
    if len(files) < 5:
        more = sorted(glob.glob(os.path.join(input_dir, '**/*.musicxml'), recursive=True) +
                      glob.glob(os.path.join(input_dir, '**/*.mxl'), recursive=True))
        files = list(dict.fromkeys(files + more))

    print(f"Found {len(files)} files")

    if sample and sample < len(files):
        random.seed(42)
        files = random.sample(files, sample)
        print(f"Sampling {sample} files")

    os.makedirs(output_dir, exist_ok=True)
    reports = []
    errors = []

    for i, fpath in enumerate(files):
        name = os.path.basename(fpath)
        print(f"  [{i+1}/{len(files)}] {name}...", end=' ', flush=True)
        result = evaluate_roundtrip(fpath)
        if 'error' in result:
            print(f"ERROR: {result['error']}")
            errors.append({'file': fpath, 'error': result['error']})
        else:
            r = result['notes']['recall_pct']
            m = result['notes']['missed']
            print(f"recall={r:.1f}% missed={m}")
            reports.append(result)
            stem = Path(fpath).stem.replace('/', '_')
            with open(os.path.join(output_dir, f'{stem}.json'), 'w') as f:
                json.dump(result, f, indent=2)

    if reports:
        agg = _aggregate(reports)
        agg_path = os.path.join(output_dir, '_aggregate.json')
        with open(agg_path, 'w') as f:
            json.dump(agg, f, indent=2)
        print(f"\nAggregate saved: {agg_path}")
        _print_aggregate(agg)

    if errors:
        with open(os.path.join(output_dir, '_errors.json'), 'w') as f:
            json.dump(errors, f, indent=2)
        print(f"Errors ({len(errors)} files): saved")

    return reports


def _aggregate(reports: list) -> dict:
    """Aggregate multiple reports."""
    agg = {'num_files': len(reports)}

    def _ptiles(key, subkey=None):
        vals = []
        for r in reports:
            v = r.get(key) if subkey is None else r.get(key, {}).get(subkey)
            if isinstance(v, (int, float)) and v is not None:
                vals.append(float(v))
        if not vals:
            return None
        return _describe(vals)

    agg['notes'] = {
        'recall_pct': _ptiles('notes', 'recall_pct'),
        'missed': _ptiles('notes', 'missed'),
        'extra': _ptiles('notes', 'extra'),
    }
    agg['structure'] = {
        'key_errs': _ptiles('structure', 'key_errs'),
        'timesig_errs': _ptiles('structure', 'timesig_errs'),
    }
    agg['quantization'] = {
        'pos_jitter_mean': _ptiles('quantization', 'position_jitter_mean'),
    }

    # Markings comparison: per-file diff
    for field in ('dynamics', 'articulations', 'ornaments',
                  'arpeggios', 'slurs', 'grace_notes', 'tuplets'):
        diffs = []
        for r in reports:
            om = r.get('markings', {}).get('orig', {}).get(field, 0)
            rm = r.get('markings', {}).get('rt', {}).get(field, 0)
            diffs.append(abs(om - rm))
        if diffs:
            agg.setdefault('markings_diff', {})[field] = {
                'total_diff': sum(diffs),
                'files_with_diff': sum(1 for d in diffs if d > 0),
            }

    # Directions comparison
    for field in ('pedals', 'hairpins', 'octaves'):
        diffs = []
        for r in reports:
            od = r.get('directions', {}).get('orig', {}).get(field, 0)
            rd = r.get('directions', {}).get('rt', {}).get(field, 0)
            diffs.append(abs(od - rd))
        if diffs:
            agg.setdefault('directions_diff', {})[field] = {
                'total_diff': sum(diffs),
                'files_with_diff': sum(1 for d in diffs if d > 0),
            }

    return agg


def _describe(vals: list) -> dict:
    vals = sorted(vals)
    n = len(vals)
    return {
        'count': n, 'min': round(vals[0], 2), 'max': round(vals[-1], 2),
        'mean': round(sum(vals) / n, 2),
        'p50': round(vals[n // 2], 2),
        'p95': round(vals[int(n * 0.95)], 2) if n > 2 else round(vals[-1], 2),
    }


def _print_aggregate(agg: dict):
    """Print a readable aggregate."""
    print(f"\n{'='*60}")
    print(f"  Round-Trip Report ({agg['num_files']} files)")
    print(f"{'='*60}\n")

    # 1. Notes
    nr = agg.get('notes', {}).get('recall_pct', {})
    if nr:
        print("  1. Note Recall (pitch-based)")
        print(f"     mean={nr['mean']:.1f}%  p50={nr['p50']:.1f}%  "
              f"p95={nr['p95']:.1f}%  min={nr['min']:.1f}%")
        print()

    # 2. Structure
    sk = agg.get('structure', {}).get('key_errs', {})
    st = agg.get('structure', {}).get('timesig_errs', {})
    if sk and st:
        print("  2. Structure errors (per-file avg)")
        print(f"     Key errs:     mean={sk['mean']:.1f}  max={sk['max']:.0f}")
        print(f"     TimeSig errs: mean={st['mean']:.1f}  max={st['max']:.0f}")
        print()

    # 3. Quantization
    qm = agg.get('quantization', {}).get('pos_jitter_mean', {})
    if qm:
        print("  3. Quantization")
        print(f"     Position jitter (mu): {qm['mean']:.3f}  max={qm['max']:.3f}")
        print()

    # 4. Markings
    md = agg.get('markings_diff', {})
    if md:
        print("  4. Performance Markings (total files with mismatches)")
        for field, info in sorted(md.items()):
            name = field.replace('_', ' ').title()
            print(f"     {name:<16}: {info['files_with_diff']} files "
                  f"(total diff={info['total_diff']})")
        print()

    # 5. XML Directions
    dd = agg.get('directions_diff', {})
    if dd:
        print("  5. XML Directions (pedal, hairpin, octave)")
        for field, info in sorted(dd.items()):
            name = field.replace('_', ' ').title()
            print(f"     {name:<16}: {info['files_with_diff']} files "
                  f"(total diff={info['total_diff']})")
        print()


def main():
    parser = argparse.ArgumentParser(description='Round-trip fidelity test')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('file', nargs='?', help='Single MusicXML file')
    group.add_argument('--dir', help='Directory of MusicXML files')
    group.add_argument('--report', help='Summarize existing reports')
    parser.add_argument('--sample', type=int, default=None, help='Sample N files')
    parser.add_argument('--output', type=str, default='roundtrip_reports',
                        help='Output directory')
    args = parser.parse_args()

    if args.file:
        result = evaluate_roundtrip(args.file)
        if 'error' in result:
            print(f"ERROR: {result['error']}")
            return
        _print_single(result)

    elif args.dir:
        evaluate_directory(args.dir, sample=args.sample, output_dir=args.output)

    elif args.report:
        reports = []
        for fpath in glob.glob(os.path.join(args.report, '*.json')):
            if os.path.basename(fpath).startswith('_'):
                continue
            with open(fpath) as f:
                reports.append(json.load(f))
        print(f"Loaded {len(reports)} reports")
        agg = _aggregate(reports)
        _print_aggregate(agg)


def _fmt_pct(numerator: int, denominator: int) -> str:
    """Format as percentage, or N/A if denominator is zero."""
    if denominator == 0:
        return 'N/A'
    return f'{numerator / denominator * 100:.1f}%'


def _print_single(r: dict):
    """Pretty-print a single-file report."""
    print(f"File: {r['file']}")
    print(f"Measures: {r['total_measures']}")
    print()

    # ── Structure ──
    print("  Structure:")
    print(f"    Key errors:     {r['structure']['key_errs']}")
    print(f"    TimeSig errors:  {r['structure']['timesig_errs']}")
    print(f"    Clef errors:    {r['structure']['clef_errs']}")
    print()

    # ── Notes ──
    n = r['notes']
    print("  Notes:")
    print(f"    Original:  {n['orig_count']}")
    print(f"    RoundTrip: {n['rt_count']}")
    print(f"    Matched:   {n['matched']}")
    print(f"    Recall:    {n['recall_pct']:.1f}%")
    print(f"    Precision: {n['precision_pct']:.1f}%")
    print(f"    Missed:    {n['missed']}")
    print(f"    Extra:     {n['extra']}")
    print()

    # ── Quantization ──
    q = r['quantization']
    print("  Quantization:")
    print(f"    Position jitter: mean={q['position_jitter_mean']:.3f}  max={q['position_jitter_max']}")
    print(f"    Duration error:  mean={q['duration_error_mean']:.3f}  max={q['duration_error_max']}")
    print(f"    Velocity error:  mean={q['velocity_error_mean']:.3f}  max={q['velocity_error_max']}")
    print()

    # ── Rests ──
    print("  Rests:")
    print(f"    Original:  {r['rests']['orig_count']}")
    print(f"    RoundTrip: {r['rests']['rt_count']}")
    print()

    # ── Markings ──
    om = r.get('markings', {}).get('orig', {})
    rm = r.get('markings', {}).get('rt', {})
    print("  Performance Markings (orig → rt):")
    fields = [
        ('dynamics', 'Dynamics'),
        ('articulations', 'Articulations'),
        ('ornaments', 'Ornaments'),
        ('arpeggios', 'Arpeggios'),
        ('slurs', 'Slurs'),
        ('grace_notes', 'Grace Notes'),
        ('tuplets', 'Tuplets'),
    ]
    for key, label in fields:
        o_val = om.get(key, 0)
        r_val = rm.get(key, 0)
        diff = r_val - o_val
        diff_str = f'{diff:+d}' if diff != 0 else 'ok'
        print(f"    {label:<16}: {o_val} → {r_val}  ({diff_str})")
    print()

    # ── XML Directions ──
    od = r.get('directions', {}).get('orig', {})
    rd = r.get('directions', {}).get('rt', {})
    print("  XML Directions (orig → rt):")
    for key, label in [('pedals', 'Pedal'), ('hairpins', 'Hairpin'), ('octaves', 'Octave')]:
        o_val = od.get(key, 0)
        r_val = rd.get(key, 0)
        diff = r_val - o_val
        diff_str = f'{diff:+d}' if diff != 0 else 'ok'
        print(f"    {label:<16}: {o_val} → {r_val}  ({diff_str})")


if __name__ == '__main__':
    main()
