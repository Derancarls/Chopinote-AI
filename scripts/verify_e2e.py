#!/usr/bin/env python3
"""端到端预处理验证：MIDI / PDMX / MusicXML 三种格式全链路检查。

Usage: python scripts/verify_e2e.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import List

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from chopinote_dataset.converter import MusicXMLToREMI, PDMXToREMI
from chopinote_dataset.fast_converter import FastMIDIToREMI
from chopinote_dataset.renderer import REMIToMusicXML
from chopinote_dataset.tokenizer import REMITokenizer

GRID = 16
VEL = 8

# ── Test file paths ──────────────────────────────────────────────

# MIDI: POP909 / 041
MIDI_PATH = "/root/autodl-tmp/POP909/001/001.mid"

# MusicXML: ATEPP Ravel Jeux d'eau
MUSICXML_PATH = "/root/autodl-tmp/ATEPP-1.2/Maurice_Ravel/Jeux_d'eau,_M._30/musicxml_cleaned.musicxml"

# ── Synthetic PDMX ────────────────────────────────────────────────

def _make_pdmx_data() -> dict:
    """创建一个含 4 小节 4/4 拍的完整 PDMX。"""
    return {
        'resolution': 480,
        'barlines': [
            {'measure': 1, 'time': 0},
            {'measure': 2, 'time': 1920},
            {'measure': 3, 'time': 3840},
            {'measure': 4, 'time': 5760},
            {'measure': 5, 'time': 7680},
        ],
        'time_signatures': [
            {'measure': 1, 'numerator': 4, 'denominator': 4},
        ],
        'key_signatures': [
            {'measure': 1, 'root_str': 'C', 'mode': 'major'},
        ],
        'tempos': [
            {'time': 0, 'qpm': 120},
        ],
        'tracks': [
            {
                'program': 0,
                'notes': [
                    # measure 1: C-E-G 和弦 + beat 2 rest + beat 3-4 single notes
                    {'measure': 1, 'time': 0, 'pitch': 60, 'velocity': 80, 'duration': 480},
                    {'measure': 1, 'time': 0, 'pitch': 64, 'velocity': 80, 'duration': 480},
                    {'measure': 1, 'time': 0, 'pitch': 67, 'velocity': 80, 'duration': 480},
                    {'measure': 1, 'time': 960, 'pitch': 62, 'velocity': 70, 'duration': 480},
                    {'measure': 1, 'time': 1440, 'pitch': 65, 'velocity': 70, 'duration': 480},
                    # measure 2: scale run
                    {'measure': 2, 'time': 0, 'pitch': 60, 'velocity': 75, 'duration': 240},
                    {'measure': 2, 'time': 240, 'pitch': 62, 'velocity': 75, 'duration': 240},
                    {'measure': 2, 'time': 480, 'pitch': 64, 'velocity': 75, 'duration': 240},
                    {'measure': 2, 'time': 720, 'pitch': 65, 'velocity': 75, 'duration': 240},
                    {'measure': 2, 'time': 960, 'pitch': 67, 'velocity': 75, 'duration': 240},
                    {'measure': 2, 'time': 1200, 'pitch': 69, 'velocity': 75, 'duration': 240},
                    {'measure': 2, 'time': 1440, 'pitch': 71, 'velocity': 75, 'duration': 240},
                    {'measure': 2, 'time': 1680, 'pitch': 72, 'velocity': 75, 'duration': 240},
                    # measure 3: rest on beat 1, grace note on beat 2
                    {'measure': 3, 'time': 960, 'pitch': 64, 'velocity': 80, 'duration': 480},
                    {'measure': 3, 'time': 960, 'pitch': 65, 'velocity': 70, 'duration': 30, 'is_grace': True},
                    {'measure': 3, 'time': 1440, 'pitch': 67, 'velocity': 75, 'duration': 480},
                    # measure 4: ending
                    {'measure': 4, 'time': 0, 'pitch': 72, 'velocity': 90, 'duration': 1920},
                ],
                'annotations': [],
            },
            {
                'program': 0,
                'notes': [
                    # Bass line
                    {'measure': 1, 'time': 0, 'pitch': 48, 'velocity': 70, 'duration': 1920},
                    {'measure': 2, 'time': 0, 'pitch': 48, 'velocity': 70, 'duration': 960},
                    {'measure': 2, 'time': 960, 'pitch': 50, 'velocity': 70, 'duration': 960},
                    {'measure': 3, 'time': 0, 'pitch': 43, 'velocity': 65, 'duration': 1920},
                    {'measure': 4, 'time': 0, 'pitch': 48, 'velocity': 70, 'duration': 1920},
                ],
                'annotations': [],
            },
        ],
    }


# ── Analysis helpers ──────────────────────────────────────────────

def inspect_tokens(token_ids: List[int], tokenizer: REMITokenizer, label: str):
    """打印 token 序列关键统计。"""
    events = tokenizer.detokenize(token_ids)
    type_counts = {}
    for ttype, _ in events:
        type_counts[ttype] = type_counts.get(ttype, 0) + 1

    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    print(f"  Token IDs: {len(token_ids)}")
    print(f"  Events:    {len(events)}")
    print(f"  Bars:      {type_counts.get('<Bar>', 0)}")
    print(f"  Notes:     {type_counts.get('<Note_ON', 0)}")
    rests = type_counts.get('<Rest>', 0)
    print(f"  Rests:     {rests}")
    print(f"  Grace:     {type_counts.get('<GraceNote', 0)}")
    print(f"  Pedal:     {type_counts.get('<Pedal', 0)}")
    print(f"  Velocities:{type_counts.get('<Velocity', 0)}")
    print(f"  Durations: {type_counts.get('<Duration', 0)}")
    print(f"  KeySig:    {type_counts.get('<KeySig', 0)}")
    print(f"  TimeSig:   {type_counts.get('<TimeSig', 0)}")

    # Check: every Note_ON has Velocity + Duration after it
    missing_vel = 0
    missing_dur = 0
    for i, (ttype, _) in enumerate(events):
        if ttype == '<Note_ON':
            if i + 2 >= len(events) or events[i + 1][0] != '<Velocity':
                missing_vel += 1
            if i + 2 >= len(events) or events[i + 2][0] != '<Duration':
                missing_dur += 1
    if missing_vel or missing_dur:
        print(f"  WARNING: {missing_vel} notes missing Velocity, {missing_dur} missing Duration")
    else:
        print(f"  OK: All Note_ON have Velocity+Duration")

    # Check: every Rest has Duration
    missing_rest_dur = 0
    for i, (ttype, _) in enumerate(events):
        if ttype == '<Rest>':
            if i + 1 >= len(events) or events[i + 1][0] != '<Duration':
                missing_rest_dur += 1
    if missing_rest_dur:
        print(f"  WARNING: {missing_rest_dur} rests missing Duration")
    else:
        print(f"  OK: All Rests have Duration")

    # Print first 20 events as sample
    print(f"\n  First 20 events:")
    for i, (ttype, val) in enumerate(events[:20]):
        print(f"    [{i:3d}] {ttype:20s} {str(val):>10s}")


def bridge_test(token_ids_a: List[int], tokenizer: REMITokenizer,
                label: str) -> dict:
    """桥接测试: REMI → MusicXML → REMI 自洽性。"""
    renderer = REMIToMusicXML(GRID, VEL)
    mx_conv = MusicXMLToREMI(GRID, VEL)

    with tempfile.NamedTemporaryFile(suffix='.musicxml', delete=False) as f:
        xml_path = f.name

    try:
        events_a = tokenizer.detokenize(token_ids_a)
        renderer.render(events_a, xml_path)
        tokens_b, _ = mx_conv.convert(xml_path, collect_metadata=True)

        if not tokens_b:
            return {'error': '桥接转换返回空', 'label': label}

        events_b = tokenizer.detokenize(tokens_b)

        # Compare token sequences
        max_len = max(len(events_a), len(events_b))
        min_len = min(len(events_a), len(events_b))
        exact = sum(1 for i in range(min_len)
                     if events_a[i] == events_b[i])
        rate = exact / max_len if max_len > 0 else 0

        # Bar count comparison
        bars_a = sum(1 for e in events_a if e[0] == '<Bar>')
        bars_b = sum(1 for e in events_b if e[0] == '<Bar>')

        # Note count comparison
        notes_a = sum(1 for e in events_a if e[0] == '<Note_ON')
        notes_b = sum(1 for e in events_b if e[0] == '<Note_ON')

        return {
            'label': label,
            'events_a': len(events_a),
            'events_b': len(events_b),
            'exact_matches': exact,
            'total_comparable': max_len,
            'consistency': f'{rate:.1%}',
            'bars_a': bars_a,
            'bars_b': bars_b,
            'notes_a': notes_a,
            'notes_b': notes_b,
        }
    except Exception as e:
        return {'error': str(e), 'label': label}
    finally:
        try:
            os.remove(xml_path)
        except OSError:
            pass


# ── Main ──────────────────────────────────────────────────────────

def main():
    tokenizer = REMITokenizer(GRID, VEL)
    results = {}

    print("=" * 60)
    print("  端到端预处理验证")
    print("=" * 60)

    # ── 1. MIDI ─────────────────────────────────────────────────
    print("\n" + "█" * 60)
    print("  1. MIDI (FastMIDIToREMI)")
    print("█" * 60)

    if os.path.exists(MIDI_PATH):
        fc = FastMIDIToREMI(GRID, VEL)
        midi_tokens, midi_meta = fc.convert(MIDI_PATH, collect_metadata=True)
        inspect_tokens(midi_tokens, tokenizer, f"MIDI: {os.path.basename(MIDI_PATH)}")
        print(f"  Metadata: bars={midi_meta.get('num_measures', '?')}, "
              f"notes={midi_meta.get('num_notes', '?')}")
        results['MIDI_convert'] = 'PASS' if midi_tokens else 'FAIL (empty)'

        if midi_tokens:
            br = bridge_test(midi_tokens, tokenizer, "MIDI bridge")
            _print_bridge(br)
            results['MIDI_bridge'] = br
    else:
        print(f"  SKIP: MIDI 文件不存在: {MIDI_PATH}")
        results['MIDI_convert'] = 'SKIP'

    # ── 2. PDMX ─────────────────────────────────────────────────
    print("\n" + "█" * 60)
    print("  2. PDMX (Synthetic)")
    print("█" * 60)

    pdmx_data = _make_pdmx_data()
    pconv = PDMXToREMI(GRID, VEL)
    pdmx_tokens, pdmx_meta = pconv.convert_pdmx(pdmx_data, collect_metadata=True)
    inspect_tokens(pdmx_tokens, tokenizer, "PDMX (4-bar C major, 2 tracks)")
    print(f"  Metadata: bars={pdmx_meta.get('num_measures', '?')}, "
          f"notes={pdmx_meta.get('num_notes', '?')}, "
          f"tracks={pdmx_meta.get('num_tracks', '?')}")
    results['PDMX_convert'] = 'PASS' if pdmx_tokens else 'FAIL (empty)'

    if pdmx_tokens:
        br = bridge_test(pdmx_tokens, tokenizer, "PDMX bridge")
        _print_bridge(br)
        results['PDMX_bridge'] = br

    # ── 3. MusicXML ─────────────────────────────────────────────
    print("\n" + "█" * 60)
    print("  3. MusicXML")
    print("█" * 60)

    if os.path.exists(MUSICXML_PATH):
        mxc = MusicXMLToREMI(GRID, VEL)
        xml_tokens, xml_meta = mxc.convert(MUSICXML_PATH, collect_metadata=True)
        inspect_tokens(xml_tokens, tokenizer, f"MusicXML: {os.path.basename(MUSICXML_PATH)}")
        print(f"  Metadata: bars={xml_meta.get('num_measures', '?')}, "
              f"notes={xml_meta.get('num_notes', '?')}")
        results['MusicXML_convert'] = 'PASS' if xml_tokens else 'FAIL (empty)'

        if xml_tokens:
            br = bridge_test(xml_tokens, tokenizer, "MusicXML bridge")
            _print_bridge(br)
            results['MusicXML_bridge'] = br
    else:
        print(f"  SKIP: MusicXML 文件不存在: {MUSICXML_PATH}")
        results['MusicXML_convert'] = 'SKIP'

    # ── 4. Summary ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  验证总结")
    print("=" * 60)

    all_pass = True
    for key, val in results.items():
        if isinstance(val, dict):
            if 'error' in val:
                print(f"  {key}: FAIL — {val['error']}")
                all_pass = False
            else:
                print(f"  {key}: consistency={val.get('consistency', '?')} "
                      f"({val.get('exact_matches', 0)}/{val.get('total_comparable', 0)})")
        else:
            status = val
            if status.startswith('FAIL'):
                all_pass = False
            print(f"  {key}: {status}")

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    return 0 if all_pass else 1


def _print_bridge(br: dict):
    if 'error' in br:
        print(f"  Bridge test: FAIL — {br['error']}")
        return
    print(f"  Bridge test:")
    print(f"    Events: A={br['events_a']}  B={br['events_b']}")
    print(f"    Bars:   A={br['bars_a']}  B={br['bars_b']}")
    print(f"    Notes:  A={br['notes_a']}  B={br['notes_b']}")
    print(f"    Consistency: {br['consistency']} "
          f"({br['exact_matches']}/{br['total_comparable']})")


if __name__ == '__main__':
    sys.exit(main())
