"""REMI events → MusicXML renderer (渲染器).

Converts REMI token events back to a music21 Score and writes MusicXML.
Independent from converter and generation — only depends on tokenizer for detokenization.

Usage:
    from chopinote_dataset.renderer import REMIToMusicXML
    renderer = REMIToMusicXML(grid_size=16, velocity_levels=8)
    score = renderer.render(events)
    renderer.write(events, 'output.musicxml')
"""

import re
import sys
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

from music21 import (
    bar, chord, clef, duration, dynamics, expressions, instrument, key,
    meter, note, spanner, stream, tempo, tie, volume,
)

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from chopinote_dataset.tokenizer import REMITokenizer, key_name_to_tonic_midi

logger = logging.getLogger(__name__)

# ── Articulation map ────────────────────────────────────────────
_ARTIC_MAP = {
    'staccato': 'Staccato', 'accent': 'Accent', 'tenuto': 'Tenuto',
    'marcato': 'StrongAccent', 'pizzicato': 'Pizzicato', 'fermata': 'Fermata',
}

# ── Ornament map (music21 expressions) ──────────────────────────
_ORN_MAP = {
    'trill': 'Trill', 'mordent': 'Mordent', 'turn': 'Turn', 'tremolo': 'Tremolo',
}

# ── Key fifths → default alter per step (for accidental cleanup) ─
_KEY_SIG_ALTER: dict[int, dict[str, int]] = {
    -7: {'C':-1,'D':-1,'E':-1,'F':-1,'G':-1,'A':-1,'B':-1},
    -6: {'C':-1,'D':-1,'E':-1,'F':-1,'G':-1,'A':-1,'B': 0},
    -5: {'C':-1,'D': 0,'E':-1,'F':-1,'G':-1,'A':-1,'B': 0},
    -4: {'C':-1,'D': 0,'E':-1,'F':-1,'G': 0,'A':-1,'B': 0},
    -3: {'C': 0,'D': 0,'E':-1,'F':-1,'G': 0,'A': 0,'B': 0},
    -2: {'C': 0,'D': 0,'E':-1,'F': 0,'G': 0,'A': 0,'B': 0},
    -1: {'C': 0,'D': 0,'E': 0,'F':-1,'G': 0,'A': 0,'B': 0},
     0: {'C': 0,'D': 0,'E': 0,'F': 0,'G': 0,'A': 0,'B': 0},
     1: {'C': 0,'D': 0,'E': 0,'F': 1,'G': 0,'A': 0,'B': 0},
     2: {'C': 0,'D': 0,'E': 0,'F': 1,'G': 1,'A': 0,'B': 0},
     3: {'C': 0,'D': 0,'E': 0,'F': 1,'G': 1,'A': 1,'B': 0},
     4: {'C': 0,'D': 0,'E': 1,'F': 1,'G': 1,'A': 1,'B': 0},
     5: {'C': 0,'D': 1,'E': 1,'F': 1,'G': 1,'A': 1,'B': 0},
     6: {'C': 0,'D': 1,'E': 1,'F': 1,'G': 1,'A': 1,'B': 1},
     7: {'C': 1,'D': 1,'E': 1,'F': 1,'G': 1,'A': 1,'B': 1},
}


class REMIToMusicXML:
    """REMI token events → music21 Score → MusicXML."""

    def __init__(self, grid_size: int = 16, velocity_levels: int = 8):
        self.grid_size = grid_size
        self.velocity_levels = velocity_levels
        self.quarter_per_position = 4.0 / grid_size
        self.tokenizer = REMITokenizer(grid_size, velocity_levels)

    # ── Public API ──────────────────────────────────────────────

    def render_from_tokens(self, token_ids: list[int],
                           output_path: str | None = None):
        """Detokenize then render to music21 Score. Optionally write to file."""
        events = self.tokenizer.detokenize(token_ids)
        return self.render(events, output_path)

    def render(self, events: list, output_path: str | None = None):
        """Convert REMI events to music21 Score. Optionally write MusicXML."""
        parsed = self._parse_events(events)
        score = self._build_score(parsed)
        if output_path:
            score.write('musicxml', fp=output_path)
            self._inject_directions(output_path, parsed)
            self._cleanup_accidentals(output_path)
        return score

    def write(self, events: list, output_path: str):
        """Shortcut: render events and write to MusicXML file."""
        return self.render(events, output_path)

    # ── Core ────────────────────────────────────────────────────

    def _events_to_score(self, events):
        """Build a music21 Score from REMI events."""
        parsed = self._parse_events(events)
        return self._build_score(parsed)

    def _parse_events(self, events):
        """Parse REMI events into structured per-measure-per-position data.

        Returns a dict with all information needed to build a Score.
        """
        parts: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        key_per_measure: dict = {}
        timesig_per_measure: dict = {}
        tempo_per_measure: dict = {}
        clef_per_measure: dict = defaultdict(list)

        # ── Performance events ──
        dynamics_events: list = []
        pedal_events: list = []
        hairpin_events: list = []
        octave_events: list = []
        slur_events: list = []
        arpeggio_events: list = []

        cur_measure = -1
        cur_position = 0
        current_key_name: Optional[str] = None
        current_timesig: Optional[str] = None
        current_tempo: Optional[int] = None
        current_prog: int = 0
        current_sub: int = 0

        # ── Pending note accumulation (list for chord support) ──
        pending_notes: list[dict] = []  # [{interval, vel, prog, sub, articulations, ornaments, tuplet}]
        pending_rest: bool = False
        pending_rest_prog: int = 0
        pending_rest_sub: int = 0
        pending_grace_type: Optional[str] = None

        # ── Position-level accumulators (cleared on position change) ──
        pos_articulations: list = []
        pos_ornaments: list = []

        # ── Tuplet tracking (spans multiple notes) ──
        current_tuplet: Optional[tuple] = None

        for token_type, value in events:
            # ── Skip special tokens ──
            if token_type in ('<PAD>', '<BOS>', '<EOS>', '<MASK>'):
                continue

            # ── Measure boundary ──
            if token_type == '<Bar>':
                if cur_measure >= 0:
                    key_per_measure[cur_measure] = current_key_name
                    timesig_per_measure[cur_measure] = current_timesig
                    tempo_per_measure[cur_measure] = current_tempo
                cur_measure += 1
                cur_position = 0

            # ── Position ──
            elif token_type == '<Position':
                cur_position = value
                pos_articulations.clear()
                pos_ornaments.clear()

            # ── Context tokens ──
            elif token_type == '<Key':
                current_key_name = value

            elif token_type == '<TimeSig':
                current_timesig = value

            elif token_type == '<Tempo':
                current_tempo = value

            elif token_type == '<Clef':
                clef_per_measure[cur_measure].append((current_prog, current_sub, value))

            elif token_type == '<Program':
                s = str(value)
                if '_' in s:
                    parts_l = s.split('_')
                    current_prog = int(parts_l[0])
                    current_sub = int(parts_l[1])
                else:
                    current_prog = int(s)
                    current_sub = 0

            # ── Tuplet context ──
            elif token_type == '<TupletStart':
                current_tuplet = _parse_tuplet_ratio(value)
            elif token_type == '<TupletEnd>':
                current_tuplet = None

            # ── Grace note context (applies to NEXT note only) ──
            elif token_type == '<GraceNote':
                pending_grace_type = value

            # ── Note events ──
            elif token_type == '<Note_ON':
                # Chord support: accumulate multiple Note_ON before Duration
                pending_notes.append({
                    'interval': value,
                    'vel': 0,
                    'prog': current_prog,
                    'sub': current_sub,
                    'articulations': list(pos_articulations),
                    'ornaments': list(pos_ornaments),
                    'tuplet': current_tuplet,
                })
                pending_rest = False

            elif token_type == '<Velocity':
                # Apply velocity to ALL pending notes
                for pn in pending_notes:
                    pn['vel'] = value

            elif token_type == '<Duration':
                if pending_notes:
                    tonic = key_name_to_tonic_midi(current_key_name)
                    for pn in pending_notes:
                        midi_pitch = tonic + pn['interval']
                        midi_pitch = max(0, min(127, midi_pitch))
                        parts[(pn['prog'], pn['sub'])][cur_measure][cur_position].append({
                            'type': 'note',
                            'pitch': midi_pitch,
                            'velocity': pn['vel'],
                            'duration': value,
                            'articulations': list(pn['articulations']),
                            'ornaments': list(pn['ornaments']),
                            'grace_type': pending_grace_type,
                            'tuplet': pn['tuplet'],
                        })
                    pending_notes.clear()
                    pending_grace_type = None
                elif pending_rest:
                    parts[(pending_rest_prog, pending_rest_sub)][cur_measure][cur_position].append({
                        'type': 'rest',
                        'duration': value,
                    })
                    pending_rest = False

            elif token_type == '<Rest>':
                pending_rest = True
                pending_rest_prog = current_prog
                pending_rest_sub = current_sub
                pending_notes.clear()

            # ── Expressive markings (position-level + pending) ──
            elif token_type == '<Artic':
                pos_articulations.append(value)
                for pn in pending_notes:
                    pn['articulations'].append(value)
            elif token_type == '<Ornament':
                pos_ornaments.append(value)
                for pn in pending_notes:
                    pn['ornaments'].append(value)

            # ── Performance events (position-based) ──
            elif token_type == '<Dynamic':
                dynamics_events.append((cur_measure, cur_position,
                                        current_prog, current_sub, value))
            elif token_type == '<Pedal':
                pedal_events.append((cur_measure, cur_position,
                                     current_prog, current_sub, value))
            elif token_type == '<Hairpin':
                hairpin_events.append((cur_measure, cur_position,
                                       current_prog, current_sub, value))
            elif token_type == '<Octave':
                octave_events.append((cur_measure, cur_position,
                                      current_prog, current_sub, value))
            elif token_type == '<Slur':
                slur_events.append((cur_measure, cur_position,
                                    current_prog, current_sub, value))
            elif token_type == '<Arpeggio>':
                arpeggio_events.append((cur_measure, cur_position,
                                        current_prog, current_sub))

            # ── Informational tokens (skip for rendering) ──
            elif token_type in ('<Beat', '<Bass', '<Anticipate',
                                '<Repeat', '<Jump'):
                pass

        # Last measure context
        if cur_measure >= 0:
            key_per_measure[cur_measure] = current_key_name
            timesig_per_measure[cur_measure] = current_timesig
            tempo_per_measure[cur_measure] = current_tempo

        total_measures = cur_measure + 1 if cur_measure >= 0 else 0

        return {
            'parts': dict(parts),
            'key_per_measure': key_per_measure,
            'timesig_per_measure': timesig_per_measure,
            'tempo_per_measure': tempo_per_measure,
            'clef_per_measure': dict(clef_per_measure),
            'dynamics': dynamics_events,
            'pedal_events': pedal_events,
            'hairpin_events': hairpin_events,
            'octave_events': octave_events,
            'slur_events': slur_events,
            'arpeggio_events': arpeggio_events,
            'total_measures': total_measures,
        }

    def _build_score(self, parsed: dict):
        """Build music21 Score from parsed data."""
        score = stream.Score()
        part_keys = sorted(parsed['parts'].keys())
        if not part_keys:
            return score

        total_measures = parsed['total_measures']
        if total_measures == 0:
            return score

        last_key_name = None
        last_timesig_str = None

        # ── Index performance events by (measure, position, prog, sub) ──
        dyn_index = _index_events(parsed['dynamics'])
        arp_index = _index_events(parsed['arpeggio_events'], has_value=False)

        # ── Slur pairing ──
        slur_spans = _pair_slurs(parsed['slur_events'])

        for prog, sub in part_keys:
            part_data = parsed['parts'][(prog, sub)]
            part_stream = stream.Part()
            part_stream.id = f'P{prog}_{sub}'

            try:
                inst = instrument.fromMidiProgram(prog)
                part_stream.append(inst)
            except Exception:
                pass

            # ── Track open slurs for this part ──
            part_slur_spans = [
                (sm, sp, em, ep) for (sm, sp, em, ep, p, sb) in slur_spans
                if (p, sb) == (prog, sub)
            ]
            # Position-keyed lookup for start/end
            slur_starts: dict = defaultdict(list)
            slur_ends: dict = defaultdict(list)
            for sm, sp, em, ep in part_slur_spans:
                slur_starts[(sm, sp)].append((em, ep))
                slur_ends[(em, ep)].append((sm, sp))
            pending_slurs: list = []  # [(end_m, end_pos, accumulated_notes), ...]

            for m in range(total_measures):
                measure_obj = stream.Measure(number=m + 1)

                # ── Time signature ──
                ts_str = parsed['timesig_per_measure'].get(m)
                if ts_str and ts_str in REMITokenizer.TIME_SIGNATURES:
                    num, den = ts_str.split('/')
                    ts = meter.TimeSignature(f'{num}/{den}')
                    measure_obj.timeSignature = ts
                    last_timesig_str = ts_str

                # ── Key signature ──
                key_name = parsed['key_per_measure'].get(m)
                if key_name and key_name != last_key_name:
                    try:
                        k = key.Key(key_name)
                        ks = key.KeySignature(k.sharps)
                        measure_obj.append(ks)
                        last_key_name = key_name
                    except Exception:
                        pass

                # ── Clef ──
                for c_prog, c_sub, clef_name in parsed['clef_per_measure'].get(m, []):
                    if (c_prog, c_sub) == (prog, sub):
                        _append_clef(measure_obj, clef_name)

                # ── Notes / Rests / Dynamics ──
                positions = sorted(part_data.get(m, {}).keys())
                for pos in positions:
                    note_items = part_data[m][pos]
                    offset = pos * self.quarter_per_position

                    # Dynamics at this position
                    for d_prog, d_sub, d_val in dyn_index.get((m, pos), []):
                        if (d_prog, d_sub) == (prog, sub):
                            measure_obj.insert(offset, dynamics.Dynamic(d_val))

                    # Separate notes and rests
                    notes_at_pos = [ni for ni in note_items if ni['type'] == 'note']
                    rests_at_pos = [ni for ni in note_items if ni['type'] == 'rest']

                    notes_created: list = []  # note objects for slur accumulation
                    if notes_at_pos:
                        # Grace notes first
                        grace_notes = [n for n in notes_at_pos if n.get('grace_type')]
                        regular_notes = [n for n in notes_at_pos if not n.get('grace_type')]

                        for gn in grace_notes:
                            g_note = self._make_grace_note(gn)
                            measure_obj.insert(offset, g_note)

                        if regular_notes:
                            if len(regular_notes) == 1:
                                ni = regular_notes[0]
                                n = self._make_note(ni, offset)
                                measure_obj.append(n)
                                notes_created.append(n)
                                if ni.get('tuplet'):
                                    _apply_tuplet(n, ni['tuplet'])
                            else:
                                c = chord.Chord(
                                    [ni['pitch'] for ni in regular_notes],
                                    quarterLength=regular_notes[0]['duration'] * self.quarter_per_position,
                                )
                                c.offset = offset
                                c.volume.velocity = self._vel_to_midi(
                                    regular_notes[0]['velocity'])
                                for ni in regular_notes:
                                    for art_name in ni.get('articulations', []):
                                        _attach_articulation(c, art_name)
                                    for orn_name in ni.get('ornaments', []):
                                        _attach_ornament(c, orn_name)
                                if regular_notes[0].get('tuplet'):
                                    _apply_tuplet(c, regular_notes[0]['tuplet'])
                                # Arpeggio
                                if (prog, sub) in arp_index.get((m, pos), []):
                                    try:
                                        c.expressions.append(expressions.ArpeggioMark())
                                    except Exception:
                                        pass
                                measure_obj.append(c)
                                notes_created.append(c)

                    # Slur starts
                    for em, ep in slur_starts.get((m, pos), []):
                        pending_slurs.append((em, ep, []))
                    # Accumulate note objects into all pending slurs
                    for n_obj in notes_created:
                        for _, _, acc in pending_slurs:
                            acc.append(n_obj)
                    # Slur ends
                    for sm, sp in slur_ends.get((m, pos), []):
                        for i, (em2, ep2, acc) in enumerate(pending_slurs):
                            if (em2, ep2) == (m, pos):
                                if len(acc) >= 2:
                                    s = spanner.Slur(acc)
                                    measure_obj.insert(offset, s)
                                pending_slurs.pop(i)
                                break

                    if rests_at_pos:
                        ri = rests_at_pos[0]
                        r = note.Rest(
                            quarterLength=ri.get('duration', 1) * self.quarter_per_position)
                        r.offset = offset
                        measure_obj.append(r)

                part_stream.append(measure_obj)

            score.append(part_stream)

        # ── Tempo marks ──
        for m, bpm in parsed['tempo_per_measure'].items():
            if bpm is not None and m < total_measures:
                mm = tempo.MetronomeMark(number=bpm)
                parts_list = list(score.parts)
                if parts_list:
                    measures = list(parts_list[0].getElementsByClass('Measure'))
                    if m < len(measures):
                        measures[m].insert(0, mm)

        return score

    def _make_note(self, ni: dict, offset: float) -> note.Note:
        """Create a music21 Note from parsed note info."""
        dur = ni['duration'] * self.quarter_per_position
        n = note.Note(ni['pitch'], quarterLength=dur)
        n.offset = offset
        vel = self._vel_to_midi(ni['velocity'])
        n.volume.velocity = vel

        for art_name in ni.get('articulations', []):
            _attach_articulation(n, art_name)
        for orn_name in ni.get('ornaments', []):
            _attach_ornament(n, orn_name)

        return n

    def _make_grace_note(self, ni: dict) -> note.Note:
        """Create a grace note from parsed note info."""
        gn = note.Note(ni['pitch'])
        gt = ni.get('grace_type', 'grace')
        gn.duration = duration.GraceDuration('eighth')
        gn.duration.slash = (gt == 'acciaccatura')
        gn.volume.velocity = self._vel_to_midi(ni['velocity'])
        return gn

    def _vel_to_midi(self, velocity_level: int) -> int:
        """Convert velocity level (0~7) back to MIDI velocity (0~127)."""
        bucket_size = 128 // self.velocity_levels
        return velocity_level * bucket_size + bucket_size // 2

    # ── XML Post-processing ─────────────────────────────────────

    def _inject_directions(self, filepath: str, parsed: dict):
        """Inject pedal, hairpin, and octave directions into MusicXML.

        music21 Spanner subclasses (PedalMark, Crescendo, Ottava) don't serialize
        correctly, so we inject them directly into the written XML.
        """
        part_keys = sorted(parsed['parts'].keys())
        key_to_occurrence = {k: i + 1 for i, k in enumerate(part_keys)}

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        def _inject_into_measure(xml, bar_num, xml_snippet, target_occ):
            pattern = rf'(<measure[^>]*?number="{bar_num}"[^>]*>.*?)(</measure>)'
            matches = list(re.finditer(pattern, xml, re.DOTALL))
            if target_occ > len(matches):
                return xml
            m = matches[target_occ - 1]
            return (xml[:m.start()] + m.group(1) + '\n' + xml_snippet + '\n    '
                    + m.group(2) + xml[m.end():])

        def _inject(ev_list, xml_tmpl_fn):
            nonlocal content
            for ev in ev_list:
                m_num, pos, prog, sub, val = ev
                target = key_to_occurrence.get((prog, sub), 1)
                content = _inject_into_measure(content, m_num + 1,
                                               xml_tmpl_fn(val), target)

        # Pedal
        _inject(parsed['pedal_events'],
                lambda v: (
                    '      <direction placement="below">'
                    '\n        <direction-type>'
                    f'\n          <pedal type="{v}" line="yes" sign="yes"/>'
                    '\n        </direction-type>'
                    '\n      </direction>'
                ))

        # Hairpin (crescendo/diminuendo)
        _inject(parsed['hairpin_events'],
                lambda v: (
                    '      <direction placement="below">'
                    '\n        <direction-type>'
                    f'\n          <wedge type="{"crescendo" if v == "cresc" else "diminuendo"}"'
                    ' number="1" spread="0"/>'
                    '\n        </direction-type>'
                    '\n      </direction>'
                ))

        # Octave shifts
        for ev in parsed['octave_events']:
            m_num, pos, prog, sub, val = ev
            target = key_to_occurrence.get((prog, sub), 1)
            if val == 'end':
                snippet = (
                    '      <direction placement="below">'
                    '\n        <direction-type>'
                    '\n          <octave-shift type="stop" size="8" number="1"/>'
                    '\n        </direction-type>'
                    '\n      </direction>'
                )
            else:
                is_down = 'b' in val
                shift_type = 'down' if is_down else 'up'
                snippet = (
                    '      <direction placement="below">'
                    '\n        <direction-type>'
                    f'\n          <octave-shift type="{shift_type}" size="8" number="1"/>'
                    '\n        </direction-type>'
                    '\n      </direction>'
                )
            content = _inject_into_measure(content, m_num + 1, snippet, target)

        if content != content:  # never true, just to avoid unused warning pattern
            pass

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

    def _cleanup_accidentals(self, filepath: str):
        """Remove unnecessary natural signs from MusicXML.

        Per measure: only keep natural signs that actually cancel a preceding
        accidental or override the key signature default.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        measure_re = re.compile(r'(<measure[^>]*>.*?</measure>)', re.DOTALL)
        fifths_re = re.compile(r'<fifths>(-?\d+)</fifths>')
        step_re = re.compile(r'<step>([A-G])</step>')
        alter_re = re.compile(r'<alter>(-?\d+)</alter>')
        note_re = re.compile(r'<note[^>]*>.*?</note>', re.DOTALL)

        def _process_measure(measure_xml: str) -> str:
            fifths = 0
            fm = fifths_re.search(measure_xml)
            if fm:
                fifths = int(fm.group(1))
            altered_steps: set[str] = set()

            for nb in note_re.finditer(measure_xml):
                note_xml = nb.group(0)
                sm = step_re.search(note_xml)
                if not sm:
                    continue
                step = sm.group(1)
                am = alter_re.search(note_xml)
                alter_val = int(am.group(1)) if am else 0
                default = _KEY_SIG_ALTER.get(fifths, {}).get(step, 0)

                if alter_val != default:
                    altered_steps.add(step)

                if '<accidental>natural</accidental>' in note_xml:
                    needed = (step in altered_steps) or (default != 0 and alter_val == 0)
                    if not needed:
                        cleaned = re.sub(
                            r'\s*<accidental>natural</accidental>\s*\n?',
                            '\n', note_xml, count=1)
                        measure_xml = measure_xml.replace(note_xml, cleaned, 1)
            return measure_xml

        result = measure_re.sub(lambda m: _process_measure(m.group(1)), content)

        if result != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(result)


# ── Module-level helpers ────────────────────────────────────────

def _parse_tuplet_ratio(val: str) -> tuple:
    """Parse '3:2' → (3, 2)."""
    parts = val.split(':')
    return (int(parts[0]), int(parts[1]))


def _index_events(ev_list: list, has_value: bool = True) -> dict:
    """Index events by (measure, position, prog, sub)."""
    idx = defaultdict(list)
    for ev in ev_list:
        if has_value:
            m, pos, prog, sub, val = ev
            idx[(m, pos)].append((prog, sub, val))
        else:
            m, pos, prog, sub = ev
            idx[(m, pos)].append((prog, sub))
    return dict(idx)


def _pair_slurs(slur_events: list) -> list:
    """Pair slur start/end events into (start_m, start_pos, end_m, end_pos, prog, sub)."""
    open_slurs: dict[Tuple, Tuple[int, int]] = {}  # (prog, sub) → (m, pos)
    spans = []
    for ev in slur_events:
        m, pos, prog, sub, val = ev
        key = (prog, sub)
        if val == 'start':
            open_slurs[key] = (m, pos)
        elif val == 'end':
            if key in open_slurs:
                sm, sp = open_slurs.pop(key)
                spans.append((sm, sp, m, pos, prog, sub))
    return spans


def _append_clef(measure_obj, clef_name: str):
    """Append a clef object to a measure."""
    clef_map = {
        'treble': clef.TrebleClef, 'bass': clef.BassClef,
        'alto': clef.AltoClef, 'tenor': clef.TenorClef,
    }
    cls = clef_map.get(clef_name)
    if cls:
        measure_obj.append(cls())


def _attach_articulation(target, art_name: str):
    """Attach an articulation to a Note or Chord."""
    from music21 import articulations as art21
    art_map = {
        'staccato': art21.Staccato, 'accent': art21.Accent,
        'tenuto': art21.Tenuto, 'marcato': art21.StrongAccent,
        'pizzicato': art21.Pizzicato,
    }
    cls = art_map.get(art_name.lower() if isinstance(art_name, str) else '')
    if cls:
        target.articulations.append(cls())
    elif art_name == 'fermata':
        from music21 import expressions as ex
        target.expressions.append(ex.Fermata())


def _attach_ornament(target, orn_name: str):
    """Attach an ornament to a Note or Chord."""
    from music21 import expressions as ex
    orn_map = {
        'trill': ex.Trill, 'mordent': ex.Mordent,
        'turn': ex.Turn, 'tremolo': ex.Tremolo,
    }
    cls = orn_map.get(orn_name.lower() if isinstance(orn_name, str) else '')
    if cls:
        target.expressions.append(cls())


def _apply_tuplet(note_or_chord, tuplet_ratio: tuple):
    """Apply tuplet duration scaling to a Note or Chord."""
    actual, normal = tuplet_ratio
    t = duration.Tuplet()
    t.numberNotesActual = actual
    t.numberNotesNormal = normal
    note_or_chord.duration.tuplets = (t,)
