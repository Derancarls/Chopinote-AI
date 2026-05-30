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
        # Divisions per quarter note for raw MusicXML output.
        # grid_size=16 → 4 divisions/quarter, so each position step = 1 division.
        self.raw_divisions = int(1.0 / self.quarter_per_position)

    # ── Public API ──────────────────────────────────────────────

    def render_from_tokens(self, token_ids: list[int],
                           output_path: str | None = None,
                           fast_path: bool = False):
        """Detokenize then render to music21 Score. Optionally write to file.

        Args:
            fast_path: If True, write MusicXML directly without music21.
                       ~100x faster, suitable for C evaluation review.
                       Does NOT support MIDI export or advanced features.
        """
        events = self.tokenizer.detokenize(token_ids)
        return self.render(events, output_path, fast_path=fast_path)

    def render(self, events: list, output_path: str | None = None,
               fast_path: bool = False):
        """Convert REMI events to music21 Score. Optionally write MusicXML."""
        parsed = self._parse_events(events)
        if fast_path and output_path:
            self._render_raw_xml(parsed, output_path)
            return None  # No music21 Score in fast path
        score = self._build_score(parsed)
        if output_path:
            score.write('musicxml', fp=output_path)
            self._inject_directions(output_path, parsed)
            self._cleanup_accidentals(output_path)
        return score

    def _render_raw_xml(self, parsed: dict, output_path: str):
        """Write MusicXML directly from parsed events — no music21.

        ~100x faster than building a music21 Score. Generates valid
        MusicXML 4.0 suitable for C evaluation review (review_musicxml,
        compare_tokens_to_xml).

        Does NOT handle: MIDI export, grace notes, tuplets, slurs.
        """
        import xml.etree.ElementTree as ET

        root = ET.Element('score-partwise', version='4.0')
        tree = ET.ElementTree(root)

        part_keys = sorted(parsed['parts'].keys())
        total_measures = parsed['total_measures']
        divisions = self.raw_divisions  # e.g. 4 for grid_size=16

        # ── Part list ──
        part_list = ET.SubElement(root, 'part-list')
        for prog, sub in part_keys:
            score_part = ET.SubElement(part_list, 'score-part', id=f'P{prog}_{sub}')
            part_name = ET.SubElement(score_part, 'part-name')
            part_name.text = f'Program {prog}' + (f'_{sub}' if sub else '')

        # ── Key fifths mapping (avoid music21 import) ──
        _KEY_TO_FIFTHS = {
            'C': 0, 'G': 1, 'D': 2, 'A': 3, 'E': 4, 'B': 5, 'F#': 6, 'C#': 7,
            'F': -1, 'Bb': -2, 'Eb': -3, 'Ab': -4, 'Db': -5, 'Gb': -6, 'Cb': -7,
        }

        # ── Clef name → (sign, line) mapping ──
        _CLEF_TO_SIGN_LINE = {
            'treble': ('G', '2'), 'bass': ('F', '4'),
            'alto': ('C', '3'), 'tenor': ('C', '4'),
        }

        # ── Parts ──
        for prog, sub in part_keys:
            part_data = parsed['parts'][(prog, sub)]
            part_el = ET.SubElement(root, 'part', id=f'P{prog}_{sub}')

            last_key_name = None
            for m in range(total_measures):
                measure_el = ET.SubElement(part_el, 'measure', number=str(m + 1))

                # Attributes
                ts_str = parsed['timesig_per_measure'].get(m)
                key_name = parsed['key_per_measure'].get(m)
                clefs = parsed['clef_per_measure'].get(m, [])

                if m == 0 or ts_str or (key_name and key_name != last_key_name) or clefs:
                    attr_el = ET.SubElement(measure_el, 'attributes')
                    div_el = ET.SubElement(attr_el, 'divisions')
                    div_el.text = str(divisions)

                    if ts_str and ts_str in REMITokenizer.TIME_SIGNATURES:
                        num, den = ts_str.split('/')
                        time_el = ET.SubElement(attr_el, 'time')
                        b_el = ET.SubElement(time_el, 'beats')
                        b_el.text = num
                        bt_el = ET.SubElement(time_el, 'beat-type')
                        bt_el.text = den

                    if key_name and key_name != last_key_name:
                        fifths = _KEY_TO_FIFTHS.get(key_name.replace(' minor', '').replace(' major', ''), 0)
                        key_el = ET.SubElement(attr_el, 'key')
                        f_el = ET.SubElement(key_el, 'fifths')
                        f_el.text = str(fifths)
                        last_key_name = key_name

                    for c_prog, c_sub, clef_name in clefs:
                        if (c_prog, c_sub) == (prog, sub):
                            cl_sign, cl_line = _CLEF_TO_SIGN_LINE.get(
                                clef_name, ('G', '2'))
                            clef_el = ET.SubElement(attr_el, 'clef')
                            ET.SubElement(clef_el, 'sign').text = cl_sign
                            ET.SubElement(clef_el, 'line').text = cl_line

                # ── Notes / Rests ──
                positions = sorted(part_data.get(m, {}).keys())
                for pos in positions:
                    items = part_data[m][pos]
                    for ni in items:
                        if ni['type'] == 'rest':
                            note_el = ET.SubElement(measure_el, 'note')
                            ET.SubElement(note_el, 'rest')
                            dur = ni.get('duration', 1)
                            ET.SubElement(note_el, 'duration').text = str(int(dur))
                        elif ni['type'] == 'note':
                            note_el = ET.SubElement(measure_el, 'note')
                            # Pitch: stored as integer MIDI number in parsed events
                            midi_pitch = ni.get('pitch')
                            if midi_pitch is not None:
                                step_str, octave, alter = _midi_to_step_octave(midi_pitch)
                                pitch_el = ET.SubElement(note_el, 'pitch')
                                ET.SubElement(pitch_el, 'step').text = step_str
                                ET.SubElement(pitch_el, 'octave').text = str(octave)
                                if alter != 0:
                                    ET.SubElement(pitch_el, 'alter').text = str(alter)
                            dur = ni.get('duration', 1)
                            ET.SubElement(note_el, 'duration').text = str(int(dur))

        tree.write(output_path, xml_declaration=True, encoding='utf-8')

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

        Uses ElementTree for DOM-level manipulation — O(S + E×log M) instead of
        the old O(E×S) regex+string-rebuild approach.
        """
        import xml.etree.ElementTree as ET

        part_keys = sorted(parsed['parts'].keys())
        key_to_occurrence = {k: i + 1 for i, k in enumerate(part_keys)}

        tree = ET.parse(filepath)
        root = tree.getroot()
        ns = root.tag.split('}')[0] + '}' if '}' in root.tag else ''

        def _ns(local: str) -> str:
            return f'{ns}{local}'

        parts = root.findall(_ns('part'))
        if not parts:
            return

        def _inject_into_part(part_el, bar_num: int, xml_snippet: str):
            for measure in part_el.findall(_ns('measure')):
                if measure.get('number') == str(bar_num):
                    direction_el = ET.fromstring(xml_snippet)
                    # Insert before the first note or at the end
                    first_note = measure.find(_ns('note'))
                    if first_note is not None:
                        # Insert before first note
                        note_idx = list(measure).index(first_note)
                        measure.insert(note_idx, direction_el)
                    else:
                        measure.append(direction_el)
                    return

        def _inject(ev_list, xml_tmpl_fn):
            for ev in ev_list:
                m_num, pos, prog, sub, val = ev
                target = key_to_occurrence.get((prog, sub), 1)
                if target <= len(parts):
                    _inject_into_part(
                        parts[target - 1], m_num + 1, xml_tmpl_fn(val))

        # Pedal
        _inject(parsed['pedal_events'],
                lambda v: (
                    '<direction placement="below">'
                    '<direction-type>'
                    f'<pedal type="{v}" line="yes" sign="yes"/>'
                    '</direction-type>'
                    '</direction>'
                ))

        # Hairpin (crescendo/diminuendo)
        _inject(parsed['hairpin_events'],
                lambda v: (
                    '<direction placement="below">'
                    '<direction-type>'
                    f'<wedge type="{"crescendo" if v == "cresc" else "diminuendo"}"'
                    ' number="1" spread="0"/>'
                    '</direction-type>'
                    '</direction>'
                ))

        # Octave shifts
        for ev in parsed['octave_events']:
            m_num, pos, prog, sub, val = ev
            target = key_to_occurrence.get((prog, sub), 1)
            if target > len(parts):
                continue
            if val == 'end':
                snippet = (
                    '<direction placement="below">'
                    '<direction-type>'
                    '<octave-shift type="stop" size="8" number="1"/>'
                    '</direction-type>'
                    '</direction>'
                )
            else:
                is_down = 'b' in val
                shift_type = 'down' if is_down else 'up'
                snippet = (
                    '<direction placement="below">'
                    '<direction-type>'
                    f'<octave-shift type="{shift_type}" size="8" number="1"/>'
                    '</direction-type>'
                    '</direction>'
                )
            _inject_into_part(parts[target - 1], m_num + 1, snippet)

        tree.write(filepath, xml_declaration=True, encoding='utf-8')

    def _cleanup_accidentals(self, filepath: str):
        """Remove unnecessary natural signs from MusicXML.

        Per measure: only keep natural signs that actually cancel a preceding
        accidental or override the key signature default.

        Uses ElementTree for DOM-level manipulation — O(M×N) instead of the
        old O(M×N²) regex+str.replace approach.
        """
        import xml.etree.ElementTree as ET

        tree = ET.parse(filepath)
        root = tree.getroot()
        ns = root.tag.split('}')[0] + '}' if '}' in root.tag else ''

        def _ns(local: str) -> str:
            return f'{ns}{local}'

        changed = False

        for part_el in root.findall(_ns('part')):
            for measure in part_el.findall(_ns('measure')):
                fifths = 0
                # Parse key signature for this measure
                attr_el = measure.find(_ns('attributes'))
                if attr_el is not None:
                    key_el = attr_el.find(_ns('key'))
                    if key_el is not None:
                        fifths_el = key_el.find(_ns('fifths'))
                        if fifths_el is not None and fifths_el.text:
                            try:
                                fifths = int(fifths_el.text)
                            except ValueError:
                                pass

                altered_steps: set[str] = set()

                # First pass: track altered steps
                for note_el in measure.findall(_ns('note')):
                    step_el = note_el.find(_ns('pitch'))
                    if step_el is None:
                        continue
                    step_name_el = step_el.find(_ns('step'))
                    if step_name_el is None:
                        continue
                    step = step_name_el.text
                    alter_el = step_el.find(_ns('alter'))
                    alter_val = int(alter_el.text) if alter_el is not None and alter_el.text else 0
                    default = _KEY_SIG_ALTER.get(fifths, {}).get(step, 0)

                    if alter_val != default:
                        altered_steps.add(step)

                # Second pass: remove unnecessary naturals
                for note_el in list(measure.findall(_ns('note'))):
                    step_el = note_el.find(_ns('pitch'))
                    if step_el is None:
                        continue
                    step_name_el = step_el.find(_ns('step'))
                    if step_name_el is None:
                        continue
                    step = step_name_el.text
                    alter_el = step_el.find(_ns('alter'))
                    alter_val = int(alter_el.text) if alter_el is not None and alter_el.text else 0
                    default = _KEY_SIG_ALTER.get(fifths, {}).get(step, 0)

                    accidental_el = note_el.find(_ns('accidental'))
                    if accidental_el is not None and accidental_el.text == 'natural':
                        needed = (step in altered_steps) or (default != 0 and alter_val == 0)
                        if not needed:
                            note_el.remove(accidental_el)
                            changed = True

        if changed:
            tree.write(filepath, xml_declaration=True, encoding='utf-8')


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


def _midi_to_step_octave(midi_pitch: int) -> tuple[str, int, int]:
    """Convert MIDI pitch number to (step, octave, alter).
    e.g. 60 → ('C', 4, 0), 61 → ('C', 4, 1), 70 → ('A', 4, 1)
    """
    _STEP_NAMES = ['C', 'C', 'D', 'D', 'E', 'F', 'F', 'G', 'G', 'A', 'A', 'B']
    _ALTERS =      [ 0,   1,   0,   1,   0,   0,   1,   0,   1,   0,   1,   0]
    octave = (midi_pitch // 12) - 1
    pc = midi_pitch % 12
    return _STEP_NAMES[pc], octave, _ALTERS[pc]


def _apply_tuplet(note_or_chord, tuplet_ratio: tuple):
    """Apply tuplet duration scaling to a Note or Chord."""
    actual, normal = tuplet_ratio
    t = duration.Tuplet()
    t.numberNotesActual = actual
    t.numberNotesNormal = normal
    note_or_chord.duration.tuplets = (t,)
