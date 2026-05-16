"""
Fast MIDI → REMI converter using mido (up to 80x faster than music21-based version).

Reads MIDI files directly with mido and produces identical REMI token format
as the original MIDIToREMI, but without the music21 overhead.
"""
import os
import json
import hashlib
import uuid
import time
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, asdict

import mido
import numpy as np

from .tokenizer import REMITokenizer, key_name_to_tonic_midi

logger = logging.getLogger(__name__)
_NO_KEY_WARNED = False

# ── 调号 MIDI 映射 ──────────────────────────────────────────
_KEY_SIG_MAP = {
    -7: 'C#m', -6: 'F#m', -5: 'Bm', -4: 'Em', -3: 'Am',
    -2: 'Dm', -1: 'Gm', 0: 'Am', 1: 'Em', 2: 'Bm', 3: 'F#m',
    4: 'C#m', 5: 'G#m', 6: 'D#m', 7: 'A#m',
}
_KEY_SIG_MAJOR = {
    -7: 'Cb', -6: 'Gb', -5: 'Db', -4: 'Ab', -3: 'Eb',
    -2: 'Bb', -1: 'F', 0: 'C', 1: 'G', 2: 'D', 3: 'A',
    4: 'E', 5: 'B', 6: 'F#', 7: 'C#',
}
_KEY_PC_MAP = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
    'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11,
}
_DRUM_CHANNEL = 9

# Supported time signatures in REMI (synced with REMITokenizer.TIME_SIGNATURES)
REMI_TIME_SIGS = set(REMITokenizer.TIME_SIGNATURES)


def _key_name_from_sf(sharps_flats: int, minor: bool) -> str:
    """Convert MIDI key signature (sharps_flats, minor) to key name."""
    if minor:
        return _KEY_SIG_MAP.get(sharps_flats, 'C')
    return _KEY_SIG_MAJOR.get(sharps_flats, 'C')


class FastMIDIToREMI:
    """Fast MIDI → REMI converter using mido (no music21 dependency).

    Produces the same REMI event format as the original MIDIToREMI:
    list of (event_type, value) tuples ready for tokenization.
    """

    def __init__(self, grid_size: int = 16, velocity_levels: int = 8):
        self.grid_size = grid_size
        self.velocity_levels = velocity_levels
        self.quarter_per_position = 4.0 / grid_size

        # Token type constants (matching REMITokenizer)
        self.BAR = '<Bar>'
        self.POSITION = '<Position'
        self.PROGRAM = '<Program'
        self.NOTE_ON = '<Note_ON'
        self.VELOCITY = '<Velocity'
        self.DURATION = '<Duration'
        self.TEMPO = '<Tempo'
        self.TIMESIG = '<TimeSig'
        self.KEY = '<Key'
        self.ANTICIPATE = '<Anticipate'
        self.BEAT = '<Beat'
        self.BASS = '<Bass'
        self.GRACE_NOTE = '<GraceNote'
        self.REST = '<Rest>'
        self.PEDAL = '<Pedal'
        self.BOS = '<BOS>'
        self.EOS = '<EOS>'

    def convert(self, file_path: str, collect_metadata: bool = False
                ) -> Tuple[List[int], dict]:
        """Parse MIDI file and convert to REMI token IDs.

        Returns:
            (token_ids, metadata)
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return ([], {}) if collect_metadata else []

        try:
            mid = mido.MidiFile(file_path)
        except Exception as e:
            logger.error(f"Failed to parse MIDI {file_path}: {e}")
            return ([], {}) if collect_metadata else []

        events = self._mid_to_events(mid)
        if not events:
            if collect_metadata:
                return [], {}
            return []

        full_events = [(self.BOS, None)] + events + [(self.EOS, None)]

        # Tokenize events
        # We need the REMITokenizer for this, so import here
        from .tokenizer import REMITokenizer
        tokenizer = REMITokenizer(self.grid_size, self.velocity_levels)
        token_ids = tokenizer.tokenize(full_events)

        if collect_metadata:
            metadata = {
                'grid_size': self.grid_size,
                'velocity_levels': self.velocity_levels,
                'num_events': len(events) - 2,  # minus BOS/EOS
                'num_measures': sum(1 for e in events if e[0] == self.BAR),
                'num_notes': sum(1 for e in events if e[0] == self.NOTE_ON),
                'source_format': 'midi_fast',
            }
            return token_ids, metadata
        return token_ids

    def _mid_to_events(self, mid: mido.MidiFile) -> List[Tuple[str, Optional[Any]]]:
        """Convert mido MidiFile → REMI event list."""
        ticks_per_beat = mid.ticks_per_beat
        if ticks_per_beat <= 0:
            return []

        # ── 1. Merge all tracks into a single event stream ──────────
        # Collect (absolute_tick, channel, track_idx, msg) events
        raw_events: List[Tuple[int, int, int, mido.Message]] = []
        for track_idx, track in enumerate(mid.tracks):
            abs_tick = 0
            for msg in track:
                abs_tick += msg.time
                if msg.type == 'note_on' or msg.type == 'note_off':
                    raw_events.append((abs_tick, msg.channel if hasattr(msg, 'channel') else 0,
                                       track_idx, msg))
                elif msg.type in ('program_change',):
                    raw_events.append((abs_tick, msg.channel if hasattr(msg, 'channel') else 0,
                                       track_idx, msg))
                elif msg.type == 'set_tempo':
                    raw_events.append((abs_tick, 0, track_idx, msg))
                elif msg.type == 'time_signature':
                    raw_events.append((abs_tick, 0, track_idx, msg))
                elif msg.type == 'key_signature':
                    raw_events.append((abs_tick, 0, track_idx, msg))
                elif msg.type == 'control_change' and hasattr(msg, 'control') and msg.control == 64:
                    raw_events.append((abs_tick, msg.channel if hasattr(msg, 'channel') else 0,
                                       track_idx, msg))

        if not raw_events:
            return []

        # Sort by tick, then channel, then track
        raw_events.sort(key=lambda x: (x[0], x[1], x[2]))

        # ── 2. Build tempo + time signature maps ────────────────────
        # Default tempo: 500000 μs/beat = 120 BPM
        tempo_map: List[Tuple[int, float]] = [(0, 500000.0)]
        time_sig_map: List[Tuple[int, int, int]] = []  # (tick, numerator, denominator)
        key_sig_map: List[Tuple[int, int, bool]] = []  # (tick, sharps_flats, minor)
        program_map: Dict[int, int] = {}  # channel → program
        pedal_events: List[Tuple[int, bool]] = []  # (tick, is_down)

        for tick, channel, track_idx, msg in raw_events:
            if msg.type == 'set_tempo':
                tempo_map.append((tick, msg.tempo))
            elif msg.type == 'time_signature':
                time_sig_map.append((tick, msg.numerator, msg.denominator))
            elif msg.type == 'key_signature':
                # mido key_signature: key='C', 'Am', etc. -> convert to our format
                key_str = msg.key
                minor = key_str.endswith('m')
                root = key_str[:-1] if minor else key_str
                # Estimate sharps_flats from key name (reverse lookup)
                sf = 0
                for k, v in _KEY_PC_MAP.items():
                    if k == root:
                        # approximate - just store the key name for later
                        pass
                key_sig_map.append((tick, key_str, minor))
            elif msg.type == 'program_change':
                program_map[channel] = msg.program
            elif msg.type == 'control_change' and msg.control == 64:
                pedal_events.append((tick, msg.value >= 64))

        # Sort tempo/time_sig/key maps by tick
        tempo_map.sort(key=lambda x: x[0])
        time_sig_map.sort(key=lambda x: x[0])
        key_sig_map.sort(key=lambda x: x[0])

        # ── 3. Build tick → time (quarter note) conversion ──────────
        # Using tempo map to compute cumulative time
        def _tick_to_quarter(tick: int) -> float:
            """Convert absolute tick to quarter note position.

            At 4/4 time, 1 beat = 1 quarter note.
            quarter_notes = ticks / ticks_per_beat
            Tempo doesn't affect quarter note position (it affects absolute time).
            """
            return tick / ticks_per_beat

        # ── 4. Convert to quarter note positions and build notes ────
        # Track note_on/note_off pairs
        pending_notes: Dict[Tuple[int, int, int], Tuple[int, int, float]] = {}
        # (channel, pitch, track) → (tick_on, velocity, start_quarter)
        notes: List[Tuple[float, int, int, int, int, float]] = []
        # (start_quarter, channel, pitch, velocity, duration_quarter, track)

        meta_at_beat: Dict[int, List[Tuple[str, Any]]] = {}  # quarter position → meta events
        programs_used: Dict[int, int] = {}  # channel → program
        key_signatures: Dict[float, str] = {}

        for tick, channel, track_idx, msg in raw_events:
            qn = _tick_to_quarter(tick)

            if msg.type == 'note_on' and msg.velocity > 0:
                key = (channel, msg.note, track_idx)
                pending_notes[key] = (tick, msg.velocity, qn)
                programs_used[channel] = programs_used.get(channel, 0)

            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                key = (channel, msg.note, track_idx)
                if key in pending_notes:
                    tick_on, vel, start_qn = pending_notes.pop(key)
                    dur_qn = qn - start_qn
                    if dur_qn > 0:
                        notes.append((start_qn, channel, msg.note, vel, dur_qn, track_idx))

        # Handle remaining hanging notes (use a default duration)
        for key, (tick_on, vel, start_qn) in pending_notes.items():
            channel, pitch, track_idx = key
            dur_qn = 0.25  # default: 1/16 note
            notes.append((start_qn, channel, pitch, vel, dur_qn, track_idx))

        if not notes:
            return []

        notes.sort(key=lambda x: x[0])

        # ── 5. Build measure boundaries ────────────────────────────
        # Calculate measure boundaries based on time signatures
        beats_per_bar: List[Tuple[float, float]] = [(0, 4.0)]  # (start_quarter, quarter_notes_per_bar)
        ts_idx = 0
        while ts_idx < len(time_sig_map):
            tick = time_sig_map[ts_idx][0]
            num = time_sig_map[ts_idx][1]
            den = time_sig_map[ts_idx][2]
            qn_per_bar = 4.0 * num / den
            start_qn = _tick_to_quarter(tick)
            beats_per_bar.append((start_qn, qn_per_bar))
            ts_idx += 1
        beats_per_bar.sort(key=lambda x: x[0])

        def _get_measure(qn: float) -> Tuple[int, float]:
            """Get (measure_number, position_in_measure_in_quarter_notes) for a quarter note time."""
            total_qn = 0.0
            meas = 0
            bb_idx = 0
            cur_qn_per_bar = beats_per_bar[0][1]
            eps = 1e-9

            while True:
                bar_end = total_qn + cur_qn_per_bar
                if qn < bar_end - eps:
                    return meas, max(0.0, qn - total_qn)
                if qn <= bar_end + eps:
                    # qn at bar boundary → next measure, pos 0
                    for next_idx in range(bb_idx + 1, len(beats_per_bar)):
                        if abs(beats_per_bar[next_idx][0] - bar_end) < 0.01 or beats_per_bar[next_idx][0] <= bar_end + 1e-6:
                            cur_qn_per_bar = beats_per_bar[next_idx][1]
                            bb_idx = next_idx
                            break
                    return meas + 1, 0.0
                meas += 1
                total_qn = bar_end
                for next_idx in range(bb_idx + 1, len(beats_per_bar)):
                    if abs(beats_per_bar[next_idx][0] - bar_end) < 0.01 or beats_per_bar[next_idx][0] <= bar_end:
                        cur_qn_per_bar = beats_per_bar[next_idx][1]
                        bb_idx = next_idx
                        break

        # Get tempo and time sig at a quarter note time
        def _get_tempo_bpm(qn: float) -> Optional[int]:
            bpm = None
            for tempo_tick, tempo_us in reversed(tempo_map):
                tempo_qn = _tick_to_quarter(tempo_tick)
                if tempo_qn <= qn:
                    bpm = int(round(60000000.0 / tempo_us))
                    break
            if bpm is None:
                bpm = 120
            bpm = max(30, min(240, bpm))
            return (bpm // 10) * 10

        def _get_time_sig(qn: float) -> Optional[Tuple[int, int]]:
            ts = None
            for ts_tick, num, den in reversed(time_sig_map):
                ts_qn = _tick_to_quarter(ts_tick)
                if ts_qn <= qn:
                    ts = (num, den)
                    break
            return ts

        def _get_key_name(qn: float) -> Optional[str]:
            key = None
            for ks_tick, ks_str, minor in reversed(key_sig_map):
                ks_qn = _tick_to_quarter(ks_tick)
                if ks_qn <= qn:
                    key = ks_str
                    break
            return key

        # ── 6. Group notes into measures ────────────────────────────────
        program_counts: Dict[int, int] = {}
        channel_program_map: Dict[int, Tuple[int, int]] = {}

        used_channels = set(ch for _, ch, _, _, _, _ in notes)
        for ch in used_channels:
            prog = program_map.get(ch, 0)
            if ch == _DRUM_CHANNEL or prog >= 112:  # Drum channel / drum programs
                continue
            sub_cnt = program_counts.get(prog, 0)
            sub = sub_cnt if sub_cnt < 4 else 0
            channel_program_map[ch] = (prog, sub)
            program_counts[prog] = sub_cnt + 1

        if not channel_program_map:
            return []

        first_qn = notes[0][0]
        master_ts = _get_time_sig(first_qn) or (4, 4)
        master_num, master_den = master_ts
        first_key = _get_key_name(first_qn)
        if first_key is None:
            global _NO_KEY_WARNED
            if not _NO_KEY_WARNED:
                logger.info("MIDI 无调号信息，默认 C 大调")
                _NO_KEY_WARNED = True

        measure_notes: Dict[int, List] = {}
        measure_tempos: Dict[int, int] = {}
        measure_ts: Dict[int, Tuple[int, int]] = {}

        for start_qn, channel, pitch, vel, dur_qn, track_idx in notes:
            meas, pos_qn = _get_measure(start_qn)
            if meas not in measure_notes:
                measure_notes[meas] = []
                bpm = _get_tempo_bpm(start_qn)
                if bpm:
                    measure_tempos[meas] = bpm
                ts_info = _get_time_sig(start_qn)
                if ts_info:
                    measure_ts[meas] = ts_info

            pos = min(self.grid_size - 1,
                      int(round(pos_qn / self.quarter_per_position)))
            vel_level = min(self.velocity_levels - 1,
                            vel // (128 // self.velocity_levels))
            dur_positions = max(1, min(self.grid_size,
                                       int(round(dur_qn / self.quarter_per_position))))

            GRACE_DURATION_THRESHOLD_QN = 0.25 * self.quarter_per_position
            is_grace = dur_qn < GRACE_DURATION_THRESHOLD_QN

            if channel in channel_program_map:
                prog, sub = channel_program_map[channel]
                measure_notes[meas].append((pos, prog, sub, pitch, vel_level, dur_positions, is_grace))

        # ── 7. Precompute measure start quarter positions ────────────────
        all_measures = sorted(set(list(measure_notes.keys()) +
                                  list(measure_tempos.keys()) +
                                  list(measure_ts.keys())))
        if not all_measures:
            return []

        def _build_measure_starts(max_measure: int) -> List[float]:
            starts = [0.0]
            cur_bar_len = beats_per_bar[0][1]
            bb_idx = 0
            for m in range(max_measure):
                next_start = starts[-1] + cur_bar_len
                for next_idx in range(bb_idx + 1, len(beats_per_bar)):
                    ts_qn = beats_per_bar[next_idx][0]
                    if abs(ts_qn - next_start) < 0.01 or ts_qn <= next_start + 0.001:
                        cur_bar_len = beats_per_bar[next_idx][1]
                        bb_idx = next_idx
                        break
                starts.append(next_start)
            return starts

        measure_starts = _build_measure_starts(all_measures[-1])

        # ── 8. Precompute beat positions per measure ────────────────────
        measure_beats: Dict[int, List[Tuple[int, int]]] = {}
        for m in all_measures:
            ts = _get_time_sig(measure_starts[m])
            if ts:
                num, den = ts
            else:
                num, den = master_num, master_den
            beat_interval_qn = 4.0 / den
            beat_spacing = max(1, int(round(beat_interval_qn / self.quarter_per_position)))
            beats = []
            for beat_num in range(num):
                bp = beat_num * beat_spacing
                if bp < self.grid_size and beat_num < 16:
                    beats.append((bp, beat_num + 1))
            measure_beats[m] = beats

        # ── 9. Pre-scan bass notes ──────────────────────────────────────
        pos_bass: Dict[Tuple[int, int], int] = {}
        for m, notes_in_m in measure_notes.items():
            for pos, prog, sub, pitch, vel_level, dur_pos, *_ in notes_in_m:
                key = (m, pos)
                if key not in pos_bass or pitch < pos_bass[key]:
                    pos_bass[key] = pitch

        # ── 10. Convert pedal events to (measure, pos) ──────────────────
        pedal_at_pos: Dict[Tuple[int, int], List[str]] = {}
        for tick, is_down in pedal_events:
            qn = _tick_to_quarter(tick)
            meas, pos_qn = _get_measure(qn)
            pos = min(self.grid_size - 1, int(round(pos_qn / self.quarter_per_position)))
            action = 'start' if is_down else 'end'
            pedal_at_pos.setdefault((meas, pos), []).append(action)

        # Also include pedal measures in all_measures
        if pedal_at_pos:
            for m, _ in pedal_at_pos:
                if m not in all_measures:
                    all_measures.append(m)
            all_measures.sort()
        if all_measures[-1] > len(measure_starts) - 1:
            measure_starts = _build_measure_starts(all_measures[-1])

        # ── 11. Build event sequence ────────────────────────────────────
        events: List[Tuple[str, Optional[Any]]] = []
        initial_key = first_key or 'C'
        last_key_name: Optional[str] = None

        for m in all_measures:
            key_at_m = _get_key_name(measure_starts[m])

            # ANTICIPATE: emit BEFORE BAR if key changes in the next measure
            if m > 0 and key_at_m and key_at_m != last_key_name:
                events.append((self.ANTICIPATE, key_at_m))

            events.append((self.BAR, None))

            # Key signature (after BAR)
            if m == 0:
                events.append((self.KEY, initial_key))
                last_key_name = initial_key
            elif key_at_m and key_at_m != last_key_name:
                events.append((self.KEY, key_at_m))
                last_key_name = key_at_m

            # Tempo
            bpm = measure_tempos.get(m) or _get_tempo_bpm(measure_starts[m])
            if bpm and (not events[-1][0] == self.TEMPO or events[-1][1] != bpm):
                events.append((self.TEMPO, bpm))

            # Time signature
            ts = measure_ts.get(m) or _get_time_sig(measure_starts[m])
            if ts:
                ts_str = f'{ts[0]}/{ts[1]}'
                if ts_str in REMI_TIME_SIGS:
                    events.append((self.TIMESIG, ts_str))

            notes_in_m = measure_notes.get(m, [])
            beats_in_m = measure_beats.get(m, [])

            # Group notes by position
            pos_notes: Dict[int, List] = {}
            for pos, prog, sub, pitch, vel_level, dur_pos, is_grace in notes_in_m:
                if pos not in pos_notes:
                    pos_notes[pos] = []
                pos_notes[pos].append((prog, sub, pitch, vel_level, dur_pos, is_grace))

            for pos in pos_notes:
                pos_notes[pos].sort(key=lambda x: (x[0], x[1]))

            all_positions = sorted(set(list(pos_notes.keys()) +
                                       [b[0] for b in beats_in_m]))

            cur_prog = -1
            cur_sub = 0

            for pos in all_positions:
                events.append((self.POSITION, pos))
                cur_prog = -1
                cur_sub = 0

                # Beat token
                for bp, beat_num in beats_in_m:
                    if bp == pos:
                        events.append((self.BEAT, beat_num))
                        break

                # Rest token: beat position with no notes from any channel
                if not pos_notes.get(pos):
                    events.append((self.REST, None))

                # Sustained pedal events at this position
                for action in pedal_at_pos.get((m, pos), []):
                    events.append((self.PEDAL, action))

                # Bass token
                bass_pc = pos_bass.get((m, pos))
                if bass_pc is not None:
                    events.append((self.BASS, bass_pc % 12))

                # Notes at this position
                for prog, sub, pitch, vel_level, dur_pos, is_grace in pos_notes.get(pos, []):
                    if prog != cur_prog or sub != cur_sub:
                        if sub == 0:
                            events.append((self.PROGRAM, str(prog)))
                        else:
                            events.append((self.PROGRAM, f'{prog}_{sub}'))
                        cur_prog = prog
                        cur_sub = sub

                    tonic = key_name_to_tonic_midi(initial_key)
                    interval = max(-60, min(60, pitch - tonic))

                    if is_grace:
                        events.append((self.GRACE_NOTE, 'grace'))
                        events.append((self.NOTE_ON, interval))
                    else:
                        events.append((self.NOTE_ON, interval))
                        events.append((self.VELOCITY, vel_level))
                        events.append((self.DURATION, dur_pos))

        return events


# ── Fast file processing helper ──────────────────────────────

@dataclass
class FastMusicMetadata:
    """Minimal metadata for fast MIDI processing."""
    file_id: str
    file_path: str
    composer: str
    title: str
    genre: str
    duration_seconds: float
    num_measures: int
    num_notes: int
    num_tokens: int
    time_signature: str
    key_signature: str
    tempo: Optional[float]
    instruments: List[str]
    has_chords: bool
    has_polyphony: bool
    processing_time: float
    hash_md5: str


def compute_file_hash(file_path: str) -> str:
    """Compute MD5 hash of a file. Uses .hash sidecar as cache if available.

    If a .hash sidecar file exists next to the source file, reads it instead
    of re-hashing the content. Otherwise computes the hash and writes the sidecar
    for future use.
    """
    sidecar = file_path + '.hash'
    if os.path.exists(sidecar):
        try:
            with open(sidecar) as f:
                cached = f.read().strip()
                if cached and len(cached) == 32:
                    return cached
        except (OSError, ValueError):
            pass

    h = _compute_file_hash_raw(file_path)

    try:
        with open(sidecar, 'w') as f:
            f.write(h)
    except OSError:
        pass

    return h


def _compute_file_hash_raw(file_path: str) -> str:
    """Compute MD5 hash from file content (no sidecar, always reads file)."""
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_file_id(file_path: str) -> str:
    stem = Path(file_path).stem
    # Truncate stem to avoid excessively long filenames
    if len(stem) > 60:
        stem = stem[:60]
    h = compute_file_hash(file_path)[:8]
    u = uuid.uuid4().hex[:8]
    return f"{stem}_{h}_{u}"


def process_midi_file_fast(file_path: str, output_dir: str,
                           grid_size: int = 16, velocity_levels: int = 8,
                           min_notes: int = 10, max_notes: int = 50000,
                           min_tokens: int = 50, max_tokens: int = 16384,
                           min_size_kb: int = 1, max_size_mb: int = 50) -> Optional[Dict]:
    """Fast single MIDI file processing.

    Returns result dict or None if file fails checks.
    """
    # Quick size check (avoid parsing tiny/huge files)
    try:
        fsize = os.path.getsize(file_path) / 1024
    except OSError:
        return None
    if fsize < min_size_kb or fsize > max_size_mb * 1024:
        return None

    t0 = time.time()

    # Parse with mido
    try:
        mid = mido.MidiFile(file_path)
    except Exception:
        return None

    # Quick note count check (fast scan without full conversion)
    note_on_count = 0
    ticks_per_beat = mid.ticks_per_beat
    has_tempo = False
    total_ticks = 0
    program_set = set()
    drum_ch = False

    for track in mid.tracks:
        for msg in track:
            total_ticks += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                note_on_count += 1
            if msg.type == 'set_tempo':
                has_tempo = True
            if msg.type == 'program_change':
                program_set.add(msg.program)
            if hasattr(msg, 'channel') and msg.channel == 9:
                drum_ch = True

    if note_on_count < min_notes or note_on_count > max_notes:
        return None

    # Convert
    converter = FastMIDIToREMI(grid_size, velocity_levels)
    try:
        token_ids, metadata = converter.convert(file_path, collect_metadata=True)
    except Exception:
        return None

    if not token_ids or len(token_ids) < min_tokens + 2 or len(token_ids) > max_tokens:
        return None

    # Generate file paths
    fid = generate_file_id(file_path)
    token_path = os.path.join(output_dir, "tokens", f"{fid}.tokens")
    meta_path = os.path.join(output_dir, "metadata", f"{fid}.meta.json")
    os.makedirs(os.path.dirname(token_path), exist_ok=True)
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)

    # Duration estimate
    dur_seconds = total_ticks / (ticks_per_beat * (120.0 / 60)) if ticks_per_beat > 0 else 0

    meta = {
        'file_id': fid,
        'file_path': file_path,
        'num_tokens': len(token_ids),
        'num_notes': note_on_count,
        'num_measures': metadata.get('num_measures', 0),
        'duration_seconds': dur_seconds,
        'processing_time': time.time() - t0,
        'hash_md5': compute_file_hash(file_path),
        'has_tempo': has_tempo,
        'has_drum': drum_ch,
        'programs': list(program_set),
    }

    # Write token file
    with open(token_path, 'w', encoding='utf-8') as f:
        json.dump(token_ids, f)

    # Write metadata
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    return {
        'file_id': fid,
        'original_path': file_path,
        'token_path': token_path,
        'metadata_path': meta_path,
        'num_tokens': len(token_ids),
    }
