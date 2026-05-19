"""对齐测试：PDMX/MIDI 转换器中与 MusicXML 不一致的修复点。"""
import json
import os
import tempfile

import pytest

from chopinote_dataset.converter import PDMXToREMI
from chopinote_dataset.fast_converter import FastMIDIToREMI
from chopinote_dataset.tokenizer import REMITokenizer


# ── 辅助函数 ──────────────────────────────────────────────────

def _make_pdmx_with_rests() -> dict:
    """创建一个含 2 小节 4/4 拍的 PDMX dict，第二拍无音符应补 rest。"""
    return {
        'resolution': 480,
        'barlines': [
            {'measure': 1, 'time': 0},
            {'measure': 2, 'time': 1920},  # 4 beats * 480 ticks
            {'measure': 3, 'time': 3840},
        ],
        'time_signatures': [
            {'measure': 1, 'numerator': 4, 'denominator': 4},
        ],
        'tempos': [
            {'time': 0, 'qpm': 120},
        ],
        'tracks': [
            {
                'program': 0,
                'notes': [
                    # measure 1, beat 1 only — beats 2-4 should get rests
                    {'measure': 1, 'time': 0, 'pitch': 60, 'velocity': 80, 'duration': 480},
                ],
                'annotations': [],
            },
        ],
    }


def _make_pdmx_with_grace() -> dict:
    """创建一个含倚音的 PDMX dict。"""
    return {
        'resolution': 480,
        'barlines': [
            {'measure': 1, 'time': 0},
            {'measure': 2, 'time': 1920},
        ],
        'time_signatures': [
            {'measure': 1, 'numerator': 4, 'denominator': 4},
        ],
        'tempos': [
            {'time': 0, 'qpm': 120},
        ],
        'key_signatures': [
            {'measure': 1, 'root_str': 'C', 'mode': 'major'},
        ],
        'tracks': [
            {
                'program': 0,
                'notes': [
                    {'measure': 1, 'time': 0, 'pitch': 60, 'velocity': 80,
                     'duration': 480, 'is_grace': False},
                    # Grace note
                    {'measure': 1, 'time': 0, 'pitch': 62, 'velocity': 70,
                     'duration': 30, 'is_grace': True},
                ],
                'annotations': [],
            },
        ],
    }


def _create_small_midi_with_grace(tmpdir) -> str:
    """创建含倚音的 MIDI 文件，返回路径。"""
    import mido
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    track.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))
    track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, time=0))
    track.append(mido.Message('program_change', program=0, channel=0, time=0))

    # Regular note C4
    track.append(mido.Message('note_on', note=60, velocity=80, channel=0, time=0))
    track.append(mido.Message('note_off', note=60, velocity=0, channel=0, time=480))

    # Grace note (very short): D4
    track.append(mido.Message('note_on', note=62, velocity=70, channel=0, time=0))
    track.append(mido.Message('note_off', note=62, velocity=0, channel=0, time=10))

    path = os.path.join(tmpdir, 'test_grace.mid')
    mid.save(path)
    return path


def _create_small_midi_with_pedal(tmpdir) -> str:
    """创建含延音踏板的 MIDI 文件，返回路径。"""
    import mido
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    track.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))
    track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, time=0))
    track.append(mido.Message('program_change', program=0, channel=0, time=0))

    # Note + pedal
    track.append(mido.Message('control_change', control=64, value=127, channel=0, time=0))
    track.append(mido.Message('note_on', note=60, velocity=80, channel=0, time=0))
    track.append(mido.Message('note_off', note=60, velocity=0, channel=0, time=480))
    track.append(mido.Message('control_change', control=64, value=0, channel=0, time=0))

    path = os.path.join(tmpdir, 'test_pedal.mid')
    mid.save(path)
    return path


# ── 测试: PDMX Rest 有 Duration ────────────────────────────

class TestPDMXRestHasDuration:
    """PDMX 自动生成的 Rest 必须有 Duration token。"""

    def _get_events(self, pdmx_data: dict):
        """内部辅助：将 PDMX dict 转换为事件列表。"""
        converter = PDMXToREMI(16, 8)
        tokenizer = REMITokenizer(16, 8)
        token_ids = converter.convert_pdmx(pdmx_data)
        if not token_ids:
            return []
        return tokenizer.detokenize(token_ids)

    def test_rest_event_exists(self):
        """PDMX rest 生成后有 <Rest> 事件。"""
        pdmx = _make_pdmx_with_rests()
        events = self._get_events(pdmx)
        rest_events = [e for e in events if e[0] == '<Rest>']
        assert len(rest_events) >= 1, (
            f'需要至少 1 个 Rest 事件（静拍），实际 {len(rest_events)}')

    def test_rest_has_duration(self):
        """每个 Rest 事件后应紧跟 <Duration> token。"""
        pdmx = _make_pdmx_with_rests()
        events = self._get_events(pdmx)
        for i, (ttype, _) in enumerate(events):
            if ttype == '<Rest>':
                assert i + 1 < len(events), 'Rest 后应有 Duration'
                assert events[i + 1][0] == '<Duration', (
                    f'Rest 后应为 Duration, 实际 {events[i + 1][0]}')

    def test_rest_duration_value(self):
        """4/4 拍下拍长 = grid/4 = 4 个 position。"""
        pdmx = _make_pdmx_with_rests()
        events = self._get_events(pdmx)
        for i, (ttype, _) in enumerate(events):
            if ttype == '<Rest>':
                if i + 1 < len(events) and events[i + 1][0] == '<Duration':
                    dur = events[i + 1][1]
                    assert dur == 4, f'4/4 拍 rest duration 应为 4, 实际 {dur}'


# ── 测试: PDMX Grace Note ───────────────────────────────────

class TestPDMXGraceNote:
    """倚音应有完整 VELOCITY + DURATION 子 token。"""

    def _get_events(self, pdmx_data: dict):
        converter = PDMXToREMI(16, 8)
        tokenizer = REMITokenizer(16, 8)
        token_ids = converter.convert_pdmx(pdmx_data)
        return tokenizer.detokenize(token_ids) if token_ids else []

    def test_grace_has_velocity_and_duration(self):
        """倚音后应有 VELOCITY 和 DURATION。"""
        events = self._get_events(_make_pdmx_with_grace())
        for i, (ttype, _) in enumerate(events):
            if ttype == '<GraceNote':
                # GraceNote → NOTE_ON → VELOCITY → DURATION
                assert i + 3 < len(events), '倚音序列不完整'
                assert events[i + 1][0] == '<Note_ON', (
                    f'GraceNote 后应为 Note_ON，实际 {events[i + 1][0]}')
                assert events[i + 2][0] == '<Velocity', (
                    f'Note_ON 后应为 Velocity，实际 {events[i + 2][0]}')
                assert events[i + 3][0] == '<Duration', (
                    f'Velocity 后应为 Duration，实际 {events[i + 3][0]}')


# ── 测试: FastMIDI Grace Note 补齐 ─────────────────────────

class TestFastMIDIGraceNote:
    """FastMIDI 倚音必须有 VELOCITY + DURATION。"""

    def test_grace_has_velocity_and_duration(self, tmpdir):
        path = _create_small_midi_with_grace(str(tmpdir))
        converter = FastMIDIToREMI(16, 8)
        tokenizer = REMITokenizer(16, 8)

        token_ids, _ = converter.convert(path, collect_metadata=True)
        assert token_ids, '转换结果不应为空'

        events = tokenizer.detokenize(token_ids)
        grace_found = False
        for i, (ttype, _) in enumerate(events):
            if ttype == '<GraceNote':
                grace_found = True
                assert i + 3 < len(events), '倚音序列不完整'
                # NOTE_ON
                assert events[i + 1][0] == '<Note_ON', (
                    f'GraceNote 后应为 Note_ON，实际 {events[i + 1][0]}')
                # VELOCITY
                assert events[i + 2][0] == '<Velocity', (
                    f'Note_ON 后应为 Velocity，实际 {events[i + 2][0]}')
                # DURATION
                assert events[i + 3][0] == '<Duration', (
                    f'Velocity 后应为 Duration，实际 {events[i + 3][0]}')
                # Grace note duration should be 1
                assert events[i + 3][1] == 1, (
                    f'倚音 duration 应为 1，实际 {events[i + 3][1]}')
                break

        assert grace_found, '未找到倚音事件'

    def test_regular_note_unaffected(self, tmpdir):
        """非倚音音符不应被此修复影响。"""
        path = _create_small_midi_with_grace(str(tmpdir))
        converter = FastMIDIToREMI(16, 8)
        tokenizer = REMITokenizer(16, 8)

        token_ids, _ = converter.convert(path, collect_metadata=True)
        events = tokenizer.detokenize(token_ids)

        for i, (ttype, _) in enumerate(events):
            if ttype == '<Note_ON':
                # If not preceded by GraceNote, should have Velocity+Duration
                if i > 0 and events[i - 1][0] != '<GraceNote':
                    assert i + 2 < len(events)
                    assert events[i + 1][0] == '<Velocity'
                    assert events[i + 2][0] == '<Duration'


# ── 测试: FastMIDI Pedal 格式 ─────────────────────────────

class TestFastMIDIPedal:
    """FastMIDI 踏板事件格式必须与 tokenizer 兼容。"""

    def test_pedal_events_emitted(self, tmpdir):
        path = _create_small_midi_with_pedal(str(tmpdir))
        converter = FastMIDIToREMI(16, 8)
        tokenizer = REMITokenizer(16, 8)

        token_ids, _ = converter.convert(path, collect_metadata=True)
        assert token_ids, '转换结果不应为空'

        events = tokenizer.detokenize(token_ids)
        pedal_events = [e for e in events if e[0] == '<Pedal']
        assert len(pedal_events) >= 2, (
            f'需要至少 2 个 Pedal 事件 (start+end)，实际 {len(pedal_events)}')

    def test_pedal_event_values(self, tmpdir):
        """Pedal 事件的 value 必须为 'start' 或 'end'。"""
        path = _create_small_midi_with_pedal(str(tmpdir))
        converter = FastMIDIToREMI(16, 8)
        tokenizer = REMITokenizer(16, 8)

        token_ids, _ = converter.convert(path, collect_metadata=True)
        events = tokenizer.detokenize(token_ids)

        pedal_values = set()
        for ttype, val in events:
            if ttype == '<Pedal':
                assert val in ('start', 'end'), (
                    f'Pedal value 必须为 start/end，实际 {val}')
                pedal_values.add(val)

        assert 'start' in pedal_values
        assert 'end' in pedal_values


# ── 测试: FastMIDI Rest 有 Duration ─────────────────────────

class TestFastMIDIRestHasDuration:
    """FastMIDI 自动生成的 Rest 必须有 Duration token。"""

    def test_rest_has_duration(self, tmpdir):
        """创建一个仅含 1 个音符的 MIDI，其他位置应为 rest + Duration。"""
        import mido
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)

        track.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))
        track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, time=0))
        track.append(mido.Message('program_change', program=0, channel=0, time=0))
        # Single note on beat 1
        track.append(mido.Message('note_on', note=60, velocity=80, channel=0, time=0))
        track.append(mido.Message('note_off', note=60, velocity=0, channel=0, time=120))

        path = os.path.join(str(tmpdir), 'test_rest.mid')
        mid.save(path)

        converter = FastMIDIToREMI(16, 8)
        tokenizer = REMITokenizer(16, 8)
        token_ids, _ = converter.convert(path, collect_metadata=True)
        assert token_ids, '转换结果不应为空'

        events = tokenizer.detokenize(token_ids)
        for i, (ttype, _) in enumerate(events):
            if ttype == '<Rest>':
                assert i + 1 < len(events), 'Rest 后应有 Duration'
                assert events[i + 1][0] == '<Duration', (
                    f'Rest 后应为 Duration，实际 {events[i + 1][0]}')
