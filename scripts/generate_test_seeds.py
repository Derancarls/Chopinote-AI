"""
生成 4 轨 4 小节测试 Seed: Piano (RH/LH) + Violin I + Violin II

用法:
    python scripts/generate_test_seeds.py
"""
from pathlib import Path
from music21 import stream, note, chord, tempo, key, meter, dynamics, metadata, clef, instrument

INSTRUMENTS = {
    'piano':  0,
    'violin': 40,
}

MIDI_PROG_MAP = {
    'Piano (Right)': INSTRUMENTS['piano'],
    'Piano (Left)':  INSTRUMENTS['piano'],
    'Violin I':  INSTRUMENTS['violin'],
    'Violin II': INSTRUMENTS['violin'],
}


def _make_instrument(prog: int):
    """创建带指定 MIDI program 的乐器对象。"""
    if prog == 0:
        return instrument.Piano()
    inst = instrument.Instrument()
    inst.midiProgram = prog
    return inst


def save(rh, lh, vn1, vn2, name, k, ts='4/4', bpm=100):
    s = stream.Score()
    s.insert(0, metadata.Metadata(title=name))

    for part_data in [
        ('Piano (Right)', clef.TrebleClef(), rh, 'mf', k, ts, bpm, True),
        ('Piano (Left)', clef.BassClef(), lh, 'mp', k, ts, bpm, False),
        ('Violin I', clef.TrebleClef(), vn1, 'mf', k, ts, bpm, False),
        ('Violin II', clef.TrebleClef(), vn2, 'mp', k, ts, bpm, False),
    ]:
        name_part, clf, notes, dyn, kk, tss, bpmm, is_first = part_data
        p = stream.Part()
        p.partName = name_part
        if is_first:
            p.insert(0, key.Key(kk))
            p.insert(0, meter.TimeSignature(tss))
            p.insert(0, tempo.MetronomeMark(bpmm))
        p.insert(0, _make_instrument(MIDI_PROG_MAP[name_part]))
        p.insert(0, clf)
        p.insert(0, dynamics.Dynamic(dyn))
        _fill_measures(p, notes, tss)
        s.append(p)

    path = Path('data/test_seeds') / f'{name}.musicxml'
    path.parent.mkdir(parents=True, exist_ok=True)
    s.write('musicxml', fp=str(path))
    print(f'  [OK] {path}')


def _fill_measures(part, notes, ts='4/4'):
    time_range = 4.0
    m = stream.Measure()
    current = 0.0
    for n in notes:
        if current + n.quarterLength > time_range + 0.01:
            part.append(m)
            m = stream.Measure()
            current = 0.0
        m.append(n)
        current += n.quarterLength
    if len(m) > 0:
        part.append(m)


# ════════════════════ 4 轨 4 小节 ════════════════════
# C 大调，4/4 拍，100 BPM

# 右手钢琴：和弦 + 旋律
rh = [
    chord.Chord(['C4', 'E4', 'G4'], quarterLength=2),
    note.Note(67, quarterLength=0.5), note.Note(69, quarterLength=0.5),
    note.Note(71, quarterLength=0.5), note.Note(72, quarterLength=0.5),
    chord.Chord(['F4', 'A4', 'C5'], quarterLength=2),
    note.Note(74, quarterLength=0.5), note.Note(72, quarterLength=0.5),
    note.Note(71, quarterLength=0.5), note.Note(69, quarterLength=0.5),
    chord.Chord(['G4', 'B4', 'D5'], quarterLength=2),
    note.Note(76, quarterLength=0.5), note.Note(74, quarterLength=0.5),
    note.Note(72, quarterLength=0.5), note.Note(71, quarterLength=0.5),
    chord.Chord(['C4', 'E4', 'G4'], quarterLength=2),
    note.Note(72, quarterLength=0.5), note.Note(71, quarterLength=0.5),
    note.Note(69, quarterLength=0.5), note.Note(67, quarterLength=0.5),
]

# 左手钢琴：八度低音 + 五度
lh = [
    note.Note(36, quarterLength=2), note.Note(48, quarterLength=2),
    note.Note(41, quarterLength=2), note.Note(53, quarterLength=2),
    note.Note(43, quarterLength=2), note.Note(55, quarterLength=2),
    note.Note(36, quarterLength=2), note.Note(48, quarterLength=2),
]

# 小提琴 I：长音旋律线
vn1 = [
    note.Note(67, quarterLength=4),
    note.Note(72, quarterLength=4),
    note.Note(74, quarterLength=4),
    note.Note(67, quarterLength=4),
]

# 小提琴 II：和声长音
vn2 = [
    note.Note(60, quarterLength=4),
    note.Note(65, quarterLength=4),
    note.Note(67, quarterLength=4),
    note.Note(60, quarterLength=4),
]

save(rh, lh, vn1, vn2, 'seed_4tracks_4bars', 'C', bpm=100)
print('生成完成')
