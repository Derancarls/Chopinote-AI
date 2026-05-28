#!/usr/bin/env python3
"""用 music21 生成 4 小节钢琴种子（C大调 I-IV-V-I）。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from music21 import stream, note, tempo, meter, key, instrument, clef

s = stream.Score()

# ── 配置 ───────────────────────────────────────────────────────
bpm = 100
k = key.Key('C')
ts = meter.TimeSignature('4/4')

# ── 右手：旋律 ────────────────────────────────────────────────
rh = stream.Part()
rh.insert(0, instrument.Piano())
rh.insert(0, clef.TrebleClef())
rh.insert(0, k)
rh.insert(0, tempo.MetronomeMark(bpm))
rh.append(ts)

mel = [
    ['E4', 'G4', 'C5', 'E5', 'D5', 'C5', 'B4', 'C5'],
    ['F4', 'A4', 'C5', 'F5', 'E5', 'D5', 'C5', 'B4'],
    ['G4', 'B4', 'D5', 'G5', 'F5', 'E5', 'D5', 'C5'],
    ['C4', 'E4', 'G4', 'C5', 'C5', 'B4', 'C5', 'E5'],
]
for bar_notes in mel:
    m = stream.Measure()
    for pn in bar_notes:
        n = note.Note(pn, quarterLength=0.5)
        n.volume.velocity = 90
        m.append(n)
    rh.append(m)

# ── 左手：伴奏 ────────────────────────────────────────────────
lh = stream.Part()
lh.insert(0, instrument.Piano())
lh.insert(0, clef.BassClef())
lh.insert(0, k)
lh.append(ts)

acc = [
    ['C2', 'C3', 'E3', 'G3', 'C2', 'C3', 'E3', 'G3'],
    ['F2', 'F3', 'A3', 'C4', 'F2', 'F3', 'A3', 'C4'],
    ['G2', 'G3', 'B3', 'D4', 'G2', 'G3', 'B3', 'D4'],
    ['C2', 'C3', 'E3', 'G3', 'C2', 'C3', 'E3', 'G3'],
]
for bar_notes in acc:
    m = stream.Measure()
    for pn in bar_notes:
        n = note.Note(pn, quarterLength=0.5)
        n.volume.velocity = 80
        m.append(n)
    lh.append(m)

s.append(rh)
s.append(lh)

out = '/tmp/seed_4bars.musicxml'
s.write('musicxml', fp=out)

from music21 import converter
c = converter.parse(out)
parts = c.parts
print(f'种子已生成: {out}')
print(f'声部数: {len(parts)}')
for i, p in enumerate(parts):
    measures = p.getElementsByClass(stream.Measure)
    print(f'  声部 {i+1}: {len(measures)} 小节')
