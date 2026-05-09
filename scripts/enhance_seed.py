"""
给 seed_4tracks_4bars.musicxml 添加力度、踏板、演奏法、连奏线记号。
所有添加的记号均被 REMITokenizer 支持。
"""
from pathlib import Path
from music21 import (
    converter, dynamics, expressions, articulations, spanner,
    note, chord
)

SEED = Path('data/test_seeds/seed_4tracks_4bars.musicxml')

score = converter.parse(str(SEED))
parts = score.parts

# ============================================================
# Part 0: Piano Right
# ============================================================
print("Part 0: Piano Right ...")
measures = list(parts[0].getElementsByClass('Measure'))

# Dynamics: mf → cresc → f → dim → mp → p
measures[0].insert(0, dynamics.Dynamic('mf'))
measures[1].insert(0, dynamics.Crescendo())
dur = measures[1].duration.quarterLength
measures[1].insert(dur - 1.0, dynamics.Dynamic('f'))
measures[2].insert(0, dynamics.Diminuendo())
dur = measures[2].duration.quarterLength
measures[2].insert(dur - 1.0, dynamics.Dynamic('mp'))
dur = measures[3].duration.quarterLength
measures[3].insert(dur - 1.0, dynamics.Dynamic('p'))

# Pedal: sustain each measure
for m in measures:
    m.insert(0, expressions.PedalMark('start'))
    dur = m.duration.quarterLength
    m.insert(dur - 0.5, expressions.PedalMark('stop'))

# Articulations: staccato on eighth-note scale runs
for m in measures:
    for n in m.notes:
        if isinstance(n, note.Note) and n.duration.quarterLength <= 0.5:
            n.articulations.append(articulations.Staccato())

# Slurs: slur each group of consecutive eighth notes
for m in measures:
    all_notes = list(m.notes)
    groups = []
    cur = []
    for n in all_notes:
        is_eighth = (isinstance(n, note.Note) and n.duration.quarterLength <= 0.5)
        if is_eighth:
            cur.append(n)
        else:
            if len(cur) >= 3:
                groups.append(cur)
            cur = []
    if len(cur) >= 3:
        groups.append(cur)

    for g in groups:
        slur = spanner.Slur()
        for sn in g:
            slur.addSpannedElements([sn])
        # Insert into measure so flatten() picks it up
        m.insert(0, slur)

# ============================================================
# Part 1: Piano Left (bass)
# ============================================================
print("Part 1: Piano Left ...")
measures = list(parts[1].getElementsByClass('Measure'))
for m in measures:
    m.insert(0, dynamics.Dynamic('p'))
    m.insert(0, expressions.PedalMark('start'))
    dur = m.duration.quarterLength
    m.insert(dur - 0.5, expressions.PedalMark('stop'))

# ============================================================
# Part 2: Violin I
# ============================================================
print("Part 2: Violin I ...")
measures = list(parts[2].getElementsByClass('Measure'))
measures[0].insert(0, dynamics.Dynamic('mp'))
measures[1].insert(0, dynamics.Crescendo())
dur = measures[1].duration.quarterLength
measures[1].insert(dur - 1.0, dynamics.Dynamic('mf'))
measures[2].insert(0, dynamics.Diminuendo())
dur = measures[2].duration.quarterLength
measures[2].insert(dur - 1.0, dynamics.Dynamic('p'))

# Tenuto on whole notes (violin long tones)
for m in measures:
    for n in m.notes:
        if isinstance(n, note.Note) and n.duration.quarterLength >= 3.0:
            n.articulations.append(articulations.Tenuto())

# ============================================================
# Part 3: Violin II
# ============================================================
print("Part 3: Violin II ...")
measures = list(parts[3].getElementsByClass('Measure'))
measures[0].insert(0, dynamics.Dynamic('mp'))
dur = measures[3].duration.quarterLength
measures[3].insert(dur - 1.0, dynamics.Dynamic('pp'))
for m in measures:
    for n in m.notes:
        if isinstance(n, note.Note) and n.duration.quarterLength >= 3.0:
            n.articulations.append(articulations.Tenuto())

# ============================================================
# Write
# ============================================================
score.write('musicxml', fp=str(SEED))
print(f"Done → {SEED}")
