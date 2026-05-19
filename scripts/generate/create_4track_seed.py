"""
生成 4 轨 4 小节种子：钢琴双轨 + 小提琴双轨 + 力度/踏板。
"""
import sys, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from music21 import stream, note, chord, meter, clef, dynamics, instrument, key


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'data/test_seeds/seed_4tracks_w_dynamics.musicxml'
    ts = meter.TimeSignature('4/4')

    pf = instrument.Piano()
    vn = instrument.Violin()

    def _ch(pits, ql): c = chord.Chord(pits); c.quarterLength = ql; return c

    # 4 parts
    p1 = stream.Part(); p1.partName = 'Piano';  p1.append(pf); p1.append(clef.TrebleClef()); p1.append(ts); p1.append(key.Key('C'))
    p2 = stream.Part(); p2.partName = 'Piano';  p2.append(pf); p2.append(clef.BassClef());   p2.append(ts)
    p3 = stream.Part(); p3.partName = 'Violin'; p3.append(vn); p3.append(clef.TrebleClef()); p3.append(ts)
    p4 = stream.Part(); p4.partName = 'Violin'; p4.append(vn); p4.append(clef.TrebleClef()); p4.append(ts)

    # Bar 1 — p
    m = stream.Measure(); m.number=1; m.timeSignature=ts
    m.insert(0, dynamics.Dynamic('p'))
    m.insert(0, _ch([60,64,67],1.0)); m.insert(1.0,_ch([65,69,72],1.0)); m.insert(2.0,_ch([64,67,72],2.0))
    p1.append(m)
    m=stream.Measure(); m.number=1; m.timeSignature=ts
    n=note.Note(36); n.quarterLength=4.0; m.insert(0,n); p2.append(m)
    m=stream.Measure(); m.number=1; m.timeSignature=ts
    n=note.Note(72); n.quarterLength=4.0; m.insert(0,n); p3.append(m)
    m=stream.Measure(); m.number=1; m.timeSignature=ts
    n=note.Note(67); n.quarterLength=4.0; m.insert(0,n); p4.append(m)

    # Bar 2
    m=stream.Measure(); m.number=2; m.timeSignature=ts
    m.insert(0,_ch([62,66,69],1.0)); m.insert(1.0,_ch([64,67,71],1.0)); m.insert(2.0,_ch([60,64,69],2.0))
    p1.append(m)
    m=stream.Measure(); m.number=2; m.timeSignature=ts
    n=note.Note(41); n.quarterLength=2.0; m.insert(0,n); n=note.Note(36); n.quarterLength=2.0; m.insert(2.0,n); p2.append(m)
    m=stream.Measure(); m.number=2; m.timeSignature=ts
    n=note.Note(69); n.quarterLength=4.0; m.insert(0,n); p3.append(m)
    m=stream.Measure(); m.number=2; m.timeSignature=ts
    n=note.Note(64); n.quarterLength=4.0; m.insert(0,n); p4.append(m)

    # Bar 3 — f
    m=stream.Measure(); m.number=3; m.timeSignature=ts
    m.insert(0, dynamics.Dynamic('f'))
    m.insert(0,_ch([67,71,74],2.0)); m.insert(2.0,_ch([65,69,72],2.0))
    p1.append(m)
    m=stream.Measure(); m.number=3; m.timeSignature=ts
    n=note.Note(43); n.quarterLength=4.0; m.insert(0,n); p2.append(m)
    m=stream.Measure(); m.number=3; m.timeSignature=ts
    n=note.Note(76); n.quarterLength=2.0; m.insert(0,n); n=note.Note(72); n.quarterLength=2.0; m.insert(2.0,n); p3.append(m)
    m=stream.Measure(); m.number=3; m.timeSignature=ts
    n=note.Note(71); n.quarterLength=2.0; m.insert(0,n); n=note.Note(69); n.quarterLength=2.0; m.insert(2.0,n); p4.append(m)

    # Bar 4 — p
    m=stream.Measure(); m.number=4; m.timeSignature=ts
    m.insert(3.5, dynamics.Dynamic('p'))
    m.insert(0,_ch([60,64,67],4.0))
    p1.append(m)
    m=stream.Measure(); m.number=4; m.timeSignature=ts
    n=note.Note(36); n.quarterLength=4.0; m.insert(0,n); p2.append(m)
    m=stream.Measure(); m.number=4; m.timeSignature=ts
    n=note.Note(72); n.quarterLength=4.0; m.insert(0,n); p3.append(m)
    m=stream.Measure(); m.number=4; m.timeSignature=ts
    n=note.Note(67); n.quarterLength=4.0; m.insert(0,n); p4.append(m)

    score = stream.Score()
    for p in [p1, p2, p3, p4]: score.append(p)
    score.write('musicxml', fp=out)

    # 后注入 pedal + wedge 到钢琴声部
    with open(out, 'r', encoding='utf-8') as f:
        xml = f.read()

    parts = re.findall(r'(<part[> ][^>]*>.*?</part>)', xml, re.DOTALL)
    part1 = parts[0]
    measures = re.findall(r'(<measure[> ][^>]*>.*?</measure>)', part1, re.DOTALL)

    pedal_start = '<direction placement=\"below\"><direction-type><pedal type=\"start\" line=\"yes\"/></direction-type></direction>'
    pedal_end   = '<direction placement=\"below\"><direction-type><pedal type=\"stop\" line=\"yes\"/></direction-type></direction>'
    wedge_cresc = '<direction placement=\"below\"><direction-type><wedge type=\"crescendo\" number=\"1\"/></direction-type></direction>'
    wedge_dim   = '<direction placement=\"below\"><direction-type><wedge type=\"diminuendo\" number=\"1\"/></direction-type></direction>'

    edits = {0: [pedal_start], 1: [wedge_cresc], 3: [wedge_dim, pedal_end]}
    for idx, dirs in edits.items():
        m = measures[idx]
        nm = m
        for d in dirs:
            nm = nm.replace('</measure>', f'\n      {d}\n    </measure>', 1)
        part1 = part1.replace(m, nm)

    xml = xml.replace(parts[0], part1)
    with open(out, 'w', encoding='utf-8') as f:
        f.write(xml)

    print(f'已生成: {out}')
    print(f'  pedal: {xml.count("<pedal")} | wedge: {xml.count("<wedge")} | dynamics: {xml.count("<dynamics")}')


if __name__ == '__main__':
    main()
