#!/usr/bin/env python3
"""
v0.2.x → v0.3.0 .tokens 迁移脚本。
将旧词表 (929) 的 .tokens 文件重编码为新词表 (542)。
只做 token ID 级别操作，不重新转换原始数据。

映射规则:
  <Key C> / <Key Am>        → 提取主音 → <Tonic C> / <Tonic A>
  <Anticipate X>            → 丢弃
  <Program N>               → <Program N> <Voice 0>
  <Program N_M>             → <Program N> <Voice M>
  <Chord X> / <Chord 7> / <Inv X> → 丢弃
  其他 token                → 原样保留

用法: python scripts/migrate_to_v4.py --input-dir tokens_v3 --output-dir tokens_v4 --num-workers 16
"""
import os, sys, json, argparse, multiprocessing as mp
from collections import Counter
from chopinote_dataset.tokenizer import REMITokenizer


# ── 旧词表 (929, frozen) ──────────────────────────────────
class OldTokenizer:
    """v0.2.x 929-token vocab, frozen for migration only."""

    def __init__(self, grid_size=16, velocity_levels=8):
        self.grid_size = grid_size
        self.velocity_levels = velocity_levels
        self._t2i = {}
        self._i2t = {}
        self._build()

    def _build(self):
        idx = 0
        for t in ['<PAD>', '<BOS>', '<EOS>', '<MASK>']:
            self._t2i[t] = idx; self._i2t[idx] = t; idx += 1
        self._t2i['<Bar>'] = idx; self._i2t[idx] = '<Bar>'; idx += 1
        for i in range(self.grid_size):
            t = f'<Position {i}>'
            self._t2i[t] = idx; self._i2t[idx] = t; idx += 1
        for prog in range(128):
            t = f'<Program {prog}>'
            self._t2i[t] = idx; self._i2t[idx] = t; idx += 1
            for sub in range(1, 4):
                t = f'<Program {prog}_{sub}>'
                self._t2i[t] = idx; self._i2t[idx] = t; idx += 1
        for interval in range(-60, 61):
            t = f'<Note_ON {interval}>'
            self._t2i[t] = idx; self._i2t[idx] = t; idx += 1
        for level in range(self.velocity_levels):
            t = f'<Velocity {level}>'
            self._t2i[t] = idx; self._i2t[idx] = t; idx += 1
        for d in range(1, self.grid_size + 1):
            t = f'<Duration {d}>'
            self._t2i[t] = idx; self._i2t[idx] = t; idx += 1
        for clef in ('treble', 'bass', 'alto', 'tenor', 'soprano', 'c_1', 'c_2', 'c_5', 'percussion'):
            t = f'<Clef {clef}>'
            self._t2i[t] = idx; self._i2t[idx] = t; idx += 1
        for dyn in ('pppp', 'ppp', 'pp', 'p', 'mp', 'mf', 'f', 'ff', 'fff', 'ffff',
                    'sfz', 'sfp', 'sf', 'fz', 'fp', 'rf', 'rfz', 'sffz', 'sfpp'):
            t = f'<Dynamic {dyn}>'
            self._t2i[t] = idx; self._i2t[idx] = t; idx += 1
        for hp in ('cresc', 'dim'):
            t = f'<Hairpin {hp}>'
            self._t2i[t] = idx; self._i2t[idx] = t; idx += 1
        for art in ('staccato', 'accent', 'tenuto', 'marcato', 'pizzicato', 'fermata'):
            t = f'<Artic {art}>'
            self._t2i[t] = idx; self._i2t[idx] = t; idx += 1
        for orn in ('trill', 'mordent', 'turn', 'tremolo'):
            t = f'<Ornament {orn}>'
            self._t2i[t] = idx; self._i2t[idx] = t; idx += 1
        for ped in ('start', 'end'):
            t = f'<Pedal {ped}>'
            self._t2i[t] = idx; self._i2t[idx] = t; idx += 1
        for sl in ('start', 'end'):
            t = f'<Slur {sl}>'
            self._t2i[t] = idx; self._i2t[idx] = t; idx += 1
        for rpt in ('start', 'end', 'volta_1', 'volta_2'):
            t = f'<Repeat {rpt}>'
            self._t2i[t] = idx; self._i2t[idx] = t; idx += 1
        for jmp in ('da_capo', 'dal_segno', 'segno', 'coda', 'fine'):
            t = f'<Jump {jmp}>'
            self._t2i[t] = idx; self._i2t[idx] = t; idx += 1
        for bpm in range(30, 241, 10):
            t = f'<Tempo {bpm}>'
            self._t2i[t] = idx; self._i2t[idx] = t; idx += 1
        for ratio in ('3:2', '5:4', '6:4', '7:4', '7:8', '5:6', '9:8', '10:8',
                      '11:8', '13:8', '14:8', '15:8', '17:8', '19:8', '21:8',
                      '22:8', '2:3', '4:3', '4:5', '4:6'):
            t = f'<TupletStart {ratio}>'
            self._t2i[t] = idx; self._i2t[idx] = t; idx += 1
        self._t2i['<TupletEnd>'] = idx; self._i2t[idx] = '<TupletEnd>'; idx += 1
        for ts in ('2/4', '3/4', '4/4', '5/4', '6/4', '2/2', '3/2', '4/2',
                   '3/8', '6/8', '9/8', '12/8', '5/8', '7/8'):
            t = f'<TimeSig {ts}>'
            self._t2i[t] = idx; self._i2t[idx] = t; idx += 1
        self._t2i['<Rest>'] = idx; self._i2t[idx] = '<Rest>'; idx += 1
        for gn in ('acciaccatura', 'appoggiatura', 'grace'):
            t = f'<GraceNote {gn}>'
            self._t2i[t] = idx; self._i2t[idx] = t; idx += 1
        for key_name in ('C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'F', 'Bb', 'Eb', 'Ab', 'Db', 'Gb', 'Cb',
                         'Am', 'Em', 'Bm', 'F#m', 'C#m', 'G#m', 'D#m', 'A#m', 'Dm', 'Gm', 'Cm', 'Fm', 'Bbm', 'Ebm', 'Abm'):
            t = f'<Key {key_name}>'
            self._t2i[t] = idx; self._i2t[idx] = t; idx += 1
        for beat_num in range(1, 17):
            t = f'<Beat {beat_num}>'
            self._t2i[t] = idx; self._i2t[idx] = t; idx += 1
        for oct_val in ('8va', '8vb', '15ma', '15mb', 'end'):
            t = f'<Octave {oct_val}>'
            self._t2i[t] = idx; self._i2t[idx] = t; idx += 1
        self._t2i['<Arpeggio>'] = idx; self._i2t[idx] = '<Arpeggio>'; idx += 1
        for bass_pc in range(12):
            t = f'<Bass {bass_pc}>'
            self._t2i[t] = idx; self._i2t[idx] = t; idx += 1
        for key_name in ('C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'F', 'Bb', 'Eb', 'Ab', 'Db', 'Gb', 'Cb',
                         'Am', 'Em', 'Bm', 'F#m', 'C#m', 'G#m', 'D#m', 'A#m', 'Dm', 'Gm', 'Cm', 'Fm', 'Bbm', 'Ebm', 'Abm'):
            t = f'<Anticipate {key_name}>'
            self._t2i[t] = idx; self._i2t[idx] = t; idx += 1
        for sec_name in ('exposition', 'development', 'recapitulation',
                         'theme1', 'theme2', 'themen', 'intro', 'coda',
                         'bridge', 'cadenza', 'transition', 'variation', 'episode',
                         '0', '1', '2', '3', '4', '5', '6', '7'):
            t = f'<Section {sec_name}>'
            self._t2i[t] = idx; self._i2t[idx] = t; idx += 1
        self._t2i['<SecSum>'] = idx; self._i2t[idx] = '<SecSum>'; idx += 1
        for func in ('I', 'i', 'ii', 'ii°', 'iii', 'III', 'IV', 'iv', 'V', 'vi', 'VI', 'vii°',
                     'N', 'It6', 'Fr6', 'Ger6'):
            t = f'<Chord {func}>'
            self._t2i[t] = idx; self._i2t[idx] = t; idx += 1
        self._t2i['<Chord 7>'] = idx; self._i2t[idx] = '<Chord 7>'; idx += 1
        for inv in ('Root', '1st', '2nd', '3rd'):
            t = f'<Inv {inv}>'
            self._t2i[t] = idx; self._i2t[idx] = t; idx += 1

    def decode(self, tid):
        return self._i2t.get(tid, '<MASK>')

    @property
    def vocab_size(self):
        return len(self._t2i)


# ── 映射逻辑 ──────────────────────────────────────────────

# 降号→升号映射 (新 Tonic 词表只用升号拼写)
_FLAT_TO_SHARP = {'Bb': 'A#', 'Eb': 'D#', 'Ab': 'G#', 'Db': 'C#', 'Gb': 'F#', 'Cb': 'B'}


def key_to_tonic(key_name: str) -> str:
    """<Key C> → 'C', <Key Am> → 'A', <Key Bb> → 'A#'"""
    tonic = key_name[:-1] if key_name.endswith('m') else key_name
    return _FLAT_TO_SHARP.get(tonic, tonic)


def migrate_tokens(old_ids: list[int], old_tok: OldTokenizer, new_tok: REMITokenizer) -> list[int]:
    """将一批旧 token ID 迁移为新 token ID。"""
    new_events = []

    for tid in old_ids:
        token = old_tok.decode(tid)

        # <Key X> → <Tonic X>
        if token.startswith('<Key '):
            key_name = token[5:-1]
            tonic = key_to_tonic(key_name)
            new_events.append(f'<Tonic {tonic}>')
            continue

        # <Anticipate X> → 丢弃
        if token.startswith('<Anticipate'):
            continue

        # <Program N> → <Program N> <Voice 0>
        if token.startswith('<Program ') and '_' not in token:
            prog = token[9:-1]
            new_events.append(token)
            new_events.append('<Voice 0>')
            continue

        # <Program N_M> → <Program N> <Voice M>
        if token.startswith('<Program ') and '_' in token:
            inner = token[9:-1]
            prog, sub = inner.split('_')
            new_events.append(f'<Program {prog}>')
            new_events.append(f'<Voice {sub}>')
            continue

        # Chord tokens → 丢弃
        if token.startswith('<Chord ') or token == '<Chord 7>':
            continue
        if token.startswith('<Inv '):
            continue

        # 其余原样保留
        new_events.append(token)

    # Encode with new tokenizer
    new_ids = []
    for evt in new_events:
        tid = new_tok.encode_token(evt)
        new_ids.append(tid)

    return new_ids


def process_one(args):
    """单文件处理 (multiprocessing worker)。"""
    fpath, out_dir, old_tok, new_tok, dry_run = args
    rel = os.path.relpath(fpath, start=fpath.split(os.sep + 'tokens_')[0])
    # Find the tokens_v3 base
    try:
        old_ids = json.load(open(fpath))
    except Exception as e:
        return ('error', fpath, str(e))

    if not isinstance(old_ids, list) or len(old_ids) == 0:
        return ('skip', fpath, 'empty')

    new_ids = migrate_tokens(old_ids, old_tok, new_tok)

    if dry_run:
        return ('ok', fpath, len(old_ids), len(new_ids))

    # Write output
    out_path = os.path.join(out_dir, os.path.basename(fpath))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(new_ids, f)

    return ('ok', fpath, len(old_ids), len(new_ids))


def main():
    ap = argparse.ArgumentParser(description='v0.2.x → v0.3.0 .tokens migration')
    ap.add_argument('--input-dir', default='/root/autodl-tmp/data/processed/tokens_v3')
    ap.add_argument('--output-dir', default='/root/autodl-tmp/data/processed/tokens_v4')
    ap.add_argument('--num-workers', type=int, default=16)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    # Collect files
    all_files = []
    for root, dirs, files in os.walk(args.input_dir):
        for f in files:
            if f.endswith('.tokens'):
                all_files.append(os.path.join(root, f))

    print(f"文件总数: {len(all_files)}")

    old_tok = OldTokenizer(16, 8)
    new_tok = REMITokenizer(16, 8)

    tasks = [(f, args.output_dir, old_tok, new_tok, args.dry_run) for f in all_files]

    stats = Counter()
    total_old = total_new = 0

    with mp.Pool(args.num_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(process_one, tasks, chunksize=200)):
            status = result[0]
            stats[status] += 1
            if status == 'ok':
                total_old += result[2]
                total_new += result[3]
            elif status == 'error':
                print(f"  ERROR: {result[1]}: {result[2]}")

            if (i + 1) % 50000 == 0:
                print(f"  进度: {i+1}/{len(all_files)} ({100*(i+1)/len(all_files):.0f}%)")

    print(f"\n完成! {stats['ok']} 成功, {stats['error']} 错误, {stats['skip']} 跳过")
    if total_old > 0:
        print(f"旧 token 总数: {total_old}, 新 token 总数: {total_new}")
        print(f"变化: {total_new - total_old:+d} ({(total_new/total_old - 1)*100:+.1f}%)")


if __name__ == '__main__':
    main()
