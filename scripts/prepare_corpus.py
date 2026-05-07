"""
从 music21 内置语料库和本地 MusicXML 文件准备训练数据。

用法:
    python scripts/prepare_corpus.py                          # 只处理 corpus
    python scripts/prepare_corpus.py --include-local           # 也处理 data/raw/ 下的本地文件
"""
import json
import os
import sys
from pathlib import Path
import argparse

from music21 import corpus, converter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chopinote_dataset.converter import MusicXMLToREMI


KEYBOARD_KEYWORDS = [
    'bach', 'chorale', 'bwv', 'wtc', 'invention', 'sinfonia',
    'chopin', 'ballade', 'nocturne', 'etude', 'prelude', 'mazurka',
    'scarlatti', 'sonata', 'beethoven', 'mozart', 'haydn',
    'schubert', 'schumann', 'brahms', 'debussy', 'ravel',
]


def find_pieces():
    paths = corpus.getPaths()
    suitable = []
    for p in paths:
        name = str(p).lower()
        if any(kw in name for kw in KEYBOARD_KEYWORDS):
            suitable.append(p)
    if not suitable:
        suitable = paths
    return sorted(set(suitable))


def safe_filename(name):
    return ''.join(c if c.isalnum() or c in '_-' else '_' for c in name)


def main():
    parser = argparse.ArgumentParser(description='准备训练数据')
    parser.add_argument('--include-local', action='store_true')
    parser.add_argument('--output-dir', default='data/processed')
    parser.add_argument('--augment-transpose', action='store_true',
                        help='对每首曲子做移调增强（默认 ±5 semitones）')
    parser.add_argument('--transpose-range', type=int, default=5,
                        help='移调范围 ±N semitones（默认 5）')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    token_dir = output_dir / 'tokens'
    meta_dir = output_dir / 'metadata'
    token_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    conv = MusicXMLToREMI(grid_size=16, velocity_levels=8)
    all_files = []
    converted = 0
    skipped = 0

    # ── 1. Corpus ──────────────────────────────────────────
    pieces = find_pieces()
    print(f"候选曲目: {len(pieces)}")

    for corpus_path in tqdm(pieces, desc="转换"):
        try:
            parsed = converter.parse(corpus_path)
        except Exception:
            skipped += 1
            continue

        # 提取所有 Score
        if hasattr(parsed, 'scores') and parsed.scores:
            scores = list(parsed.scores)
        elif hasattr(parsed, 'parts'):
            scores = [parsed]
        else:
            skipped += 1
            continue

        base_name = safe_filename(Path(str(corpus_path)).stem)

        for idx, score in enumerate(scores):
            try:
                tokens, meta = conv.convert_score(score, collect_metadata=True)
            except Exception:
                skipped += 1
                continue
            if not tokens:
                skipped += 1
                continue

            suffix = f'_{idx}' if len(scores) > 1 else ''
            safe = f'{base_name}{suffix}'

            with open(token_dir / f'{safe}.json', 'w', encoding='utf-8') as f:
                json.dump(tokens, f)
            if meta:
                with open(meta_dir / f'{safe}.meta.json', 'w', encoding='utf-8') as f:
                    json.dump(meta, f)

            all_files.append(str(token_dir / f'{safe}.json'))
            converted += 1

            # ── 移调增强 ─────────────────────────────────
            if args.augment_transpose:
                for semitone in range(-args.transpose_range, args.transpose_range + 1):
                    if semitone == 0:
                        continue
                    try:
                        ts = score.transpose(semitone, inPlace=False)
                        # 检查所有 MIDI 音高是否仍在 0-127 范围内
                        all_pitches = []
                        for n in ts.flat.notes:
                            if n.isNote:
                                all_pitches.append(n.pitch.midi)
                            elif n.isChord:
                                all_pitches.extend(c.pitch.midi for c in n.notes)
                        if all_pitches:
                            if min(all_pitches) < 0 or max(all_pitches) > 127:
                                skipped += 1
                                continue
                        tokens_t, meta_t = conv.convert_score(ts, collect_metadata=True)
                        if not tokens_t:
                            skipped += 1
                            continue
                        sign = '+' if semitone > 0 else ''
                        safe_t = f'{safe}_t{sign}{semitone}'
                        with open(token_dir / f'{safe_t}.json', 'w', encoding='utf-8') as f:
                            json.dump(tokens_t, f)
                        if meta_t:
                            with open(meta_dir / f'{safe_t}.meta.json', 'w', encoding='utf-8') as f:
                                json.dump(meta_t, f)
                        all_files.append(str(token_dir / f'{safe_t}.json'))
                        converted += 1
                    except Exception:
                        skipped += 1
                        continue
            # ────────────────────────────────────────────────

    # ── 2. 本地文件 ───────────────────────────────────────
    if args.include_local:
        raw_dir = Path('data/raw')
        if raw_dir.exists():
            local_files = list(raw_dir.rglob('*.musicxml')) + list(raw_dir.rglob('*.xml'))
            print(f"本地文件: {len(local_files)}")
            for path in tqdm(local_files, desc="本地文件"):
                try:
                    tokens, meta = conv.convert(str(path), collect_metadata=True)
                except Exception:
                    skipped += 1
                    continue
                if not tokens:
                    skipped += 1
                    continue
                safe = path.stem.replace(' ', '_')
                with open(token_dir / f'{safe}.json', 'w', encoding='utf-8') as f:
                    json.dump(tokens, f)
                with open(meta_dir / f'{safe}.meta.json', 'w', encoding='utf-8') as f:
                    json.dump(meta, f)
                all_files.append(str(token_dir / f'{safe}.json'))
                converted += 1

    # ── 3. 文件列表 ───────────────────────────────────────
    with open(output_dir / 'all_files.txt', 'w', encoding='utf-8') as f:
        for fp in all_files:
            f.write(fp + '\n')

    print(f"\n成功: {converted} | 跳过: {skipped}")


if __name__ == '__main__':
    main()
