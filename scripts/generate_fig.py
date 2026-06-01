#!/usr/bin/env python3
"""
Figuration 标注脚本 — v0.3.0。
从 .tokens 文件按 Voice 拆分，每 4-bar 窗口分类织体模式，生成 .fig.json。

织体类型 (11 种):
  0=none, 1=block, 2=alberti, 3=arpeggio, 4=stride,
  5=octave_tremolo, 6=walking_bass, 7=countermelody, 8=pedal,
  9=waltz, 10=broken_octave, 11=tremolo

用法:
  python scripts/generate_fig.py annotate \
      --input-dir /root/autodl-tmp/data/processed/tokens_v4 \
      --num-workers 8
"""
import os, sys, json, argparse, multiprocessing as mp
from collections import Counter

FIG_NONE = 0
FIG_BLOCK = 1
FIG_ALBERTI = 2
FIG_ARPEGGIO = 3
FIG_STRIDE = 4
FIG_OCTAVE_TREMOLO = 5
FIG_WALKING_BASS = 6
FIG_COUNTERMELODY = 7
FIG_PEDAL = 8
FIG_WALTZ = 9
FIG_BROKEN_OCTAVE = 10
FIG_TREMOLO = 11

WINDOW_BARS = 4

# 每声部单独分类的心跳
_VOICE_IDS = None  # set in main()
_NOTE_ON_MIN = _NOTE_ON_MAX = _NOTE_ON_ZERO = 0
_BAR_ID = 4
_POS_MIN = _POS_MAX = 0


def _init_constants():
    global _VOICE_IDS, _NOTE_ON_MIN, _NOTE_ON_MAX, _NOTE_ON_ZERO, _POS_MIN, _POS_MAX
    from chopinote_dataset.tokenizer import REMITokenizer
    t = REMITokenizer(16, 8)
    _VOICE_IDS = {t.encode_token(f'<Voice {v}>'): v for v in range(4)}
    note_ids = [t.encode_token(f'<Note_ON {i}>') for i in range(-60, 61)]
    _NOTE_ON_MIN = min(note_ids)
    _NOTE_ON_MAX = max(note_ids)
    _NOTE_ON_ZERO = t.encode_token('<Note_ON 0>')
    pos_ids = [t.encode_token(f'<Position {i}>') for i in range(16)]
    _POS_MIN = min(pos_ids)
    _POS_MAX = max(pos_ids)


def classify_window(notes: list[tuple[int, int, int]]) -> int:
    """分类一个窗口内的织体类型。

    notes: [(interval, duration, position), ...]
    """
    if len(notes) < 4:
        return FIG_NONE

    pitches = [n[0] for n in notes]
    durations = [n[1] for n in notes]
    positions = [n[2] for n in notes]
    intervals = [pitches[i+1] - pitches[i] for i in range(len(pitches) - 1)]

    # 1. 同 position ≥3 音 → Block
    pos_counts = Counter(positions)
    if max(pos_counts.values()) >= 3:
        return FIG_BLOCK

    # 2. 八度交替 → Octave Tremolo
    if len(intervals) >= 4 and all(abs(i) in (0, 12) for i in intervals):
        if any(abs(i) == 12 for i in intervals):
            return FIG_OCTAVE_TREMOLO

    # 3. 同音/近距离快速重复 → Tremolo
    if len(intervals) >= 4 and sum(1 for i in intervals if abs(i) <= 1) >= len(intervals) * 0.7:
        return FIG_TREMOLO

    # 4. Alberti: 3-4 音周期
    if _is_alberti(intervals):
        return FIG_ALBERTI

    # 5. Broken octave
    if all(abs(i) in (0, 12) for i in intervals):
        return FIG_BROKEN_OCTAVE

    # 6. Stride: 强拍单音 + 弱拍多音, 低音跳跃大
    if _is_stride(notes):
        return FIG_STRIDE

    # 7. Waltz: 低音-和弦-和弦
    if _is_waltz(notes):
        return FIG_WALTZ

    # 8. Walking bass: 级进
    if len(intervals) >= 4 and all(abs(i) <= 2 for i in intervals):
        return FIG_WALKING_BASS

    # 9. Pedal: 音高几乎不变
    if len(set(pitches)) <= 2:
        return FIG_PEDAL

    # 10. Countermelody: 音高方差大
    if len(pitches) >= 8 and _pitch_variance(pitches) > 30:
        return FIG_COUNTERMELODY

    # 11. 默认 → 琶音
    return FIG_ARPEGGIO


def _is_alberti(intervals: list[int]) -> bool:
    if len(intervals) < 6:
        return False
    for period in (3, 4):
        for offset in range(period):
            errors = 0
            for i in range(offset, len(intervals) - period, period):
                if abs(intervals[i] - intervals[i+period]) > 1:
                    errors += 1
            if errors <= len(intervals) // (period * 3):
                return True
    return False


def _is_stride(notes: list) -> bool:
    strong = [n for n in notes if n[2] % 4 == 0]
    if len(strong) < 2:
        return False
    strong_pitches = [s[0] for s in strong]
    jumps = [abs(strong_pitches[i+1] - strong_pitches[i]) for i in range(len(strong_pitches) - 1)]
    return any(j > 12 for j in jumps)


def _is_waltz(notes: list) -> bool:
    pos_groups = {}
    for n in notes:
        pos_groups.setdefault(n[2], []).append(n[0])
    if len(pos_groups) < 2:
        return False
    return max(len(v) for v in pos_groups.values()) >= 2


def _pitch_variance(pitches: list[int]) -> float:
    mean = sum(pitches) / len(pitches)
    return sum((p - mean) ** 2 for p in pitches) / len(pitches)


def process_file(args: tuple) -> dict:
    fpath, out_dir, dry_run = args
    try:
        with open(fpath) as f:
            ids = json.load(f)
    except Exception as e:
        return {'status': 'error', 'file': fpath, 'reason': str(e)}

    if not isinstance(ids, list) or len(ids) < 10:
        return {'status': 'skip', 'file': fpath}

    # 按 bar 和 voice 收集 Note_ON 事件
    # per_bar_voice[bar][voice] = [(interval, duration, position)]
    per_bar_voice: dict[int, dict[int, list]] = {}
    bar_idx = -1
    current_voice = 0

    for pos, tid in enumerate(ids):
        if tid == _BAR_ID:
            bar_idx += 1
            per_bar_voice.setdefault(bar_idx, {})
            continue
        if tid in _VOICE_IDS:
            current_voice = _VOICE_IDS[tid]
            continue
        if _POS_MIN <= tid <= _POS_MAX:
            per_bar_voice.setdefault(bar_idx, {})
            current_pos = tid - _POS_MIN
            continue
        if _NOTE_ON_MIN <= tid <= _NOTE_ON_MAX:
            interval = tid - _NOTE_ON_ZERO
            per_bar_voice.setdefault(bar_idx, {}).setdefault(current_voice, []).append(
                (interval, 0, current_pos))
            continue

    # 对每个 voice，按 4-bar 窗口分类
    fig_ids = [FIG_NONE] * len(ids)  # per-token fig 标注
    if bar_idx < 0:
        return {'status': 'ok', 'file': fpath, 'data': {'fig_ids': fig_ids, 'fig_changes': []}}

    all_voices = set()
    for bar in per_bar_voice.values():
        all_voices.update(bar.keys())

    fig_changes = []

    for voice in all_voices:
        # 收集该 voice 的 per-bar notes
        voice_bars: dict[int, list] = {}
        for b in range(bar_idx + 1):
            notes = per_bar_voice.get(b, {}).get(voice, [])
            if notes:
                voice_bars[b] = notes

        if len(voice_bars) < WINDOW_BARS:
            continue

        # 滑动窗口分类
        sorted_bars = sorted(voice_bars.keys())
        prev_fig = None
        for i in range(0, len(sorted_bars) - WINDOW_BARS + 1, WINDOW_BARS // 2):
            window_bars = sorted_bars[i:i+WINDOW_BARS]
            window_notes = []
            for b in window_bars:
                window_notes.extend(voice_bars[b])

            if len(window_notes) < 4:
                continue

            fig = classify_window(window_notes)
            if fig == FIG_NONE:
                continue

            # 把标注写回该窗口的 bar 范围内
            start_bar = window_bars[0]
            end_bar = window_bars[-1]

            if fig != prev_fig:
                fig_changes.append({
                    'bar': start_bar,
                    'voice': voice,
                    'fig': fig,
                })
                prev_fig = fig

    return {
        'status': 'ok',
        'file': fpath,
        'data': {
            'fig_ids': fig_ids,
            'fig_changes': fig_changes,
        },
    }


def main():
    ap = argparse.ArgumentParser(description='Figuration annotation for v0.3.0')
    ap.add_argument('command', choices=['annotate'])
    ap.add_argument('--input-dir', default='/root/autodl-tmp/data/processed/tokens_v4')
    ap.add_argument('--num-workers', type=int, default=8)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    _init_constants()

    all_files = []
    for root, dirs, files in os.walk(args.input_dir):
        for f in files:
            if f.endswith('.tokens'):
                all_files.append(os.path.join(root, f))

    print(f"文件总数: {len(all_files)}")

    tasks = [(f, args.input_dir, args.dry_run) for f in all_files]
    stats = Counter()

    with mp.Pool(args.num_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(process_file, tasks, chunksize=100)):
            if result['status'] == 'ok':
                stats['ok'] += 1
                if not args.dry_run:
                    data = result['data']
                    out_path = result['file'].replace('.tokens', '.fig.json')
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    with open(out_path, 'w') as f:
                        json.dump(data, f)
            else:
                stats[result['status']] += 1

            if (i + 1) % 100000 == 0:
                print(f"  进度: {i+1}/{len(all_files)} ({100*(i+1)/len(all_files):.0f}%), ok={stats['ok']}")

    print(f"完成! 成功={stats['ok']}, 错误={stats['error']}, 跳过={stats['skip']}")


if __name__ == '__main__':
    main()
