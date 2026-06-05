#!/usr/bin/env python3
"""
Figuration 标注脚本 — v0.3.2 gen5。
从 .tokens 文件按 Voice 拆分，每 4-bar 窗口分类织体模式，
直接将 <FigV V X> token 注入 .tokens 序列（改写原文件）。

织体类型 (11 种):
  0=none(不写token), 1=block, 2=alberti, 3=arpeggio, 4=stride,
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

FIG_IDX_TO_NAME = {
    1: 'block', 2: 'alberti', 3: 'arpeggio', 4: 'stride',
    5: 'octave_tremolo', 6: 'walking_bass', 7: 'countermelody', 8: 'pedal',
    9: 'waltz', 10: 'broken_octave', 11: 'tremolo',
}

WINDOW_BARS = 4

# 全局常量，由 _init_constants() 设置
_VOICE_IDS = None
_BAR_ID = 4
_POS_MIN = _POS_MAX = -1
_NOTE_ON_MIN = _NOTE_ON_MAX = _NOTE_ON_ZERO = 0


def _init_constants():
    global _VOICE_IDS, _BAR_ID, _POS_MIN, _POS_MAX
    global _NOTE_ON_MIN, _NOTE_ON_MAX, _NOTE_ON_ZERO
    from chopinote_dataset.tokenizer import REMITokenizer
    t = REMITokenizer(16, 8)
    _VOICE_IDS = {t.encode_token(f'<Voice {v}>'): v for v in range(4)}
    _BAR_ID = t.bar_token_id
    note_ids = [t.encode_token(f'<Note_ON {i}>') for i in range(-60, 61)]
    _NOTE_ON_MIN = min(note_ids)
    _NOTE_ON_MAX = max(note_ids)
    _NOTE_ON_ZERO = t.encode_token('<Note_ON 0>')
    pos_ids = [t.encode_token(f'<Position {i}>') for i in range(16)]
    _POS_MIN = min(pos_ids)
    _POS_MAX = max(pos_ids)


def classify_window(notes: list[tuple[int, int]]) -> int:
    """分类一个窗口内的织体类型。

    notes: [(interval, position), ...]
    """
    if len(notes) < 4:
        return FIG_NONE

    pitches = [n[0] for n in notes]
    intervals = [pitches[i+1] - pitches[i] for i in range(len(pitches) - 1)]

    # 1. 同 position ≥3 音 → Block
    pos_counts = Counter(n[1] for n in notes)
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
    strong = [n for n in notes if n[1] % 4 == 0]
    if len(strong) < 2:
        return False
    strong_pitches = [s[0] for s in strong]
    jumps = [abs(strong_pitches[i+1] - strong_pitches[i]) for i in range(len(strong_pitches) - 1)]
    return any(j > 12 for j in jumps)


def _is_waltz(notes: list) -> bool:
    pos_groups = {}
    for n in notes:
        pos_groups.setdefault(n[1], []).append(n[0])
    if len(pos_groups) < 2:
        return False
    return max(len(v) for v in pos_groups.values()) >= 2


def _pitch_variance(pitches: list[int]) -> float:
    mean = sum(pitches) / len(pitches)
    return sum((p - mean) ** 2 for p in pitches) / len(pitches)


def process_file(args: tuple) -> dict:
    """处理单个 .tokens 文件: 读取→分类→注入 FigV token→写回。"""
    fpath = args
    try:
        with open(fpath) as f:
            ids = json.load(f)
    except Exception as e:
        return {'status': 'error', 'file': fpath, 'reason': str(e)}

    if not isinstance(ids, list) or len(ids) < 10:
        return {'status': 'skip', 'file': fpath}

    # ── 第一遍: 按 bar 和 voice 收集 Note_ON 事件 ──
    # per_bar_voice[bar][voice] = [(interval, position), ...]
    per_bar_voice: dict[int, dict[int, list]] = {}
    bar_idx = -1
    current_voice = 0
    bar_positions: list[int] = []  # token 序列中每个 <Bar> 的位置

    for pos, tid in enumerate(ids):
        if tid == _BAR_ID:
            bar_idx += 1
            bar_positions.append(pos)
            per_bar_voice.setdefault(bar_idx, {})
            continue
        if tid in _VOICE_IDS:
            current_voice = _VOICE_IDS[tid]
            continue
        if _POS_MIN <= tid <= _POS_MAX:
            current_pos = tid - _POS_MIN
            continue
        if _NOTE_ON_MIN <= tid <= _NOTE_ON_MAX:
            interval = tid - _NOTE_ON_ZERO
            per_bar_voice.setdefault(bar_idx, {}).setdefault(current_voice, []).append(
                (interval, current_pos))

    if bar_idx < 0:
        return {'status': 'ok', 'file': fpath, 'injected': 0}

    # ── 第二遍: per-voice 分类 (每个 voice 独立滑动窗口) ──
    # bar_figs[bar][voice] = fig_type_idx (1-11), 0=none 不写
    bar_figs: dict[int, dict[int, int]] = {}

    all_voices = set()
    for bar in per_bar_voice.values():
        all_voices.update(bar.keys())

    for voice in all_voices:
        # 收集该 voice 的 per-bar notes
        voice_bars: dict[int, list] = {}
        for b in range(bar_idx + 1):
            notes = per_bar_voice.get(b, {}).get(voice, [])
            if notes:
                voice_bars[b] = notes

        if len(voice_bars) < WINDOW_BARS:
            continue

        sorted_bars = sorted(voice_bars.keys())

        # 滑动窗口分类
        for i in range(0, len(sorted_bars) - WINDOW_BARS + 1, max(1, WINDOW_BARS // 2)):
            window_bars = sorted_bars[i:i+WINDOW_BARS]
            window_notes = []
            for b in window_bars:
                window_notes.extend(voice_bars[b])

            if len(window_notes) < 4:
                continue

            fig = classify_window(window_notes)
            if fig == FIG_NONE:
                continue

            # 标注写回该窗口的每个 bar
            for b in window_bars:
                bar_figs.setdefault(b, {})[voice] = fig

    # ── 第三遍: 注入 <FigV V X> token 到序列中 ──
    # 在每 bar 的 <Bar> token 之后插入
    injected = 0
    new_ids = []
    # bar 号 → <FigV V X> token ID 列表
    figv_tokens_per_bar: dict[int, list[int]] = {}
    for b, voice_figs in bar_figs.items():
        figv_tokens_per_bar[b] = _build_figv_tokens(voice_figs)

    for pos, tid in enumerate(ids):
        new_ids.append(tid)
        if tid == _BAR_ID:
            bar_num = _count_bar_until(new_ids) - 1
            if bar_num in figv_tokens_per_bar:
                for fvtid in figv_tokens_per_bar[bar_num]:
                    new_ids.append(fvtid)
                    injected += 1

    # ── 写回 ──
    with open(fpath, 'w') as f:
        json.dump(new_ids, f)

    return {'status': 'ok', 'file': fpath, 'injected': injected}


def _build_figv_tokens(voice_figs: dict[int, int]) -> list[int]:
    """构建单个 bar 的 <FigV V X> token ID 列表 (按 voice 排序)。

    voice_figs: {voice_idx: fig_type_idx}, 不包含 'none' (值不会是 0)
    """
    from chopinote_dataset.tokenizer import REMITokenizer
    tk = REMITokenizer(16, 8)
    tokens = []
    for v in sorted(voice_figs.keys()):
        fi = voice_figs[v]
        if fi == FIG_NONE:
            continue
        fname = FIG_IDX_TO_NAME.get(fi)
        if fname:
            tokens.append(tk.get_figv_id(v, fname))
    return tokens


def _count_bar_until(tokens: list[int]) -> int:
    """统计 tokens 中的 <Bar> 数量。"""
    count = 0
    for tid in tokens:
        if tid == _BAR_ID:
            count += 1
    return count


def _scandir_iter(input_dir: str):
    """用 os.scandir 逐项迭代 .tokens 文件 — 避免 os.walk 列表。"""
    dirs_to_walk = [input_dir]
    while dirs_to_walk:
        d = dirs_to_walk.pop()
        try:
            with os.scandir(d) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        dirs_to_walk.append(os.path.join(d, entry.name))
                    elif entry.is_file() and entry.name.endswith('.tokens'):
                        yield os.path.join(d, entry.name)
        except OSError:
            continue


def _count_tokens_files(input_dir: str) -> int:
    """用 find 快速计数。"""
    import subprocess
    result = subprocess.run(
        ['find', input_dir, '-name', '*.tokens', '-printf', '.'],
        capture_output=True, text=True, timeout=300
    )
    return len(result.stdout)


def main():
    ap = argparse.ArgumentParser(description='Per-voice figuration annotation — v0.3.2 gen5')
    ap.add_argument('command', choices=['annotate'])
    ap.add_argument('--input-dir', default='/root/autodl-tmp/data/processed/tokens_v4')
    ap.add_argument('--num-workers', type=int, default=8)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    _init_constants()

    total = _count_tokens_files(args.input_dir)
    print(f"文件总数: {total}")

    if args.dry_run:
        print("[dry-run] 不写文件")

    stats = Counter()
    total_injected = 0

    with mp.Pool(args.num_workers) as pool:
        for i, result in enumerate(
            pool.imap_unordered(process_file, _scandir_iter(args.input_dir), chunksize=100)
        ):
            if result['status'] == 'ok':
                stats['ok'] += 1
                injected = result.get('injected', 0)
                total_injected += injected
            else:
                stats[result['status']] += 1

            if (i + 1) % 100000 == 0:
                print(f"  进度: {i+1}/{total} "
                      f"({100*(i+1)/total:.0f}%), "
                      f"ok={stats['ok']}, inj={total_injected}")

    print(f"完成! 成功={stats['ok']}, 错误={stats['error']}, "
          f"跳过={stats['skip']}, 注入={total_injected} tokens")


if __name__ == '__main__':
    main()
