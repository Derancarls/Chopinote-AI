"""A3 统计后端 — 45+ token 级音乐指标。

ABC Engine A3 层的计算核心 — 每个指标是纯函数: (tokens, tokenizer, **kwargs) -> float，返回 0~1 分数（1=最好）。

ABC Engine 层归属标识（实际调用由各层自行调度）:
    A3 = 生成前 seed 评估 + 生成中统计分析
    B1 = 硬约束监测（密度、同度、休止链）
    B2 = 趋势检测（窗口 vs seed 基线漂移）
    C  = 生成后全量评价 + 进化触发
"""

from __future__ import annotations

import math
from typing import Callable


# ═══════════════════════════════════════════════════════════════
#  辅助函数
# ═══════════════════════════════════════════════════════════════

def _tokens_by_bar(tokens: list[int], bar_id: int) -> list[list[int]]:
    """将 token 序列按 BAR token 分节。"""
    bars = [[]]
    for tid in tokens:
        if tid == bar_id:
            bars.append([])
        else:
            bars[-1].append(tid)
    return [b for b in bars if b]


def _note_on_intervals(tokens: list[int], tokenizer) -> list[int]:
    """提取 NOTE_ON 的 interval 值列表。"""
    intervals = []
    for t in tokens:
        s = tokenizer.decode_token(t)
        if s.startswith("<Note_ON"):
            val = int(s[len("<Note_ON") + 1:-1])
            intervals.append(val)
    return intervals


def _pitch_class_dist(tokens: list[int], tokenizer,
                      tonic_midi: int = 60) -> list[float]:
    """从 token 序列计算 12 音级分布。"""
    counts = [0.0] * 12
    total = 0.0
    for t in tokens:
        s = tokenizer.decode_token(t)
        if s.startswith("<Note_ON"):
            interval = int(s[len("<Note_ON") + 1:-1])
            pc = (tonic_midi + interval) % 12
            counts[pc] += 1.0
            total += 1.0
    if total == 0:
        return [1.0 / 12] * 12
    return [c / total for c in counts]


def _velocity_list(tokens: list[int], tokenizer) -> list[int]:
    """提取 Velocity token 的值列表。"""
    vals = []
    for t in tokens:
        s = tokenizer.decode_token(t)
        if s.startswith("<Velocity"):
            vals.append(int(s[len("<Velocity") + 1:-1]))
    return vals


def _count_token_type(tokens: list[int], tokenizer, prefix: str) -> int:
    """统计以 prefix 开头的 token 数量。"""
    return sum(1 for t in tokens if tokenizer.decode_token(t).startswith(prefix))


def _kl_divergence(p: list[float], q: list[float], eps: float = 1e-10) -> float:
    """KL 散度。"""
    kl = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:
            kl += pi * math.log(max(pi, eps) / max(qi, eps))
    return kl


# ═══════════════════════════════════════════════════════════════
#  统计/分布指标 (B1 / B2 / C)
# ═══════════════════════════════════════════════════════════════

def density_z_score(tokens: list[int], tokenizer,
                    reference_density: float = 0.0) -> float:
    """音符密度 Z-score。0=密度偏离过大, 1=正常。"""
    bar_id = tokenizer.bar_token_id
    bars = _tokens_by_bar(tokens, bar_id)
    if len(bars) < 2:
        return 1.0
    densities = []
    for bar_tokens in bars:
        n_notes = _count_token_type(bar_tokens, tokenizer, "<Note_ON")
        densities.append(n_notes)
    mean_d = sum(densities) / len(densities)
    if reference_density > 0:
        z = (mean_d - reference_density) / max(reference_density, 0.1)
    else:
        if len(densities) < 3:
            return 1.0
        recent = densities[-min(4, len(densities)):]
        prev = densities[-min(8, len(densities)):-min(4, len(densities))]
        if not prev:
            return 1.0
        prev_mean = sum(prev) / len(prev)
        if prev_mean == 0:
            return 1.0
        z = (sum(recent) / len(recent) - prev_mean) / prev_mean
    return 1.0 / (1.0 + abs(z) * 3)


def pitch_class_kl(tokens: list[int], tokenizer,
                   reference: list[float] | None = None) -> float:
    """音级分布 KL 散度。1=与参考一致, 0=完全不同。"""
    dist = _pitch_class_dist(tokens, tokenizer)
    if reference:
        ref = reference
    else:
        ref = [1.0 / 12] * 12
    kl = _kl_divergence(dist, ref)
    return math.exp(-kl * 5)


def interval_kl(tokens: list[int], tokenizer,
                reference: list[float] | None = None) -> float:
    """音程分布 KL 散度。"""
    intervals = _note_on_intervals(tokens, tokenizer)
    if not intervals:
        return 0.5
    counts = [0] * 25
    for iv in intervals:
        counts[min(abs(iv), 24)] += 1
    total = sum(counts)
    dist = [c / total for c in counts]
    if reference:
        ref = reference
    else:
        ref = [0.04] * 25
        ref[2] = 0.12
        ref[0] = 0.03
        ref[1] = 0.08
        ref[12] = 0.04
        ref_t = sum(ref)
        ref = [r / ref_t for r in ref]
    kl = _kl_divergence(dist, ref)
    return math.exp(-kl * 3)


def rest_ratio_score(tokens: list[int], tokenizer,
                     reference: float | None = None) -> float:
    """休止比例分数。"""
    rest_count = _count_token_type(tokens, tokenizer, "<Rest")
    total_events = _count_token_type(tokens, tokenizer, "<Note_ON") + rest_count
    if total_events == 0:
        return 0.5
    ratio = rest_count / total_events
    if reference is not None:
        delta = abs(ratio - reference)
        return 1.0 / (1.0 + delta * 5)
    if 0.03 <= ratio <= 0.25:
        return 1.0
    if ratio < 0.03:
        return 0.7
    return max(0.0, 1.0 - (ratio - 0.25) * 3.0)


def velocity_consistency(tokens: list[int], tokenizer,
                         reference_mean: float | None = None) -> float:
    """力度一致性。"""
    vals = _velocity_list(tokens, tokenizer)
    if len(vals) < 3:
        return 1.0
    mean_v = sum(vals) / len(vals)
    if reference_mean is not None:
        return 1.0 - min(abs(mean_v - reference_mean) / 40.0, 1.0)
    var_v = sum((v - mean_v) ** 2 for v in vals) / len(vals)
    cv = math.sqrt(var_v) / max(mean_v, 1)
    if 0.08 <= cv <= 0.40:
        return 1.0
    if cv < 0.08:
        return max(0.1, cv / 0.08)
    return max(0.0, 1.0 - (cv - 0.40) * 2.0)


def dissonance_ratio(tokens: list[int], tokenizer) -> float:
    """协和度 — 不协和音程比例。"""
    intervals = _note_on_intervals(tokens, tokenizer)
    if len(intervals) < 3:
        return 0.5
    dissonant = {1, 6, 11}
    count = sum(1 for iv in intervals if abs(iv) % 12 in dissonant)
    ratio = count / len(intervals)
    if 0.02 <= ratio <= 0.15:
        return 1.0
    if ratio < 0.02:
        return 0.7
    return max(0.0, 1.0 - (ratio - 0.15) * 6.0)


def syncopation_ratio(tokens: list[int], tokenizer) -> float:
    """切分音比例。Position token 判断弱拍起音。"""
    positions = []
    for t in tokens:
        s = tokenizer.decode_token(t)
        if s.startswith("<Position"):
            positions.append(int(s[len("<Position") + 1:-1]))
    if not positions:
        return 0.5
    sync = sum(1 for p in positions if p % 4 != 0)
    ratio = sync / len(positions)
    if 0.05 <= ratio <= 0.35:
        return 1.0
    if ratio < 0.05:
        return 0.6
    return max(0.0, 1.0 - (ratio - 0.35) * 3.0)


def duration_entropy(tokens: list[int], tokenizer) -> float:
    """节奏多样性 — 时值分布熵。"""
    durs = []
    for t in tokens:
        s = tokenizer.decode_token(t)
        if s.startswith("<Duration"):
            durs.append(int(s[len("<Duration") + 1:-1]))
    if len(durs) < 3:
        return 0.3
    counts = {}
    for d in durs:
        counts[d] = counts.get(d, 0) + 1
    total = len(durs)
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
    max_ent = math.log2(len(counts)) if len(counts) > 1 else 1
    normalized = entropy / max_ent if max_ent > 0 else 0
    if 0.3 <= normalized <= 0.6:
        return 1.0
    if normalized < 0.3:
        return 0.3 + normalized * 2.33
    return max(0.0, 1.0 - (normalized - 0.6) * 2.5)


def register_span(tokens: list[int], tokenizer,
                  reference_span: float | None = None) -> float:
    """音域跨度分数。"""
    intervals = _note_on_intervals(tokens, tokenizer)
    if len(intervals) < 3:
        return 0.3
    span = max(intervals) - min(intervals)
    if reference_span is not None and reference_span > 0:
        ratio = span / max(reference_span, 1)
        return 1.0 / (1.0 + abs(ratio - 1.0) * 2)
    ratio = span / 87.0
    if 0.25 <= ratio <= 0.75:
        return 1.0
    if ratio < 0.25:
        return max(0.1, ratio / 0.25)
    return max(0.2, 1.0 - (ratio - 0.75) * 2.0)


def melodic_direction(tokens: list[int], tokenizer) -> float:
    """旋律方向变化率。"""
    intervals = _note_on_intervals(tokens, tokenizer)
    if len(intervals) < 4:
        return 0.5
    directions = []
    for i in range(len(intervals) - 1):
        diff = intervals[i + 1] - intervals[i]
        if diff > 0:
            directions.append(1)
        elif diff < 0:
            directions.append(-1)
        else:
            directions.append(0)
    if len(directions) < 3:
        return 0.5
    changes = sum(1 for i in range(1, len(directions))
                  if directions[i] != 0 and directions[i] != directions[i - 1])
    ratio = changes / max(len(directions) - 1, 1)
    if 0.25 <= ratio <= 0.65:
        return 1.0
    if ratio < 0.25:
        return max(0.1, ratio / 0.25)
    return max(0.0, 1.0 - (ratio - 0.65) * 3.0)


def interval_step_ratio(tokens: list[int], tokenizer) -> float:
    """级进/大跳比例（级进 <=2 半音）。"""
    intervals = _note_on_intervals(tokens, tokenizer)
    if not intervals:
        return 0.5
    steps = sum(1 for iv in intervals if abs(iv) <= 2)
    return steps / len(intervals)


def key_consistency(tokens: list[int], tokenizer) -> float:
    """调性稳定性（KEY token 变化检测）。"""
    keys = set()
    for t in tokens:
        s = tokenizer.decode_token(t)
        if s.startswith("<Key"):
            keys.add(s)
    if len(keys) <= 1:
        return 1.0
    return max(0.0, 1.0 - (len(keys) - 1) * 0.5)


# ═══════════════════════════════════════════════════════════════
#  合法性指标 (A / B2 / C)
# ═══════════════════════════════════════════════════════════════

def pitch_range_check(tokens: list[int], tokenizer,
                      tonic_midi: int = 60) -> float:
    """音域 21-108 检查。0=越界, 1=合规。"""
    for t in tokens:
        s = tokenizer.decode_token(t)
        if s.startswith("<Note_ON"):
            interval = int(s[len("<Note_ON") + 1:-1])
            abs_pitch = tonic_midi + interval
            if abs_pitch < 21 or abs_pitch > 108:
                return 0.0
    return 1.0


def empty_measure_check(tokens: list[int], tokenizer) -> float:
    """空小节检查：连续 2+ 空小节即警告，3+ 严重扣分。"""
    bar_id = tokenizer.bar_token_id
    bars = _tokens_by_bar(tokens, bar_id)
    empty_streak = 0
    max_streak = 0
    for bar_tokens in bars:
        has_note = any(
            tokenizer.decode_token(t).startswith("<Note_ON")
            for t in bar_tokens
        )
        if has_note:
            empty_streak = 0
        else:
            empty_streak += 1
            max_streak = max(max_streak, empty_streak)
    if max_streak < 2:
        return 1.0
    return max(0.0, 1.0 - (max_streak - 1) * 0.3)


# ═══════════════════════════════════════════════════════════════
#  B1 硬约束指标（密度/同度/休止链/单节奏/极密度）
# ═══════════════════════════════════════════════════════════════

def unison_chain_check(tokens: list[int], tokenizer) -> float:
    """检测连续 8+ 个同音（interval=0）。"""
    intervals = _note_on_intervals(tokens, tokenizer)
    if len(intervals) < 4:
        return 1.0
    max_chain = 0
    current_chain = 0
    for iv in intervals:
        if iv == 0:
            current_chain += 1
            max_chain = max(max_chain, current_chain)
        else:
            current_chain = 0
    if max_chain >= 8:
        return max(0.0, 1.0 - (max_chain - 7) * 0.15)
    return 1.0


def rest_streak_check(tokens: list[int], tokenizer) -> float:
    """检测同一 position 连续 4+ 个 Rest token。"""
    rest_count = 0
    max_streak = 0
    for t in tokens:
        s = tokenizer.decode_token(t)
        if s == '<Rest>':
            rest_count += 1
            max_streak = max(max_streak, rest_count)
        elif s.startswith('<Note_ON') or s.startswith('<Position'):
            rest_count = 0
    if max_streak >= 4:
        return max(0.0, 1.0 - (max_streak - 3) * 0.25)
    return 1.0


def mono_rhythm_check(tokens: list[int], tokenizer) -> float:
    """检测连续 4 小节全部使用同一时值类型。"""
    bar_id = tokenizer.bar_token_id
    bars = _tokens_by_bar(tokens, bar_id)
    if len(bars) < 4:
        return 1.0
    recent_bars = bars[-min(8, len(bars)):]
    bar_dur_sets = []
    for bar_tokens in recent_bars:
        durs = set()
        for t in bar_tokens:
            s = tokenizer.decode_token(t)
            if s.startswith('<Duration'):
                durs.add(int(s[len('<Duration') + 1:-1]))
        bar_dur_sets.append(durs)
    if len(bar_dur_sets) < 4:
        return 1.0
    mono_streak = 0
    max_mono = 0
    for i in range(1, len(bar_dur_sets)):
        if bar_dur_sets[i] == bar_dur_sets[i - 1] and len(bar_dur_sets[i]) <= 2:
            mono_streak += 1
            max_mono = max(max_mono, mono_streak)
        else:
            mono_streak = 0
    if max_mono >= 4:
        return max(0.0, 1.0 - (max_mono - 3) * 0.2)
    return 1.0


def extreme_density_check(tokens: list[int], tokenizer) -> float:
    """检测单小节 >40 notes 或 <1 note。"""
    bar_id = tokenizer.bar_token_id
    bars = _tokens_by_bar(tokens, bar_id)
    if len(bars) < 2:
        return 1.0
    recent_bars = bars[-min(4, len(bars)):]
    for bar_tokens in recent_bars:
        nn = _count_token_type(bar_tokens, tokenizer, '<Note_ON')
        if nn > 40:
            return 0.1
        if nn < 1:
            return 0.2
    return 1.0


def max_polyphony_check(tokens: list[int], tokenizer,
                        max_per_hand: int = 6) -> float:
    """检测同位置单声部最大同时发音数 — 超过人手生理极限则扣分。"""
    current_pos = None
    pos_counts: list[int] = []
    current_count = 0

    for t in tokens:
        s = tokenizer.decode_token(t)
        if s.startswith('<Position'):
            if current_pos is not None:
                pos_counts.append(current_count)
            current_count = 0
            current_pos = True  # mark position tracking active
        elif s.startswith('<Note_ON'):
            current_count += 1
        elif s.startswith('<Bar'):
            if current_pos is not None:
                pos_counts.append(current_count)
            current_count = 0
            current_pos = None

    if current_count > 0:
        pos_counts.append(current_count)

    if not pos_counts:
        return 1.0

    max_notes = max(pos_counts)
    if max_notes <= max_per_hand:
        return 1.0
    return max(0.0, 1.0 - (max_notes - max_per_hand) * 0.15)


# ═══════════════════════════════════════════════════════════════
#  B1/B2 旋律/和声指标
# ═══════════════════════════════════════════════════════════════

def bar_boundary_melody(tokens: list[int], tokenizer,
                        tonic_midi: int = 60) -> float:
    """检查小节边界的旋律衔接流畅度。"""
    bar_id = tokenizer.bar_token_id
    bars = _tokens_by_bar(tokens, bar_id)
    if len(bars) < 2:
        return 0.8
    boundary_intervals = []
    prev_last_note = None
    for bar_tokens in bars:
        notes = [t for t in bar_tokens
                 if tokenizer.decode_token(t).startswith('<Note_ON')]
        if not notes:
            continue
        first_val = int(tokenizer.decode_token(notes[0]).split(' ')[1].rstrip('>'))
        if prev_last_note is not None:
            boundary_intervals.append(first_val - prev_last_note)
        prev_last_note = int(tokenizer.decode_token(notes[-1]).split(' ')[1].rstrip('>'))
    if len(boundary_intervals) < 2:
        return 0.8
    big_jumps = sum(1 for iv in boundary_intervals if abs(iv) > 12)
    reversals = 0
    for i in range(1, len(boundary_intervals)):
        if (boundary_intervals[i] * boundary_intervals[i - 1] < 0
            and abs(boundary_intervals[i]) > 8
            and abs(boundary_intervals[i - 1]) > 8):
            reversals += 1
    score = 1.0
    score -= big_jumps / max(len(boundary_intervals), 1) * 0.5
    score -= reversals / max(len(boundary_intervals), 1) * 0.8
    return max(0.0, score)


def parallel_fifths_check(tokens: list[int], tokenizer,
                          tonic_midi: int = 60) -> float:
    """平行五度/八度检测 — 跟踪相邻和弦间的声部引导。"""
    bar_id = tokenizer.bar_token_id
    bars = _tokens_by_bar(tokens, bar_id)
    if len(bars) < 2:
        return 1.0
    window_bars = bars[-min(4, len(bars)):]

    # 收集所有和弦（按 position 分组的 Note_ON interval 列表）
    all_chords: list[list[int]] = []
    for bar_tokens in window_bars:
        current_chord: list[int] = []
        for t in bar_tokens:
            s = tokenizer.decode_token(t)
            if s.startswith('<Position'):
                if current_chord:
                    all_chords.append(current_chord)
                    current_chord = []
            elif s.startswith('<Note_ON'):
                current_chord.append(int(s[len('<Note_ON') + 1:-1]))
        if current_chord:
            all_chords.append(current_chord)

    if len(all_chords) < 2:
        return 1.0

    parallel_count = 0
    voice_pair_checks = 0

    for ci in range(len(all_chords) - 1):
        chord_a = sorted(all_chords[ci])
        chord_b = sorted(all_chords[ci + 1])
        na, nb = len(chord_a), len(chord_b)
        if na < 2 or nb < 2:
            continue
        n_voices = min(na, nb)
        for i in range(n_voices):
            for j in range(i + 1, n_voices):
                interval_a = abs(chord_a[j] - chord_a[i])
                interval_b = abs(chord_b[j] - chord_b[i])
                if interval_a % 12 not in (7, 0) or interval_a == 0:
                    continue
                if interval_b % 12 not in (7, 0) or interval_b == 0:
                    continue
                if chord_a[i] == chord_b[i] and chord_a[j] == chord_b[j]:
                    continue
                voice_pair_checks += 1
                if interval_a % 12 == interval_b % 12:
                    parallel_count += 1

    if voice_pair_checks < 2:
        return 0.9
    ratio = parallel_count / voice_pair_checks
    if ratio > 0.3:
        return max(0.0, 1.0 - ratio * 1.5)
    return 1.0


def token_type_kl(tokens: list[int], tokenizer,
                  reference: list[float] | None = None) -> float:
    """Token 类型分布 KL 散度。"""
    _TT_PREFIXES = [
        '<Note_ON', '<Duration', '<Position', '<Rest', '<Velocity',
        '<Bar', '<Key', '<TimeSig', '<Tempo', '<Program',
        '<Chord ', '<Pedal', '<Slur', '<Hairpin', '<Dynamic',
        '<Artic', '<Ornament', '<GraceNote', '<Tuplet',
        '<Octave', '<Arpeggio', '<Tie',
    ]
    counts = [0] * len(_TT_PREFIXES)
    for t in tokens:
        s = tokenizer.decode_token(t)
        for i, prefix in enumerate(_TT_PREFIXES):
            if s.startswith(prefix):
                counts[i] += 1
                break
    total = sum(counts)
    if total == 0:
        return 0.5
    dist = [c / total for c in counts]
    if reference is not None:
        ref = reference
    else:
        ref = [1.0 / len(_TT_PREFIXES)] * len(_TT_PREFIXES)
    kl = _kl_divergence(dist, ref)
    return math.exp(-kl * 4)


def melodic_contour_match(tokens: list[int], tokenizer,
                          seed_contour: list[float] | None = None) -> float:
    """旋律轮廓匹配 — 对比生成段与 seed 的整体旋律形状。"""
    bar_id = tokenizer.bar_token_id
    bars = _tokens_by_bar(tokens, bar_id)
    if len(bars) < 2:
        return 0.7
    envelope = []
    for bar_tokens in bars:
        notes = [int(tokenizer.decode_token(t).split(' ')[1].rstrip('>'))
                 for t in bar_tokens
                 if tokenizer.decode_token(t).startswith('<Note_ON')]
        if notes:
            envelope.append((min(notes), max(notes)))
        else:
            envelope.append((0, 0))
    contour = []
    for i in range(len(envelope)):
        window = envelope[max(0, i - 1):min(len(envelope), i + 2)]
        valid = [w for w in window if w != (0, 0)]
        if valid:
            contour.append(sum(w[1] for w in valid) / len(valid))
        else:
            contour.append(0.0)
    if seed_contour is None or len(seed_contour) < 2:
        return 0.7
    if len(contour) < 2:
        return 0.7
    c_max = max(contour) if max(contour) > 0 else 1
    s_max = max(seed_contour) if max(seed_contour) > 0 else 1
    contour_norm = [c / c_max for c in contour]
    seed_norm = [s / s_max for s in seed_contour]
    n = max(len(contour_norm), len(seed_norm))
    if n < 2:
        return 0.7
    step_c = len(contour_norm) / n
    step_s = len(seed_norm) / n
    diffs = []
    for i in range(n):
        ci = contour_norm[min(int(i * step_c), len(contour_norm) - 1)]
        si = seed_norm[min(int(i * step_s), len(seed_norm) - 1)]
        diffs.append(abs(ci - si))
    avg_diff = sum(diffs) / len(diffs)
    return max(0.0, 1.0 - avg_diff * 3)


# ═══════════════════════════════════════════════════════════════
#  和弦评价指标 (B1 / B2 / C)
# ═══════════════════════════════════════════════════════════════

_CHORD_TONES: dict[str, list[int]] = {
    'I': [0, 4, 7],       'i': [0, 3, 7],
    'ii': [2, 5, 9],      'ii°': [2, 5, 8],
    'iii': [4, 7, 11],    'III': [4, 8, 11],
    'IV': [5, 9, 12],     'iv': [5, 8, 12],
    'V': [7, 11, 14],     'vi': [9, 12, 16],
    'VI': [8, 12, 15],    'vii°': [11, 14, 17],
    'N': [1, 5, 8],       'It6': [8, 12, 18],
    'Fr6': [8, 12, 14, 18], 'Ger6': [8, 12, 15, 18],
}


def _extract_chord_sequence(tokens: list[int], tokenizer) -> list[str]:
    """从 token 序列中提取和弦功能序列。"""
    chords = []
    for t in tokens:
        s = tokenizer.decode_token(t)
        if s.startswith('<Chord ') and not s.startswith('<Chord 7>'):
            func = s[len('<Chord '):-1]
            chords.append(func)
    return chords


def _progression_score(prev_func: str, curr_func: str) -> float:
    """基于功能和声规则评估相邻和弦的进行合理性。"""
    strong = {
        ('V', 'I'), ('V', 'i'), ('V', 'vi'), ('IV', 'I'), ('IV', 'i'),
        ('ii', 'V'), ('ii°', 'V'), ('vii°', 'I'), ('vii°', 'i'),
        ('I', 'IV'), ('i', 'iv'), ('I', 'V'), ('i', 'V'),
    }
    if (prev_func, curr_func) in strong:
        return 1.0
    weak = {('I', 'ii'), ('I', 'vi'), ('i', 'VI'), ('VI', 'V')}
    if (prev_func, curr_func) in weak:
        return 0.5
    if prev_func == curr_func:
        return 0.8
    return 0.3


def chord_melody_alignment(tokens: list[int], tokenizer,
                           tonic_midi: int = 60) -> float:
    """和弦-旋律一致性 — 检查旋律音是否落在当前和弦音内。"""
    chords = _extract_chord_sequence(tokens, tokenizer)
    if len(chords) < 2:
        return 0.5

    melody_intervals = []
    for t in tokens:
        s = tokenizer.decode_token(t)
        if s.startswith('<Note_ON'):
            melody_intervals.append(int(s[len('<Note_ON') + 1:-1]))

    if len(melody_intervals) < 3:
        return 0.5

    notes_per_chord = max(1, len(melody_intervals) // len(chords))
    aligned = 0
    total = 0
    for i, chord_func in enumerate(chords):
        chord_tones = _CHORD_TONES.get(chord_func)
        if chord_tones is None:
            continue
        start = i * notes_per_chord
        end = start + notes_per_chord if i < len(chords) - 1 else len(melody_intervals)
        for j in range(start, min(end, len(melody_intervals))):
            interval = melody_intervals[j]
            pc = (tonic_midi + interval) % 12
            if pc in {t % 12 for t in chord_tones}:
                aligned += 1
            total += 1

    if total == 0:
        return 0.5
    return aligned / total


def progression_validity(tokens: list[int], tokenizer) -> float:
    """和声进行合理性 — 相邻和弦进行是否符合功能和声规则。"""
    chords = _extract_chord_sequence(tokens, tokenizer)
    if len(chords) < 2:
        return 0.5

    scores = []
    for i in range(len(chords) - 1):
        scores.append(_progression_score(chords[i], chords[i + 1]))

    return sum(scores) / len(scores)


def cadence_quality(tokens: list[int], tokenizer) -> float:
    """终止式质量 — 检查段落结尾是否有合理终止式。"""
    chords = _extract_chord_sequence(tokens, tokenizer)
    if len(chords) < 3:
        return 0.5

    last_two = (chords[-2], chords[-1]) if len(chords) >= 2 else None
    last_three = (chords[-3], chords[-2], chords[-1]) if len(chords) >= 3 else None

    if last_two in [('V', 'I'), ('V', 'i')]:
        return 1.0  # Authentic
    elif last_two in [('IV', 'I'), ('iv', 'i')]:
        return 0.8  # Plagal
    elif chords[-1] == 'V':
        return 0.6  # Half
    elif last_two in [('V', 'vi'), ('V', 'VI')]:
        return 0.5  # Deceptive
    elif last_three and last_three[0] in ('ii', 'ii°') and last_three[1] == 'V' and last_three[2] in ('I', 'i'):
        return 1.0  # Perfect: ii-V-I
    return 0.1


def harmonic_rhythm_score(tokens: list[int], tokenizer,
                          reference_density: float | None = None) -> float:
    """和声节奏 — 检查和弦变化频率是否合理。"""
    chords = _extract_chord_sequence(tokens, tokenizer)
    if len(chords) < 2:
        return 0.5

    changes = sum(1 for i in range(1, len(chords)) if chords[i] != chords[i - 1])
    total = len(tokens)
    if total == 0:
        return 0.5

    if reference_density is not None and reference_density > 0:
        window_density = changes / total
        ratio = min(window_density, reference_density) / max(window_density, reference_density, 1e-8)
        return ratio
    else:
        change_rate = total / max(1, changes + 1)
        if 8 <= change_rate <= 32:
            return 1.0
        elif change_rate < 4:
            return max(0.0, change_rate / 4.0)
        elif change_rate > 64:
            return max(0.0, 1.0 - (change_rate - 64) / 64.0)
        else:
            return 0.7


# ═══════════════════════════════════════════════════════════════
#  指标注册表 — 简化版（去掉 Phase/Flag/MetricDef 编排层，
#  仅保留名称→函数的映射，供 A3/B/C 按需调用）
# ═══════════════════════════════════════════════════════════════

METRIC_FUNCTIONS: dict[str, Callable] = {
    # 统计/分布
    'density_z': density_z_score,
    'pitch_class_kl': pitch_class_kl,
    'interval_kl': interval_kl,
    'rest_ratio': rest_ratio_score,
    'velocity_consistency': velocity_consistency,
    'dissonance_ratio': dissonance_ratio,
    'syncopation_ratio': syncopation_ratio,
    'duration_entropy': duration_entropy,
    'register_span': register_span,
    'melodic_direction': melodic_direction,
    'interval_step_ratio': interval_step_ratio,
    'key_consistency': key_consistency,
    # 合法性
    'pitch_range': pitch_range_check,
    'empty_measure': empty_measure_check,
    # B1 硬约束
    'unison_chain': unison_chain_check,
    'rest_streak': rest_streak_check,
    'mono_rhythm': mono_rhythm_check,
    'extreme_density': extreme_density_check,
    'max_polyphony': max_polyphony_check,
    # 旋律/和声
    'bar_boundary_melody': bar_boundary_melody,
    'parallel_fifths': parallel_fifths_check,
    'token_type_kl': token_type_kl,
    'melodic_contour': melodic_contour_match,
    # 和弦
    'chord_melody_alignment': chord_melody_alignment,
    'progression_validity': progression_validity,
    'cadence_quality': cadence_quality,
    'harmonic_rhythm': harmonic_rhythm_score,
}


def compute_metric(name: str, tokens: list[int], tokenizer, **kwargs) -> float | None:
    """按名称调用指标。"""
    fn = METRIC_FUNCTIONS.get(name)
    if fn is None:
        return None
    return fn(tokens, tokenizer, **kwargs)


def compute_all_metrics(tokens: list[int], tokenizer, **kwargs) -> dict[str, float]:
    """计算所有已注册指标，跳过无返回的。"""
    results = {}
    for name, fn in METRIC_FUNCTIONS.items():
        try:
            val = fn(tokens, tokenizer, **kwargs)
            if val is not None:
                results[name] = val
        except TypeError:
            # 函数不接受额外 kwargs → 用无 kwargs 调用
            try:
                val = fn(tokens, tokenizer)
                if val is not None:
                    results[name] = val
            except Exception:
                results[name] = 0.0
        except Exception:
            results[name] = 0.0
    return results
