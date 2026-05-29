"""A2 动机提取 — Phase 1 纯规则驱动，零模型依赖。

提供:
  identify_landmarks()   → 从 A3 bar_log 选地标 bar
  purify_tokens()        → 剥离演奏层，保留结构层
  extract_dna()          → 从提纯 token 提取 MotifDNA
"""

from __future__ import annotations

import math
from .database import MotifDNA, BarStats


# 提纯时剥离的 token 前缀（演奏/装饰层）
_PURIFY_DROP_PREFIXES = [
    'Ornament', 'Grace', 'Artic.', 'Pedal',
    'Octave', 'Hairpin', 'Dynamic', 'Slur',
]


def identify_landmarks(
    bar_log: list[BarStats],
    baselines: dict,
) -> list[tuple[str, int]]:
    """从 A3 bar_log 选地标 bar — 语义驱动，非固定偏移。

    返回 [(label_tag, bar_index), ...]：
      - statement:    第一个完整 bar
      - climax:       密度最大的 bar
      - distinctive:  PC 分布 vs baseline KL 散度最大的 bar
    """
    if not bar_log:
        return []

    landmarks = []
    bars = [b for b in bar_log if b.bar >= 0]  # 排除临时 bar

    if not bars:
        return []

    # 1. 主题陈述: 第一个 bar
    landmarks.append(('statement', bars[0].bar))

    if len(bars) < 2:
        return landmarks

    # 2. 高潮点: 密度最大
    climax = max(bars, key=lambda b: b.density)
    if climax.bar != landmarks[0][1]:  # 不和 statement 重复
        landmarks.append(('climax', climax.bar))

    # 3. 最独特: PC 分布 vs baseline KL 散度最大
    baseline_pc = baselines.get('pitch_class_dist')
    if baseline_pc and isinstance(baseline_pc, list) and len(baseline_pc) > 0:
        best_kl = -1.0
        best_bar = None
        for b in bars:
            if b.pitch_class_dist and len(b.pitch_class_dist) == len(baseline_pc):
                kl = _kl_divergence(b.pitch_class_dist, baseline_pc)
                if kl > best_kl:
                    best_kl = kl
                    best_bar = b
        if best_bar and best_bar.bar not in {t[1] for t in landmarks}:
            landmarks.append(('distinctive', best_bar.bar))

    return landmarks


def purify_tokens(tokens: list[int], tokenizer) -> list[int]:
    """剥离演奏层，保留结构层。

    去掉: ornament / grace / articulation / pedal / octave / hairpin / dynamic / slur
    归一化: Velocity → 固定值 4
    保留: Note_ON, Duration, Position, Chord, Inv, Bar, Key, Section, Program, Tempo, TimeSig
    """
    purified = []
    for tid in tokens:
        ts = tokenizer.decode_token(tid)
        if not ts:
            purified.append(tid)
            continue

        # 跳过演奏细节
        if any(ts.startswith(p) for p in _PURIFY_DROP_PREFIXES):
            continue

        # Velocity 归一化
        if ts.startswith('Velocity'):
            vel_tid = tokenizer.encode_token('<Velocity 4>')
            if vel_tid != tokenizer.mask_token_id:
                purified.append(vel_tid)
            continue

        purified.append(tid)
    return purified


def extract_dna(tokens: list[int], tokenizer,
                tonic_midi: int = 60) -> MotifDNA:
    """从提纯后的 token 提取结构化特征。

    Args:
        tokens: 提纯后的 token（或原始 token，建议先 purify）
        tokenizer: REMI tokenizer
        tonic_midi: 主音 MIDI 编号，用于计算 scale_degrees

    Returns:
        MotifDNA
    """
    dna = MotifDNA()
    note_pitches: list[int] = []
    durations: list[float] = []
    strong_beats: list[bool] = []
    chord_funcs: list[int] = []
    current_chord = 0

    _PREFIX_NOTE = tokenizer.NOTE_ON
    _PREFIX_DUR = tokenizer.DURATION
    _PREFIX_POS = tokenizer.POSITION
    _PREFIX_CHORD = tokenizer.CHORD

    for i, tid in enumerate(tokens):
        ts = tokenizer.decode_token(tid)

        if ts.startswith(_PREFIX_NOTE):
            try:
                pitch = int(ts.split(' ')[1])
            except (IndexError, ValueError):
                pitch = 0
            note_pitches.append(pitch)

            # scale degree: (semitones from tonic) % 12 → degree
            degree = ((pitch - tonic_midi) % 12)
            dna.scale_degrees.append(_semitone_to_degree(degree))

            dna.chord_at_position.append(current_chord)

        elif ts.startswith(_PREFIX_DUR):
            try:
                dur = float(ts.split(' ')[1])
            except (IndexError, ValueError):
                dur = 1.0
            durations.append(dur)

        elif ts.startswith(_PREFIX_POS):
            try:
                pos = int(ts.split(' ')[1].split('_')[0])
            except (IndexError, ValueError):
                pos = 0
            # Position 0 = 强拍
            strong_beats.append(pos == 0)

        elif ts.startswith(_PREFIX_CHORD):
            try:
                func_name = ts.split(' ')[1].rstrip('>')
            except IndexError:
                func_name = ''
            # 简化：用 hash 映射到 int
            current_chord = hash(func_name) % 1000 if func_name else 0

    # contour: 相邻 Note_ON 的半音差
    if len(note_pitches) >= 2:
        dna.contour = [note_pitches[i] - note_pitches[i - 1]
                       for i in range(1, len(note_pitches))]

    # rhythm: 归一化时长比
    if durations:
        mean_dur = sum(durations) / len(durations)
        dna.rhythm = [d / max(mean_dur, 1e-8) for d in durations]

    dna.strong_beat_mask = strong_beats

    # register
    if note_pitches:
        dna.register_centroid = sum(note_pitches) / len(note_pitches)
        dna.ambitus = (min(note_pitches), max(note_pitches))

    return dna


def _semitone_to_degree(semitone: int) -> int:
    """半音程 → 调内音级（0-based C 为准）。"""
    MAJOR_DEGREES = {0: 1, 2: 2, 4: 3, 5: 4, 7: 5, 9: 6, 11: 7}
    return MAJOR_DEGREES.get(semitone, 0)


def _kl_divergence(p: list[float], q: list[float]) -> float:
    """KL(P||Q)，加小 epsilon 防除零。"""
    eps = 1e-8
    s = 0.0
    for pi, qi in zip(p, q):
        pi = max(pi, eps)
        qi = max(qi, eps)
        s += pi * math.log(pi / qi)
    return max(0.0, s)
