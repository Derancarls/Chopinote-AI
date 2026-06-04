"""A2 动机提取 — 纯规则驱动，零模型依赖。

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
    _PREFIX_TONIC = tokenizer.TONIC  # v0.3.0: Tonic 替代 Chord 作和声上下文化理

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

        elif ts.startswith(_PREFIX_TONIC):
            try:
                tonic_name = ts.split(' ')[1].rstrip('>')
            except IndexError:
                tonic_name = 'C'
            # 用 tonic_name → tonic_id (0-11) 作和声上下文
            current_chord = hash(tonic_name) % 1000 if tonic_name else 0

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


# ═══════════════════════════════════════════════════════════════
#  MotifTransform — 动机变形算子 (v0.3.3-opt1)
# ═══════════════════════════════════════════════════════════════


class MotifTransform:
    """动机变形算子集合。

    输入 MotifDNA，输出变形后的 MotifDNA。
    所有算子无副作用——返回新的 MotifDNA，不修改输入。

    用法:
        dna = extract_dna(tokens, tokenizer)
        inverted = MotifTransform.inversion(dna)
        stretched = MotifTransform.augmentation(dna, factor=2.0)
    """

    @staticmethod
    def retrograde(dna: MotifDNA) -> MotifDNA:
        """逆行: 时间和方向同时反转。"""
        n = len(dna.contour)
        return MotifDNA(
            contour=[-c for c in reversed(dna.contour)],
            rhythm=list(reversed(dna.rhythm)),
            scale_degrees=list(reversed(dna.scale_degrees)),
            strong_beat_mask=list(reversed(dna.strong_beat_mask)),
            register_centroid=dna.register_centroid,
            ambitus=dna.ambitus,
            chord_at_position=list(reversed(dna.chord_at_position)) if len(dna.chord_at_position) == n else [],
        )

    @staticmethod
    def inversion(dna: MotifDNA) -> MotifDNA:
        """倒影: 音程方向反转，保持时值和节奏。"""
        return MotifDNA(
            contour=[-c for c in dna.contour],
            rhythm=list(dna.rhythm),
            scale_degrees=list(dna.scale_degrees),
            strong_beat_mask=list(dna.strong_beat_mask),
            register_centroid=dna.register_centroid,
            ambitus=dna.ambitus,
            chord_at_position=list(dna.chord_at_position),
        )

    @staticmethod
    def augmentation(dna: MotifDNA, factor: float = 2.0) -> MotifDNA:
        """增值: 时值拉伸。"""
        return MotifDNA(
            contour=list(dna.contour),
            rhythm=[r * factor for r in dna.rhythm],
            scale_degrees=list(dna.scale_degrees),
            strong_beat_mask=list(dna.strong_beat_mask),
            register_centroid=dna.register_centroid,
            ambitus=dna.ambitus,
            chord_at_position=list(dna.chord_at_position),
        )

    @staticmethod
    def diminution(dna: MotifDNA, factor: float = 0.5) -> MotifDNA:
        """减值: 时值压缩。"""
        return MotifDNA(
            contour=list(dna.contour),
            rhythm=[max(0.25, r * factor) for r in dna.rhythm],
            scale_degrees=list(dna.scale_degrees),
            strong_beat_mask=list(dna.strong_beat_mask),
            register_centroid=dna.register_centroid,
            ambitus=dna.ambitus,
            chord_at_position=list(dna.chord_at_position),
        )

    @staticmethod
    def retrograde_inversion(dna: MotifDNA) -> MotifDNA:
        """逆行倒影: 先倒影再逆行。"""
        return MotifTransform.retrograde(
            MotifTransform.inversion(dna)
        )

    @staticmethod
    def fragment(dna: MotifDNA, keep_ratio: float = 0.5) -> MotifDNA:
        """碎片化: 只保留动机头部。"""
        n = max(1, int(len(dna.contour) * keep_ratio))
        return MotifDNA(
            contour=dna.contour[:n],
            rhythm=dna.rhythm[:n],
            scale_degrees=dna.scale_degrees[:n],
            strong_beat_mask=dna.strong_beat_mask[:n],
            register_centroid=dna.register_centroid,
            ambitus=dna.ambitus,
            chord_at_position=dna.chord_at_position[:n],
        )

    @staticmethod
    def sequence(dna: MotifDNA, step: int) -> MotifDNA:
        """模进: 整体移高/移低 step 个音级。"""
        return MotifDNA(
            contour=list(dna.contour),
            rhythm=list(dna.rhythm),
            scale_degrees=[max(1, min(7, s + step)) for s in dna.scale_degrees],
            strong_beat_mask=list(dna.strong_beat_mask),
            register_centroid=dna.register_centroid + step * 2,
            ambitus=(dna.ambitus[0] + step * 2, dna.ambitus[1] + step * 2),
            chord_at_position=list(dna.chord_at_position),
        )

    @staticmethod
    def interval_expand(dna: MotifDNA, ratio: float = 1.5) -> MotifDNA:
        """音程扩展: 等比例拉大音程。"""
        return MotifDNA(
            contour=[int(c * ratio) for c in dna.contour],
            rhythm=list(dna.rhythm),
            scale_degrees=list(dna.scale_degrees),
            strong_beat_mask=list(dna.strong_beat_mask),
            register_centroid=dna.register_centroid,
            ambitus=dna.ambitus,
            chord_at_position=list(dna.chord_at_position),
        )

    @staticmethod
    def rhythmic_vary(dna: MotifDNA, pattern: list[float] | None = None) -> MotifDNA:
        """节奏变奏: 保持 contour + scale_degrees, 替换节奏型。

        pattern 为 None 时用默认切分节奏 [1.0, 0.5, 0.5, 1.0, ...]。
        """
        if pattern is None:
            n = len(dna.contour)
            pattern = [1.0 if i % 2 == 0 else 0.5 for i in range(n)]
        # 拉伸/压缩 pattern 匹配 contour 长度
        contour_n = len(dna.contour)
        if contour_n == 0:
            return MotifDNA(
                contour=[], rhythm=[], scale_degrees=[], strong_beat_mask=[],
                register_centroid=dna.register_centroid, ambitus=dna.ambitus,
                chord_at_position=[],
            )
        if len(pattern) > contour_n:
            pattern = pattern[:contour_n]
        elif len(pattern) < contour_n:
            factor = contour_n / len(pattern)
            pattern = [pattern[int(i / factor)] for i in range(contour_n)]
        return MotifDNA(
            contour=list(dna.contour),
            rhythm=list(pattern),
            scale_degrees=list(dna.scale_degrees),
            strong_beat_mask=list(dna.strong_beat_mask),
            register_centroid=dna.register_centroid,
            ambitus=dna.ambitus,
            chord_at_position=list(dna.chord_at_position),
        )


# ═══════════════════════════════════════════════════════════════
#  DNA → Token 渲染
# ═══════════════════════════════════════════════════════════════

# 调内音级 → 半音程映射 (C 大调为基准)
_SCALE_DEGREE_TO_SEMITONE = {1: 0, 2: 2, 3: 4, 4: 5, 5: 7, 6: 9, 7: 11}


def render_dna_to_tokens(
    dna: MotifDNA,
    tokenizer,
    tonic_midi: int = 60,
    voice: int = 0,
    velocity: int = 5,
) -> list[int]:
    """将 MotifDNA 渲染为 token ID 序列。

    输出: [Voice V, Note_ON interval, Velocity X, Duration D, ...]
    不含 Position/Bar token——调用者负责在合适的 Position 下插入。

    Args:
        dna: 动机 DNA
        tokenizer: REMITokenizer
        tonic_midi: 主音 MIDI 编号 (默认 C4=60)
        voice: 声部编号 0-3
        velocity: 力度等级 0-7 (默认 5)

    Returns:
        token ID 列表
    """
    if not dna.contour or not dna.scale_degrees:
        return []

    tokens = [tokenizer.encode_token(f'<Voice {voice}>')]

    # 起始音: 从 scale_degrees[0] 推算 interval
    first_degree = max(1, min(7, dna.scale_degrees[0]))
    semitone_offset = _SCALE_DEGREE_TO_SEMITONE.get(first_degree, 0)
    reg = int(dna.register_centroid)
    start_midi = tonic_midi + semitone_offset + (reg - tonic_midi) // 12 * 12
    start_midi = max(21, min(108, start_midi))

    current_midi = start_midi
    for i, contour_step in enumerate(dna.contour):
        current_midi += contour_step
        current_midi = max(21, min(108, current_midi))
        interval = int(current_midi - tonic_midi)
        interval = max(-60, min(60, interval))

        tokens.append(tokenizer.encode_token(f'<Note_ON {interval}>'))

        # Velocity
        if dna.strong_beat_mask and i < len(dna.strong_beat_mask) and dna.strong_beat_mask[i]:
            tokens.append(tokenizer.encode_token(f'<Velocity {min(7, velocity + 1)}>'))
        else:
            tokens.append(tokenizer.encode_token(f'<Velocity {velocity}>'))

        # Duration: rhythm 值映射到 grid 单位 (1 beat ≈ grid_size/4 = 4)
        dur_val = dna.rhythm[i] if i < len(dna.rhythm) else 1.0
        dur_grid = max(1, min(16, round(dur_val * 4)))
        tokens.append(tokenizer.encode_token(f'<Duration {dur_grid}>'))

    return tokens


def render_dna_to_guidance(
    dna: MotifDNA,
    tokenizer,
    tonic_midi: int = 60,
    voice: int = 0,
) -> list[int]:
    """将 MotifDNA 渲染为引导 token 序列 (仅 Note_ON，无 Duration)。

    用于 apply_motif_guidance() 的目标序列——只约束音高，
    不约束时值 (留给模型自由发挥)。

    Returns:
        [Note_ON_id, Note_ON_id, ...]  纯音高序列
    """
    if not dna.contour or not dna.scale_degrees:
        return []

    first_degree = max(1, min(7, dna.scale_degrees[0]))
    semitone_offset = _SCALE_DEGREE_TO_SEMITONE.get(first_degree, 0)
    reg = int(dna.register_centroid)
    start_midi = tonic_midi + semitone_offset + (reg - tonic_midi) // 12 * 12
    start_midi = max(21, min(108, start_midi))

    tokens = []
    current_midi = start_midi
    for contour_step in dna.contour:
        current_midi += contour_step
        current_midi = max(21, min(108, current_midi))
        interval = int(current_midi - tonic_midi)
        interval = max(-60, min(60, interval))
        tokens.append(tokenizer.encode_token(f'<Note_ON {interval}>'))

    return tokens


# ═══════════════════════════════════════════════════════════════
#  兼容别名 (旧 API)
# ═══════════════════════════════════════════════════════════════


def invert_contour(contour: list[int]) -> list[int]:
    """倒影：音程方向反转。"""
    return MotifTransform.inversion(MotifDNA(contour=contour)).contour


def fragment_tokens(tokens: list[int], keep_ratio: float = 0.5) -> list[int]:
    """碎片化：保留前 keep_ratio 比例 token。"""
    if not tokens or keep_ratio <= 0:
        return []
    if keep_ratio >= 1.0:
        return list(tokens)
    return tokens[:max(1, int(len(tokens) * keep_ratio))]


def diminish_tokens(tokens: list[int], tokenizer, factor: float = 0.5) -> list[int]:
    """减值：Duration token × factor。"""
    import re
    result = []
    _PREFIX_DUR = tokenizer.DURATION
    for tid in tokens:
        ts = tokenizer.decode_token(tid)
        if ts.startswith(_PREFIX_DUR):
            m = re.search(r'([\d.]+)', ts)
            if m:
                orig = float(m.group(1))
                new_val = max(0.0625, orig * factor)
                new_ts = re.sub(r'[\d.]+', str(new_val), ts)
                new_tid = tokenizer.encode_token(new_ts)
                if new_tid != tokenizer.mask_token_id:
                    result.append(new_tid)
                    continue
        result.append(tid)
    return result


def contour_distance(a: list[int], b: list[int]) -> float:
    """两个 contour 的归一化距离 [0, 1]。"""
    if not a or not b:
        return 1.0
    n = min(len(a), len(b))
    a_slice = a[:n]
    b_slice = b[:n]
    max_diff = max(abs(x) for x in a_slice + b_slice) or 1
    dist = sum(abs(ai - bi) / max_diff for ai, bi in zip(a_slice, b_slice)) / n
    return dist


def contour_similarity(a: list[int], b: list[int]) -> float:
    """两个 contour 的相似度 [0, 1]，1 = 完全相同。"""
    return max(0.0, 1.0 - contour_distance(a, b))
