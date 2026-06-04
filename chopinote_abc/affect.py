"""情感色彩量化系统 (Affective Profile) — v0.3.3-opt5。

八维情感空间 + 情绪解析器 + B2 参数联动。
纯规则, 零数据, 零训练。从 token + SSF 场实时计算。

Phase 1: AffectCalculator (八维计算)
Phase 2: 情绪解析 + B2 联动 (AFFECT_PARAM_MAP 见 decision.py)

用法:
    from chopinote_abc.affect import AffectCalculator, AffectVector
    av = AffectCalculator.compute(note_intervals=[0,4,7], ...)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from collections import Counter


# ═══════════════════════════════════════════════════════════════
#  AffectVector — 八维情感向量
# ═══════════════════════════════════════════════════════════════

DIMENSION_NAMES = [
    'brightness', 'tension', 'stability', 'energy',
    'warmth', 'depth', 'motion', 'closure',
]

# 短名 → 全名映射 (用于用户输入解析)
_DIM_ALIASES: dict[str, str] = {
    '亮': 'brightness', '暗': 'brightness',
    '紧张': 'tension', '松弛': 'tension', '不安': 'tension',
    '稳定': 'stability', '漂泊': 'stability',
    '激烈': 'energy', '宁静': 'energy', '安静': 'energy', '静': 'energy',
    '温暖': 'warmth', '冷': 'warmth', '暖': 'warmth',
    '深邃': 'depth', '浅': 'depth',
    '流动': 'motion', '停滞': 'motion', '跳跃': 'motion',
    '闭合': 'closure', '开放': 'closure',
}

# 情绪词 → 预置名映射
_EMOTION_TO_PRESET: dict[str, str] = {
    '辉煌': 'majestic', '壮丽': 'majestic',
    '忧伤': 'melancholy', '悲伤': 'melancholy', '忧愁': 'melancholy',
    '紧张': 'anxious', '焦虑': 'anxious', '不安': 'anxious',
    '宁静': 'serene', '平静': 'serene', '安详': 'serene', '平和': 'serene',
    '激情': 'passionate', '热情': 'passionate', '热烈': 'passionate',
    '神秘': 'mysterious', '诡异': 'mysterious',
    '活泼': 'playful', '欢快': 'playful', '轻快': 'playful',
    '阴沉': 'dark', '黑暗': 'dark', '阴暗': 'dark',
}


@dataclass
class AffectVector:
    """八维情感向量, 每维 ∈ [0, 1]."""
    brightness: float = 0.5
    tension: float = 0.5
    stability: float = 0.5
    energy: float = 0.5
    warmth: float = 0.5
    depth: float = 0.5
    motion: float = 0.5
    closure: float = 0.5

    def to_list(self) -> list[float]:
        return [getattr(self, d) for d in DIMENSION_NAMES]

    @classmethod
    def from_list(cls, values: list[float]) -> 'AffectVector':
        return cls(**{d: values[i] for i, d in enumerate(DIMENSION_NAMES)})

    @classmethod
    def neutral(cls) -> 'AffectVector':
        return cls()

    def blend(self, other: 'AffectVector', weight: float = 0.5) -> 'AffectVector':
        """加权混合两个向量。"""
        w = max(0.0, min(1.0, weight))
        return AffectVector(**{
            d: getattr(self, d) * (1 - w) + getattr(other, d) * w
            for d in DIMENSION_NAMES
        })

    def __getitem__(self, key: str) -> float:
        return getattr(self, key)

    def __setitem__(self, key: str, value: float):
        setattr(self, key, max(0.0, min(1.0, value)))


# ═══════════════════════════════════════════════════════════════
#  AffectCalculator
# ═══════════════════════════════════════════════════════════════

# 音程不协和权重
_INTERVAL_DISSONANCE = {
    0: 0.0,     # unison
    1: 0.7, 2: 0.7,  # seconds
    3: 0.2, 4: 0.2,  # thirds
    5: 0.1,          # perfect fourth (mild)
    6: 1.0,          # tritone
    7: 0.1,          # perfect fifth
    8: 0.2, 9: 0.2,  # sixths
    10: 0.7, 11: 0.7,  # sevenths
    12: 0.0,         # octave
}

# 节拍位权重 (grid_size=16: 0=downbeat, 4/8/12=beats)
_BEAT_WEIGHTS = {0: 1.0, 4: 0.6, 8: 1.0, 12: 0.6}

# 调号亮度: 升号数 → [0,1]
_KEY_BRIGHTNESS: dict[str, float] = {
    'Gb': 0.08, 'Db': 0.15, 'Ab': 0.22, 'Eb': 0.28, 'Bb': 0.35,
    'F': 0.42, 'C': 0.50, 'G': 0.58, 'D': 0.65,
    'A': 0.72, 'E': 0.78, 'B': 0.85, 'F#': 0.92,
    'C#': 0.08, 'Gb-': 0.08, 'D#': 0.22, 'A#': 0.35,
    'D-': 0.15, 'E-': 0.28, 'A-': 0.22, 'B-': 0.35,
}


class AffectCalculator:
    """八维情感向量实时计算器。

    每 bar 调用 compute()，纯规则，零参数。
    输入来自 token 序列 + SSF 场 + 终止式类型。
    """

    @staticmethod
    def compute(
        note_intervals: list[int],          # Note_ON interval 值列表
        beat_positions: list[int],           # 每个 note 的 beat position (0-15)
        durations: list[int],                # 每个 note 的 duration (grid units)
        tonic_field: list[float],            # 12-dim 段落级 TonicField
        tonic_name: str = 'C',               # 主音名
        cadence_type: str | None = None,     # PAC/IAC/HC/DC/PC or None
    ) -> AffectVector:
        """对单个 bar 计算全部八维。

        Args:
            note_intervals: 该 bar 内的 Note_ON interval 序列
            beat_positions:  每个 note 的 beat position (0-15)
            durations:       每个 note 的 duration (grid units)
            tonic_field:     12-dim 段落级 SSF TonicField
            tonic_name:      主音名 (用于 Brightness)
            cadence_type:    终止式类型 (用于 Closure)

        Returns:
            AffectVector
        """
        n = len(note_intervals)
        if n == 0:
            return AffectVector(
                brightness=_calc_key_brightness(tonic_name),
                stability=_calc_tonic_anchoring(tonic_field),
            )

        return AffectVector(
            brightness=AffectCalculator._compute_brightness(
                note_intervals, tonic_name),
            tension=AffectCalculator._compute_tension(
                note_intervals, beat_positions, durations),
            stability=AffectCalculator._compute_stability(tonic_field),
            energy=AffectCalculator._compute_energy(
                n, beat_positions, durations),
            warmth=AffectCalculator._compute_warmth(note_intervals),
            depth=0.5,  # needs section-level context, default neutral
            motion=AffectCalculator._compute_motion(note_intervals),
            closure=AffectCalculator._compute_closure(cadence_type),
        )

    @staticmethod
    def compute_section(bar_vectors: list[AffectVector]) -> AffectVector:
        """对段落内多个 bar 的向量取平均。"""
        if not bar_vectors:
            return AffectVector()
        n = len(bar_vectors)
        return AffectVector(**{
            d: sum(getattr(v, d) for v in bar_vectors) / n
            for d in DIMENSION_NAMES
        })

    # ── 各维度实现 ──────────────────────────────────

    @staticmethod
    def _compute_brightness(
        intervals: list[int], tonic_name: str,
    ) -> float:
        key_b = _calc_key_brightness(tonic_name)
        if len(intervals) < 3:
            return key_b
        # mode_ratio: 大三度 vs 小三度比例
        major_3rds = 0
        minor_3rds = 0
        for i in range(len(intervals)):
            for j in range(i + 1, len(intervals)):
                diff = abs(intervals[i] - intervals[j]) % 12
                if diff == 4:   # major third
                    major_3rds += 1
                elif diff == 3:  # minor third
                    minor_3rds += 1
        total = major_3rds + minor_3rds
        mode_ratio = major_3rds / total if total > 0 else 0.5
        return key_b * 0.6 + mode_ratio * 0.4

    @staticmethod
    def _compute_tension(
        intervals: list[int], beat_positions: list[int],
        durations: list[int],
    ) -> float:
        total_weight = 0.0
        weighted_sum = 0.0
        for i, interval in enumerate(intervals):
            iv_mod = abs(interval) % 12
            iv_w = _INTERVAL_DISSONANCE.get(iv_mod, 0.3)
            pos = beat_positions[i] if i < len(beat_positions) else 0
            beat_w = _BEAT_WEIGHTS.get(pos, 0.3)
            dur = durations[i] if i < len(durations) else 1
            w = iv_w * beat_w * dur
            weighted_sum += w
            total_weight += dur
        if total_weight == 0:
            return 0.5
        return max(0.0, min(1.0, weighted_sum / total_weight))

    @staticmethod
    def _compute_stability(tonic_field: list[float]) -> float:
        if len(tonic_field) < 12:
            return 0.5
        tf_max = max(tonic_field) or 1.0
        tonic_anchoring = tonic_field[0] / tf_max
        # local_deviation approximated from tonic_field spread
        mean_tf = sum(tonic_field) / 12.0
        deviation = sum(abs(v - mean_tf) for v in tonic_field) / 12.0
        dev_norm = min(1.0, deviation / 0.3)
        return tonic_anchoring * 0.5 + (1.0 - dev_norm) * 0.5

    @staticmethod
    def _compute_energy(
        n_notes: int, beat_positions: list[int], durations: list[int],
    ) -> float:
        density = min(1.0, n_notes / 16.0)
        # syncopation: off-beat notes (positions not 0,4,8,12)
        off_beat = sum(1 for p in beat_positions
                       if p % 4 != 0 and p < 16) if beat_positions else 0
        sync_ratio = off_beat / max(1, len(beat_positions))
        # dotted: duration values that are uneven (1.5x or 0.75x of standard)
        dotted = sum(1 for d in durations if d % 2 != 0 or d >= 3)
        dotted_ratio = dotted / max(1, len(durations))
        return max(0.0, min(1.0,
            density * 0.5 + sync_ratio * 0.3 + dotted_ratio * 0.2))

    @staticmethod
    def _compute_warmth(intervals: list[int]) -> float:
        # register ratio: intervals < 0 means below tonic (mid-low register)
        low = sum(1 for iv in intervals if iv < 0)
        reg_ratio = low / max(1, len(intervals))
        # interval type ratios
        count_36 = 0
        count_27 = 0
        total = 0
        for i in range(len(intervals)):
            for j in range(i + 1, len(intervals)):
                diff = abs(intervals[i] - intervals[j]) % 12
                if diff in (3, 4, 8, 9):
                    count_36 += 1
                elif diff in (1, 2, 10, 11):
                    count_27 += 1
                total += 1
        ratio_36 = count_36 / max(1, total)
        ratio_27 = count_27 / max(1, total)
        return max(0.0, min(1.0,
            reg_ratio * 0.4 + ratio_36 * 0.4 + (1.0 - ratio_27) * 0.2))

    @staticmethod
    def _compute_motion(intervals: list[int]) -> float:
        if len(intervals) < 2:
            return 0.5
        melodic = [intervals[i] - intervals[i - 1]
                   for i in range(1, len(intervals))]
        steps = sum(1 for m in melodic if 1 <= abs(m) <= 2)
        leaps = sum(1 for m in melodic if abs(m) > 4)
        n_moves = len(melodic)
        step_r = steps / n_moves if n_moves > 0 else 0.0
        leap_r = leaps / n_moves if n_moves > 0 else 0.0
        # direction changes
        dir_changes = 0
        for i in range(2, len(melodic)):
            if melodic[i - 1] != 0 and melodic[i] != 0:
                if (melodic[i - 1] > 0) != (melodic[i] > 0):
                    dir_changes += 1
        dir_rate = dir_changes / max(1, n_moves - 1)
        return max(0.0, min(1.0,
            step_r * 0.6 + (1.0 - leap_r) * 0.3 + dir_rate * 0.1))

    @staticmethod
    def _compute_closure(cadence_type: str | None) -> float:
        """单 bar 的闭合感 (近似)。段落级 Closure 由 compute_section_closure() 算。"""
        if cadence_type is None:
            return 0.5
        # PAC=强闭合, IAC=弱闭合, HC=开放, DC=意外, PC=柔和闭合
        closure_map = {'PAC': 0.95, 'IAC': 0.65, 'HC': 0.25,
                       'DC': 0.15, 'PC': 0.55}
        return closure_map.get(cadence_type, 0.5)

    @staticmethod
    def compute_section_closure(
        cadence_types: list[str], n_sections: int = 1,
    ) -> float:
        """段落级闭合感, 综合所有终止式。"""
        if not cadence_types:
            return 0.5
        counts = Counter(cadence_types)
        total = len(cadence_types)
        pac_ratio = counts.get('PAC', 0) / max(1, total)
        hc_ratio = counts.get('HC', 0) / max(1, total)
        cad_density = total / max(1, n_sections)
        return max(0.0, min(1.0,
            pac_ratio * 1.0 + (1.0 - hc_ratio) * 0.4
            + min(1.0, cad_density * 0.2) * 0.2))

    @staticmethod
    def compute_section_depth(
        tonic_fields: list[list[float]],
        n_tonic_changes: int = 0,
        n_bars: int = 16,
    ) -> float:
        """段落级深度: SSF 场熵 + 转调频率。"""
        if not tonic_fields:
            return 0.5
        # entropy of mean tonic field
        mean_tf = [0.0] * 12
        for tf in tonic_fields:
            for i in range(12):
                mean_tf[i] += tf[i]
        n = len(tonic_fields)
        mean_tf = [v / n for v in mean_tf]
        # active PC count → entropy proxy
        active = sum(1 for v in mean_tf if v > 0.1)
        entropy = active / 12.0
        # tonic change rate
        change_rate = n_tonic_changes / max(1, n_bars)
        return max(0.0, min(1.0,
            entropy * 0.5 + min(1.0, change_rate * 5.0) * 0.5))


def _calc_key_brightness(tonic_name: str) -> float:
    return _KEY_BRIGHTNESS.get(tonic_name, 0.5)


def _calc_tonic_anchoring(tonic_field: list[float]) -> float:
    if not tonic_field or max(tonic_field) == 0:
        return 0.5
    return tonic_field[0] / max(max(tonic_field), 0.01)


# ═══════════════════════════════════════════════════════════════
#  情绪预置表
# ═══════════════════════════════════════════════════════════════

AFFECT_PRESETS: dict[str, AffectVector] = {
    'majestic': AffectVector(
        brightness=0.75, tension=0.30, stability=0.85, energy=0.80,
        warmth=0.60, depth=0.40, motion=0.50, closure=0.90),
    'melancholy': AffectVector(
        brightness=0.20, tension=0.35, stability=0.70, energy=0.25,
        warmth=0.70, depth=0.50, motion=0.75, closure=0.40),
    'anxious': AffectVector(
        brightness=0.40, tension=0.80, stability=0.20, energy=0.70,
        warmth=0.30, depth=0.75, motion=0.40, closure=0.15),
    'serene': AffectVector(
        brightness=0.50, tension=0.10, stability=0.90, energy=0.10,
        warmth=0.55, depth=0.30, motion=0.60, closure=0.80),
    'passionate': AffectVector(
        brightness=0.55, tension=0.60, stability=0.50, energy=0.85,
        warmth=0.70, depth=0.50, motion=0.50, closure=0.70),
    'mysterious': AffectVector(
        brightness=0.25, tension=0.45, stability=0.30, energy=0.20,
        warmth=0.30, depth=0.80, motion=0.55, closure=0.20),
    'playful': AffectVector(
        brightness=0.65, tension=0.15, stability=0.75, energy=0.60,
        warmth=0.45, depth=0.20, motion=0.85, closure=0.60),
    'dark': AffectVector(
        brightness=0.15, tension=0.55, stability=0.40, energy=0.30,
        warmth=0.50, depth=0.60, motion=0.35, closure=0.30),
}

STYLE_PRESETS: dict[str, AffectVector] = {
    'nocturne': AffectVector(
        brightness=0.30, tension=0.25, stability=0.70, energy=0.20,
        warmth=0.75, depth=0.45, motion=0.80, closure=0.50),
    'etude': AffectVector(
        brightness=0.55, tension=0.40, stability=0.65, energy=0.90,
        warmth=0.40, depth=0.35, motion=0.60, closure=0.55),
    'waltz': AffectVector(
        brightness=0.55, tension=0.20, stability=0.80, energy=0.55,
        warmth=0.55, depth=0.25, motion=0.75, closure=0.70),
    'march': AffectVector(
        brightness=0.70, tension=0.25, stability=0.90, energy=0.75,
        warmth=0.60, depth=0.20, motion=0.40, closure=0.90),
    'chorale': AffectVector(
        brightness=0.45, tension=0.15, stability=0.95, energy=0.15,
        warmth=0.65, depth=0.50, motion=0.85, closure=0.85),
    'impromptu': AffectVector(
        brightness=0.50, tension=0.35, stability=0.55, energy=0.65,
        warmth=0.50, depth=0.40, motion=0.70, closure=0.45),
    'lullaby': AffectVector(
        brightness=0.40, tension=0.05, stability=0.92, energy=0.05,
        warmth=0.80, depth=0.15, motion=0.65, closure=0.85),
    'scherzo': AffectVector(
        brightness=0.60, tension=0.30, stability=0.65, energy=0.85,
        warmth=0.35, depth=0.30, motion=0.90, closure=0.55),
    'elegy': AffectVector(
        brightness=0.20, tension=0.40, stability=0.60, energy=0.15,
        warmth=0.60, depth=0.60, motion=0.55, closure=0.60),
    'fanfare': AffectVector(
        brightness=0.85, tension=0.20, stability=0.90, energy=0.90,
        warmth=0.55, depth=0.15, motion=0.35, closure=0.95),
}


# ═══════════════════════════════════════════════════════════════
#  情绪解析器
# ═══════════════════════════════════════════════════════════════

def parse_affective_intent(
    text: str,
    style: str | None = None,
) -> AffectVector:
    """解析用户自然语言输入 → 八维目标向量。

    解析顺序:
      1. style hint ("夜曲") → 取风格预置
      2. 情感关键词 → 覆盖对应维度
      3. 未指定 → 风格值 or 中性 0.5

    Args:
        text: 用户情感描述, 如 "温暖而流动", "忧伤", "辉煌而紧张"
        style: 风格提示, 如 "nocturne", "etude"

    Returns:
        AffectVector 目标向量
    """
    # 1. 风格预置
    if style and style.lower() in STYLE_PRESETS:
        vec = AffectVector(**{
            d: getattr(STYLE_PRESETS[style.lower()], d)
            for d in DIMENSION_NAMES
        })
    else:
        vec = AffectVector()

    if not text.strip():
        return vec

    # 2. 解析关键词
    text = text.replace('而', ' ').replace('但', ' ').replace('不', '')
    words = text.split()

    # 先检查情绪词 (映射到预置)
    matched_preset: AffectVector | None = None
    for word in words:
        word = word.strip()
        if not word:
            continue
        preset_name = _EMOTION_TO_PRESET.get(word)
        if preset_name and preset_name in AFFECT_PRESETS:
            matched_preset = AFFECT_PRESETS[preset_name]
            break

    if matched_preset:
        # 以预置为基础
        for d in DIMENSION_NAMES:
            setattr(vec, d, getattr(matched_preset, d))

    # 维度关键词覆盖
    dim_mods: dict[str, float] = {}
    for word in words:
        word = word.strip()
        if not word:
            continue
        dim_name = _DIM_ALIASES.get(word, word)
        if dim_name in DIMENSION_NAMES:
            dim_mods[dim_name] = max(dim_mods.get(dim_name, 0.0), 0.8)

    for dim_name, target_val in dim_mods.items():
        setattr(vec, dim_name, target_val)

    return vec
