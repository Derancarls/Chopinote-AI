"""指标注册表：定义每个指标的阶段归属及 token 级实现。

阶段定义:
    A  = 生成前 seed 评估（异步）
    B1 = 生成中局部（最近 N 节自身流畅性）
    B2 = 生成中全局（每 S 节一块 vs seed 整体）
    C  = 生成后全量评价（MusicXML 级）
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Flag, auto
from typing import Callable, Optional


class Phase(Flag):
    """指标生效阶段（位掩码，一个指标可跨多个阶段）。"""
    A = auto()
    B1 = auto()
    B2 = auto()
    C = auto()

    @classmethod
    def from_str(cls, s: str) -> "Phase":
        m = cls(0)
        for ch in s.strip().split(","):
            ch = ch.strip()
            if ch == "A":
                m |= cls.A
            elif ch == "B1":
                m |= cls.B1
            elif ch == "B2":
                m |= cls.B2
            elif ch == "C":
                m |= cls.C
        return m


@dataclass
class MetricDef:
    """指标定义。"""
    name: str                          # 唯一标识，如 "density_z"
    label: str                         # 可读标签，如 "音符密度"
    phases: Phase                      # 生效阶段
    fn_tokens: Optional[Callable] = None  # token 级实现（B1/B2 用）
    fn_score: Optional[Callable] = None   # Score 级实现（现有 evaluator）
    weight: float = 0.0                # 聚合权重（仅 C 阶段综合打分用）
    description: str = ""


# ── 已提取的常量 ──────────────────────────────────────

# NOTE_ON 的 interval 范围：-60 ~ +60，对应 MIDI 偏移
# Position 颗粒度
# Rest token ID

# ── token 级辅助函数 ───────────────────────────────────


def _tokens_by_bar(tokens: list[int], bar_id: int) -> list[list[int]]:
    """将 token 序列按 BAR token 分节。"""
    bars = [[]]
    for tid in tokens:
        if tid == bar_id:
            bars.append([])
        else:
            bars[-1].append(tid)
    return [b for b in bars if b is not None]


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


def kl_divergence(p: list[float], q: list[float], eps: float = 1e-10) -> float:
    """KL 散度。"""
    kl = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:
            kl += pi * math.log(max(pi, eps) / max(qi, eps))
    return kl


# ── token 级指标实现 ──────────────────────────────────
# 每个函数签名: (tokens, tokenizer, **kwargs) -> float
# 返回分数 0~1，1=最好


def _density_z_tokens(tokens: list[int], tokenizer,
                      reference_density: float = 0.0) -> float:
    """密度 Z-score。reference_density=0 时用滑动窗口自身。"""
    from chopinote_model.generate import tokenizer as _tk
    bar_id = tokenizer.bar_token_id
    bars = _tokens_by_bar(tokens, bar_id)
    if len(bars) < 2:
        return 1.0
    # 去掉最后一个未完成小节
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


def _pitch_class_kl_tokens(tokens: list[int], tokenizer,
                           reference: list[float] | None = None) -> float:
    """pitch class KL 散度。无 reference 时 vs 均匀分布。"""
    dist = _pitch_class_dist(tokens, tokenizer)
    if reference:
        ref = reference
    else:
        ref = [1.0 / 12] * 12
    kl = kl_divergence(dist, ref)
    return math.exp(-kl * 5)


def _interval_kl_tokens(tokens: list[int], tokenizer,
                        reference: list[float] | None = None) -> float:
    """interval 分布 KL 散度。"""
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
    kl = kl_divergence(dist, ref)
    return math.exp(-kl * 3)


def _rest_ratio_tokens(tokens: list[int], tokenizer,
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


def _velocity_consistency_tokens(tokens: list[int], tokenizer,
                                 reference_mean: float | None = None) -> float:
    """力度连续性。"""
    vals = _velocity_list(tokens, tokenizer)
    if len(vals) < 3:
        return 1.0
    mean_v = sum(vals) / len(vals)
    if reference_mean is not None:
        return 1.0 - min(abs(mean_v - reference_mean) / 40.0, 1.0)
    # 自身变异系数
    var_v = sum((v - mean_v) ** 2 for v in vals) / len(vals)
    cv = math.sqrt(var_v) / max(mean_v, 1)
    if 0.08 <= cv <= 0.40:
        return 1.0
    if cv < 0.08:
        return max(0.1, cv / 0.08)
    return max(0.0, 1.0 - (cv - 0.40) * 2.0)


def _dissonance_ratio_tokens(tokens: list[int], tokenizer) -> float:
    """不协和音程比例。"""
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


def _syncopation_ratio_tokens(tokens: list[int], tokenizer) -> float:
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


def _duration_entropy_tokens(tokens: list[int], tokenizer) -> float:
    """时值分布熵。"""
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


def _register_span_tokens(tokens: list[int], tokenizer,
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


def _melodic_direction_tokens(tokens: list[int], tokenizer) -> float:
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


def _interval_shift_tokens(tokens: list[int], tokenizer) -> float:
    """步进/大跳比例（级进 <=2 半音）。"""
    intervals = _note_on_intervals(tokens, tokenizer)
    if not intervals:
        return 0.5
    steps = sum(1 for iv in intervals if abs(iv) <= 2)
    return steps / len(intervals)


def _key_consistency_tokens(tokens: list[int], tokenizer) -> float:
    """调性稳定性（KEY  token 变化检测）。"""
    keys = set()
    for t in tokens:
        s = tokenizer.decode_token(t)
        if s.startswith("<Key"):
            keys.add(s)
    if len(keys) <= 1:
        return 1.0
    return max(0.0, 1.0 - (len(keys) - 1) * 0.5)


def _empty_measure_tokens(tokens: list[int], tokenizer) -> float:
    """空小节检查：连续 3+ 空小节 → 低分。"""
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


def _pitch_range_tokens(tokens: list[int], tokenizer,
                        tonic_midi: int = 60) -> float:
    """音域 21-108 检查。"""
    for t in tokens:
        s = tokenizer.decode_token(t)
        if s.startswith("<Note_ON"):
            interval = int(s[len("<Note_ON") + 1:-1])
            abs_pitch = tonic_midi + interval
            if abs_pitch < 21 or abs_pitch > 108:
                return 0.0
    return 1.0


# ── 注册表 ────────────────────────────────────────────
# 每个指标注册其阶段归属 + 实现函数


def _register_all() -> dict[str, MetricDef]:
    """构建全量指标注册表。"""
    registry: dict[str, MetricDef] = {}

    def reg(name, label, phases, fn_tokens=None, weight=0.0, desc=""):
        registry[name] = MetricDef(
            name=name, label=label,
            phases=Phase.from_str(phases),
            fn_tokens=fn_tokens, weight=weight, description=desc,
        )

    # ── 合法性（A / B2 / C） ──────────────────────
    reg("pitch_range", "音域合规", "A,B2,C",
        fn_tokens=_pitch_range_tokens, desc="MIDI 21-108")
    reg("empty_measure", "空小节检测", "A,B2,C",
        fn_tokens=_empty_measure_tokens, desc="连续空小节 ≥2 警告")
    reg("tuplet_pair", "Tuplet配对", "A,B2,C",
        desc="TUPLET_START/END 配对")
    reg("tie_pair", "Tie配对", "A,B2,C",
        desc="TIE_START/STOP 配对")

    # ── 统计/分布（B1 / B2 / C） ─────────────────
    reg("pitch_class_kl", "音级分布KL", "B1,B2,C",
        fn_tokens=_pitch_class_kl_tokens, weight=0.12)
    reg("interval_kl", "音程分布KL", "B1,B2,C",
        fn_tokens=_interval_kl_tokens, weight=0.08)
    reg("density_z", "音符密度", "B1,B2,C",
        fn_tokens=_density_z_tokens, weight=0.10)
    reg("rest_ratio", "休止比例", "B1,B2,C",
        fn_tokens=_rest_ratio_tokens, weight=0.04)
    reg("register_span", "音域跨度", "B1,B2,C",
        fn_tokens=_register_span_tokens, weight=0.04)
    reg("velocity_consistency", "力度一致性", "B1,B2,C",
        fn_tokens=_velocity_consistency_tokens, weight=0.04)
    reg("dissonance_ratio", "协和度", "B1,C",
        fn_tokens=_dissonance_ratio_tokens, weight=0.06)
    reg("syncopation_ratio", "切分音比例", "B1,C",
        fn_tokens=_syncopation_ratio_tokens, weight=0.05)
    reg("duration_entropy", "节奏多样性", "B1,B2,C",
        fn_tokens=_duration_entropy_tokens, weight=0.06)
    reg("melodic_direction", "旋律方向", "B1,C",
        fn_tokens=_melodic_direction_tokens, weight=0.03)
    reg("interval_shift", "音程偏移", "B1,B2,C",
        fn_tokens=_interval_shift_tokens)
    reg("key_consistency", "调性稳定性", "B2,C",
        fn_tokens=_key_consistency_tokens, weight=0.10)

    # ── B1 局部专用 ─────────────────────────────
    reg("density_delta", "密度波动", "B1",
        fn_tokens=_density_z_tokens)
    reg("articulation_delta", "演奏法密度", "B1")

    # ── B2 全局专用 ─────────────────────────────
    reg("token_type_kl", "Token类型分布KL", "B2,C", weight=0.05)

    # ── C 阶段专用（全量评价时启动） ─────────────
    reg("self_similarity", "自相似性", "C", weight=0.08)
    reg("pitch_entropy", "音高熵", "C", weight=0.05)
    reg("chromaticism_index", "半音化程度", "C", weight=0.05)
    reg("harmonic_rhythm", "和声节奏", "C", weight=0.04)
    reg("polyphony_mean", "平均复音数", "C", weight=0.03)
    reg("texture_variance", "织体变化", "C", weight=0.03)
    reg("contour_arc", "拱形结构", "C", weight=0.03)

    return registry


# 单例注册表
REGISTRY: dict[str, MetricDef] = _register_all()


def get_metrics(phase: Phase) -> list[MetricDef]:
    """获取指定阶段的指标列表。"""
    return [m for m in REGISTRY.values() if m.phases & phase]


def get_metric_names(phase: Phase) -> list[str]:
    """获取指定阶段的指标名列表。"""
    return [m.name for m in get_metrics(phase)]


def filter_metrics_by_names(names: list[str]) -> list[MetricDef]:
    """按名称筛选指标，保持输入顺序。"""
    return [REGISTRY[n] for n in names if n in REGISTRY]
