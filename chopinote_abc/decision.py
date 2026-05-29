"""B 决策层硬约束 + 温区退火 — Phase 1 规则实现。

提供:
  BHardBans — B 的硬约束 token 屏蔽
  apply_zone_temperature() — 段内冷→热→冷温度曲线
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .database import SectionPlan


@dataclass
class BHardBans:
    """B 的硬约束 — 检测到硬违规直接 logits[mask] = -inf。

    与主 Transformer 的 generate_step(logit_bans=...) 接口对接。
    """

    banned_tokens: set[int] = field(default_factory=set)

    # ── 声部引导违规 ──

    parallel_fifths: set[int] = field(default_factory=set)
    parallel_octaves: set[int] = field(default_factory=set)
    voice_crossing: set[int] = field(default_factory=set)

    # ── 音域违规 ──

    out_of_range: set[int] = field(default_factory=set)
        # 超出乐器音域的 Note_ON token
    extreme_leap: set[int] = field(default_factory=set)
        # 超过 12 半音跳跃的 Note_ON token

    # ── 和弦违规 ──

    unresolved_leading_tone: set[int] = field(default_factory=set)
        # 导音未解决的 token
    contour_violation: set[int] = field(default_factory=set)
        # 发展部 contour 严重偏离 target_contour 的 token

    def merge_all(self) -> set[int]:
        """合并所有违规 token 为统一集合 → 传给 generate_step。"""
        all_banned = set(self.banned_tokens)
        for attr in ('parallel_fifths', 'parallel_octaves', 'voice_crossing',
                     'out_of_range', 'extreme_leap',
                     'unresolved_leading_tone', 'contour_violation'):
            all_banned.update(getattr(self, attr, set()))
        return all_banned

    def clear(self):
        """每 bar 开始前清空。"""
        self.banned_tokens.clear()
        self.parallel_fifths.clear()
        self.parallel_octaves.clear()
        self.voice_crossing.clear()
        self.out_of_range.clear()
        self.extreme_leap.clear()
        self.unresolved_leading_tone.clear()
        self.contour_violation.clear()

    def add_out_of_range(self, token_ids: set[int]):
        """批量添加超出音域的 token。"""
        self.out_of_range.update(token_ids)

    def add_contour_violating(self, token_ids: set[int]):
        """发展部 contour 偏离 target 时禁 token。"""
        self.contour_violation.update(token_ids)

    def has_bans(self) -> bool:
        return bool(self.merge_all())


def apply_zone_temperature(
    section: SectionPlan,
    bar_idx: int,
    base_temperature: float,
) -> float:
    """段内温度退火 — 冷→热→冷，区间边界 1 bar 线性过渡。

    Args:
        section: 当前段落规划
        bar_idx: 段内 bar 位置（0=段首）
        base_temperature: 基础 temperature

    Returns:
        调整后的 temperature 值
    """
    total = section.bars
    if total <= 3:
        # 太短不分区
        return base_temperature

    cold_entry_end = max(1, int(total * 0.15))
    hot_end = total - max(1, int(total * 0.15))

    if bar_idx < cold_entry_end:
        # 冷区
        scale = 0.8
    elif bar_idx == cold_entry_end:
        # 过渡 bar：冷→热
        scale = 1.05  # (0.8 + 1.3) / 2
    elif bar_idx < hot_end - 1:
        # 热区（-1 为尾过渡预留）
        scale = 1.3 + section.innovation_budget
    elif bar_idx == hot_end - 1:
        # 过渡 bar：热→冷
        hot_scale = 1.3 + section.innovation_budget
        scale = (hot_scale + 0.7) / 2
    else:
        # 冷区（收束）
        scale = 0.7

    return base_temperature * scale


def _compute_deviation_surprise(
    bar_stats,
    baselines: dict,
) -> float:
    """Phase 1 创新判定：用 A3 统计量近似"意外度"。

    意外度 ≈ KL(token_type_dist || baseline) / max_kl
    纯统计计算，不需要模型 forward。
    """
    from .motif import _kl_divergence

    pc_dist = bar_stats.pitch_class_dist
    baseline_pc = baselines.get('pitch_class_dist')
    if not pc_dist or not baseline_pc or not isinstance(baseline_pc, list):
        return 0.0

    return _kl_divergence(pc_dist, baseline_pc)
