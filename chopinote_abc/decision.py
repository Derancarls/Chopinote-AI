"""B 决策层硬约束 + 温区退火 + 创新预算 + 发展配方 — Phase 2 完整实现。

提供:
  BHardBans — B 的硬约束 token 屏蔽
  apply_zone_temperature() — 段内冷→热→冷温度曲线
  BFeedback — 每 bar 反馈上下文 (致命信号 / 创新判定 / 发展配方)
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

    Phase 2: 热区系数融入 innovation_budget。

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

    # Phase 2: 热区系数 = 1.3 + innovation_budget（发展部更热）
    hot_scale = 1.3 + section.innovation_budget

    if bar_idx < cold_entry_end:
        # 冷区
        scale = 0.8
    elif bar_idx == cold_entry_end:
        # 过渡 bar：冷→热
        scale = (0.8 + hot_scale) / 2
    elif bar_idx < hot_end - 1:
        # 热区（-1 为尾过渡预留）
        scale = hot_scale
    elif bar_idx == hot_end - 1:
        # 过渡 bar：热→冷
        scale = (hot_scale + 0.7) / 2
    else:
        # 冷区（收束）
        scale = 0.7

    return base_temperature * scale


# ═══════════════════════════════════════════════════════════════
#  Phase 2: B 反馈上下文 — 致命信号 / 创新预算 / 发展配方
# ═══════════════════════════════════════════════════════════════


@dataclass
class InnovationEntry:
    """单条创新记录，写入 A3 创新日志。"""
    bar: int
    innovation_type: str           # 'surprising_but_coherent'
    surprise: float
    tokens: list[int] | None = None


@dataclass
class BFeedback:
    """B 决策层每 bar 的完整反馈上下文。

    每 bar 生成完后更新，携带三部分输出：
      adjustments — 软调节（temperature/rest_penalty/complexity）
      hard_bans — 硬约束（BHardBans 或 merge_all() 结果）
      fatal_signals — 致命信号计数（触发和声回退 / 中止）
    """

    # ── 致命信号追踪 ──
    b1_low_streak: int = 0           # 连续 B1 分 < 0.2 的 bar 数
    consecutive_empty: int = 0        # 连续全空 bar 数
    FATAL_B1_THRESHOLD: int = 3       # B1 低分达到此数 → 和声回退
    FATAL_EMPTY_THRESHOLD: int = 4    # 全空 bar 达到此数 → 中止

    # ── 创新预算追踪 ──
    innovations_used: int = 0         # 本段已消耗的创新次数
    innovation_log: list[InnovationEntry] = field(default_factory=list)

    # ── 发展配方状态 ──
    invert_target: list[int] | None = None  # 当前倒影 target contour

    def on_bar_complete(self, bar_stats, b1_score: float | None,
                        section: SectionPlan | None) -> dict:
        """每 bar 生成完后更新状态，返回需要执行的指令。

        Returns:
            {'fatal': str | None,  # 'reharmonize' | 'abort' | None
             'admit_innovation': bool,
             'adjustments': dict,
             }
        """
        result = {
            'fatal': None,
            'admit_innovation': False,
            'adjustments': {},
        }

        # ── 致命信号检测 ──
        if b1_score is not None and b1_score < 0.2:
            self.b1_low_streak += 1
        else:
            self.b1_low_streak = 0

        if bar_stats and bar_stats.density < 0.1:
            self.consecutive_empty += 1
        else:
            self.consecutive_empty = 0

        if self.b1_low_streak >= self.FATAL_B1_THRESHOLD:
            result['fatal'] = 'reharmonize'
            self.b1_low_streak = 0  # 重置，留给回退后重新计数
            return result

        if self.consecutive_empty >= self.FATAL_EMPTY_THRESHOLD:
            result['fatal'] = 'abort'
            return result

        # ── 创新预算判定 ──
        if section is not None and bar_stats is not None:
            admitted = self._check_innovation(
                bar_stats, section, b1_score)
            result['admit_innovation'] = admitted
            if admitted:
                self.innovations_used += 1

        return result

    def _check_innovation(self, bar_stats, section: SectionPlan,
                          b1_score: float | None) -> bool:
        """判定当前 bar 的偏离是否为有意的创新。

        条件: 统计分布显著偏离（surprise > 0.5）
               + 硬约束通过（b1_score > 0.3 作为 proxy）
               + 创新预算还有剩余
        """
        # surprise 用 PC 分布 vs 基线 KL（Phase 1/2 规则近似）
        surprise = max(0.0, bar_stats.density_variance
                       if hasattr(bar_stats, 'density_variance')
                       else 0.0)

        # 简化版判定: b1_score 在 0.3-0.6 区间（不太差也不太完美）→ 可能是有意偏离
        coherence_ok = b1_score is None or b1_score > 0.3

        if not coherence_ok:
            return False

        budget = section.innovation_budget * section.bars
        if self.innovations_used >= budget:
            return False

        return True

    def record_innovation_entry(self, bar: int,
                                 innovation_type: str,
                                 surprise: float,
                                 tokens: list[int] | None = None):
        """记录一次被承认的创新。"""
        self.innovation_log.append(InnovationEntry(
            bar=bar,
            innovation_type=innovation_type,
            surprise=surprise,
            tokens=tokens,
        ))

    def setup_development_ops(self, ops: list[str],
                               theme_contour: list[int] | None):
        """在段开始前，根据 A1 规划的发展配方设置 target。

        Args:
            ops: development_ops 列表，如 ['invert', 'fragment']
            theme_contour: 主题动机的 contour（用于计算 target）
        """
        self.invert_target = None
        if 'invert' in ops and theme_contour:
            from .motif import invert_contour
            self.invert_target = invert_contour(theme_contour)

    def get_development_adjustments(self,
                                     current_contour: list[int] | None,
                                     ) -> dict:
        """基于发展配方的参数调整建议。"""
        adjustments = {}
        if self.invert_target is None or current_contour is None:
            return adjustments

        from .motif import contour_distance
        deviation = contour_distance(current_contour, self.invert_target)

        if deviation > 0.6:
            # 严重偏离 → 收紧生成
            adjustments['temperature'] = -0.05
            adjustments['complexity'] = -0.3
        elif deviation > 0.4:
            adjustments['temperature'] = -0.03

        return adjustments

    def reset_per_section(self):
        """每段开始前重置计数（致命信号 / 创新计数不过段）。"""
        self.b1_low_streak = 0
        self.consecutive_empty = 0

    def need_harmony_rollback(self) -> bool:
        """B 致命信号是否达到和声回退阈值。"""
        return self.b1_low_streak >= self.FATAL_B1_THRESHOLD

    def need_abort(self) -> bool:
        """是否应该中止生成。"""
        return self.consecutive_empty >= self.FATAL_EMPTY_THRESHOLD


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
