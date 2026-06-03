"""B 决策层 — 硬约束 + 温区退火 + 创新预算 + 发展配方。

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

    # ── DurSat 时值饱和度违规 ──

    duration_overflow: set[int] = field(default_factory=set)
        # 会导致 cum_dur + dur > grid_size 的 Duration token
    note_on_banned: bool = False
        # True = 该声部剩余空间连 Dur 1 都放不下，禁止所有 Note_ON
    bar_early: bool = False
        # True = 小节活跃声部未满，禁止 Bar token

    # ── v0.3.2: VoicePlan 兜底 ──
    inactive_voice_tokens: set[int] = field(default_factory=set)
        # VoicePlan 关闭的声部 token，模型永远不能采样

    def ban_inactive_voices(self, active_voices: list[int], tokenizer):
        """v0.3.2: 禁止模型采样 VoicePlan 已关闭的声部 token。

        A1 不为 inactive 声部预插框架 token，但模型可能自己尝试采样
        <Voice N>。此方法作为 B1 硬兜底。
        """
        self.inactive_voice_tokens.clear()
        for v in range(4):
            if v not in active_voices:
                tid = tokenizer.encode_token(f'{tokenizer.VOICE} {v}>')
                self.inactive_voice_tokens.add(tid)

    def ban_context_tokens(self, tokenizer):
        """v0.3.2: 框架-内容分离后，此方法为空操作。

        框架 token (Program/Tempo/TimeSig/Bar/Tonic/Clef/Position/Voice 等)
        由 A1 预插入，模型永不采样，无需禁令。
        仅保留 ~6 条 B1 硬约束规则 (音域/平行/交错/跳跃/DurSat Rule1+2)。
        """
        # v0.3.2: no-op — framework tokens are pre-inserted by A1
        pass

    def merge_all(self) -> set[int]:
        """合并所有违规 token 为统一集合 → 传给 generate_step。"""
        all_banned = set(self.banned_tokens)
        for attr in ('parallel_fifths', 'parallel_octaves', 'voice_crossing',
                     'out_of_range', 'extreme_leap',
                     'unresolved_leading_tone', 'contour_violation',
                     'duration_overflow', 'inactive_voice_tokens'):
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
        self.duration_overflow.clear()
        self.inactive_voice_tokens.clear()
        self.note_on_banned = False
        self.bar_early = False

    # ── DurSat 硬约束 ──────────────────────────────────────

    def ban_overflow_durations(self, voice_id: int, cur_pos: int,
                               tokenizer, grid_size: int = 16,
                               cum_dur: list[int] | None = None):
        """Rule 1: 禁用会导致 cum_dur + dur > grid_size 的 Duration 值。

        Args:
            voice_id: 当前声部索引 (0-3)
            cur_pos: 当前 Position 值
            tokenizer: REMITokenizer 实例（用于获取 Duration token ID）
            grid_size: 小节格位数（默认 16）
            cum_dur: 四声部累计时长列表，voice_id 为 None 时忽略（向后兼容）
        """
        self.duration_overflow.clear()
        remaining = grid_size - (cum_dur[voice_id] if cum_dur else 0)
        for dur_val in range(1, grid_size + 1):
            if cur_pos + dur_val > grid_size:
                dur_tid = tokenizer.encode_token(f'<Duration {dur_val}>')
                self.duration_overflow.add(dur_tid)

    def ban_note_on_if_full(self, voice_id: int, cur_pos: int,
                            tokenizer, grid_size: int = 16,
                            cum_dur: list[int] | None = None) -> bool:
        """Rule 2: 当前剩余空间连 Duration 1 都放不下时，禁止该声部所有 Note_ON。

        Returns:
            True 如果 Note_ON 应被全面禁止
        """
        remaining = grid_size - (cum_dur[voice_id] if cum_dur else 0)
        if cur_pos + 1 > grid_size or remaining <= 0:
            self.note_on_banned = True
            return True
        self.note_on_banned = False
        return False

    def ban_bar_if_not_full(self, active_voices: set[int],
                            cum_dur: list[int] | None = None,
                            grid_size: int = 16) -> bool:
        """Rule 3: 小节活跃声部未满时禁止 Bar token。

        只检查当前 bar 内实际出现过的声部（active_voices）。
        从未发声的声部不参与检查——否则单旋律/两声部会永久卡住。

        Returns:
            True 如果 Bar token 应被禁止
        """
        if not active_voices:
            self.bar_early = False
            return False  # 空 bar — 不禁止，由 C 层事后检测
        if cum_dur is None:
            self.bar_early = False
            return False
        all_active_full = all(
            cum_dur[v] >= grid_size - 1 for v in active_voices
        )  # tolerance=1
        if not all_active_full:
            self.bar_early = True
            return True
        self.bar_early = False
        return False

    def get_note_on_banned_ids(self, tokenizer) -> set[int]:
        """获取所有 Note_ON token ID（当 note_on_banned=True 时使用）。"""
        banned = set()
        for interval in range(-60, 61):
            banned.add(tokenizer.encode_token(f'<Note_ON {interval}>'))
        return banned

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

    # ── v0.3.2: 乐句追踪 ──
    phrase_state: object | None = None   # PhraseState 实例 (避免循环导入)

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

        条件: 密度偏离基线（density 不在正常范围）
               + 硬约束通过（b1_score > 0.3 作为 proxy）
               + 创新预算还有剩余
        """
        # 用 density 偏离度作为 surprise proxy: 显著偏离基线 → 可能是有意创新
        density = getattr(bar_stats, 'density', 0.0)
        rest_ratio = getattr(bar_stats, 'rest_ratio', 0.0)

        # 密度过高(>12)或过低(<2) → 偏离; 休止过多(>0.4) → 不可能是创新
        if rest_ratio > 0.4:
            return False
        is_deviating = (density > 12.0 or density < 2.0)

        # 硬约束通过（b1_score 不能太差）
        coherence_ok = b1_score is None or b1_score > 0.3

        if not coherence_ok:
            return False
        if not is_deviating:
            return False  # 没偏离就不需要创新预算

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


def _compute_deviation_surprise(
    bar_stats,
    baselines: dict,
) -> float:
    """创新判定：用 A3 统计量近似"意外度"。

    意外度 ≈ KL(token_type_dist || baseline) / max_kl
    纯统计计算，不需要模型 forward。
    """
    from .motif import _kl_divergence

    pc_dist = bar_stats.pitch_class_dist
    baseline_pc = baselines.get('pitch_class_dist')
    if not pc_dist or not baseline_pc or not isinstance(baseline_pc, list):
        return 0.0

    return _kl_divergence(pc_dist, baseline_pc)
