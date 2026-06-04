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


# ═══════════════════════════════════════════════════════════════
#  主题发展引擎 (v0.3.3-opt1)
# ═══════════════════════════════════════════════════════════════

import random as _random

@dataclass
class DevelopmentAction:
    """一次主题发展动作。

    由 B2 的 select_development_action() 产生，
    告诉采样循环：当前 bar 该用什么变形、移调多少、哪个声部。
    """
    transform: str | None = None
        # 'retrograde' / 'inversion' / 'augmentation' / 'diminution' /
        # 'retrograde_inversion' / 'fragment' / 'sequence' /
        # 'interval_expand' / 'rhythmic_vary' / None (=原形)
    transpose: int = 0                 # 半音移调量
    voice: int = 0                     # 目标声部 0-3
    guidance_strength: float = 0.5     # 引导强度 [0, 1]


# ── 发展策略表 ────────────────────────────────────────────────

DEVELOPMENT_STRATEGIES: dict[str, dict] = {
    'statement': {
        'transform': None,
        'transpose': 0,
        'guidance_strength': 0.6,
        'voice': 0,
    },
    'restatement': {
        'transform': None,
        'transpose': 7,                # 属调重申
        'guidance_strength': 0.4,
        'voice': 0,
    },
    'development': {
        'transform_weights': {
            'fragment': 0.25,
            'sequence': 0.25,
            'inversion': 0.15,
            'diminution': 0.15,
            'interval_expand': 0.1,
            'retrograde': 0.1,
        },
        'transpose_range': (-5, 7),
        'guidance_strength': 0.35,
    },
    'climax': {
        'transform': 'augmentation',
        'transpose': 0,
        'guidance_strength': 0.5,
    },
    'closing': {
        'transform': 'fragment',
        'transpose': 0,
        'guidance_strength': 0.15,
    },
}

# 乐句类型 → 策略映射
PHRASE_TO_STRATEGY = {
    'antecedent': 'statement',
    'consequent': 'restatement',
    'extension': 'development',
    'closing': 'closing',
    'transition': 'development',
}


def select_development_action(
    section_type: str,
    phrase_type: str | None = None,
    bar_in_section: int = 0,
    total_bars: int = 32,
) -> DevelopmentAction:
    """根据段落类型和位置选择发展策略。

    Args:
        section_type: 段落类型 ('exposition'/'development'/'recapitulation'/...)
        phrase_type: 乐句类型 ('antecedent'/'consequent'/'extension'/'closing'/...)
        bar_in_section: 段内 bar 位置
        total_bars: 段总 bar 数

    Returns:
        DevelopmentAction
    """
    # 1. 乐句类型优先
    if phrase_type and phrase_type in PHRASE_TO_STRATEGY:
        strategy_key = PHRASE_TO_STRATEGY[phrase_type]
    else:
        # 2. 按段内位置推断
        progress = bar_in_section / max(1, total_bars)
        if progress < 0.1:
            strategy_key = 'statement'
        elif progress < 0.75:
            # 发展部用 development 策略，其他段用 restatement
            strategy_key = 'development' if section_type == 'development' else 'restatement'
        elif progress < 0.9:
            strategy_key = 'climax' if section_type == 'development' else 'closing'
        else:
            strategy_key = 'closing'

    strat = DEVELOPMENT_STRATEGIES[strategy_key]

    # 确定变形算子
    transform = strat.get('transform')
    if transform is None and 'transform_weights' in strat:
        weights = strat['transform_weights']
        transform = _random.choices(list(weights.keys()), weights=list(weights.values()), k=1)[0]

    # 确定移调量
    transpose = strat.get('transpose', 0)
    if isinstance(transpose, tuple):
        lo, hi = transpose
        transpose = _random.randint(lo, hi)

    return DevelopmentAction(
        transform=transform,
        transpose=transpose,
        voice=strat.get('voice', 0),
        guidance_strength=strat.get('guidance_strength', 0.4),
    )


def apply_motif_guidance(
    target_tokens: list[int],
    logits,                        # torch.Tensor or list
    guidance_strength: float = 0.5,
    note_on_range: tuple[int, int] | None = None,
) -> None:
    """为 logits 添加动机引导偏置（原地修改）。

    不是强制输出目标 token，而是提高目标 token 及其邻域的概率。
    引导强度控制「写动机」vs「自由发挥」的平衡。

    Args:
        target_tokens: 从 MotifDNA 渲染的目标 token ID 序列
        logits: 当前 step 的 logits (torch.Tensor, shape [vocab_size] or [1, vocab_size])
        guidance_strength: 引导强度 [0, 1]，段首高、段末低
        note_on_range: Note_ON token 的 ID 范围 (min_id, max_id)，
                       用于邻域扩展。为 None 时不做邻域扩展。
    """
    if not target_tokens or guidance_strength <= 0:
        return

    try:
        import torch
        is_torch = isinstance(logits, torch.Tensor)
    except ImportError:
        is_torch = False

    # 取当前该匹配到的目标位置 (按生成步数推进)
    # 调用者负责维护 position 指针; 这里对单 token 做偏置
    target_id = target_tokens[0]  # 第一个目标 token

    boost = guidance_strength * 5.0

    if is_torch:
        logits_flat = logits.view(-1)
        vocab_size = logits_flat.shape[0]

        # 提高目标 token
        if 0 <= target_id < vocab_size:
            logits_flat[target_id] += boost

        # 邻域扩展: 相邻音高也有小幅 boost
        if note_on_range is not None:
            lo, hi = note_on_range
            neighbor_boost = guidance_strength * 2.0
            for offset in [-2, -1, 1, 2]:
                neighbor = target_id + offset
                if lo <= neighbor <= hi:
                    logits_flat[neighbor] += neighbor_boost
    else:
        # plain list
        if 0 <= target_id < len(logits):
            logits[target_id] += boost


def build_note_on_range(tokenizer) -> tuple[int, int]:
    """构建 Note_ON token 的 ID 范围，供邻域扩展使用。"""
    lo = tokenizer.encode_token('<Note_ON -60>')
    hi = tokenizer.encode_token('<Note_ON 60>')
    return (lo, hi)


# ═══════════════════════════════════════════════════════════════
#  长程张力曲线参数联动 (v0.3.3-opt3)
# ═══════════════════════════════════════════════════════════════

@dataclass
class DramaticParams:
    """B2 沿 DramaticCurve 动态调参的全部输出。

    tension:     当前目标紧张度 [0, 1]
    derivative:  紧张度变化方向 (正=爬升, 负=回落)
    temperature: 采样温度
    target_density: 目标音符密度 (notes/bar)
    dissonance_tolerance: 不协和容忍度
    register_range: (min_octave_offset, max_octave_offset)
    rest_penalty: 休止符惩罚
    ssf_constraint: SSF 约束强度
    innovation_budget: 创新预算
    cadence_strength: 终止式期待强度
    """
    tension: float = 0.5
    derivative: float = 0.0
    temperature: float = 1.0
    target_density: float = 8.0
    dissonance_tolerance: float = 0.2
    register_range: tuple[float, float] = (2.0, 6.0)
    rest_penalty: float = 1.0
    ssf_constraint: float = 0.5
    innovation_budget: float = 0.2
    cadence_strength: float = 0.5


def apply_dramatic_params(
    tension: float,
    derivative: float,
    base_temperature: float = 1.0,
) -> DramaticParams:
    """沿 DramaticCurve 连续调参，替代段级温区退火。

    Args:
        tension: 当前目标紧张度 [0, 1], 来自 DramaticCurve.get_tension()
        derivative: 紧张度变化率, 来自 DramaticCurve.get_derivative()
        base_temperature: 基础温度

    Returns:
        DramaticParams 完整参数集
    """
    t = max(0.0, min(1.0, tension))
    d = max(-0.5, min(0.5, derivative))

    return DramaticParams(
        tension=t,
        derivative=d,
        temperature=base_temperature * (0.7 + t * 0.6),
        target_density=3.0 + t * 12.0,
        dissonance_tolerance=0.05 + t * 0.5,
        register_range=(2.0 - t * 1.5, 5.0 + t * 1.5),
        rest_penalty=2.0 - t * 1.8,
        ssf_constraint=0.8 - t * 0.5,
        innovation_budget=0.1 + max(0.0, d) * 0.4,
        cadence_strength=0.3 + max(0.0, -d) * 0.6 if d < -0.03 else 0.3,
    )


# ═══════════════════════════════════════════════════════════════
#  对位意识 ContourBias (v0.3.3-opt4)
# ═══════════════════════════════════════════════════════════════

class ContourBias:
    """对位方向偏置: 鼓励反向进行 + 不完全→完全协和解决。

    在每 step 采样前调用 compute_bias()，修改 logits 偏置 Note_ON token。
    轻量级: 无参数、无 embedding，纯规则。

    用法:
        cb = ContourBias(tokenizer)
        bias = cb.compute_bias(voice=0, prev_notes={0: 7, 1: 4, 2: 0, 3: -5},
                               prev_prev_notes={}, active_voices=[0,1,2,3])
        for tid, val in bias.items():
            logits[tid] += val
    """

    def __init__(self, tokenizer):
        self.tk = tokenizer
        # 预计算 Note_ON token ID → interval 映射
        self._interval_of: dict[int, int] = {}
        for token_str, token_id in tokenizer._token_to_id.items():
            if token_str.startswith('<Note_ON '):
                try:
                    self._interval_of[token_id] = int(
                        token_str.split()[1].rstrip('>'))
                except (ValueError, IndexError):
                    pass
        self._note_ids = sorted(self._interval_of.keys())

    def compute_bias(
        self,
        current_voice: int,
        prev_notes: dict[int, int],         # {voice: prev_note_interval}
        prev_prev_notes: dict[int, int],    # {voice: prev_prev_note_interval}
        active_voices: list[int],
    ) -> dict[int, float]:
        """计算当前 step 每个 Note_ON token 的对位偏置。

        Returns:
            {token_id: bias_value} 加到 logits 上的偏置
        """
        bias: dict[int, float] = {}
        my_prev = prev_notes.get(current_voice)
        if my_prev is None:
            return bias

        for other_voice in active_voices:
            if other_voice == current_voice:
                continue
            prev_other = prev_notes.get(other_voice)
            if prev_other is None:
                continue
            prev_prev_other = prev_prev_notes.get(other_voice)

            for tid in self._note_ids:
                candidate = self._interval_of[tid]
                my_move = candidate - my_prev
                other_move = self._estimate_other_move(
                    other_voice, prev_other, prev_prev_other)

                b = 0.0
                b += self._contrary_bonus(my_move, other_move)
                b += self._parallel_penalty(
                    my_move, other_move, candidate, my_prev, prev_other)
                b += self._resolution_bonus(
                    candidate, my_prev, prev_other, other_move)

                if b != 0.0:
                    bias[tid] = bias.get(tid, 0.0) + b

        return bias

    @staticmethod
    def _estimate_other_move(voice: int, prev_interval: int,
                             prev_prev_interval: int | None) -> int:
        """估计另一声部当前 step 的移动方向。

        使用 prev_prev → prev 的趋势推算 (假设方向持续)。
        无法推算时返回 0 (假设静止)。
        """
        if prev_prev_interval is not None:
            return prev_interval - prev_prev_interval
        return 0

    @staticmethod
    def _contrary_bonus(my_move: int, other_move: int) -> float:
        """反向进行 +0.2, 严格平行 -0.15, 同向 -0.05。"""
        if my_move == 0 or other_move == 0:
            return 0.0
        my_dir = 1 if my_move > 0 else -1
        other_dir = 1 if other_move > 0 else -1
        if my_dir != other_dir:
            return 0.20
        elif my_move == other_move:
            return -0.15
        else:
            return -0.05

    @staticmethod
    def _parallel_penalty(
        my_move: int, other_move: int,
        candidate_interval: int, prev_my_interval: int,
        prev_other_interval: int,
    ) -> float:
        """平行五度/八度: 更强惩罚 -1.0。

        cur_vert = |candidate - (prev_other + other_move)| % 12
        prev_vert = |prev_my - prev_other| % 12
        """
        if my_move == 0 or other_move == 0:
            return 0.0
        if my_move != other_move:
            return 0.0

        new_other = prev_other_interval + other_move
        cur = abs(candidate_interval - new_other) % 12
        prev = abs(prev_my_interval - prev_other_interval) % 12

        if cur == 7 and prev == 7:
            return -1.0
        if cur == 0 and prev == 0:
            return -1.0
        return 0.0

    @staticmethod
    def _resolution_bonus(
        candidate_interval: int,
        prev_my_interval: int,
        prev_other_interval: int,
        other_move: int,
    ) -> float:
        """不完全协和→完全协和解决: +0.15, 半音解决额外 +0.10。"""
        new_other = prev_other_interval + other_move
        prev_vert = abs(prev_my_interval - prev_other_interval) % 12
        cur_vert = abs(candidate_interval - new_other) % 12

        prev_imperfect = prev_vert in (3, 4, 8, 9)
        cur_perfect = cur_vert in (0, 5, 7)

        if prev_imperfect and cur_perfect:
            # 半音解决 (导音→主音): 两声部音程差 = 1 或 2 或 11 半音
            semitone_res = abs(candidate_interval - new_other) % 12 in (1, 2, 11)
            return 0.15 + (0.10 if semitone_res else 0.0)
        return 0.0


# ═══════════════════════════════════════════════════════════════
#  情感色彩 B2 参数联动 (v0.3.3-opt5)
# ═══════════════════════════════════════════════════════════════

AFFECT_PARAM_MAP: dict[str, dict[str, callable]] = {
    'brightness': {
        'major_key_bias':       lambda b: +0.4 * (b - 0.5),
    },
    'tension': {
        'dissonance_tolerance':  lambda t: t,
        'temperature':           lambda t: 0.8 + 0.4 * t,
    },
    'stability': {
        'tonic_field_strength':  lambda s: 0.3 + 0.7 * s,
        'modulation_allowance':  lambda s: 1.0 - s,
    },
    'energy': {
        'density_ceiling':       lambda e: 4.0 + 12.0 * e,
        'rest_penalty':          lambda e: 2.0 - 1.5 * e,
    },
    'warmth': {
        'register_bias_low':     lambda w: +0.5 * (w - 0.5),
        'register_bias_high':    lambda w: -0.5 * (w - 0.5),
    },
    'depth': {
        'harmonic_rhythm':       lambda d: 0.5 + d * 2.0,
    },
    'motion': {
        'step_bonus':            lambda m: +0.3 * (m - 0.5),
        'leap_penalty':          lambda m: -0.2 * (m - 0.5),
    },
    'closure': {
        'cadence_strength':      lambda c: c,
    },
}


@dataclass
class AffectBias:
    """B2 情感偏置结果: 8 维映射到具体参数值。

    由 apply_affect_bias() 产生, 采样循环读取对应字段。
    """
    major_key_bias: float = 0.0
    dissonance_tolerance: float = 0.2
    temperature: float = 1.0
    tonic_field_strength: float = 0.65
    modulation_allowance: float = 0.5
    density_ceiling: float = 10.0
    rest_penalty: float = 1.25
    register_bias_low: float = 0.0
    register_bias_high: float = 0.0
    harmonic_rhythm: float = 1.0
    step_bonus: float = 0.0
    leap_penalty: float = 0.0
    cadence_strength: float = 0.5


def apply_affect_bias(target: 'AffectVector') -> AffectBias:
    """将目标八维向量映射为 B2 可用的具体参数偏置。

    Args:
        target: AffectVector, 来自 A1 情绪解析器或用户输入

    Returns:
        AffectBias, 每个字段可直接用于采样循环调参
    """
    result = AffectBias()
    for dim_name, param_map in AFFECT_PARAM_MAP.items():
        dim_val = getattr(target, dim_name, 0.5)
        for param, transform in param_map.items():
            if hasattr(result, param):
                setattr(result, param, transform(dim_val))
    return result
