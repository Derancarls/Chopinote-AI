"""A/B/C 三阶段反馈控制器。

A: 生成前 seed 评估 → SeedProfile + GenerationParams
B: 生成中实时反馈 → 每小节调整参数（B1 局部流畅 + B2 全局漂移）
C: 生成后全量评价 → 低分重试 + RL reward
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from chopinote_model.generate import SeedProfile, GenerationParams

from chopinote_evaluator.registry import (
    REGISTRY, Phase,
    get_metrics, get_metric_names,
    _tokens_by_bar,
    _note_on_intervals,
    _pitch_class_dist,
    _velocity_list,
    _count_token_type,
    kl_divergence,
    _pitch_class_kl_tokens, _interval_kl_tokens,
    _density_z_tokens, _rest_ratio_tokens,
    _velocity_consistency_tokens, _dissonance_ratio_tokens,
    _syncopation_ratio_tokens, _duration_entropy_tokens,
    _register_span_tokens, _melodic_direction_tokens,
    _interval_shift_tokens, _key_consistency_tokens,
    _empty_measure_tokens, _pitch_range_tokens,
    _unison_chain_tokens, _rest_streak_tokens,
    _mono_rhythm_tokens, _extreme_density_tokens,
    _max_polyphony_per_position_tokens,
    _bar_boundary_melody_tokens, _parallel_fifths_tokens,
    _melodic_contour_tokens, _token_type_kl_tokens,
    progression_validity_tokens, harmonic_rhythm_score_tokens,
    cadence_quality_tokens, chord_melody_alignment_tokens,
    _extract_chord_sequence,
)

logger = logging.getLogger(__name__)


@dataclass
class BPhaseResult:
    """B 阶段每小节输出。"""
    bar: int
    b1_score: float
    b2_score: float
    combined_score: float
    b1_details: dict[str, float]
    b2_details: dict[str, float]
    adjustments_applied: dict[str, float]


@dataclass
class CPhaseResult:
    """C 阶段输出。"""
    report: Any           # EvaluationReport
    retry_count: int
    final: bool
    adjustments: dict[str, Any] | None = None


# ── A 阶段数据结构 ──────────────────────────────────


@dataclass
class SeedSection:
    """seed 内部的段落信息。"""
    type: str
    start_bar: int
    n_bars: int
    key: str = 'C'


@dataclass
class SectionStyleTarget:
    """某段落类型的目标参数区间。"""
    temperature: tuple[float, float, float] = (1.0, 0.7, 1.4)
    key_bias_strength: tuple[float, float, float] = (2.0, 0.5, 3.0)
    complexity: tuple[float, float, float] = (5.0, 1.0, 10.0)
    density_target: float = 8.0
    rest_ratio_target: tuple[float, float] = (0.05, 0.25)
    harmonic_rhythm: float = 16.0
    b1_threshold_mult: float = 1.0
    b2_threshold_mult: float = 1.0
    confidence: str = 'high'

    @classmethod
    def from_seed_section(cls, bar_tokens_list: list[list[int]], tokenizer,
                          tonic_midi: int = 60) -> 'SectionStyleTarget':
        """从 seed 中某段落类型的 token 统计出风格目标。

        Args:
            bar_tokens_list: 该段落类型的所有小节的 token 列表的列表
            tokenizer: REMI tokenizer
            tonic_midi: 主音 MIDI 编号

        Returns:
            SectionStyleTarget with confidence='high' (从真实数据推导)
        """
        if not bar_tokens_list:
            return cls(confidence='low')

        # 合并所有 token
        all_tokens = [t for bar in bar_tokens_list for t in bar]
        n_bars = len(bar_tokens_list)

        # 密度统计
        densities = []
        for bar_tokens in bar_tokens_list:
            nn = sum(1 for t in bar_tokens
                     if tokenizer.decode_token(t).startswith('<Note_ON'))
            densities.append(nn)
        density_mean = sum(densities) / max(len(densities), 1)

        # 休止比例
        rest_count = sum(1 for t in all_tokens
                        if tokenizer.decode_token(t) == '<Rest>')
        note_count = sum(1 for t in all_tokens
                        if tokenizer.decode_token(t).startswith('<Note_ON'))
        rest_ratio = rest_count / max(rest_count + note_count, 1)

        # 复杂度：基于短时值比例 + 密度 + 演奏法多样性
        short_durs = sum(1 for t in all_tokens
                        if tokenizer.decode_token(t).startswith('<Duration')
                        and int(tokenizer.decode_token(t).split(' ')[1].rstrip('>')) <= 4)
        total_durs = sum(1 for t in all_tokens
                        if tokenizer.decode_token(t).startswith('<Duration'))
        short_ratio = short_durs / max(total_durs, 1)
        complexity_raw = density_mean * 0.5 + short_ratio * 10.0 * 0.5
        complexity_val = max(1.0, min(10.0, complexity_raw))

        # 力度均值 → temperature target（力度高 ≈ 情绪激烈 ≈ 温度偏高）
        velocities = [int(tokenizer.decode_token(t).split(' ')[1].rstrip('>'))
                     for t in all_tokens
                     if tokenizer.decode_token(t).startswith('<Velocity')]
        vel_mean = sum(velocities) / max(len(velocities), 1) if velocities else 4
        temp_target = max(0.7, min(1.4, vel_mean / 4.0 * 1.0))

        # 和声节奏：chord change 频率
        chords = _extract_chord_sequence(all_tokens, tokenizer)
        changes = sum(1 for i in range(1, len(chords)) if chords[i] != chords[i - 1])
        harm_rhythm = len(all_tokens) / max(changes, 1)

        # b1/b2 阈值乘数：seed 段落通常用 1.0（正常约束）
        b1_mult = 1.0
        b2_mult = 1.0

        return cls(
            temperature=(temp_target, max(0.5, temp_target - 0.3), min(2.0, temp_target + 0.3)),
            key_bias_strength=(2.0, 0.5, 3.0),
            complexity=(complexity_val, max(1.0, complexity_val - 3.0), min(10.0, complexity_val + 2.0)),
            density_target=density_mean,
            rest_ratio_target=(max(0.01, rest_ratio - 0.1), min(0.5, rest_ratio + 0.1)),
            harmonic_rhythm=max(4.0, harm_rhythm),
            b1_threshold_mult=b1_mult,
            b2_threshold_mult=b2_mult,
            confidence='high',
        )


@dataclass
class SectionStyleProfile:
    """从 seed 中提取的各段落类型风格目标。"""
    styles: dict[str, SectionStyleTarget] = field(default_factory=dict)
    fallback: SectionStyleTarget = field(default_factory=SectionStyleTarget)


@dataclass
class HarmonyContext:
    """A 阶段提取的和声上下文。"""
    chord_density_per_bar: list[float] = field(default_factory=list)
    final_cadence: list[str] = field(default_factory=list)
    cadence_patterns: dict[str, int] = field(default_factory=dict)
    harmonic_complexity: float = 0.5
    chord_density_per_bar_mean: float | None = None
    seed_contour: list[float] = field(default_factory=list)


@dataclass
class SectionPlan:
    """A 阶段输出的段落排期。"""
    type: str
    start_bar: int
    n_bars: int
    key: str = 'C'
    min_bars: int = 4
    max_bars: int = 32


@dataclass
class SeedBlueprint:
    """A 阶段输出：seed 的完整结构化蓝图。"""
    profile: SeedProfile | None = None
    sections: list[SeedSection] = field(default_factory=list)
    form_hint: str = 'through_composed'
    style_profile: SectionStyleProfile = field(default_factory=SectionStyleProfile)
    harmony: HarmonyContext = field(default_factory=HarmonyContext)
    section_plan: list[SectionPlan] = field(default_factory=list)


@dataclass
class BarDiagnosis:
    """C 阶段逐小节诊断。"""
    bar: int
    issues: list[str] = field(default_factory=list)
    severity: float = 0.0
    suggestion: str | None = None


# ── A 阶段：生成前评估 ──────────────────────────────────


class PreGenerationEvaluator:
    """A 阶段：解析 seed token 序列，提取 SeedProfile + 设定 GenerationParams。"""

    def __init__(self, tokenizer, default_complexity: float | None = None):
        self.tokenizer = tokenizer
        self.default_complexity = default_complexity

    def evaluate(self, seed_tokens: list[int]) -> tuple[SeedProfile, GenerationParams]:
        """从 seed tokens 提取结构画像，返回 SeedProfile + GenerationParams。"""
        bar_id = self.tokenizer.bar_token_id
        tokenizer = self.tokenizer

        # ── 基本信息 ────────────────────────────────────────
        all_bars = _tokens_by_bar(seed_tokens, bar_id)
        # all_bars[0] 是第一个 BAR 之前的头信息（BOS/Key/TimeSig/Tempo），不计入小节
        real_bars = all_bars[1:] if len(all_bars) > 1 else []
        # 去掉末尾空小节（最后一个 BAR token 产生）
        while real_bars and not any(t for t in real_bars[-1]):
            real_bars.pop()
        n_bars = len(real_bars)

        # 调性/拍号/速度（从序列开头提取）
        tonic_key = None
        tonic_midi = 60
        time_sig = '4/4'
        tempo = 120
        for tid in seed_tokens[:80]:
            s = tokenizer.decode_token(tid)
            if s.startswith('<Key') and tonic_key is None:
                tonic_key = s[len('<Key') + 1:-1]
            elif s.startswith('<TimeSig') and time_sig == '4/4':
                time_sig = s[len('<TimeSig') + 1:-1]
            elif s.startswith('<Tempo'):
                try:
                    tempo = int(s[len('<Tempo') + 1:-1])
                except ValueError:
                    pass

        # tonic_key → tonic_midi
        if tonic_key:
            try:
                from chopinote_dataset.tokenizer import key_name_to_tonic_midi
                tonic_midi = key_name_to_tonic_midi(tonic_key)
            except Exception:
                pass

        key_pitch_classes = None
        if tonic_key:
            try:
                from chopinote_model.generate import KEY_TO_DIATONIC_PITCHES
                key_pitch_classes = KEY_TO_DIATONIC_PITCHES.get(tonic_key)
            except Exception:
                pass

        # ── 声部/乐器 ────────────────────────────────────
        programs = []
        for tid in seed_tokens:
            s = tokenizer.decode_token(tid)
            if s.startswith('<Program'):
                val = s[len('<Program') + 1:-1]
                parts = val.split('_')
                prog = int(parts[0])
                sub = int(parts[1]) if len(parts) > 1 else 0
                pair = (prog, sub)
                if pair not in programs:
                    programs.append(pair)
        voice_count = len(programs)

        # ── 逐节密度 ────────────────────────────────────
        density_series = []
        for bar_tokens in real_bars:
            nn = _count_token_type(bar_tokens, tokenizer, '<Note_ON')
            density_series.append(nn)
        bar_density = sum(density_series) / max(len(density_series), 1)

        # ── 分布 ────────────────────────────────────────
        pitch_class_dist = _pitch_class_dist(seed_tokens, tokenizer, tonic_midi)

        intervals = _note_on_intervals(seed_tokens, tokenizer)
        interval_counts = [0] * 25
        for iv in intervals:
            interval_counts[min(abs(iv), 24)] += 1
        total_iv = sum(interval_counts)
        interval_dist = [c / max(total_iv, 1) for c in interval_counts]

        # ── 力度 ────────────────────────────────────────
        velocities = _velocity_list(seed_tokens, tokenizer)
        velocity_mean = sum(velocities) / max(len(velocities), 1)

        # ── 休止比例 ────────────────────────────────────
        rest_count = _count_token_type(seed_tokens, tokenizer, '<Rest')
        note_count = _count_token_type(seed_tokens, tokenizer, '<Note_ON')
        rest_ratio = rest_count / max(note_count + rest_count, 1)

        # ── 合法性检查（A 阶段） ────────────────────────────
        # 只检查空小节的空段和 tuplet/tie 配对
        tuplet_ok = True
        if '<TupletStart' in [tokenizer.decode_token(t) for t in seed_tokens]:
            start_count = _count_token_type(seed_tokens, tokenizer, '<TupletStart')
            end_count = _count_token_type(seed_tokens, tokenizer, '<TupletEnd')  # This won't work
            # Better: check by string
            start_c = sum(1 for t in seed_tokens if 'TupletStart' in tokenizer.decode_token(t))
            end_c = sum(1 for t in seed_tokens if 'TupletEnd' in tokenizer.decode_token(t))
            tuplet_ok = start_c <= end_c + 1  # last one may not be closed

        # ── SeedProfile ───────────────────────────────────
        profile = SeedProfile(
            n_bars=n_bars,
            bar_density=bar_density,
            tonic_key=tonic_key,
            tonic_midi=tonic_midi,
            key_pitch_classes=key_pitch_classes,
            tempo=tempo,
            programs=programs,
            pitch_class_dist=pitch_class_dist,
            interval_dist=interval_dist,
            velocity_mean=velocity_mean,
            rest_ratio=rest_ratio,
            density_series=density_series,
            voice_count=voice_count,
            time_sig=time_sig,
        )

        # ── GenerationParams ──────────────────────────────
        # 根据 seed 特性智能设定初值
        complexity = self.default_complexity
        if complexity is None:
            complexity = max(1.0, min(10.0, bar_density * 0.8))

        params = GenerationParams(
            complexity=complexity,
            lock_key=True,
            lock_time=True,
            lock_tempo=True,
            lock_program=True,
        )

        return profile, params

    def extract_blueprint(self, seed_tokens: list[int]) -> SeedBlueprint:
        """A 阶段核心：从 seed tokens 提取完整结构化蓝图。"""
        profile, params = self.evaluate(seed_tokens)
        blueprint = SeedBlueprint(profile=profile)

        # ── 和声上下文 ──────────────────────────────────────
        bar_id = self.tokenizer.bar_token_id
        bars = _tokens_by_bar(seed_tokens, bar_id)
        real_bars = bars[1:] if len(bars) > 1 else []
        while real_bars and not any(t for t in real_bars[-1]):
            real_bars.pop()

        chord_density_per_bar = []
        for bar_tokens in real_bars:
            chords = _extract_chord_sequence(bar_tokens, self.tokenizer)
            changes = sum(1 for i in range(1, len(chords)) if chords[i] != chords[i - 1])
            chord_density_per_bar.append(changes)

        final_cadence = []
        for bar_tokens in reversed(real_bars[-3:]):
            chords = _extract_chord_sequence(bar_tokens, self.tokenizer)
            if chords:
                final_cadence = chords[-2:]
                break

        # ── seed 旋律轮廓（用于 B2 melodic_contour 匹配） ──
        seed_contour = []
        for bar_tokens in real_bars:
            notes = [int(self.tokenizer.decode_token(t).split(' ')[1].rstrip('>'))
                     for t in bar_tokens
                     if self.tokenizer.decode_token(t).startswith('<Note_ON')]
            if notes:
                seed_contour.append(float(max(notes)))
            else:
                seed_contour.append(0.0)

        harmony = HarmonyContext(
            chord_density_per_bar=chord_density_per_bar,
            final_cadence=final_cadence,
            chord_density_per_bar_mean=sum(chord_density_per_bar) / max(len(chord_density_per_bar), 1) if chord_density_per_bar else 0.05,
            seed_contour=seed_contour,
        )
        blueprint.harmony = harmony

        # ── 段落检测（优先 6 信号 Viterbi，fallback 密度变化点） ──
        key = profile.tonic_key or 'C'
        sections = self._detect_sections_annotator(real_bars, key)
        if not sections and profile.density_series and len(profile.density_series) >= 4:
            sections = self._detect_sections_simple(profile.density_series, key)
        blueprint.sections = sections
        if sections:
            form = self._infer_form(sections)
            blueprint.form_hint = form
            # 构建 section_plan
            blueprint.section_plan = self._build_section_plan(sections, total_bars=max(32, profile.n_bars * 2))
            # 构建 style_profile
            blueprint.style_profile = self._build_style_profile(
                sections, real_bars=real_bars)

        return blueprint

    def _detect_sections_simple(self, density_series: list[float], key: str) -> list[SeedSection]:
        """简化的段落检测：基于密度变化点（fallback）。"""
        if len(density_series) < 4:
            return []
        mean_d = sum(density_series) / len(density_series)
        sections = []
        current_type = 'theme1'
        current_start = 0
        for i in range(1, len(density_series)):
            window_prev = density_series[max(0, i - 2):i]
            window_curr = density_series[i:min(len(density_series), i + 2)]
            avg_prev = sum(window_prev) / len(window_prev) if window_prev else mean_d
            avg_curr = sum(window_curr) / len(window_curr) if window_curr else mean_d
            if abs(avg_curr - avg_prev) > mean_d * 0.5 and i - current_start >= 4:
                sections.append(SeedSection(type=current_type, start_bar=current_start,
                                           n_bars=i - current_start, key=key))
                current_start = i
                current_type = 'development' if current_type == 'theme1' else 'theme2'
        sections.append(SeedSection(type=current_type, start_bar=current_start,
                                   n_bars=len(density_series) - current_start, key=key))
        return sections

    def _detect_sections_annotator(self, real_bars: list[list[int]],
                                    key: str) -> list[SeedSection]:
        """使用 structure_annotator 的 6 信号 Viterbi 方法检测段落。

        调用 structure_annotator 的 token 级接口（纯规则，零模型依赖）：
        extract_bar_features → 6 信号 → 加权融合 → Viterbi → infer_section_types。
        """
        try:
            from scripts.structure_annotator import (
                extract_bar_features, compute_key_change_signal,
                compute_density_shift_signal, compute_repeat_signal,
                compute_tempo_change_signal, compute_program_change_signal,
                compute_silence_gap_signal, viterbi_segmentation,
                infer_section_types, SIGNAL_WEIGHTS, MIN_SECTION_BARS,
                BOUNDARY_THRESHOLD, SECTION_TOKEN_NAMES,
            )
        except ImportError:
            return []

        if len(real_bars) < MIN_SECTION_BARS * 2:
            return []

        # 解码所有 token → 字符串
        all_token_strs = []
        for bar_tokens in real_bars:
            for t in bar_tokens:
                all_token_strs.append(self.tokenizer.decode_token(t))
            all_token_strs.append('<Bar>')

        bars = extract_bar_features(all_token_strs)
        if len(bars) < MIN_SECTION_BARS * 2:
            return []

        # 计算 6 信号
        n = len(bars)
        fused = sum(
            fn(bars) * SIGNAL_WEIGHTS[wn]
            for fn, wn in [
                (compute_key_change_signal, 'key_change'),
                (compute_density_shift_signal, 'density_shift'),
                (compute_repeat_signal, 'repeat'),
                (compute_tempo_change_signal, 'tempo_change'),
                (compute_program_change_signal, 'program_change'),
                (compute_silence_gap_signal, 'silence_gap'),
            ]
        )
        fused = fused.astype(float)
        total_w = sum(SIGNAL_WEIGHTS.values())
        if total_w > 0:
            fused = fused / total_w

        boundaries = viterbi_segmentation(fused.astype(float), MIN_SECTION_BARS)

        # 过滤低置信度边界
        for i in range(n):
            if boundaries[i] and fused[i] < BOUNDARY_THRESHOLD:
                boundaries[i] = 0

        if boundaries.sum() == 0:
            return []

        section_type_ids = infer_section_types(bars, boundaries)
        if not section_type_ids:
            return []

        # 边界索引 → SeedSection 列表
        boundary_list = [0] + [i for i in range(1, n) if boundaries[i]] + [n - 1]
        boundary_list = sorted(set(boundary_list))
        sections = []
        for s in range(len(boundary_list) - 1):
            start = boundary_list[s]
            end = boundary_list[s + 1]
            n_bars = max(1, end - start)
            type_id = section_type_ids[s] if s < len(section_type_ids) else 14
            type_name = SECTION_TOKEN_NAMES.get(type_id, 'theme1')
            sections.append(SeedSection(
                type=type_name, start_bar=start,
                n_bars=n_bars, key=key,
            ))
        return sections

    def _infer_form(self, sections: list[SeedSection]) -> str:
        """根据段落序列推断曲式。"""
        types = [s.type for s in sections]
        if not types:
            return 'through_composed'
        if 'theme1' in types and 'development' in types and types[0] == types[-1]:
            return 'ternary'
        if types.count('theme1') >= 2:
            return 'sonata'
        return 'through_composed'

    def _build_section_plan(self, sections: list[SeedSection], total_bars: int = 64) -> list[SectionPlan]:
        """根据 seed 段落构建生成段排期，按比例适配 total_bars。"""
        # 先算各段的原始期望长度
        raw_lengths = [max(4, min(32, int(sec.n_bars * 1.5))) for sec in sections]
        raw_total = sum(raw_lengths)
        # 按比例缩放到 total_bars
        if raw_total > 0:
            scale = total_bars / raw_total
        else:
            scale = 1.0

        plan = []
        bar_offset = 0
        for i, sec in enumerate(sections):
            n_bars = max(4, min(32, int(raw_lengths[i] * scale)))
            plan.append(SectionPlan(
                type=sec.type,
                start_bar=bar_offset,
                n_bars=n_bars,
                key=sec.key,
                min_bars=max(4, n_bars // 2),
                max_bars=min(32, n_bars * 2),
            ))
            bar_offset += n_bars
        return plan

    def _build_style_profile(self, sections: list[SeedSection],
                            real_bars: list[list[int]] | None = None) -> SectionStyleProfile:
        """从 seed 段落构建风格目标，优先从实际 token 统计推导。"""
        styles = {}
        for sec in sections:
            # 收集该段落对应的 bar tokens
            sec_bars = []
            if real_bars and sec.start_bar < len(real_bars):
                end = min(sec.start_bar + sec.n_bars, len(real_bars))
                sec_bars = real_bars[sec.start_bar:end]

            if sec_bars:
                styles[sec.type] = SectionStyleTarget.from_seed_section(
                    sec_bars, self.tokenizer,
                    tonic_midi=60,
                )
            else:
                styles[sec.type] = SectionStyleTarget(confidence='low')
        return SectionStyleProfile(styles=styles, fallback=SectionStyleTarget())


# ── B 阶段调整规则 ────────────────────────────────────

# B1（局部流畅）：指标评分 < 阈值时触发调整
# key = 指标名, value = {param: max_delta_per_call}
B1_ADJUSTMENT_RULES: dict[str, dict[str, float]] = {
    # 统计层
    'density_z': {'rest_penalty': 0.3, 'temperature': -0.05},
    'dissonance_ratio': {'temperature': -0.15},
    'velocity_consistency': {'temperature': -0.10},
    'rest_ratio': {'rest_penalty': 0.25},
    'register_span': {'complexity': -0.5},
    'duration_entropy': {'temperature': -0.10},
    'syncopation_ratio': {'temperature': -0.08},
    'melodic_direction': {'temperature': -0.08},
    'interval_shift': {'temperature': -0.05},
    'empty_measure': {'rest_penalty': 2.0, 'temperature': 0.15},
    'unison_chain': {'temperature': 0.2, 'complexity': 1.0},
    'rest_streak': {'rest_penalty': 3.0, 'temperature': 0.15},
    'mono_rhythm': {'complexity': 1.0},
    'extreme_density': {'complexity': 1.0, 'temperature': -0.1},
    'max_polyphony_per_position': {'complexity': -1.0, 'rest_penalty': 3.0},
    # 和声层
    'progression_validity': {'key_bias_strength': 1.0, 'temperature': -0.1},
    'harmonic_rhythm': {'complexity': -0.5},
    # 旋律层
    'bar_boundary_melody': {'temperature': -0.15, 'key_bias_strength': 0.5},
    'parallel_fifths': {'temperature': -0.15, 'key_bias_strength': 0.8},
}

# B2（全局漂移）
B2_ADJUSTMENT_RULES: dict[str, dict[str, float]] = {
    # 统计层
    'pitch_class_kl': {'key_bias_strength': 0.5},
    'interval_kl': {'key_bias_strength': 0.3},
    'density_z': {'rest_penalty': 0.3, 'temperature': -0.05},
    'rest_ratio': {'rest_penalty': 0.4},
    'register_span': {'temperature': -0.10},
    'velocity_consistency': {'temperature': -0.08},
    'key_consistency': {'key_bias_strength': 0.8},
    'token_type_kl': {'temperature': -0.10, 'rest_penalty': 0.2},
    'empty_measure': {'rest_penalty': 2.5, 'temperature': 0.2, 'complexity': 1.5},
    # 和声层
    'progression_validity': {'key_bias_strength': 1.0, 'temperature': -0.1},
    'harmonic_rhythm': {'complexity': -0.8, 'key_bias_strength': 0.5},
    'chord_melody_alignment': {'key_bias_strength': 0.8, 'complexity': -0.5},
    'cadence_quality': {'key_bias_strength': 1.5, 'temperature': -0.15},
    'cadence_placement': {'key_bias_strength': 1.2, 'temperature': -0.1},
    # 旋律层
    'melodic_contour': {'complexity': 1.0, 'key_bias_strength': 0.5},
}

B1_DEFAULT_THRESHOLD = 0.55
B2_DEFAULT_THRESHOLD = 0.50

# B2 趋势检测：连续 N 块下降 → 额外惩罚乘数
B2_TREND_WINDOW = 3
B2_TREND_PENALTY = 0.5  # 趋势下降时额外 delta 乘数

# 调整衰减半衰期（bar 数）
ADJUSTMENT_HALF_LIFE = 4

# B2 段落类型容忍度乘数
SECTION_B2_TOLERANCE: dict[str, float] = {
    'exposition': 1.0, 'theme1': 1.0, 'theme2': 0.9,
    'recapitulation': 0.8, 'development': 0.4, 'bridge': 0.5,
    'transition': 0.4, 'coda': 0.6, 'cadenza': 0.15,
    'intro': 0.7, 'variation': 0.6, 'episode': 0.3,
}


class NarrowFeedbackController:
    """B 阶段：生成中实时反馈。

    B1：只看最近 N 节自身流畅性，不依赖 seed。
    B2：每 S 节一块 vs seed 整体，检测漂移趋势。
    支持段落感知 + 和声/旋律指标 + 衰减 + 中段预检。
    """

    def __init__(
        self,
        seed_profile: SeedProfile,
        tokenizer,
        blueprint: SeedBlueprint | None = None,
        local_bars: int = 4,
        local_weight: float = 0.5,
        global_weight: float = 0.5,
        b1_threshold: float = B1_DEFAULT_THRESHOLD,
        b2_threshold: float = B2_DEFAULT_THRESHOLD,
        b2_extra_pull: float = 1.0,
    ):
        self.seed_profile = seed_profile
        self.tokenizer = tokenizer
        self.blueprint = blueprint
        self.local_bars = local_bars
        self.local_weight = local_weight
        self.global_weight = global_weight
        self.b1_threshold = b1_threshold
        self.b2_threshold = b2_threshold
        self.b2_extra_pull = b2_extra_pull

        # B1 滚动状态
        self._prev_window_tokens: list[int] | None = None
        self._prev_pitch_class_dist: list[float] | None = None
        self._prev_interval_dist: list[float] | None = None
        self._prev_bars_count = 0

        # B2 块状态
        self._b2_block_scores: list[float] = []
        self._last_b2_bar = 0

        # B2 窗口大小 = seed 小节数
        self._b2_block_size = max(1, seed_profile.n_bars)

        # 参数调整累加（用于衰减）
        self._cumulative_adjustments: dict[str, float] = {}

        # 段落感知状态
        self._current_section_type: str | None = None
        self._section_adjusted: set[int] = set()  # 已调整过的段落（防震荡）
        if blueprint and blueprint.section_plan:
            self._section_plan = blueprint.section_plan
            self._style_profile = blueprint.style_profile
        else:
            self._section_plan = []
            self._style_profile = SectionStyleProfile()

    def on_bar(
        self,
        full_tokens: list[int],
        bar_count: int,
        current_params: GenerationParams,
    ) -> dict[str, float]:
        """每小节完成后调用，返回参数调整量。"""
        adjustments: dict[str, float] = {}
        b1_details: dict[str, float] = {}
        b2_details: dict[str, float] = {}

        bar_id = self.tokenizer.bar_token_id
        tokenizer = self.tokenizer
        bars = _tokens_by_bar(full_tokens, bar_id)

        if len(bars) < 2:
            return {}

        # ── 段落感知：检测段落切换 ─────────────────────
        near_boundary = None
        if self._section_plan:
            prev_section = self._current_section_type
            self._current_section_type = self._get_current_section(bar_count)
            if prev_section != self._current_section_type and self._current_section_type:
                self._enter_section(self._current_section_type)
            near_boundary = self._near_boundary(bar_count)
            # 边界阈值调整
            if near_boundary == 'entering':
                self.b1_threshold = B1_DEFAULT_THRESHOLD * 1.2
            elif near_boundary == 'leaving':
                self.b1_threshold = B1_DEFAULT_THRESHOLD * 0.5
                self.b2_threshold = B2_DEFAULT_THRESHOLD * 0.5
            else:
                style_target = self._style_profile.styles.get(self._current_section_type)
                if style_target:
                    self.b1_threshold = B1_DEFAULT_THRESHOLD * style_target.b1_threshold_mult
                    self.b2_threshold = B2_DEFAULT_THRESHOLD * style_target.b2_threshold_mult

        # ── B1：局部流畅性 ──────────────────────────────
        b1_score, b1_details = self._compute_b1(bars, bar_count, tokenizer)
        if b1_details:
            b1_adj = self._b1_adjustments(b1_details, bar_count)
            for k, v in b1_adj.items():
                adjustments[k] = adjustments.get(k, 0.0) + v

        # ── B2：全局漂移 ────────────────────────────────
        b2_score, b2_details = self._compute_b2(bars, bar_count, tokenizer)
        if b2_details:
            b2_adj = self._b2_adjustments(b2_details, bar_count)
            for k, v in b2_adj.items():
                adjustments[k] = adjustments.get(k, 0.0) + v

        # ── B2 段落长度调整 ─────────────────────────────
        if self._section_plan and bar_count > 0:
            self._adjust_section_length(bar_count)

        # ── 衰减历史调整 ───────────────────────────────
        self._apply_decay()

        if not adjustments:
            return {}

        # ── 更新累计 ────────────────────────────────────
        for k, v in adjustments.items():
            self._cumulative_adjustments[k] = self._cumulative_adjustments.get(k, 0.0) + v

        # ── 日志（info 级别，结构化输出）───────────────
        combined = b1_score * self.local_weight + b2_score * self.global_weight
        self._log_bar_state(bar_count, b1_score, b2_score, combined,
                           adjustments, current_params, near_boundary)

        return adjustments

    def _log_bar_state(self, bar_count, b1_score, b2_score, combined,
                      adjustments, current_params, near_boundary):
        """结构化日志输出。"""
        total_bars = self.blueprint.section_plan[-1].start_bar + self.blueprint.section_plan[-1].n_bars if (self.blueprint and self.blueprint.section_plan) else '?'
        logger.info(
            "B|bar=%d/%s sec=%s near=%s | b1=%.3f b2=%.3f | "
            "ΔT=%+.2f Δrest=%+.1f Δkey=%+.1f | "
            "params: T=%.2f rest=%.1f key=%.1f cplx=%.1f",
            bar_count, total_bars, self._current_section_type or '-', near_boundary or '-',
            b1_score, b2_score,
            adjustments.get('temperature', 0), adjustments.get('rest_penalty', 0),
            adjustments.get('key_bias_strength', 0),
            current_params.temperature, current_params.rest_penalty,
            current_params.key_bias_strength, current_params.complexity,
        )

    def _compute_b1(
        self,
        bars: list[list[int]],
        bar_count: int,
        tokenizer,
    ) -> tuple[float, dict[str, float]]:
        """B1：最近 local_bars 节的局部流畅性。"""
        n = min(self.local_bars, len(bars) - 1)
        if n < 2:
            return 0.5, {}

        window = bars[-n:]  # 最近 n 节
        flat = [t for bar in window for t in bar]
        window_notes = _note_on_intervals(flat, tokenizer)

        scores: dict[str, float] = {}

        # 空小节检测：即使音符极少也必须运行
        try:
            scores['empty_measure'] = _empty_measure_tokens(flat, tokenizer)
        except Exception:
            pass

        if len(window_notes) < 3:
            if scores:
                avg = sum(scores.values()) / len(scores)
                return avg, scores
            return 0.5, {}

        try:
            scores['pitch_class_kl'] = _pitch_class_kl_tokens(
                flat, tokenizer, reference=self._prev_pitch_class_dist,
            )
        except Exception:
            pass

        try:
            scores['interval_kl'] = _interval_kl_tokens(
                flat, tokenizer, reference=self._prev_interval_dist,
            )
        except Exception:
            pass

        for name, fn in [
            ('density_z', _density_z_tokens),
            ('rest_ratio', _rest_ratio_tokens),
            ('velocity_consistency', _velocity_consistency_tokens),
            ('dissonance_ratio', _dissonance_ratio_tokens),
            ('syncopation_ratio', _syncopation_ratio_tokens),
            ('duration_entropy', _duration_entropy_tokens),
            ('register_span', _register_span_tokens),
            ('melodic_direction', _melodic_direction_tokens),
            ('interval_shift', _interval_shift_tokens),
            ('unison_chain', _unison_chain_tokens),
            ('rest_streak', _rest_streak_tokens),
            ('mono_rhythm', _mono_rhythm_tokens),
            ('extreme_density', _extreme_density_tokens),
            ('max_polyphony_per_position', _max_polyphony_per_position_tokens),
        ]:
            try:
                scores[name] = fn(flat, tokenizer)
            except Exception:
                pass

        # ── B1 和声/旋律指标 ────────────────────────────
        try:
            scores['progression_validity'] = progression_validity_tokens(flat, tokenizer)
        except Exception:
            pass
        try:
            scores['harmonic_rhythm'] = harmonic_rhythm_score_tokens(flat, tokenizer)
        except Exception:
            pass
        try:
            scores['bar_boundary_melody'] = _bar_boundary_melody_tokens(flat, tokenizer)
        except Exception:
            pass
        try:
            scores['parallel_fifths'] = _parallel_fifths_tokens(flat, tokenizer)
        except Exception:
            pass

        # 更新前一个窗口的统计
        self._prev_pitch_class_dist = _pitch_class_dist(flat, tokenizer,
                                                         self.seed_profile.tonic_midi)
        self._prev_interval_dist = [0.04] * 25
        intervals = _note_on_intervals(flat, tokenizer)
        if intervals:
            counts = [0] * 25
            for iv in intervals:
                counts[min(abs(iv), 24)] += 1
            t = sum(counts)
            if t > 0:
                self._prev_interval_dist = [c / t for c in counts]

        # B1 聚合分 = 平均
        avg = sum(scores.values()) / max(len(scores), 1)
        return avg, scores

    def _compute_b2(
        self,
        bars: list[list[int]],
        bar_count: int,
        tokenizer,
    ) -> tuple[float, dict[str, float]]:
        """B2：最近 S 节 vs seed 整体的漂移检测。"""
        profile = self.seed_profile
        block = self._b2_block_size

        # 排除 seed 小节，只评新生成内容
        all_bars = bars[1:]  # 去掉第一个空节（BOS 前的）
        gen_only = all_bars[profile.n_bars:]  # 去掉 seed 的 N 个小节
        if len(gen_only) < max(2, block):
            return 0.5, {}
        # 用最近 block 节
        window = gen_only[-block:]
        flat = [t for bar in window for t in bar]

        window_notes = _note_on_intervals(flat, tokenizer)

        scores: dict[str, float] = {}

        # 空小节检测：即使音符极少也必须运行（empty_measure 不需要音符）
        try:
            scores['empty_measure'] = _empty_measure_tokens(flat, tokenizer)
        except Exception:
            pass

        if len(window_notes) < 3:
            if scores:
                avg = sum(scores.values()) / len(scores)
                return avg, scores
            return 0.5, {}

        # 与 seed 对比的指标
        try:
            scores['pitch_class_kl'] = _pitch_class_kl_tokens(
                flat, tokenizer, reference=profile.pitch_class_dist,
            )
        except Exception:
            pass

        try:
            scores['interval_kl'] = _interval_kl_tokens(
                flat, tokenizer, reference=profile.interval_dist,
            )
        except Exception:
            pass

        try:
            scores['density_z'] = _density_z_tokens(
                flat, tokenizer, reference_density=profile.bar_density,
            )
        except Exception:
            pass

        try:
            scores['rest_ratio'] = _rest_ratio_tokens(
                flat, tokenizer, reference=profile.rest_ratio,
            )
        except Exception:
            pass

        try:
            scores['register_span'] = _register_span_tokens(
                flat, tokenizer, reference_span=max(profile.interval_dist) * 87 if max(profile.interval_dist) > 0.05 else None,
            )
        except Exception:
            pass

        try:
            scores['velocity_consistency'] = _velocity_consistency_tokens(
                flat, tokenizer, reference_mean=profile.velocity_mean,
            )
        except Exception:
            pass

        # 不与 seed 对比的指标（empty_measure 已在上面处理）
        for name, fn in [
            ('key_consistency', _key_consistency_tokens),
            ('pitch_range', _pitch_range_tokens),
            ('duration_entropy', _duration_entropy_tokens),
            ('interval_shift', _interval_shift_tokens),
            ('token_type_kl', _token_type_kl_tokens),
        ]:
            try:
                scores[name] = fn(flat, tokenizer)
            except Exception:
                pass

        # ── B2 和声/旋律指标 ────────────────────────────
        try:
            scores['progression_validity'] = progression_validity_tokens(flat, tokenizer)
        except Exception:
            pass
        try:
            seed_harmonic_density = None
            if self.blueprint and self.blueprint.harmony.chord_density_per_bar_mean:
                seed_harmonic_density = self.blueprint.harmony.chord_density_per_bar_mean
            scores['harmonic_rhythm'] = harmonic_rhythm_score_tokens(
                flat, tokenizer, reference_density=seed_harmonic_density)
        except Exception:
            pass
        try:
            scores['chord_melody_alignment'] = chord_melody_alignment_tokens(
                flat, tokenizer, tonic_midi=profile.tonic_midi)
        except Exception:
            pass
        try:
            scores['cadence_quality'] = cadence_quality_tokens(flat, tokenizer)
        except Exception:
            pass
        try:
            scores['cadence_placement'] = self._cadence_placement_check(
                flat, bar_count, tokenizer)
        except Exception:
            pass
        try:
            seed_contour = None
            if self.blueprint and self.blueprint.harmony.seed_contour:
                seed_contour = self.blueprint.harmony.seed_contour
            scores['melodic_contour'] = _melodic_contour_tokens(
                flat, tokenizer, seed_contour=seed_contour)
        except Exception:
            pass

        # B2 块评分记录（用于趋势检测）
        avg = sum(scores.values()) / max(len(scores), 1)
        if bar_count - self._last_b2_bar >= block:
            self._b2_block_scores.append(avg)
            self._last_b2_bar = bar_count
            if len(self._b2_block_scores) > 10:
                self._b2_block_scores = self._b2_block_scores[-10:]

        return avg, scores

    def _b1_adjustments(
        self, scores: dict[str, float], bar_count: int,
    ) -> dict[str, float]:
        """B1 指标 → 参数调整。"""
        adj: dict[str, float] = {}
        for metric_name, score in scores.items():
            if metric_name not in B1_ADJUSTMENT_RULES:
                continue
            if score >= self.b1_threshold:
                continue
            deficit = (self.b1_threshold - score) / self.b1_threshold
            for param, max_delta in B1_ADJUSTMENT_RULES[metric_name].items():
                adj[param] = adj.get(param, 0.0) + max_delta * deficit
        return adj

    def _b2_adjustments(
        self, scores: dict[str, float], bar_count: int,
    ) -> dict[str, float]:
        """B2 指标 → 参数调整（含趋势乘数 + 段落容忍度）。"""
        # 段落特定容忍度：development/cadenza 等段落天然偏离 seed，降低调整力度
        section_tolerance = SECTION_B2_TOLERANCE.get(
            self._current_section_type or '', 1.0
        )
        adj: dict[str, float] = {}
        for metric_name, score in scores.items():
            if metric_name not in B2_ADJUSTMENT_RULES:
                continue
            if score >= self.b2_threshold:
                continue
            deficit = (self.b2_threshold - score) / self.b2_threshold
            for param, max_delta in B2_ADJUSTMENT_RULES[metric_name].items():
                adj[param] = adj.get(param, 0.0) + max_delta * deficit * section_tolerance

        # 趋势检测：最近 N 块持续下降 → 额外惩罚
        if len(self._b2_block_scores) >= B2_TREND_WINDOW:
            recent = self._b2_block_scores[-B2_TREND_WINDOW:]
            if all(recent[i] <= recent[i - 1] for i in range(1, len(recent))):
                for param in list(adj.keys()):
                    adj[param] = adj[param] * (1.0 + B2_TREND_PENALTY)
                logger.debug("B2 trend: decreasing over last %d blocks, penalty applied",
                             B2_TREND_WINDOW)

        return adj

    # ── 段落感知方法 ──────────────────────────────────

    def _get_current_section(self, bar_count: int) -> str | None:
        """返回当前 bar 所在的段落类型名。"""
        for i, section in enumerate(self._section_plan):
            sec_start = section.start_bar
            sec_end = sec_start + section.n_bars
            if i + 1 < len(self._section_plan):
                next_start = self._section_plan[i + 1].start_bar
                if sec_start <= bar_count < next_start:
                    return section.type
            else:
                if bar_count >= sec_start:
                    return section.type
        # fallback：查找包含此 bar 的段落
        for section in self._section_plan:
            if section.start_bar <= bar_count < section.start_bar + section.n_bars:
                return section.type
        return None

    def _get_current_section_end(self, bar_count: int) -> int | None:
        """返回当前 bar 所在段落的结束 bar 号。"""
        for i, section in enumerate(self._section_plan):
            if i + 1 < len(self._section_plan):
                sec_start = section.start_bar
                next_start = self._section_plan[i + 1].start_bar
                if sec_start <= bar_count < next_start:
                    return next_start
            else:
                if bar_count >= section.start_bar:
                    return section.start_bar + section.n_bars
        return None

    def _enter_section(self, section_type: str):
        """进入新段落时，主动设置参数姿态。"""
        target = self._style_profile.styles.get(
            section_type, self._style_profile.fallback
        )
        self.b1_threshold = B1_DEFAULT_THRESHOLD * target.b1_threshold_mult
        self.b2_threshold = B2_DEFAULT_THRESHOLD * target.b2_threshold_mult
        logger.info("B|进入段落 %s: b1_thr=%.2f b2_thr=%.2f confidence=%s",
                    section_type, self.b1_threshold, self.b2_threshold,
                    target.confidence)

    def _near_boundary(self, bar_count: int) -> str | None:
        """检测当前 bar 是否接近段落边界。"""
        for i, section in enumerate(self._section_plan):
            sec_start = section.start_bar
            if i + 1 < len(self._section_plan):
                sec_end = self._section_plan[i + 1].start_bar
            else:
                sec_end = sec_start + section.n_bars
            if sec_start <= bar_count < sec_end:
                pos_in_section = bar_count - sec_start
                section_n_bars = sec_end - sec_start
                if pos_in_section < 2:
                    return 'entering'
                if pos_in_section >= section_n_bars - 2:
                    return 'leaving'
                return None
        return None

    def _cadence_placement_check(self, flat: list[int], bar_count: int, tokenizer) -> float:
        """检查终止式是否出现在段落边界附近。"""
        section_end = self._get_current_section_end(bar_count)
        if section_end is None:
            return 1.0
        bars_to_end = section_end - bar_count
        if bars_to_end > 3:
            return 1.0
        chords = _extract_chord_sequence(flat, tokenizer)
        if len(chords) < 2:
            return 0.7
        last_two = (chords[-2], chords[-1])
        cadence_pairs = {('V', 'I'), ('V', 'i'), ('IV', 'I'), ('iv', 'i'),
                         ('V', 'vi'), ('vii°', 'I'), ('vii°', 'i')}
        if last_two in cadence_pairs:
            return 1.0
        if bars_to_end <= 1 and last_two not in cadence_pairs:
            return 0.3
        return 0.6

    def _adjust_section_length(self, bar_count: int):
        """根据最近 B2 质量动态调整当前段落剩余长度。"""
        if not self._b2_block_scores:
            return
        current_section = self._current_section_type
        if current_section is None:
            return
        # 找到当前段落在 section_plan 中的索引
        sec_idx = None
        for i, sec in enumerate(self._section_plan):
            if sec.type == current_section:
                if sec.start_bar <= bar_count:
                    sec_idx = i
        if sec_idx is None or sec_idx in self._section_adjusted:
            return

        section_end = self._get_current_section_end(bar_count)
        if section_end is None:
            return
        remaining = section_end - bar_count

        recent_scores = self._b2_block_scores[-4:]
        avg_score = sum(recent_scores) / len(recent_scores)

        if avg_score > 0.75 and remaining <= 6:
            extend = min(2, self._section_plan[sec_idx].max_bars - self._section_plan[sec_idx].n_bars)
            if extend > 0:
                self._section_plan[sec_idx].n_bars += extend
                for later in self._section_plan[sec_idx + 1:]:
                    later.start_bar += extend
                self._section_adjusted.add(sec_idx)
                logger.info("B|bar=%d: B2 高质量 (%.2f), 延长当前段 +%d bar",
                           bar_count, avg_score, extend)
        elif avg_score < 0.35 and remaining >= 6:
            shorten = min(2, remaining - 4)
            if shorten > 0:
                self._section_plan[sec_idx].n_bars -= shorten
                for later in self._section_plan[sec_idx + 1:]:
                    later.start_bar -= shorten
                self._section_adjusted.add(sec_idx)
                logger.info("B|bar=%d: B2 低质量 (%.2f), 缩短当前段 -%d bar",
                           bar_count, avg_score, shorten)

    def _apply_decay(self):
        """每 bar 对累积调整量做指数衰减。"""
        decay_factor = 0.5 ** (1.0 / ADJUSTMENT_HALF_LIFE)
        for k in list(self._cumulative_adjustments.keys()):
            self._cumulative_adjustments[k] *= decay_factor
            if abs(self._cumulative_adjustments[k]) < 0.01:
                del self._cumulative_adjustments[k]

    def _mid_generation_check(self, generated_tokens: list[int], tokenizer) -> dict | None:
        """快速致命检查（仅检查明确失败的模式）。"""
        bar_id = tokenizer.bar_token_id
        bars = _tokens_by_bar(generated_tokens, bar_id)
        if len(bars) < 4:
            return None

        # 检查连续 4+ 空小节
        empty_streak = 0
        for bar_tokens in bars[-8:]:
            has_note = any(tokenizer.decode_token(t).startswith('<Note_ON')
                         for t in bar_tokens)
            if has_note:
                empty_streak = 0
            else:
                empty_streak += 1
                if empty_streak >= 4:
                    return {'reason': '连续 4+ 空小节', 'bar': len(bars)}

        # 检查连续 8+ 同音
        intervals = _note_on_intervals(generated_tokens[-500:], tokenizer)
        if len(intervals) >= 8:
            same_streak = 0
            for i in range(1, len(intervals)):
                if intervals[i] == intervals[i - 1]:
                    same_streak += 1
                    if same_streak >= 8:
                        return {'reason': '连续 8+ 同音', 'bar': len(bars)}
                else:
                    same_streak = 0

        return None

    # ── 状态查询 ────────────────────────────────────────

    @property
    def b2_trend(self) -> str | None:
        """返回 B2 趋势描述。"""
        if len(self._b2_block_scores) < 2:
            return None
        recent = self._b2_block_scores[-min(4, len(self._b2_block_scores)):]
        if all(recent[i] >= recent[i - 1] for i in range(1, len(recent))):
            return 'improving'
        if all(recent[i] <= recent[i - 1] for i in range(1, len(recent))):
            return 'declining'
        return 'stable'

    def get_stats(self) -> dict:
        """返回当前状态快照。"""
        return {
            'b2_block_scores': self._b2_block_scores,
            'b2_trend': self.b2_trend,
            'cumulative_adjustments': dict(self._cumulative_adjustments),
        }


# ── C 阶段：生成后评价 ──────────────────────────────────

REWARD_LOG_DIR = os.environ.get(
    'CHOPINOTE_REWARD_DIR',
    '/root/autodl-tmp/chopinote/rewards',
)


class PostGenerationFilter:
    """C 阶段：生成后全量评价 + 低分重试 + RL reward 日志。"""

    def __init__(
        self,
        tokenizer,
        benchmarks: dict | None = None,
        benchmarks_path: str | None = None,
        alpha: float = 0.3,
        group: str = 'all',
        retry_threshold: float = 0.55,
        max_retries: int = 3,
        reward_dir: str = REWARD_LOG_DIR,
    ):
        self.tokenizer = tokenizer
        self.benchmarks = benchmarks
        self.benchmarks_path = benchmarks_path
        self.alpha = alpha
        self.group = group
        self.retry_threshold = retry_threshold
        self.max_retries = max_retries
        self.reward_dir = reward_dir

    def evaluate(
        self,
        musicxml_path: str,
        seed_path: str | None = None,
        checkpoint: str | None = None,
    ):
        """运行全量评价，返回 EvaluationReport。"""
        from chopinote_evaluator.score import Evaluator
        from chopinote_evaluator.benchmarks.build_benchmarks import load_benchmarks

        bm = self.benchmarks
        if bm is None:
            bm = load_benchmarks(self.benchmarks_path)

        ev = Evaluator(benchmarks=bm, alpha=self.alpha, group=self.group)

        # 当无 seed_path 时，仍尝试用 seed 的 key/time sig 约束生成阶段的连贯性
        report = ev.evaluate(
            score_path=musicxml_path,
            seed_path=seed_path,
            checkpoint=checkpoint,
        )
        return report

    def should_retry(self, report) -> bool:
        """判断是否需要重试。"""
        return report.total_score < self.retry_threshold

    def log_reward(
        self,
        report,
        params: GenerationParams | None = None,
        retry_count: int = 0,
        seed_info: dict | None = None,
        extra: dict | None = None,
    ):
        """记录生成结果到 reward 日志（JSONL）。"""
        os.makedirs(self.reward_dir, exist_ok=True)

        entry = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_score': report.total_score,
            'alpha': getattr(report, 'alpha', None),
            'general_score': report.general.get('score', 0) if report.general else 0,
            'specific_score': (
                report.specific.get('score', 0)
                if report.specific else 0
            ),
            'legality_passed': report.legality.passed if report.legality else True,
            'legality_issues': len(report.legality.issues) if report.legality else 0,
            'retry_count': retry_count,
            'seed_info': seed_info or {},
            'params': params.__dict__ if params else {},
        }

        if extra:
            entry.update(extra)

        log_path = os.path.join(self.reward_dir, 'reward_log.jsonl')
        try:
            with open(log_path, 'a') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        except OSError as e:
            logger.warning("Failed to write reward log: %s", e)

        return log_path

    def diagnose_bars(self, report, tokenizer, seed_profile=None) -> list[BarDiagnosis]:
        """逐小节诊断，定位具体问题。"""
        diagnoses: list[BarDiagnosis] = []

        # 扫描合法性问题
        if hasattr(report, 'legality') and report.legality:
            for issue in report.legality.issues:
                bar = getattr(issue, 'measure', 0)
                if bar > 0:
                    diagnoses.append(BarDiagnosis(
                        bar=bar,
                        issues=[str(issue)],
                        severity=0.8,
                        suggestion='检查该小节音域/配对/空小节',
                    ))

        # 扫描理论违规
        if report.general and isinstance(report.general, dict):
            theory = report.general.get('theory', {})
            violations = theory.get('violations', []) if isinstance(theory, dict) else []
            for v in violations:
                m = v.get('measure', 0) if isinstance(v, dict) else getattr(v, 'measure', 0)
                if m > 0:
                    desc = str(v.get('type', 'unknown')) if isinstance(v, dict) else str(v)
                    diagnoses.append(BarDiagnosis(
                        bar=m,
                        issues=[desc],
                        severity=0.6,
                        suggestion='检查该小节和声/声部进行',
                    ))

        # 如果无问题，返回空
        if not diagnoses:
            return []

        # 合并同小节的诊断
        merged: dict[int, BarDiagnosis] = {}
        for d in diagnoses:
            if d.bar in merged:
                merged[d.bar].issues.extend(d.issues)
                merged[d.bar].severity = max(merged[d.bar].severity, d.severity)
            else:
                merged[d.bar] = d
        return sorted(merged.values(), key=lambda d: d.bar)

    def smart_rollback(
        self,
        diagnoses: list[BarDiagnosis],
        section_plan: list[SectionPlan] | None = None,
        min_rollback_bars: int = 4,
    ) -> int | None:
        """根据诊断 + 段落结构选择最优退回点。

        返回退回的目标 bar 号（1-indexed），或 None（不退回）。
        """
        if not diagnoses:
            return None

        # 策略 1：如果问题集中在某个段落 → 退回该段开头
        if section_plan:
            for sec in section_plan:
                sec_bars = [d for d in diagnoses
                           if sec.start_bar <= d.bar < sec.start_bar + sec.n_bars]
                if len(sec_bars) >= 2:
                    rollback_bar = sec.start_bar + 1
                    if rollback_bar > min_rollback_bars:
                        return rollback_bar

        # 策略 2：退回第一个 severity > 0.7 的 bar 之前
        for d in diagnoses:
            if d.severity > 0.7 and d.bar > min_rollback_bars:
                return max(1, d.bar - 1)

        # 策略 3：退回第一个问题的前一个 bar
        first_bar = diagnoses[0].bar
        if first_bar > min_rollback_bars:
            return max(1, first_bar - 1)

        return None

    def find_problem_bars(self, report) -> list[int]:
        """从评价报告中找出有问题的具体小节号。

        扫描合法性问题的 measure 字段和理论违规的 measure，
        返回问题小节列表（升序，去重）。
        """
        bars: set[int] = set()
        if hasattr(report, 'legality') and report.legality:
            for issue in report.legality.issues:
                if issue.measure > 0:
                    bars.add(issue.measure)

        if report.general and isinstance(report.general, dict):
            theory = report.general.get('theory', {})
            violations = theory.get('violations', []) if isinstance(theory, dict) else []
            for v in violations:
                m = v.get('measure', 0) if isinstance(v, dict) else getattr(v, 'measure', 0)
                if m > 0:
                    bars.add(m)

        return sorted(bars)

    def get_rollback_point(
        self,
        report,
        tokenizer,
        full_tokens: list[int],
        seed_bars: int = 0,
        min_rollback_bars: int = 4,
    ) -> int | None:
        """找到第一个问题小节前的退回到 token 位置。

        返回 token 序列中退回点的索引（退回点之后的 token 将被丢弃重新生成）。
        退回点 = 问题小节的前一个小节的 BAR token 位置。
        如果问题无法定位或退回后生成内容太少，返回 None。
        """
        problem_bars = self.find_problem_bars(report)
        if not problem_bars:
            return None

        # 取第一个问题小节
        first_bad_bar = problem_bars[0]
        # 退回一整个小节，确保上下文干净
        rollback_bar = max(1, first_bad_bar - 1)
        # 至少保留 min_rollback_bars 节的生成内容（不含 seed）
        if rollback_bar <= seed_bars + min_rollback_bars:
            # 问题太靠前，退回意义不大
            return None

        # 在第 rollback_bar 小节对应的 BAR token 位置截断
        bar_id = tokenizer.bar_token_id
        bar_count = 0
        for i, tid in enumerate(full_tokens):
            if tid == bar_id:
                bar_count += 1
                if bar_count == rollback_bar:
                    return i  # 退回点：从此处截断

        return None

    def get_retry_adjustments(self, report, retry_count: int) -> dict[str, Any]:
        """根据评价报告生成重试参数调整（返回 DELTAs，apply_adjustments 做加法 + 裁切）。"""
        adj: dict[str, Any] = {}

        # 合法性失败 → 大幅降低温度
        if hasattr(report, 'legality') and report.legality and not report.legality.passed:
            adj['temperature'] = -min(0.7, retry_count * 0.15)
            adj['rest_penalty'] = retry_count * 0.5
            return adj

        general = report.general or {}
        specific = report.specific or {}

        # 统计评分低 → 降低温度 + 增 key_bias
        stat_score = general.get('statistical_score', 0.5)
        if stat_score < 0.5:
            adj['temperature'] = -retry_count * 0.1
            adj['key_bias_strength'] = retry_count * 0.5

        # 理论评分低 → 降低复杂度
        theory_score = general.get('theory', {}).get('score', 0.5)
        if theory_score < 0.5:
            adj['complexity'] = -retry_count * 1.0

        # 一致性评分低 → 增加 rest_penalty + 降低温度
        if specific:
            cons_score = specific.get('consistency_score', 0.5)
            if cons_score < 0.5:
                adj['temperature'] = adj.get('temperature', 0.0) - 0.1
                adj['rest_penalty'] = adj.get('rest_penalty', 0.0) + retry_count * 0.3

        return adj
