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


# ── B 阶段调整规则 ────────────────────────────────────

# B1（局部流畅）：指标评分 < 阈值时触发调整
# key = 指标名, value = {param: max_delta_per_call}
B1_ADJUSTMENT_RULES: dict[str, dict[str, float]] = {
    'density_z': {'rest_penalty': 0.3, 'temperature': -0.05},
    'dissonance_ratio': {'temperature': -0.15},
    'velocity_consistency': {'temperature': -0.10},
    'rest_ratio': {'rest_penalty': 0.25},
    'register_span': {'complexity': -0.5},
    'duration_entropy': {'temperature': -0.10},
    'syncopation_ratio': {'temperature': -0.08},
    'melodic_direction': {'temperature': -0.08},
    'interval_shift': {'temperature': -0.05},
}

# B2（全局漂移）
B2_ADJUSTMENT_RULES: dict[str, dict[str, float]] = {
    'pitch_class_kl': {'key_bias_strength': 0.5},
    'interval_kl': {'key_bias_strength': 0.3},
    'density_z': {'rest_penalty': 0.3, 'temperature': -0.05},
    'rest_ratio': {'rest_penalty': 0.4},
    'register_span': {'temperature': -0.10},
    'velocity_consistency': {'temperature': -0.08},
    'key_consistency': {'key_bias_strength': 0.8},
    'token_type_kl': {'temperature': -0.10, 'rest_penalty': 0.2},
}

B1_DEFAULT_THRESHOLD = 0.55
B2_DEFAULT_THRESHOLD = 0.50

# B2 趋势检测：连续 N 块下降 → 额外惩罚乘数
B2_TREND_WINDOW = 3
B2_TREND_PENALTY = 0.5  # 趋势下降时额外 delta 乘数


class NarrowFeedbackController:
    """B 阶段：生成中实时反馈。

    B1：只看最近 N 节自身流畅性，不依赖 seed。
    B2：每 S 节一块 vs seed 整体，检测漂移趋势。
    """

    def __init__(
        self,
        seed_profile: SeedProfile,
        tokenizer,
        local_bars: int = 4,
        local_weight: float = 0.5,
        global_weight: float = 0.5,
        b1_threshold: float = B1_DEFAULT_THRESHOLD,
        b2_threshold: float = B2_DEFAULT_THRESHOLD,
        b2_extra_pull: float = 1.0,
    ):
        self.seed_profile = seed_profile
        self.tokenizer = tokenizer
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

        # 参数调整累加（用于日志或衰减）
        self._cumulative_adjustments: dict[str, float] = {}

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

        # 如果小节数不足，跳过
        if len(bars) < 2:
            return {}

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

        if not adjustments:
            return {}

        # ── 更新累计 ────────────────────────────────────
        for k, v in adjustments.items():
            self._cumulative_adjustments[k] = self._cumulative_adjustments.get(k, 0.0) + v

        # ── 日志 ────────────────────────────────────────
        combined = b1_score * self.local_weight + b2_score * self.global_weight
        logger.debug(
            "B|bar=%d b1=%.3f b2=%.3f combined=%.3f adj=%s",
            bar_count, b1_score, b2_score, combined, adjustments,
        )

        return adjustments

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
        if len(window_notes) < 3:
            return 0.5, {}

        scores: dict[str, float] = {}

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
        ]:
            try:
                scores[name] = fn(flat, tokenizer)
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

        # 至少需要 S 节生成内容 + 排除 seed 部分
        gen_bars = bars[1:]  # 去掉第一个空节（BOS 前的）
        if len(gen_bars) < max(2, block):
            return 0.5, {}
        # 用最近 block 节
        window = gen_bars[-block:]
        flat = [t for bar in window for t in bar]

        window_notes = _note_on_intervals(flat, tokenizer)
        if len(window_notes) < 3:
            return 0.5, {}

        scores: dict[str, float] = {}

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

        # 不与 seed 对比的指标
        for name, fn in [
            ('key_consistency', _key_consistency_tokens),
            ('empty_measure', _empty_measure_tokens),
            ('pitch_range', _pitch_range_tokens),
            ('duration_entropy', _duration_entropy_tokens),
            ('interval_shift', _interval_shift_tokens),
        ]:
            try:
                scores[name] = fn(flat, tokenizer)
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
        """B2 指标 → 参数调整（含趋势乘数）。"""
        adj: dict[str, float] = {}
        for metric_name, score in scores.items():
            if metric_name not in B2_ADJUSTMENT_RULES:
                continue
            if score >= self.b2_threshold:
                continue
            deficit = (self.b2_threshold - score) / self.b2_threshold
            for param, max_delta in B2_ADJUSTMENT_RULES[metric_name].items():
                adj[param] = adj.get(param, 0.0) + max_delta * deficit

        # 趋势检测：最近 N 块持续下降 → 额外惩罚
        if len(self._b2_block_scores) >= B2_TREND_WINDOW:
            recent = self._b2_block_scores[-B2_TREND_WINDOW:]
            if all(recent[i] <= recent[i - 1] for i in range(1, len(recent))):
                for param in list(adj.keys()):
                    adj[param] = adj[param] * (1.0 + B2_TREND_PENALTY)
                logger.debug("B2 trend: decreasing over last %d blocks, penalty applied",
                             B2_TREND_WINDOW)

        return adj

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
