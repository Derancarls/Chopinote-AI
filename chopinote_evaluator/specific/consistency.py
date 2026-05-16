"""风格一致性 — 衡量种子和生成片段之间的音乐风格连续程度。

输入: (seed, continuation) 两个 Score 对象
输出: 各指标 0~1 分数
"""

from __future__ import annotations

import math

from chopinote_evaluator.parser import Score, score_to_note_count
from chopinote_evaluator.benchmarks.build_benchmarks import kl_divergence

# 默认对比小节数
DEFAULT_CONTEXT_BARS = 4


class ConsistencyEvaluator:
    """风格一致连续性评估器。

    比较种子尾部和生成首部的统计分布差异。
    """

    def evaluate(self, seed: Score, continuation: Score,
                 n_bars: int = DEFAULT_CONTEXT_BARS) -> dict[str, float]:
        """评估风格一致性。

        参数:
            seed: 种子部分（用户编写的）
            continuation: 生成的续写
            n_bars: 边界两侧各取多少小节对比

        返回:
            {指标名: score (0~1)}
        """
        seed_seg = self._tail(seed, n_bars)
        gen_seg = self._head(continuation, n_bars)

        if not seed_seg.measures or not gen_seg.measures:
            return {'error': 0.0, 'detail': '边界片段为空'}

        return {
            'pitch_class_kl': self._pitch_boundary_kl(seed_seg, gen_seg),
            'density_delta': self._density_delta(seed_seg, gen_seg),
            'interval_shift': self._interval_shift(seed_seg, gen_seg),
            'velocity_delta': self._velocity_delta(seed_seg, gen_seg),
            'articulation_delta': self._articulation_delta(seed_seg, gen_seg),
        }

    def aggregate(self, scores: dict[str, float]) -> float:
        """一致性分数加权平均。"""
        weights = {
            'pitch_class_kl': 0.30,
            'density_delta': 0.20,
            'interval_shift': 0.20,
            'velocity_delta': 0.15,
            'articulation_delta': 0.15,
        }
        total_w = 0.0
        weighted = 0.0
        for k, w in weights.items():
            if k in scores and isinstance(scores[k], (int, float)):
                weighted += scores[k] * w
                total_w += w
        return weighted / max(total_w, 0.001)

    # ── 指标实现 ──────────────────────────────────────────

    @staticmethod
    def _pitch_boundary_kl(seed_seg: Score, gen_seg: Score) -> float:
        """种子尾部 vs 生成首部的 pitch class KL 散度 → score。"""
        def pc_dist(score):
            counts = [0] * 12
            for m in score.measures:
                for n in m.notes:
                    if not n.is_rest and n.pitch is not None:
                        counts[n.pitch % 12] += 1
            total = sum(counts)
            if total == 0:
                return None
            return [c / total for c in counts]

        seed_dist = pc_dist(seed_seg)
        gen_dist = pc_dist(gen_seg)

        if seed_dist is None or gen_dist is None:
            return 0.5

        kl = kl_divergence(gen_dist, seed_dist)
        return math.exp(-kl * 5)

    @staticmethod
    def _density_delta(seed_seg: Score, gen_seg: Score) -> float:
        """音符密度差异。"""
        def density(score):
            notes = sum(1 for m in score.measures for n in m.notes if not n.is_rest)
            measures = len(score.measures)
            return notes / max(measures, 1)

        d1 = density(seed_seg)
        d2 = density(gen_seg)
        delta = abs(d1 - d2) / max(d1, d2, 0.01)
        return 1.0 / (1.0 + delta * 3)

    @staticmethod
    def _interval_shift(seed_seg: Score, gen_seg: Score) -> float:
        """音程分布偏移（步进 vs 大跳比例）。"""
        def step_ratio(score):
            intervals = []
            for m in score.measures:
                notes = [n for n in m.notes if not n.is_rest and n.pitch is not None]
                for i in range(len(notes) - 1):
                    intervals.append(abs(notes[i + 1].pitch - notes[i].pitch))
            if not intervals:
                return 0.5
            steps = sum(1 for iv in intervals if iv <= 2)
            return steps / len(intervals)

        r1 = step_ratio(seed_seg)
        r2 = step_ratio(gen_seg)
        return max(0.0, 1.0 - abs(r1 - r2) * 2)

    @staticmethod
    def _velocity_delta(seed_seg: Score, gen_seg: Score) -> float:
        """力度均值差。"""
        def avg_vel(score):
            velocities = []
            for m in score.measures:
                for n in m.notes:
                    if not n.is_rest:
                        velocities.append(n.velocity)
            if not velocities:
                return 64.0
            return sum(velocities) / len(velocities)

        v1 = avg_vel(seed_seg)
        v2 = avg_vel(gen_seg)
        return 1.0 - min(abs(v1 - v2) / 40.0, 1.0)

    @staticmethod
    def _articulation_delta(seed_seg: Score, gen_seg: Score) -> float:
        """演奏法密度差。"""
        def art_ratio(score):
            total = sum(1 for m in score.measures for n in m.notes)
            arts = sum(1 for m in score.measures for n in m.notes if n.articulation)
            return arts / max(total, 1)

        return 1.0 - abs(art_ratio(seed_seg) - art_ratio(gen_seg))

    # ── 辅助 ──────────────────────────────────────────────

    @staticmethod
    def _tail(score: Score, n_bars: int) -> Score:
        """取 Score 最后 n 个小节。"""
        measures = score.measures[-n_bars:] if len(score.measures) > n_bars else score.measures
        return Score(measures=measures, tempo=score.tempo, programs=score.programs)

    @staticmethod
    def _head(score: Score, n_bars: int) -> Score:
        """取 Score 前 n 个小节。"""
        measures = score.measures[:n_bars] if len(score.measures) > n_bars else score.measures
        return Score(measures=measures, tempo=score.tempo, programs=score.programs)
