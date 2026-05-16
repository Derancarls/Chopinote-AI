"""统计分布对比 — 将输入乐谱的分布与基准对比，输出 0~1 分数。
"""

from __future__ import annotations

import math
from typing import Any

from chopinote_evaluator.parser import Score, score_to_duration_seconds
from chopinote_evaluator.benchmarks.build_benchmarks import (
    kl_divergence,
    cosine_similarity,
    get_ks_profile,
    load_benchmarks,
)


class StatisticalEvaluator:
    """统计分布对比评估器。

    加载基准分布后，对输入乐谱计算各指标与基准的偏差分数。

    用法:
        benchmarks = load_benchmarks()
        eval = StatisticalEvaluator(benchmarks, group='all')
        result = eval.evaluate_global(score)
    """

    # 各指标权重（总和 = 1.0）
    WEIGHTS = {
        'pitch_class_kl': 0.12,
        'interval_dist_kl': 0.08,
        'density_z': 0.10,
        'key_consistency': 0.10,
        'self_similarity': 0.08,
        'token_type_kl': 0.05,
        'duration_entropy': 0.06,
        'pitch_entropy': 0.05,
        'dissonance_ratio': 0.06,
        'chromaticism_index': 0.05,
        'register_span': 0.04,
        'syncopation_ratio': 0.05,
        'dynamic_variance': 0.04,
        'rest_ratio': 0.04,
        'melodic_direction': 0.03,
        'contour_arc': 0.03,
        'polyphony_mean': 0.03,
        'texture_variance': 0.03,
        'harmonic_rhythm': 0.04,
    }

    def __init__(self, benchmarks: dict[str, Any] | None = None,
                 group: str = 'all'):
        """
        参数:
            benchmarks: build_benchmarks 产出的全量基准 dict
            group: 对比锚点名（如 'all', 'timesig_4_4', 'source_musescore', ...）
        """
        self.benchmarks = benchmarks or {}
        self.group_data = self._select_group(group)

    def _select_group(self, group: str) -> dict:
        """从基准中选择对应分组的数据。"""
        if not self.benchmarks:
            return {}
        return self.benchmarks.get(group, self.benchmarks.get('all', {}))

    def evaluate_global(self, score: Score) -> dict[str, float]:
        """对完整乐谱做统计对比。

        返回:
            {指标名: score (0~1)}
        """
        return {
            'pitch_class_kl': self._pitch_class_kl_score(score),
            'interval_dist_kl': self._interval_dist_kl_score(score),
            'density_z': self._density_z_score(score),
            'key_consistency': self._key_consistency_score(score),
            'self_similarity': self._self_similarity_score(score),
            'token_type_kl': self._token_type_kl_score(score),
            'duration_entropy': self._duration_entropy_score(score),
            'pitch_entropy': self._pitch_entropy_score(score),
            'dissonance_ratio': self._dissonance_ratio_score(score),
            'chromaticism_index': self._chromaticism_index_score(score),
            'register_span': self._register_span_score(score),
            'syncopation_ratio': self._syncopation_ratio_score(score),
            'dynamic_variance': self._dynamic_variance_score(score),
            'rest_ratio': self._rest_ratio_score(score),
            'melodic_direction': self._melodic_direction_score(score),
            'contour_arc': self._contour_arc_score(score),
            'polyphony_mean': self._polyphony_mean_score(score),
            'texture_variance': self._texture_variance_score(score),
            'harmonic_rhythm': self._harmonic_rhythm_score(score),
        }

    def evaluate_boundary(self, seed: Score, continuation: Score,
                          n_bars: int = 4) -> dict[str, float]:
        """边界统计对比（种子尾部 vs 生成首部），供狭义层使用。"""
        seed_seg = Score(measures=seed.measures[-n_bars:] if len(seed.measures) > n_bars else seed.measures)
        gen_seg = Score(measures=continuation.measures[:n_bars] if len(continuation.measures) > n_bars else continuation.measures)
        return {
            'pitch_class_kl': self._pitch_class_kl_score(gen_seg, seed_seg),
            'density_z': self._density_z_score(gen_seg, seed_seg),
        }

    def aggregate_score(self, scores: dict[str, float]) -> float:
        """对指标分数做加权平均。"""
        total_weight = 0.0
        weighted_sum = 0.0
        for key, weight in self.WEIGHTS.items():
            if key in scores:
                weighted_sum += scores[key] * weight
                total_weight += weight
        return weighted_sum / max(total_weight, 0.001)

    # ── 各指标实现 ──────────────────────────────────────────

    def _pitch_class_kl_score(self, score: Score,
                              reference: Score | None = None) -> float:
        """12 音级分布 KL 散度 → score。

        有 reference 时：对比两个 Score 的分布（边界对比）。
        无 reference 时：对比基准库的 pitch class 分布（如无，用 K-S profile）。
        """
        counts = [0] * 12
        for m in score.measures:
            for n in m.notes:
                if not n.is_rest and n.pitch is not None:
                    counts[n.pitch % 12] += 1

        total = sum(counts)
        if total == 0:
            return 0.0

        dist = [c / total for c in counts]

        if reference is not None:
            # 与参考 Score 对比
            ref_counts = [0] * 12
            for m in reference.measures:
                for n in m.notes:
                    if not n.is_rest and n.pitch is not None:
                        ref_counts[n.pitch % 12] += 1
            ref_total = sum(ref_counts)
            if ref_total == 0:
                return 0.5
            ref_dist = [c / ref_total for c in ref_counts]
            kl = kl_divergence(dist, ref_dist)
        else:
            # 与基准对比
            bm_dist = self.group_data.get('pitch_class_dist')
            if bm_dist:
                kl = kl_divergence(dist, bm_dist)
            else:
                # fallback: 用本曲调性的 K-S profile
                ks = self._detect_key_profile(score)
                if ks:
                    kl = kl_divergence(dist, ks)
                else:
                    return 0.5  # 无法判断

        return math.exp(-kl * 5)

    def _interval_dist_kl_score(self, score: Score) -> float:
        """音程分布 KL 散度 → score。"""
        intervals = []
        for m in score.measures:
            notes = [n for n in m.notes if not n.is_rest and n.pitch is not None]
            for i in range(len(notes) - 1):
                interval = min(abs(notes[i + 1].pitch - notes[i].pitch), 24)
                intervals.append(interval)

        if not intervals:
            return 0.0

        # 构建分布（0-24 半音，25 bins）
        bins = 25
        counts = [0] * bins
        for iv in intervals:
            counts[min(iv, bins - 1)] += 1
        total = len(intervals)
        dist = [c / total for c in counts]

        # 基准：假设一个温和的期望分布（级进为主）
        expected = [0.10] * bins
        expected[0] = 0.05   # 同音
        expected[1] = 0.15   # 小二度
        expected[2] = 0.20   # 大二度（级进最多）
        expected[3] = 0.08   # 小三度
        expected[4] = 0.08   # 大三度
        expected[5] = 0.05   # 纯四度
        expected[7] = 0.07   # 纯五度
        expected[12] = 0.05  # 八度
        # 重归一化
        exp_total = sum(expected)
        expected = [e / exp_total for e in expected]

        bm_dist = self.group_data.get('interval_dist')
        if bm_dist:
            # 从基准 dict 转 list
            ref = [bm_dist.get(str(i), 0.001) for i in range(bins)]
            ref_total = sum(ref)
            ref = [v / ref_total for v in ref]
            kl = kl_divergence(dist, ref)
        else:
            kl = kl_divergence(dist, expected)

        return math.exp(-kl * 3)

    def _density_z_score(self, score: Score,
                         reference: Score | None = None) -> float:
        """音符密度 Z-score → score。"""
        total_notes = sum(1 for m in score.measures for n in m.notes if not n.is_rest)
        dur = score_to_duration_seconds(score)

        if reference is not None:
            # 与参考 Score 对比密度
            ref_notes = sum(1 for m in reference.measures for n in m.notes if not n.is_rest)
            ref_dur = score_to_duration_seconds(reference)
            density = total_notes / max(dur, 0.01)
            ref_density = ref_notes / max(ref_dur, 0.01)
            if ref_density > 0:
                ratio = density / ref_density
                return 1.0 / (1.0 + abs(ratio - 1.0) * 3)
            return 0.5
        else:
            density = total_notes / max(dur, 0.01)
            bm_stats = self.group_data.get('note_density', {})
            if bm_stats:
                mean = bm_stats.get('mean', 8.0)
                std = max(bm_stats.get('std', 4.0), 0.01)
            else:
                mean, std = 8.0, 4.0
            z = (density - mean) / std
            return 1.0 / (1.0 + abs(z))

    def _key_consistency_score(self, score: Score) -> float:
        """Krumhansl-Schmuckler key finding — 调性稳定性。"""
        if len(score.measures) <= 1:
            return 1.0

        # 对每小节找最匹配的调性
        keys_per_measure = []
        for m in score.measures:
            profile = [0.0] * 12
            for n in m.notes:
                if not n.is_rest and n.pitch is not None:
                    profile[n.pitch % 12] += 1.0
            total_pc = sum(profile)
            if total_pc > 0:
                profile = [p / total_pc for p in profile]
                key = self._ks_match(profile)
                keys_per_measure.append(key)

        if not keys_per_measure:
            return 0.5

        # 调性变化次数
        changes = 0
        for i in range(1, len(keys_per_measure)):
            if keys_per_measure[i] != keys_per_measure[i - 1]:
                changes += 1

        stability = 1.0 - changes / (len(keys_per_measure) - 1)
        # 调性变化太多或太少都有问题
        if stability < 0.3:
            stability *= 0.5  # 频繁转调惩罚
        return max(0.0, stability)

    def _self_similarity_score(self, score: Score) -> float:
        """自相似性矩阵 — 块结构强度。"""
        block_size = 4  # 每块 4 小节
        n_measures = len(score.measures)
        if n_measures < 8:
            return 0.5  # 太短无法判断

        # 分块并提取特征
        blocks = []
        for i in range(0, n_measures, block_size):
            block_measures = score.measures[i:i + block_size]
            if len(block_measures) < 2:
                continue

            profile = [0.0] * 12
            density = 0.0
            for m in block_measures:
                for n in m.notes:
                    if not n.is_rest and n.pitch is not None:
                        profile[n.pitch % 12] += 1.0
                    if not n.is_rest:
                        density += 1.0

            total_pc = sum(profile)
            if total_pc > 0:
                profile = [p / total_pc for p in profile]

            # 特征 = pitch_class_profile + [norm_density]
            features = profile + [density / max(len(block_measures), 1)]
            blocks.append(features)

        if len(blocks) < 3:
            return 0.5

        # 余弦相似度矩阵
        n = len(blocks)
        diag_sum = 0.0
        off_diag_sum = 0.0
        off_diag_count = 0

        for i in range(n):
            for j in range(n):
                sim = cosine_similarity(blocks[i], blocks[j])
                if i == j:
                    diag_sum += sim
                else:
                    off_diag_sum += sim
                    off_diag_count += 1

        diag_mean = diag_sum / n
        off_diag_mean = off_diag_sum / max(off_diag_count, 1)

        if off_diag_mean > 0:
            strength = diag_mean / max(off_diag_mean, 0.01)
        else:
            strength = 2.0

        # 截断到 [0, 1]
        return min(1.0, strength / 3.0)

    def _token_type_kl_score(self, score: Score) -> float:
        """Token 类型分布 KL → score。

        从 Score 估计各类 token 占比（note/rest/duration/articulation）
        与基准的 token_type_dist 对比。
        """
        # 从 Score 中统计"伪 token 类型"
        note_count = 0
        rest_count = 0
        art_count = 0
        tie_count = 0
        total = 0

        for m in score.measures:
            for n in m.notes:
                total += 1
                if n.is_rest:
                    rest_count += 1
                else:
                    note_count += 1
                if n.articulation:
                    art_count += len(n.articulation)
                if n.is_tie_start:
                    tie_count += 1

        if total == 0:
            return 0.0

        # 构建分布：note, rest, articulation, tie
        dist = {
            'note': note_count / total,
            'rest': rest_count / total,
            'articulation': art_count / max(total, 1),
            'tie': tie_count / max(total, 1),
        }

        # 与基准对比
        bm_dist = self.group_data.get('token_type_dist', {})
        if not bm_dist:
            return 0.5

        # KL 需要两个分布维度对齐
        all_keys = sorted(set(list(dist.keys()) + list(bm_dist.keys())))
        p = [dist.get(k, 0.001) for k in all_keys]
        q = [bm_dist.get(k, 0.001) for k in all_keys]

        # 归一化
        p_total = sum(p)
        q_total = sum(q)
        p = [v / p_total for v in p]
        q = [v / q_total for v in q]

        kl = kl_divergence(p, q)
        return math.exp(-kl * 3)

    # ── 新增指标：节奏 ──────────────────────────────────────

    def _duration_entropy_score(self, score: Score) -> float:
        """时值分布熵 — 衡量节奏多样性。

        将时值归入标准类别，计算 Shannon 熵。
        熵过低 → 节奏单一；熵过高 → 节奏混乱。
        """
        from math import log2

        thresholds = [0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
        counts = [0] * (len(thresholds) + 1)
        total = 0
        for m in score.measures:
            for n in m.notes:
                if not n.is_rest and n.duration > 0:
                    placed = False
                    for i, t in enumerate(thresholds):
                        if abs(n.duration - t) < t * 0.15:
                            counts[i] += 1
                            placed = True
                            break
                    if not placed:
                        counts[-1] += 1
                    total += 1

        if total < 2:
            return 0.2

        probs = [c / total for c in counts if c > 0]
        if len(probs) < 2:
            return 0.3

        entropy = -sum(p * log2(p) for p in probs)
        max_entropy = log2(len(counts))
        normalized = entropy / max(max_entropy, 0.01)

        # 最佳区域 0.3-0.6，峰值 0.45
        if 0.3 <= normalized <= 0.6:
            return 1.0
        elif normalized < 0.3:
            return 0.3 + normalized * 2.33
        else:
            return max(0.0, 1.0 - (normalized - 0.6) * 2.5)

    # ── 新增指标：音高 ──────────────────────────────────────

    def _pitch_entropy_score(self, score: Score) -> float:
        """12 音级熵 — 衡量音高使用广度。

        过低的熵意味着只用少数几个音，过高意味着近乎无调性均匀分布。
        """
        from math import log2

        counts = [0.0] * 12
        total = 0
        for m in score.measures:
            for n in m.notes:
                if not n.is_rest and n.pitch is not None:
                    counts[n.pitch % 12] += 1.0
                    total += 1

        if total < 3:
            return 0.1

        probs = [c / total for c in counts if c > 0]
        if len(probs) < 3:
            return 0.2

        entropy = -sum(p * log2(p) for p in probs)
        max_entropy = log2(12)  # ≈ 3.585
        normalized = entropy / max_entropy

        # 最佳 0.55-0.85（使用 7-10 个音级）
        if 0.55 <= normalized <= 0.85:
            return 1.0
        elif normalized < 0.55:
            return normalized / 0.55
        else:
            return max(0.0, 1.0 - (normalized - 0.85) * 4.0)

    # ── 新增指标：协和度 ────────────────────────────────────

    def _dissonance_ratio_score(self, score: Score) -> float:
        """不协和音程比例 → score。

        统计旋律中 m2/M7/TT（高度不协和）的比例。
        比例过低=缺乏张力，过高=刺耳。
        """
        dissonant_classes = {1, 2, 6, 10, 11}  # m2, M2, TT, m7, M7 in interval class
        # 实际我们只关心高度不协和：m2(1), M7(11), TT(6)
        high_dissonance = {1, 6, 11}

        total_intervals = 0
        dissonant_count = 0
        for m in score.measures:
            notes = [n for n in m.notes if not n.is_rest and n.pitch is not None]
            for i in range(len(notes) - 1):
                interval = abs(notes[i + 1].pitch - notes[i].pitch) % 12
                if interval in high_dissonance:
                    dissonant_count += 1
                total_intervals += 1

        if total_intervals < 3:
            return 0.5

        ratio = dissonant_count / total_intervals
        # 理想范围 0.02-0.15
        if 0.02 <= ratio <= 0.15:
            return 1.0
        elif ratio < 0.02:
            return 0.7  # 有点太协和
        else:
            return max(0.0, 1.0 - (ratio - 0.15) * 6.0)

    def _chromaticism_index_score(self, score: Score) -> float:
        """半音化指数 — 调外音比例。

        检测当前调式的音阶，统计调外音占比。
        适量的半音化是表现力的重要来源（如肖邦）。
        """
        # 确定主导调性
        if len(score.measures) < 2:
            return 0.5

        # 用 K-S 算法找最匹配的调性
        key_profile = self._detect_key_profile(score)
        if key_profile is None:
            return 0.5

        # 确定调性后，找该调的自然音阶
        total_profile = [0.0] * 12
        total_notes = 0
        for m in score.measures:
            for n in m.notes:
                if not n.is_rest and n.pitch is not None:
                    total_profile[n.pitch % 12] += 1.0
                    total_notes += 1

        if total_notes < 5:
            return 0.5

        # 找主导调性
        key_name = self._ks_match([p / total_notes for p in total_profile])
        diatonic = self._diatonic_set(key_name)
        if diatonic is None:
            return 0.5

        chromatic_count = sum(total_profile[i] for i in range(12) if i not in diatonic)
        ratio = chromatic_count / total_notes

        # 理想范围 0.02-0.20
        if 0.02 <= ratio <= 0.20:
            return 1.0
        elif ratio < 0.02:
            return 0.75
        else:
            return max(0.0, 1.0 - (ratio - 0.20) * 4.0)

    @staticmethod
    def _diatonic_set(key_name: str) -> set[int] | None:
        """返回调性的自然音阶 pitch class set。"""
        major_scale = [0, 2, 4, 5, 7, 9, 11]     # W-W-H-W-W-W-H
        minor_scale = [0, 2, 3, 5, 7, 8, 10]     # W-H-W-W-H-W-W (natural minor)

        tonic_map = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
                     'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
                     'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11}
        parts = key_name.split('_')
        tonic_name = parts[0]
        mode = parts[1] if len(parts) > 1 else 'major'
        tonic = tonic_map.get(tonic_name)
        if tonic is None:
            return None
        scale = major_scale if mode == 'major' else minor_scale
        return {(tonic + s) % 12 for s in scale}

    # ── 新增指标：音域 ──────────────────────────────────────

    def _register_span_score(self, score: Score) -> float:
        """音域利用率 — pitch 跨度 / 钢琴全音域 88 键。"""
        pitches = []
        for m in score.measures:
            for n in m.notes:
                if not n.is_rest and n.pitch is not None:
                    pitches.append(n.pitch)

        if len(pitches) < 3:
            return 0.1

        span = max(pitches) - min(pitches)
        ratio = span / 87.0  # 归一化到 0-1

        # 理想跨度 0.25-0.75 (约 2-5 个八度)
        if 0.25 <= ratio <= 0.75:
            return 1.0
        elif ratio < 0.25:
            return max(0.1, ratio / 0.25)
        else:
            return max(0.2, 1.0 - (ratio - 0.75) * 2.0)

    # ── 新增指标：节奏切分 ──────────────────────────────────

    def _syncopation_ratio_score(self, score: Score) -> float:
        """切分音比例 — 弱拍起音 / 总起音数。

        弱拍 = 非正拍位置（beat 的 1/2, 1/4 等位置）。
        """
        sync_count = 0
        total_onsets = 0
        for m in score.measures:
            beats, unit = m.time_signature
            beat_dur = 4.0 / unit  # 每拍在归一化时值中的长度
            for n in m.notes:
                if n.is_rest:
                    continue
                total_onsets += 1
                pos_in_beat = (n.onset % beat_dur) / beat_dur if beat_dur > 0 else 0
                # 距正拍 > 0.15 拍视为切分（不是正拍上的音）
                if pos_in_beat > 0.15 and pos_in_beat < 0.85:
                    sync_count += 1

        if total_onsets < 3:
            return 0.5

        ratio = sync_count / total_onsets
        # 理想范围 0.05-0.35
        if 0.05 <= ratio <= 0.35:
            return 1.0
        elif ratio < 0.05:
            return 0.6
        else:
            return max(0.0, 1.0 - (ratio - 0.35) * 3.0)

    # ── 新增指标：力度 ──────────────────────────────────────

    def _dynamic_variance_score(self, score: Score) -> float:
        """力度变化程度 — 速度值的变异系数 CV = std/mean。"""
        velocities = []
        for m in score.measures:
            for n in m.notes:
                if not n.is_rest and n.velocity > 0:
                    velocities.append(float(n.velocity))

        if len(velocities) < 3:
            return 0.3

        mean_v = sum(velocities) / len(velocities)
        if mean_v == 0:
            return 0.0
        variance = sum((v - mean_v) ** 2 for v in velocities) / len(velocities)
        std_v = math.sqrt(variance)
        cv = std_v / mean_v

        # CV 理想范围 0.08-0.40
        if 0.08 <= cv <= 0.40:
            return 1.0
        elif cv < 0.08:
            return max(0.1, cv / 0.08)
        else:
            return max(0.0, 1.0 - (cv - 0.40) * 2.0)

    # ── 新增指标：休止 ──────────────────────────────────────

    def _rest_ratio_score(self, score: Score) -> float:
        """休止符比例 — 休止事件 / 总事件数。

        适当的休止对分句和呼吸至关重要。
        """
        rest_count = 0
        total = 0
        for m in score.measures:
            for n in m.notes:
                total += 1
                if n.is_rest:
                    rest_count += 1

        if total < 3:
            return 0.5

        ratio = rest_count / total
        # 理想范围 0.03-0.25
        if 0.03 <= ratio <= 0.25:
            return 1.0
        elif ratio < 0.03:
            return 0.7
        else:
            return max(0.0, 1.0 - (ratio - 0.25) * 3.0)

    # ── 新增指标：旋律方向 ──────────────────────────────────

    def _melodic_direction_score(self, score: Score) -> float:
        """旋律方向变化率 — 衡量旋律轮廓的多样性。

        方向变化次数 / (音符数 - 2)。完全直线上升得 0，频繁变换得高分。
        """
        directions = []
        for m in score.measures:
            notes = [n for n in m.notes if not n.is_rest and n.pitch is not None]
            for i in range(len(notes) - 1):
                diff = notes[i + 1].pitch - notes[i].pitch
                if diff > 0:
                    directions.append(1)    # 上行
                elif diff < 0:
                    directions.append(-1)   # 下行
                else:
                    directions.append(0)    # 同音

        if len(directions) < 3:
            return 0.5

        # 方向变化次数
        changes = 0
        for i in range(1, len(directions)):
            if directions[i] != 0 and directions[i] != directions[i - 1]:
                changes += 1

        ratio = changes / max(len(directions) - 1, 1)
        # 理想变化率 0.25-0.65
        if 0.25 <= ratio <= 0.65:
            return 1.0
        elif ratio < 0.25:
            return max(0.1, ratio / 0.25)
        else:
            return max(0.0, 1.0 - (ratio - 0.65) * 3.0)

    def _contour_arc_score(self, score: Score) -> float:
        """旋律拱形结构 — 检测上行后下行（或反之）的古典拱形。

        统计 4 音符滑动窗口中"上下"或"下上"模式的比例。
        """
        patterns = []
        for m in score.measures:
            notes = [n for n in m.notes if not n.is_rest and n.pitch is not None]
            for i in range(len(notes) - 3):
                d1 = notes[i + 1].pitch - notes[i].pitch
                d2 = notes[i + 2].pitch - notes[i + 1].pitch
                d3 = notes[i + 3].pitch - notes[i + 2].pitch
                # 拱形：↑↑↓ 或 ↓↓↑
                if (d1 > 0 and d2 > 0 and d3 < 0) or (d1 < 0 and d2 < 0 and d3 > 0):
                    patterns.append(1)
                else:
                    patterns.append(0)

        if not patterns:
            return 0.6

        ratio = sum(patterns) / len(patterns)
        # 理想范围 0.05-0.30（偶有拱形即好）
        if ratio >= 0.05:
            return min(1.0, ratio / 0.15)
        else:
            return 0.4 + ratio * 6.0

    # ── 新增指标：织体 ──────────────────────────────────────

    def _polyphony_mean_score(self, score: Score) -> float:
        """同时发音数均值 — 衡量织体厚度。

        对每拍快照统计同时发声的音符数，取均值。
        """
        snapshots = _snapshot_polyphony(score)
        if not snapshots:
            return 0.0

        mean_poly = sum(snapshots) / len(snapshots)

        # 钢琴音乐理想范围 2.0-6.0
        if 2.0 <= mean_poly <= 6.0:
            return 1.0
        elif mean_poly < 2.0:
            return max(0.1, mean_poly / 2.0)
        else:
            return max(0.0, 1.0 - (mean_poly - 6.0) * 0.3)

    def _texture_variance_score(self, score: Score) -> float:
        """织体变化程度 — 同时发音数的方差 / 均值。

        衡量厚薄织体交替使用的程度。方差适中 = 富有层次变化。
        """
        snapshots = _snapshot_polyphony(score)
        if len(snapshots) < 3:
            return 0.3

        mean_poly = sum(snapshots) / len(snapshots)
        if mean_poly == 0:
            return 0.0
        variance = sum((s - mean_poly) ** 2 for s in snapshots) / len(snapshots)
        cv = math.sqrt(variance) / mean_poly

        # CV 理想范围 0.2-0.7
        if 0.2 <= cv <= 0.7:
            return 1.0
        elif cv < 0.2:
            return max(0.2, cv / 0.2)
        else:
            return max(0.0, 1.0 - (cv - 0.7) * 1.5)

    # ── 新增指标：和声节奏 ──────────────────────────────────

    def _harmonic_rhythm_score(self, score: Score) -> float:
        """和声节奏 — 平均每小节和弦变化次数。"""
        from chopinote_evaluator.general.harmony import analyze_harmony

        harmony = analyze_harmony(score)
        if not harmony.chords or len(score.measures) < 2:
            return 0.5

        chords_per_meas = len(harmony.chords) / len(score.measures)
        # 理想范围 0.5-2.5 个和弦/小节
        if 0.5 <= chords_per_meas <= 2.5:
            return 1.0
        elif chords_per_meas < 0.5:
            return max(0.2, chords_per_meas / 0.5)
        else:
            return max(0.0, 1.0 - (chords_per_meas - 2.5) * 0.5)

    # ── 辅助方法 ────────────────────────────────────────────

    @staticmethod
    def _ks_match(profile: list[float]) -> str:
        """用 K-S 算法找最匹配的调性。"""
        best_key = 'C_major'
        best_corr = -1.0

        for key_name in list(KS_MAJOR_KEYS) + list(KS_MINOR_KEYS):
            ks_profile = get_ks_profile(key_name)
            if ks_profile is None:
                continue

            # 皮尔逊相关
            n = len(profile)
            sum_p = sum(profile)
            sum_ks = sum(ks_profile)
            sum_pp = sum(x * x for x in profile)
            sum_ksks = sum(x * x for x in ks_profile)
            sum_pks = sum(profile[i] * ks_profile[i] for i in range(n))

            denom = math.sqrt((sum_pp - sum_p * sum_p / n) * (sum_ksks - sum_ks * sum_ks / n))
            if denom == 0:
                continue
            corr = (sum_pks - sum_p * sum_ks / n) / denom

            if corr > best_corr:
                best_corr = corr
                best_key = key_name

        return best_key

    @staticmethod
    def _detect_key_profile(score: Score) -> list[float] | None:
        """检测乐谱主调性，返回对应的 K-S profile。"""
        profile = [0.0] * 12
        for m in score.measures:
            for n in m.notes:
                if not n.is_rest and n.pitch is not None:
                    profile[n.pitch % 12] += 1.0
        total = sum(profile)
        if total == 0:
            return None
        profile = [p / total for p in profile]

        # 找最匹配的调性
        key = StatisticalEvaluator._ks_match(profile)
        return get_ks_profile(key)


# K-S key 列表
KS_MAJOR_KEYS = ['C_major', 'G_major', 'D_major', 'A_major', 'E_major', 'B_major',
                  'F#_major', 'C#_major', 'F_major', 'Bb_major', 'Eb_major', 'Ab_major']

KS_MINOR_KEYS = ['A_minor', 'E_minor', 'B_minor', 'F#_minor', 'C#_minor', 'G#_minor',
                  'D#_minor', 'A#_minor', 'D_minor', 'G_minor', 'C_minor', 'F_minor']


# ── 模块级辅助 ─────────────────────────────────────────────


def _snapshot_polyphony(score: Score) -> list[int]:
    """每 1/4 拍采样同时发声的音符数，返回快照列表。"""
    from chopinote_evaluator.parser import score_to_duration_seconds

    dur_sec = score_to_duration_seconds(score)
    if dur_sec <= 0:
        return []

    # 每 1/4 拍采一次样
    tempo = score.tempo or 120
    beat_sec = 60.0 / tempo
    step_sec = beat_sec / 4.0

    # 将音符展开为 (start_sec, end_sec) 列表
    events = []
    current_time = 0.0
    for m in score.measures:
        beats, unit = m.time_signature
        measure_dur_sec = beats * (60.0 / tempo) * (4.0 / unit)
        for n in m.notes:
            if not n.is_rest:
                onset_frac = n.onset / (beats * 4.0 / unit) if (beats * 4.0 / unit) > 0 else 0
                dur_frac = n.duration / (beats * 4.0 / unit) if (beats * 4.0 / unit) > 0 else 0
                start = current_time + onset_frac * measure_dur_sec
                end = start + dur_frac * measure_dur_sec
                events.append((start, end))
        current_time += measure_dur_sec

    if not events:
        return []

    max_time = max(e[1] for e in events)
    if max_time <= 0:
        return []

    snapshots = []
    t = 0.0
    while t < max_time:
        count = sum(1 for s, e in events if s <= t < e)
        snapshots.append(count)
        t += step_sec

    return snapshots
