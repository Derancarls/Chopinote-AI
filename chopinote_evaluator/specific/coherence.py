"""结构衔接 — 衡量种子和生成之间的结构连续程度。

包括调性匹配、声部数延续、速度连续性、终止式匹配。
"""

from __future__ import annotations

from chopinote_evaluator.parser import Score


class CoherenceEvaluator:
    """结构衔接评估器。"""

    def evaluate(self, seed: Score, continuation: Score) -> dict[str, float]:
        """评估结构衔接。

        返回:
            {指标名: score (0~1)}
        """
        return {
            'key_match': self._key_match(seed, continuation),
            'voice_count_delta': self._voice_count_delta(seed, continuation),
            'tempo_continuity': self._tempo_continuity(seed, continuation),
            'cadence_match': self._cadence_match(seed, continuation),
        }

    def aggregate(self, scores: dict[str, float]) -> float:
        """衔接分数加权平均。"""
        weights = {
            'key_match': 0.35,
            'voice_count_delta': 0.25,
            'tempo_continuity': 0.25,
            'cadence_match': 0.15,
        }
        total_w = 0.0
        weighted = 0.0
        for k, w in weights.items():
            if k in scores:
                weighted += scores[k] * w
                total_w += w
        return weighted / max(total_w, 0.001)

    # ── 指标实现 ──────────────────────────────────────────

    @staticmethod
    def _key_match(seed: Score, continuation: Score) -> float:
        """调性匹配：种子最后 2 小节 vs 生成前 2 小节的调性一致则 1.0。"""
        seed_key = _detect_key_segment(seed, -2)
        gen_key = _detect_key_segment(continuation, 2)

        if seed_key and gen_key:
            return 1.0 if seed_key == gen_key else 0.0

        # fallback: 比较 key_signature
        def last_key(score, n_back=2):
            for m in reversed(score.measures[-n_back:] if len(score.measures) > n_back else score.measures):
                if m.key_signature and m.key_signature != 'unknown':
                    return m.key_signature
            return None

        sk = last_key(seed)
        gk = last_key(continuation)
        if sk and gk:
            return 1.0 if sk.split('_')[0] == gk.split('_')[0] else 0.0

        return 0.5  # 无法判断

    @staticmethod
    def _voice_count_delta(seed: Score, continuation: Score) -> float:
        """活跃声部数量差。"""
        def active_voices(score):
            voices = set()
            for m in score.measures:
                for n in m.notes:
                    if not n.is_rest:
                        voices.add((n.staff, n.voice))
            return len(voices)

        delta = abs(active_voices(seed) - active_voices(continuation))
        return max(0.0, 1.0 - delta * 0.5)

    @staticmethod
    def _tempo_continuity(seed: Score, continuation: Score) -> float:
        """速度变化比例。"""
        t1 = seed.tempo or 120
        t2 = continuation.tempo or 120
        if t1 <= 0:
            t1 = 120
        if t2 <= 0:
            t2 = 120
        ratio = t2 / t1
        if 0.8 <= ratio <= 1.25:
            return 1.0
        return max(0.0, 1.0 - min(abs(ratio - 1.0) * 2.0, 1.0))

    @staticmethod
    def _cadence_match(seed: Score, continuation: Score) -> float:
        """种子尾部终止式与续写开头的和声连贯性。

        检测种子末尾的和弦进行，判断终止式类型，
        并与续写首部的调性/和弦进行对比。
        """
        from chopinote_evaluator.general.harmony import analyze_harmony, CHORD_TEMPLATES

        # 取种子最后 4 小节的和声分析
        seed_tail = Score(
            measures=seed.measures[-4:] if len(seed.measures) > 4 else seed.measures,
            tempo=seed.tempo, programs=seed.programs,
        )
        gen_head = Score(
            measures=continuation.measures[:4] if len(continuation.measures) > 4 else continuation.measures,
            tempo=continuation.tempo, programs=continuation.programs,
        )

        seed_harmony = analyze_harmony(seed_tail)
        gen_harmony = analyze_harmony(gen_head)

        # 获取种子末尾的终止式类型
        seed_cadence = _detect_cadence(seed_harmony)

        # 获取种子和生成的主调性
        seed_key = _detect_key_segment(seed, -2)
        gen_key = _detect_key_segment(continuation, 2)

        # 评分逻辑
        score = 0.5  # baseline

        # 如果种子有明确终止式，续写应与之协调
        if seed_cadence == 'PAC' or seed_cadence == 'IAC':
            # 种子已稳定结束，续写可以从同调开始
            if seed_key and gen_key and seed_key == gen_key:
                score = 0.9
            else:
                score = 0.6
        elif seed_cadence == 'HC':
            # 半终止 → 续写应回到主调
            if seed_key and gen_key and seed_key == gen_key:
                score = 0.85
            else:
                score = 0.5
        elif seed_cadence == 'DC':
            # 阻碍终止 → 续写应最终解决
            score = 0.75
        else:
            # 无明显终止式，检查调性匹配
            if seed_key and gen_key and seed_key == gen_key:
                score = 0.8
            else:
                score = 0.6

        # 如有和弦 analysis 数据，增强评分
        if seed_harmony.chords and gen_harmony.chords:
            # 检查种子最后一个和弦 → 续写第一个和弦的进行
            last_chord = seed_harmony.chords[-1]
            first_chord = gen_harmony.chords[0] if gen_harmony.chords else None

            if first_chord:
                # V → I 是好的衔接
                if _is_dominant(last_chord) and _is_tonic(first_chord, seed_key):
                    score = min(1.0, score + 0.1)
                elif last_chord.root == first_chord.root and last_chord.quality == first_chord.quality:
                    score = min(1.0, score + 0.05)

        return score


# ── 辅助 ────────────────────────────────────────────────────


# ── 终止式检测辅助 ─────────────────────────────────────────


def _detect_cadence(harmony) -> str | None:
    """从和声分析结果中检测终止式类型。

    返回: 'PAC', 'IAC', 'HC', 'DC', 'PC', 或 None
    """
    if not harmony.chords or len(harmony.chords) < 2:
        return None

    # 取最后两个和弦
    second_last = harmony.chords[-2]
    last = harmony.chords[-1]

    # V-I: PAC 或 IAC
    if _is_dominant(second_last) and _is_tonic(last, None):
        # PAC: 两个和弦都是原位，且旋律在 tonic
        if second_last.inversion == 0 and last.inversion == 0:
            return 'PAC'
        return 'IAC'

    # V-vi: DC
    if _is_dominant(second_last) and _is_submediant(last):
        return 'DC'

    # 停在 V: HC
    if _is_dominant(last):
        return 'HC'

    # IV-I: PC
    if _is_subdominant(second_last) and _is_tonic(last, None):
        return 'PC'

    return None


def _is_dominant(chord) -> bool:
    """判断是否为属功能（V 或 vii°）。"""
    return chord.quality in ('M', 'dom7') or (chord.quality in ('dim', 'dim7', 'hdim7'))


def _is_tonic(chord, key_name: str | None) -> bool:
    """判断是否为主功能（I 或 vi）。"""
    return chord.quality in ('M', 'm', 'maj7', 'min7')


def _is_subdominant(chord) -> bool:
    """判断是否为下属功能（IV 或 ii）。"""
    return chord.quality in ('M', 'm', 'min7', 'maj7', 'sus4')


def _is_submediant(chord) -> bool:
    """判断是否为下中音（vi）。"""
    return chord.quality in ('m', 'min7')


def _detect_key_segment(score: Score, n_bars: int) -> str | None:
    """检测 Score 片段的主调性。

    参数:
        score: 乐谱
        n_bars: 正数=前 n 小节，负数=后 n 小节
    """
    if n_bars > 0:
        segment = score.measures[:n_bars]
    else:
        segment = score.measures[n_bars:]

    if not segment:
        return None

    # 取第一个有明确调性的小节
    for m in segment:
        if m.key_signature and m.key_signature != 'unknown':
            return m.key_signature

    return None
