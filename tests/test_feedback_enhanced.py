"""ABC Engine v2 回归测试：A3 指标 + B 决策 + C 评分。

不需要 GPU，所有测试基于合成 token 序列和 rule-based 逻辑。
"""
import pytest
import math
from chopinote_abc.metrics import (
    unison_chain_check, rest_streak_check, mono_rhythm_check,
    extreme_density_check, bar_boundary_melody, melodic_contour_match,
    compute_metric, compute_all_metrics, METRIC_FUNCTIONS,
    density_z_score, pitch_class_kl, interval_kl,
    rest_ratio_score, velocity_consistency, dissonance_ratio,
    syncopation_ratio, duration_entropy, register_span,
    melodic_direction, key_consistency, pitch_range_check,
    empty_measure_check, max_polyphony_check,
    parallel_fifths_check, token_type_kl,
    chord_melody_alignment, progression_validity,
    cadence_quality, harmonic_rhythm_score,
)
from chopinote_abc.constraints import Violation, evaluate_theory


# ═══════════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════════

def _make_note_sequence(tokenizer, intervals, velocity=4, duration=8):
    """合成连续的 Note_ON/Bar/Position 序列。"""
    tokens = []
    bar = tokenizer.encode_token('<Bar>')
    pos = tokenizer.encode_token('<Position 0>')
    vel = tokenizer.encode_token(f'<Velocity {velocity}>')
    dur = tokenizer.encode_token(f'<Duration {duration}>')
    for iv in intervals:
        tokens.append(bar)
        tokens.append(pos)
        tokens.append(tokenizer.encode_token(f'<Note_ON {iv}>'))
        tokens.append(vel)
        tokens.append(dur)
    return tokens


# ═══════════════════════════════════════════════════════════════
#  Token 级指标测试
# ═══════════════════════════════════════════════════════════════

class TestTokenMetrics:
    def test_unison_chain_normal(self, tokenizer):
        """正常旋律无同音连续。"""
        tokens = _make_note_sequence(tokenizer, [60, 62, 64, 65])
        score = unison_chain_check(tokens, tokenizer)
        assert score == 1.0

    def test_unison_chain_triggers(self, tokenizer):
        """连续 10 个同音应扣分。"""
        score = unison_chain_check(_make_note_sequence(tokenizer, [0] * 10), tokenizer)
        assert score < 0.6

    def test_rest_streak_normal(self, tokenizer):
        """单个休止不触发。"""
        bar = tokenizer.encode_token('<Bar>')
        rest = tokenizer.encode_token('<Rest>')
        dur = tokenizer.encode_token('<Duration 8>')
        tokens = [bar, tokenizer.encode_token('<Position 0>'),
                  tokenizer.encode_token('<Note_ON 60>'), dur, rest]
        score = rest_streak_check(tokens, tokenizer)
        assert score == 1.0

    def test_rest_streak_triggers(self, tokenizer):
        """连续 6 个 Rest 应扣分。"""
        rest = tokenizer.encode_token('<Rest>')
        score = rest_streak_check([rest] * 6, tokenizer)
        assert score < 0.6

    def test_extreme_density(self, tokenizer):
        """单小节 >40 notes 应触发 — 需要至少 2 个 bar 才能检测。"""
        bar = tokenizer.encode_token('<Bar>')
        note_on = tokenizer.encode_token('<Note_ON 60>')
        # 构造 2 个 bar，第 2 个 bar 有 50 个 note
        tokens = [bar, tokenizer.encode_token('<Note_ON 62>')] + [bar] + [note_on] * 50
        score = extreme_density_check(tokens, tokenizer)
        assert score < 0.3

    def test_density_z_score(self, tokenizer):
        """密度 Z-score 在正常范围。"""
        tokens = _make_note_sequence(tokenizer, [60, 62, 64, 67, 65, 62, 60])
        score = density_z_score(tokens, tokenizer)
        assert 0 <= score <= 1

    def test_metric_registry_complete(self):
        """指标注册表完整性。"""
        assert len(METRIC_FUNCTIONS) >= 25
        assert 'density_z' in METRIC_FUNCTIONS
        assert 'pitch_class_kl' in METRIC_FUNCTIONS
        assert 'chord_melody_alignment' in METRIC_FUNCTIONS
        assert 'cadence_quality' in METRIC_FUNCTIONS
        assert 'parallel_fifths' in METRIC_FUNCTIONS
        assert 'unison_chain' in METRIC_FUNCTIONS
        assert 'rest_streak' in METRIC_FUNCTIONS
        assert 'melodic_contour' in METRIC_FUNCTIONS

    def test_compute_metric_by_name(self, tokenizer):
        """按名称调用指标。"""
        tokens = _make_note_sequence(tokenizer, [60, 62])
        val = compute_metric('density_z', tokens, tokenizer)
        assert val is not None
        assert 0 <= val <= 1

    def test_compute_all_metrics(self, tokenizer):
        """计算全部指标。"""
        tokens = _make_note_sequence(tokenizer, [60, 62, 64, 67, 65, 62, 60])
        results = compute_all_metrics(tokens, tokenizer)
        assert len(results) >= 15
        assert all(0 <= v <= 1 for v in results.values())

    def test_bar_boundary_melody(self, tokenizer):
        """小节边界旋律衔接。"""
        tokens = _make_note_sequence(tokenizer, [60, 62, 64, 67, 65])
        score = bar_boundary_melody(tokens, tokenizer)
        assert score >= 0.4  # 级进旋律应该不低

    def test_melodic_contour_no_seed(self, tokenizer):
        """无 seed 参考时返回中性分。"""
        tokens = _make_note_sequence(tokenizer, [60, 65])
        score = melodic_contour_match(tokens, tokenizer, seed_contour=None)
        assert score == 0.7


# ═══════════════════════════════════════════════════════════════
#  和弦指标测试
# ═══════════════════════════════════════════════════════════════

class TestChordMetrics:
    def test_cadence_quality_perfect(self, tokenizer):
        """V-I 应识别为完全终止。"""
        tokens = []
        for chord in ['IV', 'V', 'I']:
            tokens.append(tokenizer.encode_token(f'<Chord {chord}>'))
        score = cadence_quality(tokens, tokenizer)
        assert score >= 0.9

    def test_progression_validity(self, tokenizer):
        """I-IV-V-I 合理进行应得中等分。"""
        tokens = []
        for chord in ['I', 'IV', 'V', 'I']:
            tokens.append(tokenizer.encode_token(f'<Chord {chord}>'))
        score = progression_validity(tokens, tokenizer)
        assert score >= 0.4  # I-IV(1.0) + IV-V(0.5) + V-I(1.0) / 3 ≈ 0.83


# ═══════════════════════════════════════════════════════════════
#  B 决策层测试
# ═══════════════════════════════════════════════════════════════

class TestBDecision:
    def test_bhardbans_init(self):
        """BHardBans 初始化为空。"""
        from chopinote_abc.decision import BHardBans
        bans = BHardBans()
        assert bans.has_bans() == False

    def test_bhardbans_ban_note(self):
        """添加硬禁 token。"""
        from chopinote_abc.decision import BHardBans
        bans = BHardBans()
        bans.parallel_fifths.update([1, 2, 3])
        assert bans.has_bans() == True

    def test_zone_temperature_callable(self):
        """温区退火可调用。"""
        from chopinote_abc.decision import apply_zone_temperature
        assert callable(apply_zone_temperature)


# ═══════════════════════════════════════════════════════════════
#  Tokenizer 兼容
# ═══════════════════════════════════════════════════════════════

class TestTokenizer:
    def test_vocab_size(self, tokenizer):
        assert tokenizer.vocab_size == 929
