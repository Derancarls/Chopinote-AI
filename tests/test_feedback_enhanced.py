"""评价模块增强测试：A 阶段蓝图 + B 阶段段落感知 + C 阶段诊断。

不需要 GPU，所有测试基于合成 token 序列和 rule-based 逻辑。
"""
import pytest
import math
from chopinote_evaluator.feedback_controller import (
    SeedSection, SectionStyleTarget, SectionStyleProfile, HarmonyContext,
    SectionPlan, SeedBlueprint, BarDiagnosis, PreGenerationEvaluator,
    NarrowFeedbackController, PostGenerationFilter, B1_ADJUSTMENT_RULES,
    B2_ADJUSTMENT_RULES, SECTION_B2_TOLERANCE,
)


# ═══════════════════════════════════════════════════════════════
#  A 阶段数据结构
# ═══════════════════════════════════════════════════════════════

class TestADataStructures:
    def test_seed_section_defaults(self):
        sec = SeedSection(type='theme1', start_bar=0, n_bars=8)
        assert sec.key == 'C'

    def test_section_style_target_defaults(self):
        target = SectionStyleTarget()
        assert target.temperature == (1.0, 0.7, 1.4)
        assert target.density_target == 8.0
        assert target.harmonic_rhythm == 16.0

    def test_section_style_profile(self):
        styles = {
            'theme1': SectionStyleTarget(temperature=(1.0, 0.8, 1.5)),
            'development': SectionStyleTarget(temperature=(1.3, 0.8, 2.0)),
        }
        profile = SectionStyleProfile(styles=styles)
        assert profile.styles['theme1'].temperature[0] == 1.0
        assert profile.styles['development'].temperature[0] == 1.3

    def test_harmony_context(self):
        ctx = HarmonyContext(
            chord_density_per_bar=[0.5, 0.3, 0.8],
            chord_density_per_bar_mean=0.53,
        )
        assert len(ctx.chord_density_per_bar) == 3
        assert ctx.chord_density_per_bar_mean == 0.53

    def test_section_plan(self):
        plan = SectionPlan(type='theme1', start_bar=0, n_bars=8, key='C')
        assert plan.min_bars == 4
        assert plan.max_bars == 32

    def test_blueprint_with_sections(self):
        blueprint = SeedBlueprint(
            sections=[SeedSection(type='intro', start_bar=0, n_bars=4)],
            form_hint='through_composed',
        )
        assert blueprint.form_hint == 'through_composed'
        assert len(blueprint.sections) == 1

    def test_bar_diagnosis(self):
        d = BarDiagnosis(bar=5, issues=['empty_measure'], severity=0.8)
        assert d.suggestion is None
        d2 = BarDiagnosis(bar=12, issues=['density_drop'], severity=0.6,
                          suggestion='检查该小节密度')
        assert d2.suggestion == '检查该小节密度'


# ═══════════════════════════════════════════════════════════════
#  A 阶段：PreGenerationEvaluator 拓展 (extract_blueprint)
# ═══════════════════════════════════════════════════════════════

class TestPreGenerationEvaluatorEnhanced:
    def test_extract_blueprint_returns_blueprint(self, tokenizer):
        """extract_blueprint 返回 SeedBlueprint 且含默认字段。"""
        from chopinote_evaluator.feedback_controller import PreGenerationEvaluator
        ev = PreGenerationEvaluator(tokenizer)
        seed = [tokenizer.BOS, tokenizer.encode_token('<Bar>'),
                tokenizer.encode_token('<Position 0>'),
                tokenizer.encode_token('<Note_ON 60>'),
                tokenizer.encode_token('<Velocity 4>'),
                tokenizer.encode_token('<Duration 8>'),
                tokenizer.EOS]
        bp = ev.extract_blueprint(seed)
        assert isinstance(bp, SeedBlueprint)
        # 即使 seed 极短，也应安全返回
        assert bp.form_hint == 'through_composed'
        assert bp.harmony.chord_density_per_bar_mean is not None

    def test_detect_sections_empty_on_short_seed(self, tokenizer):
        """短 seed 不进行段落检测。"""
        ev = PreGenerationEvaluator(tokenizer)
        bp = ev.extract_blueprint([tokenizer.BOS, tokenizer.EOS])
        assert bp.sections == []

    def test_infer_form_ternary(self, tokenizer):
        """theme1 → theme2 → theme1 被识别为 ternary。"""
        ev = PreGenerationEvaluator(tokenizer)
        sections = [
            SeedSection(type='theme1', start_bar=0, n_bars=8),
            SeedSection(type='development', start_bar=8, n_bars=8),
            SeedSection(type='theme1', start_bar=16, n_bars=8),
        ]
        form = ev._infer_form(sections)
        assert form == 'ternary'

    def test_infer_form_through_composed(self, tokenizer):
        """无法匹配的模式返回 through_composed。"""
        ev = PreGenerationEvaluator(tokenizer)
        sections = [SeedSection(type='intro', start_bar=0, n_bars=4)]
        form = ev._infer_form(sections)
        assert form == 'through_composed'

    def test_build_section_plan(self, tokenizer):
        """_build_section_plan 返回合理的排期。"""
        ev = PreGenerationEvaluator(tokenizer)
        sections = [SeedSection(type='theme1', start_bar=0, n_bars=8)]
        plan = ev._build_section_plan(sections, total_bars=32)
        assert len(plan) == 1
        assert plan[0].min_bars >= 2
        assert plan[0].max_bars >= 8
        # n_bars 应基于原始长度缩放
        assert plan[0].n_bars >= 4


# ═══════════════════════════════════════════════════════════════
#  B 阶段调整规则完整性
# ═══════════════════════════════════════════════════════════════

class TestBAdjustmentRules:
    def test_b1_has_all_sections(self):
        """B1 规则包含统计/和声/旋律所有调整项。"""
        required = {'density_z', 'dissonance_ratio', 'rest_ratio',
                    'empty_measure', 'unison_chain', 'rest_streak',
                    'progression_validity', 'harmonic_rhythm',
                    'bar_boundary_melody', 'parallel_fifths'}
        for r in required:
            assert r in B1_ADJUSTMENT_RULES, f'B1 缺少 {r}'

    def test_b2_has_all_sections(self):
        """B2 规则包含统计/和声/旋律所有调整项。"""
        required = {'pitch_class_kl', 'interval_kl', 'key_consistency',
                    'empty_measure', 'progression_validity',
                    'harmonic_rhythm', 'cadence_quality',
                    'chord_melody_alignment', 'melodic_contour'}
        for r in required:
            assert r in B2_ADJUSTMENT_RULES, f'B2 缺少 {r}'

    def test_b1_rules_have_valid_params(self):
        """B1 调整规则参数均为正确定义的参数名。"""
        valid_params = {'temperature', 'rest_penalty', 'key_bias_strength',
                        'complexity'}
        for rule_name, adjustments in B1_ADJUSTMENT_RULES.items():
            for param, delta in adjustments.items():
                assert param in valid_params, f'B1 {rule_name}: 未知参数 {param}'
                assert isinstance(delta, (int, float)), f'B1 {rule_name}: delta 应为数字'

    def test_b2_rules_have_valid_params(self):
        """B2 调整规则参数均为正确定义的参数名。"""
        valid_params = {'temperature', 'rest_penalty', 'key_bias_strength',
                        'complexity'}
        for rule_name, adjustments in B2_ADJUSTMENT_RULES.items():
            for param, delta in adjustments.items():
                assert param in valid_params, f'B2 {rule_name}: 未知参数 {param}'
                assert isinstance(delta, (int, float)), f'B2 {rule_name}: delta 应为数字'


# ═══════════════════════════════════════════════════════════════
#  SECTION_B2_TOLERANCE
# ═══════════════════════════════════════════════════════════════

class TestSectionB2Tolerance:
    def test_development_tolerance_low(self):
        """development 段落应允许大幅漂移。"""
        assert SECTION_B2_TOLERANCE.get('development', 1.0) <= 0.5

    def test_cadenza_tolerance_very_low(self):
        """cadenza 段落几乎不限制。"""
        assert SECTION_B2_TOLERANCE.get('cadenza', 1.0) <= 0.2

    def test_theme1_tolerance_strict(self):
        """theme1 段落应严格 vs seed。"""
        assert SECTION_B2_TOLERANCE.get('theme1', 0) >= 0.8

    def test_all_major_types_present(self):
        """所有主要段落类型都有容忍度定义。"""
        for t in ('exposition', 'theme1', 'theme2', 'development', 'bridge',
                  'transition', 'coda', 'cadenza', 'intro'):
            assert t in SECTION_B2_TOLERANCE, f'缺少 {t}'


# ═══════════════════════════════════════════════════════════════
#  C 阶段诊断
# ═══════════════════════════════════════════════════════════════

class TestCPhaseDiagnosis:
    def test_empty_diagnoses_list(self, tokenizer):
        """无问题时返回空列表。"""
        # 模拟一个合法性问题都没有的 report
        class MockReport:
            legality = None
            general = {}
        filter = PostGenerationFilter(tokenizer)
        diagnoses = filter.diagnose_bars(MockReport(), tokenizer)
        assert diagnoses == []

    def test_diagnose_with_legality_issues(self, tokenizer):
        """合法性问题生成 BarDiagnosis。"""
        class MockIssue:
            def __init__(self):
                self.measure = 5
            def __str__(self):
                return 'empty_measure'

        class MockLegality:
            def __init__(self):
                self.issues = [MockIssue()]
                self.passed = False

        class MockReport:
            def __init__(self):
                self.legality = MockLegality()
                self.general = {}

        filter = PostGenerationFilter(tokenizer)
        diagnoses = filter.diagnose_bars(MockReport(), tokenizer)
        assert len(diagnoses) == 1
        assert diagnoses[0].bar == 5
        assert diagnoses[0].severity == 0.8

    def test_smart_rollback_empty(self):
        """无诊断时不退回。"""
        filter = PostGenerationFilter(None)
        result = filter.smart_rollback([])
        assert result is None

    def test_smart_rollback_first_issue(self):
        """返回第一个严重问题的前一个 bar。"""
        filter = PostGenerationFilter(None)
        diagnoses = [BarDiagnosis(bar=8, issues=['empty'], severity=0.8)]
        result = filter.smart_rollback(diagnoses, min_rollback_bars=4)
        assert result == 7  # bar 8-1=7

    def test_smart_rollback_too_early(self):
        """前 4 bar 有问题是，不退回。"""
        filter = PostGenerationFilter(None)
        diagnoses = [BarDiagnosis(bar=3, issues=['empty'], severity=0.9)]
        result = filter.smart_rollback(diagnoses, min_rollback_bars=4)
        assert result is None

    def test_smart_rollback_section(self):
        """问题集中在某段落时，退回该段开头。"""
        filter = PostGenerationFilter(None)
        section_plan = [SectionPlan(type='theme1', start_bar=0, n_bars=8)]
        diagnoses = [
            BarDiagnosis(bar=5, issues=['density_drop'], severity=0.6),
            BarDiagnosis(bar=6, issues=['empty'], severity=0.7),
        ]
        result = filter.smart_rollback(diagnoses, section_plan, min_rollback_bars=4)
        # sec.start_bar=0, 0+1=1 但 1 <= 4(rollback_min), 所以转策略3
        # 策略3: first_bar=5, 5>4 → 返回 4
        assert result == 4


# ═══════════════════════════════════════════════════════════════
#  B 阶段段落感知逻辑
# ═══════════════════════════════════════════════════════════════

class TestBSectionAwareness:
    def test_get_current_section(self, tokenizer):
        """_get_current_section 正确识别段落。"""
        from chopinote_evaluator.feedback_controller import (
            NarrowFeedbackController, SeedProfile
        )
        from chopinote_model.generate import GenerationParams

        profile = SeedProfile(n_bars=8, bar_density=8.0)
        bp = SeedBlueprint(
            sections=[SeedSection(type='theme1', start_bar=0, n_bars=8)],
            section_plan=[SectionPlan(type='theme1', start_bar=0, n_bars=8)],
        )
        controller = NarrowFeedbackController(profile, tokenizer, blueprint=bp)
        current = controller._get_current_section(4)
        assert current == 'theme1'

    def test_get_current_section_none(self, tokenizer):
        """无 section_plan 时返回 None。"""
        from chopinote_evaluator.feedback_controller import (
            NarrowFeedbackController, SeedProfile
        )
        profile = SeedProfile(n_bars=8, bar_density=8.0)
        controller = NarrowFeedbackController(profile, tokenizer, blueprint=None)
        current = controller._get_current_section(4)
        assert current is None

    def test_near_boundary_entering(self, tokenizer):
        """前 2 bar 标记 entering。"""
        from chopinote_evaluator.feedback_controller import (
            NarrowFeedbackController, SeedProfile
        )
        profile = SeedProfile(n_bars=8, bar_density=8.0)
        bp = SeedBlueprint(
            section_plan=[SectionPlan(type='theme1', start_bar=0, n_bars=8)]
        )
        controller = NarrowFeedbackController(profile, tokenizer, blueprint=bp)
        assert controller._near_boundary(0) == 'entering'
        assert controller._near_boundary(1) == 'entering'

    def test_near_boundary_leaving(self, tokenizer):
        """后 2 bar 标记 leaving。"""
        from chopinote_evaluator.feedback_controller import (
            NarrowFeedbackController, SeedProfile
        )
        profile = SeedProfile(n_bars=8, bar_density=8.0)
        bp = SeedBlueprint(
            section_plan=[SectionPlan(type='theme1', start_bar=0, n_bars=8)]
        )
        controller = NarrowFeedbackController(profile, tokenizer, blueprint=bp)
        assert controller._near_boundary(6) == 'leaving'

    def test_near_boundary_internal(self, tokenizer):
        """中间 bar 返回 None。"""
        from chopinote_evaluator.feedback_controller import (
            NarrowFeedbackController, SeedProfile
        )
        profile = SeedProfile(n_bars=8, bar_density=8.0)
        bp = SeedBlueprint(
            section_plan=[SectionPlan(type='theme1', start_bar=0, n_bars=8)]
        )
        controller = NarrowFeedbackController(profile, tokenizer, blueprint=bp)
        assert controller._near_boundary(4) is None


# ═══════════════════════════════════════════════════════════════
#  B1/B2 新增指标 (registry.py)
# ═══════════════════════════════════════════════════════════════

class TestBNewMetrics:
    def test_unison_chain_normal(self, tokenizer):
        """正常音程不触发同音链警告。"""
        from chopinote_evaluator.registry import _unison_chain_tokens
        tokens = _make_note_sequence(tokenizer, [60, 62, 64, 65])
        score = _unison_chain_tokens(tokens, tokenizer)
        assert score == 1.0, f'预期 1.0, 得 {score}'

    def test_unison_chain_triggers(self, tokenizer):
        """连续 8+ 个同音（interval=0 = 主音）应扣分。"""
        from chopinote_evaluator.registry import _unison_chain_tokens
        # Note_ON 0 表示主音, 10 个连续主音应触发 (§4.4)
        tokens = _make_note_sequence(tokenizer, [0] * 10)
        score = _unison_chain_tokens(tokens, tokenizer)
        assert score < 1.0, f'预期 < 1.0, 得 {score}'

    def test_rest_streak_normal(self, tokenizer):
        """正常休止分布不触发。"""
        from chopinote_evaluator.registry import _rest_streak_tokens
        tokens = [tokenizer.encode_token('<Bar>'),
                  tokenizer.encode_token('<Position 0>'),
                  tokenizer.encode_token('<Note_ON 60>'),
                  tokenizer.encode_token('<Duration 8>'),
                  tokenizer.encode_token('<Rest>')]
        score = _rest_streak_tokens(tokens, tokenizer)
        assert score == 1.0

    def test_rest_streak_triggers(self, tokenizer):
        """连续 4+ Rest 应扣分。"""
        from chopinote_evaluator.registry import _rest_streak_tokens
        rest = tokenizer.encode_token('<Rest>')
        tokens = [rest] * 6
        score = _rest_streak_tokens(tokens, tokenizer)
        assert score < 1.0

    def test_mono_rhythm(self, tokenizer):
        """同一时值类型重复 4+ 节应降低分数。"""
        from chopinote_evaluator.registry import _mono_rhythm_tokens
        bar = tokenizer.encode_token('<Bar>')
        dur8 = tokenizer.encode_token('<Duration 8>')
        # 5 bar 全用 Duration 8
        tokens = []
        for _ in range(5):
            tokens.extend([bar, dur8])
        score = _mono_rhythm_tokens(tokens, tokenizer)
        assert score < 1.0

    def test_extreme_density_too_many(self, tokenizer):
        """单小节 >40 notes 应触发。"""
        from chopinote_evaluator.registry import _extreme_density_tokens
        bar = tokenizer.encode_token('<Bar>')
        note_on = tokenizer.encode_token('<Note_ON 60>')
        tokens = [bar] + [note_on] * 50
        score = _extreme_density_tokens(tokens, tokenizer)
        assert score < 0.5

    def test_bar_boundary_melody_smooth(self, tokenizer):
        """平滑小节边界得高分。"""
        from chopinote_evaluator.registry import _bar_boundary_melody_tokens
        bar = tokenizer.encode_token('<Bar>')
        pos = tokenizer.encode_token('<Position 0>')
        dur = tokenizer.encode_token('<Duration 8>')
        vel = tokenizer.encode_token('<Velocity 4>')
        # 连续阶梯上行（音程小于 12）
        n1 = tokenizer.encode_token('<Note_ON 60>')
        n2 = tokenizer.encode_token('<Note_ON 62>')
        n3 = tokenizer.encode_token('<Note_ON 64>')
        tokens = [bar, pos, n1, vel, dur, bar, pos, n2, vel, dur, bar, pos, n3, vel, dur]
        score = _bar_boundary_melody_tokens(tokens, tokenizer)
        assert score >= 0.5

    def test_melodic_contour_no_seed(self, tokenizer):
        """无 seed 参考时返回中性分。"""
        from chopinote_evaluator.registry import _melodic_contour_tokens
        bar = tokenizer.encode_token('<Bar>')
        tokens = [bar, tokenizer.encode_token('<Note_ON 60>')]
        score = _melodic_contour_tokens(tokens, tokenizer, seed_contour=None)
        assert score == 0.7

    def test_melodic_contour_with_seed(self, tokenizer):
        """匹配 seed 轮廓应得高分。"""
        from chopinote_evaluator.registry import _melodic_contour_tokens
        bar = tokenizer.encode_token('<Bar>')
        notes = [tokenizer.encode_token(f'<Note_ON {p}>') for p in [60, 64, 67, 72]]
        tokens = [bar] + notes
        score = _melodic_contour_tokens(tokens, tokenizer, seed_contour=[60.0, 65.0, 68.0, 72.0])
        assert score > 0.0


# ═══════════════════════════════════════════════════════════════
#  指标 Phase 注册一致性
# ═══════════════════════════════════════════════════════════════

class TestMetricRegistration:
    def test_harmonic_rhythm_in_b1(self):
        """harmonic_rhythm 注册于 B1。"""
        from chopinote_evaluator.registry import REGISTRY
        m = REGISTRY.get('harmonic_rhythm')
        assert m is not None
        assert 'B1' in str(m.phases)

    def test_progression_validity_in_b1(self):
        """progression_validity 注册于 B1。"""
        from chopinote_evaluator.registry import REGISTRY
        m = REGISTRY.get('progression_validity')
        assert m is not None
        assert 'B1' in str(m.phases)

    def test_chord_melody_alignment_in_b2(self):
        """chord_melody_alignment 注册于 B2。"""
        from chopinote_evaluator.registry import REGISTRY
        m = REGISTRY.get('chord_melody_alignment')
        assert m is not None
        assert 'B2' in str(m.phases)

    def test_cadence_quality_in_b1b2(self):
        """cadence_quality 注册于 B1,B2,C。"""
        from chopinote_evaluator.registry import REGISTRY
        m = REGISTRY.get('cadence_quality')
        assert m is not None
        assert 'B1' in str(m.phases) and 'C' in str(m.phases)

    def test_new_b1_metrics_registered(self):
        """B1 新增 4 个统计指标全部注册。"""
        from chopinote_evaluator.registry import REGISTRY
        for name in ('unison_chain', 'rest_streak', 'mono_rhythm', 'extreme_density'):
            assert name in REGISTRY, f'{name} 未注册'

    def test_new_melody_metrics_registered(self):
        """旋律相关指标已注册。"""
        from chopinote_evaluator.registry import REGISTRY
        for name in ('bar_boundary_melody', 'parallel_fifths', 'melodic_contour'):
            assert name in REGISTRY, f'{name} 未注册'


# ═══════════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════════

def _make_note_sequence(tokenizer, intervals, velocity=4, duration=8):
    """合成连续的 Note_ON 序列（无 bar/position 封装）。"""
    tokens = []
    for iv in intervals:
        tokens.append(tokenizer.encode_token(f'<Note_ON {iv}>'))
    return tokens
