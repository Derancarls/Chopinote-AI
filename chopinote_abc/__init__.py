"""ABC Engine v2 — A(感知层) B(决策层) C(进化层) 三位一体认知架构。

A1DB: 框架库（结构 + 和声 + 运行时覆盖）
A2DB: 动机摘要库（提纯 token + MotifDNA）
A3DB: 统计库（每 bar 实时统计 + 段快照 + 基线 + 创新日志）

B1: 硬约束（声部音域、平行禁止、时值溢出、音符密度）
B2: 调参（温区退火、创新预算、致命信号检测）
C:  进化（MusicXML 审查、Token↔XML 对比、DPO 偏好学习）
"""

from .database import (
    A1DB, A2DB, A3DB,
    SectionPlan, ChordAtBar, SeedContext, StructuralFix,
    MotifDNA, MotifRecord,
    BarStats, SectionStats,
    PhrasePlan, PhraseState,          # v0.3.2 gen4: 乐句层
    DramaticCurve,                    # v0.3.3-opt3: 长程张力曲线
    compute_voice_independence,        # v0.3.3-opt4: 声部独立性
    compute_novelty_bonus, compute_diversity_bonus, write_reward_log,
)
from .planner import (
    plan_structure, plan_harmony, reharmonize_from_bar,
    tonal_progression_template,
    plan_phrases_for_section,         # v0.3.2 gen4: 乐句规划
    DRAMATIC_TEMPLATES,               # v0.3.3-opt3: 曲式张力模板
    build_dramatic_curve,             # v0.3.3-opt3: 曲线生成器
)
from .motif import (
    identify_landmarks, purify_tokens, extract_dna,
    invert_contour, fragment_tokens, diminish_tokens,
    contour_distance, contour_similarity,
    MotifTransform, render_dna_to_tokens, render_dna_to_guidance,
)
from .decision import (
    BHardBans, apply_zone_temperature,
    BFeedback, InnovationEntry,
    DevelopmentAction, select_development_action,
    apply_motif_guidance, build_note_on_range,
    DramaticParams, apply_dramatic_params,  # v0.3.3-opt3: 张力曲线参数联动
    ContourBias,                              # v0.3.3-opt4: 对位方向偏置
    AFFECT_PARAM_MAP, AffectBias,            # v0.3.3-opt5: 情感B2联动
    apply_affect_bias,                        # v0.3.3-opt5: 情感→参数映射
)
from .affect import (
    AffectVector, AffectCalculator,           # v0.3.3-opt5: 八维情感计算
    AFFECT_PRESETS, STYLE_PRESETS,           # v0.3.3-opt5: 情绪/风格预置表
    parse_affective_intent,                   # v0.3.3-opt5: 自然语言→情感向量
    DIMENSION_NAMES,                          # v0.3.3-opt5: 维度名列表
)
from .metrics import (
    METRIC_FUNCTIONS,
    compute_metric,
    compute_all_metrics,
    density_z_score,
    pitch_class_kl,
    interval_kl,
    rest_ratio_score,
    velocity_consistency,
    dissonance_ratio,
    syncopation_ratio,
    duration_entropy,
    register_span,
    melodic_direction,
    key_consistency,
    pitch_range_check,
    empty_measure_check,
    unison_chain_check,
    rest_streak_check,
    mono_rhythm_check,
    extreme_density_check,
    max_polyphony_check,
    bar_boundary_melody,
    parallel_fifths_check,
    token_type_kl,
    melodic_contour_match,
    chord_melody_alignment,
    progression_validity,
    cadence_quality,
    harmonic_rhythm_score,
)
from .constraints import (
    TokenConstraint,
    check_parallel_fifths_octaves_tokens,
    check_voice_crossing_tokens,
    check_extreme_jump_tokens,
    Violation,
    separate_voices,
    evaluate_theory,
    SCORE_RULES,
)
from .scoring import (
    EvalReport,
    evaluate_generation,
    BarInspection,
    CFeedback,
    review_musicxml,
    compare_tokens_to_xml,
    c_review_to_feedback,
    apply_c_feedback_to_bans,
)
from .logging import (
    ABCGenerationLogger,
    GenerationSummary,
)
from .parser import (
    Note, Measure, Score,
    parse_musicxml, parse_musicxml_string,
    score_to_duration_seconds, score_to_note_count,
)
