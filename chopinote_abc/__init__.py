"""ABC Engine v2 — A(感知层) B(决策层) C(进化层) 三位一体认知架构。

A1DB: 框架库（结构 + 和声 + 运行时覆盖）
A2DB: 动机摘要库（提纯 token + MotifDNA）
A3DB: 统计库（每 bar 实时统计 + 段快照 + 基线 + 创新日志）

Phase 2: 和声回退 + C→A1 闭环 + 创新预算 + 发展配方 + 新颖性加分
"""

from .database import (
    A1DB, A2DB, A3DB,
    SectionPlan, ChordAtBar, SeedContext, StructuralFix,
    MotifDNA, MotifRecord,
    BarStats, SectionStats,
    compute_novelty_bonus, compute_diversity_bonus, write_reward_log,
)
from .planner import (
    plan_structure, plan_harmony, reharmonize_from_bar,
    tonal_progression_template,
)
from .motif import (
    identify_landmarks, purify_tokens, extract_dna,
    invert_contour, fragment_tokens, diminish_tokens,
    contour_distance, contour_similarity,
)
from .decision import (
    BHardBans, apply_zone_temperature,
    BFeedback, InnovationEntry,
    _compute_deviation_surprise,
)
