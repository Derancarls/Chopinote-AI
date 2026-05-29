"""ABC Engine v2 — A(感知层) B(决策层) C(进化层) 三位一体认知架构。

A1DB: 框架库（结构 + 和声 + 运行时覆盖）
A2DB: 动机摘要库（提纯 token + MotifDNA）
A3DB: 统计库（每 bar 实时统计 + 段快照 + 基线）

Phase 1 全链路规则驱动，零模型依赖（除主 Transformer）。
"""

from .database import (
    A1DB, A2DB, A3DB,
    SectionPlan, ChordAtBar, SeedContext,
    MotifDNA, MotifRecord,
    BarStats, SectionStats,
)
from .planner import (
    plan_structure, plan_harmony, reharmonize_from_bar,
    tonal_progression_template,
)
from .motif import (
    identify_landmarks, purify_tokens, extract_dna,
)
from .decision import (
    BHardBans, apply_zone_temperature,
)
