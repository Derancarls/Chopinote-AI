# 生成-评价反馈闭环

## A/B1/B2/C 四阶段架构

```
用户输入 MusicXML
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│  A 阶段：生成前评估（异步，~微秒级）                      │
│                                                          │
│  输入: seed token 序列                                    │
│  过程:                                                   │
│    1. 提取 SeedProfile（小节数/密度/调性/速度/声部/分布） │
│    2. 设定 GenerationParams 初值                          │
│                                                          │
│  输出: SeedProfile + GenerationParams                    │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│  B 阶段：生成中实时反馈（每小节触发，token 级）             │
│                                                          │
│  ┌────────────────────────┐  ┌────────────────────────┐  │
│  │  B1 局部层              │  │  B2 全局层              │  │
│  │  最近 N 节自身流畅性     │  │  每 S 节一块 vs seed   │  │
│  │  不依赖 seed            │  │  检测漂移趋势           │  │
│  │                        │  │                        │  │
│  │  调整: temperature     │  │  调整: key_bias_strength│  │
│  │        rest_penalty    │  │        rest_penalty    │  │
│  │        complexity      │  │        temperature     │  │
│  └───────────┬────────────┘  └───────────┬────────────┘  │
│              │                           │               │
│              ▼                           ▼               │
│        combined = B1×local_w + B2×global_w               │
│        低分 → 增大拉回力度                                  │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│  C 阶段：生成后全量评价（异步跑一次）                      │
│                                                          │
│  输入: 生成的 MusicXML                                   │
│  过程:                                                   │
│    1. 复用 chopinote_evaluator 全量 ~50 项指标            │
│    2. 合法性 → 广义(统计+理论) → 狭义(一致+衔接+模型)    │
│    3. 综合评分 < 阈值 → 建议重试                          │
│                                                          │
│  输出: EvaluationReport → reward_log.jsonl               │
└──────────────────────────────────────────────────────────┘
```

## 文件结构

| 文件 | 角色 |
|------|------|
| `chopinote_evaluator/registry.py` | 指标注册表：定义每项指标的阶段归属(A/B1/B2/C) + token 级实现 |
| `chopinote_evaluator/feedback_controller.py` | A/B/C 控制器实现 |
| `chopinote_model/generate.py` | SeedProfile + GenerationParams 数据容器 |
| `chopinote_cli/main.py` | CLI 集成：`--feedback` 标志 + 参数透传 |

## 指标分阶段归属

### A 阶段（生成前 seed 评估）

| 指标 | 作用 |
|------|------|
| n_bars | seed 小节数 |
| bar_density | 平均 notes/bar |
| tonic_key | 主调 |
| tonic_midi | 主音 MIDI 编号 |
| key_pitch_classes | 调性自然音级集合 |
| programs | 乐器列表 [(prog, sub), ...] |
| pitch_class_dist | 12 维音级分布 |
| interval_dist | 25 维音程分布 |
| velocity_mean | 力度均值 |
| rest_ratio | 休止比例 |
| density_series | 逐节密度数组 |
| voice_count | 声部数 |
| time_sig | 拍号 |

### B1 阶段（生成中局部流畅）

指标 | 评分逻辑 | 触发调整
------|----------|----------
pitch_class_kl | vs 前一个窗口 KL | temperature ↓
interval_kl | vs 前一个窗口 KL | temperature ↓
density_z | 滑动窗口 Z-score | rest_penalty ↑, temperature ↓
dissonance_ratio | 0.02~0.15 最佳区间 | temperature ↓
velocity_consistency | CV 0.08~0.40 最佳区间 | temperature ↓
rest_ratio | 0.03~0.25 最佳区间 | rest_penalty ↑
register_span | 25%~75% 最佳区间 | complexity ↓
duration_entropy | 0.3~0.6 最佳区间 | temperature ↓
syncopation_ratio | 0.05~0.35 最佳区间 | temperature ↓
melodic_direction | 25%~65% 变化率 | temperature ↓
interval_shift | 级进/大跳比 | temperature ↓

**窗口**: 最近 N 节（默认 4 节），不足时跳过。
**阈值**: 0.55，低于阈值按亏缺比例应用 B1_ADJUSTMENT_RULES。

### B2 阶段（生成中全局漂移）

指标 | 评分逻辑 | 触发调整
------|----------|----------
pitch_class_kl | vs seed 音级分布 KL | key_bias_strength ↑
interval_kl | vs seed 音程分布 KL | key_bias_strength ↑
density_z | vs seed 平均密度 Z-score | rest_penalty ↑, temperature ↓
rest_ratio | vs seed 休止比例差 | rest_penalty ↑
register_span | vs seed 音域跨度比 | temperature ↓
velocity_consistency | vs seed 力度均值差 | temperature ↓
key_consistency | KEY token 变化检测 | key_bias_strength ↑
token_type_kl | vs seed token 类型分布 KL | temperature ↓, rest_penalty ↑
empty_measure | 连续空小节 ≥2 | —
pitch_range | MIDI 21-108 检查 | —

**窗口**: seed 小节数 S，不足 S 节时跳过。
**趋势检测**: 最近 B2_TREND_WINDOW=3 块连续下降 → 额外 B2_TREND_PENALTY=0.5 乘数。
**阈值**: 0.50。

### C 阶段（生成后全量评价）

| 维度 | 包含 |
|------|------|
| 合法性 (7项) | note_density, pitch_range, empty_measures, time_sig_align, voice_overlap, tuplet_pairing, tie_pairing |
| 统计 (19项) | pitch_class_kl, interval_kl, density_z, rest_ratio, register_span, velocity_consistency, dissonance_ratio, syncopation_ratio, duration_entropy, key_consistency, self_similarity, pitch_entropy, chromaticism_index, harmonic_rhythm, polyphony_mean, texture_variance, contour_arc, melodic_direction, interval_shift |
| 理论 (9项) | parallel_fifths, parallel_octaves, hidden_fifths, hidden_octaves, voice_distance, voice_crossing, leading_tone, tritone_resolution, cross_relation |
| 狭义一致 (5项) | pitch_class_kl, density_delta, interval_shift, velocity_delta, articulation_delta |
| 狭义衔接 (4项) | key_match, voice_count_delta, tempo_continuity, cadence_match |
| 模型自洽 (可选) | perplexity, boundary_test |

## CLI 用法

```bash
# 基础生成（无反馈）
chopin checkpoints/step_X.pt input.musicxml

# 启用 A→B→C 反馈闭环
chopin checkpoints/step_X.pt input.musicxml --feedback

# 调整 B1/B2 权重
chopin checkpoints/step_X.pt input.musicxml --feedback \
    --local-weight 0.3 --global-weight 0.7

# 调整重试阈值
chopin checkpoints/step_X.pt input.musicxml --feedback \
    --retry-threshold 0.6 --max-retries 5
```

## 参数裁剪范围

GenerationParams.apply_adjustments() 自动裁剪到安全范围：

| 参数 | 范围 |
|------|------|
| temperature | [0.3, 2.5] |
| rest_penalty | [0.0, 10.0] |
| key_bias_strength | [0.0, 5.0] |
| complexity | [1.0, 10.0] |
| top_k | [1, 100] |
| prog_switch_strength | [0.0, 5.0] |

## RL Reward 日志

C 阶段自动记录到 `$CHOPINOTE_REWARD_DIR/reward_log.jsonl`（默认 `/root/autodl-tmp/chopinote/rewards/`）：

```json
{
  "timestamp": "2026-05-17 12:00:00",
  "total_score": 0.72,
  "general_score": 0.68,
  "specific_score": 0.75,
  "legality_passed": true,
  "retry_count": 0,
  "params": {"temperature": 1.0, "rest_penalty": 0.0, ...}
}
```

## 现有评价指标注册表

```
A:  pitch_range, empty_measure, tuplet_pair, tie_pair
    (提取: n_bars, bar_density, tonic_key, tempo, programs, ...)

B1: pitch_class_kl, interval_kl, density_z, rest_ratio,
    register_span, velocity_consistency, dissonance_ratio,
    syncopation_ratio, duration_entropy, melodic_direction,
    interval_shift, density_delta, articulation_delta

B2: pitch_class_kl, interval_kl, density_z, rest_ratio,
    register_span, velocity_consistency, interval_shift,
    key_consistency, empty_measure, pitch_range, token_type_kl

C:  全量 ~50 项（注册表中已显式标注 Phase）
```

详见 `chopinote_evaluator/registry.py` 中 `REGISTRY` 的定义。
