# Chopinote-AI

> Give it a few bars — it finishes the piece.

Chopinote-AI is a **decoder-only Transformer** (1.21B parameters) that composes classical piano music in the style of your input. Feed it the first few measures of a piece (MusicXML), and it generates a stylistically coherent continuation — with structural awareness of musical form, functional harmony understanding, and real-time quality control.

Output is standard MusicXML, editable in MuseScore, Finale, Sibelius, or any notation software.

---

## Why Chopinote-AI?

### 1. Section-Aware Attention — It Understands Musical Form

Most music generation models treat every bar equally. Chopinote-AI explicitly models **musical structure** through learned section biases:

| Bias | Role |
|------|------|
| **α** | same-section boost — notes within the same section attend more strongly |
| **β** | bar-distance decay — nearby bars matter more than distant ones |
| **γ** | section-type affinity — fast sections pay more attention to other fast sections |
| **δ** | section-boundary reset — attention resets at section boundaries |

This means the model understands when it's in an **exposition** vs **development** vs **recapitulation** section, and generates accordingly. It won't introduce a new theme where a recapitulation belongs.

### 2. Chord-Aware Attention — It Understands Harmony

Chopinote-AI models **functional harmony** with dedicated chord biases:

| Bias | Role |
|------|------|
| **γ** (shared) | harmonic function affinity — tonic attends to tonic, dominant to dominant |
| **ε** | chord-to-chord — harmonic progression coherence across bars |
| **ζ** | inversion awareness — chord voicing (root position, first inversion, etc.) |

Combined with a **ChordPredictionHead** that learns to predict chord functions (Tonic, Dominant, Subdominant, etc.) and inversions, the model develops an internal model of harmonic progression — not just statistical co-occurrence of notes.

### 3. Dual Position Encoding — Both Local and Global Context

| Encoding | Scope | Purpose |
|----------|-------|---------|
| **RoPE** | token position within sequence | relative token distance, transposition-invariant |
| **Bar Embedding** | bar number within the piece | measure-level structure, phrase grouping |

RoPE captures fine-grained ornamentation (grace notes, trills) while bar embedding tracks the larger phrase and period structure.

### 4. Interval-Based Relative Pitch — Key Transposition Is Native

Notes are encoded as **intervals relative to the tonic** (major 3rd = 4, perfect 5th = 7), not absolute MIDI numbers. This means:

- The model **natively generalizes across keys** — it doesn't need to re-learn patterns in C major vs G major
- Melodic and harmonic intervals are first-class citizens
- Transposition is a **zero-cost operation**

### 5. Multi-Task Learning — Structural Supervision Without Extra Data

During training, the model simultaneously learns three tasks from the same token stream:

| Task | Head | What It Learns |
|------|------|----------------|
| **Next-token prediction** | LM head | note-by-note fluency and style |
| **Section prediction** | SectionPredictionHead | bar boundaries, key changes, section types (A/B/Sonata) |
| **Chord prediction** | ChordPredictionHead | harmonic function, chord inversions, cadence patterns |

The section and chord sidecar annotations (`.sec.json`, `.chord.json`) are automatically generated — no manual annotation required.

### 6. Feedback-Controlled Generation — Quality in Real Time

Generation runs through a **four-stage feedback loop**:

```
Pre-generation:  Profile your input → set initial sampling parameters
    │
Bar-by-bar:      B1 (local fluency) + B2 (global drift) → adjust on the fly
    │
Post-generation:  Full evaluation → keep or suggest regenerate
```

This catches degeneration before it compounds, and ensures every output meets quality thresholds.

### 7. Comprehensive Evaluation

20+ musical metrics organized into four layers:

| Layer | What | Examples |
|-------|------|----------|
| **Legality** | Basic validity | voice range, empty bars, time signature alignment |
| **Statistics** | Surface features | pitch class distribution, density, rhythmic variety |
| **Theory** | Harmony & voice leading | chord progression validity, cadence quality, harmonic rhythm |
| **Coherence** | Structural integrity | style consistency between input and generated sections |

### 8. Memory-Efficient Architecture

Training a 1.21B model on 32 GB VRAM requires deliberate design:

- **Bias recomputation inside gradient checkpointing** — section and chord biases are recomputed from raw data (~1 MiB) instead of stored as full attention masks (256 MiB per layer)
- **Bias detach** — combined bias is detached before SDPA, eliminating 8 GiB of attention matrix materialization during backward
- **Gradient checkpointing** with `use_reentrant=True` — only stores input activations
- **BF16 autocast** — no GradScaler needed, native half-precision training

---

## Architecture

```
Input Token IDs (B, T)
    │
    ▼
┌─────────────────────┐
│  Token Embedding     │  d_model=2048
│  + RoPE (position)   │
│  + Bar Embedding     │  measure-level position
└─────────┬───────────┘
          │
┌─────────▼──────────────────────────────────┐
│           24× TransformerBlock             │
│  ┌─────────────────────────────────────┐   │
│  │  Multi-Head Self-Attention           │   │
│  │  • RoPE rotary position encoding    │   │
│  │  • Section bias (α/β/γ/δ)           │   │
│  │  • Chord bias (γ/ε/ζ + δ dedup)     │   │
│  │  • Flash / mem-efficient attention  │   │
│  └──────────────┬──────────────────────┘   │
│  ┌──────────────▼──────────────────────┐   │
│  │  Position-wise FFN (d_ff=8192)       │   │
│  │  Residual + LayerNorm                │   │
│  └─────────────────────────────────────┘   │
└─────────┬──────────────────────────────────┘
          │
          ▼
┌─────────────────────┐  ┌──────────────────────┐  ┌─────────────────────┐
│  LM Head             │  │  SectionPredictionHead │  │  ChordPredictionHead │
│  (next token logits)  │  │  (bars/key/type)      │  │  (func/inv)          │
└─────────────────────┘  └──────────────────────┘  └─────────────────────┘
```

### Specs

| Parameter | Value |
|-----------|-------|
| Parameters | 1.21B |
| Layers | 24 |
| Attention heads | 32 |
| d_model | 2048 |
| d_ff | 8192 |
| Vocab size | 929 |
| Context length | 4096 tokens |
| Position encoding | RoPE |
| Precision | BF16 training, FP8/BF16 inference (auto-detected) |
| Peak VRAM (training, bs=8) | 22.3 GiB |
| Checkpoint size | ~4.5 GB |

### Tokenization (REMI)

REMI (REvamped MIDI) tokenization encodes music as a flat token sequence:

- **Note events**: pitch (as interval from tonic), duration, velocity (8 levels)
- **Time events**: beat, bar line, tempo
- **Structure tokens**: section markers, chord function labels

Each token belongs to one of 27 semantically grouped types for per-type accuracy tracking.

---

## Generation Pipeline

### Three-Stage Generation

| Stage | What | Length |
|-------|------|--------|
| **1. Structure Plan** | Generate section boundaries, keys, and type labels | ~12-20 tokens |
| **2. Harmony Skeleton** | Generate chord function + inversion sequence | ~60-120 tokens |
| **3. Note Filling** | Generate full notes with section + chord attention bias | remaining tokens |

This compositionally decomposes the generation task, ensuring structural and harmonic coherence before committing to individual notes.

### Feedback Loop

```
┌─────────────────────────────────────────────────────────────┐
│  A — Pre-Generation                                        │
│  Analyze seed → style profile → set temperature, top-k, etc. │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  B — Mid-Generation (per-bar)                              │
│  ┌──────────────────┐  ┌──────────────────┐                 │
│  │ B1 — Local       │  │ B2 — Global      │                 │
│  │ recent measures  │  │ generated vs seed│                 │
│  │ feel off?        │  │ drifting apart?  │                 │
│  │ → adjust temp    │  │ → pull params    │                 │
│  └──────────────────┘  └──────────────────┘                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  C — Post-Generation                                       │
│  Full evaluation → score → keep if good, suggest retry if not │
└─────────────────────────────────────────────────────────────┘
```

---

## Getting Started

### Quick Start

```bash
# Continue a piece
chopin checkpoints/step_N.pt input.musicxml -o output.musicxml

# With feedback loop (recommended)
chopin checkpoints/step_N.pt input.musicxml --feedback

# Generate multiple variants, pick the best
chopin checkpoints/step_N.pt input.musicxml -n 5 --feedback

# Use a style preset
chopin checkpoints/step_N.pt input.musicxml --preset romantic
chopin checkpoints/step_N.pt input.musicxml --preset baroque
chopin checkpoints/step_N.pt input.musicxml --preset jazz

# Custom config file
chopin checkpoints/step_N.pt input.musicxml --config my_cfg.yaml

# CLI overrides config defaults
chopin checkpoints/step_N.pt input.musicxml --temp 1.2 --max-bars 64
```

### GPU Auto-Configuration

On first run, Chopinote-AI automatically detects your hardware and selects the optimal inference settings:

| Hardware | Auto-Selected |
|----------|---------------|
| **RTX 5090 / Blackwell+** (≥24 GB) | FP8 precision, torch.compile, TF32 on |
| **RTX 4080 / Ampere+** (≥24 GB) | BF16 precision, torch.compile, TF32 on |
| **RTX 3080 / Ampere** (≥8 GB) | BF16 precision, torch.compile off |
| **CPU only** | FP32 precision, 16 threads |

No manual tuning needed — the model handles precision, thread count, memory fraction, and compilation automatically.

### Configuration File

Parameters follow a priority chain: **CLI args > --preset > config file > defaults**.

```bash
# Auto-detected config locations (first found wins):
#   1. ./chopinote_config.yaml
#   2. ~/.chopinote/config.yaml
#   3. Built-in chopinote_cli/generation_config.yaml

# All 20 generation parameters are configurable — see the built-in YAML for details.
```

### Available Presets

| Preset | Style |
|--------|-------|
| `romantic` | Chopin, Liszt — rubato, chromatic harmony, expansive phrasing |
| `baroque` | Bach, Handel — contrapuntal, dance rhythms, terraced dynamics |
| `classical` | Mozart, Haydn — balanced phrases, clear cadences, Alberti bass |
| `jazz` | Gershwin, Brubeck — extended chords, swung rhythms, ii-V-I progressions |
| `church` | Pipe organ — solemn, sustained, liturgical |
| `minimal` | Reich, Glass — sparse texture, slow tempo, hypnotic repetition |

---

## Training

### Curriculum (Two-Phase)

| Phase | Data | Steps | LR | Warmup | Focus |
|-------|------|-------|----|--------|-------|
| 1 — Pretrain | MIDI (MAESTRO, Lakh, GiantMIDI, etc.) | 120K | 1.5e-4 | 4K | Note fluency, basic structure |
| 2 — Fine-tune | MusicXML (ASAP, ATEPP, openscore, internal) | 50K | 1.0e-4 | 2K | Articulation, expression, performance markings |

### Hardware

- **GPU**: RTX 5090 — 32 GB VRAM
- **Batch**: bs=8, grad_accum=4 (effective bs=32)
- **Data**: ~1.62M files, ~13.7B tokens, ~400 GB on disk
- **Speed**: ~5-6 s/step on RTX 5090 (bf16)

### Training Data

| Type | Source | Size |
|------|--------|------|
| MIDI | MAESTRO, Lakh, GiantMIDI, POP909, MusicNet, EMOPIA | ~1.37M files |
| MusicXML | ASAP, ATEPP, Openscore, internal corpus | ~4.1K files |
| PDMX | Pop music scores | ~250K files |

---

## Project Structure

```
chopinote_model/        Core model and training
├── model.py            MusicTransformer (24 layers, RoPE, sec/chord bias)
├── config.py           ModelConfig, TrainingConfig, PhaseConfig, TokenLossMask
├── train.py            Trainer, multi-task loss, curriculum learning
├── dataset.py          TokenDataset with LRU cache, sidecar auto-loading
├── generate.py         Three-stage generation with KV cache
├── fp8_linear.py       FP8 mixed-precision linear layer (Blackwell _scaled_mm)
├── auto_config.py      Hardware detection + optimal inference/training config

chopinote_dataset/      Data processing pipeline
├── tokenizer.py        REMITokenizer (vocab 929, grid_size=16)
├── converter.py        MusicXML/PDMX/MIDI → REMI
├── fast_converter.py   mido-based ~80x faster MIDI conversion

chopinote_evaluator/    Music quality evaluation
├── feedback_controller.py  A/B1/B2/C feedback generation loop
├── registry.py          20+ metric registry
├── score.py            MusicXML-level scoring
└── general/            General metrics (statistics, theory, harmony)
└── specific/           Specific metrics (coherence, consistency)

chopinote_cli/          CLI entry point
├── main.py             Inference CLI with feedback + presets + config
├── config.py           Config loading, validation, auto-discovery
├── presets.py          Style presets (romantic, baroque, classical, etc.)
└── generation_config.yaml  Built-in defaults (20 parameters)

scripts/                Utility scripts
├── train/              Training: curriculum, DPO, launch control
├── preprocess/         Data preprocessing pipeline
├── generate/           Batch generation, roundtrip testing
└── analysis/           Token analysis, verification
```

---

## Related Work

Chopinote-AI builds on ideas from:

- **Music Transformer** (Huang et al.) — relative attention for music
- **REMI** (Huang & Yang) — MIDI tokenization for event-based music generation
- **Structured Music Transformer** (Jiang et al.) — section-aware attention
- **Chord Music Transformer** (Chen et al.) — chord-aware self-attention
- **RLHF for text** — adapted as DPO preference tuning for musical quality

---

## License

Internal research project. See license file for details.

---

---

# Chopinote-AI 中文说明

> 输入一段开头，AI 为你续写完整的古典钢琴曲。

Chopinote-AI 是一个 **1.21B 参数的 decoder-only Transformer**，专门用于古典钢琴曲的智能续写。给定前几小节（MusicXML），它以自回归方式逐 token 生成风格匹配的后续乐章，输出标准 MusicXML，可直接在打谱软件中打开和编辑。

---

## 核心优势

### 1. 段落感知注意力 — 模型理解曲式结构

大多数音乐生成模型平等对待每一小节。Chopinote-AI 通过可学习的段落偏置显式建模音乐结构：

| 偏置 | 作用 |
|------|------|
| **α** | 同段落增强——同段落内的音符相互注意力更强 |
| **β** | 小节距离衰减——相邻小节比远处小节更重要 |
| **γ** | 段落类型亲和——快板段落更关注其他快板段落 |
| **δ** | 段落边界重置——注意力在段落边界处重置 |

模型知道何时处于 **呈示部**、**展开部** 还是 **再现部**，不会在需要再现时引入新材料。

### 2. 和声感知注意力 — 模型理解功能和声

通过专用和声偏置建模调性和声：

| 偏置 | 作用 |
|------|------|
| **γ**（共享） | 和声功能亲和——主功能关注主功能，属功能关注属功能 |
| **ε** | 和弦序列——跨小节的和声进行连贯性 |
| **ζ** | 转位感知——和弦排列（原位、第一转位等） |

配合 **ChordPredictionHead** 预测和弦功能（主、属、下属等）和转位，模型建立了和声进行的内在模型，而不仅仅是音符的统计共现。

### 3. 双重位置编码 — 兼顾局部与全局

| 编码 | 范围 | 用途 |
|------|------|------|
| **RoPE** | 序列内 token 位置 | 相对距离，移调不变 |
| **小节嵌入** | 整曲的小节号 | 乐句结构，周期性模式 |

RoPE 捕捉细碎的装饰音、颤音等，小节嵌入追踪更大的乐句和段落结构。

### 4. 音程制相对音高 — 移调是天生的

音符编码为 **相对主音的音程**（大三度=4，纯五度=7），而非绝对 MIDI 编号：

- 模型 **天生跨调性泛化**——不需在 C 大调和 G 大调上分别学习
- 旋律和声音程是一等公民
- 移调是零成本操作

### 5. 多任务学习 — 无需额外标注的结构监督

训练时从同一 token 流同时学习三个任务：

| 任务 | 预测头 | 学习内容 |
|------|--------|----------|
| **下一 token 预测** | LM head | 逐音符流畅性与风格 |
| **段落预测** | SectionPredictionHead | 小节边界、调性变化、段落类型 |
| **和弦预测** | ChordPredictionHead | 和声功能、和弦转位、终止式 |

段落和和弦标注（`.sec.json`、`.chord.json`）由脚本自动生成，无需人工标注。

### 6. 带实时反馈的生成控制

四阶段反馈闭环：

```
生成前：  分析输入开头 → 提取风格画像 → 设定采样参数初值
    │
逐小节：  B1（局部流畅度）+ B2（全局漂移）→ 实时调整参数
    │
生成后：  全面评分 → 达标保留，不达标建议重试
```

在退化累积之前及时纠正，确保每次输出达到质量阈值。

### 7. 全面评价体系

20+ 音乐指标分四层：

| 层级 | 内容 | 示例 |
|------|------|------|
| **合法性** | 基本有效性 | 音域检查、空小节、拍号对齐 |
| **统计** | 表面特征 | 音级分布、密度、节奏多样性 |
| **理论** | 和声与声部进行 | 和弦进行有效性、终止质量、和声节奏 |
| **连贯性** | 结构完整性 | 输入与生成部分的风格一致性 |

### 8. 显存高效架构

在 32 GB 显存上训练 1.21B 模型需要精心设计：

- **Bias 重计算** — 段落/和声偏置在 checkpoint 内从原始数据（~1 MiB）重算，而非存储为完整注意力掩码（每层 256 MiB）
- **Bias detach** — 合并偏置在 SDPA 前分离，消除反向传播中 8 GiB 注意力矩阵的显存占用
- **Gradient checkpointing** — 仅存储输入激活值
- **BF16 混合精度** — 无需 GradScaler，原生半精度训练

---

## 模型架构

```
输入 Token ID (B, T)
    │
    ▼
┌─────────────────────┐
│  Token 嵌入          │  d_model=2048
│  + RoPE（位置编码）    │
│  + 小节嵌入           │  小节级别位置编码
└─────────┬───────────┘
          │
┌─────────▼──────────────────────────────────┐
│           24× TransformerBlock             │
│  ┌─────────────────────────────────────┐   │
│  │  多头自注意力                        │   │
│  │  • RoPE 旋转位置编码                 │   │
│  │  • 段落偏置 (α/β/γ/δ)               │   │
│  │  • 和声偏置 (γ/ε/ζ + δ 去重)        │   │
│  │  • Flash / mem-efficient attention  │   │
│  └──────────────┬──────────────────────┘   │
│  ┌──────────────▼──────────────────────┐   │
│  │  前馈网络 (d_ff=8192)                │   │
│  │  残差连接 + 层归一化                  │   │
│  └─────────────────────────────────────┘   │
└─────────┬──────────────────────────────────┘
          │
          ▼
┌─────────────────────┐  ┌──────────────────────┐  ┌─────────────────────┐
│  LM 预测头           │  │  段落预测头           │  │  和弦预测头          │
│  (下一 token logits)  │  │  (小节数/调性/类型)   │  │  (功能/转位)          │
└─────────────────────┘  └──────────────────────┘  └─────────────────────┘
```

### 规格

| 项目 | 值 |
|------|-----|
| 参数量 | 1.21B |
| 层数 | 24 |
| 注意力头 | 32 |
| d_model | 2048 |
| d_ff | 8192 |
| 词表大小 | 929 |
| 上下文长度 | 4096 tokens |
| 位置编码 | RoPE |
| 精度 | BF16 训练，FP8/BF16 推理（自动检测） |
| 峰值显存（训练，bs=8） | 22.3 GiB |
| 模型大小 | ~4.5 GB |

### Token 化（REMI）

- **音符事件**：音高（相对主音音程）、时值、力度（8 级）
- **时间事件**：拍点、小节线、速度标记
- **结构标记**：段落边界、和弦功能标签

每种 token 属于 27 个语义分组之一，支持按类型追踪准确率。

---

## 生成流程

### 三阶段生成

| 阶段 | 内容 | 长度 |
|------|------|------|
| **1. 结构规划** | 生成段落边界、调性、类型标签 | ~12-20 tokens |
| **2. 和声骨架** | 生成和弦功能 + 转位序列 | ~60-120 tokens |
| **3. 音符填充** | 带段落+和声偏置生成完整音符 | 剩余 tokens |

这种分层分解确保在写具体音符之前，结构和和声骨架已经确定。

---

## 快速上手

```bash
# 续写
chopin checkpoints/step_N.pt input.musicxml -o output.musicxml

# 启用反馈闭环（推荐）
chopin checkpoints/step_N.pt input.musicxml --feedback

# 生成多份变体，选最好的
chopin checkpoints/step_N.pt input.musicxml -n 5 --feedback

# 指定风格预设
chopin checkpoints/step_N.pt input.musicxml --preset romantic
chopin checkpoints/step_N.pt input.musicxml --preset baroque
chopin checkpoints/step_N.pt input.musicxml --preset jazz

# 自定义配置文件
chopin checkpoints/step_N.pt input.musicxml --config my_cfg.yaml

# CLI 覆盖配置默认值
chopin checkpoints/step_N.pt input.musicxml --temp 1.2 --max-bars 64
```

### GPU 自动适配

首次运行自动检测硬件并选择最优推理配置：

| 硬件 | 自动选择 |
|------|----------|
| **RTX 5090 / Blackwell+** (≥24 GB) | FP8 精度、torch.compile、TF32 开 |
| **RTX 4080 / Ampere+** (≥24 GB) | BF16 精度、torch.compile、TF32 开 |
| **RTX 3080 / Ampere** (≥8 GB) | BF16 精度、torch.compile 关 |
| **纯 CPU** | FP32 精度、16 线程 |

### 配置文件

优先级：**CLI 参数 > --preset > 配置文件 > 内置默认值**。

```bash
# 自动检测顺序（先找到的生效）：
#   1. ./chopinote_config.yaml
#   2. ~/.chopinote/config.yaml
#   3. 内置 chopinote_cli/generation_config.yaml
```

### 风格预设

| 预设 | 风格 |
|------|------|
| `romantic` | 肖邦、李斯特 — 弹性速度、半音化和声、宽广乐句 |
| `baroque` | 巴赫、亨德尔 — 对位、舞曲节奏、阶梯式力度 |
| `classical` | 莫扎特、海顿 — 平衡乐句、清晰终止、阿尔贝蒂低音 |
| `jazz` | 格什温、Brubeck — 扩展和弦、摇摆节奏、ii-V-I |
| `church` | 管风琴 — 庄严、持续、礼仪性 |
| `minimal` | Reich、Glass — 稀疏织体、慢速、催眠重复 |

---

## 训练

### 两阶段课程学习

| 阶段 | 数据 | 步数 | 学习率 | Warmup | 重点 |
|------|------|------|--------|--------|------|
| 1 — 预训练 | MIDI | 120K | 1.5e-4 | 4K | 音符流畅性、基础结构 |
| 2 — 微调 | MusicXML | 50K | 1.0e-4 | 2K | 演奏法、表情、力度标记 |

### 硬件

- **GPU**: RTX 5090 — 32 GB 显存
- **Batch**: bs=8, grad_accum=4（有效 bs=32）
- **数据**: ~1.62M 文件，~13.7B tokens，~400 GB
- **速度**: RTX 5090 约 5-6 秒/步 (bf16)

### 训练数据

| 类型 | 来源 | 规模 |
|------|------|------|
| MIDI | MAESTRO, Lakh, GiantMIDI, POP909, MusicNet, EMOPIA | ~1.37M 文件 |
| MusicXML | ASAP, ATEPP, Openscore, 内建语料库 | ~4.1K 文件 |
| PDMX | 流行乐谱 | ~250K 文件 |

---

## 致谢

Chopinote-AI 的设计受益于以下工作：

- **Music Transformer** (Huang et al.) — 音乐相对注意力
- **REMI** (Huang & Yang) — 基于事件的中序列表征
- **Structured Music Transformer** (Jiang et al.) — 段落感知注意力
- **Chord Music Transformer** (Chen et al.) — 和弦感知自注意力

---

*Chopinote-AI — 让古典音乐创作从空白五线谱开始，而不是从已有作品开始。*
