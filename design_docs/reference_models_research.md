# 音乐生成模型参考研究

> 调研时间：2026-05-09
> 目的：梳理当前主流音乐生成模型的架构设计，为 Chopinote-AI 后续改进提供参考

---

## 一、Tokenization 体系对比

### 1.1 REMI（本项目当前使用）
- 将音符分解为 Position + Duration + Velocity + 表现力标记等属性序列
- 显式 Bar/Beat/Position 时间网格（grid_size=16 → 16 格/四分音符）
- vocab=837，但序列较长（一首完整曲子 ~数千 token）

### 1.2 Performance Encoding（Magenta Music Transformer / Performance RNN）
- NoteOn + NoteOff + TimeShift 表示起止
- 时间更灵活（TimeShift 可到 1ms），但无显式网格
- 节拍感较弱，但表现力更细腻（可编码 rubato、微时值偏移）

### 1.3 Compound Word (CP) Tokenization
- **代表工作**：Compound Word Transformer (Hsiao et al.)
- 将所有属性（pitch/duration/velocity/instrument）压缩为一个复合 token
- 序列长度是 REMI 的约 1/6，训练速度显著提升
- 代价：并行预测子字段丢失了属性间依赖（如 pitch-duration 关联）
- 改进方案：

| 方法 | 思路 | 代价 |
|------|------|------|
| Nested Music Transformer (NMT) | sub-decoder + 交叉注意力，自回归展开子字段 | 增加推理内存和延迟 |
| **Delay Pattern (DP)** | 子字段错开步长，自回归预测，无额外参数 | ~1.7% 速度损失，质量接近 REMI |

- 实测（Time-Shifted Token Scheduling, 2025）：MMT-DP 生成速度 **62.47 notes/sec** vs REMI+ 的 20.42 notes/sec（3× 加速），质量主观评分相当

### 1.4 音频 Tokenization（与本项目无关，但了解边界）
- **EnCodec (Meta)**：神经音频编解码器，将波形压缩为离散 token
- **RVQ (Residual Vector Quantization)**：分层码本（codebook 1=粗结构，后续=细节）
- **MusicGen Delay Pattern**：用交错延迟让单个 Transformer 自回归预测所有 RVQ 层级

---

## 二、位置编码

### 2.1 绝对位置编码（本项目当前使用）
- 标准 GPT 方式：position_ids → Embedding + 加到 QKV
- 位置 i 和 j 的交互与它们的绝对位置相关，而非相对距离
- 局限：音乐中的模式是移位不变的（大三和弦在任何调上都是大三和弦），绝对编码要求模型在每个位置独立学习

### 2.2 相对位置注意力（Music Transformer, Google, 2018）
- **论文**：*Music Transformer: Generating Music with Long-Term Structure* (Huang et al.)
- 在注意力分数计算中直接编码 token 之间的相对距离 `i - j`
- 核心优势：模型学到"和前一个音差 2 个半音"的抽象规则，而非"C 大调时 position 42 是 E"
- 实现方式：相对位置偏置 `relative_bias[i][j] = embed[i - j]` 加到 attention logits
- Memory-efficient 变体：`dot_product_relative_v2`，用 skewing 技巧避免 O(n²) 内存
- 论文证明：在音乐困惑度上显著优于绝对位置编码

### 2.3 多维相对注意力 MRA（Moonbeam, QMUL, 2025）
- 将相对距离分解为三个独立维度：
  - **时间距离** → 节奏模式
  - **音高距离** → 音程关系
  - **小节距离** → 和声进行
- 每个维度有自己的可学习偏置，最后相加
- 更细粒度地建模音乐的多维结构

### 2.4 MuseNet 风格结构嵌入（OpenAI, 2019）
- 四种结构嵌入叠加：Part（段落位置）、Type（token 类型）、Time（时间差）、Pitch-Class（音级）
- 2024 复现研究（*Practical and Reproducible Symbolic Music Generation*）：用 **sinusoidal 初始化** 替代 learned embedding 能显著提升结构生成质量

---

## 三、注意力机制

### 3.1 全注意力（本项目当前使用）
- FlashAttention (`F.scaled_dot_product_attention`) 实现
- O(n²) 复杂度，seq_len=4096 时约 16M 注意力分数
- 无法扩展到 seq_len=8192（d_model=1024 下 OOM）

### 3.2 FC-Attention（Museformer, Microsoft, NeurIPS 2022）
- **论文**：*Museformer: Transformer with Fine- and Coarse-Grained Attention for Music Generation*
- **精细注意力**：每个 token 直接关注与音乐结构最相关的小节的全部 token（前 1、2、4、8、12、16、24、32 个小节，基于训练数据的相似性统计选定）
- **粗粒度注意力**：对其他小节，只关注每个小节的 **summary token**（每个小节附加一个可学习的汇总 token），而非全部 token
- 结果：可建模 3× 以上标准全注意力的序列长度
- 训练时自动学习哪些小节在结构上相关——这对有重复模式的音乐特别有效

### 3.3 线性注意力（RWKV-7, 2025）
- RWKV 架构：RNN-like 线性注意力，理论上支持无限上下文
- MIREX 2025 结果：**20M 参数 RWKV-7** 在钢琴续写任务上与 **780M Anticipatory Transformer** 质量相当（连贯性 3.57 vs 3.70，音乐性 3.50 vs 3.45）
- 意义：对单乐器符号音乐生成，模型大小不是决定性因素，数据质量和 tokenization 可能更重要

### 3.4 Anticipatory Music Transformer（Stanford, TMLR 2024）
- **论文**：*Anticipatory Music Transformer* (Thickstun et al.)
- 核心创新：在序列中插入"控制点"(control tokens) 作为 stopping times，让生成条件化于未来事件
- 2024 年为旧金山交响乐团生成了贝多芬《致爱丽丝》的小提琴伴奏
- MIREX 2025 最高分：连贯性 3.70、结构 3.69、创意 3.30
- 780M 参数

---

## 四、生成控制机制

### 4.1 本项目当前方法
- 推理时 logit 操作链：温度→复杂度偏置→Rest惩罚→词表截断→Program锁定→音高范围→调性偏置→复音上限→top-k→采样
- 这套控制链在学术论文中几乎不存在——学术模型通常只做温度+top-k
- 多轨生成（任意 Program 组合）+ 乐器级复音上限
- 预设系统 7 种风格

### 4.2 学术/工业模型的生成控制

| 模型 | 控制方式 | 粒度 |
|------|---------|------|
| Music Transformer | 无（自由续写） | - |
| Museformer | 无（自由续写） | - |
| MuseNet | token-level conditioning | 粗 |
| MusicGen (Meta) | Text prompt + melody condition | 语义级 |
| Suno | Text prompt + style tags | 语义级 |
| **Anticipatory Transformer** | **Control tokens 在序列中定点引导** | **事件级** |

Anticipatory 的控制方式最值得关注：它不是推理时 logit 操作，而是在训练时学习"在特定时间点达到特定目标"，因此生成过渡更自然。

### 4.3 潜在结合方向
- 你的调性偏置（推理时硬约束）+ Anticipatory（训练时平滑过渡）→ 更自然的调性转换
- 复杂度控制的 Anticipatory 版本：插入密度控制点让模型提前调整织体

---

## 五、模型架构整体对比

| 模型 | 时间 | 参数量 | Tokenization | 位置编码 | 注意力机制 | 生成控制 | 多轨 | 表现力 | 代码开源 |
|------|------|--------|------------|---------|-----------|---------|------|--------|---------|
| **Chopinote-AI** | 2026 | **156M** | REMI (837) | 绝对 | FlashAttention | **极其丰富** | ✅ | ✅ 全套 | ✅ |
| Music Transformer | 2018 | ~50M | Performance | **相对位置** | 相对注意力 | 基础 | ❌ | ❌ | ✅ |
| MuseNet | 2019 | 72M-1.5B | MIDI-like | 绝对+结构嵌入 | GPT | 基础 | ✅ | ❌ | ❌ |
| Museformer | 2022 | ~200M | REMI-like | 绝对 | **FC-Attention** | 基础 | ✅ | ❌ | ✅ |
| MusicGen | 2023 | 300M-3.3B | EnCodec (音频) | 绝对 | 标准 | Text+Melody | ❌ | 音频级 | ✅ |
| Anticipatory Transformer | 2024 | 780M | MIDI-like | 到达时间编码 | 标准 | **控制点引导** | ✅ | ❌ | ✅ |
| MuPT | 2025 | ~300M | REMI+ | 绝对 | 标准 | 基础 | ✅ | ❌ | ✅ |
| Moonbeam | 2025 | 未知 | REMI-like | **MRA(多维相对)** | MRA | 基础 | ✅ | ❌ | ❌ |
| RWKV-7 | 2025 | **20M** | MIDI-like | 线性注意力 | RWKV | 基础 | ❌ | ❌ | ✅ |
| Cadenza | 2024 | ~100M | PerTok (CP变体) | RoPE | VAE+Bi-Encoder | 作曲/演奏分离 | ✅ | **力度+微时值** | ✅ |
| xLSTM | 2025 | 可变 | MIDI-like | 相对 | LSTM+注意力混合 | 基础 | ✅ | ❌ | ✅ |

---

## 六、值得关注的训练策略

### 6.1 数据增强（Magenta / Music Transformer）
- 时间拉伸 ±5%
- 音高移调 ±3 semitones
- 随机裁剪
- 与本项目的 ±5 semitones 移调增强思路一致

### 6.2 课程学习 / 分阶段训练（本项目）
- Phase 1 MIDI 预训练（屏蔽表现力 token loss）
- Phase 2 MusicXML 微调（全量 token）
- 类似策略在文献中较少被系统研究，是本项目的独特贡献

### 6.3 作曲-演奏分离（Cadenza, 2024）
- Composer（VAE with RoPE）：生成乐谱骨架（音高、节奏、和声）
- Performer（Bi-Encoder）：给骨架添加力度、微时值偏移等演奏细节
- 和本项目的两阶段训练有精神上的相似，但 Cadenza 是两个独立的模型

### 6.4 Scale Law 验证（Lehmkuhl et al., NeurIPS 2025 Workshop）
- 系统对比了模型规模（最高 950M）和数据多样性
- 结论：模型规模和训练数据多样性都显著影响生成质量
- 950M 模型的输出在 Turing-style 盲听测试中常被误认为人类作品

---

## 七、对本项目最有价值的改进方向（按投入产出比排序）

### 优先级 1：相对位置注意力
- **参考**：Music Transformer (Google) / Moonbeam MRA (QMUL)
- **改动量**：~50 行代码，修改 attention 分数计算
- **预期收益**：模式学习更高效，对音程、和弦等相对关系建模更好
- **为什么优先**：论文验证最充分，改动最小，收益明确

### 优先级 2：Anticipatory 控制点
- **参考**：Anticipatory Music Transformer (Stanford)
- **改动量**：改数据格式 + 训练时插入 control token
- **预期收益**：替代/补充推理时调性偏置，生成更平滑的调性/风格过渡
- **创新空间**：推理时动态插入控制点（论文只做了训练时固定插入）

### 优先级 3：Compound Word + Delay Pattern
- **参考**：Compound Word Transformer / MusicGen Delay Pattern
- **改动量**：需改 tokenizer 和 output head
- **预期收益**：序列压缩 6×，可支持 seq_len=8192
- **代价**：多轨+表现力下 vocab 设计复杂

### 优先级 4：FC-Attention（如果做超长序列）
- **参考**：Museformer (Microsoft)
- **改动量**：较大，需重写 attention mask 逻辑
- **预期收益**：3× 当前序列长度而不 OOM
- **场景**：需要 seq_len>4096 且不想改 tokenization 时

### 优先级 5：Cadenza 两层分离架构
- **参考**：Cadenza (2024)
- **改动量**：需要两个独立模型
- **预期收益**：更细粒度的表现力控制
- **与本项目关联**：当前两阶段训练已隐含了骨架→细节的学习，可以进一步显式化

---

## 八、值得关注的数据集

| 数据集 | 格式 | 规模 | 特点 |
|--------|------|------|------|
| MAESTRO v3.0.0 | MIDI + 音频对齐 | 1,276 曲 | 钢琴独奏，高精度 |
| Lakh MIDI | MIDI | 176,581 曲 | 多乐器，多风格 |
| Aria-MIDI Pruned | MIDI | 820,944 曲 | 大规模爬取 |
| Bread MIDI | MIDI | ~1,000,000 曲 | HuggingFace |
| GiantMIDI-Piano | MIDI | 10,854 曲 | 钢琴独奏，需联系作者 |
| PDMX | PDMX JSON | ~254,000 曲 | 多轨 MIDI 元数据 |
| ASAP | MusicXML+MIDI | 242 曲 | 对齐数据集 |
| MusicCaps (评估) | 音频+文本 | 5,521 曲 | Google 评估集 |

---

## 九、关键引用

- Music Transformer: Huang et al., 2018, *Music Transformer: Generating Music with Long-Term Structure*
- Museformer: Yu et al., 2022, *Museformer: Transformer with Fine- and Coarse-Grained Attention for Music Generation*, NeurIPS 2022
- Anticipatory Music Transformer: Thickstun et al., 2024, TMLR
- MusicGen: Copet et al., 2023, *Simple and Controllable Music Generation*, Meta AI
- MusicLM: Agostinelli et al., 2023, *MusicLM: Generating Music from Text*, Google
- Compound Word Transformer: Hsiao et al., 2021
- Nested Music Transformer: 2024
- MuPT: Qu et al., ICLR 2025
- Moonbeam: Marxer et al., 2025, QMUL
- Cadenza: 2024, PerTok tokenizer + Composer/Performer 分离
- RWKV-7: MIREX 2025, 20M 参数匹敌 780M
- Time-Shifted Token Scheduling: 2025, Delay Pattern 系统性评估
- Scale Law: Lehmkuhl et al., 2025, NeurIPS Workshop
