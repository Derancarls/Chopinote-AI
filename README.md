# Chopinote-AI

> 输入一段开头，AI 为你续写完整的古典钢琴曲。

给定一首曲子的前几小节（MusicXML 格式），Chopinote-AI 以自回归方式逐 token 生成风格匹配的后续乐章，输出标准 MusicXML，可在 MuseScore、Finale、Sibelius 等打谱软件中直接打开和编辑。

---

## 核心架构

### Decoder-only Transformer (1.01B)

纯 GPT 风格的自回归架构，24 层 Transformer decoder，d_model=2048，32 注意力头，在 RTX 5090 (32GB) 上以 BF16 训练。与大多数音乐生成模型不同，Chopinote-AI 的设计围绕一个核心理念：**让音乐结构成为模型的原生属性，而非事后附加的约束**。

### 音程制相对音高编码（核心创新）

传统 REMI 方案将 Note_ON 编码为绝对 MIDI 音高（如 `Note_ON 60` = 中央 C），但 Chopinote-AI 存储的是**到主音的半音程**（如 `Note_ON 0` = 主音，`Note_ON -3` = 下方小三度）。这使得：

- **调性归纳是模型的原生能力**——模型学到的是音程关系而非绝对音高，同一段旋律在不同调性上是相同的 token 序列
- **转调自然泛化**——模型无需为每个调性重新学习音高分布
- **数据效率更高**——调性信息不必从零开始学习，训练信号集中在真正的音乐模式上

### Anticipate Token（提前调性预告）

在调性变化发生的**前一小节**插入预告 token（如 `<Anticipate Key G>`），提前通知模型即将到来的转调。这给注意力机制一个和声准备的窗口期，使转调过渡更自然——这是其他开源 REMI 方案中少见的显式设计。

### 小节嵌入 + RoPE 双重位置感知

模型同时使用两种位置信息：

- **RoPE（旋转位置编码）**：编码 token 在序列中的绝对和相对位置，支持 KV cache 推理加速。针对 Blackwell GPU 做了 cuDNN 注意力后端优先调度，并重写为 `torch.compile` 友好的 reshape+stack 模式。
- **小节嵌入（Measure Embedding）**：在每个 token 的嵌入上叠加其所属小节的嵌入向量。模型不仅知道"第几个 token"，还知道"第几个小节"，这对音乐的长程结构建模至关重要。

两者结合使模型同时拥有序列级和小节级的双重位置感知。

### 两阶段课程学习

大多数音乐生成模型使用单一训练策略，但 Chopinote-AI 模拟人类学琴的路径：

| 阶段 | 数据 | 学习目标 | Loss 屏蔽 |
|------|------|----------|-----------|
| **Phase 1 (预训练)** | MIDI（120K steps） | 音符骨架：音高、节奏、和声、结构 | 屏蔽所有表现力 token（力度/踏板/演奏法等 69 个 token 不计入 loss） |
| **Phase 2 (微调)** | MusicXML（50K steps） | 表现力细节：力度、踏板、装饰音、表情记号 | 全量 loss，所有 token 参与学习 |

MIDI 数据量远大于 MusicXML，先让模型掌握"写什么音符"，再学习"怎么写好听"。

### 乐器级物理约束

对每个乐器施加物理合理的复音上限：钢琴 ≤10 声部、弦乐/铜管/木管 ≤2、合成器 ≤6，并且 subtrack 级音域划分（小提琴、中提琴、大提琴、低音提琴独立）。生成时自动将超出音域的 Note_ON logits 置为 -inf，确保输出符合乐器物理限制。

### 轻量化预设控制系统

内置 7 种风格预设（巴洛克、古典、浪漫、爵士、极简、管风琴圣咏、默认），每种预设封装了温度、top-k、复杂度、乐器、以及锁定策略的组合。同时支持 tunable 参数覆盖，无需重新训练即可切换生成风格。

---

## 训练质量评估体系

28 种 token 类型的 per-type accuracy 独立追踪，每 1000 步在验证集上评估。训练时 TensorBoard 实时监控 loss / 学习率 / 梯度范数 / 各类准确率，运行状态通过 Launch Control 面板集中可视化。

---

## 技术亮点

| 特性 | 说明 |
|------|------|
| **RoPE + cuDNN Attention** | Blackwell GPU 原生优化，SDPA 后端优先使用 cuDNN Attention |
| **FP8 可选量化** | 利用 RTX 5090 `torch._scaled_mm` 做 FP8 矩阵乘法，权重量化缓存避免重复计算 |
| **Weight Tying** | 输入/输出嵌入共享权重矩阵，减少 1.7M 参数 |
| **Gradient Checkpointing** | 整块 TransformerBlock（Attention+FFN）换入换出，激活显存节省 3.2GB |
| **BF16 Autocast** | BF16 训练，无需 GradScaler，tf32 matmul 精度 |
| **原子 Checkpoint 写入** | 临时文件 + rename 防写坏，自动清理历史存档 |
| **25 Worker 并行预处理** | MD5 侧车缓存避免重复扫描，120 秒超时防 worker 挂死 |

---

## 使用的公开数据集

Chopinote-AI 在以下公开数据集的预处理结果上训练：

**MIDI 数据集：**
- [MAESTRO v3.0.0](https://magenta.tensorflow.org/datasets/maestro) — 超过 200 小时的钢琴演奏 MIDI，由国际钢琴比赛录音对齐
- [Lakh MIDI Dataset (lmd_full)](https://colinraffel.com/projects/lmd/) — 约 45,000 首 MIDI 歌曲，涵盖多种风格
- [GiantMIDI-Piano](https://github.com/bytedance/GiantMIDI-Piano) — 10,854 首古典钢琴曲的 MIDI 转录
- [POP909](https://github.com/music-x-lab/POP909-Dataset) — 909 首流行钢琴曲，带主旋律/伴奏分离
- [Aria MIDI](https://github.com/seheonnn/Aria-MIDI-Dataset) — 多风格 MIDI 语料库
- [Bread MIDI](https://github.com/pierredandurand/Bread-MIDI-Dataset) — 精选 MIDI 数据集
- [MusicNet](https://homes.cs.washington.edu/~nfloehr/MusicNet/) — 330 首古典音乐录音 + 对齐标注
- [EMOPIA](https://github.com/annahung31/EMOPIA) — 1,087 首流行钢琴曲，带情感标注

**MusicXML 数据集：**
- [ASAP](https://github.com/fosfrancesco/asap-dataset) — 222 首古典钢琴曲的同步录音 + 乐谱 (MusicXML + MIDI)
- [ATEPP](https://github.com/CPJKU/ATEPP) — 144 首带有演奏法标注的古典钢琴曲
- [Openscore](https://github.com/OpenScore) — 社区贡献的开放乐谱合集

**其他格式数据集：**
- PDMX（Pop Music XML）— 流行音乐乐谱数据集，包含丰富的 annotation（演奏法/力度/踏板/装饰音等）

**内建语料库（config.yaml）：**
- Chopin Complete Works (~250 首)
- Bach Chorales (371 首)
- Beethoven Sonatas (32 首)
