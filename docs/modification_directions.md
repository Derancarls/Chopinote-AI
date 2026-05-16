# 待办任务与优化方向

> 整合自：tonality_and_harmony_scheme.md / unlimited_tracks_B_voice_scheme.md / conditional_embedding_path_b.md / reference_models_research.md
> 已移除已实现项：相对位置注意力、Tier 1 Key 调性标记、乐器分轨改进、CLI 预设系统、复杂度控制、复音上限、标记系统、两阶段训练

---

## 一、当前生成质量瓶颈

### 1. 和声理解不足

**现象：**
- 生成中出现大量不和谐音程（小二度、增四度等）
- 调性音高偏置（key_bias）存在但力度不够，模型经常偏离调性

**可能方向：**
- 调性偏置强度（key_bias_strength）从默认 2.0 调高
- 训练数据中增加和声进行（cadence、functional harmony）的标注或采样权重
- 后处理和声校正：检测并修正明显的不和谐音程
- 模型仅 10 层 768 dim，可能容量不足以充分学习调性和声

---

## 二、编码方案改进

### 2. 相对音高编码（音程制）

NOTE_ON 从绝对 MIDI 音高（0-127）改为存储到主调主音的**半音程**（interval），让模型学到音程关系而非绝对音高。

**改点：**
- tokenizer：NOTE_ON 范围 0-127 → -60 ~ +60，vocab 837→830
- converter：三个 converter 在 NOTE_ON 发射时计算 `interval = pitch - tonic`
- generate.py：推理时将 interval 转回绝对音高
- main.py：音高限制/调性偏置改为 interval 空间计算
- 所有旧 token 数据必须重新处理

### 3. Tier 2：Chord 功能和弦标记

在 harmonic rhythm 节点插入和弦功能 token：`<Chord I>` / `<Chord V7>` / `<Chord ii°>` 等，给模型「和声思路链」。

**关键行为：**
- Chord token 标记该 position 开始的 harmonic function
- 使用 Roman numeral 标注法
- 和弦变化时才发送（不变不重复）
- 词表约 30-40 tokens

**前置条件：**
- 依赖 Tier 1（Key 调性标记，已实现）
- 需要 music21 的 `roman.RomanNumeral` 分析
- 需先验证 chord identification 准确率再做决定

**风险：** 和弦分析对 MIDI 准确率有限，错误的 Chord token 会误导模型。

### 4. Tier 3：Scale Degree 相对音级编码

修改 embedding 层，将音符拆为 `<Degree N>` + `<Octave N>` 两个 token。模型直接学「调内音级关系」而非「绝对音高统计」。

**架构影响：**
- Embedding 层需要条件：`embedding = key_embedding + degree_embedding + octave_embedding`
- 输出层需要从 degree + octave 映射回 MIDI pitch
- 序列长度 ×2（每个音符从 1 个 token 变 2 个）
- 需要重训，不兼容当前权重

**前置条件：**
- Tier 1 已稳定（✅ 已完成）
- 有足够的计算资源进行重新训练

---

## 三、架构改进（参考模型研究）

### 5. Anticipatory 控制点（优先级 1）

在序列中插入控制点（control tokens）作为 stopping times，让生成条件化于未来事件。

- **改动量：** 改数据格式 + 训练时插入 control token
- **预期收益：** 替代/补充推理时调性偏置，生成更平滑的调性/风格过渡
- **创新空间：** 推理时动态插入控制点

### 6. Compound Word + Delay Pattern（优先级 2）

**参考：** Compound Word Transformer / MusicGen Delay Pattern

将所有属性（pitch/duration/velocity/instrument）压缩为一个复合 token，子字段用 Delay Pattern 错开步长自回归预测。

- **预期收益：** 序列压缩 6×，生成速度约 3× 加速
- **代价：** 多轨 + 表现力下 vocab 设计复杂

### 7. FC-Attention（优先级 3，超长序列场景）

**参考：** Museformer (Microsoft, NeurIPS 2022)

精细注意力关注结构相关小节的全部 token + 粗粒度注意力关注其他小节的 summary token。

- **预期收益：** 3× 当前序列长度而不 OOM
- **场景：** 需要 seq_len > 4096 且不想改 tokenization 时

### 8. Cadenza 两层分离架构（优先级 4）

**参考：** Cadenza (2024)

Composer（VAE with RoPE）生成乐谱骨架 + Performer（Bi-Encoder）添加演奏细节。

- **与本项目关联：** 当前两阶段训练已隐含骨架→细节学习，可进一步显式化
- **代价：** 需要两个独立模型

---

## 四、Track/Program 系统

### 9. 不限轨方案 B：Program + Voice

用 `<Program N>` 标识乐器种类 + `<Voice N>` 标识同一乐器内的声部分层，替代当前固定 Track_L/Track_R。

**编码方式：**
```
<Position 0>
  <Program 0>
    <Voice 0> <Note_ON 72> <Vel 5> <Dur 4>   ← 钢琴右手
    <Voice 1> <Note_ON 36> <Vel 5> <Dur 4>   ← 钢琴左手
  <Program 40>
    <Note_ON 55> <Vel 4> <Dur 2>              ← 小提琴
```

**与当前对比：**
- 当前：`<Track_L>` / `<Track_R>`（2 轨）
- 方案 B：128 Program × N Voice = 理论上无限组合
- 子轨数无上限（管弦乐队写作中同一乐器组大量分部）
- 序列比当前方案稍短

**转换逻辑：**
- 移除 Track_L/Track_R 发射
- 按 `(measure, position, program)` 分组，每组开头插入 `<Program N>`
- 同一 program 内多轨时，在每轨第一个 note 前插入 `<Voice M>`

---

## 五、条件控制

### 10. 条件嵌入层（路径 B）

在模型 embedding 层加入可学习的条件嵌入，让模型显式学习跟随条件，支持风格插值。

**架构变更：**
```
当前：input = token_embedding(input_ids) + pos_embedding
改为：input = token_embedding(input_ids) + pos_embedding + cond_embedding
```

**条件嵌入表：** key_embed (30) + time_embed (14) + tempo_embed (22) + style_embed (N) + program_embed (128)

**优点：**
- 条件被模型显式学习，服从度高
- 支持风格插值（如 0.7×baroque + 0.3×romantic）
- CFG 可提升生成质量
- CLI 参数体系可复用

**代价：**
- 需改模型架构 + 重训
- 不兼容当前权重
- 风格标签需额外标注

---

## 六、数据

### 待下载数据
| 数据集 | 格式 | 规模 | 状态 |
|--------|------|------|------|
| Bach Chorales | MusicXML | 371 曲 | ❌ 未下载，目录为空 |
| Beethoven Sonatas | MusicXML | 32 曲 | ❌ 未下载，目录为空 |
| Chopin Complete Works | MusicXML | 250 曲 | ❌ 未下载，目录为空 |
| GiantMIDI-Piano（完整） | MIDI | ~10,854 曲 | ❌ 仅有 160 首 evaluation 子集 |
| POP909 | MIDI | 909 曲 | ✅ 已下载，未解压 |
| EMOPIA | MIDI | 1,087 曲 | ✅ 已下载，未解压 |
| ATEPP | MIDI | 144 曲 | ✅ 已下载，未解压 |
| MusicNet | MIDI | 330 曲 | ✅ 已下载，已解压 |

### 数据预处理
- 新数据集（POP909/EMOPIA/ATEPP/MusicNet）需解压并预处理
- 若实现相对音高编码，所有 token 数据需重新转换
- 需重新运行 `split_dataset.py` 生成最终 train/val/test 划分
