# 调性与和声方案（Tier 1 / 2 / 3）

## 概述

分三阶段引入调性和声学概念，模型逐步从「绝对音高统计」进化为「调性感知 → 和声功能理解 → 相对音级推理」。

---

## Tier 1：Key 调性标记（推荐优先实现）

### 思路

在 token 序列中加入调性标记 `<Key G>` / `<Key Cm>`，让模型通过 attention 感知当前调性。

### 编码方式

```
<BOS> <Key C> <TimeSig 4/4> <Position 0> <Program 0> <Note_ON 60> ...
```

- 24 个 Key token：C, G, D, A, E, B, F#, C#, F, Bb, Eb, Ab, Db, Gb, Cb (major) + Am, Em, Bm, F#m, C#m, G#m, D#m, A#m, Dm, Gm, Cm, Fm, Bbm, Ebm, Abm, C#m (minor) 等
- 实际：12 major + 12 minor = 24 tokens
- 变调时重新发射 Key token
- 只标记「当前主调」，不需要每次都发——调不变则不重复

### 关键行为

- Key token 作用域：从出现到下一个 Key token 或 EOS
- 调不变时不重复发送
- 多调性乐曲保留全局 key（主调），不追踪每小节临时转调

### 信息来源（converter）

- **MusicXML**：直接从 key signature 提取（`<key><fifths>1</fifths><mode>major</mode></key>`）
- **PDMX**：使用 music21 的 Krumhansl-Schmuckler key finder （`analysis.key.KeyAnalyzer`）从音符分布推断
  - 小节级别分析 → 多数决确定全局主调
  - 置信度过低时（< 50%）不发射 Key token，留给模型自行推断
- **人工验证**：首次实现后抽检 100 首验证准确率

### 信息量

- 词表增量：+24（12 major + 12 minor）
- 序列长度增量：+1 per piece（通常只有开头发一次）
- 训练数据无需额外标注，自动推导

### 对模型的影响

- Token embedding 层学会 key 与音级的关系
- Attention 层将音符行为关联到调性上下文
- 生成时更少出现「跑调」音符
- 为 Tier 2 的和弦分析提供必要的调性上下文

### 实现要点

1. tokenizer.py：新增 KEY 常量，24 个 key tokens
2. converter.py：MusicXML 从 key signature 提取，PDMX 用 key finder 推断
3. generate.py：不需要改动（Key token 会自动被模型学习）
4. config.py：vocab_size += 24

---

## Tier 2：Chord 功能和弦标记（后续阶段）

### 思路

在 harmonic rhythm 节点插入和弦功能 token：`<Chord I>` / `<Chord V7>` / `<Chord ii°>` 等，给模型「和声思路链」。

### 编码方式

```
<Position 0> <Chord I> <Note_ON 60> <Note_ON 64> <Note_ON 67>
<Position 4> <Chord V7> <Note_ON 62> <Note_ON 66> <Note_ON 69>
```

### 关键行为

- Chord token 标记该 position 开始的 harmonic function
- 使用 Roman numeral 标注法（大写=major，小写=minor，°=dim，+=aug）
- 和弦变化时才发送（不变不重复）
- 一个和弦持续期间，下方音符皆为该和弦的 voicing / arpeggiation

### 信息来源

- 依赖 Tier 1（需要 key context）
- music21 的 `roman.RomanNumeral` 分析
- 需要先做 chord identification（MIDI 上有歧义，可接受准确率 70-80%）

### 词表

约 30-40 tokens（I / ii / iii / IV / V / vi / vii° 及其七和弦 variant）

### 风险

- 和弦分析对 MIDI 准确率有限（缺少 voicing 信息）
- 错误的 Chord token 会误导模型，比没有更差
- 需先验证 chord identification 准确率再做决定

---

## Tier 3：Scale Degree 相对音级编码（未来探索）

### 思路

修改 embedding 层，让模型学的不是绝对音高（Note_ON 60 = C4），而是音级 + 八度（Scale degree 1-7 + octave）。

### 编码方式

```
<Degree 1> <Octave 4> <Vel 5> <Dur 4>     ← 主音（在 C 大调中是 C4）
<Degree 5> <Octave 4> <Vel 5> <Dur 4>     ← 属音（G4）
<Degree 7> <Octave 4> <Vel 5> <Dur 4>     ← 导音（B4 → 倾向主音）
```

### 关键行为

- 每个音符拆为 `<Degree N>` + `<Octave N>` 两个 token
- 调性变化时所有音级重新映射
- Key 决定了 degree → MIDI pitch 的映射表

### 架构影响

- 不再是简单的词表扩展，而是**输入表示的改变**
- Embedding 层需要条件：`embedding = key_embedding + degree_embedding + octave_embedding`
- 输出层需要从 degree + octave 映射回 MIDI pitch
- 生成时 degree 和 octave 两个 token 需要联合采样（或顺序采样）

### 优势

- 模型直接学到「调内音级关系」，而非「绝对音高统计」
- 跨调泛化：学到的和声进行规则可迁移到所有调
- 生成结果天然在调内

### 代价

- 重构输入/输出层，需要重新训练
- 序列长度 ×2（每个音符从 1 个 token 变 2 个）
- 需要验证联合采样的可靠性

### 前置条件

- Tier 1 已稳定
- 模型对调性已有感知
- 有足够的计算资源进行重新训练
