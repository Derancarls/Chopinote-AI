# 项目现状与差距分析：从当前模型到"宏伟的乐谱"

日期: 2026-05-16
基线: v0.1.2-rope, 1.01B, RoPE 位置编码, REMI 872 vocab

---

## 一、当前已实现

### 数据层
- REMI tokenizer (872 vocab, 28 token types) — 覆盖音符/力度/演奏法/踏板/装饰音/反复/跳转等
- 4 种 converter: FastMIDIToREMI / MusicXMLToREMI / PDMXToREMI / MIDIToREMI
- 1.7M+ 源文件预处理管道，hash sidecar 缓存
- 训练/验证/测试集合划分

### 模型层
- MusicTransformer: 1.01B, d_model=2048, d_ff=8192, n_layers=20, n_heads=32
- RoPE 位置编码，measure_embedding 注入小节结构
- gradient checkpointing, bf16, weight tying
- KV cache 推理加速
- FP8Linear 可选量化

### 推理层
- 自回归采样 (top-k, temperature)
- Preset 系统 (baroque/romantic/jazz 等 7 种风格)
- MusicXML 导出（基础音符/和弦/力度/踏板/渐强渐弱/演奏法）

---

## 二、差距分层

### 第 0 层：模型从未训到收敛

**现状：** ALiBi 和 RoPE 两个版本都没有完整训练经历。

**影响：** 任何架构层面的讨论（d_ff 减半、Grouped MLP、batch size）在模型训到收敛前都是推测。loss 曲线、生成质量、过拟合程度这些基础数据全部缺失。

**优先级：最高。先跑完一次训练再谈其他。**

---

### 第 1 层：长程结构（架构级天花板）

Transformer 的自注意力局部偏重和 max_seq_len=4096 限制，使模型天然擅长局部连贯（乐句级）但不擅长全局结构（乐章级）。

**具体问题：**
- 4096 tokens ≈ 100-200 小节，不够一首完整的奏鸣曲/回旋曲
- 即便增大 max_seq_len，softmax attention 在几千 token 后注意力分布发散，远端信息难以利用
- 没有显式的 section/part/phase 层级建模

**可能方向：**
- 二阶段生成：先写"草稿"（压缩的段落级序列），再展开为细节
- 层次化 tokenization：增加 paragraph/section token 作为结构锚点
- 记忆增强：在注意力之外增加一个可读写的全局状态

---

### 第 2 层：可控生成（产品级缺失）

当前生成控制接口只有 temperature、top-k、key lock 三个参数。

**缺失的能力：**
- **曲式控制**：回旋曲 (ABACA)、奏鸣曲式、变奏曲——没有表达方式
- **和声约束**：不能指定调性/和弦进行
- **结构重复**：不能要求"第 8-16 小节高八度重复"
- **段落衔接**：没有 transition 的显式控制
- **条件生成**：没有 composer style conditioning、no text prompt

**可能方向：**
- 结构 prompt：在输入序列中嵌入结构 token（RepeatSection, DaCapo 等）作为条件
- 和声骨架先验：先生成 chord progression 再填充细节
- 基于 preset 系统的扩展：preset 从简单参数集进化为结构模板

---

### 第 3 层：表现力输出（乐谱质量细节）

MusicXML 导出实现了基础机能，但距离出版级乐谱有差距。

**缺失／不完善：**
- 连奏线 (slur/tie) —— 缺失，严重影响可读性
- 八度标记 (ottava) —— 缺失，高音区一片上加线
- 声部分配 (voice) —— 缺失，和弦音头朝向混乱
- 延音踏板 —— 按 position 插入，无智能对齐
- 装饰音标记 —— 倚音已有，trill/mordent 等缺失
- 休止符合并 —— 多声部休止符可能重叠
- 谱号切换 —— 高/低音谱号未按音域自动选择

---

### 第 4 层：评估体系（迭代工具缺失）

没有 loss 以外的任何质量指标。模型改进只能靠耳朵听。

**缺失的指标：**
- 和声合理性：pitch histogram、key consistency、dissonance rate
- 节奏多样性：note density、groove consistency、空拍分布
- 结构重复度：self-similarity matrix 分析
- 生成多样性：inter-annotation distance (MMD)
- 可控性：条件控制 vs 输出的对应关系量化

---

## 三、推荐优先级排序

```
优先级 1 ─── 训练到收敛（0 到 1）
                │
优先级 2 ─── B=8 + grad_accum=4 加速训练（送的）
                │
优先级 3 ─── 长程结构建模（架构天花板）
                │
优先级 4 ─── 可控生成接口（产品差异化）
                │
优先级 5 ─── MusicXML 表现力输出完善（乐谱质量）
                │
优先级 6 ─── 评估指标体系（迭代工具）
```

第 1-2 步可以在同一轮训练中完成。第 3 步需要单独的设计和实验。
第 4-6 步可以并行推进，不阻塞训练。
