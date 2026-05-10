# Bass Note Token 设计方案

## 动机

当前模型在和声理解上存在不足（小二度、增四度等不和谐音程），主要原因之一是模型缺少「当前和声的根基音」信息。

Bass Note token 标记每个 position 上**所有发音乐器中最低的音的音级**（pitch class），给模型一个清晰的「低音在走什么和声功能」的信号。它不依赖任何外部标注，从现有音符数据中即可 100% 准确提取。

与完整罗马数字和弦标记（Tier 2）相比：Bass Note 检测零错误、词表 +12、覆盖全部数据。

---

## 编码方式

### Token 格式

```
<Bass 0>   ← C（最低音音级为 C）
<Bass 1>   ← C#/Db
...
<Bass 11>  ← B
```

共 12 个 token，对应 12 个 pitch class。

### 在序列中的位置

当前序列结构：
```
<Bar N> <Position X> <Program A> <Note_ON P1> <Vel V1> <Dur D1> <Program B> <Note_ON P2> ...
```

改为：
```
<Bar N> <Position X> <Bass Y> <Program A> <Note_ON P1> <Vel V1> <Dur D1> <Program B> <Note_ON P2> ...
```

- Bass token 紧跟在 Position 之后，在任何 Program/Note_ON 之前
- 同一 position 的所有 Program 共享同一个 Bass token（低音是全局和声概念）
- 该 position 无音符（全部休止）时，不发射 Bass token

### 信息量

词表增量：+12（vocab 830 → 842）
序列长度增量：+0～1 per position（有音符的 position 才发）

---

## 检测逻辑

三个 converter（MusicXMLToREMI / PDMXToREMI / MIDIToREMI）的改动模式相同。

### 步骤

1. **Pre-scan pass**：在事件排序后、正式组装前，遍历 `merged` 列表，收集每个 `(measure, position)` 的所有普通音符（kind='n'）的 pitch
2. **计算最低音**：对每个 `(measure, position)`，取 pitch 最小值，`bass_pc = min_pitch % 12`
3. **发射**：在 `_assemble_events` 主循环中，发射 Position token 之后、Program token 之前，检查该 `(measure, position)` 是否有 bass_pc，有则发射 `<Bass bass_pc>`

```python
# pre-scan pass
pos_bass: dict[tuple[int, int], int] = {}
for m_idx, pos, _prog, _sub, _pri, kind, data in merged:
    if kind == 'n':
        pitch = data[0]
        key = (m_idx, pos)
        if key not in pos_bass or pitch < pos_bass[key]:
            pos_bass[key] = pitch

# 组装时，在 Position 后插入
if pos != cur_pos:
    events.append((REMITokenizer.POSITION, pos))
    cur_pos = pos
    cur_program = -1

    bass_pc = pos_bass.get((cur_measure, pos))
    if bass_pc is not None:
        events.append((REMITokenizer.BASS, bass_pc % 12))
```

---

## 修改文件清单

### chopinote_dataset/tokenizer.py

| 改动 | 说明 |
|------|------|
| 新增 `BASS = '<Bass'` 常量 | token 类型前缀 |
| 新增 `build_vocab` 中 12 个 token：`<Bass 0>` ~ `<Bass 11>` | vocab 830→842 |
| 新增 `detokenize` 分支 | `<Bass N>` → `(BASS, N)` |

### chopinote_dataset/converter.py

| Converter | 改动位置 | 改动 |
|-----------|---------|------|
| MusicXMLToREMI._assemble_events | ~行 364（排序后）+ ~行 381（Position 后） | pre-scan + 发射 |
| PDMXToREMI._convert_pdmx（组装部分） | 同模式 | 同上 |
| MIDIToREMI._convert_score（组装部分） | ~行 930 区域 | 同上 |

### chopinote_model/config.py

```python
vocab_size: int = 842  # 830 → 842
```

### chopinote_model/generate.py

| 函数 | 改动 |
|------|------|
| `tokens_to_notes` | Bass token 加入 skip 列表（不生成音符） |
| `generate_with_progress` | 无需改动（Bass 由模型自然预测） |

### chopinote_cli/main.py

无需改动。Bass token 作为上下文 token 存在，不参与 logit 操作链。

---

## 生成端行为

Bass token 不需要特殊的 logit 干预：

- **训练时**：Bass token 在序列中作为普通 token，模型通过 attention 学习低音与上方音符的关系
- **推理时**：模型在 Position 后自然预测 Bass token，后续 NOTE_ON 的分布受其影响
- **tokens_to_notes**：遇到 `<Bass N>` 直接跳过（不产生音符）

这意味着：
- 如果模型学到「在 C 大调中 Bass=0 时，NOTE_ON 更可能预测 0, +4, +7（C大三和弦）」，调性偏置的作用被 Bass token 替代了一部分
- 低音进行（bass line）会变得更连贯，因为模型显式看到了每个 position 的最低音

---

## 数据重处理

由于 vocab 变化（830→842）且 token 序列内容变化，所有旧 token 数据必须重新转换：

```bash
python scripts/prepare_corpus.py
python scripts/prepare_corpus.py --midi
python scripts/preprocess_pdmx.py
```

### 兼容性矩阵

| 项目 | 兼容？ |
|------|--------|
| 旧 checkpoint（830 vocab） | ❌ embedding/lm_head 形状变化 |
| 旧 token 数据文件 | ❌ 无 Bass token，且后续 ID 偏移 |
| best.pt / step_94000_best.pt | ❌ 必须从头训练或在新数据上微调 |
| 训练管线（train.py, dataset.py） | ✅ 无需改动 |

---

## 验证方法

1. **Bass 检测正确性**：对任意 MusicXML，检查最低音的音级是否与 `<Bass N>` 一致
2. **编解码循环验证**：转换 → 逆转换 → music21 音高对比
3. **生成测试**：使用旧权重生成一次（无 Bass token）→ 观察新权重训练后生成的低音是否更连贯
4. **词表对齐检查**：`tokenizer.vocab_size == 842`，Bass token ID 不与其他 token 冲突

---

## 与现有调性系统的关系

| 组件 | 作用 | 与 Bass 的关系 |
|------|------|---------------|
| `<Key C>` | 标记全局调性（曲子用哪个调） | Bass 基于实际最低音，Key 基于谱面调号 |
| 调性偏置（key_bias）| 推理时提升调内音 | Bass 是数据层面的信号，两者互补 |
| 相对音高编码 | NOTE_ON 存 interval 而非绝对音高 | 正交，Bass 也在 interval 空间中计算 |

Bass token 不替代现有调性系统，而是补充——让模型在每个 position 显式知道低音音级，从而自己学会「低音走 C → 上面更可能出现 E-G」这种和声关系。
