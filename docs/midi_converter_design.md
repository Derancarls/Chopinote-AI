# MIDI Converter 设计记录

## 两个 MIDI 转换器

### 1. `FastMIDIToREMI` (fast_converter.py)
- 基于 `mido`，直接解析 MIDI 二进制
- 速度快 ~80x，用于大规模数据预处理

### 2. `MIDIToREMI` (converter.py)
- 基于 `music21`，解析开销大
- 仅用于对比验证或小规模处理

## 输出等价性

两个转换器输出的 REMI 事件类型**完全一致**：

| Token | Fast (mido) | Music21 (music21) |
|---|---|---|
| BAR | ✓ | ✓ |
| POSITION | ✓ | ✓ |
| PROGRAM | ✓ | ✓ |
| NOTE_ON | ✓ | ✓ |
| VELOCITY | ✓ | ✓ |
| DURATION | ✓ | ✓ |
| TEMPO | ✓ | ✓ |
| TIMESIG | ✓ | ✓ |
| KEY | ✓ | ✓ |
| ANTICIPATE | ✓ (2026-05-11 后补) | ✓ |
| BEAT | ✓ | ✓ |
| BASS | ✓ | ✓ |
| BOS/EOS | ✓ | ✓ |

**差异：**
- 无 — 补完 ANTICIPATE 后功能完全等价
- Fast 不 reject MIDI Type 2（极罕见，无实际影响）

## 鼓过滤

两者都通过 program 号过滤（GM drum programs 112-127），逻辑相同：
```python
# Fast
if prog >= 112: continue

# Music21
_DRUM_PROGRAMS = set(range(112, 128))
if prog in self._DRUM_PROGRAMS: continue
```

不检查 channel 9。如果某 MIDI 的鼓轨在 channel 9 但没有 program_change 到 112+，鼓会漏过。这在实践中几乎不存在（标准 GM 鼓轨必定有 program 112+ 或 drum channel 标记）。

## 为什么 Fast 这么快

music21 的 `converter.parse()` 做了大量工作：建立完整的 Score/Parts/Measures/Notes 对象模型、自动处理时间映射、MusicXML 兼容层。对于 MIDI → REMI 这个场景大部分是浪费的。mido 直接把 MIDI 事件按 tick 排序，省掉所有中间对象。

## ANTICIPATE 补充说明

ANTICIPATE token 预告下一个小节的调性变化，出现在 BAR 之前：
```
<Anticipate G> <Bar> ... <Key G>
```

- 两个转换器都通过 MIDI `key_signature` 元事件检测调性变化
- MIDI key signature 在爬虫数据集中出现率极低（<0.1%），大部分 MIDI 不包含
- 新增此 token 不需要重转已有数据
