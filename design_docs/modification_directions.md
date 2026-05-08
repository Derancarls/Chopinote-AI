# 修改方向记录

## 2026-05-09 — 乐器分轨改进（已完成）

**改动：** 针对「小提琴声部出现和弦」和「钢琴音高窜到小提琴声部」的修复。

### 措施
1. **乐器级复音上限** — 新增 `INSTRUMENT_POLYPHONY_CAP` 字典，按乐器类别设定同 position 每 track 的最大同时发音数：
   - 弦乐/铜管/木管/贝斯：上限 2（单音旋律乐器）
   - 钢琴：上限 10（可弹和弦）
   - 其他：4-8
   - 新增 `get_polyphony_cap(program)` 函数查询

2. **Per-track polyphony 追踪** — polyphony 限制从全局统一 cap 改为 per-(program, subtrack) 独立追踪，不同乐器不再互相影响

3. **弦乐 subtrack 级音域** — 扩展 `SUBTRACK_RANGES` 加入 Violin/Viola/Cello/Contrabass/Tremolo/Pizzicato/Ensemble 的 subtrack 划分

### 文件
- `chopinote_model/generate.py`: 新增常量和函数，扩展 SUBTRACK_RANGES
- `chopinote_cli/main.py`: 重构 polyphony 追踪逻辑，默认 max_polyphony 10→8

## 2026-05-08 — 实测问题记录

### 1. 乐器分轨混乱

**现象：**
- 小提琴声部出现同一时间 4-5 个音（弦乐器本质是单音旋律乐器，不该有和弦式织体）
- 钢琴的音高跑到了小提琴声部，声部器乐边界模糊

**可能方向：**
- subtrack 级音域约束需要加强，目前仅对钢琴 0-7 有 SUBTRACK_RANGES 细分（右手 48-96，左手 28-72），其他乐器未做 subtrack 级限制
- training data 中弦乐声部多为单音织体，但模型可能从钢琴的多音模式中学习了跨声部复调
- 可考虑在 logit 操作链中针对非钢琴乐器增加 polyphony 上限（如弦乐 ≤2 音/拍）

### 2. 和声理解不足

**现象：**
- 生成中出现大量不和谐音程（小二度、增四度等）
- 调性音高偏置（key_bias）存在但力度不够，模型经常偏离调性

**可能方向：**
- 调性偏置强度（key_bias_strength）从默认 2.0 调高
- 训练数据中增加和声进行（cadence、functional harmony）的标注或采样权重
- 后处理和声校正：检测并修正明显的不和谐音程
- 模型仅 10 层 768 dim，可能容量不足以充分学习调性和声
