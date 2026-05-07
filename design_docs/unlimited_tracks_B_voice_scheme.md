# 不限轨方案 B：Program + Voice

## 设计思路

用 `<Program N>` 标识乐器种类，用 `<Voice N>` 标识同一乐器内的声部分层，替代当前固定 `Track_L/Track_R`。

## 编码方式

```
<Position 0>
  <Program 0>
    <Voice 0> <Note_ON 72> <Vel 5> <Dur 4>   ← 钢琴右手（高音谱）
    <Voice 1> <Note_ON 36> <Vel 5> <Dur 4>   ← 钢琴左手（低音谱）
  <Program 40>
    <Note_ON 55> <Vel 4> <Dur 2>              ← 小提琴（Voice 不出现=Voice 0）
```

## 关键行为

- `<Voice N>` 作用域为最近一次 `<Program N>`，换 Program 后 Voice 重置
- 同一 Program 内 Voice 不变时不重复发送
- 仅一个声部时不发 Voice token（默认 Voice 0）

## 信息量

与方案 A 完全等价，不丢失任何信息。

## 与 A 的对比

| | A：`Program N_M` | B：`Program` + `Voice` |
|---|---|---|
| 词表增量 | +128 Program | +128 Program + 少量 Voice |
| 模型理解难度 | 直觉——单个 token 携带完整语义 | 需要学「Voice 在最近 Program 作用域内」的层级关系 |
| 序列长度 | 稍长（换子轨重发完整词） | 稍短（Voice 比 Program_N_M 短） |
| 子轨数上限 | 需硬编码子轨数量 | 无上限 |

## 何时改用 B

- 需要不限子轨数（如管弦乐队写作中同一乐器组有大量分部）
- 更关注序列长度压缩
- 模型已具备足够的 Transformer 深度来理解 Program/Voice 层级

## 与当前 tokenizer 对比

当前：`<Track_L>` / `<Track_R>`（2 轨）
方案 B：128 Program × N Voice = 理论上无限组合

## 转换逻辑（converter.py 改动）

- 移除 `<Track_L>`、`<Track_R>` 发射
- 按 `(measure, position, program)` 分组，每个 program 组开头插入 `<Program N>`
- 同一 program 内多轨时，在每轨的第一个 note 前插入 `<Voice M>`
- 排序 key：`(measure, position, program, voice, priority)`
