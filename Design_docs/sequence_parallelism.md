# 序列并行 (Sequence Parallelism) 设计方案

日期: 2026-05-15
硬件: RTX 5090 32GB × N, PyTorch 2.8.0+cu128
基线: v0.1.2, B=2, T=4096, 1.01B

---

## 一、动机

当前训练瓶颈：
- B=2 已是最小有效 batch
- T=4096, B=2 时 attention 偏置张量 (1, 32, 4096, 4096) bf16 = 1GB
- 想扩展到 T=8192 或 T=16384 时，attention bias → 4GB / 16GB（单卡放不下）
- 单卡 batch=2 利用率已经偏低，减小 batch 进一步降低效率

序列并行通过将 seq 维度切分到多卡，使长序列训练成为可能。

---

## 二、方案对比

### 方案 A: DeepSpeed Ulysses

**原理：** 将 (B, T, d_model) 沿 T 切分到 P 卡，QKV 投影后通过 all-to-all 转置为 (B, T/P, d_model) → 每卡做完整 attention 的 1/P。

```
输入: (B, T, d_model) → chunk → (B, T/P, d_model) 到每卡
QKV 投影: 每卡计算局部 QKV
All-to-All: 转置为按 head 切分 (B, T, d_model/P)
Local Attention: 每卡做完整序列但少 head
```

| 特性 | 值 |
|------|-----|
| 通信 | 2 × all-to-all |
| 通信量 | 2 × d_model × T × sizeof(bf16) |
| 对 attention bias 支持 | **需自定义**（bias 需要全局信息） |
| 额外代码 | ~200 行 |

**T=4096, P=2 通信量:** 2 × 2048 × 4096 × 2 = **32 MB**（可忽略）

### 方案 B: Megatron-LM Sequence Parallelism

**原理：** 将 T 切分到 P 卡，在 column-parallel / row-parallel 的通信边界做 all-reduce / reduce-scatter。

```
每卡持有 (B, T/P, d_model)
LayerNorm: 局部计算（独立）
Attention: QKV 投影后 all-gather → 局部 attention → reduce-scatter
MLP: column parallel → all-reduce → row parallel
```

| 特性 | 值 |
|------|-----|
| 通信 | all-gather + reduce-scatter |
| 通信量 | 2 × d_model × T × sizeof(bf16) |
| 集成难度 | 高（需改写 Linear + Norm） |
| 灵活性 | 与 tensor parallelism 绑定 |

### 方案 C: Ring Attention (Distributed Flash Attention)

**原理：** 将 T 切分到 P 卡，每卡计算局部 attention block，通过 P-1 次点对点通信将 key/value 块在各卡间轮转。

```
每卡持有 (B, T/P, d_model) 的 q, k, v 分块
for round in range(P):
    当前 block attn = softmax(q @ k^T / sqrt(d)) @ v
    send kv_block → next rank
    recv kv_block ← prev rank
```

| 特性 | 值 |
|------|-----|
| 通信 | P-1 次 P2P (send/recv) |
| 通信量 | (P-1) × 2 × d_model × T/P × sizeof(bf16) |
| **对 attention bias 支持** | **分块后各块 bias 独立**，天然支持 |
| 额外代码 | ~150 行（复用 flash attention） |
| 峰值显存 | 1/P × causal mask 显存 |

---

## 三、推荐方案：Ring Attention

### 选择理由

本模型使用了 1GB 的 attention bias（rel + measure + pad + causal mask），DeepSpeed Ulysses 和 Megatron SP 都不允许在 attention 内接自定义 bias 张量。Ring Attention 在切分 seq 的同时保持了每个 block 内的完整 bias 计算，最适配当前架构。

### 实现路径

```
第一阶段 （不改 forward，验证通信）
  └─ ring_attention.py: ring_p2p 通信原语
  └─ 单机 2×RTX 5090 验证正确性

第二阶段 （集成 model.py）
  └─ CausalSelfAttention.forward 接入 ring attention
  └─ bias 生成改为局部：每卡只生成自己分块的 bias
  └─ 支持 gradient checkpointing

第三阶段 （集成 train.py）
  └─ DistributedSampler 按 rank 分数据
  └─ DDP 包装模型
  └─ eval 同步 total_sum / total_tokens
```

### 通信开销估算

| T | P | 单轮通信量 | 总通信量 | BW (NVLink 900 GB/s) | 延迟 |
|---|----|-----------|---------|----------------------|------|
| 4096 | 2 | 1 MB | 1 MB | 900 GB/s | ~1 µs |
| 8192 | 2 | 2 MB | 2 MB | 900 GB/s | ~2 µs |
| 16384 | 4 | 2 MB | 6 MB | 900 GB/s | ~7 µs |

通信开销可忽略。Ring Attention 在 2 卡场景几乎无开销。

---

## 四、注意事项

1. **attention bias 生成方式：** 当前 `_build_total_bias` 生成完整 (1, nH, T, T) bias，需要改为 `_build_block_bias(local_start, local_end, full_T)` 生成局部 bias 块。
2. **训练脚本修改：** `run_curriculum_training.py` 需要 `torchrun` 启动，default world_size=1 向后兼容。
3. **gradient checkpointing：** Ring Attention 的 checkpoint 粒度需要调整——不 checkpoint 整个 block 而是单个 ring iteration。
4. **验证 loss 对齐：** 多卡推理时需要在 rank 0 上聚合 total_sum / total_tokens。
