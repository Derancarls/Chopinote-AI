# MLP 参数量优化：Grouped / Cross-Layer 共享方案

日期: 2026-05-15
硬件: RTX 5090 32GB, PyTorch 2.8.0+cu128
基线: v0.1.2, 1.01B params, d_model=2048, d_ff=8192, n_layers=20

---

## 一、现状分析

当前 MLP (FeedForward) 为标准 GPT 架构：

```
FFN(x) = GELU(x @ W1) @ W3 + x @ W2  (d_ff=8192)
```

每层参数量：
- W1: (2048, 8192) = 16.8M
- W2: (8192, 2048) = 16.8M
- W3: (2048, 8192) = 16.8M
- **每层合计: 50.3M**

20 层 MLP 总参数量: **1.006B** (占模型总参数量 ~99%)

**瓶颈：** MLP 占了模型几乎全部参数。attention 仅占 ~2% (QKV+O = 4 × 2048×64×32 ≈ 16.8M/层 × bias ≈ 17M，20 层 = 340M，但 weight tying 后实际上 embedding 占大头)。

---

## 二、优化方案

### 方案 A: Cross-Layer FFN Sharing (ALBERT 风格)

所有层共享同一套 MLP 权重 W1/W2/W3。

| 指标 | 基线 | 共享后 |
|------|------|--------|
| MLP 参数 | 1,006M | 50.3M |
| 总参数 | ~1,010M | ~55M |
| 理论加速 | 1.0x | 受 memory-bound 限制，~1.0x |
| VRAM (optimizer states) | ~8G (Adam) | ~0.5G |

**优点：** 参数量降到 1/20，适合小数据场景，过拟合风险低。
**缺点：** 表达能力受限，RTX 5090 上计算量没变（仍是 d_ff=8192 的 matmul），只省了参数量和 optimizer states 的 VRAM。
**适合场景：** 数据量 < 10B tokens，需要防止过拟合。

### 方案 B: Grouped MLP (GQA 风格)

受 GQA 启发，将 FFN 的中间维度 d_ff 分成 g 组，每组处理 d_model/g 的输入子空间：

```
x_split = x.chunk(g, dim=-1)                    # g × (B, T, d_model/g)
hidden = [GELU(x_i @ W1_i) for x_i in x_split]  # 每组独立投影
hidden = torch.cat(hidden, dim=-1)               # (B, T, d_ff)
output = hidden @ W3
```

当 g > 1 时，参数从 `d_model × d_ff` 降为 `g × (d_model/g) × (d_ff/g) = d_model × d_ff / g`。

| g 值 | 每层 MLP 参数 | 总 MLP 参数 | 节省 |
|------|--------------|-------------|------|
| 1 (基线) | 50.3M | 1,006M | — |
| 2 | 25.2M | 503M | 50% |
| 4 | 12.6M | 252M | 75% |
| 8 | 6.3M | 126M | 87.5% |

**优点：** 参数和计算量同比例下降，实际加速可测。
**缺点：** 每组只看到部分输入维度，可能影响表示质量。

### 方案 C: FFN Width Reduction

直接减小 d_ff：

| d_ff | 每层 MLP 参数 | 总 MLP 参数 | FLOPs 比例 |
|------|--------------|-------------|-----------|
| 8192 (基线) | 50.3M | 1,006M | 1.0x |
| 4096 | 25.2M | 503M | 0.5x |
| 2048 | 12.6M | 252M | 0.25x |

**优点：** 最简单直接，实际加速明显。
**缺点：** d_ff 减小直接降低模型容量，需要实验确定最优值。

### 方案 D: Low-Rank MLP (LoRA-like)

将 W1 分解为 A×B 两个低秩矩阵：

```
FFN(x) = GELU(x @ A1 @ B1) @ (B3 @ A3) + x @ A2 @ B2
```

| 秩 r | 每层 MLP 参数 | 总 MLP 参数 |
|------|--------------|-------------|
| 1024 | 12.6M | 252M |
| 512 | 6.3M | 126M |
| 256 | 3.2M | 63M |

**注意：** 需要 3 个 matmul 而不是 2 个（前向 A1→B1 串联），latency 可能增加。

---

## 三、推荐方案

**短期（1-2 天）：方案 C**——d_ff 减半到 4096。改动最小 (config.py 一行)，参数量 503M，从 1B 降到 0.5B，训练速度预计 2x+。

**中期：方案 B**——Grouped MLP g=4。保留表达力，参数量 252M，兼顾质量和速度。

**不做：方案 A、D。** 方案 A 省参数量不省计算，VRAM 不是当前瓶颈。方案 D 分解后串行 matmul 增加延迟。
