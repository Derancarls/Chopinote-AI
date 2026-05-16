# 多维相对注意力优化 + FP8 混合精度设计方案

日期: 2026-05-14
硬件: RTX 5090 32GB (Blackwell sm_120), PyTorch 2.8.0+cu128

---

## 一、现状分析

### 当前注意力架构 (model.py:37-72)

`_build_total_bias()` 每层每 step 做 4 件事:

1. `rel_bias(dist)` — Embedding lookup (8191, 32) → (T, T_kv, nH), permute → (1, nH, T, T_kv)
2. `pad_bias` — 加到 total_bias
3. `measure_bias(dist)` — Embedding lookup (511, 32) → permute → add
4. `causal mask` — triu fill → add

结果: (1, 32, 4096, 4096) bf16 = 1GB 的 attn_mask 传给 SDPA。

两个 `nn.Embedding` 被 MATERIALIZED 成完整的 `(nH, T, T_kv)` 张量再传入 SDPA。
FlashAttention 需要把这个 1GB bias 从 HBM 读进 SRAM，这是纯带宽开销。

### 模型维度

| 参数 | 值 |
|------|-----|
| d_model | 2048 |
| n_layers | 20 |
| n_heads | 32 |
| head_dim | 64 |
| d_ff | 8192 |
| max_seq_len | 4096 |
| max_measures | 256 |
| vocab_size | 872 |
| 总参数量 | ~1.01B |

---

## 二、多维相对注意力优化

### 路线 A: ALiBi 风格斜率 (推荐)

将 per-distance learned embedding 替换为 per-head 可学习标量斜率:

```python
# 当前: 262,432 参数 (8191*32), 1GB 中间张量/layer
self.rel_bias = nn.Embedding(8191, 32, dtype=torch.bfloat16)
self.measure_bias = nn.Embedding(511, 32, dtype=torch.bfloat16)

# 优化后: 64 参数, 0 中间张量 (fused into SDPA kernel)
self.rel_slope = nn.Parameter(torch.ones(32) * 0.01)     # (n_heads,)
self.meas_slope = nn.Parameter(torch.ones(32) * 0.01)    # (n_heads,)
```

Bias 计算简化为:

```python
def _build_total_bias(self, T, T_kv, mask, measure_ids, device, dtype):
    # ALiBi slopes: 距离 × 可学习斜率 → 直接传入 SDPA attn_mask
    # FlashAttention-3 可融合此简单函数，无需物化 (nH, T, T_kv) 张量
    i = torch.arange(T, device=device).view(-1, 1)
    j = torch.arange(T_kv, device=device).view(1, -1)
    pos_dist = (i - j).abs().float().unsqueeze(0)        # (1, T, T_kv)
    pos_bias = -self.rel_slope.view(-1, 1, 1) * pos_dist # (nH, T, T_kv)

    if measure_ids is not None:
        m_q = measure_ids[-T:].unsqueeze(-1)
        m_k = measure_ids[:T_kv].unsqueeze(0)
        meas_dist = (m_q - m_k).abs().float().unsqueeze(0)
        meas_bias = -self.meas_slope.view(-1, 1, 1) * meas_dist
        pos_bias = pos_bias + meas_bias

    # Pad mask + causal 保持不变
    ...
    return total_bias.unsqueeze(0)
```

**为什么质量风险低:**
- ALiBi 在 LLM 中广泛验证 (Press et al., 2022)
- 对于音乐的相对音高/节奏关系，距离衰减是更自然的归纳偏置
- 原始的 per-distance embedding 可能对大部分距离过拟合
- 64 个可学习参数 vs 262,432 — 更强的正则化

**收益:**
- 节省 ~1GB/layer 的中间张量（可用于增大 batch）
- 消除 4 次 kernel launch (Embedding + permute + 3× add_)
- 偏置可直接由 FlashAttention kernel 内部计算
- 预估注意力部分加速: ~1.15×

**风险:**
- 音乐中某些特定距离可能有特殊意义（如八度=12 半音），线性斜率无法建模非单调关系
- 验证方法: 在小模型 (4 层) 上对比 val loss 曲线

### 路线 B: 对数分桶 (保留非线性的中间方案)

```python
# 将 8191 个距离映射到 64 个对数桶
buckets = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192,
           256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]
self.rel_bias = nn.Embedding(64, 32)  # 2048 参数 vs 262,144
```

优点: 保留每个桶的独立学习能力。对数间隔对音乐有意义（近处精细、远处粗粒度）。

### 路线 C: 核融合 (不改变模型，纯工程优化)

保持现有 Embedding，将 4 个 add 操作融合成一个 Triton kernel:

```python
@triton.jit
def fused_bias_kernel(total_bias, rel_bias, measure_bias, pad_mask, ...):
    # 单次 kernel 完成所有加法 + causal fill
```

优点: 不改变模型质量，仅减少 kernel launch overhead (~10-15% 加速)。

---

## 三、FP8 混合精度可行性

### 实测数据 (RTX 5090, _scaled_mm vs torch.mm)

| 矩阵乘法 | 形状 | BF16 | FP8 (e4m3fn) | 加速比 |
|----------|------|------|--------------|--------|
| QKV 投影 | 8192×2048×6144 | 0.97ms | 0.46ms | **2.10×** |
| Out 投影 | 8192×2048×2048 | 0.39ms | 0.17ms | **2.26×** |
| FFN 上投影 | 8192×2048×8192 | 1.25ms | 0.65ms | **1.94×** |
| FFN 下投影 | 8192×8192×2048 | 1.34ms | 0.70ms | **1.92×** |
| 8K×8K 方阵 | 8192×8192×8192 | 4.79ms | 2.44ms | **1.96×** |
| 16K×16K 方阵 | 16384×16384×16384 | 37.86ms | 19.60ms | **1.93×** |

Blackwell FP8 tensor core 实战: ~450 TFLOPS vs BF16 ~220 TFLOPS，稳定 2×。

### 即插即用的障碍

PyTorch 2.8 的现实:
- `torch.amp.autocast('cuda', dtype=torch.float8_e4m3fn)` → **不支持 nn.Linear**（底层 `addmm_cuda` 没有 FP8 实现）
- `torch._scaled_mm` 输出 FP8 张量 → `.backward()` **失败**（FP8 不支持 sum/mean 等 reduction 算子）
- 没有 `torch.ao.float8`、没有 `Float8Linear`、没有 TransformerEngine
- 唯一可用原语: `torch._scaled_mm(A_fp8, B_fp8, scale_A, scale_B, out_dtype=torch.bfloat16)`

### 实现路线: 自定义 FP8Linear autograd Function

```python
class FP8Linear(torch.autograd.Function):
    """FP8 线性层，master weights 保持 bf16，forward/backward 用 FP8 matmul。"""

    @staticmethod
    def forward(ctx, x, weight, x_scale, w_scale):
        # FP8 cast + scaled_mm → BF16 output
        M, K = x.shape
        K2, N = weight.shape
        x_fp8 = x.reshape(-1, K).to(torch.float8_e4m3fn)
        w_fp8_t = weight.t().contiguous().to(torch.float8_e4m3fn)  # (N, K) col-major
        out = torch._scaled_mm(x_fp8, w_fp8_t, x_scale, w_scale,
                               out_dtype=torch.bfloat16)
        ctx.save_for_backward(x, weight, x_scale, w_scale)
        return out.reshape(*x.shape[:-1], N)

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, x_scale, w_scale = ctx.saved_tensors
        M, K = x.reshape(-1, x.size(-1)).shape
        N = weight.size(0)

        # grad_input = grad_out @ W^T  (FP8 matmul)
        go_flat = grad_out.reshape(-1, N)
        go_fp8 = go_flat.to(torch.float8_e4m3fn)
        w_fp8 = weight.to(torch.float8_e4m3fn)  # (N, K) row-major
        go_scale = compute_scale(go_flat)
        grad_x = torch._scaled_mm(go_fp8, w_fp8, go_scale, w_scale,
                                  out_dtype=torch.bfloat16)
        grad_x = grad_x.reshape_as(x)

        # grad_weight = x^T @ grad_out  (FP8 matmul)
        x_flat = x.reshape(-1, K)
        x_fp8_t = x_flat.t().contiguous().to(torch.float8_e4m3fn)  # (K, M) col-major
        go_fp8_t = go_flat.t().contiguous().to(torch.float8_e4m3fn)  # (N, M) col-major
        grad_w = torch._scaled_mm(x_fp8_t, go_fp8_t, x_scale, go_scale,
                                  out_dtype=torch.bfloat16)
        grad_w = grad_w.t()  # → (N, K)

        return grad_x, grad_w, None, None
```

### 延迟缩放 (Delayed Scaling)

每步用上一步的 amax 做 scale，避免同步 barrier:

```python
def compute_scale(t):
    """FP8 e4m3fn max = 448, 计算动态缩放因子。"""
    amax = t.abs().max()
    return torch.clamp(448.0 / amax, max=1e4)  # clamp 防止极小 scale 溢出
```

```python
# 训练循环中:
# Step 0 (warmup): BF16 forward, 记录初始 scale
# Step N: 用 scale[N-1] 做 FP8 matmul，同时计算 scale[N] 供 Step N+1 使用
# 不需要额外的 amax 同步 — 在 FP8 cast 时顺手完成（已有 abs().max()）
```

### 哪些层该用 FP8

| 操作 | 总时间占比 | FP8? | 原因 |
|------|-----------|------|------|
| QKV 投影 | ~18% | ✅ | 大矩阵乘法，2× 加速 |
| Out 投影 | ~6% | ✅ | 同上 |
| FFN ↑ | ~24% | ✅ | 同上 |
| FFN ↓ | ~24% | ✅ | 同上 |
| SDPA | ~15% | ❌ | FlashAttention 内部已高度优化 |
| bias 构建 | ~5% | ❌ | Embedding + 小 op，不适合 FP8 |
| LayerNorm | ~3% | ❌ | 必须在 fp32 |
| Embedding | ~5% | ❌ | 查表操作 |

**预期总加速**: 0.72(线性层) / 2 + 0.28(其余) = 0.36 + 0.28 = 0.64 → **~1.56× 总训练加速**

### FP8 梯度精度策略

- **前向 activation**: e4m3fn (4-bit exponent, 3-bit mantissa) — 精度/范围平衡
- **反向梯度**: e5m2 (5-bit exponent, 2-bit mantissa) — 更大的动态范围防止下溢
  - 或: gradient scaling (mul 32768 before FP8 cast) + e4m3fn
- **Master weights**: bf16 (不变)
- **Optimizer states**: fp32 (不变)

---

## 四、组合方案与预估

| 方案 | 注意力加速 | FP8加速 | 总加速 | 质量风险 | 工程量 | 显存节省 |
|------|-----------|---------|--------|----------|--------|----------|
| A: 仅 ALiBi 斜率 | ~1.15× | — | ~1.15× | 低 | ~50行 | ~20GB |
| B: 仅 FP8 线性层 | — | ~1.55× | ~1.55× | 中 | ~200行 | — |
| **C: ALiBi + FP8** | ~1.15× | ~1.55× | **~1.75×** | 低-中 | ~250行 | ~20GB |
| D: 分桶 + FP8 | ~1.10× | ~1.55× | ~1.70× | 低 | ~250行 | ~18GB |

**推荐路线 C**: ALiBi 斜率 + FP8 线性层。
- 预期: 170K steps 训练从 ~100h → ~57h
- 显存节省 ~20GB (1GB/layer × 20 layers)，可增大 batch 或序列长度

---

## 五、实施路线图

### Phase 1: ALiBi 斜率 (1天)
1. 修改 `CausalSelfAttention._build_total_bias` — 用斜率替换 Embedding
2. 小模型验证 (4层, 5000 steps) — 对比 val loss
3. 确认无损后全量切换

### Phase 2: FP8 线性层 (2-3天)
1. 实现 `FP8Linear` autograd Function + 延迟缩放
2. 替换 `CausalSelfAttention.qkv`, `out_proj` 和 `TransformerBlock.ffn` 中的 nn.Linear
3. 前 100 steps BF16 warmup → 切换 FP8
4. 对比 loss curve + grad norm 分布

### Phase 3: 组合验证 (1天)
1. 完整 20 层模型训练
2. TensorBoard 对比: loss 曲线、grad norm、FP8 scale 稳定性
3. 生成质量 A/B test

---

## 六、风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| FP8 梯度下溢 | 中 | loss 不收敛 | gradient scaling; e5m2 for grads; 前 100 steps BF16 warmup |
| ALiBi 质量退化 | 低 | val loss 高于 baseline | 小模型快速验证; 回退到分桶方案 |
| 延迟 scaling 发散 | 低 | 训练崩溃 | 监控 scale 变化; 异常时 fallback 到同步 scaling |
| _scaled_mm 输入布局敏感 | 中 | 多一次 transpose | 预存 col-major weight clone; benchmark 确认无退化 |

---

## 附录: FP8 Benchmark 脚本

见 `scripts/bench_fp8.py`
