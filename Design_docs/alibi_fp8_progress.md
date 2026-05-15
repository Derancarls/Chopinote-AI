# ALiBi + FP8 混合精度实现进度

日期: 2026-05-14
状态: **进行中** — 核心代码已完成，测试正在验证

---

## 已完成

### Step 1: ALiBi 斜率 ✅
`chopinote_model/model.py`:
- `rel_bias Embedding(8191, 32)` → `rel_slope Parameter(32)` (标量斜率)
- `measure_bias` 保留为 `nn.Embedding`（乐句结构非线性感知）
- `_build_total_bias` 改用 ALiBi: `bias = -slope[h] * |i-j|`
- 返回 shape `(1, nH, T, T_kv)` 与 SDPA 兼容

### Step 2: FP8Linear 模块 ✅
`chopinote_model/fp8_linear.py` — 新建文件:
- `compute_scale(t)` — 动态缩放 `scale = amax / 448`
- `FP8LinearFn(torch.autograd.Function)` — forward/backward 用 `_scaled_mm`
  - **关键**: `.T` 不调用 `.contiguous()` 保持 column-major strides
  - forward: `(x / scale).to(fp8)` @ `(w^T / scale).to(fp8)` → bf16
  - backward: 两个 matmul 同样 FP8 量化 → bf16
- `FP8Linear(nn.Module)` — `use_fp8` flag + 延迟缩放
  - fallback: `F.linear(x.to(bf16), weight, bias)` 等效 nn.Linear
  - FP8 路径: `FP8LinearFn.apply` 用上一拍的 scales

### Step 3: 模型集成 ✅
`chopinote_model/model.py`:
- `self.qkv` → FP8Linear(2048, 6144, bias=False)
- `self.out_proj` → FP8Linear(2048, 2048, bias=False)
- `self.ffn[0]` → FP8Linear(2048, 8192, bias=True)
- `self.ffn[2]` → FP8Linear(8192, 2048, bias=True)
- `self.lm_head` → 保持 nn.Linear (vocab=872 太小 + weight tied)
- 新增 `MusicTransformer.set_fp8_mode(bool)` — 递归设置所有 FP8Linear

### Step 4: 配置 ✅
`chopinote_model/config.py`:
- `TrainingConfig.use_fp8: bool = False`
- `TrainingConfig.fp8_warmup_steps: int = 100`

### Step 5: 训练集成 ✅
`chopinote_model/train.py`:
- `_run_training_loop`: warmup 前 `set_fp8_mode(False)`
- warmup 步数到达后 `set_fp8_mode(True)` + 日志
- TensorBoard: `train/fp8_scale_x`, `train/fp8_scale_w` (平均 scale 监控)

### Step 6: 导出 ✅
`chopinote_model/__init__.py`: 导出 `FP8Linear`

---

## 待完成

### 测试验证 (已完成 ✅)
- [x] ALiBi bias shape 测试通过 — (1, nH, T, T_kv) bf16
- [x] 模型 forward/backward 测试通过
- [x] **FP8Linear 测试** — 42 个单元测试全部通过:
  - backward column-major 修复: `weight.t().contiguous().t()` 代替 `weight.T`
  - bias dtype: `out + bias.to(torch.bfloat16)` 防止 float32 提升
  - FP8 backward gradient cosine similarity > 0.98 vs BF16 baseline
  - 维度对齐: 非 16 倍数正常 (K=127, N=63)
  - 多 FP8Linear 堆叠 backward 稳定
  - 大 batch (4096×2048) 不 OOM
- [x] **ALiBi + 模型集成** — 13 个测试全部通过:
  - bias 形状 (1, nH, T, T_kv), causal mask, padding mask, KV cache
  - measure_bias 保留 Embedding, ALiBi slope 递减
  - FP8 模式完整 forward/backward

### Step 7: FP8 训练稳定性修复 ✅
`chopinote_model/fp8_linear.py`:
- **Fresh scales**: 每步计算当前 scale 并立即使用（不再延迟），消除 dropout 导致 scale 不匹配 → FP8 溢出 NaN
- **M 维填充**: backward `grad_weight` 的 `_scaled_mm` 要求尾维被 16 整除，M=B*T 不满足时自动 pad

### 小模型训练验证 ✅
- 4 层模型, 100 步 FP8 训练 (dropout=0.15, seq_len=512)
- 无 NaN, 无梯度爆炸, loss 稳定下降 (6.74→6.70)
- 每步 0.05s (含 backward + optimizer)

---
### 下一步
1. ~~跑通 FP8Linear 完整 forward+backward 测试~~ ✅
2. ~~与 BF16 baseline 对比数值精度 (cosine similarity)~~ ✅
3. ~~小模型 (4 层) 训练 100 steps 验证 loss 不 NaN~~ ✅
4. ~~**全量 20 层模型训练验证**~~ ✅
   - 1.01B 参数, 50 步 FP8 训练: 无 NaN, loss 7.19→6.79, VRAM 10.1GB
5. 生产环境启用: 修改 `train.py` warmup 步数后正式启动全量训练

---

## 关键技术要点

- **ALiBi**: 只换 rel_bias，保留 measure_bias Embedding
- **FP8 量化**: `scale = amax / 448` → `x_fp8 = (x / scale).to(fp8)` → `_scaled_mm` 用 scale 反量化
- **Fresh scale**: 每步用当前 scale 立即量化，消除 dropout 分布偏移 → NaN
- **Column-major**: `weight.T` 不 contiguous → strides (1, K) 即 column-major → `_scaled_mm` 接受
- **State dict 兼容**: FP8Linear 的 weight/bias key 与 nn.Linear 完全一致，checkpoint 无痛
- **Gradient checkpoint**: FP8 时用 `use_reentrant=True`（`use_reentrant=False` 追踪中间张量 metadata，与 FP8 内部 `.to(fp8)` dtype 转换冲突）

### Step 8: Checkpoint 兼容修复 ✅
- BF16 模式: `use_reentrant=False`（无 FP8，正常追踪）
- FP8 模式: `use_reentrant=True`（跳过 metadata 验证，兼容自定义 autograd Function）
