# 模型缩放方案设计记录

## 背景

当前模型 ~206M 参数（d_model=1024, 12层, 16头, d_ff=4096），数据量 ~16B tokens，tokens/参数 ≈ 78，严重过拟合。需要根据硬件条件放大模型。

## 数据量

- MIDI tokens: ~15.8B (158亿)
- PDMX/MusicXML: 尚未完全量化，预计总量 ~16-20B
- 全局 tokens/参数 目标：接近 Chinchilla 最优（~20 tokens/param）

## Tier 方案总览

| 配置 | 当前 | Tier 1 | Tier 2 | Tier 3 |
|------|------|--------|--------|--------|
| d_model | 1024 | 2048 | 2560 | 3072 |
| n_layers | 12 | 32 | 32 | 40 |
| n_heads | 16 | 32 | 32 | 48 |
| d_ff | 4096 | 8192 | 10240 | 12288 |
| **参数量** | **206M** | **2.16B** | **3.37B** | **6.06B** |
| tokens/参数 | 78 | 7.5 | 4.8 | 2.7 |
| 模型+优化器 VRAM | ~2.5GB | ~26GB | ~40GB | ~73GB |
| 需要显存 | 24GB ✅ | **48GB** ✅ | **48GB 极限** ⚠️ | **96GB** ✅ |
| 适配显卡 | RTX 4090 | RTX PRO 6000 Ada | RTX PRO 6000 Ada | RTX PRO 6000 Blackwell |

## 针对 RTX 5090（32GB）的方案

RTX 5090 为 Blackwell 架构，32GB VRAM，混合精度训练下每参数 ~12 字节占用。

| 方案 | d_model | layers | heads | d_ff | 参数量 | 预估VRAM | tokens/参数 |
|------|---------|--------|-------|------|--------|----------|------------|
| 保守 | 1536 | 24 | 24 | 6144 | 0.91B | ~16GB | 17.4 |
| **均衡 (推荐)** | **1792** | **24** | **28** | **7168** | **1.24B** | **~20GB** | **12.8** |
| 激进 | 2048 | 24 | 32 | 8192 | **1.22B** (实测) | ~21GB | 13.0 |

### 推荐理由

均衡方案：
- 留 ~12GB 余量给 batch_size=4~8 的激活值和 CUDA 开销
- 12.8 tokens/param，在单卡 32GB 上接近合理利用
- 训练时长：Phase 1 (MIDI 16B tok) 约 **7~9 天**（按 RTX 5090 ~150-200 TFLOPS 持续吞吐估算）
- Phase 2 视 PDMX/MusicXML 数据量再加 ~30-50%

### 同步调整的超参数

| 参数 | 当前值 | 推荐值 |
|------|--------|--------|
| lr | 2e-4 | 1.5e-4 |
| warmup_steps | 2000 | 4000 |
| dropout | 0.1 | 0.15 |
| weight_decay | 0.1 | 不变 |

## 硬件对比

| 显卡 | 显存 | 推荐模型 | 训练时长 (Phase 1) |
|------|------|---------|-------------------|
| RTX 4090 | 24GB | 206M (当前) | 已跑完 |
| RTX 5090 (Blackwell) | 32GB | **1.24B** | ~7-9 天 |
| RTX PRO 6000 Ada | 48GB | 2.16B (Tier 1) | ~12-16 天 |
| RTX PRO 6000 Blackwell | 96GB | 6.06B (Tier 3) | ~25-38 天 |

## Phase 1 训练参数参考（1.24B 方案）

- batch_size: 4
- grad_accum_steps: 8 (effective batch = 32)
- seq_len: 4096
- mixed_precision: fp16 (或 fp8 如果支持)
- total_steps (Phase 1): ~120k (16B tokens / 131k tokens-per-step)
- Total_steps (Phase 2): ~30-50k
