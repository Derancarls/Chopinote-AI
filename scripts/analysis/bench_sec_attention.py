"""Benchmark: 段落感知手动 attention vs 标准 SDPA 快速路径。

测量带 sec_bias 时的性能损失，为训练 batch_size 调整提供参考。
"""
import time
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

# cuDNN > Flash > Efficient
_SDPA_BACKENDS = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
try:
    _SDPA_BACKENDS.insert(0, SDPBackend.CUDNN_ATTENTION)
except AttributeError:
    pass

_NEG_INF = float('-inf')


def manual_attention(q, k, v, sec_bias, causal=True, mask=None, dropout_p=0.0):
    """手动 attention 路径（与 CausalSelfAttention.forward 一致）。"""
    B, nH, T, hd = q.shape
    T_kv = k.size(2)
    scale = hd ** -0.5
    attn = (q @ k.transpose(-2, -1)) * scale

    if sec_bias is not None:
        attn = attn + sec_bias

    if causal:
        causal_mask = torch.triu(
            torch.full((T, T_kv), _NEG_INF, device=q.device, dtype=attn.dtype),
            diagonal=T_kv - T + 1)
        attn = attn + causal_mask[None, None, :, :]

    if mask is not None:
        m = mask[0] if mask.dim() == 2 else mask
        pad = torch.where(m.bool(), 0.0, _NEG_INF).to(dtype=attn.dtype)
        attn = attn + pad.view(1, 1, 1, -1)

    attn = F.softmax(attn, dim=-1)
    if dropout_p > 0:
        attn = F.dropout(attn, p=dropout_p)
    return attn @ v


def sdpa_attention(q, k, v, causal=True, mask=None, dropout_p=0.0):
    """标准 SDPA 快速路径。"""
    if mask is not None and causal:
        m = mask[0] if mask.dim() == 2 else mask
        pad = torch.where(m.bool(), 0.0, _NEG_INF).to(dtype=q.dtype)
        if pad.size(0) < k.size(2):
            pad = F.pad(pad, (0, k.size(2) - pad.size(0)), value=_NEG_INF)
        attn_mask = pad.view(1, 1, 1, -1)
    else:
        attn_mask = None

    with sdpa_kernel(_SDPA_BACKENDS):
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=causal)


def build_fake_sec_bias(B, nH, T):
    """构造模拟段落偏置（包含 α/β/γ/δ 四源的实际 scale）。"""
    dtype = torch.bfloat16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    bias = torch.zeros(B, 1, T, T, device=device, dtype=dtype)
    # 模拟段落结构：每 128 token 一段
    n_sections = max(1, T // 128)
    for s in range(n_sections):
        lo = s * 128
        hi = min((s + 1) * 128, T)
        bias[..., lo:hi, lo:hi] += 0.5  # α: 同实例
        if s + 1 < n_sections:
            bias[..., lo:hi, hi:hi + 128] -= 0.05  # γ: 跨类型
            bias[..., lo:hi, hi:hi + 128] += 0.2  # δ: 边界

    return bias


def bench_forward(q, k, v, sec_bias, mask, causal, fn, name, warmup=5, repeat=20):
    """运行 benchmark。"""
    # warmup
    for _ in range(warmup):
        fn(q, k, v, sec_bias, causal=causal, mask=mask)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeat):
        fn(q, k, v, sec_bias, causal=causal, mask=mask)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / repeat * 1000  # ms

    # VRAM estimate
    if torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated() / 1024**3
        torch.cuda.reset_peak_memory_stats()
    else:
        mem = 0.0

    print(f'  {name:30s}  {elapsed:8.2f} ms  (VRAM peak: {mem:.2f} GiB)')
    return elapsed, mem


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16

    configs = [
        # (B, nH, T, head_dim)  — 典型训练/推理配置
        (1, 32, 512, 64),       # 短序列推理
        (1, 32, 2048, 64),      # 中序列推理
        (1, 32, 4096, 64),      # 满序列 prefill
        (8, 32, 2048, 64),      # 训练 batch=8
    ]

    print('=' * 70)
    print('段落感知 Attention 性能对比 (bf16)')
    print(f'设备: {device}')
    print('=' * 70)

    for B, nH, T, hd in configs:
        print(f'\n--- B={B}, nH={nH}, T={T}, head_dim={hd} ---')

        q = torch.randn(B, nH, T, hd, device=device, dtype=dtype)
        k = torch.randn(B, nH, T, hd, device=device, dtype=dtype)
        v = torch.randn(B, nH, T, hd, device=device, dtype=dtype)
        sec_bias = build_fake_sec_bias(B, nH, T)
        mask = torch.ones(B, T, device=device)

        # Manual with sec_bias
        t_manual, m_manual = bench_forward(
            q, k, v, sec_bias, mask, True,
            lambda q, k, v, sb, **kw: manual_attention(q, k, v, sb, **kw),
            'Manual (sec_bias)')

        # SDPA without sec_bias
        t_sdpa, m_sdpa = bench_forward(
            q, k, v, None, mask, True,
            lambda q, k, v, sb, **kw: sdpa_attention(q, k, v, **kw),
            'SDPA (no sec_bias)')

        # SDPA without sec_bias, no mask
        t_sdpa_nomask, _ = bench_forward(
            q, k, v, None, None, True,
            lambda q, k, v, sb, **kw: sdpa_attention(q, k, v, mask=None, causal=kw.get('causal', True)),
            'SDPA (is_causal only)')

        # Print comparison
        if t_sdpa > 0:
            ratio = t_manual / t_sdpa
            print(f'  {"":30s}  {"":>8s}  Manual/SDPA = {ratio:.1f}x')
        if t_sdpa_nomask > 0:
            ratio2 = t_manual / t_sdpa_nomask
            print(f'  {"":30s}  {"":>8s}  Manual/SDPA(causal_only) = {ratio2:.1f}x')

    print('\n' + '=' * 70)
    print('总结: 对比 Manual (sec_bias) vs SDPA 快速路径的性能差异')
    print('如果 Manual/SDPA > 2x，训练时 batch_size 可能需要下调。')


if __name__ == '__main__':
    main()
