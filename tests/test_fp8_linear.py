"""FP8Linear 单元测试 — forward/backward 正确性 + BF16 baseline 对比。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from chopinote_model.fp8_linear import FP8Linear, FP8LinearFn, compute_scale


# ==============================================================
# 辅助函数
# ==============================================================

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f, b_f = a.flatten().float(), b.flatten().float()
    return F.cosine_similarity(a_f, b_f, dim=0).item()


# ==============================================================
# compute_scale
# ==============================================================

class TestComputeScale:
    def test_basic(self):
        t = torch.tensor([[1.0, -2.0, 3.0]])
        s = compute_scale(t)
        assert s.item() == pytest.approx(3.0 / 448.0)

    def test_zero_input(self):
        t = torch.zeros(16, 16)
        s = compute_scale(t)
        assert s.item() == pytest.approx(1e-4)

    def test_clamp_min(self):
        t = torch.tensor([1e-6])
        s = compute_scale(t)
        assert s.item() == pytest.approx(1e-4)

    def test_bfloat16_input(self):
        t = torch.tensor([[1.0, -2.0, 3.0]], dtype=torch.bfloat16)
        s = compute_scale(t)
        assert s.dtype == torch.float32


# ==============================================================
# FP8LinearFn autograd Function
# ==============================================================

class TestFP8LinearFn:

    def _make_inputs(self, M=2048, K=2048, N=2048):
        x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
        w = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
        bias = torch.randn(N, dtype=torch.float32, device='cuda')
        sx = compute_scale(x)
        sw = compute_scale(w)
        w_fp8_fwd = (w.T.float() / sw).to(torch.float8_e4m3fn)
        w_fp8_bwd = (w.T.contiguous().T.float() / sw).to(torch.float8_e4m3fn)
        return x, w, w_fp8_fwd, w_fp8_bwd, bias, sx, sw

    def test_forward_shape(self):
        x, w, w_fwd, w_bwd, bias, sx, sw = self._make_inputs(M=64, K=128, N=256)
        out = FP8LinearFn.apply(x, w_fwd, w_bwd, w, bias, sx, sw)
        assert out.shape == (64, 256), f"Expected (64, 256), got {out.shape}"
        assert out.dtype == torch.bfloat16

    def test_forward_no_bias(self):
        x, w, w_fwd, w_bwd, _, sx, sw = self._make_inputs(M=64, K=128, N=256)
        out = FP8LinearFn.apply(x, w_fwd, w_bwd, w, None, sx, sw)
        assert out.shape == (64, 256)

    @pytest.mark.parametrize("M", [1, 16, 2048])
    @pytest.mark.parametrize("K", [128, 2048])
    @pytest.mark.parametrize("N", [128, 2048])
    def test_forward_various_shapes(self, M, K, N):
        x, w, w_fwd, w_bwd, bias, sx, sw = self._make_inputs(M=M, K=K, N=N)
        out = FP8LinearFn.apply(x, w_fwd, w_bwd, w, bias, sx, sw)
        assert out.shape == (M, N)

    def test_backward_gradients_exist(self):
        x, w, w_fwd, w_bwd, bias, sx, sw = self._make_inputs(M=64, K=128, N=32)
        x.requires_grad = True
        w.requires_grad = True
        bias.requires_grad = True

        out = FP8LinearFn.apply(x, w_fwd, w_bwd, w, bias, sx, sw)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "grad_x is None"
        assert w.grad is not None, "grad_w is None"
        assert bias.grad is not None, "grad_bias is None"
        assert x.grad.shape == x.shape
        assert w.grad.shape == w.shape
        assert bias.grad.shape == bias.shape

    def test_backward_grad_shape(self):
        x, w, w_fwd, w_bwd, bias, sx, sw = self._make_inputs(M=16, K=64, N=32)
        x.requires_grad = True
        w.requires_grad = True

        out = FP8LinearFn.apply(x, w_fwd, w_bwd, w, bias, sx, sw)
        loss = out.sum()
        loss.backward()

        assert x.grad.shape == (16, 64), f"Expected (16,64), got {x.grad.shape}"
        assert w.grad.shape == (32, 64), f"Expected (32,64), got {w.grad.shape}"

    def test_backward_bf16_dtype(self):
        x, w, w_fwd, w_bwd, bias, sx, sw = self._make_inputs(M=16, K=64, N=32)
        x.requires_grad = True
        w.requires_grad = True

        out = FP8LinearFn.apply(x, w_fwd, w_bwd, w, bias, sx, sw)
        loss = out.sum()
        loss.backward()

        assert x.grad.dtype == torch.bfloat16, f"Expected bfloat16, got {x.grad.dtype}"
        assert w.grad.dtype == torch.bfloat16

    def test_backward_no_bias(self):
        x, w, w_fwd, w_bwd, _, sx, sw = self._make_inputs(M=16, K=64, N=32)
        x.requires_grad = True
        w.requires_grad = True

        out = FP8LinearFn.apply(x, w_fwd, w_bwd, w, None, sx, sw)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert w.grad is not None


# ==============================================================
# FP8Linear nn.Module
# ==============================================================

class TestFP8LinearModule:

    def test_init(self):
        layer = FP8Linear(128, 256)
        assert layer.weight.shape == (256, 128)
        assert layer.bias is not None
        assert layer.bias.shape == (256,)
        assert layer.bias.dtype == torch.float32
        assert layer.use_fp8 is False

    def test_init_no_bias(self):
        layer = FP8Linear(128, 256, bias=False)
        assert layer.bias is None

    def test_forward_bf16_fallback(self):
        layer = FP8Linear(64, 32).cuda()
        layer.use_fp8 = False
        x = torch.randn(16, 64, dtype=torch.bfloat16, device='cuda')
        out = layer(x)
        assert out.shape == (16, 32)
        assert out.dtype == torch.bfloat16

    def test_forward_bf16_vs_nn_linear(self):
        """BF16 fallback path 应等效于 nn.Linear."""
        fp8 = FP8Linear(64, 32).cuda()
        fp8.use_fp8 = False

        ref = nn.Linear(64, 32, bias=True).cuda().to(torch.bfloat16)
        # 复制权重确保数值一致
        ref.weight.data.copy_(fp8.weight.data)
        ref.bias.data.copy_(fp8.bias.data.to(torch.bfloat16))

        x = torch.randn(16, 64, dtype=torch.bfloat16, device='cuda')
        out_fp8 = fp8(x)
        out_ref = ref(x)

        assert torch.allclose(out_fp8, out_ref, atol=1e-6), \
            f"BF16 fallback mismatch! max diff: {(out_fp8 - out_ref).abs().max().item()}"

    def test_forward_fp8_basic(self):
        layer = FP8Linear(64, 32).cuda()
        layer.use_fp8 = True
        x = torch.randn(16, 64, dtype=torch.bfloat16, device='cuda')
        out = layer(x)
        assert out.shape == (16, 32)
        assert out.dtype == torch.bfloat16
        assert torch.isfinite(out).all(), "FP8 forward 输出包含 NaN 或 Inf"

    def test_forward_fp8_various_shapes(self):
        for M, K, N in [(1, 128, 64), (64, 128, 256), (2048, 2048, 2048)]:
            layer = FP8Linear(K, N).cuda()
            layer.use_fp8 = True
            x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
            out = layer(x)
            assert out.shape == (M, N), f"({M},{K},{N}): expected ({M},{N}), got {out.shape}"
            assert torch.isfinite(out).all()

    def test_fp8_forward_consistency(self):
        """多次 FP8 forward 应稳定（不 NaN）。"""
        layer = FP8Linear(128, 128).cuda()
        layer.use_fp8 = True
        for _ in range(10):
            x = torch.randn(32, 128, dtype=torch.bfloat16, device='cuda')
            layer._update_scales(x)
            sw = compute_scale(layer.weight)
            w_fwd = (layer.weight.T.float() / sw).to(torch.float8_e4m3fn)
            w_bwd = (layer.weight.T.contiguous().T.float() / sw).to(torch.float8_e4m3fn)
            out = FP8LinearFn.apply(x, w_fwd, w_bwd, layer.weight, layer.bias,
                                     layer._scale_x, sw)
            assert torch.isfinite(out).all(), f"NaN at iteration {_}"

    def test_backward_fp8(self):
        layer = FP8Linear(128, 64).cuda()
        layer.use_fp8 = True
        x = torch.randn(16, 128, dtype=torch.bfloat16, device='cuda')
        x.requires_grad = True

        out = layer(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert layer.weight.grad is not None
        assert x.grad.shape == x.shape
        assert layer.weight.grad.shape == layer.weight.shape
        assert torch.isfinite(x.grad).all(), "grad_x 包含 NaN/Inf"
        assert torch.isfinite(layer.weight.grad).all(), "grad_w 包含 NaN/Inf"

    def test_fp8_vs_bf16_numerical(self):
        """FP8 forward 与 BF16 baseline 的 cosine similarity。"""
        layer_fp8 = FP8Linear(256, 256).cuda()
        layer_fp8.use_fp8 = True

        layer_bf16 = nn.Linear(256, 256, bias=True).cuda().to(torch.bfloat16)
        # 复制权重使 FP8/BF16 参数完全一致
        layer_bf16.weight.data.copy_(layer_fp8.weight.data)
        layer_bf16.bias.data.copy_(layer_fp8.bias.data.to(torch.bfloat16))

        x = torch.randn(64, 256, dtype=torch.bfloat16, device='cuda')
        # warmup: 用同一个 x 初始化 scales，使 FP8 路径使用正确的 scales
        layer_fp8._update_scales(x)
        sw = compute_scale(layer_fp8.weight)
        w_fwd = (layer_fp8.weight.T.float() / sw).to(torch.float8_e4m3fn)
        w_bwd = (layer_fp8.weight.T.contiguous().T.float() / sw).to(torch.float8_e4m3fn)
        layer_fp8.use_fp8 = True
        out_fp8 = FP8LinearFn.apply(x, w_fwd, w_bwd, layer_fp8.weight, layer_fp8.bias,
                                     layer_fp8._scale_x, sw)
        out_bf16 = layer_bf16(x)

        sim = cosine_sim(out_fp8, out_bf16)
        # FP8 e4m3fn 有 3-bit mantissa，预期 cos > 0.99
        assert sim > 0.99, f"cosine similarity {sim:.6f} < 0.99"

    def test_fp8_backward_cosine_sim(self):
        """FP8 backward 梯度与 BF16 baseline 的 cosine similarity。"""
        layer_fp8 = FP8Linear(256, 128).cuda()
        layer_fp8.use_fp8 = True

        layer_bf16 = nn.Linear(256, 128, bias=True).cuda().to(torch.bfloat16)
        layer_bf16.weight.data.copy_(layer_fp8.weight.data)
        layer_bf16.bias.data.copy_(layer_fp8.bias.data.to(torch.bfloat16))

        x = torch.randn(64, 256, dtype=torch.bfloat16, device='cuda', requires_grad=True)

        # FP8 forward/backward with fresh scales
        layer_fp8._update_scales(x.detach())
        sw = compute_scale(layer_fp8.weight)
        w_fwd = (layer_fp8.weight.T.float() / sw).to(torch.float8_e4m3fn)
        w_bwd = (layer_fp8.weight.T.contiguous().T.float() / sw).to(torch.float8_e4m3fn)
        out_fp8 = FP8LinearFn.apply(x, w_fwd, w_bwd, layer_fp8.weight, layer_fp8.bias,
                                     layer_fp8._scale_x, sw)
        out_fp8.sum().backward()
        grad_x_fp8 = x.grad.clone()
        grad_w_fp8 = layer_fp8.weight.grad.clone()

        # Reset
        x.grad = None
        layer_fp8.weight.grad = None

        x_bf16 = x.detach().clone().requires_grad_(True)
        out_bf16 = layer_bf16(x_bf16)
        out_bf16.sum().backward()

        sim_x = cosine_sim(grad_x_fp8, x_bf16.grad)
        sim_w = cosine_sim(grad_w_fp8, layer_bf16.weight.grad)

        assert sim_x > 0.98, f"grad_x cosine sim {sim_x:.6f} < 0.98"
        assert sim_w > 0.98, f"grad_w cosine sim {sim_w:.6f} < 0.98"

    def test_fp8_mode_toggle(self):
        layer = FP8Linear(64, 32).cuda()
        x = torch.randn(16, 64, dtype=torch.bfloat16, device='cuda')

        layer.use_fp8 = False
        out_bf16 = layer(x)

        layer.use_fp8 = True
        out_fp8 = layer(x)

        assert out_bf16.shape == out_fp8.shape
        assert torch.isfinite(out_fp8).all()

    def test_state_dict_compatible_with_linear(self):
        """FP8Linear state_dict key 应与 nn.Linear 一致。"""
        fp8 = FP8Linear(64, 32)
        keys = set(fp8.state_dict().keys())
        assert keys == {'weight', 'bias'}, f"Unexpected keys: {keys}"
        assert fp8.state_dict()['weight'].shape == (32, 64)

    def test_multi_layer_backward(self):
        """多个 FP8Linear 堆叠后的反向传播不崩溃。"""
        dims = [128, 256, 128, 64]
        layers = [FP8Linear(dims[i], dims[i+1]).cuda() for i in range(len(dims)-1)]
        for l in layers:
            l.use_fp8 = True

        x = torch.randn(16, dims[0], dtype=torch.bfloat16, device='cuda', requires_grad=True)
        for l in layers:
            x = l(x)
        x.sum().backward()

        for i, l in enumerate(layers):
            assert l.weight.grad is not None, f"Layer {i} grad is None"
            assert torch.isfinite(l.weight.grad).all(), f"Layer {i} grad has NaN/Inf"

    def test_fp8_forward_large_batch(self):
        """大 batch 不 OOM。"""
        layer = FP8Linear(2048, 2048).cuda()
        layer.use_fp8 = True
        x = torch.randn(4096, 2048, dtype=torch.bfloat16, device='cuda')
        out = layer(x)
        assert out.shape == (4096, 2048)
        assert torch.isfinite(out).all()


# ==============================================================
# 维度对齐约束 — FP8 _scaled_mm 要求 dims 为 16 的倍数
# ==============================================================

class TestFP8DimensionAlignment:

    @pytest.mark.parametrize("K,N", [
        (128, 64),   # 标准对齐
        (128, 63),   # N 不是 16 倍数
        (127, 64),   # K 不是 16 倍数
        (127, 63),   # 都不是
        (16, 16),    # 最小对齐
        (2048, 872), # vocab_size 不是 16 倍数 (lm_head 保持 BF16)
    ])
    def test_non_aligned_dims(self, K, N):
        """非 16 倍数维度应正常工作（_scaled_mm 内部会 pad）。"""
        layer = FP8Linear(K, N).cuda()
        layer.use_fp8 = True
        x = torch.randn(4, K, dtype=torch.bfloat16, device='cuda')
        out = layer(x)
        assert out.shape == (4, N), f"Mismatch for K={K}, N={N}"
        assert torch.isfinite(out).all()

    def test_backward_non_aligned_M(self):
        """M (B*T) 不是 16 倍数时 backward 正常。"""
        layer = FP8Linear(256, 128).cuda()
        layer.use_fp8 = True
        M = 1022  # e.g. B=2, T=511
        x = torch.randn(M, 256, dtype=torch.bfloat16, device='cuda', requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None and x.grad.shape == x.shape
        assert layer.weight.grad is not None and layer.weight.grad.shape == layer.weight.shape
        assert torch.isfinite(x.grad).all()
        assert torch.isfinite(layer.weight.grad).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-x'])
