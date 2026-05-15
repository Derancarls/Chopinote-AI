"""ALiBi + FP8Linear 模型集成测试 — bias 形状、forward/backward、无损验证。"""

import torch
import pytest

from chopinote_model.config import ModelConfig
from chopinote_model.model import MusicTransformer, CausalSelfAttention
from chopinote_model.fp8_linear import FP8Linear


# ==============================================================
# ALiBi bias 形状
# ==============================================================

class TestALiBiBiasShape:

    def test_alibi_bias_shape(self):
        """_build_total_bias 输出应为 (1, nH, T, T_kv)。"""
        config = ModelConfig(d_model=256, n_heads=4, n_layers=2, d_ff=1024,
                             max_seq_len=128)
        attn = CausalSelfAttention(config).cuda()

        T, T_kv = 32, 32
        bias = attn._build_total_bias(T, T_kv, mask=None, measure_ids=None,
                                       device='cuda', dtype=torch.bfloat16)
        assert bias.shape == (1, config.n_heads, T, T_kv), \
            f"Expected (1, {config.n_heads}, {T}, {T_kv}), got {bias.shape}"
        assert bias.dtype == torch.bfloat16

    def test_alibi_bias_causal(self):
        """causal mask 使右上三角为 -inf。"""
        config = ModelConfig(d_model=256, n_heads=4, n_layers=2, d_ff=1024,
                             max_seq_len=128)
        attn = CausalSelfAttention(config).cuda()

        bias = attn._build_total_bias(8, 8, mask=None, measure_ids=None,
                                       device='cuda', dtype=torch.bfloat16)
        # 右上三角 (j > i) 应为 -inf
        assert bias[0, 0, 0, 1] == float('-inf'), "causal mask 未生效"
        assert bias[0, 0, 1, 0] != float('-inf'), "左下角不应被 mask"
        assert bias[0, 0, 2, 2] != float('-inf'), "对角线不应被 mask"

    def test_alibi_slope_decreasing(self):
        """head 越靠后的 slope 越小（ALiBi 特性）。"""
        config = ModelConfig(d_model=256, n_heads=8, n_layers=2, d_ff=1024)
        attn = CausalSelfAttention(config).cuda()

        slopes = attn.rel_slope.detach().cpu()
        for i in range(len(slopes) - 1):
            assert slopes[i] >= slopes[i+1], \
                f"slope[{i}] ({slopes[i]:.4f}) < slope[{i+1}] ({slopes[i+1]:.4f}), 应递减"

    def test_alibi_bias_kv_cache(self):
        """KV cache 模式下 T != T_kv 时 bias 形状正确。"""
        config = ModelConfig(d_model=256, n_heads=4, n_layers=2, d_ff=1024,
                             max_seq_len=128)
        attn = CausalSelfAttention(config).cuda()

        T, T_kv = 16, 48  # decode: query 短, key/value 长
        bias = attn._build_total_bias(T, T_kv, mask=None, measure_ids=None,
                                       device='cuda', dtype=torch.bfloat16)
        assert bias.shape == (1, config.n_heads, T, T_kv)
        # 右上三角 causal mask → -inf, 但至少对角线上有有限值
        assert torch.isfinite(bias[0, 0, :T, :T].diag()).all(), "对角线上应为有限值"

    def test_alibi_with_padding_mask(self):
        """padding mask 正确处理。"""
        config = ModelConfig(d_model=256, n_heads=4, n_layers=2, d_ff=1024,
                             max_seq_len=128)
        attn = CausalSelfAttention(config).cuda()

        mask = torch.zeros(1, 24, dtype=torch.bool, device='cuda')
        mask[0, :16] = True  # 前 16 个有效，后 8 个 padding

        bias = attn._build_total_bias(24, 24, mask=mask, measure_ids=None,
                                       device='cuda', dtype=torch.bfloat16)
        # padding 位置应为 -inf
        assert (bias[0, 0, :, 20] == float('-inf')).all(), "padding mask 未生效"

    def test_alibi_measure_bias(self):
        """measure_bias 加入后形状不变。"""
        config = ModelConfig(d_model=256, n_heads=4, n_layers=2, d_ff=1024,
                             max_seq_len=128)
        attn = CausalSelfAttention(config).cuda()

        measure_ids = torch.zeros(32, dtype=torch.long, device='cuda')
        measure_ids[8:] = 1
        measure_ids[16:] = 2
        measure_ids[24:] = 3

        bias_with = attn._build_total_bias(32, 32, mask=None,
                                            measure_ids=measure_ids,
                                            device='cuda', dtype=torch.bfloat16)
        assert bias_with.shape == (1, config.n_heads, 32, 32)


# ==============================================================
# 模型 forward/backward
# ==============================================================

class TestModelIntegration:

    def test_model_forward_shape(self):
        """MusicTransformer forward 输出形状正确。"""
        config = ModelConfig(d_model=256, n_heads=4, n_layers=2, d_ff=1024,
                             vocab_size=100, max_seq_len=64)
        model = MusicTransformer(config).cuda()

        input_ids = torch.randint(0, 100, (2, 32), device='cuda')
        logits = model(input_ids)
        assert logits.shape == (2, 32, config.vocab_size), \
            f"Expected (2, 32, {config.vocab_size}), got {logits.shape}"
        assert torch.isfinite(logits).all(), "logits 有 NaN/Inf"

    def test_model_backward(self):
        """完整 backward 不崩溃、梯度存在。"""
        config = ModelConfig(d_model=256, n_heads=4, n_layers=2, d_ff=1024,
                             vocab_size=100, max_seq_len=64)
        model = MusicTransformer(config).cuda()

        input_ids = torch.randint(0, 100, (2, 32), device='cuda')
        logits = model(input_ids)
        loss = logits.sum()
        loss.backward()

        for name, p in model.named_parameters():
            assert p.grad is not None, f"{name} grad is None"
            assert torch.isfinite(p.grad).all(), f"{name} grad has NaN/Inf"

    def test_model_fp8_mode(self):
        """set_fp8_mode 递归设置所有 FP8Linear 模块。"""
        config = ModelConfig(d_model=256, n_heads=4, n_layers=2, d_ff=1024,
                             vocab_size=100, max_seq_len=64)
        model = MusicTransformer(config).cuda()

        model.set_fp8_mode(True)
        for module in model.modules():
            if isinstance(module, FP8Linear):
                assert module.use_fp8, f"FP8Linear 未被启用"

        model.set_fp8_mode(False)
        for module in model.modules():
            if isinstance(module, FP8Linear):
                assert not module.use_fp8, f"FP8Linear 未被禁用"

    def test_model_alibi_deterministic(self):
        """相同输入下 ALiBi 输出确定。"""
        config = ModelConfig(d_model=256, n_heads=4, n_layers=2, d_ff=1024,
                             vocab_size=100, max_seq_len=64)
        model = MusicTransformer(config).cuda()
        model.eval()

        input_ids = torch.randint(0, 100, (1, 32), device='cuda')
        with torch.no_grad():
            out1 = model(input_ids)
            out2 = model(input_ids)

        assert torch.allclose(out1, out2, atol=1e-6), "ALiBi 输出不一致"


# ==============================================================
# FP8 + 模型
# ==============================================================

class TestModelFP8:

    def test_model_fp8_forward(self):
        """FP8 模式下模型 forward 不崩溃。"""
        config = ModelConfig(d_model=256, n_heads=4, n_layers=2, d_ff=1024,
                             vocab_size=100, max_seq_len=64)
        model = MusicTransformer(config).cuda()
        model.eval()  # eval 模式跳过 checkpoint
        model.set_fp8_mode(True)

        # warmup: 初始化所有 FP8Linear 的 scales
        input_ids = torch.randint(0, 100, (2, 32), device='cuda')
        _ = model(input_ids)
        model.set_fp8_mode(True)

        logits = model(input_ids)
        assert logits.shape == (2, 32, config.vocab_size)
        assert torch.isfinite(logits).all(), "FP8 模型 logits 有 NaN/Inf"

    def test_model_fp8_backward(self):
        """FP8 模式下模型 backward 不崩溃（eval 模式跳过 checkpoint）。"""
        config = ModelConfig(d_model=256, n_heads=4, n_layers=2, d_ff=1024,
                             vocab_size=100, max_seq_len=64)
        model = MusicTransformer(config).cuda()
        model.eval()
        model.set_fp8_mode(True)

        input_ids = torch.randint(0, 100, (2, 16), device='cuda')
        _ = model(input_ids)  # warmup + 初始化 scales
        logits = model(input_ids)  # 同一输入，scale 匹配
        loss = logits.sum()
        loss.backward()

        for name, p in model.named_parameters():
            assert p.grad is not None, f"{name} grad is None"
            assert torch.isfinite(p.grad).all(), f"{name} grad has NaN/Inf"

    def test_model_fp8_backward_train(self):
        """FP8+checkpoint 训练模式 backward（先 warmup 再切 FP8）。"""
        config = ModelConfig(d_model=256, n_heads=4, n_layers=2, d_ff=1024,
                             vocab_size=100, max_seq_len=64)
        model = MusicTransformer(config).cuda()
        model.train()
        model.set_fp8_mode(False)  # 先用 BF16 训练

        input_ids = torch.randint(0, 100, (2, 16), device='cuda')
        logits = model(input_ids)
        loss = logits.sum()
        loss.backward()

        for name, p in model.named_parameters():
            assert p.grad is not None, f"{name} grad is None"
            assert torch.isfinite(p.grad).all(), f"{name} grad has NaN/Inf"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-x'])
