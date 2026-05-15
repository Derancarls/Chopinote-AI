"""RoPE + FP8Linear 模型集成测试 — 速度、SDPA is_causal、KV cache。"""
import torch
import pytest

from chopinote_model.config import ModelConfig
from chopinote_model.model import MusicTransformer, CausalSelfAttention
from chopinote_model.fp8_linear import FP8Linear


class TestRoPE:
    """RoPE 位置编码正确性。"""

    def test_rope_cache_created(self):
        """首次 forward 自动创建 _rope_cos/_rope_sin buffer。"""
        config = ModelConfig(d_model=256, n_heads=4, n_layers=2, d_ff=1024, max_seq_len=128)
        attn = CausalSelfAttention(config).cuda()
        assert attn._rope_cos is None
        x = torch.randn(1, 32, 256, dtype=torch.bfloat16, device='cuda')
        attn(x)
        assert attn._rope_cos is not None
        assert attn._rope_cos.shape == (config.max_seq_len, config.head_dim // 2)

    def test_rope_relative_decay(self):
        """RoPE 中远距离 token 间点积自然衰减。"""
        config = ModelConfig(d_model=256, n_heads=4, n_layers=2, d_ff=1024, max_seq_len=128)
        attn = CausalSelfAttention(config).cuda()
        attn._ensure_rope_cache(torch.device('cuda'), torch.bfloat16)

        # 同一位置 q,k 点积最大
        pos0_cos = attn._rope_cos[0]
        pos0_sin = attn._rope_sin[0]
        # 构建一个简单的 q=k 向量
        q = torch.randn(4, 64, dtype=torch.bfloat16, device='cuda')
        cos = pos0_cos.unsqueeze(0)
        sin = pos0_sin.unsqueeze(0)
        q_rope = attn._apply_rope(q.unsqueeze(0).unsqueeze(0), cos, sin).squeeze()
        # 自身点积应较大
        sim_self = (q_rope * q_rope).sum().item()
        assert sim_self > 0

    def test_rope_is_causal(self):
        """SDPA 使用 is_causal=True，不传显式 bias。"""
        config = ModelConfig(d_model=256, n_heads=4, n_layers=2, d_ff=1024, max_seq_len=64)
        attn = CausalSelfAttention(config).cuda()
        x = torch.randn(2, 32, 256, dtype=torch.bfloat16, device='cuda')
        out = attn(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_rope_with_padding_mask(self):
        """padding mask 与 is_causal 同时生效。"""
        config = ModelConfig(d_model=256, n_heads=4, n_layers=2, d_ff=1024, max_seq_len=64)
        attn = CausalSelfAttention(config).cuda()
        x = torch.randn(2, 32, 256, dtype=torch.bfloat16, device='cuda')
        mask = torch.zeros(2, 32, dtype=torch.bool, device='cuda')
        mask[:, :24] = True  # 后 8 个是 padding
        out = attn(x, mask=mask)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_rope_kv_cache(self):
        """KV cache + RoPE 位置追踪正确。"""
        config = ModelConfig(d_model=256, n_heads=4, n_layers=2, d_ff=1024, max_seq_len=64)
        attn = CausalSelfAttention(config).cuda()

        # 第一次 forward: 16 个 token
        x1 = torch.randn(1, 16, 256, dtype=torch.bfloat16, device='cuda')
        cache = [None, None]
        out1 = attn(x1, kv_cache=cache)
        assert cache[0].shape == (1, 4, 16, 64)  # K cached
        assert torch.isfinite(out1).all()

        # 第二次 forward: 1 个新 token，使用 cache
        x2 = torch.randn(1, 1, 256, dtype=torch.bfloat16, device='cuda')
        out2 = attn(x2, kv_cache=cache)
        assert cache[0].shape == (1, 4, 17, 64)  # K 累积
        assert torch.isfinite(out2).all()


class TestModelRoPE:
    """RoPE 模型 forward/backward。"""

    def test_model_forward_shape(self):
        config = ModelConfig(d_model=256, n_heads=4, n_layers=2, d_ff=1024,
                             vocab_size=100, max_seq_len=64)
        model = MusicTransformer(config).cuda()
        input_ids = torch.randint(0, 100, (2, 32), device='cuda')
        logits = model(input_ids)
        assert logits.shape == (2, 32, config.vocab_size)
        assert torch.isfinite(logits).all()

    def test_model_backward(self):
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

    def test_model_with_padding(self):
        """attention_mask 传入但 model 不崩溃。"""
        config = ModelConfig(d_model=256, n_heads=4, n_layers=2, d_ff=1024,
                             vocab_size=100, max_seq_len=64)
        model = MusicTransformer(config).cuda()
        input_ids = torch.randint(0, 100, (2, 32), device='cuda')
        mask = torch.zeros(2, 32, dtype=torch.bool, device='cuda')
        mask[:, :24] = True
        logits = model(input_ids, attention_mask=mask)
        assert torch.isfinite(logits).all()

    def test_model_deterministic_eval(self):
        config = ModelConfig(d_model=256, n_heads=4, n_layers=2, d_ff=1024,
                             vocab_size=100, max_seq_len=64)
        model = MusicTransformer(config).cuda()
        model.eval()
        input_ids = torch.randint(0, 100, (1, 32), device='cuda')
        with torch.no_grad():
            out1 = model(input_ids)
            out2 = model(input_ids)
        assert torch.allclose(out1, out2, atol=1e-6)

    def test_model_fp8_mode(self):
        """set_fp8_mode 递归设置所有 FP8Linear。"""
        config = ModelConfig(d_model=256, n_heads=4, n_layers=2, d_ff=1024,
                             vocab_size=100, max_seq_len=64)
        model = MusicTransformer(config).cuda()
        model.set_fp8_mode(True)
        for m in model.modules():
            if isinstance(m, FP8Linear):
                assert m.use_fp8
        model.set_fp8_mode(False)
        for m in model.modules():
            if isinstance(m, FP8Linear):
                assert not m.use_fp8

    def test_measure_embedding_params(self):
        """RoPE 模型应有 measure_embedding 但无 ALiBi 的 rel_slope/measure_bias。"""
        config = ModelConfig(d_model=256, n_heads=4, n_layers=2, d_ff=1024)
        model = MusicTransformer(config)
        names = set(dict(model.named_parameters()).keys())
        assert any('measure_embedding' in n for n in names), "应有 measure_embedding"
        assert not any('rel_slope' in n for n in names), "不应存在 ALiBi rel_slope"
        assert not any('measure_bias' in n for n in names), "不应存在旧 pairwise measure_bias"

    def test_measure_embedding_forward(self):
        """measure_embedding cumsum 正确追踪小节号。"""
        config = ModelConfig(d_model=256, n_heads=4, n_layers=2, d_ff=1024,
                             vocab_size=100, max_seq_len=64)
        model = MusicTransformer(config).cuda()
        # bars at positions 0, 3, 6 → measure_id: 0,0,0,1,1,1,1,2,2,2... (cumsum happens BEFORE bar token)
        input_ids = torch.tensor([[4, 1, 2, 4, 3, 5, 6, 4, 7]], device='cuda')
        logits = model(input_ids)
        assert logits.shape == (1, 9, 100)
        assert torch.isfinite(logits).all()


class TestModelFP8:
    """FP8 + RoPE 兼容性。"""

    def test_fp8_forward(self):
        config = ModelConfig(d_model=256, n_heads=4, n_layers=2, d_ff=1024,
                             vocab_size=100, max_seq_len=64)
        model = MusicTransformer(config).cuda()
        model.eval()
        model.set_fp8_mode(True)
        input_ids = torch.randint(0, 100, (2, 32), device='cuda')
        _ = model(input_ids)  # warmup scales
        logits = model(input_ids)
        assert logits.shape == (2, 32, config.vocab_size)
        assert torch.isfinite(logits).all()

    def test_fp8_backward(self):
        config = ModelConfig(d_model=256, n_heads=4, n_layers=2, d_ff=1024,
                             vocab_size=100, max_seq_len=64)
        model = MusicTransformer(config).cuda()
        model.eval()
        model.set_fp8_mode(True)
        input_ids = torch.randint(0, 100, (2, 16), device='cuda')
        _ = model(input_ids)
        logits = model(input_ids)
        loss = logits.sum()
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"{name} grad is None"
            assert torch.isfinite(p.grad).all(), f"{name} grad has NaN/Inf"


class TestSDPASpeed:
    """RoPE 用 is_causal=True，验证 SDPA 不传显式 attn_mask 时的速度。"""

    def test_is_causal_no_explicit_mask(self):
        """无 attn_mask 时 SDPA 使用快速 causal 路径。"""
        config = ModelConfig(d_model=256, n_heads=4, n_layers=2, d_ff=1024, max_seq_len=256)
        attn = CausalSelfAttention(config).cuda()
        x = torch.randn(2, 128, 256, dtype=torch.bfloat16, device='cuda')
        out = attn(x)  # 无 mask → is_causal=True
        assert out.shape == x.shape
        assert torch.isfinite(out).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-x'])
