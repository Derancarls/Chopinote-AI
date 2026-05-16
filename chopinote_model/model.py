"""Decoder-only Transformer (GPT 风格) 用于音乐生成，RoPE 位置编码。"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

from .config import ModelConfig
from .fp8_linear import FP8Linear

_NEG_INF = float('-inf')

# cuDNN attention (Blackwell 优化) 优先于 flash/efficient
_SDPA_BACKENDS = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
try:
    _SDPA_BACKENDS.insert(0, SDPBackend.CUDNN_ATTENTION)
except AttributeError:
    pass


class CausalSelfAttention(nn.Module):
    """单层因果自注意力 + RoPE (支持 KV cache)。"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0

        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model
        self.max_len = config.max_seq_len

        self.qkv = FP8Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = FP8Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        # RoPE cache
        self.register_buffer('_rope_cos', None, persistent=False)
        self.register_buffer('_rope_sin', None, persistent=False)

    def _ensure_rope_cache(self, device: torch.device, dtype: torch.dtype):
        if self._rope_cos is not None and self._rope_cos.device == device:
            return
        theta = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2, device=device).float() / self.head_dim))
        pos = torch.arange(self.max_len, device=device).float()
        freqs = torch.outer(pos, theta)
        self._rope_cos = freqs.cos().to(dtype)
        self._rope_sin = freqs.sin().to(dtype)

    @staticmethod
    def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """对 (B, nH, T, head_dim) 施加 RoPE 旋转 — torch.compile 友好版本。

        将 even/odd 对 reshape 为 (-1, 2)，用 stack+flatten 替代
        empty_like + strided assignment，编译器可直接融合为单一 kernel。
        """
        T = x.shape[2]
        x_pairs = x.reshape(*x.shape[:-1], -1, 2)          # (B, nH, T, head_dim/2, 2)
        x_even, x_odd = x_pairs.unbind(-1)                  # each (B, nH, T, head_dim/2)
        c = cos[:T].unsqueeze(0).unsqueeze(0)
        s = sin[:T].unsqueeze(0).unsqueeze(0)
        r_even = x_even * c - x_odd * s
        r_odd = x_even * s + x_odd * c
        return torch.stack([r_even, r_odd], dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                kv_cache: Optional[list] = None) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        self._ensure_rope_cache(x.device, q.dtype)

        if kv_cache is not None and kv_cache[0] is not None:
            cache_len = kv_cache[0].size(2)
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        else:
            cache_len = 0

        T_kv = k.size(2)

        q = self._apply_rope(q, self._rope_cos[cache_len:cache_len + T],
                             self._rope_sin[cache_len:cache_len + T])
        k = self._apply_rope(k, self._rope_cos[:T_kv], self._rope_sin[:T_kv])

        if kv_cache is not None:
            kv_cache[0] = k
            kv_cache[1] = v

        use_causal = kv_cache is None or kv_cache[0] is None or cache_len == 0
        if mask is not None:
            m = mask[0] if mask.dim() == 2 else mask
            pad = torch.where(m.bool(), 0.0, _NEG_INF).to(dtype=q.dtype)
            if pad.size(0) < T_kv:
                pad = F.pad(pad, (0, T_kv - pad.size(0)), value=_NEG_INF)
            attn_mask = pad.view(1, 1, 1, -1)
        else:
            attn_mask = None

        with sdpa_kernel(_SDPA_BACKENDS):
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=use_causal,
            )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block: Attn → FFN 各带残差。"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            FP8Linear(config.d_model, config.d_ff),
            nn.GELU(),
            FP8Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )
        self.use_checkpointing = config.gradient_checkpointing

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                kv_cache: Optional[list] = None) -> torch.Tensor:
        need_ckpt = self.use_checkpointing and kv_cache is None and self.training
        if need_ckpt:
            _reentrant = self.attn.qkv.use_fp8
            x = torch.utils.checkpoint.checkpoint(
                self._forward, x, mask, None, use_reentrant=_reentrant)
        else:
            x = self._forward(x, mask, kv_cache)
        return x

    def _forward(self, x: torch.Tensor, mask: Optional[torch.Tensor],
                 kv_cache: Optional[list] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask, kv_cache)
        x = x + self.ffn(self.ln2(x))
        return x


class MusicTransformer(nn.Module):
    """Decoder-only Transformer 用于音乐 token 序列生成，RoPE 位置编码。"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.measure_embedding = nn.Embedding(config.max_measures + 1, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.token_embedding.weight = self.lm_head.weight
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def set_fp8_mode(self, enabled: bool):
        for module in self.modules():
            if isinstance(module, FP8Linear):
                module.use_fp8 = enabled

    def invalidate_fp8_caches(self):
        """optimizer.step() 后调用，使所有 FP8 权重量化缓存失效。"""
        for module in self.modules():
            if isinstance(module, FP8Linear):
                module.invalidate_cache()

    def set_gradient_checkpointing(self, enabled: bool):
        """运行时开关 gradient checkpointing。"""
        for block in self.blocks:
            block.use_checkpointing = enabled

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                kv_caches: Optional[list] = None,
                measure_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, \
            f'序列长度 {T} 超过 max_seq_len {self.config.max_seq_len}'

        x = self.token_embedding(input_ids)
        if measure_ids is None:
            bar_mask = (input_ids == self.config.bar_token_id).int()
            measure_ids = torch.cumsum(bar_mask, dim=1).clamp(0, self.config.max_measures)
        elif measure_ids.ndim == 1:
            measure_ids = measure_ids.unsqueeze(0)
        # KV cache 下 measure_ids 可能比 input_ids 长，截取尾部
        if measure_ids.size(1) > T:
            measure_ids = measure_ids[:, -T:]
        x = x + self.measure_embedding(measure_ids)
        x = self.dropout(x)

        for i, block in enumerate(self.blocks):
            cache = None if kv_caches is None else kv_caches[i]
            x = block(x, attention_mask, cache)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits
