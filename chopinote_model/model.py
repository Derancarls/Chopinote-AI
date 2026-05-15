"""Decoder-only Transformer (GPT 风格) 用于音乐生成。"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

from .config import ModelConfig
from .fp8_linear import FP8Linear

_NEG_INF = float('-inf')
_SDPA_BACKENDS = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]


class CausalSelfAttention(nn.Module):
    """单层因果自注意力 (支持 KV cache)。"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0

        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model
        self.max_len = config.max_seq_len
        self.max_measures = config.max_measures

        self.qkv = FP8Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = FP8Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        self.rel_slope = nn.Parameter(torch.ones(config.n_heads) * 0.01)
        self.measure_bias = nn.Embedding(
            2 * config.max_measures - 1, config.n_heads, dtype=torch.bfloat16)

    def _build_total_bias(self, T: int, T_kv: int, mask: Optional[torch.Tensor],
                          measure_ids: Optional[torch.Tensor],
                          device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        i = torch.arange(T, device=device).view(-1, 1)
        j = torch.arange(T_kv, device=device).view(1, -1)
        pos_dist = (i - j).abs().unsqueeze(0).float()              # (1, T, T_kv)
        total_bias = -self.rel_slope.view(-1, 1, 1) * pos_dist     # (nH, T, T_kv)
        total_bias = total_bias.unsqueeze(0)                       # (1, nH, T, T_kv)

        if mask is not None:
            m = mask[0]
            pad_bias = torch.where(m.bool(), 0.0, _NEG_INF).to(dtype=total_bias.dtype)
            if m.size(0) < T_kv:
                pad_bias = F.pad(pad_bias, (0, T_kv - m.size(0)), value=_NEG_INF)
            total_bias.add_(pad_bias.view(1, 1, 1, -1))

        if measure_ids is not None:
            if measure_ids.dim() == 1:
                m_q = measure_ids[-T:]
                m_k = measure_ids
            else:
                m_q = measure_ids[0, -T:]
                m_k = measure_ids[0]
            dist = m_q.unsqueeze(-1) - m_k.unsqueeze(0)
            dist = dist.clamp(0, 2 * self.max_measures - 2).long()
            meas_bias = self.measure_bias(dist)
            meas_bias = meas_bias.permute(2, 0, 1).unsqueeze(0)
            total_bias.add_(meas_bias)

        causal = torch.triu(
            torch.full((T, T_kv), _NEG_INF, device=device, dtype=total_bias.dtype),
            diagonal=1,
        )
        total_bias.add_(causal)
        return total_bias.to(dtype=dtype).contiguous()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                kv_cache: Optional[list] = None,
                measure_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None and kv_cache[0] is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
            kv_cache[0] = k
            kv_cache[1] = v
        elif kv_cache is not None:
            kv_cache[0] = k
            kv_cache[1] = v

        T_kv = k.size(2)
        need_causal = (kv_cache is None or kv_cache[0] is None)
        total_bias = self._build_total_bias(T, T_kv,
                                            mask, measure_ids if need_causal else None,
                                            x.device, q.dtype)
        with sdpa_kernel(_SDPA_BACKENDS):
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=total_bias,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False,
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

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                kv_cache: Optional[list] = None,
                measure_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        need_ckpt = kv_cache is None and self.training
        if need_ckpt:
            # use_reentrant=False 会追踪中间张量 metadata，与 FP8 内部 .to(fp8) 不兼容
            # FP8 时用 use_reentrant=True 朴素重跑
            _reentrant = self.attn.qkv.use_fp8
            x = torch.utils.checkpoint.checkpoint(
                self._forward, x, mask, measure_ids, None,
                use_reentrant=_reentrant,
            )
        else:
            x = self._forward(x, mask, measure_ids, kv_cache)
        return x

    def _forward(self, x: torch.Tensor, mask: Optional[torch.Tensor],
                 measure_ids: Optional[torch.Tensor],
                 kv_cache: Optional[list] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask, kv_cache, measure_ids)
        x = x + self.ffn(self.ln2(x))
        return x


class MusicTransformer(nn.Module):
    """Decoder-only Transformer 用于音乐 token 序列生成。"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
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

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                kv_caches: Optional[list] = None,
                measure_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, \
            f'序列长度 {T} 超过 max_seq_len {self.config.max_seq_len}'

        if measure_ids is None:
            bar_mask = (input_ids == self.config.bar_token_id).int()
            measure_ids = torch.cumsum(bar_mask, dim=1)

        x = self.token_embedding(input_ids)
        x = self.dropout(x)

        for i, block in enumerate(self.blocks):
            cache = None if kv_caches is None else kv_caches[i]
            x = block(x, attention_mask, cache, measure_ids)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits
