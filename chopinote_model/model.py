"""Decoder-only Transformer (GPT 风格) 用于音乐生成。"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig


class CausalSelfAttention(nn.Module):
    """单层因果自注意力 (支持 KV cache)。"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0

        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model
        self.max_len = config.max_seq_len

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        # 相对位置偏置表: 覆盖相对距离 [-(L-1), (L-1)]
        self.rel_bias = nn.Parameter(torch.zeros(2 * config.max_seq_len - 1, config.n_heads))
        # 小节目对偏置表: 覆盖 measure 距离 [-(M-1), (M-1)]
        self.measure_bias = nn.Parameter(torch.zeros(2 * config.max_measures - 1, config.n_heads))

    def _get_rel_bias(self, T_q: int, T_k: int) -> torch.Tensor:
        """生成相对位置偏置矩阵。

        Returns:
            (1, n_heads, T_q, T_k) — 可广播到 attention scores
        """
        i = torch.arange(T_q, device=self.rel_bias.device).view(-1, 1)   # (T_q, 1)
        j = torch.arange(T_k, device=self.rel_bias.device).view(1, -1)   # (1, T_k)
        dist = i - j + self.max_len - 1                                   # (T_q, T_k)
        dist = dist.clamp(0, 2 * self.max_len - 2)
        bias = self.rel_bias[dist]                                         # (T_q, T_k, n_heads)
        return bias.permute(2, 0, 1).unsqueeze(0)                          # (1, n_heads, T_q, T_k)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                kv_cache: Optional[list] = None,
                measure_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # KV cache
        if kv_cache is not None and kv_cache[0] is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
            kv_cache[0] = k
            kv_cache[1] = v
        elif kv_cache is not None:
            kv_cache[0] = k
            kv_cache[1] = v

        # 相对位置注意力（方案 C: 偏置通过 attn_mask 传入）
        total_bias = self._get_rel_bias(T, k.size(2))

        # ── 小节目对偏置 ──────────────────────────────────
        if measure_ids is not None:
            # measure_ids: (T_total,) 推理 或 (B, T) 训练
            if measure_ids.dim() == 1:
                m_q = measure_ids[-T:]    # (T_q,) — 当前输入的 measure_ids
                m_k = measure_ids         # (T_total,) — 全部
                dist = m_q.unsqueeze(-1) - m_k.unsqueeze(0)  # (T_q, T_k)
            else:
                m_q = measure_ids[:, -T:]  # (B, T_q)
                m_k = measure_ids          # (B, T_k)
                dist = m_q.unsqueeze(2) - m_k.unsqueeze(1)  # (B, T_q, T_k)

            dist = dist.clamp(0, 2 * self.max_measures - 2).long()
            meas_bias = self.measure_bias[dist]  # (..., T_q, T_k, n_heads)

            if meas_bias.dim() == 4:
                meas_bias = meas_bias.permute(0, 3, 1, 2)   # (B, nH, T_q, T_k)
            else:
                meas_bias = meas_bias.permute(2, 0, 1).unsqueeze(0)  # (1, nH, T_q, T_k)

            total_bias = total_bias + meas_bias
        # ──────────────────────────────────────────────────

        if kv_cache is not None and kv_cache[0] is not None:
            # 推理：单 token → attn_mask=rel_bias, 无需因果掩码
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=total_bias,
                dropout_p=0.0, is_causal=False,
            )
        else:
            # 训练：is_causal=True 处理因果掩码
            # padding 屏蔽由 loss ignore_index=-100 处理
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=total_bias,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True,
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
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                kv_cache: Optional[list] = None,
                measure_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if kv_cache is None and self.training:
            x = x + checkpoint(self._ckpt_attn, self.ln1(x), mask, measure_ids,
                               use_reentrant=False)
        else:
            x = x + self.attn(self.ln1(x), mask, kv_cache, measure_ids)
        x = x + self.ffn(self.ln2(x))
        return x

    def _ckpt_attn(self, x: torch.Tensor, mask: Optional[torch.Tensor],
                   measure_ids: Optional[torch.Tensor]) -> torch.Tensor:
        return self.attn(x, mask, None, measure_ids)


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

        # weight tying
        self.token_embedding.weight = self.lm_head.weight

        self._init_weights()

        # torch.compile 实测在此模型上无加速（瓶颈在显存带宽而非计算），已移除

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                kv_caches: Optional[list] = None,
                measure_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播。

        Args:
            input_ids: (B, T) token IDs
            attention_mask: (B, T) — 1=keep, 0=pad
            kv_caches: 每层 (k, v) 的列表，推理时传入以复用历史
            measure_ids: (T_total,) 推理用，或 None 时从 input_ids 自动计算

        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, \
            f'序列长度 {T} 超过 max_seq_len {self.config.max_seq_len}'

        # 计算 measure_ids（训练时从 input_ids 自动计算）
        if measure_ids is None:
            bar_mask = (input_ids == self.config.bar_token_id).int()
            measure_ids = torch.cumsum(bar_mask, dim=1)  # (B, T)

        x = self.token_embedding(input_ids)  # (B, T, d_model)
        x = self.dropout(x)

        for i, block in enumerate(self.blocks):
            cache = None if kv_caches is None else kv_caches[i]
            x = block(x, attention_mask, cache, measure_ids)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits

