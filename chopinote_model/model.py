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

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        # causal mask: (1, 1, max_seq_len, max_seq_len) 下三角
        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
            .view(1, 1, config.max_seq_len, config.max_seq_len),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                kv_cache: Optional[list] = None) -> torch.Tensor:
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

        # FlashAttention（自动选择 cuDNN flash / mem-eff / math 后端）
        if kv_cache is not None and kv_cache[0] is not None:
            # 推理：单 token → 不需要因果掩码
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=0.0, is_causal=False,
            )
        else:
            # 训练：因果掩码 + 可选 padding 掩码
            attn_mask = None
            if mask is not None:
                causal = self.causal_mask[:, :, :T, :k.size(2)].bool()
                padding = mask[:, None, None, :k.size(2)].bool()
                attn_mask = causal & padding
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=attn_mask is None,
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
                kv_cache: Optional[list] = None) -> torch.Tensor:
        if kv_cache is None and self.training:
            x = x + checkpoint(self._ckpt_attn, self.ln1(x), mask, use_reentrant=False)
        else:
            x = x + self.attn(self.ln1(x), mask, kv_cache)
        x = x + self.ffn(self.ln2(x))
        return x

    def _ckpt_attn(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Gradient checkpointing 包装的注意力前向。"""
        return self.attn(x, mask, None)


class MusicTransformer(nn.Module):
    """Decoder-only Transformer 用于音乐 token 序列生成。"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, config.max_seq_len, config.d_model))
        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # weight tying
        self.token_embedding.weight = self.lm_head.weight

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                kv_caches: Optional[list] = None) -> torch.Tensor:
        """前向传播。

        Args:
            input_ids: (B, T) token IDs
            attention_mask: (B, T) — 1=keep, 0=pad
            kv_caches: 每层 (k, v) 的列表，推理时传入以复用历史

        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, \
            f'序列长度 {T} 超过 max_seq_len {self.config.max_seq_len}'

        x = self.token_embedding(input_ids)  # (B, T, d_model)

        # 位置编码：对于 cached 生成，从 cache 长度开始
        cache_len = 0
        if kv_caches is not None and kv_caches[0] is not None and kv_caches[0][0] is not None:
            cache_len = kv_caches[0][0].size(2)
        x = x + self.pos_embedding[:, cache_len:cache_len + T, :]

        x = self.dropout(x)

        for i, block in enumerate(self.blocks):
            cache = None if kv_caches is None else kv_caches[i]
            x = block(x, attention_mask, cache)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits

    def compute_loss(self, input_ids: torch.Tensor,
                     attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """计算 next-token prediction loss。"""
        logits = self.forward(input_ids, attention_mask)  # (B, T, V)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction='sum',
        )
        loss = loss / max(1, (shift_labels != -100).sum())
        return loss
