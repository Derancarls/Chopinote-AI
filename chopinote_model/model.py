"""Decoder-only Transformer (GPT 风格) 用于音乐生成。"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

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

        qkv = self.qkv(x)  # (B, T, 3*C)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B, T, C)

        # split heads
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
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

        # attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))  # (B, H, T, T')
        T_k = k.size(2)

        # 因果掩码：只在首次 forward（无 KV cache）时应用，
        # 后续生成时 k 中只有历史 token，无需额外因果掩码
        if kv_cache is None or kv_cache[0] is None:
            att = att.masked_fill(self.causal_mask[:, :, :T, :T_k] == 0, float('-inf'))

        # padding mask
        if mask is not None:
            # mask: (B, T_k) — 1=keep, 0=pad
            att = att.masked_fill(mask[:, None, None, :T_k] == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ v  # (B, H, T, D)
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
        x = x + self.attn(self.ln1(x), mask, kv_cache)
        x = x + self.ffn(self.ln2(x))
        return x


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

    @torch.no_grad()
    def generate(self, seed_tokens: torch.Tensor, max_new_tokens: int = 512,
                 temperature: float = 1.0, top_k: Optional[int] = None,
                 eos_token_id: int = 2, pad_token_id: int = 0) -> torch.Tensor:
        """自回归生成。

        Args:
            seed_tokens: (1, T_seed) 初始 token 序列
            max_new_tokens: 最多生成步数
            temperature: 采样温度 (<1 更确定, >1 更随机)
            top_k: 只从前 k 个最高概率 token 采样
            eos_token_id: 遇到此 token 停止
            pad_token_id: 填充 ID

        Returns:
            (1, T_total) 完整的生成序列
        """
        self.eval()
        device = seed_tokens.device
        B = seed_tokens.shape[0]

        # 初始化 KV cache
        kv_caches = [[None, None] for _ in range(self.config.n_layers)]

        generated = seed_tokens.clone()
        next_token = seed_tokens

        for _ in range(max_new_tokens):
            logits = self.forward(next_token, kv_caches=kv_caches)  # (1, 1, V)
            logits = logits[:, -1, :] / temperature  # (1, V)

            # top-k 过滤
            if top_k is not None:
                top_k = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, top_k, dim=-1)
                logits[logits < values[:, -1:]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == eos_token_id:
                break

        return generated

    def compute_loss(self, input_ids: torch.Tensor,
                     attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """计算 next-token prediction loss。"""
        logits = self.forward(input_ids, attention_mask)  # (B, T, V)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.config.pad_token_id,
            reduction='sum',
        )
        loss = loss / max(1, (shift_labels != self.config.pad_token_id).sum())
        return loss
