"""Decoder-only Transformer (GPT 风格) 用于音乐生成。"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig

_NEG_INF = float('-inf')


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

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        # 相对位置偏置表: 覆盖相对距离 [-(L-1), (L-1)]
        # bf16 参数省一半显存，同时 GradScaler 可接受 bf16 梯度（与 fp32 同 exponent range）
        self.rel_bias = nn.Parameter(torch.zeros(2 * config.max_seq_len - 1, config.n_heads, dtype=torch.bfloat16))
        # 小节目对偏置表: 覆盖 measure 距离 [-(M-1), (M-1)]
        self.measure_bias = nn.Parameter(torch.zeros(2 * config.max_measures - 1, config.n_heads, dtype=torch.bfloat16))

    def _get_rel_bias(self, T_q: int, T_k: int) -> torch.Tensor:
        i = torch.arange(T_q, device=self.rel_bias.device).view(-1, 1)
        j = torch.arange(T_k, device=self.rel_bias.device).view(1, -1)
        dist = i - j + self.max_len - 1
        dist = dist.clamp(0, 2 * self.max_len - 2)
        bias = self.rel_bias[dist]                                        # (T_q, T_k, n_heads)
        return bias.permute(2, 0, 1).unsqueeze(0)                        # (1, nH, T_q, T_k)

    def _attn_with_bias(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                        total_bias: torch.Tensor) -> torch.Tensor:
        """逐 head 计算 attention，避免一次性分配 (B, nH, T, T) 的 4 GiB 张量。

        sm_120(Blackwell) 上 SDPA backend 不支持 float attn_mask，
        manual matmul 又在 24 GiB 占用下无法分配 (B, nH, T, T) 连续空间。
        """
        scale = self.head_dim ** -0.5
        dropout_p = self.dropout.p if self.training else 0.0
        B, nH, T, head_dim = q.shape
        T_kv = k.size(2)
        output_parts = []
        causal = torch.triu(torch.ones(T, T_kv, device=q.device, dtype=torch.bool), diagonal=1)
        causal = causal.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T_kv)
        for h in range(nH):
            # (B, 1, T, T_kv) = ~64 MiB @ bf16 (B=2,T=4096)
            score = (q[:, h:h+1] @ k[:, h:h+1].transpose(-2, -1)) * scale
            score.masked_fill_(causal, _NEG_INF)
            score = score + total_bias[:, h:h+1]
            score = F.softmax(score, dim=-1, dtype=torch.float32).to(q.dtype)
            score = F.dropout(score, p=dropout_p)
            out = score @ v[:, h:h+1]
            output_parts.append(out)
        return torch.cat(output_parts, dim=1)

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

        # 相对位置偏置（AMP 下 autocast 自动处理精度统一）
        total_bias = self._get_rel_bias(T, k.size(2))

        # padding mask（batch 内 pattern 相同，只取第一行避免广播出 (B,...) 张量）
        if mask is not None:
            T_kv = k.size(2)
            m = mask[0]  # (T,) — 所有样本 padding 模式一致
            pad_bias = torch.where(m.bool(), 0.0, _NEG_INF).to(dtype=total_bias.dtype)
            if m.size(0) < T_kv:
                pad_bias = F.pad(pad_bias, (0, T_kv - m.size(0)), value=_NEG_INF)
            total_bias += pad_bias.view(1, 1, 1, -1)  # in-place，与 total_bias 同 dtype

        # ── 小节目对偏置 ──────────────────────────────────
        if measure_ids is not None:
            # measure_ids: (T_total,) 推理 或 (B, T) 训练
            if measure_ids.dim() == 1:
                m_q = measure_ids[-T:]    # (T_q,) — 当前输入的 measure_ids
                m_k = measure_ids         # (T_total,) — 全部
            else:
                # 只用第一个样本的 measure 结构，避免广播出 (B,nH,T,T) 的 3-4 GiB 分配
                m_q = measure_ids[0, -T:]  # (T_q,)
                m_k = measure_ids[0]        # (T_k,)
            dist = m_q.unsqueeze(-1) - m_k.unsqueeze(0)  # (T_q, T_k)

            dist = dist.clamp(0, 2 * self.max_measures - 2).long()
            meas_bias = self.measure_bias[dist]              # (T_q, T_k, n_heads) fp32
            meas_bias = meas_bias.permute(2, 0, 1).unsqueeze(0)  # (1, nH, T_q, T_k)

            total_bias += meas_bias  # 同 shape，in-place 避免 1 GiB 分配
        # ──────────────────────────────────────────────────

        if kv_cache is not None and kv_cache[0] is not None:
            # 推理：单 token → attn_mask=total_bias, 无需因果掩码
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=total_bias,
                dropout_p=0.0, is_causal=False,
            )
        else:
            # 逐 head 计算，避免 (B, nH, T, T) = 2 GiB 连续分配导致 OOM
            # sm_120(Blackwell) 上 flash/mem_efficient 均不支持非空 float attn_mask
            y = self._attn_with_bias(q, k, v, total_bias)

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
            x = checkpoint(self._ckpt_block, x, mask, measure_ids,
                           use_reentrant=False)
        else:
            x = self._ckpt_block(x, mask, measure_ids)
        return x

    def _ckpt_block(self, x: torch.Tensor, mask: Optional[torch.Tensor],
                    measure_ids: Optional[torch.Tensor]) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask, None, measure_ids)
        x = x + self.ffn(self.ln2(x))
        return x


class MusicTransformer(nn.Module):
    """Decoder-only Transformer 用于音乐 token 序列生成。"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        # sm_120(Blackwell): flash attention 不支持 float attn_mask，
        # 禁用后 SDPA dispatcher 自动走 mem_efficient（支持非空 mask，无 OOM）
        torch.backends.cuda.enable_flash_sdp(False)

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

