"""Decoder-only Transformer (GPT 风格) 用于音乐生成，RoPE 位置编码 + 段落感知注意力。"""
import math
import weakref
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

from .config import ModelConfig, NO_SECTION_ID, NO_SECTION_TYPE_ID
from .fp8_linear import FP8Linear

_NEG_INF = float('-inf')

# flash/efficient attention (SDPA 4D mask 只需这两个)


class RMSNorm(nn.Module):
    """Per-head RMSNorm（QK-Norm 用，沿 head_dim 标准化）。"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_heads, T, head_dim)
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms * self.weight).to(x.dtype)


_SDPA_BACKENDS = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
try:
    _SDPA_BACKENDS.insert(0, SDPBackend.CUDNN_ATTENTION)
except AttributeError:
    pass
# 4D mask + cuDNN 会回退到 math backend → OOM
_SDPA_BACKENDS_4D = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
# 推理时只用 Flash + Efficient（Blackwell 5120 CuDNN 特定长度无可用内核）
_SDPA_BACKENDS_INFER = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]


class CausalSelfAttention(nn.Module):
    """单层因果自注意力 + RoPE (支持 KV cache + 段落偏置)。"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0

        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model
        self.max_len = config.max_seq_len
        self.use_section = config.use_section_attention
        self.attn_logit_cap = config.attn_logit_cap

        self.qkv = FP8Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = FP8Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        # ── QK-Norm（稳定注意力，防止 logit 爆炸）──
        if config.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        # ── Per-head Q/K 缩放（头特异性注意力温度）──
        if config.use_head_scale:
            self.q_head_scale = nn.Parameter(torch.ones(config.n_heads))
            self.k_head_scale = nn.Parameter(torch.ones(config.n_heads))
        else:
            self.register_buffer('q_head_scale', torch.ones(config.n_heads), persistent=False)
            self.register_buffer('k_head_scale', torch.ones(config.n_heads), persistent=False)

        # RoPE cache
        self.register_buffer('_rope_cos', None, persistent=False)
        self.register_buffer('_rope_sin', None, persistent=False)

    def _ensure_rope_cache(self, device: torch.device, dtype: torch.dtype, min_len: int = 0):
        need_len = max(self.max_len, min_len) if not self.training else self.max_len
        if self._rope_cos is not None and self._rope_cos.device == device and self._rope_cos.dtype == dtype and self._rope_cos.size(0) >= need_len:
            return
        theta = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2, device=device).float() / self.head_dim))
        pos = torch.arange(need_len, device=device).float()
        freqs = torch.outer(pos, theta)
        self._rope_cos = freqs.cos().to(dtype)
        self._rope_sin = freqs.sin().to(dtype)

    @staticmethod
    def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        T = x.shape[2]
        x_pairs = x.reshape(*x.shape[:-1], -1, 2)
        x_even, x_odd = x_pairs.unbind(-1)
        c = cos[:T].unsqueeze(0).unsqueeze(0)
        s = sin[:T].unsqueeze(0).unsqueeze(0)
        r_even = x_even * c - x_odd * s
        r_odd = x_even * s + x_odd * c
        return torch.stack([r_even, r_odd], dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                kv_cache: Optional[list] = None,
                sec_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # ── QK-Norm + Per-head scale ──────────────────────
        q = self.q_norm(q) * self.q_head_scale.to(dtype=q.dtype).view(1, -1, 1, 1)
        k = self.k_norm(k) * self.k_head_scale.to(dtype=k.dtype).view(1, -1, 1, 1)

        if kv_cache is not None and kv_cache[0] is not None:
            cache_len = kv_cache[0].size(2)
        else:
            cache_len = 0

        self._ensure_rope_cache(x.device, q.dtype, min_len=cache_len + T)

        if kv_cache is not None and kv_cache[0] is not None:
            k_new = self._apply_rope(k, self._rope_cos[cache_len:cache_len + T],
                                      self._rope_sin[cache_len:cache_len + T])
            k = torch.cat([kv_cache[0], k_new], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        else:
            cache_len = 0
            k = self._apply_rope(k, self._rope_cos[:T], self._rope_sin[:T])

        T_kv = k.size(2)

        q = self._apply_rope(q, self._rope_cos[cache_len:cache_len + T],
                             self._rope_sin[cache_len:cache_len + T])

        if kv_cache is not None:
            kv_cache[0] = k
            kv_cache[1] = v

        # 推理时始终 is_causal=True（单 token 生成等价且 SDPA 内核更稳定）
        use_causal = (kv_cache is None or kv_cache[0] is None) if not self.training else (
            kv_cache is None or kv_cache[0] is None or cache_len == 0)

        # ── 段落偏置（段落感知注意力）───────────────────────────
        if sec_bias is not None:
            # 编码为 4D attn_mask → SDPA flash/mem-efficient 分块计算
            # sec_bias (B,1,T,T_kv) 广播到所有头，不展开即无 (B,nH,T,T) 完整矩阵
            attn_mask = sec_bias.to(dtype=q.dtype)
            if use_causal:
                causal = torch.triu(
                    torch.full((T, T_kv), _NEG_INF, device=x.device, dtype=q.dtype),
                    diagonal=T_kv - T + 1)
                attn_mask = attn_mask + causal[None, None, :, :]
            if mask is not None:
                m = mask[0] if mask.dim() == 2 else mask
                pad = torch.where(m.bool(), 0.0, _NEG_INF).to(dtype=q.dtype)
                if pad.size(0) < T_kv:
                    pad = F.pad(pad, (0, T_kv - pad.size(0)), value=_NEG_INF)
                attn_mask = attn_mask + pad.view(1, 1, 1, -1)
            sdpa_backends_4d = _SDPA_BACKENDS_INFER if not self.training else _SDPA_BACKENDS_4D
            with sdpa_kernel(sdpa_backends_4d):
                y = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask,
                    dropout_p=self.dropout.p if self.training else 0.0,
                    is_causal=False)
        else:
            # 标准 SDPA 快速路径
            if mask is not None and use_causal:
                m = mask[0] if mask.dim() == 2 else mask
                pad = torch.where(m.bool(), 0.0, _NEG_INF).to(dtype=q.dtype)
                if pad.size(0) < T_kv:
                    pad = F.pad(pad, (0, T_kv - pad.size(0)), value=_NEG_INF)
                attn_mask = pad.view(1, 1, 1, -1)
            else:
                attn_mask = None

            sdpa_backends = _SDPA_BACKENDS_INFER if not self.training else _SDPA_BACKENDS
            try:
                with sdpa_kernel(sdpa_backends):
                    y = F.scaled_dot_product_attention(
                        q, k, v, attn_mask=attn_mask,
                        dropout_p=self.dropout.p if self.training else 0.0,
                        is_causal=use_causal,
                    )
            except RuntimeError:
                # Fallback: 手动 attention（Blackwell 某些长度下 SDPA 无可用内核）
                scale = self.head_dim ** -0.5
                attn = (q @ k.transpose(-2, -1)) * scale
                # ── Attention logit soft-capping（Gemma 风格）─
                if self.attn_logit_cap > 0:
                    attn = self.attn_logit_cap * torch.tanh(attn / self.attn_logit_cap)
                if attn_mask is not None:
                    attn = attn + attn_mask
                if use_causal:
                    causal = torch.triu(
                        torch.full((T, T_kv), _NEG_INF, device=x.device, dtype=q.dtype),
                        diagonal=T_kv - T + 1)
                    attn = attn + causal[None, None, :, :]
                attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
                y = attn @ v  # attn: (B,H,T,S) @ v: (B,H,S,D) → (B,H,T,D)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block: Attn → FFN 各带残差。(支持段落偏置)

    显存优化: sec_bias (B,1,T,T)=256MiB 改为传入原始数据 (B,T) 级张量
    (~1MiB), 在 _forward 内从头计算 bias. checkpoint 只存 1MiB 而非 256MiB
    → 24 层省 6 GiB. bias 计算量仅为模型的 0.14%, 可忽略.
    """

    def __init__(self, config: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
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
                kv_cache: Optional[list] = None,
                sec_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # sec_bias: inference 时是 (B,1,T,T) tensor, 训练时是 tuple of small tensors
        # 训练时 checkpoint 只存 ~1 MiB 的原始数据而非 256 MiB 的成品 bias
        need_ckpt = self.use_checkpointing and kv_cache is None and self.training
        if need_ckpt:
            x = torch.utils.checkpoint.checkpoint(
                self._forward, x, mask, None, sec_bias, use_reentrant=True)
        else:
            x = self._forward(x, mask, kv_cache, sec_bias)
        return x

    def _forward(self, x: torch.Tensor, mask: Optional[torch.Tensor],
                 kv_cache: Optional[list] = None,
                 sec_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # ── 训练路径: 从 raw 数据重建 bias ──────────────
        if isinstance(sec_bias, tuple):
            sec_ids, sec_types, bar_pos, qslice = sec_bias
            m = self._model_ref()
            sec_b = m._compute_sec_bias(sec_ids, sec_types, bar_pos, qslice)
            sec_bias = sec_b.detach() if sec_b is not None else None
        x = x + self.attn(self.ln1(x), mask, kv_cache, sec_bias)
        x = x + self.ffn(self.ln2(x))
        return x


class SectionPredictionHead(nn.Module):
    """段落属性预测头（双任务训练的辅助任务）。不预测段落长度（causal 架构下无法看到未来）。"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_model = config.d_model
        # 主音预测 (12-dim SSF TonicField regression)
        self.key_head = nn.Linear(config.d_model, config.ssf_dim)
        # 段落类型预测
        self.type_head = nn.Linear(config.d_model, config.n_section_types + 1)

    def forward(self, x: torch.Tensor) -> dict:
        return {
            'key': self.key_head(x),
            'type': self.type_head(x),
        }


class MusicTransformer(nn.Module):
    """Decoder-only Transformer 用于音乐 token 序列生成，RoPE + 段落感知注意力。"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.measure_embedding = nn.Embedding(config.max_measures + 1, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        # ── 段落感知组件 ──────────────────────────────────────
        if config.use_section_attention:
            self.section_embedding = nn.Embedding(config.max_sections + 1, config.d_model)
            self.section_type_embedding = nn.Embedding(config.n_section_types, config.d_model)
            self.sec_bias_alpha = nn.Parameter(torch.tensor(config.sec_bias_alpha_init))
            self.sec_bias_beta = nn.Parameter(torch.tensor(config.sec_bias_beta_init))
            self.sec_bias_gamma = nn.Parameter(torch.tensor(config.sec_bias_gamma_init))
            self.sec_bias_delta = nn.Parameter(torch.tensor(config.sec_bias_delta_init))
            self.register_buffer('sec_bias_decay_len', torch.tensor(config.sec_bias_decay_len, dtype=torch.float))
            self.section_head = SectionPredictionHead(config)

        # ── SSF (Sliding Scale Field) 调性编码 ────────────────
        if config.use_ssf:
            self.ssf_proj = nn.Linear(config.ssf_dim, config.d_model, bias=False)
            self.ssf_reconstruction_head = nn.Linear(config.d_model, config.ssf_dim)

        # ── 声部感知 (voice identity + bias) ──────────────────
        if config.use_voice_identity:
            self.voice_embedding = nn.Embedding(config.n_voice_ids, config.d_model)
            nn.init.zeros_(self.voice_embedding.weight)
        if config.use_voice_bias:
            self.voice_same_bonus = nn.Parameter(torch.tensor(config.voice_same_init))
            self.voice_samepos_bonus = nn.Parameter(torch.tensor(config.voice_samepos_init))

        if config.use_voice_count:
            self.voice_count_embedding = nn.Embedding(config.n_voices + 1, config.d_model)

        # ── 织体感知 ──────────────────────────────────────────
        if config.use_figuration:
            self.fig_embedding = nn.Embedding(config.n_fig_types, config.d_model)
            nn.init.zeros_(self.fig_embedding.weight)

        # ── 终止式感知 ────────────────────────────────────────
        if config.use_cadence:
            self.cadence_embedding = nn.Embedding(config.n_cadence_types, config.d_model)
            nn.init.zeros_(self.cadence_embedding.weight)

        # ── 节内位置感知 ────────────────────────────────────
        if config.use_measure_in_section:
            self.measure_in_section_embedding = nn.Embedding(
                config.max_measures_in_section + 1, config.d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(config, i) for i in range(config.n_layers)
        ])
        for block in self.blocks:
            block._model_ref = weakref.ref(self)  # 用于 bias 重建 (省显存)
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.token_embedding.weight = self.lm_head.weight
        self._init_weights()
        self._init_token_maps()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif 'bias' in name and not name.startswith('sec_bias') \
                 and not name.startswith('chord_bias'):
                nn.init.zeros_(p)
        # section_embedding 的 NO_SECTION_ID 为零向量（无段落）
        if self.config.use_section_attention:
            with torch.no_grad():
                self.section_embedding.weight[NO_SECTION_ID].zero_()
        # chord_embedding[0] 为零向量（padding）
        if self.config.use_chord_attention:
            with torch.no_grad():
                self.chord_embedding.weight[0].zero_()
                self.chord_inv_embedding.weight[0].zero_()
        # voice_count_embedding / measure_in_section_embedding 零初始化（不冲击已有训练）
        if self.config.use_voice_count:
            with torch.no_grad():
                self.voice_count_embedding.weight.zero_()
        if self.config.use_measure_in_section:
            with torch.no_grad():
                self.measure_in_section_embedding.weight.zero_()

    def _compute_sec_bias(self, section_ids: torch.Tensor,
                          section_types: torch.Tensor,
                          bar_positions: torch.Tensor,
                          query_slice: int = 0) -> Optional[torch.Tensor]:
        """计算段落感知注意力偏置。

        当 query_slice > 0 时（KV cache 模式），使用全部 section_ids 计算
        (B, 1, T_full, T_kv) 偏置矩阵，然后只返回最后 query_slice 个 query 行。

        sec_bias[i][j] =
          α × same_instance(i,j)
          + β × same_type_diff_instance(i,j) × exp(-|bar_i - bar_j| / decay_len)
          - γ × diff_type(i,j)
          + δ × boundary_region(i,j) × exp(-|bar_i - bar_j| / decay_len)
        """
        if not self.config.use_section_attention:
            return None
        if section_ids is None:
            return None

        B, T_full = section_ids.shape
        device = section_ids.device
        dtype = torch.bfloat16

        # 对齐维度: (B, T, 1) vs (B, 1, T) → (B, T, T)
        sid = section_ids.unsqueeze(-1)    # (B, T_full, 1)
        st = section_types.unsqueeze(-1)   # (B, T_full, 1)
        bar = bar_positions.unsqueeze(-1)  # (B, T_full, 1)

        same_inst = (sid == sid.transpose(-2, -1)).to(dtype)      # (B, T_full, T_full)
        same_type = (st == st.transpose(-2, -1)).to(dtype)        # (B, T_full, T_full)
        bar_dist = (bar - bar.transpose(-2, -1)).abs().to(dtype)  # (B, T_full, T_full)

        # 边界区域: 检测 section_id 变化的 bar 前后 4 小节
        boundary_mask = self._compute_boundary_mask(section_ids, bar_positions)  # (B, T_full, T_full)

        # clamp 标量偏置参数，防止训练过程中梯度爆炸
        _a = self.sec_bias_alpha.clamp(0.0, self.config.sec_bias_param_max)
        _b = self.sec_bias_beta.clamp(0.0, self.config.sec_bias_param_max)
        _g = self.sec_bias_gamma.clamp(0.0, self.config.sec_bias_param_max)
        _d = self.sec_bias_delta.clamp(0.0, self.config.sec_bias_param_max)

        sec_bias = torch.zeros(B, 1, T_full, T_full, device=device, dtype=dtype)
        sec_bias += _a * same_inst.unsqueeze(1)
        sec_bias += _b * (~same_inst.bool() & same_type.bool()).to(dtype).unsqueeze(1)
        sec_bias -= _g * (~same_type.bool()).to(dtype).unsqueeze(1)
        sec_bias += _d * boundary_mask.unsqueeze(1)

        # 距离衰减（适用于同类型跨实例和边界区域）
        decay = torch.exp(-bar_dist.unsqueeze(1) / self.sec_bias_decay_len)
        sec_bias = sec_bias * decay

        # KV cache 模式: 只返回最后 query_slice 个 query 行
        if query_slice > 0 and query_slice < T_full:
            sec_bias = sec_bias[:, :, -query_slice:, :]  # (B, 1, query_slice, T_full)

        return sec_bias

    def _compute_boundary_mask(self, section_ids: torch.Tensor,
                                bar_positions: torch.Tensor) -> torch.Tensor:
        """计算边界桥接区域 mask：(B, T, T)，边界前后 4 小节的 token 间为 1。"""
        B, T = section_ids.shape
        device = section_ids.device

        # 检测 bar 层级上的边界（section_id 改变的位置）
        sid_prev = torch.cat([section_ids[:, :1], section_ids[:, :-1]], dim=1)
        boundaries = (section_ids != sid_prev).float()  # (B, T)

        # 计算每个 token 到最近边界的 bar 距离
        bar = bar_positions.float()
        boundary_bar_pos = bar * boundaries  # 边界位置的 bar number
        # 计算每对 (i, j) 到边界的距离，简化版：检查是否存在边界在 i 和 j 之间
        # 更简单的实现：i 和 j 属于不同 section 且相邻 → 边界区域
        sid = section_ids  # (B, T)
        # j 属于前一段（i 的左边第一个不同 section）
        left_boundary = (sid[:, None, :] != sid[:, :, None]).float()  # (B, T, T)

        # bar 距离 < 4 且属于不同 section → 边界区域
        bar_diff = (bar[:, None, :] - bar[:, :, None]).abs()
        near_boundary = (bar_diff <= 4.0).float()
        boundary_mask = left_boundary * near_boundary

        return boundary_mask  # (B, T, T)

    # ── v0.3.0: Voice / Fig / Cadence ID builders ──────────────

    def _build_voice_ids(self, input_ids):
        """从 token 序列追踪当前 Voice: <Voice N> 之后属于声部 N+1。"""
        B, T = input_ids.shape
        voice_ids = torch.zeros(B, T, dtype=torch.long, device=input_ids.device)
        # Voice token IDs 0-3 → voice embedding indices 1-4
        for v in range(4):
            vid = self.voice_token_to_idx.get(v, -1)
            if vid < 0:
                continue
            current = 0
            for t in range(T):
                if input_ids[0, t] == vid:
                    current = v + 1
                voice_ids[:, t] = current
        return voice_ids

    def _build_fig_ids(self, input_ids):
        """从 token 序列追踪当前 Figuration type。"""
        B, T = input_ids.shape
        fig_ids = torch.zeros(B, T, dtype=torch.long, device=input_ids.device)
        for fid in self.fig_token_ids_set:
            current = 0
            for t in range(T):
                if input_ids[0, t] == fid:
                    current = fid - min(self.fig_token_ids_set) + 1 if self.fig_token_ids_set else 0
                fig_ids[:, t] = current
        return fig_ids

    def _build_cadence_ids(self, input_ids):
        """从 token 序列追踪当前 Cadence type。"""
        B, T = input_ids.shape
        cad_ids = torch.zeros(B, T, dtype=torch.long, device=input_ids.device)
        for cid in self.cadence_token_ids_set:
            current = 0
            for t in range(T):
                if input_ids[0, t] == cid:
                    current = cid - min(self.cadence_token_ids_set) + 1 if self.cadence_token_ids_set else 0
                cad_ids[:, t] = current
        return cad_ids

    def _init_token_maps(self):
        """缓存 token ID 集合, 用于快速查找。"""
        if self.config.use_voice_identity:
            self.voice_token_to_idx = {}
            for v in range(4):
                t = f'<Voice {v}>'
                tid = self.tokenizer.encode_token(t) if hasattr(self, 'tokenizer') else -1
                if tid >= 0:
                    self.voice_token_to_idx[v] = tid
        if self.config.use_figuration:
            self.fig_token_ids_set = set()
            # Fig tokens are at the end of vocab, find them by prefix
            for tid in range(self.config.vocab_size):
                tok = self._id_to_token_safe(tid)
                if tok.startswith('<Fig '):
                    self.fig_token_ids_set.add(tid)
        if self.config.use_cadence:
            self.cadence_token_ids_set = set()
            for tid in range(self.config.vocab_size):
                tok = self._id_to_token_safe(tid)
                if tok.startswith('<Cad '):
                    self.cadence_token_ids_set.add(tid)

    def _id_to_token_safe(self, tid):
        """安全地查找 token ID → string（兼容 tokenizer 可能未初始化）。"""
        try:
            return self.tokenizer.decode_token(tid)
        except Exception:
            return '<MASK>'

    def _build_chord_group_map(self):
        """构建和弦功能 ID → 功能组 ID 映射 (buffer, 不可训练)。

        Group 0 = Tonic (T), Group 1 = Subdominant (S), Group 2 = Dominant (D).
        ID 0 = padding → group -1.
        ID 1-16 = chord functions (in CHORD_FUNCTIONS order).
        """
        TONIC, SUBDOM, DOMINANT = 0, 1, 2
        _id_to_group = {
            0:  -1,                     # padding
            1:  TONIC,    2:  TONIC,    # I, i
            3:  SUBDOM,   4:  SUBDOM,   # ii, ii°
            5:  TONIC,    6:  TONIC,    # iii, III
            7:  SUBDOM,   8:  SUBDOM,   # IV, iv
            9:  DOMINANT,               # V
            10: TONIC,    11: TONIC,    # vi, VI
            12: DOMINANT,               # vii°
            13: SUBDOM,                 # N
            14: SUBDOM,  15: SUBDOM,  16: SUBDOM,  # It6, Fr6, Ger6
        }
        group_map = torch.zeros(self.config.n_chord_funcs, dtype=torch.long)
        for cid, gid in _id_to_group.items():
            if cid < self.config.n_chord_funcs:
                group_map[cid] = gid
        self.register_buffer('chord_group_map', group_map, persistent=True)

    def _init_chord_token_sets(self):
        """构建 Chord/Inv token ID → embedding 索引映射。"""
        if not self.config.use_chord_attention:
            self._chord_func_map: dict = {}
            self._chord_inv_map: dict = {}
            return
        from chopinote_dataset.tokenizer import REMITokenizer
        t = REMITokenizer()
        # token_id → func_index (1-16)
        self._chord_func_map = {}
        for i, name in enumerate(t.CHORD_FUNCTIONS, start=1):
            tid = t.encode_token(f'<Chord {name}>')
            self._chord_func_map[tid] = i
        # token_id → inv_index (1-4)
        self._chord_inv_map = {}
        for i, name in enumerate(t.CHORD_INVERSIONS, start=1):
            tid = t.encode_token(f'<Inv {name}>')
            self._chord_inv_map[tid] = i

    def _compute_chord_bias(self, chord_func_ids: torch.Tensor,
                            bar_positions: torch.Tensor,
                            sec_bias: Optional[torch.Tensor] = None,
                            query_slice: int = 0) -> torch.Tensor:
        """计算和弦感知注意力偏置 (B, 1, T, T)。

        chord_func_ids: (B, T) 每个 token 当前和弦功能 ID (0=padding, 1-16=func)
        bar_positions: (B, T) 每个 token 小节号
        sec_bias: 段落偏置，用于边界去重

        γ (gamma): 同和弦内 token 凝聚
        ε (epsilon): 和弦切换处桥接 (仅 ±epsilon_bar_window 小节邻域)
        ζ (zeta): 同功能组不同和弦间弱正偏置
        """
        if not self.config.use_chord_attention:
            return torch.zeros(1, 1, 1, 1, device=chord_func_ids.device, dtype=torch.bfloat16)

        B, T_full = chord_func_ids.shape
        device = chord_func_ids.device
        dtype = torch.bfloat16

        cid = chord_func_ids  # (B, T_full)
        bar = bar_positions.float()  # (B, T_full)

        # ── 广播矩阵 ──
        cid_i = cid.unsqueeze(-1)  # (B, T, 1)
        cid_j = cid.unsqueeze(-2)  # (B, 1, T)
        bar_i = bar.unsqueeze(-1)
        bar_j = bar.unsqueeze(-2)

        valid_i = (cid_i != 0)  # padding = 0 = no chord
        valid_j = (cid_j != 0)
        both_valid = valid_i & valid_j  # (B, T, T)

        same_chord = (cid_i == cid_j) & both_valid  # (B, T, T)
        chord_change = both_valid & (cid_i != cid_j)

        # 功能组
        cg_i = self.chord_group_map[cid_i.clamp(0)]  # (B, T, 1)
        cg_j = self.chord_group_map[cid_j.clamp(0)]  # (B, 1, T)
        same_group = (cg_i == cg_j) & (~same_chord) & both_valid & (cg_i != -1)

        bar_dist = (bar_i - bar_j).abs()  # (B, T, T)

        # 距离衰减
        decay_full = torch.exp(-bar_dist / self.chord_decay_len)
        decay_epsilon = (bar_dist <= self.chord_epsilon_bar_window).to(dtype)

        _cg = self.chord_bias_gamma.clamp(0.0, self.config.chord_bias_param_max)
        _ce = self.chord_bias_epsilon.clamp(0.0, self.config.chord_bias_param_max)
        _cz = self.chord_bias_zeta.clamp(0.0, self.config.chord_bias_param_max)

        chord_bias = torch.zeros(B, 1, T_full, T_full, device=device, dtype=dtype)

        # γ: 同和弦凝聚 × 距离衰减
        chord_bias += _cg * same_chord.to(dtype).unsqueeze(1) * decay_full.unsqueeze(1)

        # ε: 和弦切换桥接 × 窄窗口
        chord_bias += _ce * chord_change.to(dtype).unsqueeze(1) * decay_epsilon.unsqueeze(1)

        # ζ: 同功能组弱正偏置 × 距离衰减
        chord_bias += _cz * same_group.to(dtype).unsqueeze(1) * decay_full.unsqueeze(1)

        # ── 与 sec_bias δ 去重 ──
        if sec_bias is not None:
            # 用 torch.abs + inplace clamp 避免额外中间张量
            divisor = self.config.sec_bias_delta_init + 1e-8
            delta_strength = sec_bias.abs().div_(divisor).clamp_(0, 1)
            chord_bias.mul_(1 - 0.5 * delta_strength)

        # KV cache: 只返回最后 query_slice 个 query 行
        if query_slice > 0 and query_slice < T_full:
            chord_bias = chord_bias[:, :, -query_slice:, :]

        return chord_bias

    def set_fp8_mode(self, enabled: bool):
        for module in self.modules():
            if isinstance(module, FP8Linear):
                module.use_fp8 = enabled

    def invalidate_fp8_caches(self):
        for module in self.modules():
            if isinstance(module, FP8Linear):
                module.invalidate_cache()

    def set_gradient_checkpointing(self, enabled: bool):
        for block in self.blocks:
            block.use_checkpointing = enabled

    def set_dropout(self, p: float):
        """动态设置所有 dropout 概率（用于阶梯衰减）。"""
        self.dropout.p = p
        for block in self.blocks:
            block.attn.dropout.p = p
            # FFN dropout 在 Sequential 末尾
            block.ffn[-1].p = p

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                kv_caches: Optional[list] = None,
                measure_ids: Optional[torch.Tensor] = None,
                section_ids: Optional[torch.Tensor] = None,
                section_types: Optional[torch.Tensor] = None,
                ssf_fields: Optional[torch.Tensor] = None,
                voice_count_ids: Optional[torch.Tensor] = None,
                measure_in_section_ids: Optional[torch.Tensor] = None,
                return_sec_head: bool = False) -> torch.Tensor:
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, \
            f'序列长度 {T} 超过 max_seq_len {self.config.max_seq_len}'

        # ── Token + Measure embedding ────────────────────────────
        x = self.token_embedding(input_ids)
        if measure_ids is None:
            bar_mask = (input_ids == self.config.bar_token_id).int()
            measure_ids = torch.cumsum(bar_mask, dim=1).clamp(0, self.config.max_measures)
        elif measure_ids.ndim == 1:
            measure_ids = measure_ids.unsqueeze(0)
        # 安全截断超出 max_measures 的值
        measure_ids = measure_ids.clamp(0, self.config.max_measures)
        measure_ids_full = measure_ids  # 保存完整历史，用于 sec_bias 计算
        if measure_ids.size(1) > T:
            measure_ids = measure_ids[:, -T:]
        x = x + self.measure_embedding(measure_ids)

        # ── Measure-in-section embedding ────────────────────
        if self.config.use_measure_in_section and measure_in_section_ids is not None:
            if measure_in_section_ids.ndim == 1:
                measure_in_section_ids = measure_in_section_ids.unsqueeze(0)
            ms_clamped = measure_in_section_ids[:, -T:].clamp(0, self.config.max_measures_in_section)
            x = x + self.measure_in_section_embedding(ms_clamped)

        # ── Section embedding ────────────────────────────────────
        sec_bias = None
        use_sec = (self.config.use_section_attention and section_ids is not None
                   and section_ids.ne(NO_SECTION_ID).any())
        if use_sec:
            if section_ids.ndim == 1:
                section_ids = section_ids.unsqueeze(0)
            if section_types.ndim == 1:
                section_types = section_types.unsqueeze(0)

            section_ids_full = section_ids
            section_types_full = section_types

            sec_ids_emb = section_ids[:, -T:]
            sec_types_emb = section_types[:, -T:]
            x = x + self.section_embedding(sec_ids_emb)
            x = x + self.section_type_embedding(sec_types_emb)

            in_kv_decode = kv_caches is not None and len(kv_caches) > 0 and kv_caches[0] is not None and kv_caches[0][0] is not None
            if in_kv_decode:
                sec_bias = self._compute_sec_bias(
                    section_ids_full, section_types_full, measure_ids_full, query_slice=T)
            else:
                sec_bias = self._compute_sec_bias(
                    section_ids_full[:, -T:], section_types_full[:, -T:],
                    measure_ids_full[:, -T:], query_slice=0)

        # ── Voice count embedding ────────────────────────────────
        if self.config.use_voice_count and voice_count_ids is not None:
            if voice_count_ids.ndim == 1:
                voice_count_ids = voice_count_ids.unsqueeze(0)
            voice_ids_clamped = voice_count_ids[:, -T:].clamp(0, self.config.n_voices)
            x = x + self.voice_count_embedding(voice_ids_clamped)

        # ── SSF conditioning ────────────────────────────────────
        if self.config.use_ssf and ssf_fields is not None:
            if ssf_fields.ndim == 2:
                ssf_fields = ssf_fields.unsqueeze(0)
            ssf_emb = self.ssf_proj(ssf_fields[:, -T:].to(x.dtype))
            x = x + ssf_emb

        # ── Voice identity embedding ────────────────────────────
        if self.config.use_voice_identity:
            voice_ids = self._build_voice_ids(input_ids[:, -T:])
            x = x + self.voice_embedding(voice_ids)

        # ── Figuration embedding ─────────────────────────────────
        if self.config.use_figuration:
            fig_ids = self._build_fig_ids(input_ids[:, -T:])
            x = x + self.fig_embedding(fig_ids)

        # ── Cadence embedding ────────────────────────────────────
        if self.config.use_cadence:
            cadence_ids = self._build_cadence_ids(input_ids[:, -T:])
            x = x + self.cadence_embedding(cadence_ids)

        x = self.dropout(x)

        # ── bias_data for checkpoint recompute ──────────────────
        in_kv_decode = (kv_caches is not None and len(kv_caches) > 0
                       and kv_caches[0] is not None and kv_caches[0][0] is not None)
        bias_data = None
        if use_sec:
            zeros2d = torch.zeros(B, T, dtype=torch.long, device=input_ids.device)
            if in_kv_decode:
                bias_data = (
                    section_ids_full if use_sec else zeros2d,
                    section_types_full if use_sec else zeros2d,
                    measure_ids_full,
                    T,
                )
            else:
                bias_data = (
                    section_ids_full[:, -T:] if use_sec else zeros2d,
                    section_types_full[:, -T:] if use_sec else zeros2d,
                    measure_ids_full[:, -T:],
                    0,   # query_slice: 取全部 query 行
                )

        # ── Transformer blocks ───────────────────────────────────
        for i, block in enumerate(self.blocks):
            cache = None if kv_caches is None else kv_caches[i]
            x = block(x, attention_mask, cache, bias_data)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        # ── SSF reconstruction + Section head ─────────────────────
        result = [logits]
        if return_sec_head and use_sec:
            result.append(self.section_head(x))
        if self.config.use_ssf and self.config.use_ssf_reconstruction:
            result.append(self.ssf_reconstruction_head(x))
        return result[0] if len(result) == 1 else tuple(result)
