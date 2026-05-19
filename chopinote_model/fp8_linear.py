"""FP8 混合精度线性层 — RTX 5090 Blackwell _scaled_mm 加速（v2: 权重量化缓存）。

Weight caching: 权重只在 optimizer.step() 后变化，FP8 量化结果可跨 forward/backward
复用，避免每次前向/反向都重新转换 + contiguous + transpose 的开销。

用法:
    model.set_fp8_mode(True)
    # ... training loop ...
    optimizer.step()
    model.invalidate_fp8_caches()  # 通知所有权重已更新
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_scale(t: torch.Tensor) -> torch.Tensor:
    """返回 scale = amax / 448，使 x / scale 落入 FP8 范围。"""
    amax = t.detach().abs().max().float()
    return torch.clamp(amax / 448.0, min=1e-4)


class FP8LinearFn(torch.autograd.Function):
    """FP8 矩阵乘法 autograd Function（v2: 使用缓存的列主序 FP8 权重）。"""

    @staticmethod
    def forward(ctx, x, w_fp8_fwd, w_fp8_bwd, weight_bf16, bias, scale_x, scale_w):
        M = x.reshape(-1, x.size(-1)).shape[0]
        K = weight_bf16.size(1)
        N = weight_bf16.size(0)

        x_fp8 = (x.reshape(-1, K).float() / scale_x).to(torch.float8_e4m3fn)

        out = torch._scaled_mm(x_fp8, w_fp8_fwd, scale_x, scale_w,
                               out_dtype=torch.bfloat16)
        out = out.reshape(*x.shape[:-1], N)

        if bias is not None:
            out = out + bias.to(torch.bfloat16)

        ctx.save_for_backward(x, w_fp8_bwd, weight_bf16, scale_x, scale_w)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, w_fp8_bwd, weight, scale_x, scale_w = ctx.saved_tensors
        M, K = x.reshape(-1, x.size(-1)).shape
        N = weight.size(0)

        go_flat = grad_out.reshape(-1, N)
        go_scale = compute_scale(go_flat)

        # grad_input = grad_out @ W   (M, N) @ (N, K) → (M, K)
        # _scaled_mm 要求第二参数为列主序: go (M,N) row-major @ w_bwd (N,K) col-major
        go_fp8 = (go_flat.float() / go_scale).to(torch.float8_e4m3fn)
        grad_x = torch._scaled_mm(go_fp8, w_fp8_bwd, go_scale, scale_w,
                                  out_dtype=torch.bfloat16).to(torch.bfloat16)
        grad_x = grad_x.reshape_as(x)

        # grad_weight = x^T @ grad_out   (K, M) @ (M, N) → (K, N)
        x_row = x.reshape(-1, K).t().contiguous()  # (K, M)
        go_flat_gw = go_flat
        if M % 16 != 0:
            pad = 16 - M % 16
            x_row = F.pad(x_row, (0, pad))
            go_flat_gw = F.pad(go_flat, (0, 0, 0, pad))

        x_fp8 = (x_row.float() / scale_x).to(torch.float8_e4m3fn)
        go_col = (go_flat_gw.t().contiguous().t().float() / go_scale).to(torch.float8_e4m3fn)
        grad_w = torch._scaled_mm(x_fp8, go_col, scale_x, go_scale,
                                  out_dtype=torch.bfloat16).to(torch.bfloat16)
        grad_w = grad_w.t()  # (K, N).t() → (N, K)

        grad_bias = grad_out.reshape(-1, N).sum(0) if ctx.needs_input_grad[4] else None
        return grad_x, None, None, grad_w, grad_bias, None, None


class FP8Linear(nn.Module):
    """FP8 线性层，state_dict 兼容 nn.Linear（weight/bias key 完全一致）。

    use_fp8=False 时 fallback 到普通 BF16 matmul，等效 nn.Linear。
    use_fp8=True 时使用 _scaled_mm FP8 加速 + 权重量化缓存。
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_fp8: bool = False

        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.bfloat16))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float32))
        else:
            self.register_parameter('bias', None)

        # FP8 权重缓存（仅在 use_fp8=True 且权重变化后重算）
        self._scale_x: Optional[torch.Tensor] = None
        self._scale_w: Optional[torch.Tensor] = None
        self._w_fp8_fwd: Optional[torch.Tensor] = None  # weight.T 列主序 fp8（forward）
        self._w_fp8_bwd: Optional[torch.Tensor] = None  # weight 列主序 fp8（backward grad_x）
        self._weight_version: int = -1

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        self.invalidate_cache()

    def invalidate_cache(self):
        """标记缓存的 FP8 权重量化失效（optimizer.step() 后调用）。"""
        self._weight_version = -1

    def _ensure_cache(self):
        """确保 FP8 权重缓存有效。仅在权重 _version 变化时重算。"""
        wv = self.weight._version
        if wv == self._weight_version and self._w_fp8_fwd is not None:
            return
        self._scale_w = compute_scale(self.weight)
        # Forward: weight.T 列主序 (K, N) fp8
        self._w_fp8_fwd = (self.weight.T.float() / self._scale_w).to(torch.float8_e4m3fn)
        # Backward grad_x: weight 列主序 (N, K) fp8
        # weight.T.contiguous().T: 先做列主序的 contiguous copy (开销大，但只需做一次)
        self._w_fp8_bwd = (self.weight.T.contiguous().T.float() / self._scale_w).to(torch.float8_e4m3fn)
        self._weight_version = wv

    def _fp8_compatible(self) -> bool:
        return self.in_features % 16 == 0 and self.out_features % 16 == 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_fp8 or not self._fp8_compatible():
            self._update_scales(x)
            target_dtype = self.weight.dtype
            bias = self.bias.to(target_dtype) if self.bias is not None else None
            return F.linear(x.to(target_dtype), self.weight, bias)

        self._ensure_cache()
        self._scale_x = compute_scale(x.detach())
        return FP8LinearFn.apply(x, self._w_fp8_fwd, self._w_fp8_bwd,
                                 self.weight, self.bias,
                                 self._scale_x, self._scale_w)

    @torch.no_grad()
    def _update_scales(self, x: torch.Tensor):
        self._scale_x = compute_scale(x.detach())
        self._scale_w = compute_scale(self.weight)
