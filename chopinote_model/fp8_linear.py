"""FP8 混合精度线性层 — RTX 5090 Blackwell _scaled_mm 加速。

Fresh scaling: 每步 forward 用当前输入的 scale 做 FP8 量化，立即使用不延迟，
避免 dropout 等导致分布偏移 → 延迟 scale 过小 → FP8 溢出 NaN。

FP8 e4m3fn 范围 [-448, 448]。量化流程:
  x_fp8 = (x / scale).to(fp8)  where  scale = amax / 448
  _scaled_mm(x_fp8, w_fp8, scale_x, scale_w) = (x_fp8 * scale_x) @ (w_fp8 * scale_w)^T ≈ x @ w^T
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
    """FP8 矩阵乘法 autograd Function。

    _scaled_mm(A, B, sA, sB) ≈ (A * sA) @ (B * sB)^T
    A: fp8 row-major,  B: fp8 column-major (T without contiguous).
    """

    @staticmethod
    def forward(ctx, x, weight, bias, scale_x, scale_w):
        M, K = x.reshape(-1, x.size(-1)).shape
        N = weight.size(0)

        x_fp8 = (x.reshape(-1, K).float() / scale_x).to(torch.float8_e4m3fn)
        # weight.T 保持非连续 strides = column-major layout
        w_fp8_t = (weight.T.float() / scale_w).to(torch.float8_e4m3fn)

        out = torch._scaled_mm(x_fp8, w_fp8_t, scale_x, scale_w,
                               out_dtype=torch.bfloat16)
        out = out.reshape(*x.shape[:-1], N)

        if bias is not None:
            out = out + bias.to(torch.bfloat16)

        ctx.save_for_backward(x, weight, scale_x, scale_w)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, scale_x, scale_w = ctx.saved_tensors
        M, K = x.reshape(-1, x.size(-1)).shape
        N = weight.size(0)

        go_flat = grad_out.reshape(-1, N)
        go_scale = compute_scale(go_flat)

        # grad_input = grad_out @ W     (M, N) @ (N, K) → (M, K)
        # N is always divisible by 16 (d_model/d_ff are power-of-2)
        go_fp8 = (go_flat.float() / go_scale).to(torch.float8_e4m3fn)
        w_col = (weight.t().contiguous().t().float() / scale_w).to(torch.float8_e4m3fn)
        grad_x = torch._scaled_mm(go_fp8, w_col, go_scale, scale_w,
                                  out_dtype=torch.bfloat16).to(torch.bfloat16)
        grad_x = grad_x.reshape_as(x)

        # grad_weight = x^T @ grad_out   (K, M) @ (M, N) → (K, N)
        # _scaled_mm 要求行优先矩阵尾维(M)被16整除, M=B*T 可能不满足
        x_row = x.reshape(-1, K).t().contiguous()  # (K, M)
        go_flat_gw = go_flat
        if M % 16 != 0:
            pad = 16 - M % 16
            x_row = F.pad(x_row, (0, pad))               # (K, M+pad)
            go_flat_gw = F.pad(go_flat, (0, 0, 0, pad))  # (M+pad, N)

        x_fp8 = (x_row.float() / scale_x).to(torch.float8_e4m3fn)
        go_col = (go_flat_gw.t().contiguous().t().float() / go_scale).to(torch.float8_e4m3fn)
        grad_w = torch._scaled_mm(x_fp8, go_col, scale_x, go_scale,
                                  out_dtype=torch.bfloat16).to(torch.bfloat16)
        grad_w = grad_w.t()  # (K, N).t() → (N, K)

        grad_bias = grad_out.reshape(-1, N).sum(0) if ctx.needs_input_grad[2] else None
        return grad_x, grad_w, grad_bias, None, None


class FP8Linear(nn.Module):
    """FP8 线性层，state_dict 兼容 nn.Linear（weight/bias key 完全一致）。

    use_fp8=False 时 fallback 到普通 BF16 matmul，等效 nn.Linear。
    use_fp8=True 时使用 _scaled_mm FP8 加速，每步计算 fresh scale。
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

        self._scale_x: Optional[torch.Tensor] = None
        self._scale_w: Optional[torch.Tensor] = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_fp8 or self._scale_x is None:
            self._update_scales(x)
            target_dtype = self.weight.dtype
            bias = self.bias.to(target_dtype) if self.bias is not None else None
            return F.linear(x.to(target_dtype), self.weight, bias)

        # 每步计算当前 scale 并立即使用，避免 dropout 等导致分布偏移 → 延迟 scale 过小 → FP8 溢出 NaN
        self._update_scales(x)
        return FP8LinearFn.apply(x, self.weight, self.bias, self._scale_x, self._scale_w)

    @torch.no_grad()
    def _update_scales(self, x: torch.Tensor):
        self._scale_x = compute_scale(x.detach())
        self._scale_w = compute_scale(self.weight)
