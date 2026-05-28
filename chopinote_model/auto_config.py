"""硬件自动检测与最优推理/训练配置建议。

用法:
    from chopinote_model.auto_config import detect_system, suggest_inference, print_hardware_report

    profile = detect_system()
    print_hardware_report(profile)
    cfg = suggest_inference(profile)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class CpuProfile:
    """CPU 能力检测结果。"""
    core_count: int
    thread_count: int


@dataclass
class GpuProfile:
    """GPU 能力检测结果。"""
    name: str
    vram_gb: float
    compute_capability: tuple[int, int]  # (major, minor)
    supports_bf16: bool
    supports_fp8: bool
    has_tensor_cores: bool
    multi_gpu: bool


@dataclass
class SystemProfile:
    """完整的系统硬件画像。"""
    cpu: CpuProfile
    gpu: Optional[GpuProfile]


@dataclass
class InferenceConfig:
    """推荐的推理最优配置。"""
    dtype: str               # 'fp8' | 'bf16' | 'fp32'
    torch_compile: bool      # 启用 torch.compile
    num_threads: int         # CPU 线程数
    memory_fraction: float   # GPU 显存上限比例 (0~1)
    use_tf32: bool           # TF32 matmul (Ampere+)


@dataclass
class TrainingHints:
    """推荐的训练超参（供 TrainingConfig 参考，不直接覆盖）。"""
    suggested_batch_size: int
    suggested_fp8: bool
    suggested_gradient_checkpointing: bool
    suggested_memory_fraction: float


# ── 检测函数 ──────────────────────────────────────────────────


def detect_cpu() -> CpuProfile:
    """检测 CPU 信息。"""
    thread_count = os.cpu_count() or 4
    # 尝试通过 psutil 获取物理核心数，失败时保守使用 logical/2 或直接使用 logical
    try:
        import psutil
        core_count = psutil.cpu_count(logical=False) or max(2, thread_count // 2)
    except ImportError:
        core_count = max(2, thread_count // 2)
    return CpuProfile(core_count=core_count, thread_count=thread_count)


def detect_gpu() -> Optional[GpuProfile]:
    """检测 GPU 信息。无 GPU 时返回 None。"""
    try:
        import torch
    except ImportError:
        return None

    if not torch.cuda.is_available():
        return None

    props = torch.cuda.get_device_properties(0)
    cc = (props.major, props.minor)
    name = props.name
    vram_gb = props.total_memory / (1024**3)
    multi_gpu = torch.cuda.device_count() > 1

    # BF16: Volta (7.0) 引入 Tensor Cores 且支持 bf16
    supports_bf16 = cc >= (7, 0)
    # FP8: Blackwell (10.0)
    supports_fp8 = cc >= (10, 0)
    # Tensor Cores: Volta+
    has_tensor_cores = cc >= (7, 0)

    return GpuProfile(
        name=name, vram_gb=vram_gb, compute_capability=cc,
        supports_bf16=supports_bf16, supports_fp8=supports_fp8,
        has_tensor_cores=has_tensor_cores, multi_gpu=multi_gpu,
    )


def detect_system() -> SystemProfile:
    """检测完整的系统硬件配置。"""
    return SystemProfile(cpu=detect_cpu(), gpu=detect_gpu())


# ── 推荐配置 ──────────────────────────────────────────────────


def suggest_inference(profile: SystemProfile) -> InferenceConfig:
    """根据硬件画像返回最优推理配置。"""
    # PyTorch 线程数：超过 16 线程收益递减，保守取 min(16, cpu_count-1)
    num_threads = max(1, min(16, profile.cpu.thread_count - 1))

    if profile.gpu is None:
        return InferenceConfig(
            dtype='fp32', torch_compile=False,
            num_threads=num_threads,
            memory_fraction=1.0, use_tf32=False,
        )

    gpu = profile.gpu

    # ── 精度选择 ──
    if gpu.supports_fp8 and gpu.vram_gb >= 24:
        # Blackwell+ 且显存充足 → FP8 推理（set_fp8_mode）
        dtype = 'fp8'
    elif gpu.supports_bf16:
        # Volta+ → BF16
        dtype = 'bf16'
    else:
        dtype = 'fp32'

    # ── torch.compile ──
    # Ampere (8.0)+ 有较好的 compile 支持
    torch_compile = gpu.compute_capability >= (8, 0)

    # ── 显存上限 ──
    if gpu.vram_gb >= 24:
        memory_fraction = 0.85
    elif gpu.vram_gb >= 8:
        memory_fraction = 0.80
    else:
        memory_fraction = 0.70

    # ── TF32 ──
    use_tf32 = gpu.compute_capability >= (8, 0)

    return InferenceConfig(
        dtype=dtype, torch_compile=torch_compile,
        num_threads=num_threads,
        memory_fraction=memory_fraction,
        use_tf32=use_tf32,
    )


def suggest_training(profile: SystemProfile) -> TrainingHints:
    """根据硬件画像推荐训练超参。"""
    if profile.gpu is None:
        return TrainingHints(
            suggested_batch_size=2,
            suggested_fp8=False,
            suggested_gradient_checkpointing=True,
            suggested_memory_fraction=1.0,
        )

    gpu = profile.gpu

    # ── Batch size ──
    # 1.21B 模型 bf16+gc 约占用 16-20GB
    if gpu.vram_gb >= 48:
        batch_size = 16
    elif gpu.vram_gb >= 32:
        batch_size = 8
    elif gpu.vram_gb >= 24:
        batch_size = 4
    elif gpu.vram_gb >= 16:
        batch_size = 2
    else:
        batch_size = 1

    # ── FP8 ──
    fp8 = gpu.supports_fp8 and gpu.vram_gb >= 24

    # ── Gradient checkpointing ──
    # 显存 < 32GB 时开启 checkpointing 省显存，>= 32GB 可以关掉换速度
    gradient_checkpointing = gpu.vram_gb < 32

    # ── Memory fraction ──
    if gpu.vram_gb >= 24:
        mem_frac = 0.85
    elif gpu.vram_gb >= 16:
        mem_frac = 0.80
    else:
        mem_frac = 0.75

    return TrainingHints(
        suggested_batch_size=batch_size,
        suggested_fp8=fp8,
        suggested_gradient_checkpointing=gradient_checkpointing,
        suggested_memory_fraction=mem_frac,
    )


# ── 输出 ──────────────────────────────────────────────────────


def _gpu_cc_str(cc: tuple[int, int]) -> str:
    return f'{cc[0]}.{cc[1]}'


def _format_vram(gb: float) -> str:
    if gb >= 24:
        return f'{gb:.0f} GB'
    return f'{gb:.1f} GB'


def print_hardware_report(profile: SystemProfile, inference_cfg: Optional[InferenceConfig] = None):
    """打印用户友好的硬件报告。"""
    cpu = profile.cpu
    print(f'  CPU: {cpu.thread_count} 线程 / {cpu.core_count} 核')

    if profile.gpu:
        gpu = profile.gpu
        caps = []
        if gpu.supports_bf16: caps.append('BF16')
        if gpu.supports_fp8: caps.append('FP8')
        if gpu.has_tensor_cores: caps.append('TensorCore')
        cap_str = f' [{", ".join(caps)}]' if caps else ''

        print(f'  GPU: {gpu.name} {_format_vram(gpu.vram_gb)}'
              f' (CC {_gpu_cc_str(gpu.compute_capability)}){cap_str}')
        if gpu.multi_gpu:
            print(f'       Multi-GPU: 可用')
    else:
        print('  GPU: (无)')

    if inference_cfg:
        print(f'  → 推理: dtype={inference_cfg.dtype}'
              f' | 线程={inference_cfg.num_threads}'
              f' | TF32={"on" if inference_cfg.use_tf32 else "off"}'
              f' | compile={"on" if inference_cfg.torch_compile else "off"}'
              f' | 显存上限={inference_cfg.memory_fraction:.0%}')
