"""模型超参和训练配置。"""
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Decoder-only Transformer 超参数。"""
    vocab_size: int = 236
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1024
    max_seq_len: int = 2048
    dropout: float = 0.1
    pad_token_id: int = 0

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"


@dataclass
class TrainingConfig:
    """训练配置（已适配 GTX 750 Ti 4GB VRAM）。"""
    batch_size: int = 2
    grad_accum_steps: int = 4
    lr: float = 3e-4
    warmup_steps: int = 1000
    total_steps: int = 50000
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    output_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    data_dir: str = "data/processed"

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum_steps
