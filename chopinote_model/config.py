"""模型超参和训练配置。"""
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Decoder-only Transformer 超参数（适配 RTX 4090 24GB）。"""
    vocab_size: int = 831
    d_model: int = 768
    n_layers: int = 10
    n_heads: int = 12
    d_ff: int = 3072
    max_seq_len: int = 4096
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
    """训练配置（适配 RTX 4090 24GB）。"""
    batch_size: int = 8
    grad_accum_steps: int = 4
    lr: float = 2e-4
    warmup_steps: int = 2000
    total_steps: int = 100000
    compile: bool = False
    logging_steps: int = 10
    save_steps: int = 1000
    eval_steps: int = 1000
    output_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    data_dir: str = "data/processed"

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum_steps
