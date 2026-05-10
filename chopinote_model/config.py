"""模型超参和训练配置。"""
from dataclasses import dataclass, field
from typing import List, Optional, Set


@dataclass
class ModelConfig:
    """Decoder-only Transformer 超参数（适配 RTX 4090 24GB）。"""
    vocab_size: int = 872
    d_model: int = 1024
    n_layers: int = 12
    n_heads: int = 16
    d_ff: int = 4096
    max_seq_len: int = 4096
    dropout: float = 0.1
    pad_token_id: int = 0
    bar_token_id: int = 4
    max_measures: int = 256

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"


@dataclass
class TokenLossMask:
    """定义哪些 token 类别在 loss 计算中被屏蔽（ignore_index=-100）。

    用于 MIDI 预训练阶段：屏蔽 MIDI 中不存在的表现力标记，
    使模型专注于学习音符骨架（音高/节奏/和声/声部）。
    """
    mask_clef: bool = True
    mask_dynamic: bool = True
    mask_hairpin: bool = True
    mask_artic: bool = True
    mask_ornament: bool = True
    mask_pedal: bool = True
    mask_slur: bool = True
    mask_octave: bool = True
    mask_arpeggio: bool = True
    mask_grace_note: bool = True
    mask_repeat: bool = True
    mask_jump: bool = True
    mask_tuplet: bool = True
    mask_bass: bool = False
    mask_anticipate: bool = False

    def get_masked_token_ids(self, tokenizer) -> Set[int]:
        """预计算需要屏蔽的 token ID 集合。"""
        from chopinote_dataset.tokenizer import REMITokenizer
        prefix_map = {
            'mask_clef': REMITokenizer.CLEF,
            'mask_dynamic': REMITokenizer.DYNAMIC,
            'mask_hairpin': REMITokenizer.HAIRPIN,
            'mask_artic': REMITokenizer.ARTIC,
            'mask_ornament': REMITokenizer.ORNAMENT,
            'mask_pedal': REMITokenizer.PEDAL,
            'mask_slur': REMITokenizer.SLUR,
            'mask_octave': REMITokenizer.OCTAVE,
            'mask_arpeggio': REMITokenizer.ARPEGGIO,
            'mask_grace_note': REMITokenizer.GRACE_NOTE,
            'mask_repeat': REMITokenizer.REPEAT,
            'mask_jump': REMITokenizer.JUMP,
            'mask_tuplet': REMITokenizer.TUPLET_START,
            'mask_bass': REMITokenizer.BASS,
            'mask_anticipate': REMITokenizer.ANTICIPATE,
        }
        masked_ids: Set[int] = set()
        for attr, prefix in prefix_map.items():
            if getattr(self, attr):
                for token_str, token_id in tokenizer._token_to_id.items():
                    if token_str.startswith(prefix) or token_str == REMITokenizer.TUPLET_END:
                        masked_ids.add(token_id)
        return masked_ids


@dataclass
class PhaseConfig:
    """单个分层训练阶段的配置。"""
    name: str                                        # 阶段名，如 "pretrain", "finetune"
    total_steps: int                                 # 本阶段的 optimizer 步数
    data_split_file: str                             # 数据文件列表路径
    warmup_steps: int = 2000
    lr: float = 2e-4
    loss_mask: Optional[TokenLossMask] = None        # None = 不屏蔽，全量 loss
    save_steps: int = 1000
    eval_steps: int = 1000


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
    output_dir: str = "../autodl-fs/chopinote/checkpoints"
    log_dir: str = "./logs"
    data_dir: str = "data/processed"
    phases: Optional[List[PhaseConfig]] = None  # None = 单阶段模式，向后兼容

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum_steps
