"""模型超参和训练配置。"""
import os
from dataclasses import dataclass, field
from typing import List, Optional, Set

# ── 段落感知常量 ────────────────────────────────────────────
# section_id=0 / section_type=0 统一表示"无段落"
# 所有模块必须引用此常量，禁止硬编码 0
NO_SECTION_ID: int = 0
NO_SECTION_TYPE_ID: int = 0


@dataclass
class ModelConfig:
    """Decoder-only Transformer 超参数（适配 RTX 5090 32GB）。"""
    vocab_size: int = 542                     # v0.3.0: 929 → 542
    d_model: int = 2048
    n_layers: int = 24
    n_heads: int = 32
    d_ff: int = 8192
    max_seq_len: int = 4096
    dropout: float = 0.15
    pad_token_id: int = 0
    bar_token_id: int = 4
    max_measures: int = 256
    rope_theta: float = 10000.0
    gradient_checkpointing: bool = True

    # --- 段落感知（paragraph-aware） ---
    use_section_attention: bool = True
    n_section_types: int = 22
    n_section_bars_classes: int = 128
    max_sections: int = 64
    sec_bias_decay_len: int = 16
    sec_bias_alpha_init: float = 0.5
    sec_bias_beta_init: float = 0.15
    sec_bias_gamma_init: float = 0.05
    sec_bias_delta_init: float = 0.2
    sec_loss_weight: float = 0.03
    sec_bars_loss_weight: float = 0.0
    sec_bias_param_max: float = 2.0

    # --- SSF (Sliding Scale Field) 调性编码 ---
    use_ssf: bool = True
    ssf_dim: int = 12
    use_ssf_reconstruction: bool = True
    ssf_loss_weight: float = 0.1
    ssf_tonic_weight: float = 1.0
    ssf_local_weight: float = 0.5

    # --- 声部感知（voice-aware） ---
    use_voice_count: bool = True
    n_voices: int = 16
    # v0.3.0 新增: per-voice identity embedding
    use_voice_identity: bool = True
    n_voice_ids: int = 5                    # 0=structural, 1-4=Voice 0-3 (SATB)
    use_voice_bias: bool = True
    voice_same_init: float = 0.3            # 同声部历史吸引
    voice_samepos_init: float = 0.1         # 同位置跨声部协调

    # --- 织体感知（figuration） ---
    use_figuration: bool = True
    n_fig_types: int = 12

    # --- 终止式感知（cadence） ---
    use_cadence: bool = True
    n_cadence_types: int = 6                # 0=none, 1-5=PAC/IAC/HC/DC/PC

    # --- 时值饱和度（DurSat） ---
    use_dur_sat: bool = True
    dur_sat_buckets: int = 17               # 0/16 ~ 16/16

    # --- QK-Norm（稳定注意力） ---
    use_qk_norm: bool = True
    use_head_scale: bool = True

    # --- 节内位置感知 ---
    use_measure_in_section: bool = True
    max_measures_in_section: int = 32

    # --- Attention 软上限 ---
    attn_logit_cap: float = 50.0

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"

    @classmethod
    def from_gpu(cls, profile=None) -> 'ModelConfig':
        """根据 GPU 能力返回调优后的 ModelConfig（适用于训练场景）。

        Args:
            profile: SystemProfile，为 None 时自动检测。

        Returns:
            ModelConfig 实例，架构字段保持默认值（1.21B），
            仅调优 gradient_checkpointing / use_fp8 等训练相关字段。
        """
        from chopinote_model.auto_config import detect_system, suggest_training

        if profile is None:
            profile = detect_system()
        hints = suggest_training(profile)

        cfg = cls(
            gradient_checkpointing=hints.suggested_gradient_checkpointing,
        )
        # 注入额外 hint 供 TrainingConfig 参考
        cfg._suggested_batch_size = hints.suggested_batch_size
        cfg._suggested_fp8 = hints.suggested_fp8
        cfg._suggested_memory_fraction = hints.suggested_memory_fraction
        return cfg


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
        }
        masked_ids: Set[int] = set()
        for attr, prefix in prefix_map.items():
            if getattr(self, attr):
                for token_str, token_id in tokenizer._token_to_id.items():
                    if token_str.startswith(prefix):
                        masked_ids.add(token_id)
        # mask_tuplet 控制 TupletStart 和 TupletEnd 两个 token
        if self.mask_tuplet:
            masked_ids.add(tokenizer.encode_token(REMITokenizer.TUPLET_END))
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
    max_eval_batches: int = 50


@dataclass
class TrainingConfig:
    """训练配置（适配 RTX 5090 32GB / 1.24B 模型）。"""
    batch_size: int = 32
    grad_accum_steps: int = 1
    # 以下字段仅在单阶段模式 (phases=None) 下生效
    # 多阶段模式请通过 PhaseConfig 分别设置各阶段 lr / warmup / steps
    lr: float = 1.5e-4
    warmup_steps: int = 4000
    total_steps: int = 100000
    compile: bool = False
    use_fp8: bool = True                       # FP8 Linear 加速（BF16 warmup 后自动切换）
    fp8_warmup_steps: int = 500                # BF16 warmup 步数后切换 FP8（恢复训练时 global_step >= warmup 立即切换）
    gradient_checkpointing: bool = True  # False = 关闭 checkpointing 提速（耗更多 VRAM）
    aux_head_lr_mult: float = 0.5        # section_head / ssf_* LR 乘数
    attn_bias_lr_mult: float = 0.1       # sec_bias_* / voice_same* 标量参数 LR 乘数
    logging_steps: int = 10
    save_steps: int = 1000
    eval_steps: int = 1000
    max_eval_batches: int = 100  # 限制验证批次数，100 batch × 8 = 800 样本，~6min
    # ── Token 级 loss 加权 ──
    position_token_loss_weight: float = 2.0    # Position token 预测 loss 乘数（提权强调换位）
    repetition_penalty: float = 1.2            # 连续 ≥4 同类型 token 的 loss 乘数
    max_notes_per_position: int = 8            # 训练时同 Position 音符数 > 此值 → mask 该 bar
    # ── 训练稳定化 ──
    z_loss_weight: float = 1e-4                # Z-loss 权重（压制 logit 漂移），0=禁用
    ema_beta: float = 0.999                    # 权重 EMA 衰减率，0=禁用
    # ── Dropout 阶梯衰减 ──
    dropout_schedule: dict = field(default_factory=lambda: {
        0: 0.15,      # 初始
        47000: 0.10,  # 当前恢复点
        80000: 0.08,  # Phase 1 中期
    })
    # ── DPO 自动微调（C 进化层） ──
    dpo_enabled: bool = False              # 是否启用自动 DPO
    dpo_interval_steps: int = 0            # 每多少训练步检查一次 reward_log，0=不触发
    dpo_min_new_entries: int = 20          # 至少新增多少条 reward 记录才触发
    dpo_epochs: int = 3                    # 每次 DPO 训练 epoch 数
    dpo_beta: float = 0.1                  # DPO beta
    dpo_lora_rank: int = 8                 # LoRA rank
    dpo_min_score_gap: float = 0.15        # 偏好对最小分差
    dpo_reward_dir: str = field(default_factory=lambda: os.environ.get(
        'CHOPINOTE_REWARD_DIR', '/root/autodl-tmp/chopinote/rewards'))
    # ── 自动评估生成（填充 reward_log，为 DPO 提供偏好数据） ──
    eval_enabled: bool = False               # 是否在 checkpoint 后自动跑评估生成
    eval_interval_steps: int = 5000          # 每多少步跑一次评估生成
    eval_seed_list: str = ""                 # 种子文件列表（一行一个 MusicXML 路径）
    eval_samples_per_seed: int = 2           # 每种子生成几个变体（不同温度）
    eval_max_bars: int = 48                  # 生成长度
    eval_temperatures: str = "0.9,1.1"       # 温度档位（逗号分隔）
    output_dir: str = field(default_factory=lambda: os.environ.get(
        'CHOPINOTE_OUTPUT_DIR', '/root/autodl-tmp/chopinote/checkpoints'))
    log_dir: str = field(default_factory=lambda: os.environ.get(
        'CHOPINOTE_LOG_DIR', './logs'))
    data_dir: str = field(default_factory=lambda: os.environ.get(
        'CHOPINOTE_DATA_DIR', 'data/processed'))
    phases: Optional[List[PhaseConfig]] = None  # None = 单阶段模式，向后兼容

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum_steps
