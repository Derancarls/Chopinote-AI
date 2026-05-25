"""
分层训练入口脚本。

Phase 1: MIDI 数据预训练（屏蔽表现力 token，学习音符骨架）
Phase 2: MusicXML 数据微调（全量 token，学习完整表达）

用法:
    python scripts/run_curriculum_training.py \
        --midi-train-list data/processed/midi_train.txt \
        --musicxml-train-list data/processed/train.txt \
        --val-list data/processed/val.txt \
        --phase1-steps 50000 --phase2-steps 50000
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader
torch.set_float32_matmul_precision('high')   # TF32: 免费加速，不增加显存
torch.backends.cudnn.benchmark = True        # cuDNN autotune

from chopinote_model.config import ModelConfig, TrainingConfig, PhaseConfig, TokenLossMask
from chopinote_model.model import MusicTransformer
from chopinote_model.train import Trainer
from chopinote_model.dataset import TokenDataset, collate_fn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='分层训练 Chopinote-AI 模型')
    # 数据路径
    parser.add_argument('--midi-train-list', type=str, required=True,
                        help='Phase 1 MIDI 训练文件列表')
    parser.add_argument('--musicxml-train-list', type=str, required=True,
                        help='Phase 2 MusicXML 训练文件列表')
    parser.add_argument('--val-list', type=str, default=None,
                        help='验证集文件列表（通常使用 MusicXML 验证集）')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                        help='数据目录')
    # Phase 1
    parser.add_argument('--phase1-steps', type=int, default=250000,
                        help='Phase 1 训练步数 (default: 250000)')
    parser.add_argument('--phase1-lr', type=float, default=2e-4,
                        help='Phase 1 学习率 (default: 2e-4)')
    parser.add_argument('--phase1-warmup', type=int, default=5000,
                        help='Phase 1 warmup 步数 (default: 5000)')
    # Phase 2
    parser.add_argument('--phase2-steps', type=int, default=100000,
                        help='Phase 2 训练步数 (default: 100000)')
    parser.add_argument('--phase2-lr', type=float, default=1e-4,
                        help='Phase 2 学习率 (default: 1e-4)')
    parser.add_argument('--phase2-warmup', type=int, default=2000,
                        help='Phase 2 warmup 步数 (default: 2000)')
    # 通用
    parser.add_argument('--batch-size', type=int, default=8,
                        help='batch size per step (default: 8)')
    parser.add_argument('--compile', action='store_true', default=False,
                        help='启用 torch.compile (mode=reduce-overhead)')
    parser.add_argument('--use-fp8', action='store_true', default=False,
                        help='启用 FP8 混合精度（Blackwell tensor core 加速）')
    parser.add_argument('--fp8-warmup-steps', type=int, default=100,
                        help='FP8 前用 BF16 warmup 的步数 (default: 100)')
    parser.add_argument('--no-checkpointing', action='store_true', default=False,
                        help='关闭 gradient checkpointing 提高训练速度')
    parser.add_argument('--output-dir', type=str,
                        default='/root/autodl-tmp/chopinote/checkpoints',
                        help='checkpoint 输出目录（默认 400G 数据盘）')
    parser.add_argument('--log-dir', type=str, default='./logs',
                        help='TensorBoard 日志目录 (default: ./logs)')
    parser.add_argument('--resume', type=str, default=None,
                        help='从 checkpoint 恢复（跳过之前的 phase）')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'设备: {device}')

    # Model
    model_config = ModelConfig(gradient_checkpointing=not args.no_checkpointing)
    model = MusicTransformer(model_config).to(dtype=torch.bfloat16)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'模型参数量: {total_params:,}')

    # Phase 1: MIDI 预训练，屏蔽所有表现力 token
    phase1_mask = TokenLossMask()
    phase1 = PhaseConfig(
        name='pretrain',
        total_steps=args.phase1_steps,
        data_split_file=args.midi_train_list,
        warmup_steps=args.phase1_warmup,
        lr=args.phase1_lr,
        loss_mask=phase1_mask,
    )

    # Phase 2: MusicXML 微调，全量 token
    phase2 = PhaseConfig(
        name='finetune',
        total_steps=args.phase2_steps,
        data_split_file=args.musicxml_train_list,
        warmup_steps=args.phase2_warmup,
        lr=args.phase2_lr,
        loss_mask=None,  # 不屏蔽，全量 loss
    )

    train_config = TrainingConfig(
        batch_size=args.batch_size,
        compile=args.compile,
        use_fp8=args.use_fp8,
        fp8_warmup_steps=args.fp8_warmup_steps,
        gradient_checkpointing=not args.no_checkpointing,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        data_dir=args.data_dir,
        phases=[phase1, phase2],
    )

    logger.info('=' * 60)
    logger.info('Phase 1 (预训练):')
    logger.info(f'  数据: {phase1.data_split_file}')
    logger.info(f'  步数: {phase1.total_steps}, LR: {phase1.lr}, Warmup: {phase1.warmup_steps}')
    logger.info(f'  Loss 屏蔽: 表现力 token 全部屏蔽')
    logger.info('Phase 2 (微调):')
    logger.info(f'  数据: {phase2.data_split_file}')
    logger.info(f'  步数: {phase2.total_steps}, LR: {phase2.lr}, Warmup: {phase2.warmup_steps}')
    logger.info(f'  Loss 屏蔽: 无（全量）')
    logger.info('=' * 60)

    # Trainer
    trainer = Trainer(model, model_config, train_config, device)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    # 构建验证集 DataLoader（如果提供了 --val-list）
    val_loader = None
    if args.val_list:
        val_dataset = TokenDataset(
            split_file=args.val_list,
            data_dir=args.data_dir,
            max_seq_len=model_config.max_seq_len,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=32,  # eval 无 backward，大 batch 安全，从 ~2.5h 缩到 ~8min
            shuffle=False,
            num_workers=0,          # 0: 禁用 multiprocessing，避免 worker 连接丢失崩溃
            pin_memory=False,       # False: 避免 worker 异常退出导致 pin_memory 线程崩溃
            collate_fn=collate_fn,
            drop_last=False,
        )
        logger.info(f'验证集: {len(val_dataset)} 个样本，来自 {args.val_list}')

    trainer.train(val_dataloader=val_loader)


if __name__ == '__main__':
    main()
