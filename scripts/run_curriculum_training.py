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

from chopinote_model.config import ModelConfig, TrainingConfig, PhaseConfig, TokenLossMask
from chopinote_model.model import MusicTransformer
from chopinote_model.train import Trainer

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
    parser.add_argument('--batch-size', type=int, default=4,
                        help='batch size (default: 4)')
    parser.add_argument('--output-dir', type=str,
                        default='../autodl-fs/chopinote/checkpoints',
                        help='checkpoint 输出目录')
    parser.add_argument('--resume', type=str, default=None,
                        help='从 checkpoint 恢复（跳过之前的 phase）')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'设备: {device}')

    # Model
    model_config = ModelConfig()
    model = MusicTransformer(model_config)
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
        output_dir=args.output_dir,
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

    # 多阶段训练（train() 自动检测 phases 配置）
    trainer.train(val_dataloader=None)
    # Note: val_dataloader=None，验证集在实际训练中使用 --val-list 指定
    # 如需验证，可先创建 val_loader 传入


if __name__ == '__main__':
    main()
