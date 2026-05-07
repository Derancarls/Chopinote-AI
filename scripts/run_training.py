"""
训练入口脚本。

用法:
    python scripts/run_training.py                          # 默认配置
    python scripts/run_training.py --resume ../autodl-fs/chopinote/checkpoints/step_1000.pt
    python scripts/run_training.py --total-steps 100000
"""
import argparse
import logging
import sys
from pathlib import Path

# 添加项目根目录到 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from chopinote_model.config import ModelConfig, TrainingConfig
from chopinote_model.model import MusicTransformer
from chopinote_model.dataset import create_dataloader
from chopinote_model.train import Trainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='训练 Chopinote-AI 模型')
    parser.add_argument('--resume', type=str, default=None,
                        help='从 checkpoint 恢复训练')
    parser.add_argument('--total-steps', type=int, default=None,
                        help='覆盖 TrainingConfig.total_steps')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='覆盖 batch_size')
    parser.add_argument('--lr', type=float, default=None,
                        help='覆盖学习率')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                        help='数据目录')
    parser.add_argument('--output-dir', type=str, default='../autodl-fs/chopinote/checkpoints',
                        help='checkpoint 输出目录')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'设备: {device}')

    model_config = ModelConfig()
    train_config = TrainingConfig(
        output_dir=args.output_dir,
        data_dir=args.data_dir,
    )

    # CLI 参数覆盖
    if args.total_steps:
        train_config.total_steps = args.total_steps
    if args.batch_size:
        train_config.batch_size = args.batch_size
    if args.lr:
        train_config.lr = args.lr

    logger.info(f'Model config: vocab_size={model_config.vocab_size}, '
                f'd_model={model_config.d_model}, n_layers={model_config.n_layers}')
    logger.info(f'Train config: batch_size={train_config.batch_size}, '
                f'accum={train_config.grad_accum_steps}, '
                f'total_steps={train_config.total_steps}')

    # 数据
    train_loader = create_dataloader(
        str(Path(train_config.data_dir) / 'train.txt'),
        data_dir=train_config.data_dir,
        batch_size=train_config.batch_size,
        max_seq_len=model_config.max_seq_len,
        shuffle=True,
    )
    val_loader = create_dataloader(
        str(Path(train_config.data_dir) / 'val.txt'),
        data_dir=train_config.data_dir,
        batch_size=train_config.batch_size,
        max_seq_len=model_config.max_seq_len,
        shuffle=False,
    )

    # 模型
    model = MusicTransformer(model_config)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'模型参数量: {total_params:,}')

    # Trainer
    trainer = Trainer(model, model_config, train_config, device)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
