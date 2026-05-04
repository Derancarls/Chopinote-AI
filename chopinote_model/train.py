"""训练循环。"""
import os
import time
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .config import ModelConfig, TrainingConfig

logger = logging.getLogger(__name__)


def _get_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup + cosine decay scheduler."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())
    return LambdaLR(optimizer, lr_lambda)


class Trainer:
    """模型训练器。"""

    def __init__(self, model: nn.Module, model_config: ModelConfig,
                 train_config: TrainingConfig, device: torch.device):
        self.model = model.to(device)
        self.model_config = model_config
        self.train_config = train_config
        self.device = device

        self.optimizer = AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=0.1,
            betas=(0.9, 0.95),
        )
        # 将 batch 为单位的步数转换为 optimizer-step 单位（scheduler 每 grad_accum 才 step 一次）
        opt_total = max(1, train_config.total_steps // train_config.grad_accum_steps)
        opt_warmup = max(1, train_config.warmup_steps // train_config.grad_accum_steps)
        self.scheduler = _get_scheduler(
            self.optimizer, opt_warmup, opt_total
        )

        self.global_step = 0
        self.best_loss = float('inf')
        self._last_avg_loss = float('inf')

        # 创建输出目录
        Path(train_config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(train_config.log_dir).mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, loss: float):
        """保存 checkpoint。"""
        path = Path(self.train_config.output_dir) / f'step_{self.global_step}.pt'
        torch.save({
            'step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.model_config,
        }, path)
        logger.info(f'Checkpoint saved: {path}')

        if loss < self.best_loss:
            self.best_loss = loss
            best_path = Path(self.train_config.output_dir) / 'best.pt'
            torch.save({
                'step': self.global_step,
                'model_state_dict': self.model.state_dict(),
                'loss': loss,
                'config': self.model_config,
            }, best_path)
            logger.info(f'Best model saved: {best_path}')

    def load_checkpoint(self, checkpoint_path: str):
        """恢复 checkpoint。"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['step']
        self.best_loss = checkpoint.get('loss', float('inf'))
        logger.info(f'Resumed from checkpoint: {checkpoint_path} (step {self.global_step})')

    def train(self, train_dataloader: DataLoader,
              val_dataloader: Optional[DataLoader] = None):
        """运行训练循环。"""
        config = self.train_config
        model = self.model
        optimizer = self.optimizer
        scheduler = self.scheduler

        model.train()
        total_loss = 0.0
        logging_loss = 0.0
        start_time = time.time()

        logger.info(f'开始训练 | batch_size={config.batch_size} '
                    f'accum={config.grad_accum_steps} '
                    f'effective_batch={config.effective_batch_size}')

        while self.global_step < config.total_steps:
            for batch in train_dataloader:
                if self.global_step >= config.total_steps:
                    break

                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                logits = model(input_ids, attention_mask)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                    reduction='mean',
                )
                # 先记录未除 grad_accum 的真实 loss，再做归一化 backward
                total_loss += loss.item()
                loss = loss / config.grad_accum_steps
                loss.backward()
                self.global_step += 1

                if self.global_step % config.grad_accum_steps == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                # logging
                if self.global_step % config.logging_steps == 0:
                    avg_loss = total_loss / config.logging_steps
                    self._last_avg_loss = avg_loss
                    elapsed = time.time() - start_time
                    logger.info(
                        f'Step {self.global_step}/{config.total_steps} | '
                        f'Loss: {avg_loss:.4f} | '
                        f'LR: {scheduler.get_last_lr()[0]:.2e} | '
                        f'Time: {elapsed:.1f}s'
                    )
                    total_loss = 0.0

                # 验证
                if val_dataloader is not None and \
                   self.global_step % config.eval_steps == 0:
                    val_loss = self.evaluate(val_dataloader)
                    logger.info(f'Validation loss: {val_loss:.4f}')
                    model.train()

                # 保存
                if self.global_step % config.save_steps == 0:
                    self.save_checkpoint(self._last_avg_loss)

        # 最终保存
        self.save_checkpoint(self._last_avg_loss)
        logger.info('训练完成')

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        """评估验证集 loss。"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            logits = self.model(input_ids, attention_mask)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
                reduction='mean',
            )
            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(1, num_batches)
