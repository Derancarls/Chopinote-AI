"""训练循环。"""
import os
import shutil
import time
import logging
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from .config import ModelConfig, TrainingConfig
from .dataset import TokenDataset, collate_fn

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
        if train_config.compile:
            self.model = torch.compile(self.model)
        self.model_config = model_config
        self.train_config = train_config
        self.device = device
        self.backup_dir = Path('../autodl-fs/chopinote/checkpoint_backups')

        # 去重参数（weight tying 会导致同一 tensor 作为多个 Parameter 被 yield）
        params = list(dict.fromkeys(model.parameters()))
        self.optimizer = AdamW(
            params,
            lr=train_config.lr,
            weight_decay=0.1,
            betas=(0.9, 0.95),
        )
        self.scaler = GradScaler(enabled=(device.type == 'cuda'))
        # 将 batch 为单位的步数转换为 optimizer-step 单位（scheduler 每 grad_accum 才 step 一次）
        opt_total = max(1, train_config.total_steps // train_config.grad_accum_steps)
        opt_warmup = max(1, train_config.warmup_steps // train_config.grad_accum_steps)
        self.scheduler = _get_scheduler(
            self.optimizer, opt_warmup, opt_total
        )

        self.global_step = 0
        self.best_loss = float('inf')
        self._last_avg_loss = float('inf')

        # TensorBoard 监控
        self.writer = SummaryWriter(log_dir=train_config.log_dir)
        logger.info(f'TensorBoard 日志目录: {train_config.log_dir}')

        # 创建输出目录
        Path(train_config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(train_config.log_dir).mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, loss: float):
        """保存 checkpoint（同时备份到 checkpoint_backups）。"""
        path = Path(self.train_config.output_dir) / f'step_{self.global_step}.pt'
        torch.save({
            'step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'loss': loss,
            'config': self.model_config,
        }, path)
        logger.info(f'Checkpoint saved: {path}')

        # 备份
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = self.backup_dir / f'step_{self.global_step}.pt'
        shutil.copy2(path, backup_path)
        logger.info(f'Backup saved: {backup_path}')

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
            # 备份 best.pt
            best_backup = self.backup_dir / 'best.pt'
            shutil.copy2(best_path, best_backup)
            logger.info(f'Best model backup saved: {best_backup}')

    def load_checkpoint(self, checkpoint_path: str):
        """恢复 checkpoint。"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.global_step = checkpoint['step']
        self.best_loss = checkpoint.get('loss', float('inf'))
        logger.info(f'Resumed from checkpoint: {checkpoint_path} (step {self.global_step})')

    def train(self, train_dataloader: DataLoader = None,
              val_dataloader: Optional[DataLoader] = None):
        """训练入口：根据 phases 配置自动选择单阶段或多阶段模式。

        单阶段（phases=None，向后兼容）：
            train(train_dataloader, val_dataloader)

        多阶段（phases 非空）：
            自动从 phase.data_split_file 创建 dataloader。
        """
        if self.train_config.phases is not None and len(self.train_config.phases) > 0:
            self._train_multiphase(val_dataloader)
        else:
            self._train_single_phase(train_dataloader, val_dataloader)

    def _train_single_phase(self, train_dataloader: DataLoader,
                            val_dataloader: Optional[DataLoader] = None):
        """单阶段训练循环（原有逻辑不变）。"""
        config = self.train_config
        model = self.model
        optimizer = self.optimizer
        scheduler = self.scheduler

        model.train()
        total_loss = 0.0
        start_time = time.time()

        logger.info(f'开始训练 | batch_size={config.batch_size} '
                    f'accum={config.grad_accum_steps} '
                    f'effective_batch={config.effective_batch_size}')

        _vocab_checked = False

        while self.global_step < config.total_steps:
            for batch in train_dataloader:
                if self.global_step >= config.total_steps:
                    break

                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                if not _vocab_checked:
                    max_id = max(input_ids.max().item(), labels.max().item() if labels.numel() else 0)
                    assert max_id < model.config.vocab_size, \
                        f'数据中存在 token ID {max_id} ≥ vocab_size {model.config.vocab_size}'
                    _vocab_checked = True

                with autocast(enabled=(self.device.type == 'cuda')):
                    logits = model(input_ids, attention_mask)
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1),
                        ignore_index=-100,
                        reduction='sum',
                    )
                    loss = loss / max(1, (labels != -100).sum())
                total_loss += loss.item()
                loss = loss / config.grad_accum_steps
                self.scaler.scale(loss).backward()
                self.global_step += 1

                if self.global_step % config.grad_accum_steps == 0:
                    total_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    self.writer.add_scalar('train/grad_norm', total_norm, self.global_step)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()

                if self.global_step % config.logging_steps == 0:
                    avg_loss = total_loss / config.logging_steps
                    self._last_avg_loss = avg_loss
                    self.writer.add_scalar('train/loss', avg_loss, self.global_step)
                    self.writer.add_scalar('train/lr', scheduler.get_last_lr()[0], self.global_step)
                    elapsed = time.time() - start_time
                    logger.info(
                        f'Step {self.global_step}/{config.total_steps} | '
                        f'Loss: {avg_loss:.4f} | '
                        f'LR: {scheduler.get_last_lr()[0]:.2e} | '
                        f'Time: {elapsed:.1f}s'
                    )
                    total_loss = 0.0

                if val_dataloader is not None and \
                   self.global_step % config.eval_steps == 0:
                    val_loss = self.evaluate(val_dataloader)
                    self.writer.add_scalar('val/loss', val_loss, self.global_step)
                    logger.info(f'Validation loss: {val_loss:.4f}')
                    model.train()

                if self.global_step % config.save_steps == 0:
                    self.save_checkpoint(self._last_avg_loss)

        self.save_checkpoint(self._last_avg_loss)
        self.writer.close()
        logger.info('训练完成')

    def _train_multiphase(self, val_dataloader: Optional[DataLoader] = None):
        """多阶段分层训练。

        按 phases 列表顺序执行，每阶段：
        - 从 phase.data_split_file 创建 DataLoader
        - 若 phase.loss_mask 非空，屏蔽对应 token 的 loss
        - 重建 optimizer/scheduler（使用阶段 LR 和 warmup）
        - 模型参数在阶段之间保留
        """
        config = self.train_config
        model = self.model

        for phase_idx, phase in enumerate(config.phases):
            logger.info(f'{"="*60}')
            logger.info(f'Phase {phase_idx+1}/{len(config.phases)}: {phase.name}')
            logger.info(f'  数据: {phase.data_split_file}')
            logger.info(f'  步数: {phase.total_steps}')
            logger.info(f'  LR: {phase.lr}')
            logger.info(f'  Loss 屏蔽: {phase.loss_mask is not None}')
            logger.info(f'{"="*60}')

            # 创建本阶段 dataloader
            dataset = TokenDataset(
                split_file=phase.data_split_file,
                data_dir=config.data_dir,
                max_seq_len=self.model_config.max_seq_len,
            )
            phase_loader = DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=False,
                collate_fn=collate_fn,
                drop_last=True,
            )

            # 预计算 loss 屏蔽 token ID 集合
            masked_ids: Optional[set] = None
            if phase.loss_mask is not None:
                from .config import TokenLossMask
                from chopinote_dataset.tokenizer import REMITokenizer
                t = REMITokenizer(grid_size=16, velocity_levels=8)
                masked_ids = phase.loss_mask.get_masked_token_ids(t)
                logger.info(f'  屏蔽 token 数: {len(masked_ids)}')

            # 重建 optimizer
            params = list(dict.fromkeys(model.parameters()))
            optimizer = AdamW(
                params, lr=phase.lr, weight_decay=0.1, betas=(0.9, 0.95),
            )
            self.scaler = GradScaler(enabled=(self.device.type == 'cuda'))

            opt_total = max(1, phase.total_steps // config.grad_accum_steps)
            opt_warmup = max(1, phase.warmup_steps // config.grad_accum_steps)
            scheduler = _get_scheduler(optimizer, opt_warmup, opt_total)

            # 阶段训练循环
            model.train()
            total_loss = 0.0
            start_time = time.time()
            phase_step = 0
            _vocab_checked = False

            while phase_step < phase.total_steps:
                for batch in phase_loader:
                    if phase_step >= phase.total_steps:
                        break

                    input_ids = batch['input_ids'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)

                    if not _vocab_checked:
                        max_id = max(input_ids.max().item(), labels.max().item() if labels.numel() else 0)
                        assert max_id < model.config.vocab_size, \
                            f'数据中存在 token ID {max_id} ≥ vocab_size {model.config.vocab_size}'
                        _vocab_checked = True

                    # 应用 loss 屏蔽
                    if masked_ids:
                        labels = self._apply_loss_mask(labels, masked_ids)

                    with autocast(enabled=(self.device.type == 'cuda')):
                        logits = model(input_ids, attention_mask)
                        loss = nn.functional.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            labels.view(-1),
                            ignore_index=-100,
                            reduction='sum',
                        )
                        loss = loss / max(1, (labels != -100).sum())
                    total_loss += loss.item()
                    loss = loss / config.grad_accum_steps
                    self.scaler.scale(loss).backward()
                    self.global_step += 1
                    phase_step += 1

                    if phase_step % config.grad_accum_steps == 0:
                        total_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        self.writer.add_scalar('train/grad_norm', total_norm, self.global_step)
                        self.scaler.step(optimizer)
                        self.scaler.update()
                        scheduler.step()
                        optimizer.zero_grad()

                    if phase_step % config.logging_steps == 0:
                        avg_loss = total_loss / config.logging_steps
                        self._last_avg_loss = avg_loss
                        self.writer.add_scalar('train/loss', avg_loss, self.global_step)
                        self.writer.add_scalar('train/lr', scheduler.get_last_lr()[0], self.global_step)
                        elapsed = time.time() - start_time
                        logger.info(
                            f'[{phase.name}] Step {phase_step}/{phase.total_steps} '
                            f'(global {self.global_step}) | '
                            f'Loss: {avg_loss:.4f} | '
                            f'LR: {scheduler.get_last_lr()[0]:.2e} | '
                            f'Time: {elapsed:.1f}s'
                        )
                        total_loss = 0.0

                    if val_dataloader is not None and \
                       phase_step % phase.eval_steps == 0:
                        val_loss = self.evaluate(val_dataloader)
                        self.writer.add_scalar('val/loss', val_loss, self.global_step)
                        logger.info(f'[{phase.name}] Validation loss: {val_loss:.4f}')
                        model.train()

                    if phase_step % phase.save_steps == 0:
                        self.save_checkpoint(self._last_avg_loss)

            # 阶段终了保存
            self.save_checkpoint(self._last_avg_loss)
            logger.info(f'Phase {phase.name} 完成 (global_step={self.global_step})')

        self.writer.close()
        logger.info('多阶段训练完成')

    @staticmethod
    def _apply_loss_mask(labels: torch.Tensor, masked_ids: set) -> torch.Tensor:
        """向量化 loss 屏蔽：将 labels 中属于 masked_ids 的 token 设为 -100。"""
        if not masked_ids:
            return labels
        mask_tensor = torch.tensor(list(masked_ids), device=labels.device)
        is_masked = torch.isin(labels, mask_tensor)
        labels = labels.clone()
        labels[is_masked] = -100
        return labels

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

            with autocast(enabled=(self.device.type == 'cuda')):
                logits = self.model(input_ids, attention_mask)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
                reduction='sum',
            )
            loss = loss / max(1, (labels != -100).sum())
            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(1, num_batches)
