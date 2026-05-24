"""训练循环。"""
import os
import itertools
import shutil
import time
import logging
import threading
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter

from .config import ModelConfig, TrainingConfig
from .dataset import TokenDataset, collate_fn

try:
    from liger_kernel.transformers import LigerCrossEntropyLoss
    _ce_loss = LigerCrossEntropyLoss(ignore_index=-100, reduction='sum')
    _LIGER_AVAILABLE = True
except ImportError:
    _LIGER_AVAILABLE = False

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
        self.backup_dir = Path(train_config.output_dir) / 'backups'
        self._token_id_to_type, self._type_names = self._build_token_type_map()

        # 去重参数（weight tying 会导致同一 tensor 作为多个 Parameter 被 yield）
        params = list(dict.fromkeys(model.parameters()))
        try:
            self.optimizer = AdamW(
                params, lr=train_config.lr, weight_decay=0.1,
                betas=(0.9, 0.95), fused=True,
            )
        except (RuntimeError, TypeError):
            self.optimizer = AdamW(
                params, lr=train_config.lr, weight_decay=0.1,
                betas=(0.9, 0.95),
            )
        self.scheduler = _get_scheduler(
            self.optimizer, train_config.warmup_steps, train_config.total_steps
        )

        self.global_step = 0
        self.best_loss = float('inf')
        self._last_avg_loss = float('inf')
        self._save_thread: Optional[threading.Thread] = None

        # TensorBoard 监控
        self.writer = SummaryWriter(log_dir=train_config.log_dir)
        logger.info(f'TensorBoard 日志目录: {train_config.log_dir}')

        # 创建输出目录
        Path(train_config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(train_config.log_dir).mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, loss: float):
        """异步保存 checkpoint + cleanup（上个线程确保完成）。"""
        # 等待上一个保存线程完成（避免并发写 best.pt / 旧的 step 文件）
        self._join_save_thread()

        output_dir = Path(self.train_config.output_dir)

        # ── 将数据 copy 到 CPU，避免 GPU→CPU 迁移阻塞训练 ───
        model_cpu = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        opt_cpu = {k: v.detach().cpu().clone() if hasattr(v, 'detach') else v
                   for k, v in self.optimizer.state_dict().items()}
        sched_cpu = {k: v if not hasattr(v, 'detach') else v for k, v in self.scheduler.state_dict().items()}
        step = self.global_step
        is_best = loss < self.best_loss
        if is_best:
            self.best_loss = loss
        backup_dir = str(self.backup_dir)

        def _do_save():
            _output = Path(output_dir)
            step_path = _output / f'step_{step}.pt'
            step_state = {
                'step': step,
                'model_state_dict': model_cpu,
                'optimizer_state_dict': opt_cpu,
                'scheduler_state_dict': sched_cpu,
                'loss': loss,
                'config': self.model_config,
            }
            torch.save(step_state, step_path)
            logger.info(f'Checkpoint saved: {step_path}')

            # 备份
            _backup = Path(backup_dir)
            _backup.mkdir(parents=True, exist_ok=True)
            backup_path = _backup / f'step_{step}.pt'
            shutil.copy2(step_path, backup_path)
            logger.info(f'Backup saved: {backup_path}')

            # 清理旧文件（保留最新 2 个）
            step_files = sorted(
                _output.glob('step_*.pt'),
                key=lambda p: int(p.stem.split('_')[1]),
            )
            for f in step_files[:-2]:
                f.unlink()
                logger.info(f'Deleted old checkpoint: {f}')
                b = _backup / f.name
                if b.exists():
                    b.unlink()
                    logger.info(f'Deleted old backup: {b}')

            # ── best checkpoint（原子写入）─────────────────
            if is_best:
                tmp_path = _output / 'best.tmp'
                best_path = _output / 'best.pt'
                best_state = {
                    'step': step,
                    'model_state_dict': model_cpu,
                    'loss': loss,
                    'config': self.model_config,
                }
                torch.save(best_state, tmp_path)
                tmp_path.rename(best_path)
                logger.info(f'Best model saved: {best_path}')
                best_backup = _backup / 'best.pt'
                shutil.copy2(best_path, best_backup)
                logger.info(f'Best model backup saved: {best_backup}')

        self._save_thread = threading.Thread(target=_do_save, daemon=True)
        self._save_thread.start()

    def _join_save_thread(self):
        """等待上一个异步保存完成。"""
        if self._save_thread is not None:
            self._save_thread.join()
            self._save_thread = None

    def _cleanup_old_checkpoints(self, keep_latest: int = 2):
        """保留最新的 keep_latest 个 step checkpoint，删除更早的（含备份）。"""
        output_dir = Path(self.train_config.output_dir)
        step_files = sorted(
            output_dir.glob('step_*.pt'),
            key=lambda p: int(p.stem.split('_')[1]),
        )
        for f in step_files[:-keep_latest]:
            f.unlink()
            logger.info(f'Deleted old checkpoint: {f}')
            backup = self.backup_dir / f.name
            if backup.exists():
                backup.unlink()
                logger.info(f'Deleted old backup: {backup}')

    def load_checkpoint(self, checkpoint_path: str):
        """恢复 checkpoint（支持 vocab/参数量变化，自动跳过 shape 不匹配的权重）。"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint['model_state_dict']
        model_state = self.model.state_dict()

        loaded = 0
        skipped = []
        for k, v in state_dict.items():
            if k in model_state and v.shape == model_state[k].shape:
                model_state[k] = v
                loaded += 1
            else:
                skipped.append(k)

        # ── Embedding 行拷贝（处理 vocab 扩展，复用旧行） ──────
        for key in ('token_embedding.weight', 'lm_head.weight'):
            if key in state_dict and key in model_state:
                old_emb = state_dict[key]
                new_emb = model_state[key]
                if old_emb.shape != new_emb.shape:
                    copy_rows = min(old_emb.shape[0], new_emb.shape[0])
                    new_emb[:copy_rows] = old_emb[:copy_rows]
                    logger.info(f'Embedding {key}: 复用 {copy_rows}/{new_emb.shape[0]} 行来自 checkpoint')
        # ─────────────────────────────────────────────────────

        self.model.load_state_dict(model_state)

        if skipped:
            logger.warning(f'Checkpoint 加载: {loaded} 个参数加载成功, '
                           f'{len(skipped)} 个 shape 不匹配已跳过')

        # 保存优化器/调度器状态，等 _run_training_loop 创建 optimizer 后恢复
        self._resume_opt_state = checkpoint.get('optimizer_state_dict')
        self._resume_sched_state = checkpoint.get('scheduler_state_dict')
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
        """单阶段训练循环（向后兼容）。"""
        config = self.train_config
        logger.info(f'开始训练 | batch_size={config.batch_size} '
                    f'accum={config.grad_accum_steps} '
                    f'effective_batch={config.effective_batch_size}')
        self._run_training_loop(
            train_dataloader, config.total_steps, config.warmup_steps, config.lr,
            eval_steps=config.eval_steps, save_steps=config.save_steps,
            val_dataloader=val_dataloader,
        )
        self._join_save_thread()  # 等待最后的异步保存完成
        self.writer.close()
        logger.info('训练完成')

    def _train_multiphase(self, val_dataloader: Optional[DataLoader] = None):
        """多阶段分层训练。每阶段可设置独立的 LR、warmup、数据、loss 屏蔽。"""
        config = self.train_config

        for phase_idx, phase in enumerate(config.phases):
            logger.info(f'{"="*60}')
            logger.info(f'Phase {phase_idx+1}/{len(config.phases)}: {phase.name}')
            logger.info(f'  数据: {phase.data_split_file}')
            logger.info(f'  步数: {phase.total_steps}')
            logger.info(f'  LR: {phase.lr}')
            logger.info(f'  Loss 屏蔽: {phase.loss_mask is not None}')
            logger.info(f'{"="*60}')

            dataset = TokenDataset(
                split_file=phase.data_split_file,
                data_dir=config.data_dir,
                max_seq_len=self.model_config.max_seq_len,
            )
            phase_loader = DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=2,
                persistent_workers=True,
                pin_memory=False,   # False: 避免 worker 异常退出导致 pin_memory 线程崩溃
                collate_fn=collate_fn,
                drop_last=True,
            )

            masked_ids: Optional[set] = None
            if phase.loss_mask is not None:
                from chopinote_dataset.tokenizer import REMITokenizer
                t = REMITokenizer(grid_size=16, velocity_levels=8)
                masked_ids = phase.loss_mask.get_masked_token_ids(t)
                logger.info(f'  屏蔽 token 数: {len(masked_ids)}')

            self._run_training_loop(
                phase_loader, phase.total_steps, phase.warmup_steps, phase.lr,
                phase_name=phase.name, masked_ids=masked_ids,
                eval_steps=phase.eval_steps, save_steps=phase.save_steps,
                val_dataloader=val_dataloader,
            )
            logger.info(f'Phase {phase.name} 完成 (global_step={self.global_step})')

        self.writer.close()
        logger.info('多阶段训练完成')

    def _run_training_loop(self, dataloader: DataLoader,
                           total_steps: int, warmup_steps: int, lr: float,
                           phase_name: str = '', masked_ids: Optional[set] = None,
                           eval_steps: int | None = None,
                           save_steps: int | None = None,
                           val_dataloader: Optional[DataLoader] = None):
        """共享训练循环（单阶段和多阶段共用）。"""
        config = self.train_config
        model = self.model

        params = list(dict.fromkeys(model.parameters()))
        try:
            self.optimizer = AdamW(params, lr=lr, weight_decay=0.1,
                                   betas=(0.9, 0.95), fused=True)
        except (RuntimeError, TypeError):
            self.optimizer = AdamW(params, lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
        self.scheduler = _get_scheduler(self.optimizer, warmup_steps, total_steps)

        prefix = f'[{phase_name}] ' if phase_name else ''

        # 恢复 checkpoint 中的优化器/调度器状态（_run_training_loop 新创建了 optimizer）
        if getattr(self, '_resume_opt_state', None) is not None:
            try:
                self.optimizer.load_state_dict(self._resume_opt_state)
                if self._resume_sched_state is not None:
                    self.scheduler.load_state_dict(self._resume_sched_state)
                logger.info(f'{prefix}从 checkpoint 恢复 optimizer/scheduler 状态 (step {self.global_step})')
            except Exception as e:
                logger.warning(f'{prefix}无法恢复 optimizer 状态: {e}，使用新初始化')
            self._resume_opt_state = None
            self._resume_sched_state = None

        model.train()
        model.set_gradient_checkpointing(config.gradient_checkpointing)
        _fp8_enabled = False
        if config.use_fp8 and config.fp8_warmup_steps == 0:
            model.set_fp8_mode(True)
            _fp8_enabled = True
            logger.info(f'{prefix}FP8 模式已启用 (warmup=0)')
        elif config.use_fp8:
            model.set_fp8_mode(False)
            logger.info(f'{prefix}FP8 warmup 阶段: 前 {config.fp8_warmup_steps} 步使用 BF16')

        total_loss = 0.0
        start_time = time.time()
        local_step = 0
        accum = 0
        epoch = 0
        _vocab_checked = False

        while local_step < total_steps:
            for batch in dataloader:
                if local_step >= total_steps:
                    break

                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                if not _vocab_checked:
                    max_id = max(input_ids.max().item(), labels.max().item() if labels.numel() else 0)
                    assert max_id < model.config.vocab_size, \
                        f'数据中存在 token ID {max_id} ≥ vocab_size {model.config.vocab_size}'
                    if model.config.use_section_attention and 'section_ids' in batch:
                        max_sec = batch['section_ids'].max().item()
                        max_sec_type = batch['section_types'].max().item()
                        assert max_sec <= model.config.max_sections, \
                            f'section_id {max_sec} 超出 max_sections {model.config.max_sections}'
                        assert max_sec_type < model.config.n_section_types, \
                            f'section_type {max_sec_type} ≥ n_section_types {model.config.n_section_types}'
                    _vocab_checked = True

                if masked_ids:
                    labels = self._apply_loss_mask(labels, masked_ids)

                with autocast('cuda', dtype=torch.bfloat16):
                    # ── 多任务 forward ────────────────────────────
                    model_kwargs = {}
                    if model.config.use_section_attention and 'section_ids' in batch:
                        model_kwargs['section_ids'] = batch['section_ids'].to(self.device)
                        model_kwargs['section_types'] = batch['section_types'].to(self.device)
                        model_kwargs['return_sec_head'] = True
                    if model.config.use_chord_attention and 'chord_func_ids' in batch:
                        model_kwargs['chord_func_ids'] = batch['chord_func_ids'].to(self.device)
                        model_kwargs['chord_inv_ids'] = batch['chord_inv_ids'].to(self.device)
                        model_kwargs['return_chord_head'] = True

                    output = model(input_ids, attention_mask, **model_kwargs)

                    # 解析多任务输出
                    logits = output
                    sec_head_logits = None
                    chord_head_logits = None
                    if isinstance(output, tuple):
                        if len(output) == 3:
                            logits, sec_head_logits, chord_head_logits = output
                        elif len(output) == 2:
                            # 可能是 (logits, sec_head) 或 (logits, chord_head)
                            if isinstance(output[1], dict):
                                if 'bars' in output[1]:
                                    logits, sec_head_logits = output
                                else:
                                    logits, chord_head_logits = output
                        else:
                            logits = output[0]

                    # Next-token prediction loss（主任务）
                    # NaN 诊断: 检查 logits 是否包含 NaN/Inf
                    _logits_flat = logits.view(-1, logits.size(-1))
                    if torch.isnan(_logits_flat).any() or torch.isinf(_logits_flat).any():
                        logger.error(
                            f'{prefix}NaN/Inf in logits before CE! '
                            f'logits range=[{_logits_flat.min().item():.4f}, {_logits_flat.max().item():.4f}] '
                            f'nan={torch.isnan(_logits_flat).any().item()} inf={torch.isinf(_logits_flat).any().item()} '
                            f'Resetting gradient accumulation.')
                        self.optimizer.zero_grad()
                        accum = 0
                        continue
                    # CE 转为 fp32 防止 bf16 log_softmax 下溢 NaN
                    if _LIGER_AVAILABLE:
                        loss = _ce_loss(logits.view(-1, logits.size(-1)).float(), labels.view(-1))
                    else:
                        loss = nn.functional.cross_entropy(
                            logits.view(-1, logits.size(-1)).float(),
                            labels.view(-1),
                            ignore_index=-100,
                            reduction='sum',
                        )
                    loss = loss / max(1, (labels != -100).sum())
                    sec_loss_val = 0.0
                    chord_loss_val = 0.0

                    # Section prediction loss（辅助任务）
                    if sec_head_logits is not None:
                        sec_bars_target = batch['sec_bars_target'].to(self.device)
                        sec_keys_target = batch['sec_keys_target'].to(self.device)
                        sec_types_target = batch['sec_types_target'].to(self.device)

                        has_sec_targets = (sec_bars_target != -1).any()
                        if has_sec_targets:
                            bars_loss = nn.functional.cross_entropy(
                                sec_head_logits['bars'].permute(0, 2, 1),
                                sec_bars_target, ignore_index=-1, reduction='mean',
                            )
                            keys_loss = nn.functional.cross_entropy(
                                sec_head_logits['key'].permute(0, 2, 1),
                                sec_keys_target, ignore_index=-1, reduction='mean',
                            )
                            types_loss = nn.functional.cross_entropy(
                                sec_head_logits['type'].permute(0, 2, 1),
                                sec_types_target, ignore_index=-1, reduction='mean',
                            )
                            sec_loss_val = bars_loss.item() + keys_loss.item() + types_loss.item()
                            loss = loss + model.config.sec_loss_weight * (bars_loss + keys_loss + types_loss)
                        else:
                            sec_loss_val = 0.0

                    # Chord prediction loss（辅助任务）
                    if chord_head_logits is not None:
                        chord_func_targets = batch['chord_func_targets'].to(self.device)
                        chord_inv_targets = batch['chord_inv_targets'].to(self.device)

                        has_chord_func = (chord_func_targets != -1).any()
                        has_chord_inv = (chord_inv_targets != -1).any()

                        chord_func_loss = torch.tensor(0.0, device=self.device)
                        chord_inv_loss = torch.tensor(0.0, device=self.device)

                        if has_chord_func:
                            chord_func_loss = nn.functional.cross_entropy(
                                chord_head_logits['func'].permute(0, 2, 1),
                                chord_func_targets, ignore_index=-1, reduction='mean',
                            )
                        if has_chord_inv:
                            chord_inv_loss = nn.functional.cross_entropy(
                                chord_head_logits['inv'].permute(0, 2, 1),
                                chord_inv_targets, ignore_index=-1, reduction='mean',
                            )

                        chord_loss_val = chord_func_loss.item() + chord_inv_loss.item()
                        if has_chord_func or has_chord_inv:
                            loss = loss + model.config.chord_loss_weight * (chord_func_loss + chord_inv_loss)

                total_loss += loss.item()
                loss = loss / config.grad_accum_steps
                loss.backward()
                accum += 1

                if accum % config.grad_accum_steps == 0:
                    total_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    if _fp8_enabled:
                        model.invalidate_fp8_caches()
                    local_step += 1
                    self.global_step += 1

                    if not _fp8_enabled and config.use_fp8 and \
                       local_step >= config.fp8_warmup_steps:
                        model.set_fp8_mode(True)
                        _fp8_enabled = True
                        logger.info(f'{prefix}Step {local_step}: FP8 模式已启用')

                    self.writer.add_scalar('train/grad_norm', total_norm, self.global_step)

                    if local_step % config.logging_steps == 0:
                        n_micro = config.logging_steps * config.grad_accum_steps
                        avg_loss = total_loss / n_micro
                        self._last_avg_loss = avg_loss
                        self.writer.add_scalar('train/loss', avg_loss, self.global_step)
                        self.writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0], self.global_step)
                        self.writer.add_scalar('train/grad_norm', total_norm, self.global_step)
                        if sec_loss_val > 0:
                            self.writer.add_scalar('train/sec_loss', sec_loss_val, self.global_step)
                        if chord_loss_val > 0:
                            self.writer.add_scalar('train/chord_loss', chord_loss_val, self.global_step)
                        if _fp8_enabled:
                            from .fp8_linear import FP8Linear
                            scales_x, scales_w = [], []
                            for m in model.modules():
                                if isinstance(m, FP8Linear) and m._scale_x is not None:
                                    scales_x.append(m._scale_x.item())
                                    scales_w.append(m._scale_w.item())
                            if scales_x:
                                self.writer.add_scalar('train/fp8_scale_x', sum(scales_x) / len(scales_x), self.global_step)
                                self.writer.add_scalar('train/fp8_scale_w', sum(scales_w) / len(scales_w), self.global_step)
                        elapsed = time.time() - start_time
                        sec_str = f' | Sec: {sec_loss_val:.4f}' if sec_loss_val > 0 else ''
                        chord_str = f' | Chord: {chord_loss_val:.4f}' if chord_loss_val > 0 else ''
                        logger.info(
                            f'{prefix}Step {local_step}/{total_steps}'
                            f'{f" (global {self.global_step})" if phase_name else ""} | '
                            f'Loss: {avg_loss:.4f} | '
                            f'LR: {self.scheduler.get_last_lr()[0]:.2e} | '
                            f'GN: {total_norm:.2f}{sec_str}{chord_str} | '
                            f'Time: {elapsed:.1f}s'
                        )
                        total_loss = 0.0

                    if val_dataloader is not None and eval_steps and \
                       local_step % eval_steps == 0:
                        val_metrics = self.evaluate(val_dataloader, config.max_eval_batches)
                        self.writer.add_scalar('val/loss', val_metrics['loss'], self.global_step)
                        # Log per-type accuracy
                        acc_strs = []
                        for key in sorted(val_metrics):
                            if key.startswith('acc/'):
                                self.writer.add_scalar(f'val/{key}', val_metrics[key], self.global_step)
                                acc_strs.append(f'{key}={val_metrics[key]:.4f}')
                        acc_log = '  '.join(acc_strs) if acc_strs else ''
                        logger.info(f'{prefix}Val loss: {val_metrics["loss"]:.4f}  {acc_log}')
                        model.train()

                    if save_steps and local_step % save_steps == 0:
                        self.save_checkpoint(self._last_avg_loss)
            else:  # DataLoader 自然耗尽 → 新 epoch
                epoch += 1
                logger.info(f'{prefix}DataLoader epoch {epoch} 完成 (step {local_step}/{total_steps}), 启动新 epoch')

        self.save_checkpoint(self._last_avg_loss)
        self._join_save_thread()  # 等待最后的异步保存完成

    def _build_token_type_map(self):
        """构建 token ID → 类型名映射，用于 per-token-type accuracy 统计。"""
        from chopinote_dataset.tokenizer import REMITokenizer

        tokenizer = REMITokenizer(grid_size=16, velocity_levels=8)

        # 自闭合 token（精确匹配）
        exact = {
            tokenizer.PAD: 'special', tokenizer.BOS: 'special',
            tokenizer.EOS: 'special', tokenizer.MASK: 'special',
            tokenizer.BAR: 'bar',
            tokenizer.TUPLET_END: 'tuplet',
            tokenizer.REST: 'rest',
            tokenizer.ARPEGGIO: 'arpeggio',
        }

        # 参数化 token（前缀匹配）
        prefixes = [
            ('position', tokenizer.POSITION),
            ('program', tokenizer.PROGRAM),
            ('note', tokenizer.NOTE_ON),
            ('velocity', tokenizer.VELOCITY),
            ('duration', tokenizer.DURATION),
            ('clef', tokenizer.CLEF),
            ('dynamic', tokenizer.DYNAMIC),
            ('hairpin', tokenizer.HAIRPIN),
            ('artic', tokenizer.ARTIC),
            ('ornament', tokenizer.ORNAMENT),
            ('pedal', tokenizer.PEDAL),
            ('slur', tokenizer.SLUR),
            ('repeat', tokenizer.REPEAT),
            ('jump', tokenizer.JUMP),
            ('tempo', tokenizer.TEMPO),
            ('tuplet', tokenizer.TUPLET_START),
            ('timesig', tokenizer.TIMESIG),
            ('grace_note', tokenizer.GRACE_NOTE),
            ('key', tokenizer.KEY),
            ('beat', tokenizer.BEAT),
            ('octave', tokenizer.OCTAVE),
            ('bass', tokenizer.BASS),
            ('anticipate', tokenizer.ANTICIPATE),
            ('chord', tokenizer.CHORD),
            ('inv', tokenizer.INV),
        ]

        # 收集类型名，保持有序，unknown 放最后
        seen: list = []
        for name, _ in prefixes:
            if name not in seen:
                seen.append(name)
        type_names = ['special', 'bar', *seen, 'rest', 'arpeggio', 'chord', 'inv', 'unknown']

        # 构建 vocab_size 大小的 index 张量
        num_types = len(type_names)
        name_to_idx = {n: i for i, n in enumerate(type_names)}
        type_index = torch.full((tokenizer.vocab_size,), name_to_idx['unknown'], dtype=torch.long)

        for token_id in range(tokenizer.vocab_size):
            token_str = tokenizer.decode_token(token_id)
            if token_str in exact:
                type_index[token_id] = name_to_idx[exact[token_str]]
                continue
            for name, prefix in prefixes:
                if token_str.startswith(prefix):
                    type_index[token_id] = name_to_idx[name]
                    break

        return type_index, type_names

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
    def evaluate(self, dataloader: DataLoader, max_batches: int = 0) -> dict:
        """评估验证集。返回 dict，含 'loss'、per-type acc、section/chord acc。"""
        self.model.eval()
        total_sum = 0.0
        total_tokens = 0
        sec_correct_bars = sec_correct_keys = sec_correct_types = 0
        sec_total = 0
        chord_correct_func = chord_correct_inv = 0
        chord_total_func = chord_total_inv = 0

        num_types = len(self._type_names)
        type_total = torch.zeros(num_types, dtype=torch.float, device=self.device)
        type_correct = torch.zeros(num_types, dtype=torch.float, device=self.device)
        type_idx_map = self._token_id_to_type.to(self.device)

        for i, batch in enumerate(dataloader):
            if max_batches > 0 and i >= max_batches:
                break
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            # 传递 section / chord 数据（与训练一致）
            model_kwargs = {}
            if self.model_config.use_section_attention and 'section_ids' in batch:
                model_kwargs['section_ids'] = batch['section_ids'].to(self.device)
                model_kwargs['section_types'] = batch['section_types'].to(self.device)
                model_kwargs['return_sec_head'] = True
            if self.model_config.use_chord_attention and 'chord_func_ids' in batch:
                model_kwargs['chord_func_ids'] = batch['chord_func_ids'].to(self.device)
                model_kwargs['chord_inv_ids'] = batch['chord_inv_ids'].to(self.device)
                model_kwargs['return_chord_head'] = True

            with autocast('cuda', dtype=torch.bfloat16):
                output = self.model(input_ids, attention_mask, **model_kwargs)

            # 解析输出
            logits = output
            sec_head = chord_head = None
            if isinstance(output, tuple):
                if len(output) == 3:
                    logits, sec_head, chord_head = output
                elif len(output) == 2:
                    logits, sec_head = output

            # ── Loss ──────────────────────────────────────
            if _LIGER_AVAILABLE:
                loss = _ce_loss(logits.view(-1, logits.size(-1)).float(), labels.view(-1))
            else:
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)).float(),
                    labels.view(-1),
                    ignore_index=-100,
                    reduction='sum',
                )
            n_valid = max(1, (labels != -100).sum().item())
            total_sum += loss.item()
            total_tokens += n_valid

            # ── Per-type accuracy (fully vectorized) ──────
            preds = logits.argmax(dim=-1)
            valid = labels != -100
            labels_v = labels[valid]
            types_v = type_idx_map[labels_v]
            correct_v = (preds[valid] == labels_v)

            ones = torch.ones_like(types_v, dtype=torch.float)
            type_total.scatter_add_(0, types_v, ones)
            type_correct.scatter_add_(0, types_v, correct_v.float())

            # ── Section accuracy ──────────────────────────
            if sec_head is not None and 'sec_bars_target' in batch:
                sec_bars = batch['sec_bars_target'].to(self.device)
                sec_keys = batch['sec_keys_target'].to(self.device)
                sec_types = batch['sec_types_target'].to(self.device)
                sec_mask = sec_bars != -1
                if sec_mask.any():
                    sec_total += sec_mask.sum().item()
                    sec_correct_bars += (sec_head['bars'].argmax(-1)[sec_mask] == sec_bars[sec_mask]).sum().item()
                    sec_correct_keys += (sec_head['key'].argmax(-1)[sec_mask] == sec_keys[sec_mask]).sum().item()
                    sec_correct_types += (sec_head['type'].argmax(-1)[sec_mask] == sec_types[sec_mask]).sum().item()

            # ── Chord accuracy ────────────────────────────
            if chord_head is not None and 'chord_func_targets' in batch:
                chord_func = batch['chord_func_targets'].to(self.device)
                chord_inv = batch['chord_inv_targets'].to(self.device)
                f_mask = chord_func != -1
                i_mask = chord_inv != -1
                if f_mask.any():
                    chord_total_func += f_mask.sum().item()
                    chord_correct_func += (chord_head['func'].argmax(-1)[f_mask] == chord_func[f_mask]).sum().item()
                if i_mask.any():
                    chord_total_inv += i_mask.sum().item()
                    chord_correct_inv += (chord_head['inv'].argmax(-1)[i_mask] == chord_inv[i_mask]).sum().item()

        results = {'loss': total_sum / max(1, total_tokens)}
        for i, name in enumerate(self._type_names):
            if type_total[i] > 0:
                results[f'acc/{name}'] = (type_correct[i] / type_total[i]).item()
        overall_total = type_total.sum().item()
        if overall_total > 0:
            results['acc/overall'] = (type_correct.sum() / overall_total).item()
        if sec_total > 0:
            results['acc/sec_bars'] = sec_correct_bars / sec_total
            results['acc/sec_keys'] = sec_correct_keys / sec_total
            results['acc/sec_types'] = sec_correct_types / sec_total
        if chord_total_func > 0:
            results['acc/chord_func'] = chord_correct_func / chord_total_func
        if chord_total_inv > 0:
            results['acc/chord_inv'] = chord_correct_inv / chord_total_inv

        return results
