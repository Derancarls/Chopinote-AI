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
        param_groups = self._build_param_groups(model, train_config.lr, train_config)
        try:
            self.optimizer = AdamW(
                param_groups, lr=train_config.lr, weight_decay=0.1,
                betas=(0.9, 0.95), fused=True,
            )
        except (RuntimeError, TypeError):
            self.optimizer = AdamW(
                param_groups, lr=train_config.lr, weight_decay=0.1,
                betas=(0.9, 0.95),
            )
        self.scheduler = _get_scheduler(
            self.optimizer, train_config.warmup_steps, train_config.total_steps
        )

        self.global_step = 0
        self.best_loss = float('inf')
        self._last_avg_loss = float('inf')
        self._save_thread: Optional[threading.Thread] = None

        # ── EMA 权重追踪 ────────────────────────────────────
        self._ema_beta = train_config.ema_beta
        self._ema_model: dict | None = None
        if self._ema_beta > 0:
            self._ema_model = {k: v.detach().cpu().clone()
                               for k, v in model.state_dict().items()}

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
                'ema_state_dict': self._ema_model,  # None if disabled
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
                    'ema_state_dict': self._ema_model,
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
        # ── 恢复 EMA 权重 ──
        ema = checkpoint.get('ema_state_dict')
        if ema is not None and self._ema_model is not None:
            for k in self._ema_model:
                if k in ema:
                    self._ema_model[k] = ema[k].cpu().clone()
            logger.info(f'EMA weights restored from checkpoint')
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
                num_workers=0,          # 0: 禁用 multiprocessing，避免 worker 连接丢失崩溃
                pin_memory=False,       # False: 避免 worker 异常退出导致 pin_memory 线程崩溃
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

        param_groups = self._build_param_groups(model, lr, config)
        try:
            self.optimizer = AdamW(param_groups, lr=lr, weight_decay=0.1,
                                   betas=(0.9, 0.95), fused=True)
        except (RuntimeError, TypeError):
            self.optimizer = AdamW(param_groups, lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
        self.scheduler = _get_scheduler(self.optimizer, warmup_steps, total_steps)

        prefix = f'[{phase_name}] ' if phase_name else ''

        # 恢复 checkpoint 中的优化器/调度器状态（_run_training_loop 新创建了 optimizer）
        if getattr(self, '_resume_opt_state', None) is not None:
            try:
                self.optimizer.load_state_dict(self._resume_opt_state)
                # 重建正确的参数组结构（旧 checkpoint 可能只有 1 个 flat group）
                fixed = self._build_param_groups(model, lr, config)
                # 保留已恢复的超参（betas/eps/weight_decay），避免 KeyError
                restored_hparams = {}
                for g in self.optimizer.param_groups:
                    for k, v in g.items():
                        if k != 'params':
                            restored_hparams[k] = v
                for g in fixed:
                    for k, v in restored_hparams.items():
                        g.setdefault(k, v)
                self.optimizer.param_groups = fixed
                if self._resume_sched_state is not None:
                    self.scheduler.load_state_dict(self._resume_sched_state)
                logger.info(f'{prefix}从 checkpoint 恢复 optimizer/scheduler 状态 (step {self.global_step})')
            except Exception as e:
                logger.warning(f'{prefix}无法恢复 optimizer 状态: {e}，使用新初始化')
                self._resume_opt_state = None
                # scheduler 不依赖 param groups，单独恢复
                if self._resume_sched_state is not None:
                    try:
                        self.scheduler.load_state_dict(self._resume_sched_state)
                        logger.info(f'{prefix}独立恢复 scheduler 状态 (step {self.global_step})')
                    except Exception as se:
                        logger.warning(f'{prefix}无法恢复 scheduler 状态: {se}')
                        self._resume_sched_state = None

        # 恢复训练时，用 global_step 偏移 phase 内计数器，让显示从真实进度开始
        resume_offset = self.global_step if phase_name else 0
        adjusted_total = total_steps + resume_offset

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
                    if model.config.use_ssf and 'ssf_fields' in batch:
                        model_kwargs['ssf_fields'] = batch['ssf_fields'].to(self.device)
                    if model.config.use_voice_count and 'voice_count_ids' in batch:
                        model_kwargs['voice_count_ids'] = batch['voice_count_ids'].to(self.device)
                    if model.config.use_measure_in_section and 'measure_in_section_ids' in batch:
                        model_kwargs['measure_in_section_ids'] = batch['measure_in_section_ids'].to(self.device)
                    if model.config.use_dur_sat and 'dur_sat_ids' in batch:
                        model_kwargs['dur_sat_ids'] = batch['dur_sat_ids'].to(self.device)
                    if model.config.use_section_attention and 'section_ids' in batch:
                        model_kwargs['section_ids'] = batch['section_ids'].to(self.device)
                        model_kwargs['section_types'] = batch['section_types'].to(self.device)
                        model_kwargs['return_sec_head'] = True

                    output = model(input_ids, attention_mask, **model_kwargs)

                    # 解析多任务输出: (logits, sec_head, ssf_pred)
                    logits = output
                    sec_head_logits = None
                    ssf_pred = None
                    if isinstance(output, tuple):
                        logits = output[0]
                        for o in output[1:]:
                            if isinstance(o, dict):
                                sec_head_logits = o
                            elif isinstance(o, torch.Tensor) and o.ndim == 3 and o.size(-1) == model.config.ssf_dim:
                                ssf_pred = o

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
                    loss = self._compute_weighted_loss(
                        logits, labels, input_ids, prefix)
                    # ── Z-loss（压制 logit 漂移）─────────
                    if config.z_loss_weight > 0:
                        loss = loss + config.z_loss_weight * logits.float().pow(2).mean()
                    sec_loss_val = 0.0
                    ssf_loss_val = 0.0

                    # Section prediction loss (v0.3.0: key=MSE, type=CE)
                    if sec_head_logits is not None:
                        sec_types_target = batch['sec_types_target'].to(self.device)
                        has_sec_targets = (sec_types_target != -1).any()
                        if has_sec_targets:
                            types_loss = nn.functional.cross_entropy(
                                sec_head_logits['type'].permute(0, 2, 1).float(),
                                sec_types_target, ignore_index=-1, reduction='mean',
                            )
                            sec_loss_val = types_loss.item()
                            loss = loss + model.config.sec_loss_weight * types_loss
                        # key_head → MSE regression on SSF TonicField
                        if (model.config.use_ssf and ssf_pred is not None
                                and 'ssf_fields' in batch):
                            ssf_tgt = batch['ssf_fields'].to(self.device)
                            key_pred = sec_head_logits['key'].float()
                            valid_m = (labels != -100).unsqueeze(-1).float()
                            key_loss = nn.functional.mse_loss(
                                key_pred * valid_m, ssf_tgt[:, -key_pred.size(1):].float() * valid_m,
                                reduction='sum',
                            ) / max(1, valid_m.sum())
                            sec_loss_val += key_loss.item()
                            loss = loss + model.config.sec_loss_weight * key_loss
                        else:
                            sec_loss_val = 0.0

                    # SSF reconstruction loss (12-dim MSE regression)
                    if ssf_pred is not None and model.config.use_ssf_reconstruction:
                        ssf_target = batch['ssf_fields'].to(self.device)
                        valid_mask = (labels != -100).unsqueeze(-1).float()
                        ssf_loss = nn.functional.mse_loss(
                            ssf_pred.float() * valid_mask,
                            ssf_target.float() * valid_mask,
                            reduction='sum',
                        ) / max(1, valid_mask.sum())
                        ssf_loss_val = ssf_loss.item()
                        loss = loss + model.config.ssf_loss_weight * ssf_loss

                total_loss += loss.item()
                loss = loss / config.grad_accum_steps
                loss.backward()
                accum += 1

                if accum % config.grad_accum_steps == 0:
                    total_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    # 梯度 NaN/Inf 检测：跳过 optimizer step 防止权重污染
                    if torch.isnan(torch.tensor(total_norm)) or torch.isinf(torch.tensor(total_norm)):
                        logger.error(f'{prefix}梯度 NaN/Inf (norm={total_norm}), 跳过 optimizer step!')
                        self.optimizer.zero_grad()
                        accum = 0
                        total_loss = 0.0  # 防止 NaN 污染 logging 累加器
                        continue

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    # ── EMA 更新 ────────────────────────────
                    if self._ema_model is not None:
                        beta = self._ema_beta
                        with torch.no_grad():
                            for name, p in model.named_parameters():
                                self._ema_model[name].mul_(beta).add_(
                                    p.detach().cpu(), alpha=1 - beta)

                    if _fp8_enabled:
                        model.invalidate_fp8_caches()
                    local_step += 1
                    self.global_step += 1

                    if not _fp8_enabled and config.use_fp8 and \
                       (local_step >= config.fp8_warmup_steps or self.global_step >= config.fp8_warmup_steps):
                        model.set_fp8_mode(True)
                        _fp8_enabled = True
                        logger.info(f'{prefix}Step {local_step + resume_offset}: FP8 模式已启用')

                    self.writer.add_scalar('train/grad_norm', total_norm, self.global_step)

                    if local_step % config.logging_steps == 0:
                        # ── Dropout 阶梯衰减 ──────────────────
                        ds = config.dropout_schedule
                        if ds:
                            milestones = sorted(ds.keys())
                            applicable = [m for m in milestones if self.global_step >= m]
                            if applicable:
                                target_p = ds[applicable[-1]]
                                current_p = model.dropout.p
                                if abs(current_p - target_p) > 1e-6:
                                    model.set_dropout(target_p)
                                    logger.info(f'{prefix}Step {self.global_step}: '
                                                f'dropout {current_p:.3f}→{target_p:.3f}')

                        n_micro = config.logging_steps * config.grad_accum_steps
                        avg_loss = total_loss / n_micro
                        self._last_avg_loss = avg_loss
                        self.writer.add_scalar('train/loss', avg_loss, self.global_step)
                        self.writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0], self.global_step)
                        self.writer.add_scalar('train/grad_norm', total_norm, self.global_step)
                        if sec_loss_val > 0:
                            self.writer.add_scalar('train/sec_loss', sec_loss_val, self.global_step)
                        if ssf_loss_val > 0:
                            self.writer.add_scalar('train/ssf_loss', ssf_loss_val, self.global_step)
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
                        ssf_str = f' | SSF: {ssf_loss_val:.4f}' if ssf_loss_val > 0 else ''
                        logger.info(
                            f'{prefix}Step {local_step + resume_offset}/{adjusted_total}'
                            f' | Loss: {avg_loss:.4f} | '
                            f'LR: {self.scheduler.get_last_lr()[0]:.2e} | '
                            f'GN: {total_norm:.2f}{sec_str}{ssf_str} | '
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
                        # 释放验证阶段缓存，防止 PyTorch 分配器占用显存不还
                        torch.cuda.empty_cache()

                    if save_steps and local_step % save_steps == 0:
                        self.save_checkpoint(self._last_avg_loss)
                        # ── 自动评估生成：填充 reward_log ──
                        self._run_eval_generation()
                        # ── C 进化层：DPO 自动微调 ──
                        self._check_dpo_trigger()
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
            ('tonic', tokenizer.TONIC),
            ('beat', tokenizer.BEAT),
            ('octave', tokenizer.OCTAVE),
            ('bass', tokenizer.BASS),
            ('voice', tokenizer.VOICE),
            ('fig', tokenizer.FIGURATION),
            ('cadence', tokenizer.CADENCE),
        ]

        # 收集类型名，保持有序，unknown 放最后
        seen: list = []
        for name, _ in prefixes:
            if name not in seen:
                seen.append(name)
        type_names = ['special', 'bar', *seen, 'rest', 'arpeggio', 'unknown']

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
    def _build_param_groups(model: nn.Module, lr: float,
                            train_config: TrainingConfig) -> list[dict]:
        """按模块分组设不同 LR，防止 aux heads / bias scalars 梯度爆炸。

        - backbone (全部非 aux): lr
        - aux_head (section_head, ssf_*): lr * aux_head_lr_mult
        - attn_bias (sec_bias_*, voice_*, cadence_*): lr * attn_bias_lr_mult
        """
        backbone, aux_head, attn_bias = [], [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if 'section_head' in name or 'ssf_reconstruction' in name or 'ssf_proj' in name:
                aux_head.append(p)
            elif any(k in name for k in ('sec_bias_', 'voice_same', 'voice_samepos')):
                attn_bias.append(p)
            else:
                backbone.append(p)
        groups = [
            {'params': backbone, 'lr': lr},
            {'params': aux_head, 'lr': lr * train_config.aux_head_lr_mult},
            {'params': attn_bias, 'lr': lr * train_config.attn_bias_lr_mult},
        ]
        logger.info(
            f'Param groups: backbone={len(backbone)} aux_head={len(aux_head)}'
            f' attn_bias={len(attn_bias)}'
            f' (aux_lr={lr * train_config.aux_head_lr_mult:.2e}'
            f' bias_lr={lr * train_config.attn_bias_lr_mult:.2e})')
        return groups

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

    def _compute_weighted_loss(self, logits: torch.Tensor, labels: torch.Tensor,
                                input_ids: torch.Tensor, prefix: str) -> torch.Tensor:
        """带 token 类型加权 + 和弦上限过滤 + 重复惩罚的 CE loss。"""
        cfg = self.train_config
        B, T, V = logits.shape

        # Per-token CE (sum reduction, ignore -100)
        per_token = nn.functional.cross_entropy(
            logits.view(-1, V).float(),
            labels.view(-1),
            ignore_index=-100,
            reduction='none',
        )

        # ── Token 类型映射 ──────────────────────────────────
        type_idx_map = self._token_id_to_type.to(labels.device)
        label_types = type_idx_map[labels.view(-1).clamp(0, type_idx_map.size(0) - 1)]
        valid_mask = (labels.view(-1) != -100)

        # ── Token type loss weighting ──────────────────────
        pos_type_idx = self._type_names.index('position') if 'position' in self._type_names else -1
        note_type_idx = self._type_names.index('note') if 'note' in self._type_names else -1

        weights = torch.ones(B * T, device=labels.device, dtype=torch.float)
        if pos_type_idx >= 0 and cfg.position_token_loss_weight != 1.0:
            is_pos = (label_types == pos_type_idx) & valid_mask
            weights[is_pos] = cfg.position_token_loss_weight

        # ── 重复惩罚：连续 ≥4 同类型 token → loss × penalty ─
        if cfg.repetition_penalty > 1.0:
            type_seq = label_types.view(B, T)
            for b in range(B):
                run_len = 1
                for t in range(1, T):
                    if valid_mask.view(B, T)[b, t] and type_seq[b, t] == type_seq[b, t - 1]:
                        run_len += 1
                        if run_len >= 4:
                            weights[b * T + t] *= cfg.repetition_penalty
                    else:
                        run_len = 1

        # ── 音符密度过滤：同 Position > max 个 note → mask 整 bar ─
        if note_type_idx >= 0 and cfg.max_notes_per_position > 0:
            note_ids_flat = input_ids.view(-1)
            for b in range(B):
                bar_start = 0
                pos_note_counts = {}  # position -> count of consecutive note tokens
                current_pos = -1
                bar_boundaries = []
                for t in range(T):
                    tid = input_ids[b, t].item()
                    ltype = type_seq[b, t].item()
                    if tid == self.model_config.bar_token_id:
                        bar_boundaries.append(t)
                    if ltype == pos_type_idx:
                        pos_note_counts.clear()
                        current_pos = t
                    elif ltype == note_type_idx and current_pos >= 0:
                        cnt = pos_note_counts.get(current_pos, 0) + 1
                        pos_note_counts[current_pos] = cnt
                # Mask bars where any position has > max_notes_per_position notes
                bar_boundaries.append(T)  # end marker
                for i in range(len(bar_boundaries) - 1):
                    b_start = bar_boundaries[i]
                    b_end = bar_boundaries[i + 1]
                    too_dense = any(
                        pos >= b_start and pos < b_end and cnt > cfg.max_notes_per_position
                        for pos, cnt in pos_note_counts.items()
                    )
                    if too_dense:
                        for t in range(b_start, b_end):
                            idx = b * T + t
                            weights[idx] = 0.0  # mask out entire bar
                        if i == 0 and (b_end - b_start) > 0:
                            logger.debug(
                                f'{prefix}B{b} bar {i}: masked '
                                f'(>={cfg.max_notes_per_position} notes/pos)')

        per_token = per_token * weights
        n_valid = (valid_mask & (weights > 0)).sum()
        return per_token.sum() / max(1, n_valid)

    def _run_eval_generation(self):
        """在 checkpoint 后自动评估生成，填充 reward_log。

        只在训练暂停期间运行（save_checkpoint 之后、下一个 batch 之前），
        复用当前已加载的模型，避免子进程 GPU 争抢。
        """
        cfg = self.train_config
        if not cfg.eval_enabled:
            return
        if cfg.eval_interval_steps <= 0:
            return
        if self.global_step % cfg.eval_interval_steps != 0:
            return

        if not cfg.eval_seed_list or not os.path.isfile(cfg.eval_seed_list):
            logger.warning("eval_seed_list 不存在或为空: %s", cfg.eval_seed_list)
            return

        logger.info("=" * 50)
        logger.info("自动评估生成 (step %d)", self.global_step)

        from scripts.train.batch_evaluate import run_batch_evaluation
        from chopinote_dataset.tokenizer import REMITokenizer

        tokenizer = REMITokenizer(grid_size=16, velocity_levels=8)

        with open(cfg.eval_seed_list) as f:
            seeds = [line.strip() for line in f
                     if line.strip() and not line.startswith('#')]

        temperatures = [float(x.strip())
                       for x in cfg.eval_temperatures.split(',')]

        output_dir = os.path.join(
            cfg.output_dir, 'eval_output')

        reward_log = os.path.join(cfg.dpo_reward_dir, 'reward_log.jsonl')

        # 切 eval 模式
        was_training = self.model.training
        self.model.eval()

        try:
            run_batch_evaluation(
                self.model, tokenizer, seeds, temperatures,
                samples_per_seed=cfg.eval_samples_per_seed,
                max_bars=cfg.eval_max_bars,
                output_dir=output_dir,
                reward_log_path=reward_log,
            )
        except Exception as e:
            logger.error("评估生成异常: %s", e, exc_info=True)
        finally:
            if was_training:
                self.model.train()
            torch.cuda.empty_cache()
            logger.info("自动评估生成完成")
            logger.info("=" * 50)

    def _check_dpo_trigger(self):
        """检查 reward_log 是否有足够新数据，触发 DPO 微调。"""
        cfg = self.train_config
        if not cfg.dpo_enabled:
            return
        if cfg.dpo_interval_steps <= 0:
            return
        if self.global_step % cfg.dpo_interval_steps != 0:
            return

        reward_log = Path(cfg.dpo_reward_dir) / 'reward_log.jsonl'
        if not reward_log.is_file():
            return

        new_count = self._count_new_reward_entries(reward_log)
        if new_count < cfg.dpo_min_new_entries:
            logger.info(
                f'DPO: {new_count} 条新 reward, 不足 {cfg.dpo_min_new_entries}, 跳过')
            return

        self._run_dpo_phase(reward_log)

    def _count_new_reward_entries(self, reward_log: Path) -> int:
        """统计自上次 DPO 以来新增的 reward 条目。"""
        total = 0
        try:
            with open(reward_log) as f:
                for _ in f:
                    total += 1
        except Exception:
            return 0
        last = getattr(self, '_dpo_last_processed_line', 0)
        return max(0, total - last)

    def _count_total_lines(self, path: Path) -> int:
        try:
            with open(path) as f:
                return sum(1 for _ in f)
        except Exception:
            return 0

    def _run_dpo_phase(self, reward_log: Path):
        """执行一次 DPO 微调，完成后恢复训练。"""
        logger.info('=' * 60)
        logger.info(f'DPO phase 启动 (step {self.global_step})')

        # 1. 保存 pre-DPO checkpoint
        pre_dpo_path = Path(self.train_config.output_dir) / f'step_{self.global_step}_pre_dpo.pt'
        self.save_checkpoint(self._last_avg_loss)
        self._join_save_thread()
        ckpt_path = Path(self.train_config.output_dir) / f'step_{self.global_step}.pt'
        if ckpt_path.is_file() and not pre_dpo_path.is_file():
            shutil.copy(ckpt_path, pre_dpo_path)
            logger.info(f'Pre-DPO backup: {pre_dpo_path}')

        # 2. 构建偏好对
        from scripts.train.dpo_train import (
            build_preference_dataset, apply_lora_to_model,
            compute_log_probs, dpo_loss,
            DPODataLoader, LoRALinear,
        )
        from chopinote_dataset.tokenizer import REMITokenizer

        tokenizer = REMITokenizer(grid_size=16, velocity_levels=8)
        pairs = build_preference_dataset(
            str(reward_log.parent), tokenizer,
            data_dir=self.train_config.data_dir,
            min_score_gap=self.train_config.dpo_min_score_gap,
            max_pairs=200,
        )
        if len(pairs) < 4:
            logger.info('DPO: 偏好对不足 4 对，跳过')
            if pre_dpo_path.is_file():
                pre_dpo_path.unlink()
            return

        # 3. 冻结全模型，只训练 LoRA
        for p in self.model.parameters():
            p.requires_grad_(False)

        # 4. 应用 LoRA
        lora_params, lora_param_names = apply_lora_to_model(
            self.model,
            rank=self.train_config.dpo_lora_rank,
            alpha=16.0,
        )
        logger.info(f'DPO: LoRA 可训练参数 {sum(p.numel() for p in lora_params)}')

        # 5. 参考模型 = 当前快照（不可训练）
        import copy
        ref_model = copy.deepcopy(self.model)
        ref_model.to(self.device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

        # 6. 数据
        dpo_loader = DPODataLoader(pairs, tokenizer, batch_size=1)

        # 7. 优化
        dpo_optimizer = torch.optim.AdamW(lora_params, lr=1e-4, weight_decay=0.01)
        from torch.amp import autocast

        n_batches = 0
        final_loss = 0.0
        for epoch in range(self.train_config.dpo_epochs):
            for batch in dpo_loader:
                pref_ids = batch['preferred'].to(self.device)
                rej_ids = batch['rejected'].to(self.device)
                pref_labels = batch['pref_labels'].to(self.device)
                rej_labels = batch['rej_labels'].to(self.device)

                dpo_optimizer.zero_grad()

                with autocast('cuda', dtype=torch.bfloat16):
                    policy_w = compute_log_probs(self.model, pref_ids, pref_labels)
                    policy_l = compute_log_probs(self.model, rej_ids, rej_labels)
                    with torch.no_grad():
                        ref_w = compute_log_probs(ref_model, pref_ids, pref_labels)
                        ref_l = compute_log_probs(ref_model, rej_ids, rej_labels)
                    loss, acc = dpo_loss(
                        policy_w, policy_l, ref_w, ref_l,
                        beta=self.train_config.dpo_beta,
                    )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                dpo_optimizer.step()
                n_batches += 1
                final_loss = loss.item()

            logger.info(f'DPO epoch {epoch+1}/{self.train_config.dpo_epochs} '
                        f'loss={final_loss:.4f} acc={acc:.3f}')

        # 8. 合并 LoRA
        self._merge_lora_weights()

        # 9. 清理
        del ref_model
        torch.cuda.empty_cache()

        # 10. 记录游标 + 保存 post-DPO checkpoint
        self._dpo_last_processed_line = self._count_total_lines(reward_log)
        self.save_checkpoint(self._last_avg_loss)
        self._join_save_thread()

        logger.info(f'DPO phase 完成 ({n_batches} batch, final loss={final_loss:.4f})')
        logger.info('=' * 60)

    def _merge_lora_weights(self):
        """LoRA ΔW 合并回原始权重，解冻全模型。"""
        for name, module in self.model.named_modules():
            if not hasattr(module, 'lora_a'):
                continue
            delta = (module.lora_b.T @ module.lora_a) * module.scaling
            module.original.weight.data.add_(delta.to(module.original.weight.dtype))
            # 恢复为普通 nn.Linear
            parent_name = '.'.join(name.split('.')[:-1])
            parent = self.model
            for p in name.split('.')[:-1]:
                parent = getattr(parent, p)
            if parent is not self.model:
                setattr(parent, name.split('.')[-1], module.original)

        for p in self.model.parameters():
            p.requires_grad_(True)
        logger.info('DPO: LoRA 权重已合并，全模型参数已解冻')

    # ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, max_batches: int = 0) -> dict:
        """评估验证集。返回 dict，含 'loss'、per-type acc、section acc (v0.3.0: 移除 chord acc)。"""
        self.model.eval()
        total_sum = 0.0
        total_tokens = 0
        sec_correct_types = 0
        sec_total = 0

        num_types = len(self._type_names)
        type_total = torch.zeros(num_types, dtype=torch.float, device=self.device)
        type_correct = torch.zeros(num_types, dtype=torch.float, device=self.device)
        type_idx_map = self._token_id_to_type.to(self.device)

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if max_batches > 0 and i >= max_batches:
                    break
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                # 传递 section / voice / measure_in_section 数据
                model_kwargs = {}
                if self.model_config.use_ssf and 'ssf_fields' in batch:
                    model_kwargs['ssf_fields'] = batch['ssf_fields'].to(self.device)
                if self.model_config.use_voice_count and 'voice_count_ids' in batch:
                    model_kwargs['voice_count_ids'] = batch['voice_count_ids'].to(self.device)
                if self.model_config.use_measure_in_section and 'measure_in_section_ids' in batch:
                    model_kwargs['measure_in_section_ids'] = batch['measure_in_section_ids'].to(self.device)
                if self.model_config.use_dur_sat and 'dur_sat_ids' in batch:
                    model_kwargs['dur_sat_ids'] = batch['dur_sat_ids'].to(self.device)
                if self.model_config.use_section_attention and 'section_ids' in batch:
                    model_kwargs['section_ids'] = batch['section_ids'].to(self.device)
                    model_kwargs['section_types'] = batch['section_types'].to(self.device)
                    model_kwargs['return_sec_head'] = True

                with autocast('cuda', dtype=torch.bfloat16):
                    output = self.model(input_ids, attention_mask, **model_kwargs)

                logits = output
                sec_head = None
                if isinstance(output, tuple):
                    logits = output[0]
                    for o in output[1:]:
                        if isinstance(o, dict):
                            sec_head = o

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

                # ── Per-type accuracy ─────────────────────────
                preds = logits.argmax(dim=-1)
                valid = labels != -100
                labels_v = labels[valid]
                types_v = type_idx_map[labels_v]
                correct_v = (preds[valid] == labels_v)

                ones = torch.ones_like(types_v, dtype=torch.float)
                type_total.scatter_add_(0, types_v, ones)
                type_correct.scatter_add_(0, types_v, correct_v.float())

                # ── Section accuracy ──────────────────────────
                if sec_head is not None and 'sec_types_target' in batch:
                    sec_types = batch['sec_types_target'].to(self.device)
                    sec_mask = sec_types != -1
                    if sec_mask.any():
                        sec_total += sec_mask.sum().item()
                        sec_correct_types += (sec_head['type'].argmax(-1)[sec_mask] == sec_types[sec_mask]).sum().item()

        results = {'loss': total_sum / max(1, total_tokens)}
        for i, name in enumerate(self._type_names):
            if type_total[i] > 0:
                results[f'acc/{name}'] = (type_correct[i] / type_total[i]).item()
        overall_total = type_total.sum().item()
        if overall_total > 0:
            results['acc/overall'] = (type_correct.sum() / overall_total).item()
        if sec_total > 0:
            results['acc/sec_types'] = sec_correct_types / sec_total

        return results
