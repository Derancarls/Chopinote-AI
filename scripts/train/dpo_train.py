"""DPO 偏好微调脚本。

从 reward_log.jsonl + 保存的 .tokens 文件中构建偏好对（高分段 vs 低分段），
用 DPO (Direct Preference Optimization) 微调模型。

用法:
    python scripts/dpo_train.py \
        --checkpoint /root/autodl-tmp/chopinote/checkpoints/step_9000.pt \
        --reward-dir /root/autodl-tmp/chopinote/rewards \
        --output-dir /root/autodl-tmp/chopinote/dpo_checkpoints \
        --epochs 3

原理:
    DPO 直接优化偏好概率:
    L_DPO = -log σ(β * (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))

    其中 y_w 是高分段（preferred）, y_l 是低分段（rejected）。
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from chopinote_model.config import ModelConfig
from chopinote_model.model import MusicTransformer
from chopinote_dataset.tokenizer import REMITokenizer

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s',
)
logger = logging.getLogger('dpo_train')


# ── 数据：偏好对 ├──────────────────────────────────────────

@dataclass
class PreferencePair:
    """一条偏好数据。"""
    seed_tokens: list[int]       # 种子 token 序列（含 BOS）
    preferred: list[int]         # 高分段续写
    rejected: list[int]          # 低分段续写
    preferred_score: float
    rejected_score: float


def build_preference_dataset(
    reward_dir: str,
    tokenizer,
    data_dir: str | None = None,
    top_k: int = 5,
    min_score_gap: float = 0.15,
    max_pairs: int = 200,
) -> list[PreferencePair]:
    """从 reward 日志和 token 文件构建偏好数据集。

    扫描 reward_log.jsonl，找同一个 seed 的高分/低分配对。
    .tokens 文件应与 .musicxml 同目录或 data_dir 中。
    """
    pairs: list[PreferencePair] = []
    log_path = os.path.join(reward_dir, 'reward_log.jsonl')
    if not os.path.isfile(log_path):
        logger.warning("reward_log.jsonl not found at %s", log_path)
        return pairs

    # 读取所有 entry，按 seed 分组
    entries_by_seed: dict[str, list] = defaultdict(list)
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            seed_path = entry.get('seed_info', {}).get('path', '')
            entries_by_seed[seed_path].append(entry)

    logger.info("从 reward_log 读取了 %d 条记录, %d 个 seed",
                sum(len(v) for v in entries_by_seed.values()), len(entries_by_seed))

    # 遍历每个 seed，按分数排序后配对
    for seed_path, entries in entries_by_seed.items():
        if len(entries) < 2:
            continue

        # 按分数降序排列
        sorted_entries = sorted(entries, key=lambda e: e.get('total_score', 0), reverse=True)

        # 取 top_k 中的高分和低分配对
        top_entries = sorted_entries[:top_k]
        for i in range(len(top_entries)):
            for j in range(i + 1, len(top_entries)):
                score_i = top_entries[i].get('total_score', 0)
                score_j = top_entries[j].get('total_score', 0)
                if abs(score_i - score_j) < min_score_gap:
                    continue
                high, low = (top_entries[i], top_entries[j]) if score_i > score_j else (top_entries[j], top_entries[i])

                # 加载 token 序列（各条目分别查找对应的 .tokens 文件）
                high_tokens = _find_tokens_for_entry(high, reward_dir, data_dir)
                low_tokens = _find_tokens_for_entry(low, reward_dir, data_dir)

                if high_tokens is None or low_tokens is None:
                    continue

                seed_info = high.get('seed_info') or {}
                seed_bars_count = seed_info.get('seed_bars', 0)
                seed = _extract_seed(high_tokens, tokenizer, seed_bars=seed_bars_count)
                if seed is None:
                    continue

                pref_cont = high_tokens[len(seed):] if len(high_tokens) > len(seed) else high_tokens[-256:]
                rej_cont = low_tokens[len(seed):] if low_tokens and len(low_tokens) > len(seed) else (low_tokens[-256:] if low_tokens else None)
                if rej_cont is None:
                    continue

                pair = PreferencePair(
                    seed_tokens=seed,
                    preferred=pref_cont[:512],
                    rejected=rej_cont[:512],
                    preferred_score=score_i if score_i > score_j else score_j,
                    rejected_score=score_j if score_i > score_j else score_i,
                )
                pairs.append(pair)
                logger.debug("Pair: seed=%s, high=%.3f low=%.3f",
                             seed_path, pair.preferred_score, pair.rejected_score)

                if len(pairs) >= max_pairs:
                    break
            if len(pairs) >= max_pairs:
                break

    logger.info("构建了 %d 个偏好对", len(pairs))
    return pairs


def _find_tokens_for_entry(entry, reward_dir, data_dir):
    """查找 reward 条目对应的 .tokens 文件。

    优先用 entry 中的 musicxml_path/output_path（新格式），
    回退按 timestamp 文件名匹配（旧日志兼容）。
    """
    # Method 1: 由 entry 中的 musicxml_path 推导
    musicxml_path = entry.get('musicxml_path') or entry.get('output_path')
    if musicxml_path:
        tok_path = musicxml_path.rsplit('.musicxml', 1)[0] + '.tokens'
        if os.path.isfile(tok_path):
            return _load_tokens_file(tok_path)

    # Method 2: 按 timestamp 模糊匹配文件名（旧日志向后兼容）
    ts = entry.get('timestamp', '')
    if ts:
        ts_file = ts.replace('-', '').replace(' ', '_').replace(':', '')
        for d in [data_dir, reward_dir, os.getcwd()]:
            if not d or not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                if f.endswith('.tokens') and ts_file in f:
                    tokens = _load_tokens_file(os.path.join(d, f))
                    if tokens:
                        return tokens
    return None


def _load_tokens_file(path):
    """从单个 .tokens 文件加载 token 列表。"""
    try:
        with open(path) as f:
            tokens = [int(x) for x in f.read().strip().split()]
        if len(tokens) > 20:
            return tokens
    except (OSError, ValueError):
        pass
    return None


def _load_best_tokens(tok_paths, tokenizer, **kwargs):
    """从候选 token 文件中加载 token 序列。"""
    for p in tok_paths:
        try:
            with open(p) as f:
                tokens = [int(x) for x in f.read().strip().split()]
            if len(tokens) > 20:
                return tokens
        except (OSError, ValueError):
            continue
    return None


def _extract_seed(tokens, tokenizer, seed_bars=0):
    """从完整 token 序列中提取 seed 部分。

    当 seed_bars>0 时，在第 seed_bars 个小节 BAR token 处截断。
    """
    if not tokens:
        return None
    if tokens[0] != tokenizer.bos_token_id:
        tokens = [tokenizer.bos_token_id] + tokens
    if seed_bars > 0:
        bar_id = tokenizer.bar_token_id
        bar_count = 0
        for i, tid in enumerate(tokens):
            if tid == bar_id:
                bar_count += 1
                if bar_count >= seed_bars:
                    return tokens[:i + 1]
    return tokens


# ── LoRA ────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """单层 Linear 的 LoRA 适配器。"""

    def __init__(self, original: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.original = original
        self.original.requires_grad_(False)
        in_features = original.in_features
        out_features = original.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.use_fp8 = False
        device = original.weight.device
        dtype = original.weight.dtype
        self.lora_a = nn.Parameter(torch.zeros(rank, in_features, device=device, dtype=dtype))
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_a, a=5 ** 0.5)
        nn.init.zeros_(self.lora_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.original(x)
        result = result + (x @ self.lora_a.T @ self.lora_b.T) * self.scaling
        return result


def apply_lora_to_model(model: MusicTransformer, rank: int = 8, alpha: float = 16.0) -> tuple[list[nn.Parameter], list[str]]:
    """给模型的 QKV 投影层加 LoRA，返回 (可训练参数列表, 参数名列表)。"""
    lora_params = []
    lora_param_names = []
    for name, module in model.named_modules():
        # 只给注意力层的 QKV 投影加 LoRA
        # qkv 可能是 nn.Linear 或 FP8Linear，均含 weight/bias
        if name.endswith('.qkv') and hasattr(module, 'in_features') and hasattr(module, 'out_features'):
            lora = LoRALinear(module, rank=rank, alpha=alpha)
            parent_name = '.'.join(name.split('.')[:-1])
            parent = model
            for p in name.split('.')[:-1]:
                parent = getattr(parent, p)
            child_name = name.split('.')[-1]
            setattr(parent, child_name, lora)
            lora_params.extend([lora.lora_a, lora.lora_b])
            lora_param_names.extend([f'{name}.lora_a', f'{name}.lora_b'])
            logger.info("LoRA applied to %s (rank=%d, alpha=%.1f)", name, rank, alpha)
    return lora_params, lora_param_names


# ── DPO Loss ────────────────────────────────────────────

def dpo_loss(
    policy_logps_w: torch.Tensor,  # (B,) π_θ(y_w|x) 的对数概率
    policy_logps_l: torch.Tensor,  # (B,) π_θ(y_l|x)
    ref_logps_w: torch.Tensor,     # (B,) π_ref(y_w|x)
    ref_logps_l: torch.Tensor,     # (B,) π_ref(y_l|x)
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """DPO loss.

    Returns:
        loss: 标量 loss
        accuracy: 配对准确率（π_θ(y_w) > π_θ(y_l) 的比例）
    """
    log_ratio_w = policy_logps_w - ref_logps_w
    log_ratio_l = policy_logps_l - ref_logps_l
    logits_diff = beta * (log_ratio_w - log_ratio_l)
    loss = -F.logsigmoid(logits_diff).mean()

    with torch.no_grad():
        accuracy = (log_ratio_w > log_ratio_l).float().mean().item()

    return loss, accuracy


def compute_log_probs(model, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """计算每个序列的对数概率（对非 -100 位置求和）。

    DPO 需要 π(y|x) 的总概率，即所有 continuation token 的对数概率之和。
    """
    logits = model(input_ids)
    log_probs = F.log_softmax(logits, dim=-1)
    # gather 不允许负索引，先把 -100 替换为 0 再 mask 掉
    mask = (labels != -100)
    safe_labels = labels.clone()
    safe_labels[~mask] = 0
    per_token_logp = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    return (per_token_logp * mask).sum(dim=-1)


# ── 训练 ────────────────────────────────────────────────

class DPODataLoader:
    """DPO 数据批次生成器。"""

    def __init__(self, pairs: list[PreferencePair], tokenizer, batch_size: int = 1, max_seq_len: int = 1024):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self._prepare()

    def _prepare(self):
        """预处理：将所有 pair 转为 tensor。

        DPO 需计算 π(continuation|seed)，labels 在 seed 部分设 -100
        只保留 continuation 位置做 next-token prediction loss。
        """
        self.items = []
        for pair in self.pairs:
            seed = pair.seed_tokens
            # full sequence = BOS + seed + continuation
            full_pref = seed + pair.preferred
            full_rej = seed + pair.rejected
            pref_ids = full_pref[:self.max_seq_len]
            rej_ids = full_rej[:self.max_seq_len]

            # labels: seed 位置 = -100, continuation 位置 = shifted continuation
            seed_len = min(len(seed), self.max_seq_len)
            pref_cont_len = len(pref_ids) - seed_len
            pref_labels = ([-100] * seed_len
                          + pair.preferred[1:pref_cont_len]
                          + [-100])
            rej_cont_len = len(rej_ids) - seed_len
            rej_labels = ([-100] * seed_len
                         + pair.rejected[1:rej_cont_len]
                         + [-100])

            # 确保 labels 长度 = input_ids 长度
            assert len(pref_labels) == len(pref_ids), f"{len(pref_labels)} != {len(pref_ids)}"
            assert len(rej_labels) == len(rej_ids), f"{len(rej_labels)} != {len(rej_ids)}"

            self.items.append({
                'preferred': torch.tensor(pref_ids, dtype=torch.long),
                'rejected': torch.tensor(rej_ids, dtype=torch.long),
                'pref_labels': torch.tensor(pref_labels, dtype=torch.long),
                'rej_labels': torch.tensor(rej_labels, dtype=torch.long),
            })

    def __len__(self):
        return max(1, len(self.items) // self.batch_size)

    def __iter__(self):
        indices = torch.randperm(len(self.items)).tolist()
        for i in range(0, len(indices), self.batch_size):
            batch_idx = indices[i:i + self.batch_size]
            batch = {k: torch.stack([self.items[j][k] for j in batch_idx])
                     for k in self.items[0].keys()}
            yield batch


@torch.no_grad()
def compute_ref_log_probs(model, input_ids, labels):
    """计算参考模型的 log probs（无梯度）。"""
    return compute_log_probs(model, input_ids, labels)


def train_dpo(
    model: MusicTransformer,
    ref_model: MusicTransformer,
    train_data: DPODataLoader,
    lora_params: list[nn.Parameter],
    lora_param_names: list[str],
    device: torch.device,
    epochs: int = 3,
    lr: float = 1e-4,
    beta: float = 0.1,
    output_dir: str = 'dpo_checkpoints',
    save_every: int = 50,
):
    """DPO 训练循环。"""
    optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=0.01)
    model.train()
    ref_model.eval()

    os.makedirs(output_dir, exist_ok=True)
    global_step = 0

    for epoch in range(epochs):
        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0

        for batch in train_data:
            pref_ids = batch['preferred'].to(device)
            rej_ids = batch['rejected'].to(device)
            pref_labels = batch['pref_labels'].to(device)
            rej_labels = batch['rej_labels'].to(device)

            optimizer.zero_grad()

            with autocast('cuda', dtype=torch.bfloat16):
                # 策略模型的 log probs
                policy_logps_w = compute_log_probs(model, pref_ids, pref_labels)
                policy_logps_l = compute_log_probs(model, rej_ids, rej_labels)

                # 参考模型的 log probs
                ref_logps_w = compute_ref_log_probs(ref_model, pref_ids, pref_labels)
                ref_logps_l = compute_ref_log_probs(ref_model, rej_ids, rej_labels)

                loss, acc = dpo_loss(
                    policy_logps_w, policy_logps_l,
                    ref_logps_w, ref_logps_l,
                    beta=beta,
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_acc += acc
            n_batches += 1
            global_step += 1

            if global_step % save_every == 0:
                ckpt_path = os.path.join(output_dir, f'dpo_step_{global_step}.pt')
                torch.save({
                    'step': global_step,
                    'model_state_dict': model.state_dict(),
                    'lora_state_dict': dict(zip(lora_param_names, [p.data for p in lora_params])),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                    'config': model.config,
                }, ckpt_path)
                logger.info("Checkpoint saved: %s", ckpt_path)

            if global_step % 10 == 0:
                logger.info(
                    "Epoch %d | Step %d | Loss: %.4f | Acc: %.3f",
                    epoch + 1, global_step, loss.item(), acc,
                )

        avg_loss = total_loss / max(n_batches, 1)
        avg_acc = total_acc / max(n_batches, 1)
        logger.info("Epoch %d done | Avg loss: %.4f | Avg acc: %.3f", epoch + 1, avg_loss, avg_acc)

    # 保存最终模型
    final_path = os.path.join(output_dir, 'dpo_final.pt')
    torch.save({
        'step': global_step,
        'model_state_dict': model.state_dict(),
        'config': model.config,
    }, final_path)
    logger.info("Final model saved: %s", final_path)


# ── 主入口 ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='DPO 偏好微调')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='基础模型 checkpoint 路径')
    parser.add_argument('--reward-dir', type=str,
                        default=os.environ.get('CHOPINOTE_REWARD_DIR',
                                               '/root/autodl-tmp/chopinote/rewards'),
                        help='reward 日志目录')
    parser.add_argument('--output-dir', type=str,
                        default='/root/autodl-tmp/chopinote/dpo_checkpoints',
                        help='DPO checkpoint 输出目录')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='.tokens 文件所在目录（默认从 reward-dir 查找）')
    parser.add_argument('--epochs', type=int, default=3,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='LoRA 学习率')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='DPO beta 参数')
    parser.add_argument('--lora-rank', type=int, default=8,
                        help='LoRA rank')
    parser.add_argument('--lora-alpha', type=float, default=16.0,
                        help='LoRA alpha')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='每批偏好对数')
    parser.add_argument('--min-score-gap', type=float, default=0.15,
                        help='构建偏好对所需的最小分数差')
    parser.add_argument('--max-pairs', type=int, default=200,
                        help='最大偏好对数')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("设备: %s", device)

    # 加载模型
    logger.info("加载基础模型...")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    saved_config = ckpt.get('config')
    if isinstance(saved_config, dict):
        config = ModelConfig(**saved_config)
    else:
        config = saved_config or ModelConfig()

    # 策略模型 + 参考模型（初始同权重）
    model = MusicTransformer(config)
    state_dict = ckpt['model_state_dict']
    model_state = model.state_dict()
    loaded = 0
    skipped = []
    for k, v in state_dict.items():
        if k in model_state and v.shape == model_state[k].shape:
            model_state[k] = v
            loaded += 1
        else:
            skipped.append(k)
    model.load_state_dict(model_state)
    if skipped:
        logger.warning("跳过了 %d 个 shape 不匹配的参数: %s", len(skipped), skipped[:5])
    model = model.to(device, dtype=torch.bfloat16)

    ref_model = MusicTransformer(config)
    ref_model.load_state_dict(model.state_dict())  # 初始权重相同
    ref_model = ref_model.to(device, dtype=torch.bfloat16)

    logger.info("模型加载完成。参数量: %d", sum(p.numel() for p in model.parameters()))

    # 构建数据集
    tokenizer = REMITokenizer(grid_size=16, velocity_levels=8)
    pairs = build_preference_dataset(
        args.reward_dir, tokenizer,
        data_dir=args.data_dir,
        min_score_gap=args.min_score_gap,
        max_pairs=args.max_pairs,
    )
    if not pairs:
        logger.error("没有找到偏好数据。请先运行生成命令收集数据：")
        logger.error("  chopin ... -n 5 --feedback --save-tokens")
        return

    # 应用 LoRA
    lora_params, lora_param_names = apply_lora_to_model(model, rank=args.lora_rank, alpha=args.lora_alpha)
    logger.info("LoRA 可训练参数: %d", sum(p.numel() for p in lora_params))

    # 准备 dataloader
    train_data = DPODataLoader(pairs, tokenizer, batch_size=args.batch_size)
    logger.info("训练数据: %d 个 batch, %d epoch", len(train_data), args.epochs)

    # 训练
    train_dpo(
        model, ref_model, train_data, lora_params, lora_param_names, device,
        epochs=args.epochs, lr=args.lr, beta=args.beta,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
