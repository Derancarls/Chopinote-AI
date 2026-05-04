"""Memory-efficient token sequence dataset。"""
import json
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader


class TokenDataset(Dataset):
    """从 token JSON 文件中流式加载序列片段。

    每次 __getitem__ 随机选一个文件，随机裁剪出 max_seq_len 的片段。
    """

    def __init__(self, split_file: str, data_dir: str = 'data/processed',
                 max_seq_len: int = 2048):
        """
        Args:
            split_file: train.txt / val.txt / test.txt 路径
            data_dir: 数据根目录（用于解析相对路径）
            max_seq_len: 每个样本的最大 token 数
        """
        self.max_seq_len = max_seq_len
        self.data_dir = Path(data_dir)

        # 读取文件列表
        with open(split_file, 'r', encoding='utf-8') as f:
            self.file_paths = [line.strip() for line in f if line.strip()]

        if not self.file_paths:
            raise ValueError(f'文件列表为空: {split_file}')

        # 预读取每个文件的 token 数
        self.file_lengths: list[int] = []
        for fp in self.file_paths:
            try:
                path = self.data_dir / 'tokens' / Path(fp).name
                if not path.exists():
                    path = Path(fp)
                if path.exists():
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    self.file_lengths.append(len(data))
                else:
                    self.file_lengths.append(0)
            except Exception:
                self.file_lengths.append(0)

        self.valid_indices = [
            i for i, l in enumerate(self.file_lengths)
            if l > self.max_seq_len + 1  # 至少能取一个片段 (+1 for label shift)
        ]

        if not self.valid_indices:
            # 退而求其次：任何有内容的文件都能用
            self.valid_indices = [
                i for i, l in enumerate(self.file_lengths) if l > 0
            ]
            if not self.valid_indices:
                raise ValueError('没有可用的训练文件')

        # 缓存最近加载的文件
        self._cache_key: Optional[int] = None
        self._cache_data: Optional[list[int]] = None

    def __len__(self) -> int:
        return max(len(self.valid_indices) * 4, 2048)

    def __getitem__(self, idx: int) -> dict:
        file_idx = random.choice(self.valid_indices)
        length = self.file_lengths[file_idx]

        # 加载数据（带缓存）
        if self._cache_key != file_idx:
            path = self.data_dir / 'tokens' / Path(self.file_paths[file_idx]).name
            if not path.exists():
                path = Path(self.file_paths[file_idx])
            with open(path, 'r', encoding='utf-8') as f:
                self._cache_data = json.load(f)
            self._cache_key = file_idx

        tokens = self._cache_data

        if len(tokens) <= self.max_seq_len + 1:
            # 短序列直接取全部
            seq = torch.tensor(tokens, dtype=torch.long)
        else:
            # 随机裁剪
            start = random.randint(0, len(tokens) - self.max_seq_len - 1)
            seq = torch.tensor(tokens[start:start + self.max_seq_len + 1],
                               dtype=torch.long)

        input_ids = seq[:-1]
        labels = seq[1:]

        # attention mask: 1 表示有效 token
        attention_mask = torch.ones_like(input_ids)

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }


def collate_fn(batch: list[dict]) -> dict:
    """动态 padding 到 batch 内最长序列。"""
    input_ids = [b['input_ids'] for b in batch]
    labels = [b['labels'] for b in batch]
    attention_mask = [b['attention_mask'] for b in batch]

    # pad
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100)
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask, batch_first=True, padding_value=0)

    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask,
    }


def create_dataloader(split_file: str, data_dir: str = 'data/processed',
                      batch_size: int = 2, max_seq_len: int = 2048,
                      shuffle: bool = True) -> DataLoader:
    """创建 DataLoader 的快捷函数。"""
    dataset = TokenDataset(split_file, data_dir, max_seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0,  # Windows 上避免多进程问题
        pin_memory=False,
    )
