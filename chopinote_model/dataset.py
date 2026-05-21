"""Memory-efficient token sequence dataset。"""
import json
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from .config import NO_SECTION_ID, NO_SECTION_TYPE_ID


class TokenDataset(Dataset):
    """从 token JSON 文件中流式加载序列片段。

    每次 __getitem__ 随机选一个文件，随机裁剪出 max_seq_len 的片段。
    若存在对应的 .sec.json 文件，额外加载段落标注数据用于段落感知训练。
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

        # 加载 token 长度索引（由预处理生成的 token_lengths.json）
        index_path = self.data_dir / 'token_lengths.json'
        meta_dir = self.data_dir / 'metadata_v2'
        if index_path.exists():
            import gc
            with open(index_path, 'r') as fh:
                length_index = json.load(fh)
            self.file_lengths = [
                length_index.get(Path(fp).stem, 0)
                for fp in self.file_paths
            ]
            del length_index
            gc.collect()  # 释放 ~300MB dict，避免 DataLoader fork 时 COW 复制
            # 对 length=0 的文件回退到 metadata（验证集文件可能不在索引中）
            for i, (fp, length) in enumerate(zip(self.file_paths, self.file_lengths)):
                if length == 0:
                    try:
                        path = self._resolve_path(fp)
                        if path.exists():
                            meta_path = meta_dir / (path.stem + '.meta.json')
                            if meta_path.exists():
                                with open(meta_path, 'r') as f:
                                    meta = json.load(f)
                                self.file_lengths[i] = meta['num_tokens']
                    except Exception:
                        pass
        else:
            # 回退：逐文件读取 meta
            self.file_lengths: list[int] = []
            for fp in self.file_paths:
                try:
                    path = self._resolve_path(fp)
                    if path.exists():
                        meta_path = meta_dir / (path.stem + '.meta.json')
                        if meta_path.exists():
                            with open(meta_path, 'r') as f:
                                meta = json.load(f)
                            self.file_lengths.append(meta['num_tokens'])
                        else:
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

        # FIFO 缓存最近加载的文件（随机采样下 FIFO ≈ LRU）
        self._cache: dict[int, list[int]] = {}
        self._cache_max = 128

    def _resolve_path(self, file_path: str) -> Path:
        """将 split 文件中的路径解析为实际 token 文件路径。"""
        path = self.data_dir / 'tokens_v2' / Path(file_path).name
        if not path.exists():
            fallback = Path(file_path)
            if not fallback.is_absolute():
                fallback = self.data_dir / file_path
            path = fallback
        return path

    def __len__(self) -> int:
        return len(self.valid_indices)

    def _load_tokens(self, file_idx: int) -> list[int]:
        """加载文件 tokens（带 LRU 缓存）。"""
        if file_idx in self._cache:
            return self._cache[file_idx]
        path = self._resolve_path(self.file_paths[file_idx])
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self._cache[file_idx] = data
        if len(self._cache) > self._cache_max:
            # 移除最早未使用的
            self._cache.pop(next(iter(self._cache)))
        return data

    def _load_section_data(self, file_idx: int) -> Optional[dict]:
        """加载段落标注数据。返回 None 表示无标注。"""
        path = self._resolve_path(self.file_paths[file_idx])
        sec_path = path.with_suffix('.sec.json')
        if not sec_path.exists():
            return None
        try:
            with open(sec_path, 'r') as f:
                return json.load(f)
        except Exception:
            return None

    def __getitem__(self, idx: int) -> dict:
        file_idx = self.valid_indices[idx % len(self.valid_indices)]
        tokens = self._load_tokens(file_idx)

        if len(tokens) <= self.max_seq_len + 1:
            # 短序列直接取全部
            seq = torch.tensor(tokens, dtype=torch.long)
            start = 0
        else:
            # 随机裁剪
            start = random.randint(0, len(tokens) - self.max_seq_len - 1)
            seq = torch.tensor(tokens[start:start + self.max_seq_len + 1],
                               dtype=torch.long)

        input_ids = seq[:-1]
        labels = seq[1:]

        # attention mask: 1 表示有效 token
        attention_mask = torch.ones_like(input_ids)

        # ── 段落数据加载 ────────────────────────────────────────
        T = len(input_ids)
        section_data = self._load_section_data(file_idx)
        if section_data is not None:
            sec_ids_all = section_data.get('section_ids', [])
            sec_types_all = section_data.get('section_types', [])
            n_total = len(sec_ids_all)

            # 对齐到 token 裁剪窗口：section_data 与 tokens 等长
            if start < n_total:
                sec_ids = torch.tensor(
                    sec_ids_all[start:start + T + 1][:-1], dtype=torch.long)
                sec_types = torch.tensor(
                    sec_types_all[start:start + T + 1][:-1], dtype=torch.long)
            else:
                sec_ids = torch.full((T,), NO_SECTION_ID, dtype=torch.long)
                sec_types = torch.full((T,), NO_SECTION_TYPE_ID, dtype=torch.long)

            # 构建段落预测目标数组（仅在 Section token 位置有效）
            sec_bars_target = torch.full((T,), -1, dtype=torch.long)
            sec_keys_target = torch.full((T,), -1, dtype=torch.long)
            sec_types_target = torch.full((T,), -1, dtype=torch.long)

            token_positions = section_data.get('section_token_positions', [])
            attrs_list = section_data.get('section_attrs', [])
            for i, pos in enumerate(token_positions):
                local_pos = pos - start
                if 0 <= local_pos < T and i < len(attrs_list):
                    attr = attrs_list[i]
                    sec_bars_target[local_pos] = attr.get('bars', -1)
                    sec_keys_target[local_pos] = attr.get('key', -1)
                    sec_types_target[local_pos] = attr.get('type', -1)
        else:
            sec_ids = torch.full((T,), NO_SECTION_ID, dtype=torch.long)
            sec_types = torch.full((T,), NO_SECTION_TYPE_ID, dtype=torch.long)
            sec_bars_target = torch.full((T,), -1, dtype=torch.long)
            sec_keys_target = torch.full((T,), -1, dtype=torch.long)
            sec_types_target = torch.full((T,), -1, dtype=torch.long)

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'section_ids': sec_ids,
            'section_types': sec_types,
            'sec_bars_target': sec_bars_target,
            'sec_keys_target': sec_keys_target,
            'sec_types_target': sec_types_target,
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

    result = {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask,
    }

    # ── 段落字段 padding ────────────────────────────────────────
    if 'section_ids' in batch[0]:
        section_ids = [b['section_ids'] for b in batch]
        section_types = [b['section_types'] for b in batch]
        sec_bars_target = [b['sec_bars_target'] for b in batch]
        sec_keys_target = [b['sec_keys_target'] for b in batch]
        sec_types_target = [b['sec_types_target'] for b in batch]

        result['section_ids'] = torch.nn.utils.rnn.pad_sequence(
            section_ids, batch_first=True, padding_value=0)
        result['section_types'] = torch.nn.utils.rnn.pad_sequence(
            section_types, batch_first=True, padding_value=0)
        result['sec_bars_target'] = torch.nn.utils.rnn.pad_sequence(
            sec_bars_target, batch_first=True, padding_value=-1)
        result['sec_keys_target'] = torch.nn.utils.rnn.pad_sequence(
            sec_keys_target, batch_first=True, padding_value=-1)
        result['sec_types_target'] = torch.nn.utils.rnn.pad_sequence(
            sec_types_target, batch_first=True, padding_value=-1)

    return result


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
        num_workers=0,
        pin_memory=False,   # PyTorch 2.8 + CUDA 12.8 pin_memory bug
    )
