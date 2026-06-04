"""Memory-efficient token sequence dataset。"""
import json
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from .config import NO_SECTION_ID, NO_SECTION_TYPE_ID

# ── 声部计数：预计算 Position / Note_ON token ID 集合 ────────────
_VOICE_POSITION_IDS: set[int] = set()
_VOICE_NOTEON_IDS: set[int] = set()
_VOICE_INITIALIZED = False


def _init_voice_ids():
    global _VOICE_POSITION_IDS, _VOICE_NOTEON_IDS, _VOICE_INITIALIZED
    if _VOICE_INITIALIZED:
        return
    from chopinote_dataset.tokenizer import REMITokenizer
    tk = REMITokenizer(grid_size=16, velocity_levels=8)
    for token_str, token_id in tk._token_to_id.items():
        if token_str.startswith(REMITokenizer.POSITION):
            _VOICE_POSITION_IDS.add(token_id)
        elif token_str.startswith(REMITokenizer.NOTE_ON):
            _VOICE_NOTEON_IDS.add(token_id)
    _VOICE_INITIALIZED = True


# ── 时值饱和度（DurSat）：预计算 Position/Voice/Duration token ID ──
_DURSAT_POSITION_IDS: set[int] = set()
_DURSAT_VOICE_TIDS: list[int] = [-1, -1, -1, -1]
_DURSAT_DUR_TID_TO_VAL: dict[int, int] = {}
_DURSAT_INITIALIZED = False


def _init_dursat_ids():
    global _DURSAT_POSITION_IDS, _DURSAT_VOICE_TIDS, _DURSAT_DUR_TID_TO_VAL, _DURSAT_INITIALIZED
    if _DURSAT_INITIALIZED:
        return
    from chopinote_dataset.tokenizer import REMITokenizer
    tk = REMITokenizer(grid_size=16, velocity_levels=8)
    for i in range(tk.grid_size):
        _DURSAT_POSITION_IDS.add(tk.encode_token(f'<Position {i}>'))
    for v in range(4):
        _DURSAT_VOICE_TIDS[v] = tk.encode_token(f'<Voice {v}>')
    for d in range(1, tk.grid_size + 1):
        _DURSAT_DUR_TID_TO_VAL[tk.encode_token(f'<Duration {d}>')] = d
    _DURSAT_INITIALIZED = True


def _compute_voice_counts(tokens: list[int]) -> list[int]:
    """实时计算每个 token 位置的声部计数（同 Position 下第几个 Note_ON）。"""
    _init_voice_ids()
    result = [0] * len(tokens)
    counter = 0
    for i, tid in enumerate(tokens):
        if tid in _VOICE_POSITION_IDS:
            counter = 0
        elif tid in _VOICE_NOTEON_IDS:
            result[i] = counter
            counter += 1
        else:
            result[i] = counter
    return result

def _compute_dur_sat_ids(tokens: list[int], bar_token_id: int = 4) -> list[int]:
    """按声部独立追踪 cum_dur, 在 Position token 位置填 bucket (0-16)。

    非 Position token 位置填 0。
    """
    _init_dursat_ids()
    result = [0] * len(tokens)
    cum_dur = [0, 0, 0, 0]
    current_voice = 0
    for i, tid in enumerate(tokens):
        if tid == bar_token_id:
            cum_dur = [0, 0, 0, 0]
        elif tid in _DURSAT_POSITION_IDS:
            result[i] = min(cum_dur[current_voice], 16)
        # Voice token
        for v in range(4):
            if tid == _DURSAT_VOICE_TIDS[v]:
                current_voice = v
                break
        # Duration token
        dur_val = _DURSAT_DUR_TID_TO_VAL.get(tid, 0)
        if dur_val > 0:
            cum_dur[current_voice] += dur_val
    return result


def _compute_measure_in_section(tokens: list[int],
                                  section_data: Optional[dict] = None,
                                  bar_token_id: int = 4,
                                  max_ms: int = 32) -> list[int]:
    """每个 token 在所在段落内的小节偏移（0=本节第一小节）。"""
    result = [0] * len(tokens)
    if section_data is not None and 'section_token_positions' in section_data:
        sec_positions = set(section_data['section_token_positions'])
    else:
        sec_positions = set()

    ms_in_section = 0
    for i, tid in enumerate(tokens):
        result[i] = min(ms_in_section, max_ms)
        if i in sec_positions:
            ms_in_section = 0
        elif tid == bar_token_id:
            ms_in_section += 1
    return result


# ── Key token → key_id 映射（从 token 序列追踪调性，避免用 sec.json 的聚合值）──
_TONIC_TOKEN_MAP: Optional[dict[int, int]] = None

# 12 个主音 → ID (1-12, 0=未知)
_TONIC_NAME_TO_ID = {
    'C': 1, 'C#': 2, 'D': 3, 'D#': 4, 'E': 5,
    'F': 6, 'F#': 7, 'G': 8, 'G#': 9, 'A': 10, 'A#': 11, 'B': 12,
}


def _get_tonic_token_map() -> dict[int, int]:
    """预计算 token_id → tonic_id 映射 (v0.3.0: <Tonic X> tokens)。"""
    global _TONIC_TOKEN_MAP
    if _TONIC_TOKEN_MAP is not None:
        return _TONIC_TOKEN_MAP

    from chopinote_dataset.tokenizer import REMITokenizer
    tk = REMITokenizer(grid_size=16, velocity_levels=8)
    mapping = {}
    for token_str, token_id in tk._token_to_id.items():
        if token_str.startswith('<Tonic ') and token_str.endswith('>'):
            tonic_name = token_str[7:-1]
            tid = _TONIC_NAME_TO_ID.get(tonic_name)
            if tid is not None:
                mapping[token_id] = tid
    _TONIC_TOKEN_MAP = mapping
    return mapping


def _key_ids_from_tokens(tokens: list[int]) -> list[int]:
    """扫描 token 序列，返回 per-position tonic_id (v0.3.0: <Tonic X> tokens)。"""
    tonic_map = _get_tonic_token_map()
    result = [-1] * len(tokens)
    current = -1
    for i, tid in enumerate(tokens):
        kid = tonic_map.get(tid)
        if kid is not None:
            current = kid
        if current >= 0:
            result[i] = current
    return result


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
        meta_dir = self.data_dir / 'metadata_v4'
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

        # LRU 缓存最近加载的文件
        from collections import OrderedDict
        self._cache: OrderedDict[int, list[int]] = OrderedDict()
        self._cache_max = 128

    def _resolve_path(self, file_path: str) -> Path:
        """将 split 文件中的路径解析为实际 token 文件路径。"""
        path = self.data_dir / 'tokens_v4' / Path(file_path).name
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
        self._cache.move_to_end(file_idx)
        if len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)
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

    # _load_chord_data removed in v0.3.2 audit — chord tokens removed in v0.3.0

    def _load_ssf_data(self, file_idx: int) -> Optional[dict]:
        """加载 SSF (Sliding Scale Field) 标注 (v0.3.0)。"""
        path = self._resolve_path(self.file_paths[file_idx])
        ssf_path = path.with_suffix('.ssf.json')
        if not ssf_path.exists():
            return None
        try:
            with open(ssf_path, 'r') as f:
                return json.load(f)
        except Exception:
            return None

    def _load_func_data(self, file_idx: int) -> Optional[dict]:
        """加载功能和声标注 (v0.3.3-opt2)。"""
        path = self._resolve_path(self.file_paths[file_idx])
        func_path = path.with_suffix('.func.json')
        if not func_path.exists():
            return None
        try:
            with open(func_path, 'r') as f:
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
        T = len(input_ids)

        # ── 侧边文件预加载（声部/节内计算需要段落数据）─────
        section_data = self._load_section_data(file_idx)

        # ── 声部计数 ──────────────────────────────────────────
        voice_full = _compute_voice_counts(tokens)
        if start > 0:
            voice_slice = voice_full[start:start + T + 1][:-1]
        else:
            voice_slice = voice_full[:T]
        voice_count_ids = torch.tensor(voice_slice, dtype=torch.long)

        # ── 节内位置计数 ──────────────────────────────────────
        ms_full = _compute_measure_in_section(tokens, section_data)
        if start > 0:
            ms_slice = ms_full[start:start + T + 1][:-1]
        else:
            ms_slice = ms_full[:T]
        measure_in_section_ids = torch.tensor(ms_slice, dtype=torch.long)

        # ── 时值饱和度（DurSat） ──────────────────────────────
        ds_full = _compute_dur_sat_ids(tokens)
        if start > 0:
            ds_slice = ds_full[start:start + T + 1][:-1]
        else:
            ds_slice = ds_full[:T]
        dur_sat_ids = torch.tensor(ds_slice, dtype=torch.long)

        # attention mask: 1 表示有效 token
        attention_mask = torch.ones_like(input_ids)

        # ── 段落数据构建 ────────────────────────────────────────
        if section_data is not None:
            sec_ids_all = section_data.get('section_ids', [])
            sec_types_all = section_data.get('section_types', [])
            n_total = len(sec_ids_all)

            # 校验側边文件与 token 序列长度一致（预防过期侧边文件静默错位）
            assert n_total == len(tokens), (
                f'.sec.json section_ids 长度 ({n_total}) 与 token 序列 ({len(tokens)}) 不匹配: '
                f'{self.file_paths[file_idx]}'
            )

            # 对齐到 token 裁剪窗口：section_data 与 tokens 等长
            if start < n_total:
                sec_ids = torch.tensor(
                    sec_ids_all[start:start + T + 1][:-1], dtype=torch.long)
                sec_types = torch.tensor(
                    sec_types_all[start:start + T + 1][:-1], dtype=torch.long)
            else:
                sec_ids = torch.full((T,), NO_SECTION_ID, dtype=torch.long)
                sec_types = torch.full((T,), NO_SECTION_TYPE_ID, dtype=torch.long)

            # 扫描 token 序列获取每个位置的精确 key_id（取代 sec.json 的段落聚合值）
            key_ids = _key_ids_from_tokens(tokens)

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
                    bars_val = attr.get('bars', -1)
                    # 钳制 bars 到有效范围，避免 section_head 的 CE out-of-range 崩溃
                    if bars_val > 128:
                        bars_val = 128
                    sec_bars_target[local_pos] = bars_val
                    sec_keys_target[local_pos] = key_ids[pos] if 0 <= pos < len(key_ids) else -1
                    sec_types_target[local_pos] = attr.get('type', -1)
        else:
            sec_ids = torch.full((T,), NO_SECTION_ID, dtype=torch.long)
            sec_types = torch.full((T,), NO_SECTION_TYPE_ID, dtype=torch.long)
            sec_bars_target = torch.full((T,), -1, dtype=torch.long)
            sec_keys_target = torch.full((T,), -1, dtype=torch.long)
            sec_types_target = torch.full((T,), -1, dtype=torch.long)

        # ── 和弦数据 — v0.3.0 已移除 Chord tokens, 不再需要 .chord.json ──

        # ── 功能化和声 field (v0.3.3-opt2) ──────────────────────
        func_data = self._load_func_data(file_idx)
        if func_data is not None:
            # Build bar→func_id mapping: 0=PAD, 1=T, 2=SD, 3=D, 4=SDom
            _FUNC_NAME_TO_ID = {'T': 1, 'SD': 2, 'D': 3, 'SDom': 4}
            func_per_bar: dict[int, int] = {}
            for entry in func_data.get('functions', []):
                bar = entry.get('bar', -1)
                fname = entry.get('func', '')
                func_per_bar[bar] = _FUNC_NAME_TO_ID.get(fname, 0)
            # Expand to per-token: track Bar token positions
            n_total = len(tokens)
            func_full = [0] * n_total
            bar_idx = -1
            for i, tid in enumerate(tokens):
                if tid == 4:  # bar_token_id
                    bar_idx += 1
                func_full[i] = func_per_bar.get(bar_idx, 0)
            # Align to crop window
            if start > 0:
                func_ids = torch.tensor(func_full[start:start + T + 1][:-1], dtype=torch.long)
            else:
                func_ids = torch.tensor(func_full[:T], dtype=torch.long)
        else:
            func_ids = torch.zeros(T, dtype=torch.long)

        # ── SSF field 构建 (v0.3.0) ────────────────────────────
        ssf_data = self._load_ssf_data(file_idx)
        if ssf_data is not None:
            n_total_ssf = len(tokens)
            ssf_full = torch.full((n_total_ssf, 12), 0.5, dtype=torch.float)
            tonic_fields = ssf_data.get('tonic_fields', [])
            boundaries = ssf_data.get('section_boundaries', [0])
            # 填充段落级 TonicField
            for i in range(len(boundaries)):
                b_start = boundaries[i]
                b_end = (boundaries[i + 1]
                         if i + 1 < len(boundaries)
                         else n_total_ssf)
                if i < len(tonic_fields):
                    tf = torch.tensor(tonic_fields[i], dtype=torch.float)
                    ssf_full[b_start:b_end] = tf
            # 叠加小节级 LocalField delta
            local_fields = ssf_data.get('local_fields', {})
            for bar_str, delta in local_fields.items():
                bar = int(bar_str)
                if bar < n_total_ssf:
                    ssf_full[bar] += torch.tensor(delta, dtype=torch.float)
            ssf_full = ssf_full.clamp(0.0, 1.0)
            # 对齐裁剪窗口
            if start > 0:
                ssf_fields = ssf_full[start:start + T + 1][:-1]
            else:
                ssf_fields = ssf_full[:T]
        else:
            ssf_fields = torch.full((T, 12), 0.5, dtype=torch.float)

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'ssf_fields': ssf_fields,
            'voice_count_ids': voice_count_ids,
            'measure_in_section_ids': measure_in_section_ids,
            'dur_sat_ids': dur_sat_ids,
            'section_ids': sec_ids,
            'section_types': sec_types,
            'sec_bars_target': sec_bars_target,
            'sec_keys_target': sec_keys_target,
            'sec_types_target': sec_types_target,
            'func_ids': func_ids,
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

    # ── SSF 字段 padding (v0.3.0) ──────────────────────────────
    if 'ssf_fields' in batch[0]:
        ssf_fields = [b['ssf_fields'] for b in batch]
        result['ssf_fields'] = torch.nn.utils.rnn.pad_sequence(
            ssf_fields, batch_first=True, padding_value=0.5)

    # ── 声部计数字段 padding ────────────────────────────────────
    if 'voice_count_ids' in batch[0]:
        voice_count_ids = [b['voice_count_ids'] for b in batch]
        result['voice_count_ids'] = torch.nn.utils.rnn.pad_sequence(
            voice_count_ids, batch_first=True, padding_value=0)

    # ── 节内位置字段 padding ────────────────────────────────────
    if 'measure_in_section_ids' in batch[0]:
        ms_ids = [b['measure_in_section_ids'] for b in batch]
        result['measure_in_section_ids'] = torch.nn.utils.rnn.pad_sequence(
            ms_ids, batch_first=True, padding_value=0)

    # ── 时值饱和度量字段 padding ──────────────────────────────
    if 'dur_sat_ids' in batch[0]:
        dur_sat_ids = [b['dur_sat_ids'] for b in batch]
        result['dur_sat_ids'] = torch.nn.utils.rnn.pad_sequence(
            dur_sat_ids, batch_first=True, padding_value=0)

    # ── 功能化和声字段 padding ──────────────────────────────
    if 'func_ids' in batch[0]:
        func_ids = [b['func_ids'] for b in batch]
        result['func_ids'] = torch.nn.utils.rnn.pad_sequence(
            func_ids, batch_first=True, padding_value=0)

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
        num_workers=0,          # 0: 禁用 multiprocessing，避免 worker 连接丢失崩溃
        persistent_workers=False,
        pin_memory=False,       # False: 避免 worker 异常退出导致 pin_memory 线程崩溃
    )
