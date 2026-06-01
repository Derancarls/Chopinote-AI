"""段落结构标注管道：自动检测段落边界并推断段落属性。

用法:
    python scripts/structure_annotator.py annotate \\
        --input-dir /root/autodl-tmp/data/processed/tokens_v2 \\
        --output-dir /root/autodl-tmp/data/processed/tokens_v2 \\
        --num-workers 8

    python scripts/structure_annotator.py check \\
        --input-dir /root/autodl-tmp/data/processed/tokens_v2
"""
import argparse
import json
import logging
import math
import multiprocessing
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# 导入 tokenizer 用于解码整数 token ID
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chopinote_dataset.tokenizer import REMITokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# ── 段落类型定义 ─────────────────────────────────────────────────────────────
# ID 0 reserved for "no section" (padding)
SECTION_TYPES = {
    'exposition': 1,
    'development': 2,
    'recapitulation': 3,
    'intro': 4,
    'coda': 5,
    'bridge': 6,
    'cadenza': 7,
    'transition': 8,
    'theme1': 9,
    'theme2': 10,
    'themen': 11,
    'variation': 12,
    'episode': 13,
    'section0': 14,
    'section1': 15,
    'section2': 16,
    'section3': 17,
    'section4': 18,
    'section5': 19,
    'section6': 20,
    'section7': 21,
}

# 段落 ID → tokenizer token 名映射
SECTION_TOKEN_NAMES = {
    1: 'exposition',
    2: 'development',
    3: 'recapitulation',
    4: 'intro',
    5: 'coda',
    6: 'bridge',
    7: 'cadenza',
    8: 'transition',
    9: 'theme1',
    10: 'theme2',
    11: 'themen',
    12: 'variation',
    13: 'episode',
    14: '0',
    15: '1',
    16: '2',
    17: '3',
    18: '4',
    19: '5',
    20: '6',
    21: '7',
}

# 信号权重
SIGNAL_WEIGHTS = {
    'key_change': 3.0,
    'repeat': 2.0,
    'density_shift': 1.5,
    'tempo_change': 1.5,
    'program_change': 1.0,
    'silence_gap': 1.0,
    'pattern_end': 1.0,
}

MIN_SECTION_BARS = 4       # 段落最小长度
BOUNDARY_THRESHOLD = 0.2   # 边界检测置信度阈值
DECAY_WINDOW = 16          # 衰减窗口（小节）

# Lazy tokenizer 单例（避免每个文件重复创建）
_tokenizer_instance = None


def _get_tokenizer() -> REMITokenizer:
    global _tokenizer_instance
    if _tokenizer_instance is None:
        _tokenizer_instance = REMITokenizer(grid_size=16, velocity_levels=8)
    return _tokenizer_instance


# ── Token 解析工具 ───────────────────────────────────────────────────────────

def _parse_key(token_str: str) -> Optional[str]:
    """从 '<Tonic C>' token 提取主音名 (v0.3.0)。"""
    if token_str.startswith('<Tonic ') and token_str.endswith('>'):
        return token_str[7:-1]
    return None


def _parse_tempo(token_str: str) -> Optional[int]:
    """从 '<Tempo 120>' 提取 BPM。"""
    if token_str.startswith('<Tempo ') and token_str.endswith('>'):
        try:
            return int(token_str[7:-1])
        except ValueError:
            return None
    return None


def _parse_program(token_str: str) -> Optional[int]:
    """从 '<Program N>' 或 '<Program N_M>' 提取 program number。"""
    if token_str.startswith('<Program ') and token_str.endswith('>'):
        val = token_str[9:-1]
        return int(val.split('_')[0])
    return None


def _is_dynamic_token(token_str: str) -> bool:
    return token_str.startswith('<Dynamic ')


def _is_rest(token_str: str) -> bool:
    return token_str == '<Rest>'


def _is_bar(token_str: str) -> bool:
    return token_str == '<Bar>'


def _is_note_on(token_str: str) -> bool:
    return token_str.startswith('<Note_ON ')


def _is_repeat(token_str: str) -> bool:
    return token_str.startswith('<Repeat ')


# ── 小节级特征提取 ────────────────────────────────────────────────────────────

def extract_bar_features(tokens: List[str]) -> List[dict]:
    """将 token 序列转换为小节级特征列表。

    Returns:
        每个元素是一个 dict，包含该小节的统计特征。
    """
    bars = []
    current_bar = {
        'bar_idx': 0,
        'n_tokens': 0,
        'n_notes': 0,
        'n_rests': 0,
        'note_density': 0.0,
        'key': None,
        'tempo': None,
        'programs': set(),
        'dynamics': [],
        'repeats': [],
    }
    token_idx = 0

    for token_str in tokens:
        token_idx += 1
        if _is_bar(token_str):
            bars.append(current_bar)
            current_bar = {
                'bar_idx': len(bars),
                'n_tokens': 0,
                'n_notes': 0,
                'n_rests': 0,
                'note_density': 0.0,
                'key': None,
                'tempo': None,
                'programs': set(),
                'dynamics': [],
                'repeats': [],
                '_start_token': token_idx,
            }
            continue

        current_bar['n_tokens'] += 1
        if _is_note_on(token_str):
            current_bar['n_notes'] += 1
        if _is_rest(token_str):
            current_bar['n_rests'] += 1

        k = _parse_key(token_str)
        if k is not None:
            current_bar['key'] = k

        t = _parse_tempo(token_str)
        if t is not None:
            current_bar['tempo'] = t

        p = _parse_program(token_str)
        if p is not None:
            current_bar['programs'].add(p)

        if _is_dynamic_token(token_str):
            current_bar['dynamics'].append(token_str)

        if _is_repeat(token_str):
            current_bar['repeats'].append(token_str)

    # 添加最后一小节后未遇到 <Bar>
    if current_bar['n_tokens'] > 0 or not bars:
        bars.append(current_bar)

    # 计算 note_density（每小节音符数 / 小节内 token 数）
    for bar in bars:
        total = bar['n_tokens'] or 1
        bar['note_density'] = bar['n_notes'] / total
        bar['_end_token'] = bar.get('_start_token', 0) + bar['n_tokens']

    return bars


# ── 边界检测信号 ─────────────────────────────────────────────────────────────

def compute_key_change_signal(bars: List[dict]) -> np.ndarray:
    """调性变化信号：每对相邻小节 key 不同 → 1。"""
    n = len(bars)
    signal = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        if bars[i]['key'] is not None and bars[i - 1]['key'] is not None:
            if bars[i]['key'] != bars[i - 1]['key']:
                signal[i] = 1.0
        # 单侧 key 变化也算
        elif bars[i]['key'] != bars[i - 1]['key']:
            signal[i] = 0.8
    return signal


def compute_density_shift_signal(bars: List[dict]) -> np.ndarray:
    """密度突变信号：滚动窗口 Z-score。"""
    n = len(bars)
    densities = np.array([b['note_density'] for b in bars], dtype=np.float32)
    signal = np.zeros(n, dtype=np.float32)
    window = 4
    for i in range(n):
        start = max(0, i - window)
        end = min(n, i + window + 1)
        local = densities[start:end]
        if len(local) > 2:
            mean = local.mean()
            std = local.std() + 1e-8
            z = abs(densities[i] - mean) / std
            if z > 2.0:
                signal[i] = min(z / 5.0, 1.0)
    return signal


def compute_repeat_signal(bars: List[dict]) -> np.ndarray:
    """反复标记信号：有 Repeat token → 1.0，Double barline 推断 ≈ 0.8。"""
    n = len(bars)
    signal = np.zeros(n, dtype=np.float32)
    for i, bar in enumerate(bars):
        if bar['repeats']:
            signal[i] = 1.0
        # 密度明显变化也提示反复结束
        if i >= 2 and i < n - 2:
            pre_density = np.mean([bars[i - j]['note_density'] for j in range(1, 3)])
            post_density = np.mean([bars[i + j]['note_density'] for j in range(1, 3)])
            if abs(pre_density - post_density) > 0.5:
                signal[i] = max(signal[i], 0.5)
    return signal


def compute_tempo_change_signal(bars: List[dict]) -> np.ndarray:
    """速度变化信号。"""
    n = len(bars)
    signal = np.zeros(n, dtype=np.float32)
    prev_tempo = None
    for i, bar in enumerate(bars):
        if bar['tempo'] is not None:
            if prev_tempo is not None and bar['tempo'] != prev_tempo:
                signal[i] = 1.0
            prev_tempo = bar['tempo']
    return signal


def compute_program_change_signal(bars: List[dict]) -> np.ndarray:
    """乐器变化信号。"""
    n = len(bars)
    signal = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        if bars[i]['programs'] and bars[i - 1]['programs']:
            if bars[i]['programs'] != bars[i - 1]['programs']:
                signal[i] = 1.0
    return signal


def compute_silence_gap_signal(bars: List[dict]) -> np.ndarray:
    """休止间隔信号：连续休止 > 2 拍 → 段落边界候选。"""
    n = len(bars)
    signal = np.zeros(n, dtype=np.float32)
    consec_rests = 0
    for i, bar in enumerate(bars):
        if bar['n_rests'] > 0 and bar['n_notes'] == 0:
            consec_rests += 1
            if consec_rests >= 2:  # 连续 2 小节无音符
                signal[i] = min(consec_rests / 4.0, 1.0)
        else:
            consec_rests = 0
    return signal


# ── Viterbi 分段 ────────────────────────────────────────────────────────────

def viterbi_segmentation(fused_signal: np.ndarray, min_bars: int = 4) -> np.ndarray:
    """Viterbi 解码最优分段路径。

    以小节为单位搜索，目标函数平衡「边界强度」和「段落最小长度」。

    Returns:
        (n_bars,) array, 0=非边界, 1=边界
    """
    n = len(fused_signal)
    if n < min_bars * 2:
        return np.zeros(n, dtype=np.int32)

    # DP: dp[i] = max score up to bar i
    dp = np.zeros(n, dtype=np.float32)
    back = np.full(n, -1, dtype=np.int32)
    dp[0] = 0.0

    for i in range(1, n):
        best_score = dp[i - 1]
        best_j = i - 1
        for j in range(max(0, i - 32), i - min_bars + 1):
            # 边界得分 = 边界信号强度 - 过长段落罚分
            boundary_score = fused_signal[i]
            length_penalty = 0.0 if (i - j) < 24 else 0.1 * ((i - j) - 24) / 24
            score = dp[j] + boundary_score - length_penalty
            if score > best_score:
                best_score = score
                best_j = j
        dp[i] = best_score
        back[i] = best_j

    # 回溯
    boundaries = np.zeros(n, dtype=np.int32)
    i = n - 1
    while i > 0:
        j = back[i]
        if j >= i:
            break
        if j > 0:
            boundaries[j] = 1
        i = j

    return boundaries


# ── 段落类型推断 ─────────────────────────────────────────────────────────────

def infer_section_types(bars: List[dict], boundaries: np.ndarray) -> List[int]:
    """为每个段落推断类型 ID。

    策略:
    1. Sonata detection: Key I → Key V → Development → Key I
    2. Theme tracking: 用 self-similarity 找重复模式
    3. Fallback: 编号 Section 0-7
    """
    n = len(bars)
    boundary_indices = [0] + [i for i in range(1, n) if boundaries[i]] + [n - 1]
    boundary_indices = sorted(set(boundary_indices))

    n_sections = len(boundary_indices) - 1
    if n_sections <= 0:
        return []

    # 提取每个段落的 key 序列
    section_keys = []
    for s in range(n_sections):
        start = boundary_indices[s]
        end = boundary_indices[s + 1] + 1
        keys_in_section = [bars[i]['key'] for i in range(start, min(end, n))
                          if bars[i]['key'] is not None]
        section_keys.append(keys_in_section)

    # 计算每个段落的主调（出现最多的 key）
    section_primary_keys = []
    for keys in section_keys:
        if keys:
            from collections import Counter
            section_primary_keys.append(Counter(keys).most_common(1)[0][0])
        else:
            section_primary_keys.append(None)

    # 尝试 Sonata 检测
    section_types = _detect_sonata(section_primary_keys, n_sections)

    if section_types is not None:
        return section_types

    # 用 self-similarity 检测主题重复
    section_types = _detect_theme_patterns(bars, boundary_indices, section_keys)

    if section_types is not None:
        return section_types

    # Fallback: 编号
    return [_section_fallback_id(i) for i in range(n_sections)]


def _detect_sonata(keys: List[Optional[str]], n_sections: int) -> Optional[List[int]]:
    """尝试检测奏鸣曲式段落。

    信号：I → V → 频繁转调 → I 回返
    """
    if n_sections < 3:
        return None

    # 检查起始和结束调性是否一致（回返到主调）
    first_key = keys[0] if len(keys) > 0 else None
    last_key = keys[-1] if len(keys) > 0 else None
    if first_key is None or last_key is None:
        return None

    # 简单判断：如果开始和结束调相同且有中间段
    _I_to_V = _key_interval(first_key, last_key) if first_key and last_key else None

    # 检测中间段是否调性不稳定
    mid_keys = [k for k in keys[1:-1] if k is not None]
    if len(mid_keys) < 2:
        return None
    unique_mid = len(set(mid_keys))
    mid_unstable = unique_mid > 1

    # 三段落 I → X → I 且中间不稳定 → sonata
    if n_sections >= 3 and _I_to_V is not None and mid_unstable:
        result = [_section_fallback_id(0)]
        for i in range(1, n_sections - 1):
            result.append(SECTION_TYPES['development'])
        result.append(_section_fallback_id(n_sections - 1))
        return result

    # 尝试五段落（exposition/bridge/development/recapitulation/coda）
    if n_sections >= 4:
        result = [SECTION_TYPES['exposition']]
        for i in range(1, n_sections - 2):
            result.append(SECTION_TYPES['development'])
        result.append(SECTION_TYPES['recapitulation'])
        if n_sections >= 5:
            result.append(SECTION_TYPES['coda'])
        return result
    return None


def _detect_theme_patterns(bars: List[dict], boundary_indices: List[int],
                            section_keys: List[List[Optional[str]]]) -> Optional[List[int]]:
    """用音符密度向量检测主题重复模式。"""
    n_sections = len(boundary_indices) - 1
    if n_sections < 2:
        return None

    # 提取每个段落的密度特征
    section_features = []
    for s in range(n_sections):
        start = boundary_indices[s]
        end = boundary_indices[s + 1] + 1
        densities = [bars[i]['note_density'] for i in range(start, min(end, len(bars)))]
        if densities:
            section_features.append(np.mean(densities))
        else:
            section_features.append(0.0)

    # 用密度相似度找重复模式
    result = []
    used = [False] * n_sections
    for i in range(n_sections):
        if used[i]:
            continue
        # 找与 i 最相似的后续段落
        best_j = None
        best_sim = -1.0
        for j in range(i + 1, n_sections):
            if used[j]:
                continue
            sim = 1.0 - abs(section_features[i] - section_features[j]) / (
                max(abs(section_features[i]), abs(section_features[j]), 0.01))
            if sim > 0.7 and sim > best_sim:
                best_sim = sim
                best_j = j

        if best_j is not None and not used[best_j]:
            result.append(SECTION_TYPES['theme1'] if len([r for r in result if r in (
                SECTION_TYPES['theme1'], SECTION_TYPES['theme2'])]) == 0
                         else SECTION_TYPES['theme2'])
            used[i] = True
            used[best_j] = True
            # 之间的段落标 bridge 或 transition
            for k in range(i + 1, best_j):
                if not used[k]:
                    result.append(SECTION_TYPES['bridge'])
                    used[k] = True
        elif not used[i]:
            result.append(_section_fallback_id(len(result)))
            used[i] = True

    return result


def _section_fallback_id(index: int) -> int:
    """回退：返回 Section 0-7 对应的 ID。"""
    section_num = min(index % 8, 7)
    return SECTION_TYPES[f'section{section_num}']


def _key_interval(k1: str, k2: str) -> Optional[int]:
    """计算两个调的五度圈距离（半音数）。"""
    KEY_TO_PC = {
        'C': 0, 'G': 7, 'D': 2, 'A': 9, 'E': 4, 'B': 11, 'F#': 6, 'C#': 1,
        'F': 5, 'Bb': 10, 'Eb': 3, 'Ab': 8, 'Db': 1, 'Gb': 6, 'Cb': 11,
        'Am': 9, 'Em': 4, 'Bm': 11, 'F#m': 6, 'C#m': 1, 'G#m': 8, 'D#m': 3,
        'A#m': 10, 'Dm': 2, 'Gm': 7, 'Cm': 0, 'Fm': 5, 'Bbm': 10, 'Ebm': 3,
        'Abm': 8,
    }
    if k1 in KEY_TO_PC and k2 in KEY_TO_PC:
        interval = (KEY_TO_PC[k2] - KEY_TO_PC[k1]) % 12
        return interval
    return None


# ── 主标注管道 ───────────────────────────────────────────────────────────────

def annotate_file(token_path: str, output_dir: str) -> Optional[str]:
    """标注单个文件的段落结构。

    Args:
        token_path: token JSON 文件路径
        output_dir: 输出目录

    Returns:
        生成的 .sec.json 路径，或 None（跳过）
    """
    try:
        with open(token_path, 'r', encoding='utf-8') as f:
            tokens = json.load(f)
    except Exception as e:
        logger.warning(f'读取失败 {token_path}: {e}')
        return None

    if not isinstance(tokens, list) or len(tokens) < 50:
        return None

    # 解码 token → 字符串（支持整数 ID 和字符串两种格式）
    tokenizer = _get_tokenizer()
    token_strs = []
    for t in tokens:
        if isinstance(t, int):
            token_strs.append(tokenizer.decode_token(t))
        else:
            token_strs.append(str(t))

    # 提取小节特征
    bars = extract_bar_features(token_strs)
    if len(bars) < MIN_SECTION_BARS * 2:
        return None

    # 计算各信号
    key_signal = compute_key_change_signal(bars)
    density_signal = compute_density_shift_signal(bars)
    repeat_signal = compute_repeat_signal(bars)
    tempo_signal = compute_tempo_change_signal(bars)
    program_signal = compute_program_change_signal(bars)
    silence_signal = compute_silence_gap_signal(bars)

    # 加权融合
    n = len(bars)
    fused = np.zeros(n, dtype=np.float32)
    fused += key_signal * SIGNAL_WEIGHTS['key_change']
    fused += density_signal * SIGNAL_WEIGHTS['density_shift']
    fused += repeat_signal * SIGNAL_WEIGHTS['repeat']
    fused += tempo_signal * SIGNAL_WEIGHTS['tempo_change']
    fused += program_signal * SIGNAL_WEIGHTS['program_change']
    fused += silence_signal * SIGNAL_WEIGHTS['silence_gap']
    fused = fused / sum(SIGNAL_WEIGHTS.values())  # 归一化到 [0, 1]

    # Viterbi 解码
    boundaries = viterbi_segmentation(fused, MIN_SECTION_BARS)

    # 过滤低置信度边界
    for i in range(n):
        if boundaries[i] and fused[i] < BOUNDARY_THRESHOLD:
            boundaries[i] = 0

    # 如果没有高置信度边界 → 整曲作为一个段落
    if boundaries.sum() == 0:
        return _write_single_section(token_path, output_dir, bars)

    # 推断段落类型
    section_types = infer_section_types(bars, boundaries)

    # 构建输出
    boundary_list = [i for i in range(1, n) if boundaries[i]]
    section_boundaries = [0] + boundary_list + [n - 1]
    section_boundaries = sorted(set(section_boundaries))

    if len(section_boundaries) < 2:
        return _write_single_section(token_path, output_dir, bars)

    # 计算每个段落的属性
    n_tokens = len(token_strs)
    section_ids = np.zeros(n_tokens, dtype=np.int32)
    section_types_arr = np.zeros(n_tokens, dtype=np.int32)
    section_token_positions = []
    section_attrs = []

    for s in range(len(section_boundaries) - 1):
        sec_id = s + 1  # section_id 从 1 开始
        bar_start = section_boundaries[s]
        bar_end = section_boundaries[s + 1]

        # 计算该段落占用的 token 范围
        start_bar = bars[bar_start]
        end_bar = bars[min(bar_end, len(bars) - 1)]
        token_start = start_bar.get('_start_token', 0)
        token_end = end_bar.get('_end_token', n_tokens)

        # 分配 section_id 和 section_type
        for ti in range(token_start, min(token_end, n_tokens)):
            section_ids[ti] = sec_id
            sec_type = section_types[s] if s < len(section_types) else _section_fallback_id(s)
            section_types_arr[ti] = sec_type

        # Section token 位置：该段第一个 token
        if token_start < n_tokens:
            section_token_positions.append(token_start)

        # 段落属性
        n_bars_in_sec = bar_end - bar_start
        primary_key = _get_primary_key(bars, bar_start, bar_end)
        section_attrs.append({
            'bars': n_bars_in_sec,
            'key': _key_to_id(primary_key),
            'type': sec_type if s < len(section_types) else _section_fallback_id(s),
        })

    # 生成输出
    sec_path = _get_sec_path(token_path, output_dir)
    os.makedirs(os.path.dirname(sec_path), exist_ok=True)
    with open(sec_path, 'w', encoding='utf-8') as f:
        json.dump({
            'section_ids': section_ids.tolist(),
            'section_types': section_types_arr.tolist(),
            'section_token_positions': section_token_positions,
            'section_attrs': section_attrs,
        }, f, ensure_ascii=False)

    return sec_path


def _get_sec_path(token_path: str, output_dir: str) -> str:
    return str(Path(output_dir) / (Path(token_path).stem + '.sec.json'))


def _write_single_section(token_path: str, output_dir: str, bars: List[dict]) -> str:
    """整曲作为一个段落写入（无边界检测时使用）。"""
    sec_path = _get_sec_path(token_path, output_dir)
    try:
        with open(token_path, 'r', encoding='utf-8') as f:
            tokens = json.load(f)
    except Exception:
        return None
    n_tokens = len(tokens)
    n_bars = len(bars)
    primary_key = _get_primary_key(bars, 0, n_bars)
    sec_type = _section_fallback_id(0)  # section0 (通用)
    os.makedirs(os.path.dirname(sec_path), exist_ok=True)
    with open(sec_path, 'w', encoding='utf-8') as f:
        json.dump({
            'section_ids': [1] * n_tokens,
            'section_types': [sec_type] * n_tokens,
            'section_token_positions': [0],
            'section_attrs': [
                {
                    'bars': n_bars,
                    'key': _key_to_id(primary_key),
                    'type': sec_type,
                }
            ],
        }, f, ensure_ascii=False)
    return sec_path


def _get_primary_key(bars: List[dict], start: int, end: int) -> Optional[str]:
    """提取段落主调。"""
    keys = []
    for i in range(start, min(end, len(bars))):
        if bars[i]['key'] is not None:
            keys.append(bars[i]['key'])
    if keys:
        from collections import Counter
        return Counter(keys).most_common(1)[0][0]
    return 'C'


def _key_to_id(key_name: Optional[str]) -> int:
    """调名 → ID（与 tokenizer 保持一致）。"""
    KEY_IDS = {
        'C': 1, 'C#': 2, 'Db': 2, 'D': 3, 'D#': 4, 'Eb': 4,
        'E': 5, 'F': 6, 'F#': 7, 'Gb': 7, 'G': 8, 'G#': 9, 'Ab': 9,
        'A': 10, 'A#': 11, 'Bb': 11, 'B': 12, 'Cb': 12,
        'Am': 13, 'A#m': 14, 'Bbm': 14, 'Bm': 15, 'Cm': 16,
        'C#m': 17, 'Dm': 18, 'D#m': 19, 'Ebm': 19, 'Em': 20,
        'Fm': 21, 'F#m': 22, 'Gbm': 22, 'Gm': 23, 'G#m': 24, 'Abm': 24,
    }
    if key_name in KEY_IDS:
        return KEY_IDS[key_name]
    return 0  # unknown


def _annotate_worker(args):
    """imap_unordered worker: (token_path, output_dir) -> result."""
    return annotate_file(args[0], args[1])


# ── CLI ──────────────────────────────────────────────────────────────────────

def cmd_annotate(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    token_files = list(input_dir.glob('*.tokens'))
    if args.max_files > 0:
        token_files = token_files[args.start_index:args.start_index + args.max_files]
    logger.info(f'找到 {len(token_files)} 个 token 文件 (start={args.start_index})')

    total = len(token_files)
    if args.num_workers > 1:
        annotated = 0
        task_args = [(str(f), str(output_dir)) for f in token_files]
        with multiprocessing.Pool(args.num_workers) as pool:
            for i, result in enumerate(pool.imap_unordered(
                _annotate_worker, task_args, chunksize=500
            )):
                if result is not None:
                    annotated += 1
                if (i + 1) % 50000 == 0 or (i + 1) == total:
                    logger.info(f'  [{i+1}/{total}] ✓{annotated} ({(i+1)/total*100:.0f}%)')
    else:
        annotated = 0
        for i, f in enumerate(token_files):
            result = annotate_file(str(f), str(output_dir))
            if result is not None:
                annotated += 1
            if (i + 1) % 50000 == 0 or (i + 1) == total:
                logger.info(f'  [{i+1}/{total}] ✓{annotated} ({(i+1)/total*100:.0f}%)')

    logger.info(f'标注完成: {annotated}/{len(token_files)}')


def cmd_check(args):
    """检查标注覆盖率。"""
    input_dir = Path(args.input_dir)
    token_files = list(input_dir.glob('*.tokens'))
    sec_files = list(input_dir.glob('*.sec.json'))
    logger.info(f'Token 文件: {len(token_files)}')
    logger.info(f'.sec.json 文件: {len(sec_files)}')
    logger.info(f'覆盖率: {len(sec_files) / max(len(token_files), 1) * 100:.1f}%')
    # 统计有段落内容的
    sections_found = 0
    for sf in sec_files:
        try:
            with open(sf) as f:
                data = json.load(f)
            if data.get('section_attrs'):
                sections_found += 1
        except Exception:
            pass
    logger.info(f'含段落内容: {sections_found}/{len(sec_files)}')


def main():
    parser = argparse.ArgumentParser(description='段落结构标注管道')
    sub = parser.add_subparsers(dest='command', required=True)

    p_annotate = sub.add_parser('annotate', help='执行标注')
    p_annotate.add_argument('--input-dir', required=True)
    p_annotate.add_argument('--output-dir', default=None)
    p_annotate.add_argument('--num-workers', type=int, default=1)
    p_annotate.add_argument('--verbose', action='store_true')
    p_annotate.add_argument('--max-files', type=int, default=0)
    p_annotate.add_argument('--start-index', type=int, default=0)

    p_check = sub.add_parser('check', help='检查标注覆盖率')
    p_check.add_argument('--input-dir', required=True)

    args = parser.parse_args()

    if args.command == 'annotate':
        if args.output_dir is None:
            args.output_dir = args.input_dir
        cmd_annotate(args)
    elif args.command == 'check':
        cmd_check(args)


if __name__ == '__main__':
    main()
