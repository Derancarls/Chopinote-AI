"""和弦标注管道：依赖 .sec.json 的 key 信息，模板匹配 + 置信度过滤。

输入: token 序列 + .sec.json（段落标注 + key 信息）
输出: .chord.json sidecar 文件

用法:
    python scripts/chord_annotator.py --tokens-dir data/processed/tokens_v2/ \
        --file-list data/processed/train.txt --output-suffix .chord.json
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from chopinote_dataset.tokenizer import REMITokenizer

logger = logging.getLogger(__name__)

# ── 和弦模板：根音偏移量的音级集合 ──────────────────────
CHORD_QUALITY_TEMPLATES: dict[str, list[int]] = {
    'M':     [0, 4, 7],        # 大三和弦
    'm':     [0, 3, 7],        # 小三和弦
    'dim':   [0, 3, 6],        # 减三和弦
    'aug':   [0, 4, 8],        # 增三和弦
    'dom7':  [0, 4, 7, 10],    # 属七
    'maj7':  [0, 4, 7, 11],    # 大七
    'min7':  [0, 3, 7, 10],    # 小七
    'dim7':  [0, 3, 6, 9],     # 减七
    'hdim7': [0, 3, 6, 10],    # 半减七
}

# 质量 → 罗马数字功能映射（无调性上下文，仅映射三和弦质量）
QUALITY_TO_FUNC_BASE: dict[str, str] = {
    'M': 'I', 'm': 'i', 'dim': 'ii°', 'aug': 'III',
}

# ── 30 调性 → 音阶音级 (0-11) ──────────────────────────
# 大调 = Ionian, 小调 = Aeolian
MAJOR_INTERVALS = [0, 2, 4, 5, 7, 9, 11]    # Whole-half pattern from tonic
MINOR_INTERVALS = [0, 2, 3, 5, 7, 8, 10]     # Natural minor

# 音阶音级 → 罗马数字（大调上下文）
_DEGREE_TO_MAJOR_FUNC = {
    0: 'I', 1: 'ii', 2: 'iii', 3: 'IV', 4: 'V', 5: 'vi', 6: 'vii°',
}
# 音阶音级 → 罗马数字（小调上下文）
_DEGREE_TO_MINOR_FUNC = {
    0: 'i', 1: 'ii°', 2: 'III', 3: 'iv', 4: 'V', 5: 'VI', 6: 'vii°',
}

# 功能名 → ID（按 CHORD_FUNCTIONS 顺序）
FUNC_NAME_TO_ID: dict[str, int] = {
    'I': 1, 'i': 2, 'ii': 3, 'ii°': 4, 'iii': 5, 'III': 6,
    'IV': 7, 'iv': 8, 'V': 9, 'vi': 10, 'VI': 11, 'vii°': 12,
    'N': 13, 'It6': 14, 'Fr6': 15, 'Ger6': 16,
}

# 转位名 → ID（按 CHORD_INVERSIONS 顺序）
INV_NAME_TO_ID: dict[str, int] = {
    'Root': 1, '1st': 2, '2nd': 3, '3rd': 4,
}


def _get_scale(key_name: str) -> tuple[int, list[int], bool]:
    """返回 (tonic_pc, scale_pcs, is_major) for given key name."""
    KEY_PC = {
        'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
        'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
        'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11, 'Cb': 11,
    }
    is_minor = key_name.endswith('m')
    root = key_name[:-1] if is_minor else key_name
    tonic_pc = KEY_PC.get(root, 0)
    intervals = MINOR_INTERVALS if is_minor else MAJOR_INTERVALS
    scale = [(tonic_pc + iv) % 12 for iv in intervals]
    return tonic_pc, scale, not is_minor


def _match_pcs_to_chord(pcs: set[int], key_name: str) -> Optional[dict]:
    """将 pitch class 集合匹配到和弦功能和转位。

    Returns: {func_id, inv_id, has_7th, confidence} or None
    """
    if len(pcs) < 2:
        return None

    tonic_pc, scale, is_major = _get_scale(key_name)
    degree_map = _DEGREE_TO_MAJOR_FUNC if is_major else _DEGREE_TO_MINOR_FUNC

    best_score = 0.0
    best_result = None

    pcs_sorted = sorted(pcs)
    bass_pc = pcs_sorted[0]  # 最低音

    for degree_idx, scale_pc in enumerate(scale):
        for quality, template in CHORD_QUALITY_TEMPLATES.items():
            # 该和弦在调性内的预期音级
            expected = {(scale_pc + t) % 12 for t in template}
            matched = len(pcs & expected)
            extra = len(pcs - expected)
            missing = len(expected - pcs)

            # 分数: 匹配数 - 额外音惩罚 - 缺失音惩罚
            score = matched - extra * 0.5 - missing * 0.3
            if score > best_score and matched >= len(pcs) - 1:
                best_score = score
                func_base = degree_map[degree_idx]

                # 根据质量修正功能名
                func_name = _resolve_func_name(func_base, quality, template)

                if func_name:
                    has_7th = len(template) == 4
                    inv_name = _detect_inversion(template, scale_pc, bass_pc, has_7th)

                    conf = min(1.0, score / max(1, len(pcs)))
                    best_result = {
                        'func_id': FUNC_NAME_TO_ID.get(func_name, -1),
                        'inv_id': INV_NAME_TO_ID.get(inv_name, 1),
                        'has_7th': has_7th,
                        'confidence': conf,
                        'func_name': func_name,
                        'inv_name': inv_name,
                    }

    if best_result and best_result['confidence'] >= 0.8:
        if best_result['func_id'] > 0:
            return best_result
    return None


def _resolve_func_name(base: str, quality: str, template: list[int]) -> Optional[str]:
    """根据模板质量解析最终功能名。"""
    # 基础映射：三和弦质量决定大小写
    if quality == 'M':
        # 保持大写，特殊处理有问题的
        if base.endswith('°'):
            return base  # Should not happen for M quality
        return base.upper() if base[0].islower() else base
    elif quality == 'm':
        return base.lower() if base[0].isupper() else base
    elif quality == 'dim':
        return base.rstrip('I') + 'ii°' if 'vii' in base else base + '°'
    elif quality == 'aug':
        return 'III'  # augmented is always major-based
    elif quality in ('dom7', 'maj7', 'min7', 'dim7', 'hdim7'):
        # 七和弦 — 返回三和弦基础功能名（Chord7 token 单独标记）
        if quality == 'dom7':
            return base.upper() if base in ('V',) else base
        elif quality == 'maj7':
            return base.upper()
        elif quality == 'min7':
            return base.lower()
        elif quality in ('dim7', 'hdim7'):
            return base + '°' if 'vii' in base else base
    return base


def _detect_inversion(template: list[int], root_pc: int, bass_pc: int,
                      has_7th: bool) -> str:
    """检测转位: 0=Root, 1=1st, 2=2nd, 3=3rd (仅七和弦)。"""
    if has_7th and bass_pc == (root_pc + template[3]) % 12:
        return '3rd'
    if bass_pc == (root_pc + template[2]) % 12:
        return '2nd'
    if bass_pc == (root_pc + template[1]) % 12:
        return '1st'
    return 'Root'


def _get_current_key(token_pos: int, sec_data: dict, start: int = 0) -> str:
    """根据 token 位置确定当前调性（来自 sec.json 的 section_attrs）。"""
    positions = sec_data.get('section_token_positions', [])
    attrs = sec_data.get('section_attrs', [])
    current_key = 'C'  # default
    for i, pos in enumerate(positions):
        if pos <= token_pos and i < len(attrs):
            key_idx = attrs[i].get('key', 0)
            from chopinote_dataset.tokenizer import REMITokenizer
            if 0 <= key_idx < len(REMITokenizer.KEY_NAMES):
                current_key = REMITokenizer.KEY_NAMES[key_idx]
    return current_key


def annotate_chords(token_ids: list[int], sec_data: Optional[dict] = None,
                    grid_size: int = 16) -> dict:
    """对 token 序列进行和弦标注。

    Args:
        token_ids: REMI token ID 列表
        sec_data: .sec.json 内容（用于 key 信息），None 时默认 C major
        grid_size: position 颗粒度

    Returns:
        chord.json 格式的 dict:
        {
            'chord_func_ids': [int],    # 每 token 当前和弦功能 ID (0=无和弦)
            'chord_inv_ids': [int],     # 每 token 当前转位 ID (0=无和弦)
            'chord_token_positions': [int],  # Chord token 发射位置
            'chord_attrs': [{func, inv, has_7th}],  # 每个和弦的属性
        }
    """
    tokenizer = REMITokenizer(grid_size=grid_size)
    T = len(token_ids)

    chord_func_ids = [0] * T
    chord_inv_ids = [0] * T
    chord_positions = []
    chord_attrs = []

    bar_id = tokenizer.bar_token_id

    # 按小节和拍位组织 token
    current_bar = 0
    current_position = 0
    current_key = 'C'

    # 收集每个 position 窗口内的音高
    window_pcs: set[int] = set()
    window_start = 0
    last_chord_func = 0
    last_chord_inv = 0
    last_emission_pos = -1

    # 预处理保证每小节有 <Key> token，从 token 序列追踪调性最精确。
    # sec.json 仅在序列中无 <Key> 时 fallback。
    seen_key_token = False

    for pos, tid in enumerate(token_ids):
        ts = tokenizer.decode_token(tid)

        if ts.startswith('<Key '):
            current_key = ts[len('<Key') + 1:-1]
            seen_key_token = True
        elif not seen_key_token and sec_data:
            current_key = _get_current_key(pos, sec_data)

        # 追踪位置
        if tid == bar_id:
            current_bar += 1
            current_position = 0

        if ts.startswith('<Position '):
            current_position = int(ts[len('<Position') + 1:-1])

        # 收集音高
        if ts.startswith('<Note_ON '):
            interval = int(ts[len('<Note_ON') + 1:-1])
            # 从当前 key 推导绝对 pitch class
            tonic_pc, _, _ = _get_scale(current_key)
            pc = (tonic_pc + interval) % 12
            window_pcs.add(pc)
            if window_start == 0:
                window_start = pos

        # 在拍位切换或小节线处进行和弦识别
        is_position_change = ts.startswith('<Position ') and window_pcs
        is_bar_line = tid == bar_id
        is_end = (pos == T - 1)

        if (is_position_change or is_bar_line or is_end) and len(window_pcs) >= 2:
            result = _match_pcs_to_chord(window_pcs, current_key)

            if result and result['func_id'] > 0:
                # 新和弦 — 发射 Chord token
                chord_positions.append(window_start)
                chord_attrs.append({
                    'func': result['func_id'],
                    'inv': result['inv_id'],
                    'has_7th': result['has_7th'],
                })
                last_chord_func = result['func_id']
                last_chord_inv = result['inv_id']
                last_emission_pos = window_start
            elif last_chord_func > 0:
                # 稀疏织体退化：延续上一个和弦
                pass

            window_pcs.clear()
            window_start = 0

        # 填充当前和弦上下文
        chord_func_ids[pos] = last_chord_func
        chord_inv_ids[pos] = last_chord_inv

    return {
        'chord_func_ids': chord_func_ids,
        'chord_inv_ids': chord_inv_ids,
        'chord_token_positions': chord_positions,
        'chord_attrs': chord_attrs,
    }


def _key_coverage_stats(token_ids: list[int]) -> dict:
    """统计调性覆盖：扫描 token 序列，统计缺少 <Key> 的小节。

    在每个 <Bar> 边界处检查前一小节内是否有 <Key> token（而非累积标记），
    避免首小节 <Bar> 先于 <Key> 导致的误报。最后一个小节在循环结束后检查。

    Returns: {'total_bars': int, 'keyless_bars': int, 'keyless_bar_indices': list[int]}
    """
    from chopinote_dataset.tokenizer import REMITokenizer
    tk = REMITokenizer(grid_size=16, velocity_levels=8)
    bar_id = tk.bar_token_id

    total_bars = 0
    keyless_bars = 0
    keyless_bar_indices = []
    found_key_in_measure = False

    for tid in token_ids:
        if tid == bar_id:
            # 检查前一小节（跳过首个 <Bar>，因为它无前一小节）
            if total_bars > 0 and not found_key_in_measure:
                keyless_bars += 1
                keyless_bar_indices.append(total_bars)
            total_bars += 1
            found_key_in_measure = False  # 重置，开始检查新小节
        elif tid in _key_token_ids(tk):
            found_key_in_measure = True

    # 最后一个小节收尾检查
    if total_bars > 0 and not found_key_in_measure:
        keyless_bars += 1
        keyless_bar_indices.append(total_bars)

    return {
        'total_bars': total_bars,
        'keyless_bars': keyless_bars,
        'keyless_bar_indices': keyless_bar_indices,
    }


def _key_token_ids(tk) -> set[int]:
    """返回所有 <Key X> token 的 ID 集合（缓存）。"""
    if not hasattr(_key_token_ids, '_cache'):
        _key_token_ids._cache = {
            tid for token_str, tid in tk._token_to_id.items()
            if token_str.startswith('<Key ') and token_str.endswith('>')
        }
    return _key_token_ids._cache


def process_file(token_path: str, output_suffix: str = '.chord.json'):
    """处理单个 token 文件，生成和弦标注 sidecar。"""
    token_path = Path(token_path)
    sec_path = token_path.with_suffix('.sec.json')
    chord_path = token_path.with_suffix(output_suffix)

    if not token_path.exists():
        logger.warning(f'Token 文件不存在: {token_path}')
        return

    # 加载 token 序列
    with open(token_path, 'r') as f:
        token_ids = json.load(f)

    # 加载段落数据（用于 key 信息）
    sec_data = None
    if sec_path.exists():
        try:
            with open(sec_path, 'r') as f:
                sec_data = json.load(f)
        except Exception:
            pass

    # 标注
    result = annotate_chords(token_ids, sec_data)

    # 写入
    with open(chord_path, 'w') as f:
        json.dump(result, f)

    n_chords = len(result['chord_token_positions'])
    logger.info(f'{token_path.name}: {n_chords} chords annotated → {chord_path.name}')

    # 调性覆盖检测：检查无 <Key> token 的小节（预处理查漏补缺）
    coverage = _key_coverage_stats(token_ids)
    if coverage['keyless_bars'] > 0:
        pct = coverage['keyless_bars'] / max(coverage['total_bars'], 1) * 100
        logger.warning(
            f'{token_path.name}: {coverage["keyless_bars"]}/{coverage["total_bars"]} '
            f'小节无调性标记 ({pct:.1f}%)'
        )
        if coverage['keyless_bars'] <= 5:
            logger.warning(f'  缺 key 的小节索引: {coverage["keyless_bar_indices"]}')


def process_batch(file_list: str, tokens_dir: str,
                  output_suffix: str = '.chord.json', max_files: int = 0):
    """批量处理文件列表。

    Args:
        file_list: train.txt / val.txt 等文件列表
        tokens_dir: token 文件目录
        output_suffix: 输出文件后缀
        max_files: 限制处理文件数 (0 = 不限制)
    """
    tokens_dir = Path(tokens_dir)

    with open(file_list, 'r') as f:
        paths = [line.strip() for line in f if line.strip()]

    for i, fp in enumerate(paths):
        if max_files > 0 and i >= max_files:
            break
        token_path = tokens_dir / Path(fp).name
        try:
            process_file(str(token_path), output_suffix)
        except Exception as e:
            logger.error(f'处理失败 {fp}: {e}')


def process_directory(tokens_dir: str, output_suffix: str = '.chord.json',
                      max_files: int = 0, start_index: int = 0):
    """处理目录下所有 token 文件。"""
    tokens_dir = Path(tokens_dir)
    token_files = sorted(tokens_dir.glob('*.tokens'))
    if max_files > 0:
        token_files = token_files[start_index:start_index + max_files]
    logger.info(f'找到 {len(token_files)} 个 token 文件 (start={start_index})')

    processed = 0
    errors = 0
    for i, tf in enumerate(token_files):
        if max_files > 0 and i >= max_files:
            break
        try:
            process_file(str(tf), output_suffix)
            processed += 1
        except Exception as e:
            errors += 1
            if errors <= 5:
                logger.error(f'处理失败 {tf.name}: {e}')
        if (i + 1) % 50000 == 0:
            logger.info(f'  [{i+1}/{len(token_files)}] ✓{processed} ✗{errors}')

    logger.info(f'和弦标注完成: ✓{processed} ✗{errors} / {min(len(token_files), max_files or len(token_files))}')


if __name__ == '__main__':
    import argparse
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    parser = argparse.ArgumentParser(description='和弦标注管道')
    parser.add_argument('--tokens-dir', default='/root/autodl-tmp/data/processed/tokens_v3')
    parser.add_argument('--file-list', help='文件列表 (train.txt/val.txt)')
    parser.add_argument('--single-file', help='单个 token 文件路径')
    parser.add_argument('--output-suffix', default='.chord.json')
    parser.add_argument('--max-files', type=int, default=0)
    parser.add_argument('--start-index', type=int, default=0)
    args = parser.parse_args()

    if args.single_file:
        process_file(args.single_file, args.output_suffix)
    elif args.file_list:
        process_batch(args.file_list, args.tokens_dir,
                      args.output_suffix, args.max_files)
    else:
        process_directory(args.tokens_dir, args.output_suffix,
                          args.max_files, args.start_index)
