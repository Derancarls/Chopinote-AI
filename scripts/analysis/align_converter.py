#!/usr/bin/env python3
"""PDMX/MIDI 转换器与 MusicXML 转换器对齐验证工具。

桥接测试: PDMX/MIDI → REMI → MusicXML → REMI，比较 token 序列一致性。
覆盖度报告: 静态分析各转换器覆盖的 token 类型。

Usage:
    python scripts/align_converter.py --format pdmx --data-dir /path/to/data --sample 50
    python scripts/align_converter.py --report
    python scripts/align_converter.py --format midi --single /path/to/file.mid
"""
import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from chopinote_dataset.converter import MusicXMLToREMI, PDMXToREMI
from chopinote_dataset.fast_converter import FastMIDIToREMI
from chopinote_dataset.renderer import REMIToMusicXML
from chopinote_dataset.tokenizer import REMITokenizer


# ── Token 类型覆盖度分析 ──────────────────────────────────────

def coverage_report() -> dict:
    """静态分析每个转换器覆盖了哪些 token 类型。

    通过扫描每个转换器的源码查找 REMITokenizer.<TOKEN> 及 self/空字符串引用。
    注意: _BaseREMI 中的 token 引用在子类 scan 中也可能被间接覆盖。
    """
    tokenizer = REMITokenizer(16, 8)

    token_types = {}
    for attr in dir(tokenizer):
        if attr.isupper() and not attr.startswith('_'):
            val = getattr(tokenizer, attr)
            if isinstance(val, str) and val.startswith('<'):
                token_types[attr] = val

    import inspect

    def _class_source(mod, class_name: str) -> str:
        """Return the source of a specific class within a module.
        Includes the shared base class for converter.py classes.
        """
        sources = []
        try:
            base = getattr(mod, '_BaseREMI', None)
            if base is not None:
                sources.append(inspect.getsource(base))
        except (TypeError, OSError):
            pass
        try:
            cls = getattr(mod, class_name)
            sources.append(inspect.getsource(cls))
        except (TypeError, OSError, AttributeError):
            pass
        return '\n'.join(sources)

    # Load modules
    from chopinote_dataset import converter as conv_mod
    from chopinote_dataset import fast_converter as fast_mod

    src_musicxml = _class_source(conv_mod, 'MusicXMLToREMI')
    src_pdmx = _class_source(conv_mod, 'PDMXToREMI')
    src_midi = _class_source(conv_mod, 'MIDIToREMI')
    src_fast = _class_source(fast_mod, 'FastMIDIToREMI')

    def _scan(module_src: str, class_name: str) -> set:
        """Find what token constant names appear near a class definition."""
        refs = set()
        for attr_name, tok_val in token_types.items():
            # Check if the token value string appears in module source
            # (e.g. '<Bar>' appears as REMITokenizer.BAR which resolves to '<Bar>',
            #  but the string '<Bar>' may not be in source. Check both.)
            if tok_val in module_src:
                refs.add(attr_name)
                continue
            # Check for constant reference patterns
            for pat in [f'REMITokenizer.{attr_name}',
                        f'self.{attr_name}',
                        f'.{attr_name}']:
                if pat in module_src:
                    refs.add(attr_name)
                    break
        return refs

    return {
        'MusicXMLToREMI': _scan(src_musicxml, 'MusicXMLToREMI'),
        'PDMXToREMI': _scan(src_pdmx, 'PDMXToREMI'),
        'MIDIToREMI': _scan(src_midi, 'MIDIToREMI'),
        'FastMIDIToREMI': _scan(src_fast, 'FastMIDIToREMI'),
    }


def print_coverage_report():
    """打印 markdown 格式的覆盖度报告。"""
    report = coverage_report()

    # Get token types from coverage_report
    tokenizer = REMITokenizer(16, 8)
    token_types = {}
    for attr in dir(tokenizer):
        if attr.isupper() and not attr.startswith('_'):
            val = getattr(tokenizer, attr)
            if isinstance(val, str) and val.startswith('<'):
                token_types[attr] = val

    converters = ['MusicXMLToREMI', 'PDMXToREMI', 'MIDIToREMI', 'FastMIDIToREMI']

    print('# 转换器 Token 覆盖度报告\n')
    print(f'| Token 类型 | {" | ".join(f"{c}" for c in converters)} | 备注 |')
    print(f'|{"|".join(["---"] * (len(converters) + 2))}|')

    for attr, tok_val in token_types.items():
        if tok_val in ('<PAD>', '<BOS>', '<EOS>', '<MASK>'):
            continue  # skip special tokens

        row = [f'`{tok_val}`']
        for c in converters:
            row.append('✅' if attr in report[c] else '❌')
        row.append('')
        print(f'| {" | ".join(row)} |')

    # Summary
    print('\n## 汇总\n')
    for c in converters:
        total = len(token_types) - 4  # exclude special tokens
        covered = len(report[c])
        pct = covered / total * 100
        print(f'- **{c}**: {covered}/{total} ({pct:.0f}%)')


# ── 桥接测试 ──────────────────────────────────────────────────

def bridge_roundtrip_pdmx(pdmx_path: str,
                           grid_size: int = 16,
                           velocity_levels: int = 8) -> dict:
    """PDMX → REMI → MusicXML → REMI 桥接测试。

    Returns:
        差异报告 dict
    """
    tokenizer = REMITokenizer(grid_size, velocity_levels)
    renderer = REMIToMusicXML(grid_size, velocity_levels)
    pdmx_converter = PDMXToREMI(grid_size, velocity_levels)
    mx_converter = MusicXMLToREMI(grid_size, velocity_levels)

    # Step 1: PDMX → REMI_A
    tokens_a, _ = pdmx_converter.convert(pdmx_path, collect_metadata=True)
    if not tokens_a:
        return {'error': 'PDMX 转换失败', 'path': pdmx_path}

    # Step 2: REMI_A → MusicXML (write to temp)
    tmp_musicxml = pdmx_path + '.bridge_test.musicxml'
    try:
        events_a = tokenizer.detokenize(tokens_a)
        renderer.render(events_a, tmp_musicxml)
    except Exception as e:
        return {'error': f'渲染失败: {e}', 'path': pdmx_path}

    # Step 3: MusicXML → REMI_B
    tokens_b, _ = mx_converter.convert(tmp_musicxml, collect_metadata=True)

    # Cleanup
    try:
        os.remove(tmp_musicxml)
    except OSError:
        pass

    if not tokens_b:
        return {'error': 'MusicXML 再转换失败', 'path': pdmx_path}

    return _compare_tokens(tokens_a, tokens_b, tokenizer)


def bridge_roundtrip_midi(midi_path: str,
                           grid_size: int = 16,
                           velocity_levels: int = 8,
                           use_fast: bool = True) -> dict:
    """MIDI → REMI → MusicXML → REMI 桥接测试。

    Args:
        midi_path: MIDI 文件路径
        use_fast: True = FastMIDIToREMI, False = MIDIToREMI (music21)
    """
    tokenizer = REMITokenizer(grid_size, velocity_levels)
    renderer = REMIToMusicXML(grid_size, velocity_levels)
    mx_converter = MusicXMLToREMI(grid_size, velocity_levels)

    if use_fast:
        converter = FastMIDIToREMI(grid_size, velocity_levels)
        tokens_a, _ = converter.convert(midi_path, collect_metadata=True)
    else:
        converter = type('MIDIConv', (), {})()
        conv = MIDIToREMI(grid_size, velocity_levels)
        tokens_a, _ = conv.convert(midi_path, collect_metadata=True)

    if not tokens_a:
        return {'error': 'MIDI 转换失败', 'path': midi_path}

    tmp_musicxml = midi_path + '.bridge_test.musicxml'
    try:
        events_a = tokenizer.detokenize(tokens_a)
        renderer.render(events_a, tmp_musicxml)
    except Exception as e:
        return {'error': f'渲染失败: {e}', 'path': midi_path}

    tokens_b, _ = mx_converter.convert(tmp_musicxml, collect_metadata=True)

    try:
        os.remove(tmp_musicxml)
    except OSError:
        pass

    if not tokens_b:
        return {'error': 'MusicXML 再转换失败', 'path': midi_path}

    return _compare_tokens(tokens_a, tokens_b, tokenizer)


def _compare_tokens(tokens_a: List[int], tokens_b: List[int],
                     tokenizer: REMITokenizer) -> dict:
    """比较两个 token 序列，返回结构化差异报告。"""
    # Decode tokens to events for per-type analysis
    events_a = tokenizer.detokenize(tokens_a)
    events_b = tokenizer.detokenize(tokens_b)

    # Basic stats
    total_a = len(tokens_a)
    total_b = len(tokens_b)

    # Event-level comparison
    max_len = max(len(events_a), len(events_b))
    min_len = min(len(events_a), len(events_b))

    # Per-type diff analysis
    type_mismatches = defaultdict(int)
    type_total = defaultdict(int)
    exact_matches = 0
    total_comparable = 0
    pos_offsets = 0
    pitch_diffs = 0

    for i in range(min_len):
        ea, eb = events_a[i], events_b[i]
        ta, va = ea
        tb, vb = eb

        type_total[ta] += 1

        if ta == tb and va == vb:
            exact_matches += 1
        else:
            type_mismatches[ta] += 1
            if ta == '<Position' and tb == '<Position':
                pos_offsets += 1
            if ta == '<Note_ON' and tb == '<Note_ON':
                pitch_diffs += 1

    total_comparable = min_len

    # Extra tokens (present in A but not in B)
    extra_a = max(0, len(events_a) - min_len)
    extra_b = max(0, len(events_b) - min_len)

    # Compute per-type match rate
    per_type = {}
    for ttype in sorted(set(list(type_total.keys()) + list(type_mismatches.keys()))):
        total = type_total.get(ttype, 0)
        mism = type_mismatches.get(ttype, 0)
        matched = total - mism
        rate = matched / total if total > 0 else 1.0
        per_type[ttype.split()[0]] = {
            'total': total,
            'matched': matched,
            'rate': f'{rate:.1%}',
        }

    overall_rate = exact_matches / total_comparable if total_comparable > 0 else 0

    return {
        'tokens_a': total_a,
        'tokens_b': total_b,
        'events_a': len(events_a),
        'events_b': len(events_b),
        'exact_matches': exact_matches,
        'total_comparable': total_comparable,
        'overall_consistency': f'{overall_rate:.1%}',
        'extra_events_in_a': extra_a,
        'extra_events_in_b': extra_b,
        'position_offsets': pos_offsets,
        'pitch_differences': pitch_diffs,
        'per_type': per_type,
    }


# ── 批量运行 ──────────────────────────────────────────────────

def run_batch(file_list: List[str],
              format_type: str,
              grid_size: int = 16,
              velocity_levels: int = 8) -> List[dict]:
    """对文件列表批量跑桥接测试。"""
    results = []

    if format_type == 'pdmx':
        for path in file_list:
            r = bridge_roundtrip_pdmx(path, grid_size, velocity_levels)
            results.append(r)
            _print_result(path, r)

    elif format_type == 'midi':
        for path in file_list:
            r = bridge_roundtrip_midi(path, grid_size, velocity_levels)
            results.append(r)
            _print_result(path, r)

    return results


def _print_result(path: str, result: dict):
    """打印单文件结果。"""
    if 'error' in result:
        logger.error(f'  {path}: {result["error"]}')
    else:
        logger.info(f'  {path}: consistency={result["overall_consistency"]} '
                    f'(events {result["exact_matches"]}/{result["total_comparable"]})')


def summarize(results: List[dict]):
    """汇总批量结果。"""
    total = len(results)
    errors = [r for r in results if 'error' in r]
    successes = [r for r in results if 'error' not in r]

    print(f'\n{"═" * 50}')
    print(f'桥接测试汇总')
    print(f'{"═" * 50}')
    print(f'总文件: {total}')
    print(f'成功:   {len(successes)}')
    print(f'失败:   {len(errors)}')

    if errors:
        print(f'\n错误详情:')
        for r in errors[:10]:
            print(f'  {r["path"]}: {r["error"]}')

    if successes:
        consistencies = []
        for r in successes:
            try:
                consistencies.append(float(r['overall_consistency'].rstrip('%')))
            except (ValueError, KeyError):
                pass

        if consistencies:
            avg = sum(consistencies) / len(consistencies)
            min_c = min(consistencies)
            max_c = max(consistencies)
            print(f'\n一致性: avg={avg:.1f}% min={min_c:.1f}% max={max_c:.1f}%')

            # Per-type aggregate
            type_agg = defaultdict(lambda: {'total': 0, 'matched': 0})
            for r in successes:
                for ttype, stats in r.get('per_type', {}).items():
                    type_agg[ttype]['total'] += stats['total']
                    type_agg[ttype]['matched'] += stats['matched']

            print(f'\n按 Token 类型一致性:')
            for ttype in sorted(type_agg.keys()):
                s = type_agg[ttype]
                rate = s['matched'] / s['total'] if s['total'] > 0 else 1.0
                bar_len = int(rate * 30)
                bar = '▓' * bar_len + '░' * (30 - bar_len)
                print(f'  {ttype:20s} {bar} {rate:.1%} ({s["matched"]}/{s["total"]})')


# ── CLI ────────────────────────────────────────────────────────

def find_files(data_dir: str, ext: str, max_samples: int = 0) -> List[str]:
    """递归查找指定扩展名的文件。"""
    files = []
    for root, _, fnames in os.walk(data_dir):
        for f in fnames:
            if f.endswith(ext):
                files.append(os.path.join(root, f))
    files.sort()
    if max_samples > 0 and len(files) > max_samples:
        # Deterministic sampling: take evenly spaced files
        step = len(files) / max_samples
        files = [files[int(i * step)] for i in range(max_samples)]
    return files


def main():
    parser = argparse.ArgumentParser(
        description='PDMX/MIDI ↔ MusicXML 转换器对齐验证工具')
    parser.add_argument('--format', choices=['pdmx', 'midi'],
                        help='转换器类型')
    parser.add_argument('--data-dir', help='数据目录（递归扫描）')
    parser.add_argument('--single', help='单个文件测试')
    parser.add_argument('--sample', type=int, default=50,
                        help='采样数量 (默认 50)')
    parser.add_argument('--report', action='store_true',
                        help='打印覆盖度报告')
    parser.add_argument('--use-music21-midi', action='store_true',
                        help='MIDI 测试使用 music21 MIDIToREMI (默认 FastMIDIToREMI)')

    args = parser.parse_args()

    if args.report:
        print_coverage_report()
        return

    if not args.format:
        parser.print_help()
        return

    if args.single:
        ext = '.pdmx' if args.format == 'pdmx' else '.mid'
        if not os.path.exists(args.single):
            logger.error(f'文件不存在: {args.single}')
            sys.exit(1)
        if args.format == 'pdmx':
            r = bridge_roundtrip_pdmx(args.single)
        else:
            r = bridge_roundtrip_midi(args.single, use_fast=not args.use_music21_midi)
        import json
        print(json.dumps(r, indent=2, ensure_ascii=False))
        return

    if not args.data_dir:
        logger.error('需要 --data-dir 或 --single')
        sys.exit(1)

    ext = '.pdmx' if args.format == 'pdmx' else '.mid'
    files = find_files(args.data_dir, ext, args.sample)
    if not files:
        logger.error(f'在 {args.data_dir} 中未找到 {ext} 文件')
        sys.exit(1)

    logger.info(f'找到 {len(files)} 个 {ext} 文件，运行桥接测试...')
    results = run_batch(files, args.format)
    summarize(results)


if __name__ == '__main__':
    main()
