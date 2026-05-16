"""从 metadata 预计算统计基准分布。

所有统计对比的基础参照系。从 1.6M metadata 文件中提取统计量，
按 all/genre/composer/timesig 分组聚合，输出 JSON 基准文件。

用法:
    python -m chopinote_evaluator.benchmarks.build_benchmarks
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# ── Krumhansl-Schmuckler Key Profiles ──────────────────────
# 24 个大小调的 pitch class profile（12 个半音的期望占比）
# 来源: Krumhansl & Schmuckler (1990)

KS_MAJOR_PROFILES = {
    'C':  [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    'G':  [2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29],
    'D':  [2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66],
    'A':  [3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39],
    'E':  [2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19],
    'B':  [5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52],
    'F#': [2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09],
    'C#': [4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38],
    'F':  [4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33],
    'Bb': [2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48],
    'Eb': [3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23],
    'Ab': [2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35],
}

KS_MINOR_PROFILES = {
    'A':  [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    'E':  [3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34],
    'B':  [3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69],
    'F#': [2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98],
    'C#': [3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75],
    'G#': [4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54],
    'D#': [2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53],
    'A#': [3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60],
    'D':  [2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38],
    'G':  [5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52],
    'C':  [3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68],
    'F':  [2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33],
}

# 音名转 MIDI class
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def get_ks_profile(key_name: str) -> list[float] | None:
    """根据调名获取 K-S profile 归一化向量。

    参数:
        key_name: 调名，如 "C", "G_major", "A_minor", "Bb"

    返回:
        12 元素的概率分布向量，或 None（不识别时）
    """
    # 解析调名
    key_name = key_name.strip().replace(' ', '_')
    mode = 'major'
    tonic = key_name

    if '_' in key_name:
        parts = key_name.split('_')
        tonic = parts[0]
        if len(parts) > 1:
            mode = parts[1]

    if tonic in KS_MAJOR_PROFILES and mode == 'major':
        raw = KS_MAJOR_PROFILES[tonic]
    elif tonic in KS_MINOR_PROFILES and mode == 'minor':
        raw = KS_MINOR_PROFILES[tonic]
    else:
        return None

    total = sum(raw)
    return [v / total for v in raw]


def get_all_ks_profiles() -> dict[str, list[float]]:
    """返回所有 24 个 K-S profile（归一化后）。"""
    profiles = {}
    for tonic, raw in KS_MAJOR_PROFILES.items():
        total = sum(raw)
        profiles[f'{tonic}_major'] = [v / total for v in raw]
    for tonic, raw in KS_MINOR_PROFILES.items():
        total = sum(raw)
        profiles[f'{tonic}_minor'] = [v / total for v in raw]
    return profiles


# ── Benchmark Builder ────────────────────────────────────────


def build_benchmarks(
    metadata_dir: str = '/root/autodl-tmp/data/processed/metadata/',
    output_dir: str | None = None,
    min_samples: int = 30,
) -> dict[str, Any]:
    """从 metadata 构建基准分布。

    参数:
        metadata_dir: metadata 文件所在目录
        output_dir: 输出路径，None 则不写文件
        min_samples: 分组最小样本数

    返回:
        {group_name: benchmarks_dict}
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(output_dir, exist_ok=True)

    # 收集所有 metadata
    all_files = [f for f in os.listdir(metadata_dir) if f.endswith('.meta.json')]
    print(f'读取 {len(all_files)} 个 metadata 文件...')

    # 分组收集器
    all_samples = []
    by_timesig: dict[str, list] = defaultdict(list)
    by_composer: dict[str, list] = defaultdict(list)
    by_source: dict[str, list] = defaultdict(list)

    for i, filename in enumerate(all_files):
        if i % 200000 == 0 and i > 0:
            print(f'  处理进度: {i}/{len(all_files)}')

        filepath = os.path.join(metadata_dir, filename)
        try:
            with open(filepath, 'r') as f:
                md = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        # 提取基本统计量
        sample = _extract_stats(md)
        if sample is None:
            continue

        all_samples.append(sample)

        # 按拍号分组
        ts = sample.get('time_signature')
        if ts and ts != 'unknown':
            by_timesig[ts].append(sample)

        # 按作曲家分组
        composer = sample.get('composer', '')
        if composer and composer not in ('Unknown', '', 'unknown'):
            by_composer[composer].append(sample)

        # 按来源分组
        src = sample.get('source_format', 'midi') or 'midi'
        by_source[src].append(sample)

    print(f'  有效样本: {len(all_samples)}')

    # 构建各分组基准
    benchmarks = {}

    # all 组
    if len(all_samples) >= min_samples:
        benchmarks['all'] = _aggregate(all_samples)

    # 按拍号
    for ts, samples in by_timesig.items():
        if len(samples) >= min_samples:
            benchmarks[f'timesig_{ts.replace("/", "_")}'] = _aggregate(samples)

    # 按作曲家（只保留知名作曲家）
    for composer, samples in sorted(by_composer.items(), key=lambda x: -len(x[1])):
        if len(samples) >= min_samples:
            safe_name = composer.replace(' ', '_').replace(',', '')[:30]
            benchmarks[f'composer_{safe_name}'] = _aggregate(samples)

    # 按来源
    for src, samples in by_source.items():
        if len(samples) >= min_samples:
            benchmarks[f'source_{src}'] = _aggregate(samples)

    # 写入文件
    if output_dir:
        output_path = os.path.join(output_dir, 'all.json')
        with open(output_path, 'w') as f:
            json.dump(benchmarks, f, indent=2, ensure_ascii=False)
        print(f'基准已写入: {output_path}')
        print(f'  分组数: {len(benchmarks)}')
        for name in sorted(benchmarks.keys()):
            n = benchmarks[name].get('_n_samples', 0)
            print(f'    - {name}: {n} 个样本')

    return benchmarks


def _extract_stats(md: dict) -> dict | None:
    """从单个 metadata 提取统计量。"""
    num_notes = md.get('num_notes', 0)
    num_measures = md.get('num_measures', 0)
    duration_sec = md.get('duration_seconds', 0)
    num_tokens = md.get('num_tokens', 0)

    if num_notes <= 0 or duration_sec <= 0:
        return None

    stats = {
        'num_notes': num_notes,
        'num_measures': num_measures,
        'num_tokens': num_tokens,
        'duration_seconds': duration_sec,
        'note_density': num_notes / duration_sec,
        'notes_per_measure': num_notes / max(num_measures, 1),
        'has_tempo': md.get('has_tempo', False),
        'has_drum': md.get('has_drum', False),
        'programs': md.get('programs', []),
        # 以下字段可能不存在
        'time_signature': md.get('time_signature', ''),
        'key_signature': md.get('key_signature', ''),
        'composer': md.get('composer', ''),
        'genre': md.get('genre', ''),
        'source_format': md.get('source_format', 'midi'),
        'tempo': md.get('tempo', 0),
    }
    return stats


def _aggregate(samples: list[dict]) -> dict:
    """对一组样本做聚合统计。"""
    n = len(samples)

    # 基本统计量
    densities = [s['note_density'] for s in samples if s['note_density'] > 0]
    notes_per_measure = [s['notes_per_measure'] for s in samples if s['notes_per_measure'] > 0]
    durations = [s['duration_seconds'] for s in samples if s['duration_seconds'] > 0]
    measures = [s['num_measures'] for s in samples if s['num_measures'] > 0]
    tokens = [s['num_tokens'] for s in samples if s['num_tokens'] > 0]

    # 速度标记比例
    has_tempo_count = sum(1 for s in samples if s['has_tempo'])
    has_drum_count = sum(1 for s in samples if s['has_drum'])

    # 乐器分布
    program_counts: dict[str, int] = defaultdict(int)
    for s in samples:
        for p in s.get('programs', []):
            program_counts[str(p)] += 1
    # 归一化为占比
    total_programs = sum(program_counts.values())
    program_dist = {}
    if total_programs > 0:
        top_programs = sorted(program_counts.items(), key=lambda x: -x[1])[:20]
        for k, v in top_programs:
            program_dist[k] = v / total_programs

    # 拍号分布
    ts_counts = defaultdict(int)
    for s in samples:
        ts = s.get('time_signature', '')
        if ts:
            ts_counts[ts] += 1
    ts_dist = {}
    for k, v in ts_counts.items():
        ts_dist[k] = v / n

    result = {
        '_n_samples': n,
        'note_density': _robust_stats(densities),
        'notes_per_measure': _robust_stats(notes_per_measure),
        'duration_seconds': _robust_stats(durations),
        'num_measures': _robust_stats(measures),
        'num_tokens': _robust_stats(tokens),
        'has_tempo_ratio': has_tempo_count / max(n, 1),
        'has_drum_ratio': has_drum_count / max(n, 1),
        'program_dist': program_dist,
        'time_signature_dist': ts_dist,
    }

    return result


def _robust_stats(values: list[float]) -> dict:
    """计算稳健统计量。"""
    if not values:
        return {}
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(variance)
    sorted_vals = sorted(values)

    def percentile(p):
        idx = int(n * p)
        return sorted_vals[min(idx, n - 1)]

    return {
        'mean': round(mean, 4),
        'std': round(std, 4),
        'p5': round(percentile(0.05), 4),
        'p25': round(percentile(0.25), 4),
        'p50': round(percentile(0.50), 4),
        'p75': round(percentile(0.75), 4),
        'p95': round(percentile(0.95), 4),
    }


def load_benchmarks(path: str | None = None) -> dict:
    """加载基准 JSON 文件。

    参数:
        path: 基准 JSON 路径，None 则使用默认路径

    返回:
        {group_name: benchmarks_dict}
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), 'data', 'all.json')

    if not os.path.exists(path):
        print(f'基准文件不存在: {path}')
        print('请先运行: python -m chopinote_evaluator.benchmarks.build_benchmarks')
        return {}

    with open(path, 'r') as f:
        return json.load(f)


def kl_divergence(p: list[float], q: list[float], epsilon: float = 1e-10) -> float:
    """KL(P||Q) 散度，零值做 smoothing。"""
    result = 0.0
    for pi, qi in zip(p, q):
        pi = max(pi, epsilon)
        qi = max(qi, epsilon)
        result += pi * math.log(pi / qi)
    return result


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """余弦相似度。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


if __name__ == '__main__':
    build_benchmarks()
