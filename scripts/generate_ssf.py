#!/usr/bin/env python3
"""
SSF (Sliding Scale Field) 标注脚本 — v0.3.0。
从 .tokens 文件生成 .ssf.json 侧边文件。

SSF 编码规则 (主音锚定):
  - 12 维向量, 位置 0 = 主音, 位置 i = 距主音 i 个半音
  - TonicField: 段落级 PC histogram, 归一化
  - LocalField: 小节级 delta (稀疏, 仅存 |delta| > threshold)

用法:
  python scripts/generate_ssf.py annotate \
      --input-dir /root/autodl-tmp/data/processed/tokens_v4 \
      --num-workers 16
"""
import os, sys, json, argparse, time, multiprocessing as mp
from collections import Counter


# ── 主音名 → pitch class ──────────────────────────────────
_TONIC_PC = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
    'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11, 'Cb': 11,
}

LOCAL_DELTA_THRESHOLD = 0.15


def tonic_name_to_pc(name: str) -> int:
    return _TONIC_PC.get(name, 0)


def compute_tonic_field(note_intervals: list[int], tonic_name: str) -> list[float]:
    """统计 Note_ON 区间 → 12 维 TonicField (主音在 pos 0)。

    note_intervals: Note_ON 的 interval 值 (pitch - tonic_midi)
    tonic_name: 当前主音名
    """
    counts = [0] * 12
    tonic_pc = tonic_name_to_pc(tonic_name)

    for interval in note_intervals:
        abs_pc = (tonic_pc + interval) % 12
        rotated_pos = (abs_pc - tonic_pc) % 12  # tonic 落到 pos 0
        counts[rotated_pos] += 1

    total = sum(counts)
    if total == 0:
        return [0.5] * 12

    max_count = max(counts)
    return [c / max_count for c in counts]  # 归一化到 [0, 1]


def compute_local_field(
    note_intervals: list[int], tonic_name: str, tonic_field: list[float]
) -> list[float] | None:
    """计算小节级 LocalField delta。如果 delta 太小返回 None (稀疏存储)。"""
    if len(note_intervals) < 3:
        return None

    local = compute_tonic_field(note_intervals, tonic_name)
    delta = [local[i] - tonic_field[i] for i in range(12)]

    if max(abs(d) for d in delta) < LOCAL_DELTA_THRESHOLD:
        return None  # 稀疏: 变化太小不存

    return delta


def process_file(args: tuple) -> dict:
    """处理单个 .tokens 文件。"""
    fpath, secio_dir = args

    # 加载 tokens
    try:
        with open(fpath) as f:
            ids = json.load(f)
    except Exception as e:
        return {'status': 'error', 'file': fpath, 'reason': str(e)}

    if not isinstance(ids, list) or len(ids) < 10:
        return {'status': 'skip', 'file': fpath, 'reason': 'too_short'}

    # 加载 .sec.json (段落边界)
    sec_path = fpath.replace('.tokens', '.sec.json')
    section_boundaries = []
    try:
        with open(sec_path) as f:
            sec_data = json.load(f)
        section_boundaries = sec_data.get('section_token_positions', [])
    except Exception:
        pass

    if not section_boundaries:
        section_boundaries = [0]

    # 解码 token → 提取 Note_ON 区间 和 Tonic
    # 简单扫描: 用 token ID 但不依赖完整的 tokenizer (更快)
    # Tonic token: 遍历词表前缀匹配太慢，改为在 script 启动时预计算
    note_intervals = []  # 每个小节: [(bar_idx, interval)]
    bar_idx = -1
    current_tonic = 'C'
    bar_positions = []  # token 位置列表

    # 这里用到一个简化的扫描器，不依赖完整 tokenizer
    # Tonic tokens: 需要外部传入 ID 范围
    TONIC_IDS = _get_tonic_ids()

    for pos, tid in enumerate(ids):
        if tid in TONIC_IDS:
            current_tonic = TONIC_IDS[tid]
            continue
        if tid == 4:  # <Bar>
            bar_idx += 1
            bar_positions.append(pos)
            continue
        # Note_ON IDs: 需要外部传入范围
        if _NOTE_ON_MIN <= tid <= _NOTE_ON_MAX:
            interval = tid - _NOTE_ON_ZERO
            note_intervals.append((bar_idx, pos, interval))

    if bar_idx < 0:
        bar_positions = [0]
        bar_idx = 0

    # 构建 per-bar 的 Note_ON 区间列表
    bar_notes: dict[int, list[int]] = {b: [] for b in range(bar_idx + 1)}
    for bar, pos, interval in note_intervals:
        if bar >= 0:
            bar_notes[bar].append(interval)

    # 如果 section_boundaries 只有起始值, 补上结束值
    if len(section_boundaries) == 1:
        section_boundaries.append(bar_idx + 1)

    # ── 计算 TonicField (段落级) ──
    tonic_fields = []
    for i in range(len(section_boundaries)):
        sec_start = section_boundaries[i]
        sec_end = (section_boundaries[i + 1]
                   if i + 1 < len(section_boundaries)
                   else bar_idx + 1)

        # 收集该段落内所有 Note_ON 区间
        sec_intervals = []
        for b in range(sec_start, min(sec_end, bar_idx + 1)):
            sec_intervals.extend(bar_notes.get(b, []))

        tf = compute_tonic_field(sec_intervals, current_tonic)
        tonic_fields.append(tf)

    # ── 计算 LocalField (小节级 delta) ──
    local_fields: dict[str, list[float]] = {}
    for b in range(bar_idx + 1):
        if b < len(bar_notes) and bar_notes[b]:
            # 找到该 bar 所属段落
            sec_idx = 0
            for i in range(len(section_boundaries)):
                if b >= section_boundaries[i]:
                    sec_idx = i
            tf = tonic_fields[sec_idx] if sec_idx < len(tonic_fields) else tonic_fields[-1]
            delta = compute_local_field(bar_notes[b], current_tonic, tf)
            if delta is not None:
                local_fields[str(b)] = delta

    return {
        'status': 'ok',
        'file': fpath,
        'data': {
            'tonic_fields': tonic_fields,
            'section_boundaries': section_boundaries,
            'local_fields': local_fields,
        },
    }


# ── Token ID 常量 (在 main 中初始化) ──
_NOTE_ON_MIN = 0
_NOTE_ON_MAX = 0
_NOTE_ON_ZERO = 0


def _get_tonic_ids() -> dict[int, str]:
    """获取 Tonic token ID → tonic name 映射。"""
    from chopinote_dataset.tokenizer import REMITokenizer
    t = REMITokenizer(16, 8)
    mapping = {}
    for name in t.TONIC_NAMES:
        tid = t.encode_token(f'<Tonic {name}>')
        mapping[tid] = name
    return mapping


def _init_note_on_range():
    """初始化 Note_ON token 的 ID 范围。"""
    global _NOTE_ON_MIN, _NOTE_ON_MAX, _NOTE_ON_ZERO
    from chopinote_dataset.tokenizer import REMITokenizer
    t = REMITokenizer(16, 8)
    ids = [t.encode_token(f'<Note_ON {i}>') for i in range(-60, 61)]
    _NOTE_ON_MIN = min(ids)
    _NOTE_ON_MAX = max(ids)
    _NOTE_ON_ZERO = t.encode_token('<Note_ON 0>')


# ── CLI ────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='SSF annotation for v0.3.0')
    ap.add_argument('command', choices=['annotate'])
    ap.add_argument('--input-dir', default='/root/autodl-tmp/data/processed/tokens_v4')
    ap.add_argument('--num-workers', type=int, default=16)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    _init_note_on_range()

    all_files = []
    for root, dirs, files in os.walk(args.input_dir):
        for f in files:
            if f.endswith('.tokens'):
                all_files.append(os.path.join(root, f))

    print(f"文件总数: {len(all_files)}")
    print(f"工作进程: {args.num_workers}")

    tasks = [(f, args.input_dir) for f in all_files]
    stats = Counter()
    start = time.time()

    with mp.Pool(args.num_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(process_file, tasks, chunksize=100)):
            if result['status'] == 'ok':
                stats['ok'] += 1
                if not args.dry_run:
                    data = result['data']
                    out_path = result['file'].replace('.tokens', '.ssf.json')
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    with open(out_path, 'w') as f:
                        json.dump(data, f)
            else:
                stats[result['status']] += 1

            if (i + 1) % 100000 == 0:
                elapsed = time.time() - start
                print(f"  进度: {i+1}/{len(all_files)} "
                      f"({100*(i+1)/len(all_files):.0f}%), {elapsed:.0f}s, ok={stats['ok']}")

    elapsed = time.time() - start
    print(f"完成! {elapsed:.0f}s")
    print(f"  成功: {stats['ok']}")
    print(f"  错误: {stats['error']}")
    print(f"  跳过: {stats['skip']}")


if __name__ == '__main__':
    main()
