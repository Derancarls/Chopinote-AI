#!/usr/bin/env python3
"""
SSF (Sliding Scale Field) 标注脚本 — v0.3.0 → v0.3.3 升级。
从 .tokens 文件生成 .ssf.json 侧边文件。

SSF 编码规则 (主音锚定):
  - 12 维向量, 位置 0 = 主音, 位置 i = 距主音 i 个半音
  - TonicField:  段落级 PC histogram, 归一化
  - LocalField:  小节级 delta (稀疏, 仅存 |delta| > threshold)
  - BeatField:   节拍级, 每 bar 内相邻 Position token 间的 SSF 向量 (v0.3.3 新增)

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
    """统计 Note_ON 区间 → 12 维 TonicField (主音在 pos 0)。"""
    counts = [0] * 12
    tonic_pc = tonic_name_to_pc(tonic_name)

    for interval in note_intervals:
        abs_pc = (tonic_pc + interval) % 12
        rotated_pos = (abs_pc - tonic_pc) % 12
        counts[rotated_pos] += 1

    total = sum(counts)
    if total == 0:
        return [0.5] * 12

    max_count = max(counts)
    return [c / max_count for c in counts]


def compute_local_field(
    note_intervals: list[int], tonic_name: str, tonic_field: list[float]
) -> list[float] | None:
    """计算小节级 LocalField delta。delta 太小返回 None (稀疏存储)。"""
    if len(note_intervals) < 3:
        return None

    local = compute_tonic_field(note_intervals, tonic_name)
    delta = [local[i] - tonic_field[i] for i in range(12)]

    if max(abs(d) for d in delta) < LOCAL_DELTA_THRESHOLD:
        return None

    return delta


def compute_beat_field(note_intervals: list[int], tonic_name: str) -> list[float] | None:
    """计算节拍级 SSF 向量。无音符返回 None。"""
    if not note_intervals:
        return None
    return compute_tonic_field(note_intervals, tonic_name)


def process_file(args: tuple) -> dict:
    """处理单个 .tokens 文件。"""
    fpath, secio_dir = args

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

    TONIC_IDS = _get_tonic_ids()
    POSITION_IDS = _get_position_ids()

    # ── 扫描 token 序列 ──
    bar_idx = -1
    current_tonic = 'C'
    current_position = 0
    note_intervals = []  # [(bar_idx, position_in_bar, interval)]

    for pos, tid in enumerate(ids):
        if tid in TONIC_IDS:
            current_tonic = TONIC_IDS[tid]
            continue
        if tid == 4:  # <Bar>
            bar_idx += 1
            current_position = 0
            continue
        if tid in POSITION_IDS:
            current_position = POSITION_IDS[tid]
            continue
        if _NOTE_ON_MIN <= tid <= _NOTE_ON_MAX:
            interval = tid - _NOTE_ON_ZERO
            note_intervals.append((bar_idx, current_position, interval))

    if bar_idx < 0:
        bar_idx = 0

    # ── 组织数据: bar_notes[bar] = list of intervals ──
    bar_notes: dict[int, list[int]] = {b: [] for b in range(bar_idx + 1)}
    # beat_notes[bar][position] = list of intervals
    beat_notes: dict[int, dict[int, list[int]]] = {b: {} for b in range(bar_idx + 1)}

    for bar, pos_in_bar, interval in note_intervals:
        if bar >= 0:
            bar_notes.setdefault(bar, []).append(interval)
            beat_notes.setdefault(bar, {}).setdefault(pos_in_bar, []).append(interval)

    # ── 补全 section_boundaries ──
    if len(section_boundaries) == 1:
        section_boundaries.append(bar_idx + 1)

    # ── 段落级 TonicField ──
    tonic_fields = []
    for i in range(len(section_boundaries) - 1):
        sec_start = section_boundaries[i]
        sec_end = section_boundaries[i + 1]
        sec_intervals = []
        for b in range(sec_start, min(sec_end, bar_idx + 1)):
            sec_intervals.extend(bar_notes.get(b, []))
        tonic_fields.append(compute_tonic_field(sec_intervals, current_tonic))
    # 最后一段 (sec_boundaries[-1] → end)
    last_start = section_boundaries[-1]
    sec_intervals = []
    for b in range(last_start, bar_idx + 1):
        sec_intervals.extend(bar_notes.get(b, []))
    tonic_fields.append(compute_tonic_field(sec_intervals, current_tonic))

    # ── 小节级 LocalField (delta from section TonicField) ──
    local_fields: dict[str, list[float]] = {}
    for b in range(bar_idx + 1):
        if b in bar_notes and bar_notes[b]:
            sec_idx = 0
            for i in range(len(section_boundaries)):
                if b >= section_boundaries[i]:
                    sec_idx = i
            tf = tonic_fields[sec_idx] if sec_idx < len(tonic_fields) else tonic_fields[-1]
            delta = compute_local_field(bar_notes[b], current_tonic, tf)
            if delta is not None:
                local_fields[str(b)] = delta

    # ── 节拍级 BeatField (v0.3.3 新增) ──
    beat_fields: dict[str, dict[str, list[float]]] = {}
    for b in range(bar_idx + 1):
        if b in beat_notes:
            bar_beats = {}
            for pos_in_bar, intervals in beat_notes[b].items():
                bf = compute_beat_field(intervals, current_tonic)
                if bf is not None:
                    bar_beats[str(pos_in_bar)] = bf
            if bar_beats:
                beat_fields[str(b)] = bar_beats

    return {
        'status': 'ok',
        'file': fpath,
        'data': {
            'tonic_fields': tonic_fields,
            'section_boundaries': section_boundaries,
            'local_fields': local_fields,
            'beat_fields': beat_fields,
        },
    }


# ── Token ID 常量 ────────────────────────────────────────
_NOTE_ON_MIN = 0
_NOTE_ON_MAX = 0
_NOTE_ON_ZERO = 0

_tonic_ids_cache: dict[int, str] | None = None
_position_ids_cache: dict[int, int] | None = None


def _get_tonic_ids() -> dict[int, str]:
    global _tonic_ids_cache
    if _tonic_ids_cache is not None:
        return _tonic_ids_cache
    from chopinote_dataset.tokenizer import REMITokenizer
    t = REMITokenizer(16, 8)
    mapping = {}
    for name in t.TONIC_NAMES:
        tid = t.encode_token(f'<Tonic {name}>')
        mapping[tid] = name
    _tonic_ids_cache = mapping
    return mapping


def _get_position_ids() -> dict[int, int]:
    """Position token ID → position value (0-15) 映射。"""
    global _position_ids_cache
    if _position_ids_cache is not None:
        return _position_ids_cache
    from chopinote_dataset.tokenizer import REMITokenizer
    t = REMITokenizer(16, 8)
    mapping = {}
    for i in range(t.grid_size):
        tid = t.encode_token(f'<Position {i}>')
        mapping[tid] = i
    _position_ids_cache = mapping
    return mapping


def _init_note_on_range():
    global _NOTE_ON_MIN, _NOTE_ON_MAX, _NOTE_ON_ZERO
    from chopinote_dataset.tokenizer import REMITokenizer
    t = REMITokenizer(16, 8)
    ids = [t.encode_token(f'<Note_ON {i}>') for i in range(-60, 61)]
    _NOTE_ON_MIN = min(ids)
    _NOTE_ON_MAX = max(ids)
    _NOTE_ON_ZERO = t.encode_token('<Note_ON 0>')


# ── CLI ────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='SSF annotation for v0.3.3')
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
