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
import threading, gc
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

def _scandir_iter(input_dir: str):
    """用 os.scandir 逐项迭代 — 不构建任何列表, 不触 os.walk 的 nondirs 列表。

    os.walk 对平坦目录会先把 326 万文件名全装进 nondirs 列表再 yield,
    这 ~300MB 被 fork×25 → 7.5GB 物理内存。scandir 直接迭代，零缓存。
    """
    dirs_to_walk = [input_dir]
    while dirs_to_walk:
        d = dirs_to_walk.pop()
        try:
            with os.scandir(d) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        dirs_to_walk.append(os.path.join(d, entry.name))
                    elif entry.is_file() and entry.name.endswith('.tokens'):
                        yield (os.path.join(d, entry.name), input_dir)
        except OSError:
            continue


def _count_tokens_files(input_dir: str) -> int:
    """用 find 快速计数 (不占 Python 内存)。"""
    import subprocess
    result = subprocess.run(
        ['find', input_dir, '-name', '*.tokens', '-printf', '.'],
        capture_output=True, text=True, timeout=300
    )
    return len(result.stdout)


def main():
    ap = argparse.ArgumentParser(description='SSF annotation for v0.3.3')
    ap.add_argument('command', choices=['annotate'])
    ap.add_argument('--input-dir', default='/root/autodl-tmp/data/processed/tokens_v4')
    ap.add_argument('--num-workers', type=int, default=16)
    ap.add_argument('--output-dir', default=None, help='SSF 输出目录 (默认与 .tokens 同目录)')
    ap.add_argument('--lmdb-path', default=None, help='LMDB 路径 (优先于 --output-dir)')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    ssf_output_dir = args.output_dir or args.input_dir

    _init_note_on_range()
    # 预暖缓存: 在 fork 前加载 tokenizer, 子进程直接继承 (省 25×50MB)
    _get_tonic_ids()
    _get_position_ids()

    # LMDB 模式: 提前打开
    lmdb_store = None
    if args.lmdb_path:
        from chopinote_dataset.lmdb_store import LMDBStore
        lmdb_store = LMDBStore.open(args.lmdb_path, readonly=False)
        print(f"LMDB 输出: {args.lmdb_path}")
    _get_position_ids()

    total = _count_tokens_files(args.input_dir)
    print(f"文件总数: {total}")
    print(f"工作进程: {args.num_workers}")

    stats = Counter()
    start = time.time()

    # 阶段 1: 用 find 预生成路径文件 (避免 scandir 与 25 worker I/O 争抢)
    print("阶段 1: 扫描文件路径...")
    import subprocess
    path_file = '/tmp/ssf_paths.txt'
    with open(path_file, 'wb') as f:
        subprocess.run(
            ['find', args.input_dir, '-name', '*.tokens', '-print0'],
            stdout=f, check=True
        )
    print(f"  路径文件就绪 ({os.path.getsize(path_file)//1024//1024} MB)")

    # 阶段 2: 读路径 → 提交 workers
    print("阶段 2: 启动 workers...")
    lock = threading.Lock()
    pending = 0
    completed = 0
    submitted = 0
    MAX_PENDING = args.num_workers * 4

    def _on_done(result):
        nonlocal pending, completed
        with lock:
            pending -= 1
            completed += 1
            cur = completed

        if result['status'] == 'ok':
            with lock:
                stats['ok'] += 1
            if not args.dry_run:
                data = result['data']
                if lmdb_store is not None:
                    fid = os.path.basename(result['file']).replace('.tokens', '')
                    with lmdb_store.env.begin(db=lmdb_store.main_db, write=True) as txn:
                        lmdb_store._txn_put(txn, fid, 'ssf', data)
                else:
                    fname = os.path.basename(result['file']).replace('.tokens', '.ssf.json')
                    out_path = os.path.join(ssf_output_dir, fname)
                    with open(out_path, 'w') as f:
                        json.dump(data, f)
                del data
        else:
            with lock:
                stats[result['status']] += 1

        del result
        if cur % 100000 == 0:
            elapsed = time.time() - start
            print(f"  进度: {cur}/{total} "
                  f"({100*cur/total:.0f}%), {elapsed:.0f}s, ok={stats['ok']}")

    pool = mp.Pool(args.num_workers)
    try:
        # 从文件批量读路径 (64KB buffer), 避免逐文件争抢目录 dentry
        buf = b''
        with open(path_file, 'rb', buffering=65536) as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    if buf:
                        task = (buf.decode('utf-8'), args.input_dir)
                        while True:
                            with lock:
                                if pending < MAX_PENDING:
                                    pending += 1
                                    submitted += 1
                                    break
                            time.sleep(0.001)
                        pool.apply_async(process_file, (task,), callback=_on_done)
                    break
                buf += chunk
                while b'\0' in buf:
                    path_bytes, buf = buf.split(b'\0', 1)
                    if not path_bytes:
                        continue
                    task = (path_bytes.decode('utf-8'), args.input_dir)
                    while True:
                        with lock:
                            if pending < MAX_PENDING:
                                pending += 1
                                submitted += 1
                                break
                        time.sleep(0.001)
                    pool.apply_async(process_file, (task,), callback=_on_done)

        pool.close()
        # 等待全部回调完成
        while True:
            with lock:
                if completed >= submitted:
                    break
            time.sleep(0.5)
            gc.collect()
        pool.join()
    finally:
        pool.terminate()
        pool.join()

    gc.collect()
    os.unlink(path_file)
    if lmdb_store is not None:
        lmdb_store.close()
    elapsed = time.time() - start
    print(f"完成! {elapsed:.0f}s, total={submitted}")
    print(f"  成功: {stats['ok']}")
    print(f"  错误: {stats['error']}")
    print(f"  跳过: {stats['skip']}")


if __name__ == '__main__':
    main()
