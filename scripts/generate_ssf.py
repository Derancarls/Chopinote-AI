#!/usr/bin/env python3
"""
SSF (Sliding Scale Field) 标注脚本 — v0.3.3 LMDB 版。
从 LMDB 读 token 数据，计算 SSF 三粒度 chroma 场，写回 LMDB。

SSF 编码规则 (主音锚定):
  - 12 维向量, 位置 0 = 主音, 位置 i = 距主音 i 个半音
  - TonicField:  段落级 PC histogram, 归一化
  - LocalField:  小节级 delta (稀疏, 仅存 |delta| > threshold)
  - BeatField:   节拍级, 每 bar 内相邻 Position token 间的 SSF 向量 (v0.3.3 新增)

用法:
  python scripts/generate_ssf.py annotate \
      --lmdb-path /root/autodl-tmp/data/processed/lmdb/chopinote_v4.lmdb \
      --num-workers 25
"""
import gc, os, sys, json, argparse, time, multiprocessing as mp
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


def compute_ssf_from_tokens(ids: list[int], sec_data: dict | None,
                             TONIC_IDS: dict, POSITION_IDS: dict) -> dict:
    """从 token ID 列表计算 SSF 三粒度场。纯计算，不涉及 I/O。"""
    # ── 段落边界 ──
    section_boundaries = [0]
    if sec_data:
        section_boundaries = sec_data.get('section_token_positions', [0])
    if not section_boundaries:
        section_boundaries = [0]

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
    # 最后一段
    last_start = section_boundaries[-1]
    sec_intervals = []
    for b in range(last_start, bar_idx + 1):
        sec_intervals.extend(bar_notes.get(b, []))
    tonic_fields.append(compute_tonic_field(sec_intervals, current_tonic))

    # ── 小节级 LocalField ──
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
        'tonic_fields': tonic_fields,
        'section_boundaries': section_boundaries,
        'local_fields': local_fields,
        'beat_fields': beat_fields,
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


# ── Worker ───────────────────────────────────────────────

_worker_store = None  # 全局: 每个 worker 进程的只读 LMDB 句柄


def _worker_init(lmdb_path: str):
    """Pool initializer: 每个 worker 进程打开一次只读 LMDB。"""
    global _worker_store
    from chopinote_dataset.lmdb_store import LMDBStore
    _worker_store = LMDBStore.open(lmdb_path, readonly=True, map_size=250 * 1024**3)


def _process_one(file_id: str) -> dict:
    """Worker: 从 LMDB 读 token → 计算 SSF。"""
    global _worker_store
    store = _worker_store
    if store is None:
        return {'status': 'error', 'file_id': file_id, 'reason': 'store not initialized'}

    try:
        tokens = store.get_tokens(file_id)
        sec_data = store.get_sec(file_id)
    except Exception as e:
        return {'status': 'error', 'file_id': file_id, 'reason': str(e)}

    if not isinstance(tokens, list) or len(tokens) < 10:
        return {'status': 'skip', 'file_id': file_id, 'reason': 'too_short'}

    tonic_ids = _get_tonic_ids()
    pos_ids = _get_position_ids()

    try:
        data = compute_ssf_from_tokens(tokens, sec_data, tonic_ids, pos_ids)
    except Exception as e:
        return {'status': 'error', 'file_id': file_id, 'reason': str(e)}

    return {'status': 'ok', 'file_id': file_id, 'data': data}


# ── CLI ────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='SSF annotation for v0.3.3 (LMDB)')
    ap.add_argument('command', choices=['annotate'])
    ap.add_argument('--lmdb-path', required=True, help='LMDB 路径 (读写)')
    ap.add_argument('--num-workers', type=int, default=16)
    args = ap.parse_args()

    _init_note_on_range()
    # 预暖缓存: fork 前加载 tokenizer
    _get_tonic_ids()
    _get_position_ids()

    from chopinote_dataset.lmdb_store import LMDBStore

    PENDING_FILE = '/root/autodl-tmp/ssf_pending.txt'
    DONE_FILE = '/root/autodl-tmp/ssf_done.txt'

    # ── Phase 1: 获取待处理列表 (进度文件优先, 避免重复扫描 LMDB) ──
    print("[Phase 1] 获取待处理列表...", flush=True)
    t0 = time.time()

    # 读取断点续跑记录
    done_ids = set()
    if os.path.exists(DONE_FILE):
        with open(DONE_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    done_ids.add(line)
        print(f"  已完成 (from file): {len(done_ids)}", flush=True)

    if os.path.exists(PENDING_FILE):
        # ── 快速路径: 从进度文件恢复 ──
        print("  从进度文件恢复...", flush=True)
        pending_ids = []
        with open(PENDING_FILE) as f:
            for line in f:
                line = line.strip()
                if line and line not in done_ids:
                    pending_ids.append(line)
        print(f"  待标注: {len(pending_ids)} ({time.time() - t0:.1f}s)", flush=True)
    else:
        # ── 首次: 全量 LMDB 扫描 ──
        print("  首次运行, 扫描 LMDB (仅此一次)...", flush=True)
        ro_store = LMDBStore.open(args.lmdb_path, readonly=True, map_size=250 * 1024**3)
        try:
            # 合并两次扫描为一次: 收集 :ssf 和 :tokens 的 file_id
            has_ssf = set()
            tokens_fids = []
            seen = set()
            with ro_store.env.begin(db=ro_store.main_db) as txn:
                cursor = txn.cursor()
                if cursor.set_range(b'v4:'):
                    for key, _ in cursor:
                        if not key.startswith(b'v4:'):
                            break
                        parts = key.decode('utf-8').split(':')
                        if len(parts) >= 3:
                            fid = parts[1]
                            dtype = parts[2]
                            if dtype == 'ssf':
                                has_ssf.add(fid)
                            elif dtype == 'tokens' and fid not in seen:
                                seen.add(fid)
                                if fid not in has_ssf:
                                    tokens_fids.append(fid)
                    cursor.close()
        finally:
            ro_store.close()

        # 过滤出待处理 (tokens 有但 ssf 无)
        pending_ids = [f for f in tokens_fids if f not in has_ssf]
        # 排除已完成的
        if done_ids:
            pending_ids = [f for f in pending_ids if f not in done_ids]

        print(f"  已有 SSF: {len(has_ssf)}")
        print(f"  总 tokens: {len(seen)}")
        print(f"  待标注: {len(pending_ids)}")
        print(f"  扫描耗时: {time.time() - t0:.1f}s", flush=True)

        # 保存进度文件供后续恢复
        with open(PENDING_FILE, 'w') as f:
            for fid in pending_ids:
                f.write(fid + '\n')
        print(f"  进度文件已保存: {PENDING_FILE}", flush=True)

    if not pending_ids:
        print("无需标注，全部完成。")
        return

    # ── Phase 2: 并行标注 ──
    # 关键 1: 先 fork pool, 再开 write_store (避免 LMDB fork 冲突)
    # 关键 2: 分块提交 imap, 每块 100K, 防止内部队列无限膨胀导致 CPU 暴涨
    print(f"\n[Phase 2] 启动 {args.num_workers} workers...", flush=True)
    stats = Counter()
    start = time.time()
    completed = 0
    error_samples = []
    total_pending = len(pending_ids)
    BATCH_SIZE = 2000
    WRITE_CHUNK = 100000  # 每次最多提交 10 万条到 imap 队列

    done_fh = open(DONE_FILE, 'a')
    pool = mp.Pool(args.num_workers, initializer=_worker_init,
                   initargs=(args.lmdb_path,))
    write_store = LMDBStore.open(args.lmdb_path, readonly=False, map_size=250 * 1024**3)

    try:
        # 分块处理: 避免 pool.imap_unordered 内部队列爆炸
        for chunk_start in range(0, len(pending_ids), WRITE_CHUNK):
            chunk = pending_ids[chunk_start:chunk_start + WRITE_CHUNK]
            batch = []

            for result in pool.imap_unordered(_process_one, chunk, chunksize=20):
                completed += 1
                status = result['status']
                stats[status] += 1

                if status == 'ok':
                    fid = result['file_id']
                    batch.append((fid, 'ssf', result['data']))
                    del result['data']
                elif len(error_samples) < 5:
                    error_samples.append(f"{result['file_id']}: {result.get('reason', '?')}")

                if len(batch) >= BATCH_SIZE:
                    _flush_batch(write_store, batch)
                    for fid, _, _ in batch:
                        done_fh.write(fid + '\n')
                    batch.clear()

                del result

                if completed % 50000 == 0:
                    elapsed = time.time() - start
                    rate = completed / max(1, elapsed)
                    eta = (total_pending - completed) / max(1, rate)
                    print(f"  进度: {completed}/{total_pending} "
                          f"({100*completed/total_pending:.0f}%), "
                          f"{rate:.0f} f/s, ETA {eta:.0f}s, "
                          f"ok={stats['ok']}, err={stats['error']}, skip={stats['skip']}",
                          flush=True)
                    if error_samples:
                        print(f"  错误样本: {error_samples[:3]}", flush=True)
                    gc.collect()

            # 每块结束刷盘 + 释放 imap 内部缓冲
            if batch:
                _flush_batch(write_store, batch)
                for fid, _, _ in batch:
                    done_fh.write(fid + '\n')
                batch.clear()
            done_fh.flush()
            gc.collect()

    finally:
        pool.terminate()
        pool.join()
        done_fh.close()
        write_store.close()

    elapsed = time.time() - start
    print(f"\n完成! {elapsed:.0f}s")
    print(f"  成功: {stats['ok']}")
    print(f"  错误: {stats['error']}")
    print(f"  跳过: {stats['skip']}")


def _flush_batch(store, batch: list):
    """批量写入 LMDB。"""
    if not batch:
        return
    with store.env.begin(db=store.main_db, write=True) as txn:
        for fid, dtype, data in batch:
            store._txn_put(txn, fid, dtype, data)


if __name__ == '__main__':
    main()
