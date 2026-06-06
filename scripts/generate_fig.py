#!/usr/bin/env python3
"""
Figuration 标注脚本 — v0.3.3 LMDB 版。
从 LMDB 读 token 数据，按 Voice 拆分 4-bar 窗口分类织体模式，
写回 LMDB (fig 数据类型)。

织体类型 (11 种):
  0=none, 1=block, 2=alberti, 3=arpeggio, 4=stride,
  5=octave_tremolo, 6=walking_bass, 7=countermelody, 8=pedal,
  9=waltz, 10=broken_octave, 11=tremolo

输出格式 (LMDB key: v4:<file_id>:fig):
  {
    "version": 1,
    "bar_figs": {"<bar>": {"<voice>": <fig_type>, ...}, ...}
  }

用法:
  python scripts/generate_fig.py annotate \
      --lmdb-path /root/autodl-tmp/data/processed/lmdb/chopinote_v4.lmdb \
      --num-workers 16
"""
import gc, os, sys, json, argparse, time, multiprocessing as mp
from collections import Counter

FIG_NONE = 0
FIG_BLOCK = 1
FIG_ALBERTI = 2
FIG_ARPEGGIO = 3
FIG_STRIDE = 4
FIG_OCTAVE_TREMOLO = 5
FIG_WALKING_BASS = 6
FIG_COUNTERMELODY = 7
FIG_PEDAL = 8
FIG_WALTZ = 9
FIG_BROKEN_OCTAVE = 10
FIG_TREMOLO = 11

FIG_IDX_TO_NAME = {
    1: 'block', 2: 'alberti', 3: 'arpeggio', 4: 'stride',
    5: 'octave_tremolo', 6: 'walking_bass', 7: 'countermelody', 8: 'pedal',
    9: 'waltz', 10: 'broken_octave', 11: 'tremolo',
}

WINDOW_BARS = 4

# 全局常量，fork 前初始化
_VOICE_IDS = None
_BAR_ID = 4
_POS_MIN = _POS_MAX = -1
_NOTE_ON_MIN = _NOTE_ON_MAX = _NOTE_ON_ZERO = 0


def _init_constants():
    global _VOICE_IDS, _BAR_ID, _POS_MIN, _POS_MAX
    global _NOTE_ON_MIN, _NOTE_ON_MAX, _NOTE_ON_ZERO
    from chopinote_dataset.tokenizer import REMITokenizer
    t = REMITokenizer(16, 8)
    _VOICE_IDS = {t.encode_token(f'<Voice {v}>'): v for v in range(4)}
    _BAR_ID = t.bar_token_id
    note_ids = [t.encode_token(f'<Note_ON {i}>') for i in range(-60, 61)]
    _NOTE_ON_MIN = min(note_ids)
    _NOTE_ON_MAX = max(note_ids)
    _NOTE_ON_ZERO = t.encode_token('<Note_ON 0>')
    pos_ids = [t.encode_token(f'<Position {i}>') for i in range(16)]
    _POS_MIN = min(pos_ids)
    _POS_MAX = max(pos_ids)


def classify_window(notes: list[tuple[int, int]]) -> int:
    """分类一个窗口内的织体类型。"""
    if len(notes) < 4:
        return FIG_NONE

    pitches = [n[0] for n in notes]
    intervals = [pitches[i+1] - pitches[i] for i in range(len(pitches) - 1)]

    # 1. 同 position ≥3 音 → Block
    pos_counts = Counter(n[1] for n in notes)
    if max(pos_counts.values()) >= 3:
        return FIG_BLOCK

    # 2. 八度交替 → Octave Tremolo
    if len(intervals) >= 4 and all(abs(i) in (0, 12) for i in intervals):
        if any(abs(i) == 12 for i in intervals):
            return FIG_OCTAVE_TREMOLO

    # 3. 同音/近距离快速重复 → Tremolo
    if len(intervals) >= 4 and sum(1 for i in intervals if abs(i) <= 1) >= len(intervals) * 0.7:
        return FIG_TREMOLO

    # 4. Alberti: 3-4 音周期
    if _is_alberti(intervals):
        return FIG_ALBERTI

    # 5. Broken octave
    if all(abs(i) in (0, 12) for i in intervals):
        return FIG_BROKEN_OCTAVE

    # 6. Stride: 强拍单音 + 弱拍多音, 低音跳跃大
    if _is_stride(notes):
        return FIG_STRIDE

    # 7. Waltz: 低音-和弦-和弦
    if _is_waltz(notes):
        return FIG_WALTZ

    # 8. Walking bass: 级进
    if len(intervals) >= 4 and all(abs(i) <= 2 for i in intervals):
        return FIG_WALKING_BASS

    # 9. Pedal: 音高几乎不变
    if len(set(pitches)) <= 2:
        return FIG_PEDAL

    # 10. Countermelody: 音高方差大
    if len(pitches) >= 8 and _pitch_variance(pitches) > 30:
        return FIG_COUNTERMELODY

    # 11. 默认 → 琶音
    return FIG_ARPEGGIO


def _is_alberti(intervals: list[int]) -> bool:
    if len(intervals) < 6:
        return False
    for period in (3, 4):
        for offset in range(period):
            errors = 0
            for i in range(offset, len(intervals) - period, period):
                if abs(intervals[i] - intervals[i+period]) > 1:
                    errors += 1
            if errors <= len(intervals) // (period * 3):
                return True
    return False


def _is_stride(notes: list) -> bool:
    strong = [n for n in notes if n[1] % 4 == 0]
    if len(strong) < 2:
        return False
    strong_pitches = [s[0] for s in strong]
    jumps = [abs(strong_pitches[i+1] - strong_pitches[i]) for i in range(len(strong_pitches) - 1)]
    return any(j > 12 for j in jumps)


def _is_waltz(notes: list) -> bool:
    pos_groups = {}
    for n in notes:
        pos_groups.setdefault(n[1], []).append(n[0])
    if len(pos_groups) < 2:
        return False
    return max(len(v) for v in pos_groups.values()) >= 2


def _pitch_variance(pitches: list[int]) -> float:
    mean = sum(pitches) / len(pitches)
    return sum((p - mean) ** 2 for p in pitches) / len(pitches)


def classify_tokens(ids: list[int]) -> dict:
    """从 token ID 列表分类织体 (per-bar per-voice)。

    Returns:
        {"bar_figs": {"<bar>": {"<voice>": <fig_type>, ...}, ...}}
    """
    # ── 第一遍: 按 bar 和 voice 收集 Note_ON 事件 ──
    per_bar_voice: dict[int, dict[int, list]] = {}
    bar_idx = -1
    current_voice = 0
    current_pos = 0

    for tid in ids:
        if tid == _BAR_ID:
            bar_idx += 1
            per_bar_voice.setdefault(bar_idx, {})
            continue
        if tid in _VOICE_IDS:
            current_voice = _VOICE_IDS[tid]
            continue
        if _POS_MIN <= tid <= _POS_MAX:
            current_pos = tid - _POS_MIN
            continue
        if _NOTE_ON_MIN <= tid <= _NOTE_ON_MAX:
            interval = tid - _NOTE_ON_ZERO
            per_bar_voice.setdefault(bar_idx, {}).setdefault(current_voice, []).append(
                (interval, current_pos))

    if bar_idx < 0:
        return {'bar_figs': {}}

    # ── 第二遍: per-voice 分类 (滑动窗口) ──
    bar_figs: dict[int, dict[int, int]] = {}

    all_voices = set()
    for bar in per_bar_voice.values():
        all_voices.update(bar.keys())

    for voice in all_voices:
        voice_bars: dict[int, list] = {}
        for b in range(bar_idx + 1):
            notes = per_bar_voice.get(b, {}).get(voice, [])
            if notes:
                voice_bars[b] = notes

        if len(voice_bars) < WINDOW_BARS:
            continue

        sorted_bars = sorted(voice_bars.keys())

        for i in range(0, len(sorted_bars) - WINDOW_BARS + 1, max(1, WINDOW_BARS // 2)):
            window_bars = sorted_bars[i:i+WINDOW_BARS]
            window_notes = []
            for b in window_bars:
                window_notes.extend(voice_bars[b])

            if len(window_notes) < 4:
                continue

            fig = classify_window(window_notes)
            if fig == FIG_NONE:
                continue

            for b in window_bars:
                bar_figs.setdefault(b, {})[voice] = fig

    # 转换为字符串键 (JSON 兼容)
    str_bar_figs = {}
    for b, vf in bar_figs.items():
        str_bar_figs[str(b)] = {str(v): f for v, f in vf.items()}

    return {
        'version': 1,
        'bar_figs': str_bar_figs,
    }


# ── Worker ───────────────────────────────────────────────

_worker_store = None


def _worker_init(lmdb_path: str):
    """Pool initializer: 每个 worker 进程打开一次只读 LMDB。"""
    global _worker_store
    from chopinote_dataset.lmdb_store import LMDBStore
    _worker_store = LMDBStore.open(lmdb_path, readonly=True, map_size=250 * 1024**3)


def _process_one(file_id: str) -> dict:
    """Worker: 从 LMDB 读 token → 分类织体。"""
    global _worker_store
    store = _worker_store
    if store is None:
        return {'status': 'error', 'file_id': file_id, 'reason': 'store not initialized'}

    try:
        tokens = store.get_tokens(file_id)
    except Exception as e:
        return {'status': 'error', 'file_id': file_id, 'reason': str(e)}

    if not isinstance(tokens, list) or len(tokens) < 10:
        return {'status': 'skip', 'file_id': file_id, 'reason': 'too_short'}

    try:
        data = classify_tokens(tokens)
    except Exception as e:
        return {'status': 'error', 'file_id': file_id, 'reason': str(e)}

    # 跳过无织体标注的文件
    if not data.get('bar_figs'):
        return {'status': 'skip', 'file_id': file_id, 'reason': 'no_figuration'}

    return {'status': 'ok', 'file_id': file_id, 'data': data}


# ── CLI ────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Figuration annotation (LMDB)')
    ap.add_argument('command', choices=['annotate'])
    ap.add_argument('--lmdb-path', required=True, help='LMDB 路径 (读写)')
    ap.add_argument('--num-workers', type=int, default=16)
    args = ap.parse_args()

    _init_constants()

    from chopinote_dataset.lmdb_store import LMDBStore

    PENDING_FILE = '/root/autodl-tmp/fig_pending.txt'
    DONE_FILE = '/root/autodl-tmp/fig_done.txt'

    # ── Phase 1: 获取待处理列表 ──
    print("[Phase 1] 获取待处理列表...", flush=True)
    t0 = time.time()

    done_ids = set()
    if os.path.exists(DONE_FILE):
        with open(DONE_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    done_ids.add(line)
        print(f"  已完成 (from file): {len(done_ids)}", flush=True)

    if os.path.exists(PENDING_FILE):
        # 快速路径: 从进度文件恢复
        print("  从进度文件恢复...", flush=True)
        pending_ids = []
        with open(PENDING_FILE) as f:
            for line in f:
                line = line.strip()
                if line and line not in done_ids:
                    pending_ids.append(line)
        print(f"  待标注: {len(pending_ids)} ({time.time() - t0:.1f}s)", flush=True)
    else:
        # 首次: 全量 LMDB 扫描
        print("  首次运行, 扫描 LMDB (仅此一次)...", flush=True)
        ro_store = LMDBStore.open(args.lmdb_path, readonly=True, map_size=250 * 1024**3)
        try:
            already_fig = set()
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
                            if dtype == 'fig':
                                already_fig.add(fid)
                            elif dtype == 'tokens' and fid not in seen:
                                seen.add(fid)
                                if fid not in already_fig:
                                    tokens_fids.append(fid)
                    cursor.close()
        finally:
            ro_store.close()

        pending_ids = [f for f in tokens_fids if f not in already_fig]
        if done_ids:
            pending_ids = [f for f in pending_ids if f not in done_ids]

        print(f"  已有 Fig: {len(already_fig)}")
        print(f"  总 tokens: {len(seen)}")
        print(f"  待标注: {len(pending_ids)}")
        print(f"  扫描耗时: {time.time() - t0:.1f}s", flush=True)

        with open(PENDING_FILE, 'w') as f:
            for fid in pending_ids:
                f.write(fid + '\n')
        print(f"  进度文件已保存: {PENDING_FILE}", flush=True)

    if not pending_ids:
        print("无需标注，全部完成。")
        return

    # ── Phase 2: 并行标注 ──
    print(f"\n[Phase 2] 启动 {args.num_workers} workers...", flush=True)
    stats = Counter()
    start = time.time()
    completed = 0
    error_samples = []
    skip_reasons = Counter()
    total_pending = len(pending_ids)
    BATCH_SIZE = 2000
    WRITE_CHUNK = 100000

    done_fh = open(DONE_FILE, 'a')
    pool = mp.Pool(args.num_workers, initializer=_worker_init,
                   initargs=(args.lmdb_path,))
    write_store = LMDBStore.open(args.lmdb_path, readonly=False, map_size=250 * 1024**3)

    try:
        for chunk_start in range(0, len(pending_ids), WRITE_CHUNK):
            chunk = pending_ids[chunk_start:chunk_start + WRITE_CHUNK]
            batch = []

            for result in pool.imap_unordered(_process_one, chunk, chunksize=20):
                completed += 1
                status = result['status']
                stats[status] += 1

                if status == 'ok':
                    fid = result['file_id']
                    batch.append((fid, 'fig', result['data']))
                    del result['data']
                elif status == 'error' and len(error_samples) < 5:
                    error_samples.append(f"{result['file_id']}: {result.get('reason', '?')}")
                elif status == 'skip':
                    skip_reasons[result.get('reason', '?')] += 1

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
                    if skip_reasons:
                        print(f"  跳过原因: {dict(skip_reasons.most_common(3))}", flush=True)
                    gc.collect()

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
    if not batch:
        return
    with store.env.begin(db=store.main_db, write=True) as txn:
        for fid, dtype, data in batch:
            store._txn_put(txn, fid, dtype, data)


if __name__ == '__main__':
    main()
