#!/usr/bin/env python3
"""
功能和声标注脚本 — v0.3.3 LMDB 版 (v3 模板 rewrite)。
从 LMDB 读 SSF 数据，用尖锐模板 + ratio test 分类 T/SD/D/SDom。

核心改进 (v3):
  - 模板按和弦音设计 (I=pos0,4,7 / IV=pos5,9,0 / V=pos7,11,2 / iv=pos5,8,0)
  - ratio test (best_sim / second_best) 替代 sum-based confidence
  - 节拍级精细标注: 每拍独立分类
  - 非功能拍标记为 "non-func" (不是 "none"), 保留 SSF 原始向量

三粒度:
  - section_funcs: 段落级 (从 TonicField 分类)
  - bar_funcs:     小节级 (TonicField + LocalField + Markov)
  - beat_funcs:    节拍级 (BeatField + 局部 Markov) ← 主粒度

输出格式 (LMDB key: v4:<file_id>:func):
  {
    "version": 3,
    "section_funcs": [{"section": i, "func": "T"|"SD"|"D"|"SDom"|"non-func", "ratio": 0.X}, ...],
    "bar_funcs": [...],
    "beat_funcs": [{"bar": b, "pos": p, "func": ..., "ratio": 0.X}, ...]
  }

用法:
  python scripts/annotate_function.py annotate \
      --lmdb-path /root/autodl-tmp/data/processed/lmdb/chopinote_v4.lmdb \
      --num-workers 25
"""
import sys, os, time, json, math, argparse, gc, multiprocessing as mp
from collections import Counter


# ── 功能模板 (v3: 按和弦音设计, 主音锚定, 12 维) ─────────────
# pos 0 = tonic PC. 和弦音强峰, 非和弦音接近 0.
#   T(I):    1,3,5 → pos 0, 4, 7
#   SD(IV):  4,6,1 → pos 5, 9, 0
#   D(V):    5,7,2 → pos 7,11, 2
#   SDom(iv): 4,b6,1 → pos 5, 8, 0
FUNCTION_TEMPLATES = {
    'T':    [1.0, 0.0, 0.1, 0.0, 0.8, 0.0, 0.0, 0.8, 0.0, 0.1, 0.0, 0.0],
    'SD':   [0.6, 0.0, 0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.1],
    'D':    [0.1, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.1, 0.0, 0.7],
    'SDom': [0.5, 0.0, 0.1, 0.0, 0.1, 1.0, 0.0, 0.0, 0.7, 0.1, 0.0, 0.0],
}

FUNC_NAMES = ['T', 'SD', 'D', 'SDom']

# ── Ratio test 阈值 ────────────────────────────────────────
SIM_THRESHOLD = 0.35     # best_sim 下限 (低于此 → non-func)
RATIO_THRESHOLD = 1.20   # best_sim / second_best 下限 (低于此 → ambiguous)

# ── 模板间相似度 (用于调整 Markov 权重) ─────────────────────
# SD vs SDom = 0.735 (共享 pos5), 其他 <0.45
_MARKOV_WEIGHT = 0.2  # Markov prior 权重 (降低以避免过强偏置)

# ── Markov 转移概率 ────────────────────────────────────────
MARKOV_TRANSITION = {
    'T':    {'T': 0.25, 'SD': 0.40, 'D': 0.30, 'SDom': 0.05},
    'SD':   {'T': 0.15, 'SD': 0.20, 'D': 0.55, 'SDom': 0.10},
    'D':    {'T': 0.65, 'SD': 0.10, 'D': 0.15, 'SDom': 0.10},
    'SDom': {'T': 0.10, 'SD': 0.00, 'D': 0.80, 'SDom': 0.10},
}


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def classify_ssf_vector(
    ssf_vec: list[float],
    prev_func: str | None = None,
) -> dict:
    """对单个 SSF 向量做 ratio-test 分类。

    Returns:
        {"func": "T"|"SD"|"D"|"SDom"|"non-func",
         "ratio": float,        # best_sim / second_best
         "best_sim": float,
         "reason": str | None}  # non-func 原因: "empty" | "low_sim" | "ambiguous"
    """
    # 空向量 (无音符)
    if all(v == 0.5 for v in ssf_vec) or all(v == 0.0 for v in ssf_vec):
        return {'func': 'non-func', 'ratio': 0.0, 'best_sim': 0.0, 'reason': 'empty'}

    # 对所有模板计算余弦相似度
    sim_scores = {f: cosine_similarity(ssf_vec, FUNCTION_TEMPLATES[f]) for f in FUNC_NAMES}

    # 按相似度降序排列
    sorted_funcs = sorted(sim_scores, key=sim_scores.get, reverse=True)
    best_func = sorted_funcs[0]
    second_func = sorted_funcs[1]
    best_sim = sim_scores[best_func]
    second_sim = sim_scores[second_func]

    # Check 1: 最高相似度太低
    if best_sim < SIM_THRESHOLD:
        return {'func': 'non-func', 'ratio': round(best_sim / max(second_sim, 0.01), 3),
                'best_sim': round(best_sim, 3), 'reason': 'low_sim'}

    # 有 Markov 先验时: 对 scores 做微调
    if prev_func is not None and prev_func in MARKOV_TRANSITION:
        adjusted = {}
        for f in FUNC_NAMES:
            prior = MARKOV_TRANSITION.get(prev_func, {}).get(f, 0.25)
            adjusted[f] = sim_scores[f] * (1.0 + _MARKOV_WEIGHT * prior)
        # 重新排序
        sorted_funcs = sorted(adjusted, key=adjusted.get, reverse=True)
        best_func = sorted_funcs[0]
        second_func = sorted_funcs[1]
        best_sim = sim_scores[best_func]      # ratio 仍用原始 score
        second_sim = sim_scores[second_func]

    # Check 2: ratio test — 第一名必须显著优于第二名
    ratio = best_sim / max(second_sim, 0.01)
    if ratio < RATIO_THRESHOLD:
        return {'func': 'non-func', 'ratio': round(ratio, 3),
                'best_sim': round(best_sim, 3), 'reason': 'ambiguous'}

    return {'func': best_func, 'ratio': round(ratio, 3),
            'best_sim': round(best_sim, 3), 'reason': None}


def annotate_from_ssf(ssf_data: dict) -> dict:
    """从 SSF 数据标注三粒度功能和声 (v3)。"""
    tonic_fields = ssf_data.get('tonic_fields', [])
    local_fields = ssf_data.get('local_fields', {})
    beat_fields = ssf_data.get('beat_fields', {})

    # ── 段落级: 无 Markov ──
    section_funcs = []
    for i, tf in enumerate(tonic_fields):
        result = classify_ssf_vector(tf, prev_func=None)
        result['section'] = i
        section_funcs.append(result)

    # ── 小节级: 带 Markov ──
    bar_funcs = []
    prev_bar_func = 'T'  # 初始假设

    if beat_fields:
        num_bars = max(int(k) for k in beat_fields.keys()) + 1
    elif local_fields:
        num_bars = max(int(k) for k in local_fields.keys()) + 1
    else:
        num_bars = max(1, len(tonic_fields) * 8)

    boundaries = ssf_data.get('section_boundaries', [0])

    for b in range(num_bars):
        # 找该 bar 所属 section 的 TonicField
        sec_idx = 0
        for i in range(len(boundaries)):
            if b >= boundaries[i]:
                sec_idx = i
        base_tf = tonic_fields[sec_idx] if sec_idx < len(tonic_fields) else [0.5] * 12

        # 合成 bar 级 SSF = TonicField + LocalField delta
        bar_ssf = list(base_tf)
        b_str = str(b)
        if b_str in local_fields:
            delta = local_fields[b_str]
            for j in range(12):
                bar_ssf[j] = max(0.0, min(1.0, bar_ssf[j] + delta[j]))

        # 如果该 bar 有 BeatField, 优先用 BeatField 的聚合 (更精确)
        if b_str in beat_fields:
            beats = beat_fields[b_str]
            if beats:
                # 聚合所有 beat 的 SSF
                agg = [0.0] * 12
                n = 0
                for pos_str, bf in beats.items():
                    for j in range(12):
                        agg[j] += bf[j]
                    n += 1
                bar_ssf = [v / n for v in agg]

        result = classify_ssf_vector(bar_ssf, prev_func=prev_bar_func)
        result['bar'] = b
        bar_funcs.append(result)
        if result['func'] != 'non-func':
            prev_bar_func = result['func']

    # ── 节拍级: 主粒度, 局部 Markov ──
    beat_funcs = []
    for b_str, beats in sorted(beat_fields.items(), key=lambda x: int(x[0])):
        b = int(b_str)
        prev_beat_func = 'T'
        for pos_str, bf in sorted(beats.items(), key=lambda x: int(x[0])):
            pos = int(pos_str)
            result = classify_ssf_vector(bf, prev_func=prev_beat_func)
            result['bar'] = b
            result['pos'] = pos
            beat_funcs.append(result)
            if result['func'] != 'non-func':
                prev_beat_func = result['func']

    return {
        'version': 3,
        'section_funcs': section_funcs,
        'bar_funcs': bar_funcs,
        'beat_funcs': beat_funcs,
    }


# ── Worker ───────────────────────────────────────────────

_worker_store = None


def _worker_init(lmdb_path: str):
    """Pool initializer: 每个 worker 进程打开一次只读 LMDB。"""
    global _worker_store
    from chopinote_dataset.lmdb_store import LMDBStore
    _worker_store = LMDBStore.open(lmdb_path, readonly=True, map_size=350 * 1024**3)


def _process_one(file_id: str) -> dict:
    """Worker: 从 LMDB 读 SSF → 分类功能和声。"""
    global _worker_store
    store = _worker_store
    if store is None:
        return {'status': 'error', 'file_id': file_id, 'reason': 'store not initialized'}

    try:
        ssf_data = store.get_ssf(file_id)
    except Exception as e:
        return {'status': 'error', 'file_id': file_id, 'reason': str(e)}

    if ssf_data is None:
        return {'status': 'skip', 'file_id': file_id, 'reason': 'no_ssf'}

    try:
        data = annotate_from_ssf(ssf_data)
    except Exception as e:
        return {'status': 'error', 'file_id': file_id, 'reason': str(e)}

    return {'status': 'ok', 'file_id': file_id, 'data': data}


# ── CLI ────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='功能和声标注 (LMDB, 读 SSF)')
    ap.add_argument('command', choices=['annotate'])
    ap.add_argument('--lmdb-path', required=True, help='LMDB 路径 (读写)')
    ap.add_argument('--num-workers', type=int, default=25)
    args = ap.parse_args()

    from chopinote_dataset.lmdb_store import LMDBStore

    PENDING_FILE = '/root/autodl-tmp/func_pending.txt'
    DONE_FILE = '/root/autodl-tmp/func_done.txt'

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
        # 首次: 全量 LMDB 扫描 — 所有有 SSF 的文件都处理 (覆盖 v2)
        print("  首次运行, 扫描 LMDB (仅此一次)...", flush=True)
        ro_store = LMDBStore.open(args.lmdb_path, readonly=True, map_size=350 * 1024**3)
        try:
            has_ssf = set()
            with ro_store.env.begin(db=ro_store.main_db) as txn:
                cursor = txn.cursor()
                if cursor.set_range(b'v4:'):
                    for key, _ in cursor:
                        if not key.startswith(b'v4:'):
                            break
                        parts = key.decode('utf-8').split(':')
                        if len(parts) >= 3 and parts[2] == 'ssf':
                            has_ssf.add(parts[1])
                    cursor.close()
        finally:
            ro_store.close()

        pending_ids = sorted(has_ssf)
        if done_ids:
            pending_ids = [f for f in pending_ids if f not in done_ids]

        print(f"  有 SSF: {len(has_ssf)}")
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
    write_store = LMDBStore.open(args.lmdb_path, readonly=False, map_size=350 * 1024**3)

    try:
        for chunk_start in range(0, len(pending_ids), WRITE_CHUNK):
            chunk = pending_ids[chunk_start:chunk_start + WRITE_CHUNK]
            batch = []

            for result in pool.imap_unordered(_process_one, chunk, chunksize=50):
                completed += 1
                status = result['status']
                stats[status] += 1

                if status == 'ok':
                    fid = result['file_id']
                    batch.append((fid, 'func', result['data']))
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
