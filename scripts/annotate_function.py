#!/usr/bin/env python3
"""功能和声标注脚本 — 从 .ssf.json 的 SSF 向量分类 T/SD/D/SDom。

不重复解析 .tokens，直接读 SSF 标注结果：
  - 段落级 TonicField    → section_funcs
  - 小节级 LocalField    → bar_funcs
  - 节拍级 BeatField     → beat_funcs

超过余弦相似度阈值才标注，未达标的标记为 none (func_id=0)。

Usage:
    python scripts/annotate_function.py annotate \
        --input-dir /root/autodl-tmp/data/processed/tokens_v4 \
        --num-workers 25
"""
import sys, os, time, json, logging, math, argparse
from pathlib import Path
from multiprocessing import Pool

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ── 功能 PC 模板 (主音锚定, 12 维) ──────────────────────────────
FUNCTION_TEMPLATES = {
    'T':  [1.0, 0.1, 0.2, 0.1, 0.7, 0.1, 0.1, 0.7, 0.1, 0.3, 0.1, 0.2],
    'SD': [0.5, 0.1, 0.3, 0.1, 0.3, 0.8, 0.1, 0.3, 0.6, 0.2, 0.2, 0.1],
    'D':  [0.5, 0.1, 0.4, 0.1, 0.3, 0.3, 0.1, 0.9, 0.1, 0.2, 0.6, 0.4],
    'SDom': [0.3, 0.1, 0.8, 0.1, 0.2, 0.3, 0.7, 0.3, 0.1, 0.1, 0.1, 0.1],
}

FUNC_NAMES = ['T', 'SD', 'D', 'SDom']
FUNC_TO_ID = {'T': 1, 'SD': 2, 'D': 3, 'SDom': 4}

# ── 分类阈值 ────────────────────────────────────────────────
CLASSIFY_THRESHOLD = 0.50       # cosine similarity 最低阈值
CONFIDENCE_FLOOR = 0.55         # 归一化后验最低置信度

# ── Markov 转移概率 ────────────────────────────────────────────
MARKOV_TRANSITION = {
    'T':    {'T': 0.3, 'SD': 0.4, 'D': 0.25, 'SDom': 0.05},
    'SD':   {'T': 0.15, 'SD': 0.2, 'D': 0.55, 'SDom': 0.1},
    'D':    {'T': 0.7, 'SD': 0.1, 'D': 0.15, 'SDom': 0.05},
    'SDom': {'T': 0.15, 'SD': 0.0, 'D': 0.8, 'SDom': 0.05},
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
    use_markov: bool = True,
) -> tuple[str | None, float]:
    """对单个 SSF 向量分类为功能标签。

    Args:
        ssf_vec: 12 维 SSF chroma 向量
        prev_func: 前一个功能标签 (用于 Markov 先验)
        use_markov: 是否使用 Markov 转移先验

    Returns:
        (func_name, confidence) 或 (None, 0.0)
    """
    if all(v == 0.5 for v in ssf_vec):
        return (None, 0.0)

    # Cosine similarity
    sim_scores = {}
    for fname in FUNC_NAMES:
        sim_scores[fname] = cosine_similarity(ssf_vec, FUNCTION_TEMPLATES[fname])

    best_func = max(sim_scores, key=sim_scores.get)
    best_sim = sim_scores[best_func]

    # 低于阈值 → 不标注
    if best_sim < CLASSIFY_THRESHOLD:
        return (None, 0.0)

    # Markov prior (仅对有时间顺序的级别使用)
    if use_markov and prev_func is not None:
        posterior = {}
        for fname in FUNC_NAMES:
            prior = MARKOV_TRANSITION.get(prev_func, {}).get(fname, 0.25)
            posterior[fname] = sim_scores[fname] * (0.3 + 0.7 * prior)
        best_func = max(posterior, key=posterior.get)
        best_score = posterior[best_func]
        total = sum(posterior.values())
        confidence = best_score / total if total > 0 else 0.0
    else:
        total = sum(sim_scores.values())
        confidence = best_sim / total if total > 0 else 0.0

    if confidence < CONFIDENCE_FLOOR:
        return (None, 0.0)

    return (best_func, round(confidence, 3))


# ═══════════════════════════════════════════════════════════════
#  标注逻辑
# ═══════════════════════════════════════════════════════════════

def annotate_from_ssf(ssf_data: dict) -> dict:
    """从 SSF 数据标注三粒度功能和声。

    Returns:
        {
            "version": 2,
            "section_funcs": [...],
            "bar_funcs": [...],
            "beat_funcs": [...]
        }
    """
    tonic_fields = ssf_data.get('tonic_fields', [])
    local_fields = ssf_data.get('local_fields', {})
    beat_fields = ssf_data.get('beat_fields', {})

    # ── 段落级: 不使用 Markov (段落间独立) ──
    section_funcs = []
    for i, tf in enumerate(tonic_fields):
        func, conf = classify_ssf_vector(tf, prev_func=None, use_markov=False)
        section_funcs.append({
            'section': i,
            'func': func or 'none',
            'confidence': conf,
        })

    # ── 小节级: 使用 Markov 链 ──
    bar_funcs = []
    prev_bar_func = 'T'
    # 需要知道总 bar 数 → 从 beat_fields 推断
    if beat_fields:
        num_bars = max(int(k) for k in beat_fields.keys()) + 1
    elif local_fields:
        num_bars = max(int(k) for k in local_fields.keys()) + 1
    else:
        num_bars = len(tonic_fields) * 8  # 粗略估计

    for b in range(num_bars):
        # 找到该 bar 所属 section
        sec_idx = 0
        boundaries = ssf_data.get('section_boundaries', [0])
        for i in range(len(boundaries)):
            if b >= boundaries[i]:
                sec_idx = i

        # 从 section TonicField + bar LocalField 合成 bar 级 SSF
        base_tf = tonic_fields[sec_idx] if sec_idx < len(tonic_fields) else [0.5] * 12
        bar_ssf = list(base_tf)  # copy
        b_str = str(b)
        if b_str in local_fields:
            delta = local_fields[b_str]
            for j in range(12):
                bar_ssf[j] = max(0.0, min(1.0, bar_ssf[j] + delta[j]))

        func, conf = classify_ssf_vector(bar_ssf, prev_func=prev_bar_func, use_markov=True)
        bar_funcs.append({
            'bar': b,
            'func': func or 'none',
            'confidence': conf,
        })
        if func is not None:
            prev_bar_func = func

    # ── 节拍级: 使用局部 Markov 链 ──
    beat_funcs = []
    for b_str, beats in sorted(beat_fields.items(), key=lambda x: int(x[0])):
        b = int(b_str)
        prev_beat_func = 'T'
        for pos_str, bf in sorted(beats.items(), key=lambda x: int(x[0])):
            pos = int(pos_str)
            func, conf = classify_ssf_vector(bf, prev_func=prev_beat_func, use_markov=True)
            beat_funcs.append({
                'bar': b,
                'pos': pos,
                'func': func or 'none',
                'confidence': conf,
            })
            if func is not None:
                prev_beat_func = func

    return {
        'version': 2,
        'section_funcs': section_funcs,
        'bar_funcs': bar_funcs,
        'beat_funcs': beat_funcs,
    }


# ═══════════════════════════════════════════════════════════════
#  并行标注入口
# ═══════════════════════════════════════════════════════════════

def _annotate_one(args: tuple[str, str]) -> tuple[str, bool, str]:
    """单个文件标注。"""
    token_path, output_dir = args
    try:
        # 读 SSF (不是 tokens)
        ssf_path = Path(token_path).with_suffix('.ssf.json')
        if not ssf_path.exists():
            return (token_path, False, 'no_ssf')

        with open(ssf_path, 'r') as f:
            ssf_data = json.load(f)

        result = annotate_from_ssf(ssf_data)

        out_path = Path(token_path).with_suffix('.func.json')
        with open(out_path, 'w') as f:
            json.dump(result, f, separators=(',', ':'))

        return (token_path, True, '')
    except Exception as e:
        return (token_path, False, str(e))


def annotate_all(input_dir: str, num_workers: int = 25):
    """扫描已有 .ssf.json 文件, 并行标注功能和声。"""
    token_dir = Path(input_dir)
    if not token_dir.is_dir():
        logger.error(f"目录不存在: {input_dir}")
        return

    ssf_files = list(token_dir.glob('*.ssf.json'))
    logger.info(f"找到 {len(ssf_files)} 个 .ssf.json 文件")

    tasks = []
    skipped = 0
    for sp in ssf_files:
        tp = sp.with_suffix('.tokens')
        func_path = tp.with_suffix('.func.json')
        if func_path.exists():
            skipped += 1
        else:
            tasks.append((str(tp), str(token_dir)))

    logger.info(f"已标注: {skipped}, 待标注: {len(tasks)}")
    if not tasks:
        logger.info("全部已标注")
        return

    t0 = time.time()
    completed = 0
    failed = 0

    with Pool(processes=num_workers) as pool:
        for path, ok, err in pool.imap_unordered(_annotate_one, tasks, chunksize=50):
            completed += 1
            if not ok:
                failed += 1
            if completed % 5000 == 0:
                elapsed = time.time() - t0
                rate = completed / max(1, elapsed)
                eta = (len(tasks) - completed) / max(1, rate)
                logger.info(
                    f"  进度 {completed}/{len(tasks)} "
                    f"({completed / len(tasks) * 100:.1f}%) "
                    f"| {rate:.0f} f/s | ETA {eta:.0f}s "
                    f"| 失败 {failed}"
                )

    elapsed = time.time() - t0
    logger.info(
        f"功能和声标注完成: {completed} 标注, {failed} 失败, "
        f"{skipped} 跳过 ({elapsed:.0f}s)"
    )


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='功能和声标注 (读 SSF)')
    sub = parser.add_subparsers(dest='cmd')

    annotate_p = sub.add_parser('annotate', help='标注 .ssf.json → .func.json')
    annotate_p.add_argument('--input-dir', required=True, help='tokens_v4 目录')
    annotate_p.add_argument('--num-workers', type=int, default=25)

    test_p = sub.add_parser('test', help='对单个文件测试')
    test_p.add_argument('--input', required=True, help='.tokens 文件路径 (自动找 .ssf.json)')

    args = parser.parse_args()

    if args.cmd == 'annotate':
        annotate_all(args.input_dir, args.num_workers)
    elif args.cmd == 'test':
        ssf_path = Path(args.input).with_suffix('.ssf.json')
        if not ssf_path.exists():
            print(f"SSF 文件不存在: {ssf_path}")
            sys.exit(1)
        with open(ssf_path, 'r') as f:
            ssf_data = json.load(f)
        result = annotate_from_ssf(ssf_data)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
