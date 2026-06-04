#!/usr/bin/env python3
"""功能和声标注脚本 — 从 Note_ON interval 推断 T/SD/D/SDom。

方案 B: 扫描已有 .tokens 文件, 对每个 bar 推断功能标签,
写入 .func.json 侧文件。不修改 .tokens, 不需要重转换。

Usage:
    python scripts/annotate_function.py annotate \
        --input-dir /root/autodl-tmp/data/processed/tokens_v4 \
        --num-workers 25
"""
import sys, os, time, json, logging, math, argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ── 功能 PC 模板 (主音锚定, 12 维) ──────────────────────────────
FUNCTION_TEMPLATES = {
    'T':  [1.0, 0.1, 0.2, 0.1, 0.7, 0.1, 0.1, 0.7, 0.1, 0.3, 0.1, 0.2],
    #     C    C#   D    Eb   E    F    F#   G    Ab   A    Bb   B
    'SD': [0.5, 0.1, 0.3, 0.1, 0.3, 0.8, 0.1, 0.3, 0.6, 0.2, 0.2, 0.1],
    'D':  [0.5, 0.1, 0.4, 0.1, 0.3, 0.3, 0.1, 0.9, 0.1, 0.2, 0.6, 0.4],
    'SDom': [0.3, 0.1, 0.8, 0.1, 0.2, 0.3, 0.7, 0.3, 0.1, 0.1, 0.1, 0.1],
}

FUNC_NAMES = ['T', 'SD', 'D', 'SDom']
FUNC_TO_ID = {'T': 1, 'SD': 2, 'D': 3, 'SDom': 4}

# ── Markov 转移概率 ────────────────────────────────────────────
# P(next | current), 基于经典功能和声统计
MARKOV_TRANSITION = {
    'T':    {'T': 0.3, 'SD': 0.4, 'D': 0.25, 'SDom': 0.05},
    'SD':   {'T': 0.15, 'SD': 0.2, 'D': 0.55, 'SDom': 0.1},
    'D':    {'T': 0.7, 'SD': 0.1, 'D': 0.15, 'SDom': 0.05},
    'SDom': {'T': 0.15, 'SD': 0.0, 'D': 0.8, 'SDom': 0.05},
}


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """两个向量的余弦相似度。"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class FunctionAnnotator:
    """功能和声推断器。

    对每个 bar:
    1. 收集 Note_ON interval → 主音锚定 PC histogram
    2. 计算与四类功能模板的 cosine similarity
    3. 乘 Markov 转移先验 (从前一 bar 的功能)
    4. MAP → 最高后验概率的功能标签
    """

    def __init__(self, tokenizer):
        self.tk = tokenizer
        self.bar_token_id = 4

        # 预计算 Note_ON token ID → interval 映射
        self._note_on_tid_to_interval: dict[int, int] = {}
        for token_str, token_id in tokenizer._token_to_id.items():
            if token_str.startswith('<Note_ON '):
                try:
                    interval = int(token_str.split()[1].rstrip('>'))
                    self._note_on_tid_to_interval[token_id] = interval
                except (ValueError, IndexError):
                    pass

        # 预计算 Tonic token ID → tonic_name 映射
        self._tonic_tid_to_name: dict[int, str] = {}
        for token_str, token_id in tokenizer._token_to_id.items():
            if token_str.startswith('<Tonic ') and token_str.endswith('>'):
                tonic_name = token_str[7:-1]
                self._tonic_tid_to_name[token_id] = tonic_name

        # 主音名 → chroma 偏移 (C=0)
        self._tonic_chroma: dict[str, int] = {
            'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4,
            'F': 5, 'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11,
        }

    def _build_pc_histogram(self, intervals: list[int], tonic_name: str) -> list[float]:
        """将 Note_ON interval 列表转换为主音锚定 PC histogram (12维, 归一化)。

        interval 是相对于主音 MIDI 60 的半音程 (-60..60)。
        pitch_class = (60 + interval) % 12 → tonic-anchored → (pc - tonic_chroma) % 12
        """
        hist = [0.0] * 12
        tonic_chroma = self._tonic_chroma.get(tonic_name, 0)
        for interval in intervals:
            pc = (60 + interval) % 12
            anchored = (pc - tonic_chroma) % 12
            hist[anchored] += 1.0
        # 归一化
        total = sum(hist)
        if total > 0:
            hist = [h / total for h in hist]
        return hist

    def _infer_tonic(self, tokens: list[int]) -> str:
        """从 token 序列中推断主音 (取最后一个 Tonic token)。
        无 Tonic token 时默认 C 大调。
        """
        current_tonic = 'C'
        for tid in tokens:
            tonic = self._tonic_tid_to_name.get(tid)
            if tonic is not None:
                current_tonic = tonic
        return current_tonic

    def annotate(self, tokens: list[int]) -> dict:
        """对完整 token 序列进行功能和声标注。

        Returns:
            {"version": 1, "num_bars": N, "functions": [{"bar": 0, "func": "T", "confidence": 0.92}, ...]}
        """
        # ── Step 1: 分 bar + 收集 Note_ON interval ──────────
        bars: list[list[int]] = []  # bar_idx → list of Note_ON intervals
        current_bar: list[int] = []
        current_tonic = 'C'

        for tid in tokens:
            # 跟踪主音
            tonic = self._tonic_tid_to_name.get(tid)
            if tonic is not None:
                current_tonic = tonic

            if tid == self.bar_token_id:
                if current_bar:
                    bars.append(current_bar)
                current_bar = []
            else:
                interval = self._note_on_tid_to_interval.get(tid)
                if interval is not None:
                    current_bar.append(interval)

        # 最后一个 bar
        if current_bar:
            bars.append(current_bar)

        if not bars:
            bars.append([])

        # ── Step 2: 逐 bar 推断功能 ──────────────────────────
        functions = []
        prev_func = 'T'  # 默认从 Tonic 开始

        for bar_idx, intervals in enumerate(bars):
            if not intervals:
                # 无音符 bar: 延续前一功能, 低置信度
                functions.append({
                    'bar': bar_idx,
                    'func': prev_func,
                    'confidence': 0.3,
                })
                continue

            # PC histogram
            hist = self._build_pc_histogram(intervals, current_tonic)

            # Cosine similarity with each template
            sim_scores = {}
            for fname in FUNC_NAMES:
                sim_scores[fname] = cosine_similarity(hist, FUNCTION_TEMPLATES[fname])

            # 乘 Markov transition prior (prior weight 0.7 强化句法约束)
            posterior = {}
            for fname in FUNC_NAMES:
                prior = MARKOV_TRANSITION.get(prev_func, {}).get(fname, 0.25)
                posterior[fname] = sim_scores[fname] * (0.3 + 0.7 * prior)

            # MAP
            best_func = max(posterior, key=posterior.get)
            best_score = posterior[best_func]
            # 归一化置信度
            total_post = sum(posterior.values())
            confidence = best_score / total_post if total_post > 0 else 0.5

            functions.append({
                'bar': bar_idx,
                'func': best_func,
                'confidence': round(confidence, 3),
            })

            prev_func = best_func

        return {
            'version': 1,
            'num_bars': len(bars),
            'functions': functions,
        }


# ═══════════════════════════════════════════════════════════════
#  标注入口
# ═══════════════════════════════════════════════════════════════

def _annotate_one(args: tuple[str, str]) -> tuple[str, bool, str]:
    """单个文件的标注 (worker 调用)。"""
    token_path, output_dir = args
    try:
        # 延迟加载 tokenizer (每个 worker 一个)
        import threading
        pid = threading.current_thread().ident
        tk = _get_worker_tokenizer()

        with open(token_path, 'r') as f:
            tokens = json.load(f)

        annotator = FunctionAnnotator(tk)
        result = annotator.annotate(tokens)

        # 写入同目录 (与 .tokens 同位置)
        out_path = Path(token_path).with_suffix('.func.json')
        with open(out_path, 'w') as f:
            json.dump(result, f, separators=(',', ':'))

        return (token_path, True, '')
    except Exception as e:
        return (token_path, False, str(e))


_worker_tokenizer = None
_worker_tokenizer_lock = None


def _init_worker():
    """Pool worker 初始化。"""
    global _worker_tokenizer_lock
    import threading
    _worker_tokenizer_lock = threading.Lock()


def _get_worker_tokenizer():
    """每个 worker 线程延迟初始化一个 tokenizer 实例。"""
    global _worker_tokenizer, _worker_tokenizer_lock
    if _worker_tokenizer is None:
        with _worker_tokenizer_lock:
            if _worker_tokenizer is None:
                from chopinote_dataset.tokenizer import REMITokenizer
                _worker_tokenizer = REMITokenizer(grid_size=16, velocity_levels=8)
    return _worker_tokenizer


def annotate_all(input_dir: str, num_workers: int = 25):
    """扫描 input_dir 下所有 .tokens 文件, 并行标注功能和声。

    对已有 .func.json 的文件跳过 (幂等性)。
    """
    token_dir = Path(input_dir)
    if not token_dir.is_dir():
        logger.error(f"目录不存在: {input_dir}")
        return

    # 收集所有 .tokens 文件
    all_tokens = list(token_dir.glob('*.tokens'))
    logger.info(f"找到 {len(all_tokens)} 个 .tokens 文件")

    # 跳过已有 .func.json 的文件
    tasks = []
    skipped = 0
    for tp in all_tokens:
        func_path = tp.with_suffix('.func.json')
        if func_path.exists():
            skipped += 1
        else:
            tasks.append((str(tp), str(token_dir)))

    logger.info(f"已标注: {skipped}, 待标注: {len(tasks)}")
    if not tasks:
        logger.info("全部已标注, 无需操作")
        return

    t0 = time.time()
    completed = 0
    failed = 0

    with Pool(processes=num_workers, initializer=_init_worker) as pool:
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
    parser = argparse.ArgumentParser(description='功能和声标注')
    sub = parser.add_subparsers(dest='cmd')

    annotate_p = sub.add_parser('annotate', help='标注 .tokens → .func.json')
    annotate_p.add_argument('--input-dir', required=True, help='tokens_v4 目录')
    annotate_p.add_argument('--num-workers', type=int, default=25)

    # test 子命令: 对单个文件测试
    test_p = sub.add_parser('test', help='对单个文件测试标注')
    test_p.add_argument('--input', required=True, help='.tokens 文件路径')

    args = parser.parse_args()

    if args.cmd == 'annotate':
        annotate_all(args.input_dir, args.num_workers)
    elif args.cmd == 'test':
        from chopinote_dataset.tokenizer import REMITokenizer
        tk = REMITokenizer(grid_size=16, velocity_levels=8)
        with open(args.input, 'r') as f:
            tokens = json.load(f)
        annotator = FunctionAnnotator(tk)
        result = annotator.annotate(tokens)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
