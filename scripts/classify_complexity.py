#!/usr/bin/env python3
"""
数据质量过滤 + 自动分类 + 训练集拆分 — v0.3.1-data2。

F1-F5 质量过滤:
  F1: 调性清晰度 — TonicField peakiness < 1.3 → 丢弃
  F2: 调性稳定性 — 主音变化率 > 0.5/bar → 丢弃
  F3: 结构合理性 — 无音符 / note:dur偏差 / bar密度极端 / 无小节 → 丢弃
  F4: 长度异常 — <50 或 >16384 tokens → 丢弃
  F5: Duration 越界率 — >5% 事件越界 → 丢弃

四指标自动分类:
  Texture(1-3) + Structure(1-5) + Rhythm(1-3) + Instr(1-4) → Level(1-5)

用法:
  # 扫描并分类
  python scripts/classify_complexity.py classify \
      --input-dir /root/autodl-tmp/data/processed/tokens_v4 \
      --output /root/autodl-tmp/data/processed/complexity_labels.json \
      --num-workers 16

  # 生成 train/val 拆分
  python scripts/classify_complexity.py split \
      --labels /root/autodl-tmp/data/processed/complexity_labels.json \
      --output-dir /root/autodl-tmp/data/processed/ \
      --train-ratio 0.9

设计文档: docs/curriculum_training_v0.3.x.md
"""
import os, sys, json, argparse, time, math, random, multiprocessing as mp
from collections import Counter, defaultdict

from chopinote_dataset.tokenizer import REMITokenizer

# ── 主音名 → pitch class ──────────────────────────────────
TONIC_PC = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
    'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11, 'Cb': 11,
}

GRID_SIZE = 16

# ── 全局 Token ID 范围 (在 _init_lookups 中初始化) ──
_BAR_ID = 4  # 固定: PAD=0, BOS=1, EOS=2, MASK=3, Bar=4
_POS_MIN = _POS_MAX = 0
_POS_VALUES = None   # tid → int position
_NOTE_ON_MIN = _NOTE_ON_MAX = 0
_NOTE_ON_VALUES = None  # tid → int interval
_DUR_MIN = _DUR_MAX = 0
_DUR_VALUES = None     # tid → int duration value
_VOICE_MIN = _VOICE_MAX = 0
_VOICE_VALUES = None   # tid → int voice index
_TONIC_IDS = None      # set of tonic token IDs
_TONIC_NAME = None     # tid → str tonic name
_PROG_MIN = _PROG_MAX = 0


def _init_lookups():
    """预计算 token ID 范围，加速扫描。"""
    global _POS_MIN, _POS_MAX, _POS_VALUES
    global _NOTE_ON_MIN, _NOTE_ON_MAX, _NOTE_ON_VALUES
    global _DUR_MIN, _DUR_MAX, _DUR_VALUES
    global _VOICE_MIN, _VOICE_MAX, _VOICE_VALUES
    global _TONIC_IDS, _TONIC_NAME
    global _PROG_MIN, _PROG_MAX

    t = REMITokenizer(GRID_SIZE, 8)
    vocab_size = t.vocab_size

    # Position: <Position 0> .. <Position 15>
    _POS_VALUES = [-1] * vocab_size
    pos_ids = [t.encode_token(f'<Position {i}>') for i in range(GRID_SIZE)]
    _POS_MIN, _POS_MAX = min(pos_ids), max(pos_ids)
    for i, tid in enumerate(pos_ids):
        _POS_VALUES[tid] = i

    # Note_ON: <Note_ON -60> .. <Note_ON +60>
    _NOTE_ON_VALUES = [0] * vocab_size
    note_ids = [t.encode_token(f'<Note_ON {i}>') for i in range(-60, 61)]
    _NOTE_ON_MIN, _NOTE_ON_MAX = min(note_ids), max(note_ids)
    for i, tid in enumerate(note_ids):
        _NOTE_ON_VALUES[tid] = i - 60  # interval value

    # Duration: <Duration 1> .. <Duration 16>
    _DUR_VALUES = [-1] * vocab_size
    dur_ids = [t.encode_token(f'<Duration {d}>') for d in range(1, GRID_SIZE + 1)]
    _DUR_MIN, _DUR_MAX = min(dur_ids), max(dur_ids)
    for d, tid in enumerate(dur_ids, 1):
        _DUR_VALUES[tid] = d

    # Voice: <Voice 0> .. <Voice 3>
    _VOICE_VALUES = [-1] * vocab_size
    voice_ids = [t.encode_token(f'<Voice {v}>') for v in t.VOICE_NAMES]
    _VOICE_MIN, _VOICE_MAX = min(voice_ids), max(voice_ids)
    for v, tid in enumerate(voice_ids):
        _VOICE_VALUES[tid] = int(t.VOICE_NAMES[v])

    # Tonic: <Tonic C> .. <Tonic B>
    _TONIC_IDS = set()
    _TONIC_NAME = {}
    for name in t.TONIC_NAMES:
        tid = t.encode_token(f'<Tonic {name}>')
        _TONIC_IDS.add(tid)
        _TONIC_NAME[tid] = name

    # Program: <Program 0> .. <Program 89_3>
    prog_ids = []
    for prog in t.PROGRAM_NAMES:
        prog_ids.append(t.encode_token(f'<Program {prog}>'))
        for sub in range(1, t.MAX_SUBTRACKS):
            prog_ids.append(t.encode_token(f'<Program {prog}_{sub}>'))
    _PROG_MIN, _PROG_MAX = min(prog_ids), max(prog_ids)


# ── 辅助计算 ──────────────────────────────────────────────

def _compute_entropy(values: list[int]) -> float:
    """计算整数值列表的熵 (base 2)。"""
    if not values:
        return 0.0
    counter = Counter(values)
    total = len(values)
    entropy = 0.0
    for count in counter.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def _compute_voice_duration_kl(voice_dur_lists: list[list[int]]) -> float:
    """计算声部间 Duration 分布的 JS 散度 (对称 KL 平均)。

    voice_dur_lists: list of lists, 每个内层列表是一个声部的 Duration 值序列。
    返回最大配对 JS 散度。
    """
    n_buckets = GRID_SIZE
    distributions = []
    for durs in voice_dur_lists:
        if len(durs) < 10:
            continue
        hist = [0] * n_buckets
        for d in durs:
            bucket = min(int(d) - 1, n_buckets - 1)
            hist[bucket] += 1
        total = sum(hist)
        # Laplace 平滑
        dist = [(h + 1) / (total + n_buckets) for h in hist]
        distributions.append(dist)

    if len(distributions) < 2:
        return 0.0

    max_js = 0.0
    for i in range(len(distributions)):
        for j in range(i + 1, len(distributions)):
            # JS = (KL(P||M) + KL(Q||M)) / 2
            m = [(p + q) / 2 for p, q in zip(distributions[i], distributions[j])]
            kl_pm = sum(p * math.log2(p / m_v) for p, m_v in zip(distributions[i], m) if p > 0 and m_v > 0)
            kl_qm = sum(q * math.log2(q / m_v) for q, m_v in zip(distributions[j], m) if q > 0 and m_v > 0)
            js = (kl_pm + kl_qm) / 2
            if js > max_js:
                max_js = js

    return max_js


# ── 指标计算 ──────────────────────────────────────────────

def compute_texture_score(
    pos_note_counts: dict, voice_durs: dict
) -> int:
    """织体复杂度: 1=单旋律/同度, 2=主调, 3=复调。

    pos_note_counts: {pos_index: Note_ON count}
    voice_durs: {voice_index: [duration values]}
    """
    if not pos_note_counts:
        return 1

    avg_density = sum(pos_note_counts.values()) / len(pos_note_counts)
    if avg_density <= 1.5:
        return 1

    # 检查声部间节奏独立性
    active_voices = [durs for v, durs in voice_durs.items() if len(durs) >= 10]
    if len(active_voices) >= 2:
        dur_kl = _compute_voice_duration_kl(active_voices)
        if dur_kl > 0.5:
            return 3

    return 2


def compute_structure_score(n_bars: int) -> int:
    """结构复杂度: 1=超短(1-8), 2=短(9-32), 3=中(33-96), 4=长(97-256), 5=超长(257+)。"""
    if n_bars <= 8:
        return 1
    if n_bars <= 32:
        return 2
    if n_bars <= 96:
        return 3
    if n_bars <= 256:
        return 4
    return 5


def compute_rhythm_score(dur_values: list[int]) -> int:
    """节奏复杂度: 1=简单, 2=中等, 3=复杂。"""
    if not dur_values:
        return 1

    dur_entropy = _compute_entropy(dur_values)
    max_dur = max(dur_values)

    if dur_entropy < 1.5 and max_dur <= 8:
        return 1
    if dur_entropy < 2.5:
        return 2
    return 3


def compute_instrumentation_score(program_ids: set) -> int:
    """乐器复杂度: 1=纯钢琴(≤1 Program), 2=小合奏(2-3), 3=室内乐(4-8), 4=管弦(9+)。"""
    # Map program token IDs to their base program numbers
    # Program tokens: <Program N> and <Program N_S>
    base_programs = set()
    for tid in program_ids:
        if _PROG_MIN <= tid <= _PROG_MAX:
            # We just count unique token IDs as proxy for unique programs
            # A more precise count would parse the token string, but this is sufficient
            base_programs.add(tid)
    n = len(base_programs)
    if n <= 1:
        return 1
    if n <= 3:
        return 2
    if n <= 8:
        return 3
    return 4


def compute_level(texture: int, structure: int, rhythm: int, instr: int) -> int:
    """根据四个指标计算 Difficulty Level (1-5)。

    级别定义 (最严格匹配优先):
      L1: texture≤1, structure≤2, rhythm≤1, instr=1
      L2: texture≤2, structure≤2, rhythm≤1, instr=1
      L3: texture≤2, structure≤3, rhythm≤2, instr≤2
      L4: texture≤3, structure≤3, rhythm≤2, instr≤3
      L5: texture≤3, structure≤5, rhythm≤3, instr≤4
    """
    LEVEL_CONDITIONS = [
        (1, 1, 2, 1, 1),  # (level, texture_max, structure_max, rhythm_max, instr_max)
        (2, 2, 2, 1, 1),
        (3, 2, 3, 2, 2),
        (4, 3, 3, 2, 3),
        (5, 3, 5, 3, 4),
    ]

    for level, t_max, s_max, r_max, i_max in LEVEL_CONDITIONS:
        if texture <= t_max and structure <= s_max and rhythm <= r_max and instr <= i_max:
            return level

    # 兜底: 无法匹配任何条件 → Level 5
    return 5


# ── 文件处理 ──────────────────────────────────────────────

def process_file(fpath: str) -> dict:
    """处理单个 .tokens 文件: F1-F5 过滤 → 四指标分类 → Level 分级。"""

    # 加载 tokens
    try:
        with open(fpath, 'rb') as f:
            ids = json.load(f)
    except Exception as e:
        return {'status': 'error', 'file': fpath, 'reason': str(e)}

    if not isinstance(ids, list) or len(ids) < 5:
        return {
            'status': 'reject', 'file': fpath,
            'filter': 'too_short', 'detail': {'length': len(ids) if isinstance(ids, list) else 0},
        }

    # ── 扫描 tokens, 收集所有需要的统计量 ──
    # F1: 调性清晰度
    pc_counts = [0] * 12
    tonic_pc = 0          # 当前主音 pitch class
    current_tonic_name = 'C'

    # F2: 调性稳定性
    prev_tonic_name = None
    tonic_changes = 0

    # F3: 结构合理性
    n_bars = 0
    n_notes = 0
    n_dur_events = 0
    n_pos = 0

    # F5: Duration 越界
    cum_dur = 0
    overflows = 0
    total_durations = 0

    # Texture: 织体复杂度
    pos_note_counts = defaultdict(int)   # pos → Note_ON count
    voice_durs = defaultdict(list)       # voice → [duration values]
    current_pos = -1
    current_voice = -1

    # Rhythm: 节奏复杂度
    dur_values = []

    # Instr: 乐器复杂度
    program_ids = set()

    # 扫描
    for tid in ids:
        # ── Bar ──
        if tid == _BAR_ID:
            n_bars += 1
            cum_dur = 0

        # ── Position ──
        elif _POS_MIN <= tid <= _POS_MAX:
            n_pos += 1
            current_pos = _POS_VALUES[tid]

        # ── Note_ON ──
        elif _NOTE_ON_MIN <= tid <= _NOTE_ON_MAX:
            n_notes += 1
            interval = _NOTE_ON_VALUES[tid]
            abs_pc = (tonic_pc + interval) % 12
            pc_counts[abs_pc] += 1
            if current_pos >= 0:
                pos_note_counts[current_pos] += 1

        # ── Duration ──
        elif _DUR_MIN <= tid <= _DUR_MAX:
            n_dur_events += 1
            dur_val = _DUR_VALUES[tid]
            total_durations += 1
            if cum_dur + dur_val > GRID_SIZE + 2:
                overflows += 1
            cum_dur += dur_val
            dur_values.append(dur_val)
            if current_voice >= 0:
                voice_durs[current_voice].append(dur_val)

        # ── Voice ──
        elif _VOICE_MIN <= tid <= _VOICE_MAX:
            current_voice = _VOICE_VALUES[tid]

        # ── Tonic ──
        elif tid in _TONIC_IDS:
            tonic_name = _TONIC_NAME[tid]
            tonic_pc = TONIC_PC.get(tonic_name, 0)
            current_tonic_name = tonic_name
            if prev_tonic_name is not None and tonic_name != prev_tonic_name:
                tonic_changes += 1
            prev_tonic_name = tonic_name

        # ── Program ──
        elif _PROG_MIN <= tid <= _PROG_MAX:
            program_ids.add(tid)

    # ═══════════════════════════════════════════════════════
    # F1: 调性清晰度
    # ═══════════════════════════════════════════════════════
    total_pc = sum(pc_counts)
    if total_pc < 50:
        return {
            'status': 'reject', 'file': fpath,
            'filter': 'tonality_clarity',
            'detail': {'peakiness': 0.0, 'note_count': total_pc},
        }

    probs = [c / total_pc for c in pc_counts]
    peakiness = max(probs) / (sum(probs) / 12)
    if peakiness < 1.3:
        return {
            'status': 'reject', 'file': fpath,
            'filter': 'tonality_clarity',
            'detail': {'peakiness': round(peakiness, 4), 'tonic': current_tonic_name},
        }

    # ═══════════════════════════════════════════════════════
    # F2: 调性稳定性
    # ═══════════════════════════════════════════════════════
    if n_bars >= 4:
        change_rate = tonic_changes / n_bars
        if change_rate > 0.5:
            return {
                'status': 'reject', 'file': fpath,
                'filter': 'tonality_stability',
                'detail': {'change_rate': round(change_rate, 4), 'tonic_changes': tonic_changes, 'n_bars': n_bars},
            }

    # ═══════════════════════════════════════════════════════
    # F3: 结构合理性
    # ═══════════════════════════════════════════════════════
    flags = []
    if n_notes == 0:
        flags.append('no_notes')
    if n_notes > 0 and abs(n_notes - n_dur_events) / max(n_notes, 1) > 0.3:
        flags.append('note_dur_mismatch')
    if n_bars > 8 and n_pos / n_bars < 2:
        flags.append('sparse_bars')
    if n_bars > 0 and n_pos / n_bars > 80:
        flags.append('dense_bars')
    if n_bars == 0:
        flags.append('no_bars')

    if flags:
        return {
            'status': 'reject', 'file': fpath,
            'filter': 'structural_sanity',
            'detail': {
                'flags': flags,
                'n_notes': n_notes, 'n_dur_events': n_dur_events,
                'n_bars': n_bars, 'n_pos': n_pos,
            },
        }

    # ═══════════════════════════════════════════════════════
    # F4: Token 长度
    # ═══════════════════════════════════════════════════════
    n_tokens = len(ids)
    if n_tokens < 50:
        return {
            'status': 'reject', 'file': fpath,
            'filter': 'too_short',
            'detail': {'length': n_tokens},
        }
    if n_tokens > 16384:
        return {
            'status': 'reject', 'file': fpath,
            'filter': 'too_long',
            'detail': {'length': n_tokens},
        }

    # ═══════════════════════════════════════════════════════
    # F5: Duration 越界率
    # ═══════════════════════════════════════════════════════
    if total_durations > 0:
        overflow_rate = overflows / total_durations
        if overflow_rate > 0.05:
            return {
                'status': 'reject', 'file': fpath,
                'filter': 'duration_overflow',
                'detail': {
                    'overflow_rate': round(overflow_rate, 4),
                    'overflows': overflows,
                    'total_durations': total_durations,
                },
            }

    # ═══════════════════════════════════════════════════════
    # 四指标自动分类
    # ═══════════════════════════════════════════════════════
    texture = compute_texture_score(pos_note_counts, voice_durs)
    structure = compute_structure_score(n_bars)
    rhythm = compute_rhythm_score(dur_values)
    instr = compute_instrumentation_score(program_ids)
    level = compute_level(texture, structure, rhythm, instr)

    return {
        'status': 'pass',
        'file': fpath,
        'metrics': {
            'texture': texture,
            'structure': structure,
            'rhythm': rhythm,
            'instr': instr,
            'level': level,
            'n_bars': n_bars,
            'n_notes': n_notes,
            'n_tokens': n_tokens,
            'peakiness': round(peakiness, 4),
        },
    }


# ── CLI: classify ─────────────────────────────────────────

def cmd_classify(args):
    """主分类命令: 扫描全部 .tokens 文件, 输出 complexity_labels.json。"""
    _init_lookups()

    # 收集所有 .tokens 文件
    all_files = []
    for root, dirs, files in os.walk(args.input_dir):
        for f in files:
            if f.endswith('.tokens'):
                all_files.append(os.path.join(root, f))

    print(f"输入目录: {args.input_dir}")
    print(f"文件总数: {len(all_files)}")
    print(f"工作进程: {args.num_workers}")
    print()

    tasks = all_files  # process_file 只接受 fpath
    stats = Counter()
    file_metrics = {}    # fpath → metrics dict
    rejected = {}        # fpath → rejection info
    start = time.time()

    if args.dry_run:
        # 干跑: 单进程采样
        sample = all_files[:100] if len(all_files) > 100 else all_files
        for i, fpath in enumerate(sample):
            result = process_file(fpath)
            stats[result['status']] += 1
            if i < 5:
                print(f"  [{result['status']}] {os.path.basename(fpath)}: {json.dumps(result.get('metrics', result.get('detail', {})))}")
        print(f"\n  干跑完成: {dict(stats)}")
        return

    with mp.Pool(args.num_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(process_file, tasks, chunksize=200)):
            status = result['status']
            stats[status] += 1
            fpath = result['file']

            if status == 'pass':
                file_metrics[fpath] = result['metrics']
            elif status == 'reject':
                rejected[fpath] = {
                    'filter': result['filter'],
                    'detail': result['detail'],
                }

            # 进度报告
            processed = i + 1
            if processed % 100000 == 0:
                elapsed = time.time() - start
                rate = processed / elapsed
                print(f"  进度: {processed}/{len(all_files)} "
                      f"({100 * processed / len(all_files):.0f}%), "
                      f"{elapsed:.0f}s, {rate:.0f} files/s, "
                      f"pass={stats['pass']}, reject={stats['reject']}, error={stats['error']}")

    elapsed = time.time() - start

    # ── 构建输出 ──
    # 用相对路径 (相对于 input_dir) 存储
    rel_metrics = {}
    rel_rejected = {}
    for fpath, m in file_metrics.items():
        rel = os.path.relpath(fpath, args.input_dir) if fpath.startswith(args.input_dir) else fpath
        rel_metrics[rel] = m
    for fpath, r in rejected.items():
        rel = os.path.relpath(fpath, args.input_dir) if fpath.startswith(args.input_dir) else fpath
        rel_rejected[rel] = r

    # 统计分布
    level_counts = Counter(m['level'] for m in file_metrics.values())
    texture_dist = Counter(m['texture'] for m in file_metrics.values())
    structure_dist = Counter(m['structure'] for m in file_metrics.values())
    rhythm_dist = Counter(m['rhythm'] for m in file_metrics.values())
    instr_dist = Counter(m['instr'] for m in file_metrics.values())
    reject_reasons = Counter(r['filter'] for r in rejected.values())

    output = {
        'version': 'v0.3.1-data2',
        'input_dir': args.input_dir,
        'thresholds': {
            'texture': {'1': '单旋律/同度', '2': '主调', '3': '复调'},
            'structure': {'1': '1-8 bar', '2': '9-32 bar', '3': '33-96 bar', '4': '97-256 bar', '5': '256+ bar'},
            'rhythm': {'1': '简单', '2': '中等', '3': '复杂'},
            'instr': {'1': '纯钢琴(≤1)', '2': '小合奏(2-3)', '3': '室内乐(4-8)', '4': '管弦(9+)'},
        },
        'file_metrics': rel_metrics,
        'rejected': {
            'count': len(rejected),
            'reasons': dict(reject_reasons),
            'files': rel_rejected,
        },
        'passed_count': len(file_metrics),
        'total_files': len(all_files),
        'level_counts': {
            '1': level_counts.get(1, 0),
            '2': level_counts.get(2, 0),
            '3': level_counts.get(3, 0),
            '4': level_counts.get(4, 0),
            '5': level_counts.get(5, 0),
        },
        'distribution': {
            'texture': {str(k): v for k, v in sorted(texture_dist.items())},
            'structure': {str(k): v for k, v in sorted(structure_dist.items())},
            'rhythm': {str(k): v for k, v in sorted(rhythm_dist.items())},
            'instr': {str(k): v for k, v in sorted(instr_dist.items())},
        },
    }

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, ensure_ascii=False)

    # ── 打印汇总 ──
    print(f"\n{'='*60}")
    print(f"分类完成! 总耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  总文件:   {len(all_files):>10,}")
    print(f"  通过:     {len(file_metrics):>10,} ({100*len(file_metrics)/len(all_files):.1f}%)")
    print(f"  拒绝:     {len(rejected):>10,} ({100*len(rejected)/len(all_files):.1f}%)")
    print(f"  错误:     {stats['error']:>10,}")
    print(f"\n过滤原因分布:")
    for reason, count in reject_reasons.most_common():
        print(f"  {reason:<25s}: {count:>8,}")
    print(f"\nLevel 分布:")
    for lv in range(1, 6):
        print(f"  Level {lv}: {level_counts.get(lv, 0):>10,} ({100*level_counts.get(lv,0)/max(len(file_metrics),1):.1f}%)")
    print(f"\n输出: {args.output}")


# ── CLI: split ────────────────────────────────────────────

def cmd_split(args):
    """从 complexity_labels.json 生成 train_L1~L5.txt 和 val_L1~L5.txt。"""
    random.seed(args.seed)

    with open(args.labels) as f:
        data = json.load(f)

    file_metrics = data.get('file_metrics', {})
    if not file_metrics:
        print("错误: labels 文件中没有 file_metrics")
        sys.exit(1)

    # 按 Level 分组
    by_level = defaultdict(list)
    for fpath, metrics in file_metrics.items():
        level = metrics.get('level', 5)
        by_level[level].append(fpath)

    print(f"标签文件: {args.labels}")
    print(f"总通过文件: {len(file_metrics)}")
    print(f"训练比例: {args.train_ratio}")
    for lv in range(1, 6):
        print(f"  Level {lv}: {len(by_level[lv]):,} 文件")

    # 生成 train/val 拆分
    os.makedirs(args.output_dir, exist_ok=True)
    train_counts = {}
    val_counts = {}

    for lv in range(1, 6):
        files = by_level[lv]
        random.shuffle(files)
        n_train = max(1, int(len(files) * args.train_ratio))
        train_files = sorted(files[:n_train])
        val_files = sorted(files[n_train:])

        train_path = os.path.join(args.output_dir, f'train_L{lv}.txt')
        val_path = os.path.join(args.output_dir, f'val_L{lv}.txt')

        with open(train_path, 'w') as f:
            f.write('\n'.join(train_files) + ('\n' if train_files else ''))
        with open(val_path, 'w') as f:
            f.write('\n'.join(val_files) + ('\n' if val_files else ''))

        train_counts[lv] = len(train_files)
        val_counts[lv] = len(val_files)

    # 汇总
    print(f"\n生成文件:")
    total_train = 0
    total_val = 0
    for lv in range(1, 6):
        print(f"  train_L{lv}.txt: {train_counts[lv]:>10,}   val_L{lv}.txt: {val_counts[lv]:>10,}")
        total_train += train_counts[lv]
        total_val += val_counts[lv]
    print(f"  {'总计':>12s}: {total_train:>10,}           : {total_val:>10,}")
    print(f"\n输出目录: {args.output_dir}")


# ── 入口 ──────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Chopinote-AI v0.3.1-data2: 数据质量过滤 + 自动分类 + 训练集拆分',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/classify_complexity.py classify \\
      --input-dir /root/autodl-tmp/data/processed/tokens_v4 \\
      --output complexity_labels.json --num-workers 16

  python scripts/classify_complexity.py split \\
      --labels complexity_labels.json \\
      --output-dir /root/autodl-tmp/data/processed/ --train-ratio 0.9
        """,
    )
    sub = ap.add_subparsers(dest='command', help='子命令')

    # classify
    classify_p = sub.add_parser('classify', help='扫描全部 tokens 文件, 过滤并分类')
    classify_p.add_argument('--input-dir', required=True,
                            help='tokens 文件根目录')
    classify_p.add_argument('--output', required=True,
                            help='输出 JSON 路径 (complexity_labels.json)')
    classify_p.add_argument('--num-workers', type=int, default=16,
                            help='并行进程数 (默认 16)')
    classify_p.add_argument('--dry-run', action='store_true',
                            help='仅检查前 100 个文件')

    # split
    split_p = sub.add_parser('split', help='从 labels JSON 生成 train_L*.txt / val_L*.txt')
    split_p.add_argument('--labels', required=True,
                         help='complexity_labels.json 路径')
    split_p.add_argument('--output-dir', required=True,
                         help='输出目录 (train_L1.txt ~ val_L5.txt)')
    split_p.add_argument('--train-ratio', type=float, default=0.9,
                         help='训练集比例 (默认 0.9)')
    split_p.add_argument('--seed', type=int, default=42,
                         help='随机种子 (默认 42)')

    args = ap.parse_args()

    if args.command == 'classify':
        cmd_classify(args)
    elif args.command == 'split':
        cmd_split(args)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
