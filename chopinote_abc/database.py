"""A1/A2/A3 数据库 — ABC Engine v2 的持久化状态层。

设计原则：
- A1/A3 是有状态数据库，B 和 C 可增删改查
- A2 是检索层，通过 A3 的 bar_log 选地标 + 提纯
- Phase 1 全规则驱动，零模型依赖
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════════════
#  A1DB — 框架数据库
# ═══════════════════════════════════════════════════════════════

@dataclass
class SectionPlan:
    """单个段落的规划。"""
    type: str                          # 'theme1' / 'development' / 'recapitulation' / ...
    bars: int                          # 本段小节数
    key: str                           # 调性名，如 'C' / 'G' / 'Am'
    cadence: str                       # 预期终止式: 'PAC' / 'IAC' / 'HC' / 'DC'
    start_bar: int = 0                 # 本段起始全局 bar 号（A1DB 计算）
    innovation_budget: float = 0.1     # B 的创新预算比例
    development_ops: list[str] | None = None
        # B 的发展配方，如 ['invert'] / ['fragment'] / ['diminish']
        # None / [] = 不需要发展处理
    temperature_zone: tuple[float, float, float] | None = None
        # (冷区系数, 热区系数, 冷区系数)，如 (0.8, 1.3, 0.7)
        # None = B 根据段落类型自动推断


@dataclass
class ChordAtBar:
    """单个 bar 的和声 — 仅在和弦变更时记录。"""
    bar: int           # 全局 bar 号
    func: str          # 'I' / 'IV' / 'V' / 'vi' / ... （含斜线和弦如 'V/V'）
    inv: str = 'root'  # 'root' / '1st' / '2nd' / '3rd'


@dataclass
class SeedContext:
    """seed 的结构快照 — 段 1 生成时 B 可查询。"""
    final_chord: str | None = None  # seed 最后一个和弦功能 (Phase 3 启用)
    final_key: str | None = None    # seed 最后一个已知调性
    bar_count: int = 0              # seed 包含的小节数


@dataclass
class StructuralFix:
    """C→A1 写回的修复指令。"""
    type: str                      # 'extend_section' / 'tighten_recap' / 'add_cadence' / ...
    section: int | None = None     # section_idx
    bar: int | None = None         # 全局 bar 号
    target_bars: int | None = None
    target_similarity: float | None = None
    cadence: str | None = None
    reference_label: str | None = None  # A2 label


@dataclass
class A1DB:
    """A1 框架库 — 结构 + 和声的全局导航地图。"""

    # ── 初始值（Stage 1/2 写入，生成前确定）──
    sections: list[SectionPlan] = field(default_factory=list)
    harmony: list[ChordAtBar] = field(default_factory=list)

    # ── 运行时修改（B/C 动态写入）──
    overrides: dict[int, ChordAtBar] = field(default_factory=dict)
    cadence_markers: dict[int, str] = field(default_factory=dict)
    section_adjustments: dict[int, int] = field(default_factory=dict)

    # ── seed 上下文 ──
    seed_context: SeedContext | None = None

    # ── 内部状态 ──
    _section_starts: dict[int, int] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._reindex()

    def _reindex(self):
        """重算每个 section 的全局 start_bar。"""
        bar = 0
        self._section_starts = {}
        for i, sec in enumerate(self.sections):
            self._section_starts[i] = bar
            sec.start_bar = bar
            bar += sec.bars

    @property
    def total_bars(self) -> int:
        if not self.sections:
            return 0
        last = self.sections[-1]
        return last.start_bar + last.bars

    # ── CRUD ────────────────────────────────────

    def get_section(self, bar: int) -> SectionPlan | None:
        """查询 bar 所属段落。"""
        for i, sec in enumerate(self.sections):
            if sec.start_bar <= bar < sec.start_bar + sec.bars:
                return sec
        return None

    def find_section(self, type_prefix: str) -> int | None:
        """查找第一个 type 以给定前缀开头的段落 → section_idx。"""
        for i, sec in enumerate(self.sections):
            if sec.type.startswith(type_prefix):
                return i
        return None

    def has_section_type(self, type_prefix: str) -> bool:
        return self.find_section(type_prefix) is not None

    def get_chord(self, bar: int) -> ChordAtBar | None:
        """查询 bar 的和声（优先读 overrides）。"""
        if bar in self.overrides:
            return self.overrides[bar]
        for c in self.harmony:
            if c.bar == bar:
                return c
        return None

    def override_harmony(self, from_bar: int, new_chords: list[ChordAtBar]):
        """B 改写指定范围的 harmony（和声回退用）。"""
        for c in new_chords:
            self.overrides[c.bar] = c

    def adjust_section_bars(self, section_idx: int, delta: int):
        """调整某段的小节数。"""
        self.section_adjustments[section_idx] = (
            self.section_adjustments.get(section_idx, 0) + delta)

    def insert_cadence(self, bar: int, cadence_type: str):
        """C 补终止式标记。"""
        self.cadence_markers[bar] = cadence_type

    def apply_fix(self, fix: StructuralFix):
        """C 复盘后写回（统一入口）。"""
        if fix.type == 'extend_section' and fix.section is not None:
            delta = (fix.target_bars or 0) - self.sections[fix.section].bars
            self.adjust_section_bars(fix.section, delta)
        elif fix.type == 'add_cadence' and fix.bar is not None:
            self.insert_cadence(fix.bar, fix.cadence or 'PAC')
        elif fix.type == 'tighten_recap' and fix.section is not None:
            sec = self.sections[fix.section]
            sec.innovation_budget = max(0.0, sec.innovation_budget - 0.15)
            sec.temperature_zone = (0.7, 1.0, 0.6)  # 收紧温区

    def reset_overrides(self):
        """retry 前清理运行时修改（保留 structural_fix 的持久修改）。"""
        self.overrides.clear()
        self._reindex()

    # ── Prefix 序列化 ──────────────────────────

    def build_structure_tokens(self, tokenizer) -> list[int]:
        """序列化全曲 section_plan → token 序列，供 prefix ①。"""
        tokens = []
        for sec in self.sections:
            type_token = f'<Section {sec.type}>'
            tid = tokenizer.encode_token(type_token)
            tokens.append(tid)
            key_token = f'<Key {sec.key}>'
            kid = tokenizer.encode_token(key_token)
            tokens.append(kid)
        # <SecSum> 收尾
        secsum_tid = tokenizer.encode_token('<SecSum>')
        tokens.append(secsum_tid)
        return tokens

    def build_harmony_tokens(self, tokenizer) -> list[int]:
        """序列化全曲 harmony（含 overrides）→ token 序列，供 prefix ②。"""
        tokens = []
        combined: dict[int, ChordAtBar] = {}
        for c in self.harmony:
            combined[c.bar] = c
        combined.update(self.overrides)

        last_func = None
        last_inv = None
        for bar in sorted(combined.keys()):
            c = combined[bar]
            # 去重：和弦功能+转位都没变就跳过
            if c.func == last_func and c.inv == last_inv:
                continue
            last_func = c.func
            last_inv = c.inv

            bar_tid = tokenizer.encode_token(tokenizer.BAR)
            tokens.append(bar_tid)
            chord_tid = tokenizer.encode_token(f'<Chord {c.func}>')
            tokens.append(chord_tid)
            inv_label = 'Root' if c.inv == 'root' else c.inv
            inv_tid = tokenizer.encode_token(f'<Inv {inv_label}>')
            tokens.append(inv_tid)

        return tokens


# ═══════════════════════════════════════════════════════════════
#  A3DB — 统计数据库
# ═══════════════════════════════════════════════════════════════

@dataclass
class BarStats:
    """单个 bar 的统计画像。"""
    bar: int
    density: float = 0.0               # notes/bar
    pitch_range: tuple[int, int] = (0, 0)
    velocity_mean: float = 0.0
    rest_ratio: float = 0.0
    harmonic_rhythm: float = 0.0       # tokens per chord change
    pitch_class_dist: list[float] = field(default_factory=lambda: [0.0] * 12)
    token_type_dist: list[float] = field(default_factory=lambda: [])
    note_count: int = 0
    rest_count: int = 0
    chord_change: bool = False
    b1_score: float | None = None
    b2_score: float | None = None
    innovation_meta: dict | None = None  # B 记录创新信息（可选）
    # ── DurSat 时值饱和度字段 ──
    total_duration: float = 0.0           # 本小节累计时长 (四个声部 max)
    bar_fill_ratio: float = 0.0           # total_duration / grid_size
    duration_overflow: bool = False       # 溢出标志


@dataclass
class SectionStats:
    """段级聚合统计 — 与 BarStats 同一套指标，不同窗口。"""
    section_idx: int
    density: float = 0.0
    density_std: float = 0.0
    pitch_range: tuple[int, int] = (0, 0)
    velocity_mean: float = 0.0
    velocity_dist: list[float] = field(default_factory=lambda: [0.0] * 8)
    rest_ratio: float = 0.0
    harmonic_rhythm: float = 0.0
    pitch_class_dist: list[float] = field(default_factory=lambda: [0.0] * 12)
    token_type_dist: list[float] = field(default_factory=lambda: [])
    cadence_type: str = ''


@dataclass
class A3DB:
    """A3 统计库 — 每 bar Append-Only + 段快照 + 全局累积。"""

    bar_log: list[BarStats] = field(default_factory=list)
    section_snapshots: dict[int, SectionStats] = field(default_factory=dict)
    baselines: dict[str, float | list[float]] = field(default_factory=dict)
    innovation_log: list[dict] = field(default_factory=list)

    # ── 写 ──────────────────────────────────────

    def record_bar(self, bar: int, tokens: list[int], tokenizer,
                   b1_score: float | None = None,
                   b2_score: float | None = None):
        """每 bar 完成时调用。算 stats → append bar_log。"""
        stats = BarStats(bar=bar, b1_score=b1_score, b2_score=b2_score)

        note_pitches = []
        rest_count = 0
        note_count = 0
        velocity_sum = 0.0
        velocity_count = 0
        pc_counts = [0] * 12

        _PREFIX_NOTE = tokenizer.NOTE_ON
        _PREFIX_VEL = tokenizer.VELOCITY
        _PREFIX_REST = tokenizer.REST

        for tid in tokens:
            ts = tokenizer.decode_token(tid)
            if ts.startswith(_PREFIX_NOTE):
                note_count += 1
                try:
                    pitch = int(ts.split(' ')[1])
                except (IndexError, ValueError):
                    pitch = 0
                note_pitches.append(pitch)
                pc_counts[(pitch + 12) % 12] += 1
            elif ts.startswith(_PREFIX_REST):
                rest_count += 1
            elif ts.startswith(_PREFIX_VEL):
                try:
                    velocity_sum += int(ts.split(' ')[1])
                    velocity_count += 1
                except (IndexError, ValueError):
                    pass

        total = note_count + rest_count
        stats.density = note_count
        stats.note_count = note_count
        stats.rest_count = rest_count
        stats.rest_ratio = rest_count / max(1, total)
        stats.harmonic_rhythm = 0.0  # v0.3.0: chord tokens removed, computed via SSF instead
        stats.velocity_mean = velocity_sum / max(1, velocity_count)
        stats.chord_change = False
        stats.pitch_class_dist = [c / max(1, sum(pc_counts)) for c in pc_counts]

        if note_pitches:
            stats.pitch_range = (min(note_pitches), max(note_pitches))

        # ── DurSat: 时值饱和度统计 ────────────────
        _VOICE_PREFIX = tokenizer.VOICE
        _DUR_PREFIX = tokenizer.DURATION
        _BAR_STR = tokenizer.BAR
        grid_size = tokenizer.grid_size
        cum_dur = [0, 0, 0, 0]
        current_voice = 0
        overflows = 0
        for tid in tokens:
            ts = tokenizer.decode_token(tid)
            if ts == _BAR_STR:
                cum_dur = [0, 0, 0, 0]
            elif ts.startswith(_VOICE_PREFIX):
                try:
                    current_voice = int(ts.split(' ')[1].rstrip('>'))
                except (IndexError, ValueError):
                    pass
            elif ts.startswith(_DUR_PREFIX):
                try:
                    dur_val = int(ts.split(' ')[1].rstrip('>'))
                except (IndexError, ValueError):
                    dur_val = 0
                cum_dur[current_voice] += dur_val
                if cum_dur[current_voice] > grid_size + 2:
                    overflows += 1
        stats.total_duration = float(max(cum_dur))
        stats.bar_fill_ratio = stats.total_duration / grid_size if grid_size > 0 else 0.0
        stats.duration_overflow = overflows > 0

        self.bar_log.append(stats)

    def record_innovation(self, bar: int, meta: dict):
        """B 判定某 bar 为有意的创新 → 记入创新日志，供 C 复查。"""
        self.innovation_log.append({'bar': bar, **meta})

    def snapshot_section(self, section_idx: int, tokens: list[int],
                         tokenizer, A1: A1DB | None = None):
        """段结束时调用。聚合 bar_log 中该段的记录 → 写入 snapshots。"""
        # 计算段落在 bar_log 中的实际 bar 范围（含 seed 偏移）
        seed_bars = A1.seed_context.bar_count if (A1 and A1.seed_context) else 0
        start_bar = seed_bars
        for k in range(section_idx):
            start_bar += A1.sections[k].bars
        end_bar = start_bar + (A1.sections[section_idx].bars if A1 else 0)

        bars = [b for b in self.bar_log if start_bar <= b.bar < end_bar]
        if not bars:
            return

        n = len(bars)
        densities = [b.density for b in bars]
        sec_stats = SectionStats(section_idx=section_idx)
        sec_stats.density = sum(densities) / n
        sec_stats.density_std = (
            math.sqrt(sum((d - sec_stats.density) ** 2 for d in densities) / n)
            if n > 1 else 0.0)
        sec_stats.rest_ratio = sum(b.rest_ratio for b in bars) / n
        sec_stats.velocity_mean = sum(b.velocity_mean for b in bars) / n
        sec_stats.harmonic_rhythm = sum(b.harmonic_rhythm for b in bars) / n

        # PC dist 平均
        if bars[0].pitch_class_dist:
            dim = len(bars[0].pitch_class_dist)
            sec_stats.pitch_class_dist = [
                sum(b.pitch_class_dist[i] for b in bars) / n
                for i in range(dim)
            ]

        self.section_snapshots[section_idx] = sec_stats

    def set_baseline(self, seed_tokens: list[int], tokenizer):
        """从 seed 提取基线指标（密度/音域/力度/休止/和声/PC 分布）。"""
        # 用临时 BarStats 收集 seed 统计，不污染 bar_log
        tmp = BarStats(bar=0)
        n_notes = 0
        pitches = []
        for t in seed_tokens:
            s = tokenizer.decode_token(t)
            if s.startswith(tokenizer.NOTE_ON):
                n_notes += 1
                try:
                    pitches.append(int(s[len(tokenizer.NOTE_ON) + 1:-1]))
                except ValueError:
                    pass
            elif s.startswith(tokenizer.REST):
                tmp.rest_ratio += 1
            elif s.startswith('<Velocity'):
                try:
                    tmp.velocity_mean += int(s[len('<Velocity') + 1:-1])
                except ValueError:
                    pass

        bar_count = max(1, sum(1 for t in seed_tokens if t == tokenizer.bar_token_id))
        tmp.density = n_notes / bar_count
        tmp.rest_ratio = tmp.rest_ratio / max(1, n_notes + tmp.rest_ratio)
        tmp.velocity_mean = tmp.velocity_mean / max(1, n_notes)
        if pitches:
            tmp.pitch_range = (min(pitches), max(pitches))
        tmp.pitch_class_dist = [0.0] * 12
        for p in pitches:
            tmp.pitch_class_dist[p % 12] += 1.0
        total = sum(tmp.pitch_class_dist)
        if total > 0:
            tmp.pitch_class_dist = [c / total for c in tmp.pitch_class_dist]
        tmp.harmonic_rhythm = 0.0  # seed 级不计算和声节奏

        self.baselines = {
            'pitch_class_dist': tmp.pitch_class_dist,
            'density': tmp.density,
            'rest_ratio': tmp.rest_ratio,
            'velocity_mean': tmp.velocity_mean,
            'harmonic_rhythm': tmp.harmonic_rhythm,
        }

    # ── 读 ──────────────────────────────────────

    def get_window(self, bar: int, n_bars: int) -> list[BarStats]:
        """B1 用：拿 [bar-n_bars+1, bar] 范围的统计。"""
        start = max(0, bar - n_bars + 1)
        return [b for b in self.bar_log if start <= b.bar <= bar]

    def get_trend(self, metric: str, window: int = 8) -> float:
        """B2 用：某指标的最近 window bar 斜率。"""
        recent = self.bar_log[-window:] if len(self.bar_log) >= window else self.bar_log
        if len(recent) < 2:
            return 0.0
        values = [getattr(b, metric, 0.0) for b in recent if hasattr(b, metric)]
        if len(values) < 2:
            return 0.0
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n
        num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        den = sum((i - x_mean) ** 2 for i in range(n))
        return num / max(1e-8, den)

    def get_cumulative(self) -> SectionStats:
        """C 用：从 bar 0 到当前的全曲统计画像。"""
        if not self.bar_log:
            return SectionStats(section_idx=-1)
        dummy = A3DB()
        dummy.bar_log = self.bar_log
        dummy.snapshot_section(-1, [], None)
        return dummy.section_snapshots.get(-1, SectionStats(section_idx=-1))

    def get_last_bar(self) -> BarStats:
        """B 用：刚写完的那个 bar 的完整统计。"""
        return self.bar_log[-1] if self.bar_log else BarStats(bar=-1)

    def compare_sections(self, a: int, b: int) -> dict[str, float]:
        """B2/C 用：两段统计对比（消费 section_snapshots）。"""
        snap_a = self.section_snapshots.get(a)
        snap_b = self.section_snapshots.get(b)
        if not snap_a or not snap_b:
            return {}

        results = {}
        for attr in ('density', 'rest_ratio', 'velocity_mean', 'harmonic_rhythm'):
            va = getattr(snap_a, attr, 0.0)
            vb = getattr(snap_b, attr, 0.0)
            if abs(va) + abs(vb) > 0:
                results[attr] = 1.0 - abs(va - vb) / max(abs(va) + abs(vb), 1e-8)

        # PC dist similarity (cosine)
        pa = snap_a.pitch_class_dist
        pb = snap_b.pitch_class_dist
        if pa and pb and len(pa) == len(pb):
            dot = sum(x * y for x, y in zip(pa, pb))
            na = math.sqrt(sum(x * x for x in pa))
            nb = math.sqrt(sum(y * y for y in pb))
            results['pitch_class_dist'] = dot / max(1e-8, na * nb)

        return results

    def compare_to_baseline(self, section_idx: int) -> dict[str, float]:
        """当前段 vs baseline 的偏离度。"""
        snap = self.section_snapshots.get(section_idx)
        if not snap or not self.baselines:
            return {}
        results = {}
        for metric in ('density', 'rest_ratio', 'velocity_mean'):
            val = getattr(snap, metric, 0.0)
            base = self.baselines.get(metric, 0.0)
            if isinstance(base, (int, float)) and abs(val) + abs(base) > 0:
                results[metric] = abs(val - base) / max(abs(val) + abs(base), 1e-8)
        return results

# ═══════════════════════════════════════════════════════════════
#  Phase 2: C 新颖性评估 + DPO reward log
# ═══════════════════════════════════════════════════════════════


def compute_novelty_bonus(a3, legality_violation_rate: float = 0.0) -> float:
    """C 新颖性加分：基于 A3 创新日志 + 合法性率。

    核心公式: novelty = mean_surprise × (1 - violation_rate)

    高处 surprise × 低违规 = 好创新（奖励）
    高处 surprise × 高违规 = 坏噪声（不奖励）
    """
    if not a3.innovation_log:
        return 0.0

    surprises = [entry.get('surprise', 0.0)
                 for entry in a3.innovation_log
                 if isinstance(entry.get('surprise'), (int, float))]
    if not surprises:
        return 0.0

    mean_surprise = sum(surprises) / len(surprises)
    coherence_factor = max(0.0, 1.0 - legality_violation_rate)
    return mean_surprise * coherence_factor


def compute_diversity_bonus(current_tokens: list[int],
                            previous_generations: list[list[int]],
                            tokenizer) -> float:
    """多样性奖励：当前生成 vs 之前同 seed 生成的 token type 分布差异。

    - 和之前一样 → 0（模式坍缩）→ 扣分
    - 和之前不同 → 高 → 加分
    """
    if not previous_generations:
        return 0.3  # 无历史数据时返回中性值

    # 简化实现：用 token type 分布做 KL
    from .motif import _kl_divergence

    def _token_type_dist(tokens: list[int]) -> list[float]:
        counts = [0] * 20  # 20 token type bins
        for t in tokens:
            ts = tokenizer.decode_token(t)
            if not ts:
                continue
            # 取 token type 前两个字符做 bin
            bin_idx = min(19, abs(hash(ts.split(' ')[0])) % 20)
            counts[bin_idx] += 1
        total = sum(counts) or 1
        return [c / total for c in counts]

    cur_dist = _token_type_dist(current_tokens)

    # 与所有历史生成的平均分布比较
    prev_agg = [0.0] * 20
    for prev_tokens in previous_generations:
        pd = _token_type_dist(prev_tokens)
        for i in range(20):
            prev_agg[i] += pd[i]
    n = len(previous_generations)
    mean_prev = [v / n for v in prev_agg]

    return _kl_divergence(cur_dist, mean_prev)


def write_reward_log(output_path: str, report, novelty: float, diversity: float,
                     seed_path: str = '', musicxml_path: str = '', total_score: float = 0.0,
                     seed_bars: int = 0):
    """将 C 复盘结果 + 新颖性/多样性写入 JSONL 奖励日志。

    格式与 scripts/train/dpo_train.py 的预期输入兼容。
    report 可以是 dict 或 EvalReport dataclass。
    """
    import json
    import os
    from datetime import datetime

    # 兼容 dict 和 dataclass
    if hasattr(report, 'get'):
        fixes = report.get('structural_fixes', [])
        archive_count = report.get('archive_commands_count',
                                    len(report.get('archive_commands', [])))
        total_bars = report.get('total_bars_generated', 0)
    else:
        fixes = getattr(report, 'structural_fixes', [])
        archive_count = len(getattr(report, 'archive_commands', []))
        total_bars = getattr(report, 'total_bars_generated', 0)
        if total_score == 0.0:
            total_score = getattr(report, 'total_score', 0.0)

    entry = {
        'timestamp': datetime.now().isoformat(),
        'total_score': total_score,
        'structural_fixes_count': len(fixes),
        'archive_commands_count': archive_count,
        'total_bars': total_bars,
        'novelty_bonus': novelty,
        'diversity_bonus': diversity,
        'seed_info': {'path': seed_path, 'seed_bars': seed_bars},
        'musicxml_path': musicxml_path,
        'structural_fixes': [
            {'type': f.type, 'section': f.section, 'bar': getattr(f, 'bar', None)}
            if hasattr(f, 'type') else f
            for f in fixes
        ],
    }

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')


# ═══════════════════════════════════════════════════════════════
#  A2DB — 动机摘要库
# ═══════════════════════════════════════════════════════════════

@dataclass
class MotifDNA:
    """动机的结构化特征向量 — 剥离演奏层后的纯音乐 DNA。

    B 用这些字段做可计算的可比性判断：
    - contour: "当前旋律是主题的倒影吗？"
    - rhythm: "节奏密度是主题的减半还是加倍？"
    - scale_degrees: "再现部用了调内哪几个音级？"
    """
    contour: list[int] = field(default_factory=list)
        # 相邻 Note_ON 的半音差方向序列 [+2, -1, +3, -2, ...]
    rhythm: list[float] = field(default_factory=list)
        # 归一化时长比 [1.0, 0.5, 0.5, 1.0, ...]
    scale_degrees: list[int] = field(default_factory=list)
        # 调内音级（相对 tonic）[1, 3, 5, 4, 3, 2, 1]
    strong_beat_mask: list[bool] = field(default_factory=list)
        # 强拍位置 [T, F, F, T, F, F, T, F]
    chord_at_position: list[int] = field(default_factory=list)
        # 每个音对应的和弦功能 ID [1, 1, 1, 4, 4, 5, 5, 1]
    register_centroid: float = 0.0     # 平均 MIDI 音高
    ambitus: tuple[int, int] = (0, 0)  # (最低音, 最高音)


@dataclass
class MotifRecord:
    """单条动机记录。"""
    section_idx: int
    source_bars: list[int]              # 地标 bar 索引
    label: str                          # "theme1_statement" / "theme1_climax" / ...
    purified_tokens: list[int] = field(default_factory=list)
    dna: MotifDNA = field(default_factory=MotifDNA)


@dataclass
class A2DB:
    """A2 动机摘要库 — 跨段主题记忆 + 提纯 + 发展配方。

    数据来源: A3 的 bar_log（用于选地标）+ 已生成 token（用于提纯）。
    本身不存独立数据，只存索引 + 提纯后的结构化摘要。
    """

    records: dict[str, MotifRecord] = field(default_factory=dict)

    def from_seed(self, seed_tokens: list[int], A3: A3DB, tokenizer):
        """生成开始前调用。从 seed 提取地标 → 存入 records['seed_*']。"""
        from .motif import identify_landmarks, purify_tokens, extract_dna

        if not A3.bar_log or len(A3.bar_log) < 2:
            return

        landmarks = identify_landmarks(A3.bar_log, A3.baselines)
        for label_tag, bar_idx in landmarks:
            # 从 A3 bar_log 定位该 bar 的 token 范围
            label = f'seed_{label_tag}'
            self.records[label] = MotifRecord(
                section_idx=-1,
                source_bars=[bar_idx],
                label=label,
                purified_tokens=purify_tokens(seed_tokens[-256:], tokenizer),
                dna=extract_dna(seed_tokens[-256:], tokenizer),
            )

    def from_section(self, section_idx: int, section_tokens: list[int],
                     A3: A3DB, tokenizer, A1: A1DB | None = None):
        """段生成完后调用。仅主题型段落自动触发。

        1. 查 A3.bar_log → statement / climax / distinctive bar
        2. 提取 → 提纯 → purified_tokens
        3. 提取特征 → MotifDNA
        4. 存入 records[label]
        """
        from .motif import identify_landmarks, purify_tokens, extract_dna

        section = A1.sections[section_idx] if A1 else None
        if not section:
            return

        # 计算段落在 bar_log 中的实际 bar 范围（含 seed 偏移）
        seed_bars = A1.seed_context.bar_count if A1.seed_context else 0
        start_bar = seed_bars
        for k in range(section_idx):
            start_bar += A1.sections[k].bars
        end_bar = start_bar + section.bars
        section_bars = [b for b in A3.bar_log if start_bar <= b.bar < end_bar]

        if len(section_bars) < 2:
            return

        landmarks = identify_landmarks(section_bars, A3.baselines)
        for label_tag, bar_idx in landmarks:
            label = f'{section.type}_{label_tag}'
            # 取该 bar 附近 token（简单实现：按位置切片）
            bar_offset = bar_idx - start_bar
            bar_tokens = _slice_bar_tokens(
                section_tokens, tokenizer, bar_offset,
                section.bars)

            self.records[label] = MotifRecord(
                section_idx=section_idx,
                source_bars=[bar_idx],
                label=label,
                purified_tokens=purify_tokens(bar_tokens, tokenizer),
                dna=extract_dna(purify_tokens(bar_tokens, tokenizer), tokenizer),
            )

    def archive_from_section(self, section_idx: int, tokens: list[int],
                              A3: A3DB, tokenizer, label: str,
                              A1: A1DB | None = None):
        """C 手动归档。发现发展部/插部产生了值得记住的新动机时调用。"""
        self.from_section(section_idx, tokens, A3, tokenizer, A1)
        # 重命名 label
        old_keys = [k for k in self.records if k.startswith(
            A1.sections[section_idx].type if A1 else '')]
        for old_key in old_keys:
            if label not in self.records:
                self.records[label] = self.records.pop(old_key)
                break

    # ── 读取 ────────────────────────────────────

    def get_purified_tokens(self, label: str) -> list[int]:
        """用于拼入 prefix ④。"""
        rec = self.records.get(label)
        return rec.purified_tokens if rec else []

    def get_dna(self, label: str) -> MotifDNA | None:
        """B 用来做逻辑推理（发展配方、偏离度检测）。"""
        rec = self.records.get(label)
        return rec.dna if rec else None

    def find_similar(self, dna: MotifDNA, threshold: float = 0.7) -> list[str]:
        """查库：当前这段和库里的哪个动机最像？（再现部自检用）。"""
        matches = []
        for label, rec in self.records.items():
            if not rec.dna:
                continue
            sim = _dna_similarity(dna, rec.dna)
            if sim >= threshold:
                matches.append(label)
        return matches


# ═══════════════════════════════════════════════════════════════
#  辅助函数
# ═══════════════════════════════════════════════════════════════

def _slice_bar_tokens(tokens: list[int], tokenizer,
                       bar_idx: int, total_bars: int) -> list[int]:
    """从 token 序列中取第 bar_idx 个小节的 token 片段。"""
    bar_token_id = tokenizer.bar_token_id
    bar_count = -1
    start = 0
    for i, tid in enumerate(tokens):
        if tid == bar_token_id:
            bar_count += 1
            if bar_count == bar_idx:
                start = i
            elif bar_count == bar_idx + 1:
                return tokens[start:i]
    # 最后一个小节
    if bar_count >= bar_idx:
        return tokens[start:]
    # fallback: 按比例切
    n = len(tokens)
    seg = n // max(1, total_bars)
    s = bar_idx * seg
    return tokens[s:s + seg]


def _dna_similarity(a: MotifDNA, b: MotifDNA) -> float:
    """计算两个 MotifDNA 的相似度（0-1）。"""
    scores = []

    # contour similarity
    if a.contour and b.contour:
        min_len = min(len(a.contour), len(b.contour))
        matches = sum(1 for i in range(min_len)
                      if a.contour[i] == b.contour[i])
        scores.append(matches / min_len)

    # rhythm similarity
    if a.rhythm and b.rhythm:
        min_len = min(len(a.rhythm), len(b.rhythm))
        diffs = [abs(a.rhythm[i] - b.rhythm[i]) for i in range(min_len)]
        scores.append(1.0 - sum(diffs) / max(min_len, 1))

    # register similarity
    if abs(a.register_centroid) + abs(b.register_centroid) > 0:
        reg_sim = 1.0 - abs(a.register_centroid - b.register_centroid) / max(
            abs(a.register_centroid) + abs(b.register_centroid), 1.0)
        scores.append(reg_sim)

    return sum(scores) / len(scores) if scores else 0.0
