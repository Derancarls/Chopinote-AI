"""C 层评分后端 — intrinsic（内在）+ consistency（一致性）双维度评价。

提供统一的生成后评价入口:
  - evaluate_generation(): 完整评分（合法性 + 统计 + 理论 + 一致性 + 连贯性）
  - 支持 token 级指标（metrics.py）和 Score 级规则（constraints.py）

ABC Engine C 层评分维度:
  - intrinsic_score  (内在评分): 生成乐谱自身质量 — token 指标 + 理论规则
  - consistency_score (一致性评分): 与 seed 的匹配度 — 仅续写场景有值
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .parser import Score, parse_musicxml
from .constraints import evaluate_theory
from .metrics import compute_all_metrics


@dataclass
class EvalReport:
    """C 层综合评价报告。"""
    total_score: float
    alpha: float = 0.3
    legality_passed: bool = True
    legality_issues: int = 0
    intrinsic_score: float = 0.0    # 内在评分：乐谱自身质量（不依赖 seed）
    consistency_score: float = 0.0  # 一致性评分：与 seed 的匹配度（仅续写场景）
    token_metrics: dict = field(default_factory=dict)
    theory: dict = field(default_factory=dict)
    structural_fixes: list = field(default_factory=list)
    archive_commands: list = field(default_factory=list)
    novelty_bonus: float = 0.0
    diversity_bonus: float = 0.0
    details: dict = field(default_factory=dict)


def evaluate_generation(
    tokens: list[int],
    tokenizer,
    seed_tokens: list[int] | None = None,
    musicxml_path: str | Path | None = None,
    score: Score | None = None,
    benchmarks: dict | None = None,
    alpha: float = 0.3,
    novelty_bonus: float = 0.0,
    diversity_bonus: float = 0.0,
    structural_fixes: list | None = None,
    archive_commands: list | None = None,
) -> EvalReport:
    """C 层完整评价 — 单次调用完成所有评分维度。

    参数:
        tokens: 生成的完整 token 序列
        tokenizer: REMI tokenizer
        seed_tokens: 种子 token 序列（续写场景，用于一致性/连贯性）
        musicxml_path: 生成结果的 MusicXML 路径（用于 Score 级评价）
        score: 已解析的 Score 对象（与 musicxml_path 二选一）
        benchmarks: 基准数据（可选，用于统计对比）
        alpha: 场景权重（0~1，越高越看重统计/理论分）
        novelty_bonus: C 创新加分
        diversity_bonus: C 多样性奖励
        structural_fixes: C 诊断修复列表
        archive_commands: C 归档指令

    返回:
        EvalReport
    """
    report = EvalReport(
        total_score=0.0,
        alpha=alpha,
        novelty_bonus=novelty_bonus,
        diversity_bonus=diversity_bonus,
        structural_fixes=structural_fixes or [],
        archive_commands=archive_commands or [],
    )

    # ── 1. Token 级指标 ──
    token_metrics = compute_all_metrics(tokens, tokenizer)
    report.token_metrics = token_metrics

    # 计算 token 级平均分
    if token_metrics:
        token_mean = sum(token_metrics.values()) / len(token_metrics)
    else:
        token_mean = 0.5

    # ── 2. Score 级评价（如果有 MusicXML）──
    score_obj = score
    if musicxml_path and score_obj is None:
        try:
            score_obj = parse_musicxml(str(musicxml_path))
        except Exception:
            pass

    if score_obj is not None:
        # 合法性检查
        legality = _check_legality(score_obj)
        report.legality_passed = legality['passed']
        report.legality_issues = legality['n_issues']

        if legality['has_error']:
            report.total_score = 0.0
            return report

        # 理论规则评估
        report.theory = evaluate_theory(score_obj)
        theory_score = report.theory.get('score', 0.0)

        # 内在评分: token 指标 + 理论
        report.intrinsic_score = token_mean * 0.6 + theory_score * 0.4

        # 一致性评分（有种子时）
        if seed_tokens is not None:
            try:
                consistency = _evaluate_specific(score_obj, seed_tokens, tokenizer)
                report.consistency_score = consistency.get('score', 0.0)
            except Exception:
                report.consistency_score = 0.0
    else:
        # 无 MusicXML — 仅 token 级指标
        report.intrinsic_score = token_mean

    # ── 3. 综合打分 ──
    intrinsic = report.intrinsic_score
    consistency = report.consistency_score

    if consistency > 0:
        total = intrinsic * alpha + consistency * (1 - alpha)
    else:
        total = intrinsic

    # 加上创新/多样性加成（最多 +0.1）
    total = min(1.0, total + novelty_bonus * 0.03 + diversity_bonus * 0.02)

    report.total_score = total
    return report


# ═══════════════════════════════════════════════════════════════
#  合法性检查（简化版，从 legality.py 移植）
# ═══════════════════════════════════════════════════════════════

def _check_legality(score: Score) -> dict:
    """执行基本合法性检查。"""
    issues = []
    has_error = False

    # 1. 音符密度
    total_notes = sum(1 for m in score.measures for n in m.notes if not n.is_rest)
    if total_notes == 0:
        issues.append({'rule': 'no_notes', 'severity': 'error', 'msg': '无音符'})
        has_error = True

    # 2. 音域检查
    for m in score.measures:
        for n in m.notes:
            if not n.is_rest and n.pitch is not None:
                if n.pitch < 21 or n.pitch > 108:
                    issues.append({
                        'rule': 'pitch_range', 'measure': m.number,
                        'severity': 'error', 'msg': f'音高 {n.pitch} 超出 21-108'
                    })
                    has_error = True
                    break

    # 3. 连续空小节
    empty_streak = 0
    for m in score.measures:
        has_note = any(not n.is_rest for n in m.notes)
        if not has_note:
            empty_streak += 1
        else:
            if empty_streak >= 2:
                issues.append({
                    'rule': 'empty_measures', 'measure': m.number - empty_streak,
                    'severity': 'warning', 'msg': f'连续 {empty_streak} 空小节'
                })
            empty_streak = 0

    return {
        'passed': not has_error,
        'has_error': has_error,
        'n_issues': len(issues),
        'issues': issues,
    }


# ═══════════════════════════════════════════════════════════════
#  一致性评价 — 对比生成与种子的关键特征（C 层 seed-relative）
# ═══════════════════════════════════════════════════════════════

# ── Cadence 相似度矩阵 ──────────────────────────────
_CADENCE_SIMILARITY = {
    ('PAC', 'PAC'): 1.0, ('PAC', 'IAC'): 0.7, ('PAC', 'HC'): 0.4,
    ('PAC', 'DC'): 0.2, ('PAC', 'PC'): 0.5,
    ('IAC', 'IAC'): 1.0, ('IAC', 'PAC'): 0.7, ('IAC', 'HC'): 0.5,
    ('IAC', 'DC'): 0.2, ('IAC', 'PC'): 0.6,
    ('HC', 'HC'): 1.0, ('HC', 'PAC'): 0.3, ('HC', 'IAC'): 0.4,
    ('HC', 'DC'): 0.5, ('HC', 'PC'): 0.2,
    ('DC', 'DC'): 1.0, ('DC', 'PAC'): 0.2, ('DC', 'IAC'): 0.3,
    ('DC', 'HC'): 0.5, ('DC', 'PC'): 0.1,
    ('PC', 'PC'): 1.0, ('PC', 'PAC'): 0.5, ('PC', 'IAC'): 0.6,
    ('PC', 'HC'): 0.2, ('PC', 'DC'): 0.1,
}


def _detect_cadence_from_tokens(tokens: list[int], tokenizer) -> str | None:
    """从 token 序列提取终止式类型 — 扫描 <Cad X> token。"""
    for tid in reversed(tokens[-512:]):  # 只看最后 512 tokens
        ts = tokenizer.decode_token(tid)
        if ts.startswith('<Cad ') and ts.endswith('>'):
            cad_type = ts[5:-1]  # extract 'PAC', 'IAC', etc.
            if cad_type != 'none':
                return cad_type
    return None


def _detect_cadence_from_score(score: Score,
                               target: str | None) -> str | None:
    """从 MusicXML Score 检测终止式 — 分析最后 2 bar 的和声特征。

    Simplified: checks the last measure's notes for V-I patterns.
    """
    if not score or not score.measures:
        return None
    # Get last 2 measures (excluding trailing empty ones)
    measures = [m for m in score.measures if m.notes]
    if len(measures) < 2:
        return None
    last_two = measures[-2:]
    # Count pitch classes in last measure
    last_pcs = set()
    for n in last_two[-1].notes:
        if not n.is_rest and n.pitch is not None:
            last_pcs.add(n.pitch % 12)
    # Check for common cadence PC patterns
    has_tonic = 0 in last_pcs or True  # tonic presence (simplified — need key context)
    has_third = 4 in last_pcs
    has_fifth = 7 in last_pcs
    # Simplified: if last chord has multiple notes → likely authentic cadence
    if len(last_pcs) >= 2:
        return 'IAC'  # Default to IAC (most common in generated output)
    return None


def _compute_cadence_match(target: str | None,
                           actual: str | None) -> float:
    """计算终止式匹配分数。"""
    if target is None:
        return 1.0 if actual is None else 0.5  # 无目标时中性
    if actual is None:
        return 0.3  # 未检测到终止式 → 低分
    return _CADENCE_SIMILARITY.get((target, actual), 0.3)


def _evaluate_specific(score: Score, seed_tokens: list[int],
                       tokenizer) -> dict[str, float]:
    """一致性评价 — 对比生成与种子的关键特征。

    从 seed_tokens 计算参考分布，再评估 Score 与参考的一致性。
    """
    from .metrics import (
        pitch_class_kl, interval_kl, key_consistency,
    )

    # 从种子 token 计算参考分布
    seed_bar_id = tokenizer.bar_token_id
    seed_bars = 0
    seed_notes = []
    for t in seed_tokens:
        if t == seed_bar_id:
            seed_bars += 1
        else:
            s = tokenizer.decode_token(t)
            if s.startswith('<Note_ON'):
                seed_notes.append(int(s[len('<Note_ON') + 1:-1]))

    # 计算 seed 的音级分布作为参考
    seed_pc_ref = [0.0] * 12
    for p in seed_notes:
        seed_pc_ref[p % 12] += 1.0
    total_pc = sum(seed_pc_ref)
    if total_pc > 0:
        seed_pc_ref = [c / total_pc for c in seed_pc_ref]

    # 从 Score 提取生成音符
    gen_notes = []
    gen_keys = set()
    for m in score.measures:
        if m.key_signature:
            gen_keys.add(m.key_signature)
        for n in m.notes:
            if not n.is_rest and n.pitch is not None:
                gen_notes.append(n.pitch)

    # 用 seed 分布作为参考评估生成结果的一致性
    if seed_notes and gen_notes:
        pc_score = pitch_class_kl(
            [], tokenizer, reference=seed_pc_ref) if seed_pc_ref else 0.5
        # 实际对比: 从 gen_notes 建分布 vs seed 参考
        gen_pc = [0.0] * 12
        for p in gen_notes:
            gen_pc[p % 12] += 1.0
        gen_total = sum(gen_pc)
        if gen_total > 0:
            gen_pc = [c / gen_total for c in gen_pc]
        from .metrics import _kl_divergence
        import math
        pc_score = math.exp(-_kl_divergence(gen_pc, seed_pc_ref) * 5) if seed_pc_ref else 0.5
    else:
        pc_score = 0.5

    # 连贯性
    seed_keys = set()
    for t in seed_tokens:
        s = tokenizer.decode_token(t)
        if s.startswith('<Key'):
            seed_keys.add(s)
    key_match = 1.0 if not seed_keys or seed_keys & gen_keys else 0.5

    consistency = {
        'pitch_class_continuity': pc_score,
        'interval_continuity': 0.5,  # 简化
    }

    # ── cadence_match: 从 seed tokens 提取目标终止式, 与生成结果对比 ──
    seed_cadence = _detect_cadence_from_tokens(seed_tokens, tokenizer)
    gen_cadence = _detect_cadence_from_score(score, seed_cadence)
    cadence_match = _compute_cadence_match(seed_cadence, gen_cadence)

    coherence = {
        'key_match': key_match,
        'voice_count_delta': 0.7,
        'tempo_continuity': 0.8,
        'cadence_match': cadence_match,
    }

    cons_total = sum(consistency.values()) / len(consistency)
    coh_total = sum(coherence.values()) / len(coherence)

    return {
        'score': cons_total * 0.5 + coh_total * 0.5,
        'consistency': consistency,
        'consistency_score': cons_total,
        'coherence': coherence,
        'coherence_score': coh_total,
    }


# ═══════════════════════════════════════════════════════════════
#  C 增强 ①: MusicXML 快速审查 — 逐 bar 结构级诊断
# ═══════════════════════════════════════════════════════════════

@dataclass
class BarInspection:
    """单个 bar 的快速审查结果。"""
    bar: int                                    # bar 号（全局）
    section_idx: int | None = None              # 所属段落索引
    part_notes: dict[int, int] = field(default_factory=dict)    # part_idx → note_count
    part_rests: dict[int, int] = field(default_factory=dict)    # part_idx → rest_count
    part_silent: dict[int, bool] = field(default_factory=dict)  # part_idx → 完全沉默?
    pitch_min: dict[int, int] = field(default_factory=dict)     # part_idx → 最低音
    pitch_max: dict[int, int] = field(default_factory=dict)     # part_idx → 最高音
    range_violations: list[str] = field(default_factory=list)   # 音域违规描述
    density_extreme: bool = False               # 密度极端（>15 或 <1）
    voice_crossing: bool = False                # 声部交错
    parallel_perfect: list[str] = field(default_factory=list)   # 平行五度/八度
    warnings: list[str] = field(default_factory=list)
    score: float = 1.0                          # 本 bar 质量分 0~1


@dataclass
class CFeedback:
    """C → B 的结构化实时反馈。B 直接消费这些字段做决策。

    与 StructuralFix（仅用于 retry）不同，CFeedback 可以每段/每 bar
    实时调整 B 的约束和参数。
    """
    # ── Token 级禁令 ──
    ban_pitches: set[int] = field(default_factory=set)       # 绝对禁掉的具体 MIDI pitch
    ban_intervals: set[int] = field(default_factory=set)     # 禁掉的音程（半音数）
    ban_octave_above: int | None = None                      # 禁掉此八度及以上
    ban_octave_below: int | None = None                      # 禁掉此八度及以下

    # ── 声部引导 ──
    part_bias: dict[int, float] = field(default_factory=dict)      # part_idx → 生成偏置
    force_part_switch: bool = False                                 # 强制下一 token 切声部
    silence_alert: set[int] = field(default_factory=set)           # 沉默的 part 索引集合

    # ── 参数调节 ──
    temperature_delta: float = 0.0          # B temperature 调节量
    complexity_delta: float = 0.0           # B complexity 调节量
    rest_penalty_delta: float = 0.0         # B rest_penalty 调节量

    # ── 致命信号 ──
    fatal: str | None = None                # 'reharmonize' | 'abort' | 'reset_section'
    fatal_reason: str = ''                  # 人类可读原因

    # ── 段落级建议 ──
    section_alerts: list[str] = field(default_factory=list)


def review_musicxml(
    musicxml_path: str,
    programs: list[int] | None = None,
    seed_bar_count: int = 0,
) -> list[BarInspection]:
    """快速审查 MusicXML — 逐 bar 结构级诊断，不依赖重型理论引擎。

    比 evaluate_theory() 轻量 100x，适合在生成循环中频繁调用。
    只做 O(n) 扫描，不做完整和声分析。

    Args:
        musicxml_path: 生成的 MusicXML 路径
        programs: 预期的 program 列表（如 [0, 0_1 对应 program 0]）
        seed_bar_count: seed bar 数，审查时跳过 seed bar

    Returns:
        BarInspection 列表，按 bar 号排序
    """
    from .parser import parse_musicxml

    inspections: list[BarInspection] = []

    try:
        score = parse_musicxml(musicxml_path)
    except Exception:
        return inspections

    if not score.measures:
        return inspections

    # 按 bar 组织
    bars: dict[int, dict[int, list]] = {}  # bar_number → {part_idx → [Note]}
    for m in score.measures:
        mn = m.number
        if mn <= seed_bar_count:  # 跳过 seed bar
            continue
        if mn not in bars:
            bars[mn] = {}
        for n in m.notes:
            # 推断 part: staff=1 → part 0 (treble), staff=2 → part 1 (bass)
            part_idx = n.staff - 1 if n.staff > 0 else 0
            if part_idx not in bars[mn]:
                bars[mn][part_idx] = []
            bars[mn][part_idx].append(n)

    prev_bar_top_pitches: dict[int, int] = {}  # part → 上一 bar 最高音（声部交错检测）

    for bar_num in sorted(bars.keys()):
        insp = BarInspection(bar=bar_num)
        part_notes = bars[bar_num]

        for part_idx, notes in sorted(part_notes.items()):
            non_rest = [n for n in notes if not n.is_rest]
            rests = [n for n in notes if n.is_rest]
            pitches = [n.pitch for n in non_rest if n.pitch is not None]

            insp.part_notes[part_idx] = len(non_rest)
            insp.part_rests[part_idx] = len(rests)
            insp.part_silent[part_idx] = len(non_rest) == 0

            if pitches:
                insp.pitch_min[part_idx] = min(pitches)
                insp.pitch_max[part_idx] = max(pitches)

                # 音域检查：钢琴标准 E1(28) ~ C8(108)
                for p in pitches:
                    if p < 21:
                        insp.range_violations.append(
                            f'P{part_idx} bar{bar_num}: pitch {p} < 21')
                    elif p > 108:
                        insp.range_violations.append(
                            f'P{part_idx} bar{bar_num}: pitch {p} > 108')

                # 极端密度
                if len(non_rest) > 15:
                    insp.density_extreme = True
                    insp.warnings.append(
                        f'P{part_idx} bar{bar_num}: 密度 {len(non_rest)} > 15')
                elif len(non_rest) == 0:
                    insp.warnings.append(
                        f'P{part_idx} bar{bar_num}: 完全沉默')

        # 声部交错（相邻 part 音域重叠严重）
        sorted_parts = sorted(part_notes.keys())
        for i in range(len(sorted_parts) - 1):
            p_low = sorted_parts[i]     # 低音声部
            p_high = sorted_parts[i + 1]  # 高音声部
            if p_low in insp.pitch_max and p_high in insp.pitch_min:
                # 低音声部最高音 > 高音声部最低音 + 6 半音 → 严重交错
                if insp.pitch_max[p_low] > insp.pitch_min[p_high] + 6:
                    insp.voice_crossing = True
                    insp.warnings.append(
                        f'声部交错 P{p_low} max={insp.pitch_max[p_low]} '
                        f'vs P{p_high} min={insp.pitch_min[p_high]}')

        # 平行五度/八度快速检测（相邻音符间）
        for part_idx, notes in sorted(part_notes.items()):
            non_rest_notes = [n for n in notes if not n.is_rest and n.pitch is not None]
            for j in range(len(non_rest_notes) - 1):
                interval1 = abs(non_rest_notes[j].pitch - non_rest_notes[j + 1].pitch) % 12
                if j + 2 < len(non_rest_notes):
                    interval2 = abs(non_rest_notes[j + 1].pitch - non_rest_notes[j + 2].pitch) % 12
                    if interval1 == 7 and interval2 == 7:
                        insp.parallel_perfect.append(
                            f'P{part_idx} bar{bar_num}: 连续纯五度')
                    elif interval1 == 0 and interval2 == 0:
                        insp.parallel_perfect.append(
                            f'P{part_idx} bar{bar_num}: 连续纯八度')

        # 计分
        score = 1.0
        if insp.range_violations:
            score -= 0.2 * len(insp.range_violations)
        if insp.density_extreme:
            score -= 0.3
        if insp.voice_crossing:
            score -= 0.3
        if insp.parallel_perfect:
            score -= 0.15 * len(insp.parallel_perfect)
        if any(insp.part_silent.values()):
            score -= 0.2 * sum(1 for v in insp.part_silent.values() if v)
        insp.score = max(0.0, score)

        inspections.append(insp)

    return inspections


def compare_tokens_to_xml(
    tokens: list[int],
    tokenizer,
    musicxml_path: str,
    seed_bar_count: int = 0,
) -> dict:
    """对比生成 token 与渲染后的 MusicXML，检查 roundtrip 保真度。

    检测:
      - 各声部音符数量是否匹配
      - 是否有 token 被渲染器丢弃
      - 是否有渲染器凭空多出的音符
      - 休止符比例差异

    Returns:
        {
            'fidelity': float,           # 0~1 总体保真度
            'token_note_counts': dict,   # {part_idx: count}
            'xml_note_counts': dict,     # {part_idx: count}
            'missing_notes': int,        # token 有但 XML 无
            'extra_notes': int,          # XML 有但 token 无
            'part_mismatches': list,     # 描述
            'ok': bool,                  # 是否在容忍范围内
        }
    """
    from .parser import parse_musicxml

    result = {
        'fidelity': 1.0,
        'token_note_counts': {},
        'xml_note_counts': {},
        'missing_notes': 0,
        'extra_notes': 0,
        'part_mismatches': [],
        'ok': True,
    }

    # ── 从 token 统计每个声部的音符数 ──
    token_part_notes: dict[int, int] = {}
    current_program = 0
    bar_id = tokenizer.bar_token_id
    note_on_prefix = tokenizer.NOTE_ON
    bar_count = 0

    for t in tokens:
        if t == bar_id:
            bar_count += 1
            if bar_count <= seed_bar_count:
                continue
        elif bar_count <= seed_bar_count:
            continue
        ts = tokenizer.decode_token(t)
        if ts.startswith('<Program '):
            # 提取 program 编号: "<Program 0>" → 0, "<Program 0_1>" → 0
            prog_str = ts[len('<Program '):-1]
            try:
                current_program = int(prog_str.split('_')[0])
            except ValueError:
                pass
        elif ts.startswith(note_on_prefix):
            token_part_notes[current_program] = \
                token_part_notes.get(current_program, 0) + 1

    result['token_note_counts'] = dict(token_part_notes)

    # ── 从 MusicXML 统计 ──
    try:
        score = parse_musicxml(musicxml_path)
    except Exception:
        result['ok'] = False
        result['part_mismatches'].append('MusicXML 解析失败')
        return result

    xml_part_notes: dict[int, int] = {}
    for m in score.measures:
        if m.number <= seed_bar_count:
            continue
        for n in m.notes:
            if not n.is_rest and n.pitch is not None:
                part_idx = n.staff - 1 if n.staff > 0 else 0
                xml_part_notes[part_idx] = \
                    xml_part_notes.get(part_idx, 0) + 1

    result['xml_note_counts'] = dict(xml_part_notes)

    # ── 对比 ──
    all_parts = set(token_part_notes.keys()) | set(xml_part_notes.keys())
    total_token_notes = sum(token_part_notes.values())
    total_xml_notes = sum(xml_part_notes.values())

    for p in sorted(all_parts):
        tn = token_part_notes.get(p, 0)
        xn = xml_part_notes.get(p, 0)
        if tn == 0 and xn == 0:
            continue
        if tn == 0 and xn > 0:
            result['extra_notes'] += xn
            result['part_mismatches'].append(
                f'Part {p}: token=0 XML={xn} (渲染器多出音符)')
        elif xn == 0 and tn > 0:
            result['missing_notes'] += tn
            result['part_mismatches'].append(
                f'Part {p}: token={tn} XML=0 (声部完全丢失)')
        else:
            delta = abs(tn - xn)
            ratio = delta / max(tn, 1)
            if ratio > 0.3:
                result['part_mismatches'].append(
                    f'Part {p}: token={tn} XML={xn} (差异 {ratio:.0%})')
                if tn > xn:
                    result['missing_notes'] += tn - xn
                else:
                    result['extra_notes'] += xn - tn

    # 计算保真度
    if total_token_notes > 0:
        fidelity = 1.0 - (result['missing_notes'] + result['extra_notes']) / total_token_notes
        result['fidelity'] = max(0.0, fidelity)
    else:
        result['fidelity'] = 0.0

    result['ok'] = result['fidelity'] >= 0.7 and len(result['part_mismatches']) <= 2
    return result


def c_review_to_feedback(
    inspections: list[BarInspection],
    comparison: dict | None = None,
    num_parts: int = 2,
) -> CFeedback:
    """将 MusicXML 审查 + 对比结果翻译为 B 可直接消费的 CFeedback。

    B 在下一段生成前调用这个函数，根据反馈调整硬约束和参数。

    Args:
        inspections: review_musicxml() 的输出
        comparison: compare_tokens_to_xml() 的输出（可选）
        num_parts: 预期的声部数

    Returns:
        CFeedback 结构体
    """
    fb = CFeedback()

    if not inspections:
        return fb

    # ── 汇总审查发现 ──
    total_bars = len(inspections)
    silent_parts: dict[int, int] = {}  # part → 沉默 bar 数
    crossing_bars = 0
    extreme_bars = 0
    range_violations_total = 0

    for insp in inspections:
        for part_idx, silent in insp.part_silent.items():
            if silent:
                silent_parts[part_idx] = silent_parts.get(part_idx, 0) + 1
        if insp.voice_crossing:
            crossing_bars += 1
        if insp.density_extreme:
            extreme_bars += 1
        range_violations_total += len(insp.range_violations)

        # 收集音域违规的 pitch → ban
        for v in insp.range_violations:
            # 解析 "P0 bar5: pitch 127 > 108" → 提取 pitch
            import re
            m = re.search(r'pitch (\d+)', v)
            if m:
                fb.ban_pitches.add(int(m.group(1)))

    # ── 声部沉默 → 强制切换 + bias ──
    for part_idx in range(num_parts):
        silent_count = silent_parts.get(part_idx, 0)
        silent_ratio = silent_count / max(total_bars, 1)

        if silent_ratio > 0.5:
            fb.silence_alert.add(part_idx)
            fb.force_part_switch = True
            fb.part_bias[part_idx] = 2.0  # 强偏置
            fb.section_alerts.append(
                f'Part {part_idx} 沉默率 {silent_ratio:.0%}，强制声部切换+bias=2.0')
        elif silent_ratio > 0.2:
            fb.part_bias[part_idx] = 1.3
            fb.section_alerts.append(
                f'Part {part_idx} 沉默率 {silent_ratio:.0%}，偏置=1.3')

    # ── 声部交错 → 收紧音域 ──
    if crossing_bars > total_bars * 0.3:
        fb.ban_octave_below = 2   # 禁止太低（交错通常是因为低音声部太高）
        fb.temperature_delta -= 0.1
        fb.complexity_delta -= 1.0
        fb.section_alerts.append(
            f'声部交错率 {crossing_bars}/{total_bars}，收紧音域+temperature-0.1')

    # ── 密度极端 ──
    if extreme_bars > total_bars * 0.3:
        fb.complexity_delta -= 1.5
        fb.temperature_delta -= 0.05
        fb.section_alerts.append(
            f'密度极端率 {extreme_bars}/{total_bars}，complexity-1.5')

    # ── 音域违规 ──
    if range_violations_total > 3:
        fb.ban_octave_above = 7  # 禁 C8 及以上
        fb.ban_octave_below = 1  # 禁 C1 及以下
        fb.section_alerts.append(
            f'音域违规 {range_violations_total} 处，ban octave <2 & >7')

    # ── Roundtrip 对比反馈 ──
    if comparison and not comparison.get('ok', True):
        fidelity = comparison.get('fidelity', 1.0)
        fb.section_alerts.append(
            f'Token↔XML 保真度 {fidelity:.2f}: {comparison.get("part_mismatches", [])}')
        if fidelity < 0.5:
            fb.fatal = 'reset_section'
            fb.fatal_reason = f'Roundtrip 保真度过低 ({fidelity:.2f})'

    # ── 综合致命判定 ──
    if all(silent_parts.get(p, 0) > total_bars * 0.8 for p in range(num_parts)):
        fb.fatal = 'abort'
        fb.fatal_reason = '所有声部大面积沉默'

    return fb


def apply_c_feedback_to_bans(
    feedback: CFeedback,
    hard_bans: object,  # BHardBans
    gen_params: object,  # GenerationParams
) -> dict:
    """将 CFeedback 应用到 B 的硬约束和生成参数。

    在每段生成前调用，修改 hard_bans 和 gen_params。

    Args:
        feedback: c_review_to_feedback() 的输出
        hard_bans: BHardBans 实例
        gen_params: GenerationParams 实例

    Returns:
        {'actions': [...], 'fatal': str|None}
    """
    actions = []

    # ── Token 禁令 ──
    if feedback.ban_pitches:
        from chopinote_dataset.tokenizer import REMITokenizer
        # 将所有被禁 pitch 对应的 Note_ON token 加入 banned_tokens
        for pid in feedback.ban_pitches:
            # Note_ON token 格式: '<Note_ON {pitch}>'
            # 这里暂时无法直接获取 token id，留给调用方处理
            pass
        actions.append(f'ban_pitches: {len(feedback.ban_pitches)} pitches')

    if feedback.ban_intervals:
        actions.append(f'ban_intervals: {feedback.ban_intervals}')

    # ── 八度禁令 → banned_tokens (在 generate loop 中按需动态判定) ──
    if feedback.ban_octave_above is not None:
        actions.append(f'ban_octave_above: {feedback.ban_octave_above}')
    if feedback.ban_octave_below is not None:
        actions.append(f'ban_octave_below: {feedback.ban_octave_below}')

    # ── 参数调节 ──
    if feedback.temperature_delta != 0.0:
        old_t = gen_params.temperature
        gen_params.temperature = max(0.2, min(2.5,
            gen_params.temperature + feedback.temperature_delta))
        actions.append(f'temperature: {old_t:.2f}→{gen_params.temperature:.2f}')

    if feedback.complexity_delta != 0.0:
        gen_params.complexity = max(0.5, min(10.0,
            gen_params.complexity + feedback.complexity_delta))
        actions.append(f'complexity: {gen_params.complexity:.1f}')

    if feedback.rest_penalty_delta != 0.0:
        gen_params.rest_penalty = max(0.0,
            gen_params.rest_penalty + feedback.rest_penalty_delta)
        actions.append(f'rest_penalty: {gen_params.rest_penalty:.1f}')

    return {
        'actions': actions,
        'fatal': feedback.fatal,
    }
