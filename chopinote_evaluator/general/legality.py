"""合法性检查 — 乐谱基本合规性门禁。

7 项检查，不通过（error）则总分=0。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from chopinote_evaluator.parser import Score, score_to_duration_seconds


@dataclass
class LegalityIssue:
    """合法性检查发现的问题。"""
    rule: str            # 规则名
    measure: int         # 小节号（全局适用时为 0）
    severity: str        # "error" / "warning"
    message: str         # 可读描述
    details: dict = field(default_factory=dict)


@dataclass
class LegalityResult:
    """合法性检查结果。"""
    passed: bool                  # 是否通过（无 error 级别问题）
    issues: list[LegalityIssue] = field(default_factory=list)

    @property
    def has_error(self) -> bool:
        return any(i.severity == 'error' for i in self.issues)

    @property
    def has_warning(self) -> bool:
        return any(i.severity == 'warning' for i in self.issues)


# ── 主入口 ─────────────────────────────────────────────────


def check_legality(score: Score) -> LegalityResult:
    """执行全部合法性检查。

    检查项:
        1. 音符密度 [1, 30] notes/s
        2. 音域 21-108 (MIDI)
        3. 连续空小节 >= 2 → warning
        4. 拍号对齐（每小节时值 ±5%）
        5. 声部重叠（同一 voice 内）
        6. Tuplet start/stop 配对
        7. Tie start/stop 配对

    返回:
        LegalityResult
    """
    issues = []
    for check in _CHECKS:
        issues.extend(check(score))

    has_error = any(i.severity == 'error' for i in issues)
    return LegalityResult(passed=not has_error, issues=issues)


# ── 各检查项 ───────────────────────────────────────────────


def _check_note_density(score: Score) -> list[LegalityIssue]:
    """音符密度 [1, 30] notes/s。"""
    total_notes = sum(1 for m in score.measures for n in m.notes if not n.is_rest)
    duration_sec = score_to_duration_seconds(score)
    if duration_sec <= 0:
        return [LegalityIssue('note_density_zero', 0, 'error', '乐谱时长为 0')]
    density = total_notes / duration_sec

    issues = []
    if density < 1:
        issues.append(LegalityIssue(
            'note_density_too_low', 0, 'warning',
            f'音符密度过低: {density:.1f} notes/s（< 1）',
            {'density': density, 'threshold': '1-30'}))
    elif density > 30:
        issues.append(LegalityIssue(
            'note_density_too_high', 0, 'error',
            f'音符密度过高: {density:.1f} notes/s（> 30），可能包含错误',
            {'density': density, 'threshold': '1-30'}))
    return issues


def _check_pitch_range(score: Score) -> list[LegalityIssue]:
    """所有 pitch 应在 21 (A0) - 108 (C8) 范围内。"""
    pitches = []
    for m in score.measures:
        for n in m.notes:
            if not n.is_rest and n.pitch is not None:
                pitches.append((n.pitch, m.number, n.voice))

    out_of_range = [(p, bar, v) for p, bar, v in pitches if p < 21 or p > 108]
    if not out_of_range:
        return []

    # 只报告第一个和总数
    p, bar, v = out_of_range[0]
    return [LegalityIssue(
        'pitch_out_of_range', bar, 'error',
        f'{len(out_of_range)} 个音符超出音域 21-108（MIDI {p}）',
        {'count': len(out_of_range), 'first_pitch': p, 'threshold': '21-108'})]


def _check_empty_measures(score: Score) -> list[LegalityIssue]:
    """连续 2 小节以上全休止 → warning。"""
    empty_streak = 0
    issues = []
    for m in score.measures:
        has_note = any(not n.is_rest for n in m.notes)
        if not has_note:
            empty_streak += 1
        else:
            if empty_streak >= 2:
                issues.append(LegalityIssue(
                    'too_many_empty_measures',
                    m.number - empty_streak,
                    'warning',
                    f'连续 {empty_streak} 小节全休止'))
            empty_streak = 0
    # 尾部连续空小节
    if empty_streak >= 2:
        issues.append(LegalityIssue(
            'too_many_empty_measures',
            score.measures[-1].number - empty_streak + 1,
            'warning',
            f'乐谱尾部连续 {empty_streak} 小节全休止'))
    return issues


def _check_time_sig_align(score: Score) -> list[LegalityIssue]:
    """每小节每声部总时值是否等于拍号，允许 5% 容差。

    对每个 (staff, voice) 独立检查，避免多声部累加误判。
    """
    issues = []
    for m in score.measures:
        beats, unit = m.time_signature
        expected = beats * 4.0 / unit  # 以四分音符为单位归一化
        if expected <= 0:
            continue

        # 按 (staff, voice) 分组合计时值（跳过 grace notes）
        voice_groups: dict[tuple[int, int], float] = {}
        for n in m.notes:
            if n.grace:
                continue
            key = (n.staff, n.voice)
            voice_groups[key] = voice_groups.get(key, 0.0) + n.duration

        for key, total in voice_groups.items():
            if abs(total - expected) > expected * 0.05:
                staff, voice = key
                issues.append(LegalityIssue(
                    'time_sig_mismatch', m.number, 'error',
                    f'声部(staff={staff} voice={voice}) 时值 {total:.2f} ≠ 拍号期望 {expected:.2f}',
                    {'staff': staff, 'voice': voice, 'actual': total,
                     'expected': expected, 'diff_pct': abs(total - expected) / expected * 100}))
    return issues


def _check_voice_overlap(score: Score) -> list[LegalityIssue]:
    """同一 staff 内，同一 voice 的音符不能有重叠（同时发声）。"""
    issues = []
    for m in score.measures:
        voices: dict[tuple[int, int], list] = {}
        for n in m.notes:
            if n.is_rest:
                continue
            key = (n.staff, n.voice)
            voices.setdefault(key, []).append(n)

        for key, notes in voices.items():
            sorted_notes = sorted(notes, key=lambda x: x.onset)
            for i in range(len(sorted_notes) - 1):
                n1, n2 = sorted_notes[i], sorted_notes[i + 1]
                if n1.onset + n1.duration > n2.onset + 1e-4:
                    issues.append(LegalityIssue(
                        'voice_overlap', m.number, 'warning',
                        f'声部 staff={key[0]} voice={key[1]} 音符重叠',
                        {'staff': key[0], 'voice': key[1],
                         'note1': {'pitch': n1.pitch, 'onset': n1.onset, 'duration': n1.duration},
                         'note2': {'pitch': n2.pitch, 'onset': n2.onset, 'duration': n2.duration}}))
                    break  # 每声部每小节只报一次
    return issues


def _check_tuplet(score: Score) -> list[LegalityIssue]:
    """tuplet start/stop 必须配对。"""
    issues = []
    for m in score.measures:
        starts = sum(1 for n in m.notes if n.tuplet_start)
        stops = sum(1 for n in m.notes if n.tuplet_stop)
        if starts != stops:
            issues.append(LegalityIssue(
                'tuplet_mismatch', m.number, 'error',
                f'连音 start({starts})/stop({stops}) 不匹配'))
    return issues


def _check_ties(score: Score) -> list[LegalityIssue]:
    """所有 tie 必须成对（start → stop），跨小节追踪。"""
    open_ties: set[tuple] = set()  # (pitch, voice, staff)
    issues = []
    for m in score.measures:
        for n in m.notes:
            key = (n.pitch, n.voice, n.staff)
            if n.is_tie_start:
                if key in open_ties:
                    issues.append(LegalityIssue(
                        'tie_chain', m.number, 'warning',
                        f'未闭合的连音前驱（pitch={n.pitch}）'))
                open_ties.add(key)
            if n.is_tie_stop:
                if key not in open_ties:
                    issues.append(LegalityIssue(
                        'orphan_tie_stop', m.number, 'error',
                        f'孤立的连音结尾（pitch={n.pitch}）'))
                open_ties.discard(key)

    # 未闭合的跨小节连音（在最后一个小节处报 warning）
    for pitch, voice, staff in open_ties:
        if score.measures:
            last_m = score.measures[-1].number
            issues.append(LegalityIssue(
                'unclosed_tie', last_m, 'warning',
                f'未闭合的连音（pitch={pitch} voice={voice}）'))
    return issues


# ── 检查项注册 ─────────────────────────────────────────────

_CHECKS = [
    _check_note_density,
    _check_pitch_range,
    _check_empty_measures,
    _check_time_sig_align,
    _check_voice_overlap,
    _check_tuplet,
    _check_ties,
]
