"""B 规则后端 — 从 chopinote_evaluator/general/ 移植的音乐理论规则。

分为两层:
  - Token 级: 生成时实时检查（B 层使用），不依赖 MusicXML 解析
  - Score 级: 生成后完整评估（C 层使用），依赖 parser.py 的 Score 对象
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════════════
#  Token 级约束 — B 层实时使用（不依赖 Score/MusicXML）
# ═══════════════════════════════════════════════════════════════

@dataclass
class TokenConstraint:
    """Token 级约束结果。"""
    rule: str
    banned_token_ids: list[int] = field(default_factory=list)
    severity: str = "hard"        # "hard" / "soft"
    description: str = ""


def check_parallel_fifths_octaves_tokens(
    bar_tokens: list[int],
    prev_bar_notes: list[int],
    tokenizer,
) -> list[TokenConstraint]:
    """Token 级平行五度/八度检测 — 检查当前 bar 与前一 bar 的对应声部。

    在 B 生成新 bar 时调用，如果有平行五度/八度，返回应禁止的 token ID。
    """
    # 提取当前 bar 的 note_on intervals
    cur_notes = []
    for t in bar_tokens:
        s = tokenizer.decode_token(t)
        if s.startswith('<Note_ON'):
            cur_notes.append(int(s[len('<Note_ON') + 1:-1]))

    if len(cur_notes) < 2 or len(prev_bar_notes) < 2:
        return []

    constraints = []
    n_voices = min(len(cur_notes), len(prev_bar_notes))
    cur_sorted = sorted(cur_notes)
    prev_sorted = sorted(prev_bar_notes)

    for i in range(n_voices):
        for j in range(i + 1, n_voices):
            interval_prev = abs(prev_sorted[j] - prev_sorted[i])
            interval_cur = abs(cur_sorted[j] - cur_sorted[i])

            # 纯五度/八度平行移动 → 硬约束
            if interval_prev % 12 in (7, 0) and interval_cur % 12 == interval_prev % 12:
                if interval_prev != 0 and (prev_sorted[i] != cur_sorted[i] or prev_sorted[j] != cur_sorted[j]):
                    # 找到产生违规的 token，禁止它
                    banned = [t for t in bar_tokens
                             if tokenizer.decode_token(t).startswith('<Note_ON')
                             and int(tokenizer.decode_token(t)[len('<Note_ON') + 1:-1]) == cur_sorted[j]]
                    if banned:
                        constraints.append(TokenConstraint(
                            rule='parallel_fifth_octave',
                            banned_token_ids=banned[-1:],
                            severity='hard',
                            description=f'平行{"五度" if interval_cur % 12 == 7 else "八度"}'
                        ))

    return constraints


def check_voice_crossing_tokens(
    bar_tokens: list[int],
    prev_bar_notes: list[int],
    tokenizer,
) -> list[TokenConstraint]:
    """Token 级声部交叉检测。"""
    cur_notes = []
    for t in bar_tokens:
        s = tokenizer.decode_token(t)
        if s.startswith('<Note_ON'):
            cur_notes.append(int(s[len('<Note_ON') + 1:-1]))

    if len(cur_notes) < 2:
        return []

    cur_sorted = sorted(cur_notes)
    # 简单检查：是否有低音部跳到高音部之上
    for i in range(len(cur_sorted) - 1):
        if cur_sorted[i + 1] > cur_sorted[i] + 12:  # 相邻升序音跨度 > 12 半音
            return [TokenConstraint(
                rule='voice_crossing',
                banned_token_ids=[],
                severity='soft',
                description='声部交叉'
            )]

    return []


def check_extreme_jump_tokens(
    bar_tokens: list[int],
    prev_bar_last_note: int | None,
    tokenizer,
    max_jump: int = 24,
) -> list[TokenConstraint]:
    """Token 级极端跳跃检测（> 24 半音 = 两个八度以上）。"""
    if prev_bar_last_note is None:
        return []

    cur_notes = []
    for t in bar_tokens:
        s = tokenizer.decode_token(t)
        if s.startswith('<Note_ON'):
            cur_notes.append(int(s[len('<Note_ON') + 1:-1]))

    if not cur_notes:
        return []

    first_note = cur_notes[0]
    if abs(first_note - prev_bar_last_note) > max_jump:
        return [TokenConstraint(
            rule='extreme_jump',
            banned_token_ids=[],
            severity='soft',
            description=f'极端跳跃 {abs(first_note - prev_bar_last_note)} 半音'
        )]

    return []


# ═══════════════════════════════════════════════════════════════
#  Score 级规则 — C 层使用（依赖 parser.py）
# ═══════════════════════════════════════════════════════════════


@dataclass
class Violation:
    """规则违反记录。"""
    rule: str
    measure: int
    voice: int | None
    severity: str         # "error" / "warning" / "suggestion"
    description: str
    details: dict = field(default_factory=dict)


def separate_voices(score) -> dict[int, list]:
    """返回 {voice_id: [(pitch, onset, duration, measure_number)]}。

    如果 MusicXML 有 voice 标记 → 直接使用。
    没有 voice 标记（或所有 voice=1）→ 按 staff + pitch 聚类。
    """
    raw: dict[int, list] = {}
    has_voice_info = False

    for m in score.measures:
        for n in m.notes:
            if n.is_rest:
                continue
            if n.voice > 1:
                has_voice_info = True
            raw.setdefault(n.voice, []).append(
                (n.pitch, n.onset, n.duration, m.number, n.staff))

    if has_voice_info:
        return raw

    # 没有 voice 信息 → 按 staff 分拆后聚类
    clustered: dict[int, list] = {}
    staff_notes: dict[int, list] = {}
    for m in score.measures:
        for n in m.notes:
            if n.is_rest or n.pitch is None:
                continue
            staff_notes.setdefault(n.staff, []).append(
                (n.pitch, n.onset, n.duration, m.number, n.staff))

    voice_id = 1
    for staff, notes in sorted(staff_notes.items()):
        if len(notes) < 4:
            clustered[voice_id] = notes
            voice_id += 1
        else:
            pitches = [n[0] for n in notes if n[0] is not None]
            if not pitches:
                clustered[voice_id] = notes
                voice_id += 1
                continue
            median = sorted(pitches)[len(pitches) // 2]
            low_voice = [n for n in notes if n[0] is not None and n[0] <= median]
            high_voice = [n for n in notes if n[0] is not None and n[0] > median]
            if low_voice and high_voice:
                clustered[voice_id] = low_voice
                clustered[voice_id + 1] = high_voice
                voice_id += 2
            else:
                clustered[voice_id] = notes
                voice_id += 1

    return clustered


def _voice_pitches_by_measure(voice_notes: list, measure_number: int) -> list[int]:
    """获取某声部在某小节的音高列表。"""
    return [n[0] for n in voice_notes if n[3] == measure_number and n[0] is not None]


# ── Score 级规则函数 ────────────────────────────────────────


def check_parallel_fifths_score(score) -> list[Violation]:
    """检测平行五度 (Score 级)。"""
    voices = separate_voices(score)
    voice_ids = sorted(voices.keys())
    violations = []

    for i in range(len(voice_ids)):
        for j in range(i + 1, len(voice_ids)):
            v1 = voices[voice_ids[i]]
            v2 = voices[voice_ids[j]]
            prev_is_fifth = False

            for m in score.measures:
                p1 = _voice_pitches_by_measure(v1, m.number)
                p2 = _voice_pitches_by_measure(v2, m.number)
                if p1 and p2:
                    interval = abs(min(p1) - min(p2)) % 12
                    is_fifth = interval == 7
                    if is_fifth and prev_is_fifth:
                        violations.append(Violation(
                            'parallel_fifth', m.number, voice_ids[i],
                            'warning', '平行五度'))
                        break
                    prev_is_fifth = is_fifth
                else:
                    prev_is_fifth = False

    return violations


def check_parallel_octaves_score(score) -> list[Violation]:
    """检测平行八度 (Score 级)。"""
    voices = separate_voices(score)
    voice_ids = sorted(voices.keys())
    violations = []

    for i in range(len(voice_ids)):
        for j in range(i + 1, len(voice_ids)):
            v1 = voices[voice_ids[i]]
            v2 = voices[voice_ids[j]]
            prev_is_octave = False

            for m in score.measures:
                p1 = _voice_pitches_by_measure(v1, m.number)
                p2 = _voice_pitches_by_measure(v2, m.number)
                if p1 and p2:
                    interval = abs(min(p1) - min(p2)) % 12
                    is_octave = interval == 0
                    if is_octave and prev_is_octave:
                        violations.append(Violation(
                            'parallel_octave', m.number, voice_ids[i],
                            'warning', '平行八度'))
                        break
                    prev_is_octave = is_octave
                else:
                    prev_is_octave = False

    return violations


def check_voice_distance_score(score) -> list[Violation]:
    """相邻声部间距 > 12 时警告。"""
    voices = separate_voices(score)
    voice_ids = sorted(voices.keys())
    violations = []

    if len(voice_ids) < 2:
        return violations

    for i in range(len(voice_ids) - 1):
        v_low = voices[voice_ids[i]]
        v_high = voices[voice_ids[i + 1]]
        for m in score.measures:
            p_low = _voice_pitches_by_measure(v_low, m.number)
            p_high = _voice_pitches_by_measure(v_high, m.number)
            if p_low and p_high:
                low_max = max(p_low)
                high_min = min(p_high)
                dist = high_min - low_max
                if dist > 12:
                    violations.append(Violation(
                        'voice_distance', m.number, voice_ids[i],
                        'warning', f'声部 {voice_ids[i]}-{voice_ids[i+1]} 间距 {dist} > 12'))

    return violations


def check_voice_crossing_score(score) -> list[Violation]:
    """检测声部交叉 (Score 级)。"""
    voices = separate_voices(score)
    voice_ids = sorted(voices.keys())
    if len(voice_ids) < 2:
        return []
    violations = []
    for m in score.measures:
        for i in range(len(voice_ids)):
            for j in range(i + 1, len(voice_ids)):
                p_low = _voice_pitches_by_measure(voices[voice_ids[i]], m.number)
                p_high = _voice_pitches_by_measure(voices[voice_ids[j]], m.number)
                if p_low and p_high:
                    if max(p_low) > min(p_high):
                        violations.append(Violation(
                            'voice_crossing', m.number, voice_ids[i],
                            'warning', f'声部 {voice_ids[i]}/{voice_ids[j]} 交叉'))
                        break
            else:
                continue
            break
    return violations


def check_leading_tone_score(score) -> list[Violation]:
    """导音解决检测。"""
    violations = []
    for m in score.measures:
        key_name = m.key_signature
        if not key_name:
            continue
        tonic = _key_to_tonic(key_name)
        if tonic is None:
            continue
        leading_tone = (tonic + 11) % 12
        pitches = [(n.pitch, n.onset) for n in m.notes
                    if not n.is_rest and n.pitch is not None]
        pitches.sort(key=lambda x: x[1])
        for i in range(len(pitches) - 1):
            p1_class = pitches[i][0] % 12
            p2_class = pitches[i + 1][0] % 12
            if p1_class == leading_tone and p2_class != tonic:
                violations.append(Violation(
                    'leading_tone_unresolved', m.number, None,
                    'suggestion', '导音未上行解决到主音'))
    return violations


def _key_to_tonic(key_name: str) -> int | None:
    """从调名获取 tonic MIDI class (0-11)。"""
    name = key_name.split('_')[0]
    key_map = {
        'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
        'E': 4, 'Fb': 4, 'F': 5, 'F#': 6, 'Gb': 6,
        'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10,
        'B': 11, 'Cb': 11,
    }
    return key_map.get(name)


# ── 全部 Score 级规则 ──────────────────────────────────────

SCORE_RULES = [
    check_parallel_fifths_score,
    check_parallel_octaves_score,
    check_voice_distance_score,
    check_voice_crossing_score,
    check_leading_tone_score,
]


def evaluate_theory(score) -> dict:
    """运行所有 Score 级理论规则。

    返回:
        {'score': float (0~1), 'violations': list[Violation], 'by_rule': dict}
    """
    all_violations = []
    for rule in SCORE_RULES:
        all_violations.extend(rule(score))

    penalty = sum(1 for v in all_violations if v.severity == 'error') * 0.15 + \
              sum(1 for v in all_violations if v.severity == 'warning') * 0.05

    score_val = max(0.0, 1.0 - penalty)

    by_rule: dict[str, list] = {}
    for v in all_violations:
        by_rule.setdefault(v.rule, []).append(v)

    return {
        'score': score_val,
        'violations': all_violations,
        'by_rule': {k: [{'measure': v.measure, 'severity': v.severity,
                          'description': v.description} for v in vals]
                    for k, vals in by_rule.items()},
        'n_violations': len(all_violations),
    }
