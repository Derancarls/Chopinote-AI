"""音乐理论规则引擎 — 声部分离、声部进行检查、调性检测。

规则:
- 平行五度/八度
- 声部超距（相邻声部 > 12 半音）
- 导音解决（vii → I）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from chopinote_evaluator.parser import Score


@dataclass
class Violation:
    """规则违反记录。"""
    rule: str
    measure: int
    voice: int | None
    severity: str         # "error" / "warning" / "suggestion"
    description: str
    details: dict = field(default_factory=dict)


class TheoryEvaluator:
    """理论规则评估器。"""

    def __init__(self, rules: list | None = None):
        self.rules = rules or self._default_rules()

    def evaluate(self, score: Score) -> dict:
        """运行所有规则。

        返回:
            {'score': float (0~1), 'violations': list[Violation], 'by_rule': dict}
        """
        all_violations = []
        for rule in self.rules:
            violations = rule(score)
            all_violations.extend(violations)

        # 分数：每有一个 vilation 扣分
        n_measures = max(len(score.measures), 1)
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

    @staticmethod
    def _default_rules() -> list[Callable]:
        return [
            check_parallel_fifths,
            check_parallel_octaves,
            check_hidden_fifths,
            check_hidden_octaves,
            check_voice_distance,
            check_voice_crossing,
            check_leading_tone,
            check_tritone_resolution,
            check_cross_relation,
        ]


# ── 声部分离 ────────────────────────────────────────────────


def separate_voices(score: Score) -> dict[int, list]:
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
            raw.setdefault(n.voice, []).append((n.pitch, n.onset, n.duration, m.number, n.staff))

    if has_voice_info:
        return raw

    # 没有 voice 信息 → 按 staff 分拆后聚类
    clustered: dict[int, list] = {}
    staff_notes: dict[int, list] = {}
    for m in score.measures:
        for n in m.notes:
            if n.is_rest or n.pitch is None:
                continue
            staff_notes.setdefault(n.staff, []).append((n.pitch, n.onset, n.duration, m.number, n.staff))

    voice_id = 1
    for staff, notes in sorted(staff_notes.items()):
        if len(notes) < 4:
            # 太少音符，全部分配给一个 voice
            clustered[voice_id] = notes
            voice_id += 1
        else:
            # 按音高分为两个声部
            pitches = [n[0] for n in notes]
            try:
                from sklearn.cluster import KMeans
                import numpy as np
                X = np.array(pitches).reshape(-1, 1)
                kmeans = KMeans(n_clusters=min(2, len(pitches)), n_init=1, random_state=42)
                labels = kmeans.fit_predict(X)

                low_voice = [n for i, n in enumerate(notes) if labels[i] == 0]
                high_voice = [n for i, n in enumerate(notes) if labels[i] == 1]

                # 确保 low_voice 的音高低于 high_voice
                if low_voice and high_voice and np.mean([n[0] for n in low_voice]) > np.mean([n[0] for n in high_voice]):
                    low_voice, high_voice = high_voice, low_voice

                clustered[voice_id] = low_voice
                clustered[voice_id + 1] = high_voice
                voice_id += 2
            except ImportError:
                # sklearn 不可用 → 用简单中位数分割
                pitches = [n[0] for n in notes if n[0] is not None]
                if pitches:
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
                else:
                    clustered[voice_id] = notes
                    voice_id += 1

    return clustered


def _voice_pitches_by_measure(voice_notes: list, measure_number: int) -> list[int]:
    """获取某声部在某小节的音高列表。"""
    return [n[0] for n in voice_notes if n[3] == measure_number and n[0] is not None]


# ── 规则实现 ────────────────────────────────────────────────


def check_parallel_fifths(score: Score) -> list[Violation]:
    """检测平行五度：两声部间连续出现纯五度关系。"""
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
                    # 取各自最低音计算音程
                    interval = abs(min(p1) - min(p2)) % 12
                    is_fifth = interval == 7  # 纯五度

                    if is_fifth and prev_is_fifth:
                        violations.append(Violation(
                            'parallel_fifth', m.number, voice_ids[i],
                            'warning', '平行五度'))
                        break
                    prev_is_fifth = is_fifth
                else:
                    prev_is_fifth = False

    return violations


def check_parallel_octaves(score: Score) -> list[Violation]:
    """检测平行八度。"""
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
                    is_octave = interval == 0  # 纯八度/同度

                    if is_octave and prev_is_octave:
                        violations.append(Violation(
                            'parallel_octave', m.number, voice_ids[i],
                            'warning', '平行八度'))
                        break
                    prev_is_octave = is_octave
                else:
                    prev_is_octave = False

    return violations


def check_voice_distance(score: Score) -> list[Violation]:
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
                # 低声部最高音 vs 高声部最低音
                low_max = max(p_low)
                high_min = min(p_high)
                dist = high_min - low_max

                if dist > 12:
                    violations.append(Violation(
                        'voice_distance', m.number, voice_ids[i],
                        'warning',
                        f'声部 {voice_ids[i]}-{voice_ids[i+1]} 间距 {dist} > 12'))

    return violations


def check_leading_tone(score: Score) -> list[Violation]:
    """导音解决检测（大调 vii 级 → I 级）。

    简化版本：在乐谱中找 vii → I 的半音上行模式。"""
    violations = []

    for m in score.measures:
        key_name = m.key_signature
        if not key_name:
            continue

        # 从调名获取 tonic
        tonic = _key_to_tonic(key_name)
        if tonic is None:
            continue

        leading_tone = (tonic + 11) % 12  # 导音 = tonic 下方半音

        # 在该小节找导音 → 主音的连续
        pitchn = [(n.pitch, n.onset) for n in m.notes
                   if not n.is_rest and n.pitch is not None]
        pitchn.sort(key=lambda x: x[1])

        for i in range(len(pitchn) - 1):
            p1_class = pitchn[i][0] % 12
            p2_class = pitchn[i + 1][0] % 12
            if p1_class == leading_tone and p2_class == tonic:
                # 导音解决到主音 — 这是好的，不报 violation
                pass
            elif p1_class == leading_tone and p2_class != tonic:
                # 导音未解决到主音 → 建议性警告
                 violations.append(Violation(
                    'leading_tone_unresolved', m.number, None,
                    'suggestion', '导音未上行解决到主音'))

    return violations


def check_hidden_fifths(score: Score) -> list[Violation]:
    """检测隐伏五度：外声部同向进行到纯五度且高声部跳进。"""
    voices = separate_voices(score)
    voice_ids = sorted(voices.keys())
    if len(voice_ids) < 2:
        return []

    violations = []
    v_low = voices[voice_ids[0]]   # 最低声部
    v_high = voices[voice_ids[-1]]  # 最高声部

    for m_idx in range(len(score.measures) - 1):
        m1 = score.measures[m_idx]
        m2 = score.measures[m_idx + 1]

        p_low1 = _voice_pitches_by_measure(v_low, m1.number)
        p_high1 = _voice_pitches_by_measure(v_high, m1.number)
        p_low2 = _voice_pitches_by_measure(v_low, m2.number)
        p_high2 = _voice_pitches_by_measure(v_high, m2.number)

        if not (p_low1 and p_high1 and p_low2 and p_high2):
            continue

        low1, low2 = min(p_low1), min(p_low2)
        high1, high2 = max(p_high1), max(p_high2)

        # 前一音程不是五度，后一音程是五度
        if abs(low1 - high1) % 12 != 7 and abs(low2 - high2) % 12 == 7:
            # 同向进行
            low_dir = low2 - low1
            high_dir = high2 - high1
            if (low_dir > 0 and high_dir > 0) or (low_dir < 0 and high_dir < 0):
                # 高声部跳进 (>2 半音)
                if abs(high2 - high1) > 2:
                    violations.append(Violation(
                        'hidden_fifth', m2.number, voice_ids[-1],
                        'suggestion', '隐伏五度（外声部同向进行到纯五度）'))

    return violations


def check_hidden_octaves(score: Score) -> list[Violation]:
    """检测隐伏八度：外声部同向进行到纯八度且高声部跳进。"""
    voices = separate_voices(score)
    voice_ids = sorted(voices.keys())
    if len(voice_ids) < 2:
        return []

    violations = []
    v_low = voices[voice_ids[0]]
    v_high = voices[voice_ids[-1]]

    for m_idx in range(len(score.measures) - 1):
        m1 = score.measures[m_idx]
        m2 = score.measures[m_idx + 1]

        p_low1 = _voice_pitches_by_measure(v_low, m1.number)
        p_high1 = _voice_pitches_by_measure(v_high, m1.number)
        p_low2 = _voice_pitches_by_measure(v_low, m2.number)
        p_high2 = _voice_pitches_by_measure(v_high, m2.number)

        if not (p_low1 and p_high1 and p_low2 and p_high2):
            continue

        low1, low2 = min(p_low1), min(p_low2)
        high1, high2 = max(p_high1), max(p_high2)

        if abs(low1 - high1) % 12 != 0 and abs(low2 - high2) % 12 == 0:
            low_dir = low2 - low1
            high_dir = high2 - high1
            if (low_dir > 0 and high_dir > 0) or (low_dir < 0 and high_dir < 0):
                if abs(high2 - high1) > 2:
                    violations.append(Violation(
                        'hidden_octave', m2.number, voice_ids[-1],
                        'suggestion', '隐伏八度（外声部同向进行到纯八度）'))

    return violations


def check_voice_crossing(score: Score) -> list[Violation]:
    """检测声部交叉：低声部某音高于高声部某音。"""
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


def check_tritone_resolution(score: Score) -> list[Violation]:
    """检测增四度（三全音）是否正确解决。

    增四度应向外解决到六度或向内解决到三度。
    """
    violations = []
    for m_idx in range(len(score.measures) - 1):
        m1 = score.measures[m_idx]
        m2 = score.measures[m_idx + 1]

        pitchn1 = [(n.pitch, n.onset) for n in m1.notes
                    if not n.is_rest and n.pitch is not None]
        pitchn2 = [(n.pitch, n.onset) for n in m2.notes
                    if not n.is_rest and n.pitch is not None]

        if not pitchn1 or not pitchn2:
            continue

        # 在 m1 中找三全音关系
        for i in range(len(pitchn1)):
            for j in range(i + 1, len(pitchn1)):
                interval = abs(pitchn1[i][0] - pitchn1[j][0]) % 12
                if interval == 6:  # 增四度
                    # 在 m2 中找这两个音级的解决
                    p1_next = [p for p in pitchn2 if p[0] % 12 == (pitchn1[i][0] + 1) % 12
                               or p[0] % 12 == (pitchn1[i][0] - 1) % 12]
                    p2_next = [p for p in pitchn2 if p[0] % 12 == (pitchn1[j][0] + 1) % 12
                               or p[0] % 12 == (pitchn1[j][0] - 1) % 12]
                    if not (p1_next or p2_next):
                        violations.append(Violation(
                            'tritone_unresolved', m1.number, None,
                            'suggestion', '增四度未解决'))

    return violations


def check_cross_relation(score: Score) -> list[Violation]:
    """检测交错关系：同一音级在不同声部先后出现变化半音。

    例如：女高音唱 F#，紧接着男高音唱 F 自然。
    """
    violations = []
    for m in score.measures:
        pc_voices: dict[int, dict] = {}  # pitch_class -> {staff: [pitch, onset]}
        for n in m.notes:
            if n.is_rest or n.pitch is None:
                continue
            pc = n.pitch % 12
            if pc not in pc_voices:
                pc_voices[pc] = {}
            staff_s = str(n.staff)
            if staff_s not in pc_voices[pc]:
                pc_voices[pc][staff_s] = []
            pc_voices[pc][staff_s].append((n.pitch, n.onset))

        for pc, staff_map in pc_voices.items():
            if len(staff_map) >= 2:
                # 不同声部出现同一音级但有变化半音（natural/sharp/flat）
                pitches_seen = set()
                for notes in staff_map.values():
                    for p, _ in notes:
                        pitches_seen.add(p)
                if len(pitches_seen) > 1:
                    violations.append(Violation(
                        'cross_relation', m.number, None,
                        'warning',
                        f'交错关系：音级 {["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"][pc]} '
                        f'出现变化半音'))

    return violations


def _key_to_tonic(key_name: str) -> int | None:
    """从调名获取 tonic MIDI class (0-11)。"""
    # key_name 格式: "C", "G_major", "A_minor", "Bb_major", "F#_minor"
    name = key_name.split('_')[0]  # 取调名部分
    map = {
        'C': 0, 'C#': 1, 'Db': 1,
        'D': 2, 'D#': 3, 'Eb': 3,
        'E': 4, 'Fb': 4,
        'F': 5, 'F#': 6, 'Gb': 6,
        'G': 7, 'G#': 8, 'Ab': 8,
        'A': 9, 'A#': 10, 'Bb': 10,
        'B': 11, 'Cb': 11,
    }
    return map.get(name)
