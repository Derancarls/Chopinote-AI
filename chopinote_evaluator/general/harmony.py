"""和声分析 — 逐拍和弦模板匹配。

支持程度: M/m/dim/aug/sus4/dom7/maj7/min7
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from chopinote_evaluator.parser import Score


# 和弦模板：根音相对于 MIDI note class 的偏移量
CHORD_TEMPLATES: dict[str, list[int]] = {
    'M':     [0, 4, 7],
    'm':     [0, 3, 7],
    'dim':   [0, 3, 6],
    'aug':   [0, 4, 8],
    'sus4':  [0, 5, 7],
    'dom7':  [0, 4, 7, 10],
    'maj7':  [0, 4, 7, 11],
    'min7':  [0, 3, 7, 10],
    'dim7':  [0, 3, 6, 9],
    'hdim7': [0, 3, 6, 10],
}


@dataclass
class Chord:
    """和弦标注。"""
    root: int           # 根音 MIDI class (0-11)
    quality: str        # "M", "m", "dim", "aug", ...
    inversion: int      # 0=原位, 1=第一转位, 2=第二转位, 3=第三转位
    onset_beat: float   # 起始拍位置
    duration: float     # 时值（四分音符）
    notes: list[int] = field(default_factory=list)  # 组成音的 MIDI class


@dataclass
class HarmonyAnalysis:
    """和声分析结果。"""
    chords: list[Chord] = field(default_factory=list)
    key_sequence: list[tuple[int, str]] = field(default_factory=list)  # [(小节号, key_name)]
    cadences: list[tuple[int, str, float]] = field(default_factory=list)  # [(小节号, type, confidence)]


def analyze_harmony(score: Score) -> HarmonyAnalysis:
    """逐拍做和弦标注。

    方法：
    1. 按拍分组（每拍取所有同时发声的音符）
    2. 对每拍的 pitch class 集合做模板匹配
    3. 输出和弦序列

    简化版本：只对强拍（每小节前 2 拍）做检测。
    """
    result = HarmonyAnalysis()

    for m in score.measures:
        beats, unit = m.time_signature
        beat_unit = 4.0 / unit  # 每拍的归一化时长

        # 每拍取一次和弦（简化：只取强拍位置）
        for beat_idx in range(min(beats, 4)):  # 最多前 4 拍
            onset_start = beat_idx * beat_unit
            onset_end = (beat_idx + 1) * beat_unit

            # 收集该拍内的所有音高
            pitch_classes = set()
            for n in m.notes:
                if not n.is_rest and n.pitch is not None:
                    if onset_start <= n.onset < onset_end or \
                       (n.onset < onset_start and n.onset + n.duration > onset_start):
                        pitch_classes.add(n.pitch % 12)

            if len(pitch_classes) >= 2:
                chord = _match_chord(pitch_classes)
                if chord:
                    chord.onset_beat = onset_start
                    chord.duration = beat_unit
                    result.chords.append(chord)

        # 记录调性
        result.key_sequence.append((m.number, m.key_signature))

    return result


def _match_chord(pitch_classes: set[int]) -> Chord | None:
    """对一组音高 class 做和弦模板匹配。

    先匹配 pitch class set（所有转位集合相同），
    再根据最低音（bass note）确定转位。
    """
    if len(pitch_classes) < 2:
        return None

    pcs_set = pitch_classes
    pcs_sorted = sorted(pcs_set)
    bass_pc = pcs_sorted[0]  # 最低音
    best_quality = 'M'
    best_root = 0
    best_score = 0.0

    for root in range(12):
        for quality, template in CHORD_TEMPLATES.items():
            # 和弦的 pitch class set（不论转位）
            expected = {(root + t) % 12 for t in template}

            # 计算匹配度
            matched = len(pcs_set & expected)
            extra = len(pcs_set - expected)
            missing = len(expected - pcs_set)

            score = matched - extra * 0.5 - missing * 0.3
            if score > best_score and matched >= len(pcs_set) - 1:
                best_score = score
                best_quality = quality
                best_root = root

    if best_score <= 0:
        return None

    # 确定转位：找到根音在 sorted template 中的位置，
    # 看哪个 template note 担当 bass
    template = CHORD_TEMPLATES[best_quality]
    # 找出 bass note 对应 template 中哪个音级
    bass_class = bass_pc
    inversion = 0
    for i, t in enumerate(template):
        if (best_root + t) % 12 == bass_class:
            inversion = i
            break

    return Chord(
        root=best_root,
        quality=best_quality,
        inversion=inversion,
        onset_beat=0.0,
        duration=0.0,
        notes=pcs_sorted,
    )
