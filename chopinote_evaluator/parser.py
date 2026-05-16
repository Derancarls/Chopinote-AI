"""MusicXML 解析器 — 将 MusicXML 解析为统一的 Score 中间表示。

复用 music21 做底层解析，输出 chopinote_evaluator 自己的数据结构。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import music21


# ── 数据结构 ─────────────────────────────────────────────────


@dataclass
class Note:
    """音符（或休止符）"""
    pitch: int | None           # MIDI 0-127, None=休止
    onset: float                # 从本小节开始归一化的起始拍 (四分音符=1.0)
    duration: float             # 归一化时值 (四分音符=1.0)
    duration_ticks: int         # 原始 tick 时值
    velocity: int               # 力度 0-127
    voice: int                  # 声部编号
    staff: int                  # 谱表编号 (1=右手, 2=左手)
    is_rest: bool
    is_tie_start: bool          # 连音线开始
    is_tie_stop: bool           # 连音线结束
    articulation: list[str] = field(default_factory=list)  # 演奏法标记
    tuplet_start: bool = False
    tuplet_stop: bool = False
    grace: str | None = None    # "normal" / "acciaccatura" / "appoggiatura", None=正常音符


@dataclass
class Measure:
    """小节"""
    number: int
    time_signature: tuple[int, int]  # (拍数, 分母) 如 (4,4) (3,4) (6,8)
    key_signature: str               # "C", "G", "Am", "F#", ...
    key_fifths: int                  # 调号升/降号数
    mode: str                        # "major" / "minor"
    notes: list[Note] = field(default_factory=list)
    duration_ticks: int = 0          # 本小节总 tick 数
    has_pickup: bool = False         # 是否为弱起小节


@dataclass
class Score:
    """乐谱中间表示"""
    measures: list[Measure]
    title: str = ""
    composer: str = ""
    tempo: int | None = None         # BPM
    programs: list[int] = field(default_factory=list)  # MIDI 乐器编号列表


# ── 导出函数 ─────────────────────────────────────────────────


def parse_musicxml(path: str | Path) -> Score:
    """解析 MusicXML 文件，返回 Score 对象。

    参数:
        path: MusicXML 文件路径

    返回:
        Score 对象
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    try:
        m21_score = music21.converter.parse(str(path))
    except Exception as e:
        raise ValueError(f"music21 解析失败: {e}") from e

    return _convert_score(m21_score)


def parse_musicxml_string(xml_str: str) -> Score:
    """从 MusicXML 字符串解析。

    参数:
        xml_str: MusicXML 格式字符串

    返回:
        Score 对象
    """
    try:
        m21_score = music21.converter.parse(xml_str)
    except Exception as e:
        raise ValueError(f"music21 解析失败: {e}") from e

    return _convert_score(m21_score)


def score_to_duration_seconds(score: Score) -> float:
    """计算乐谱总时长（秒），基于 tempo + 小节内容时值。"""
    tempo = score.tempo or 120  # 默认 120 BPM
    total_beats = 0.0
    for m in score.measures:
        for n in m.notes:
            total_beats += n.duration
    return total_beats / (tempo / 60)


def score_to_note_count(score: Score) -> int:
    """乐谱中非休止符的音符总数。"""
    return sum(1 for m in score.measures for n in m.notes if not n.is_rest)


# ── 内部实现 ─────────────────────────────────────────────────


def _convert_score(m21_score: music21.stream.Score) -> Score:
    """将 music21.Score 转换为内部 Score 表示。

    解析逻辑：
    1. 从 part 0 获取 tempo / 元数据
    2. 遍历所有 part，按 measure 组织音符
    3. 每个音符提取 pitch / onset / duration / voice / articulation 等信息
    """
    parts = list(m21_score.parts)
    if not parts:
        return Score(measures=[])

    # 元数据
    title = _get_metadata(m21_score, 'movementName') or _get_metadata(m21_score, 'title') or ''
    composer = _get_metadata(m21_score, 'composer') or ''

    # 速度
    tempo = _extract_tempo(m21_score)

    # 乐器
    programs = _extract_programs(parts)

    # 展开所有 part 的音符
    all_parts_notes: dict[int, list] = {}  # measure_number -> list of tuples
    measure_keys: dict[int, dict] = {}     # measure_number -> {time_sig, key_sig, ...}

    for part_idx, part in enumerate(parts):
        staff_idx = part_idx + 1  # 1-based

        # 获取拍号和调号变化
        time_sigs = _get_time_signatures(part)
        key_sigs = _get_key_signatures(part)

        current_time_sig = (4, 4)
        current_key_sig = 'C'
        current_key_fifths = 0
        current_mode = 'major'

        # 遍历小节
        for m21_measure in part.getElementsByClass(music21.stream.Measure):
            mn = m21_measure.number or 0
            if mn <= 0 and m21_measure.getDuration() == 0:
                continue  # 跳过隐式小节

            # 更新拍号和调号
            if mn in time_sigs:
                current_time_sig = time_sigs[mn]
            if mn in key_sigs:
                current_key_sig, current_key_fifths, current_mode = key_sigs[mn]

            # 初始化小节信息
            if mn not in measure_keys:
                measure_keys[mn] = {
                    'time_sig': current_time_sig,
                    'key_sig': current_key_sig,
                    'key_fifths': current_key_fifths,
                    'mode': current_mode,
                    'has_pickup': _is_pickup_measure(m21_measure),
                }

            # 处理该小节的音符
            offset_map = _build_offset_map(m21_measure)

            for m21_note in m21_measure.flatten().notesAndRests:
                if m21_note.isChord:
                    # 和弦：拆分为多个 note
                    for chord_note in m21_note.chordNotes:
                        note_data = _extract_note_data(
                            chord_note, m21_measure, staff_idx,
                            current_time_sig, offset_map
                        )
                        if note_data:
                            all_parts_notes.setdefault(mn, []).append(note_data)
                else:
                    note_data = _extract_note_data(
                        m21_note, m21_measure, staff_idx,
                        current_time_sig, offset_map
                    )
                    if note_data:
                        all_parts_notes.setdefault(mn, []).append(note_data)

    # 组装 Measure 对象（按小节号排序）
    measures = []
    for mn in sorted(measure_keys.keys()):
        mk = measure_keys[mn]
        ts = mk['time_sig']
        # 归一化时值：四分音符 = 1.0
        beat_unit = ts[1] / 4.0  # e.g. 4/4 → 1.0, 6/8 → 0.75

        measure_notes = []
        for note_data in all_parts_notes.get(mn, []):
            n = Note(
                pitch=note_data['pitch'],
                onset=note_data['onset'] * beat_unit,
                duration=note_data['duration'] * beat_unit,
                duration_ticks=note_data['duration_ticks'],
                velocity=note_data['velocity'],
                voice=note_data['voice'],
                staff=note_data['staff'],
                is_rest=note_data['is_rest'],
                is_tie_start=note_data['is_tie_start'],
                is_tie_stop=note_data['is_tie_stop'],
                articulation=note_data['articulation'],
                tuplet_start=note_data['tuplet_start'],
                tuplet_stop=note_data['tuplet_stop'],
                grace=note_data['grace'],
            )
            measure_notes.append(n)

        # 小节总 tick 数（用于拍号校验）
        duration_ticks = _measure_duration_ticks(m21_measure)

        m = Measure(
            number=mn,
            time_signature=ts,
            key_signature=mk['key_sig'],
            key_fifths=mk['key_fifths'],
            mode=mk['mode'],
            notes=measure_notes,
            duration_ticks=duration_ticks,
            has_pickup=mk['has_pickup'],
        )
        measures.append(m)

    return Score(
        measures=measures,
        title=title,
        composer=composer,
        tempo=tempo,
        programs=programs,
    )


# ── 辅助函数 ─────────────────────────────────────────────────


def _midi_pitch(n: music21.note.Note) -> int:
    """将 music21 音符转为 MIDI 音高。"""
    return n.pitch.midi


def _extract_tempo(m21_score: music21.stream.Score) -> int | None:
    """提取第一个 tempo 标记。"""
    for el in m21_score.flatten().getElementsByClass(music21.tempo.MetronomeMark):
        if el.number:
            return int(round(el.number))
    return None


def _extract_programs(parts: list) -> list[int]:
    """提取乐器 MIDI program 编号。"""
    programs = []
    for part in parts:
        for instr in part.getElementsByClass(music21.instrument.Instrument):
            if instr.midiProgram is not None:
                programs.append(instr.midiProgram)
            elif instr.midiChannel is not None:
                programs.append(instr.midiChannel)
    return programs


def _get_time_signatures(part: music21.stream.Part) -> dict[int, tuple]:
    """获取小节号 -> 拍号的映射。"""
    result = {}
    for ts in part.getElementsByClass(music21.meter.TimeSignature):
        m = ts.measureNumber
        if m is not None:
            result[m] = (ts.numerator, ts.denominator)
    return result


def _get_key_signatures(part: music21.stream.Part) -> dict[int, tuple]:
    """获取小节号 -> (调性名称, fifths, mode) 的映射。"""
    result = {}
    for ks in part.getElementsByClass(music21.key.KeySignature):
        m = ks.measureNumber
        if m is not None:
            # 转为调性名称
            key_obj = ks.asKey()
            name = key_obj.tonicPitchName  # "C", "G", "F#", ...
            mode = 'major' if key_obj.mode == 'major' else 'minor'
            # 显示完整调名如 "C major", "A minor"
            full_name = f"{name}_{mode}"
            result[m] = (full_name, ks.sharps, mode)
    return result


def _build_offset_map(m21_measure: music21.stream.Measure) -> dict[music21.note.GeneralNote, float]:
    """建立音符 -> offset 的映射（拍号归一化前）。"""
    offset_map = {}
    for el in m21_measure.flatten().notesAndRests:
        offset_map[el] = el.offset
    return offset_map


def _extract_note_data(
    m21_el: music21.note.GeneralNote,
    m21_measure: music21.stream.Measure,
    staff_idx: int,
    time_sig: tuple[int, int],
    offset_map: dict,
) -> dict | None:
    """从 music21 音符/和弦音提取内部 Note 数据。"""
    is_rest = isinstance(m21_el, music21.note.Rest)
    # 跳过未命名休止符（pickup 小节保护性休止）
    if is_rest and hasattr(m21_el, 'style') and m21_el.style.hideObjectOnPrint:
        return None

    # voice 和 staff
    voice = getattr(m21_el, 'voice', None) or 1
    if not isinstance(voice, int):
        try:
            voice = int(voice)
        except (ValueError, TypeError):
            voice = 1

    # staff 信息
    staff_info = getattr(m21_el, 'staff', None)
    if staff_info is not None:
        staff = int(staff_info)
    else:
        staff = staff_idx

    # 音高
    pitch = None
    if not is_rest:
        try:
            pitch = _midi_pitch(m21_el)
        except Exception:
            pitch = None

    # 时值（以四分音符为单位）
    qtr_duration = m21_el.duration.quarterLength

    # 起始拍（以四分音符为单位）
    onset = offset_map.get(m21_el, 0.0)

    # 力度
    velocity = 64  # 默认
    if not is_rest:
        try:
            velocity = m21_el.volume.velocity
            if velocity is None:
                velocity = 64
            else:
                velocity = int(velocity)
        except Exception:
            velocity = 64

    # 演奏法
    articulations = []
    if not is_rest:
        try:
            for art in m21_el.articulations:
                art_name = type(art).__name__.lower()
                # 标准化名称
                name_map = {
                    'staccato': 'staccato',
                    'accent': 'accent',
                    'tenuto': 'tenuto',
                    'marcato': 'marcato',
                    'strongaccent': 'accent',
                    'staccatissimo': 'staccato',
                    'spiccato': 'spiccato',
                    'articulation': 'articulation',
                }
                articulations.append(name_map.get(art_name, art_name))
        except Exception:
            pass

    # 连音线
    is_tie_start = False
    is_tie_stop = False
    if not is_rest:
        for tie in m21_el.tie or []:
            if tie.type == 'start':
                is_tie_start = True
            elif tie.type == 'stop':
                is_tie_stop = True

    # 连音
    tuplet_start = False
    tuplet_stop = False
    if hasattr(m21_el, 'duration'):
        dur = m21_el.duration
        if dur.tuplets:
            for tup in dur.tuplets:
                if tup.type == 'start':
                    tuplet_start = True
                elif tup.type == 'stop':
                    tuplet_stop = True

    # 装饰音
    grace = None
    if hasattr(m21_el, 'duration') and m21_el.duration.isGrace:
        grace = 'acciaccatura' if getattr(m21_el, 'slash', False) else 'appoggiatura'

    beat_unit = time_sig[1] / 4.0

    return {
        'pitch': pitch,
        'onset': onset / beat_unit,
        'duration': qtr_duration / beat_unit,
        'duration_ticks': int(qtr_duration * 480 * beat_unit),  # 近似 ticks
        'velocity': velocity,
        'voice': voice,
        'staff': staff,
        'is_rest': is_rest,
        'is_tie_start': is_tie_start,
        'is_tie_stop': is_tie_stop,
        'articulation': articulations,
        'tuplet_start': tuplet_start,
        'tuplet_stop': tuplet_stop,
        'grace': grace,
    }


def _is_pickup_measure(m21_measure: music21.stream.Measure) -> bool:
    """检测是否为弱起小节。"""
    return getattr(m21_measure, 'isPickup', False) or (m21_measure.duration.quarterLength > 0 and
                                                        m21_measure.paddingLeft is not None)


def _measure_duration_ticks(m21_measure: music21.stream.Measure) -> int:
    """获取小节总 tick 数。"""
    dur = m21_measure.duration
    return int(dur.quarterLength * 480)


def _get_metadata(m21_score: music21.stream.Score, key: str) -> str:
    """从 metadata 中提取字段。"""
    try:
        md = m21_score.metadata
        if md is None:
            return ''
        val = getattr(md, key, None)
        if callable(val):
            val = val()
        return str(val) if val else ''
    except Exception:
        return ''
