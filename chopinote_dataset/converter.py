"""
MusicXML → REMI 序列转换器
将双轨钢琴谱解析、按16分格对齐、转换为REMI token序列
"""
import os
from typing import List, Tuple, Optional
import logging

from music21 import (
    converter, stream, note, chord, clef,
    dynamics, tempo, bar, expressions, spanner, repeat,
)

from .tokenizer import REMITokenizer

logger = logging.getLogger(__name__)


class MusicXMLToREMI:
    """MusicXML → REMI 转换器

    解析双轨钢琴谱（左右手），按 grid 对齐并生成 REMI token 序列。
    """

    def __init__(self, grid_size: int = 16, velocity_levels: int = 8):
        self.grid_size = grid_size
        self.velocity_levels = velocity_levels
        self.quarter_per_position = 4.0 / grid_size
        self.tokenizer = REMITokenizer(grid_size, velocity_levels)

    def convert(self, file_path: str, collect_metadata: bool = False
                ) -> Tuple[List[int], dict]:
        """转换一个 MusicXML 文件为 token ID 序列。

        Returns:
            (token_ids, metadata)  — token_ids 为空时 metadata 也为空
        """
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return [], {}

        try:
            score = converter.parse(file_path)
        except Exception as e:
            logger.error(f"解析 MusicXML 失败 {file_path}: {e}")
            return [], {}

        return self._convert_score(score, file_path, collect_metadata)

    def convert_score(self, score, collect_metadata: bool = False
                      ) -> Tuple[List[int], dict]:
        """直接转换已解析的 music21 Score 对象。"""
        return self._convert_score(score, None, collect_metadata)

    # ── internal ──────────────────────────────────────────────

    def _convert_score(self, score, file_path: str | None = None,
                       collect_metadata: bool = False) -> Tuple[List[int], dict]:
        events = self._score_to_events(score)
        if not events:
            return [], {}

        full_events = [(REMITokenizer.BOS, None)] + events + [(REMITokenizer.EOS, None)]
        token_ids = self.tokenizer.tokenize(full_events)

        metadata = {}
        if collect_metadata:
            metadata = {
                'grid_size': self.grid_size,
                'velocity_levels': self.velocity_levels,
                'num_events': len(events),
                'num_measures': sum(1 for e in events if e[0] == REMITokenizer.BAR),
            }
        return token_ids, metadata

    def _clef_name(self, c) -> str:
        """Clef 对象 → 规范名称 (treble/bass/alto/tenor)。"""
        sign = c.sign.lower()
        line = c.line if c.line is not None else 2
        if sign == 'g':
            return 'treble'
        elif sign == 'f':
            return 'bass'
        elif sign == 'c':
            return 'alto' if line == 3 else 'tenor' if line == 4 else f'c_{line}'
        elif sign == 'percussion':
            return 'percussion'
        return sign

    def _score_to_events(self, score) -> List[Tuple[str, Optional[int]]]:
        """music21 Score → REMI 事件列表。"""
        parts = list(score.parts)
        if not parts:
            return []

        left_idx, right_idx = self._identify_hands(parts)

        # 收集两类事件:
        #   events: (measure, pos, part_idx, token_type, value) — 非音符标记
        #   notes:  (measure, pos, part_idx, pitch, vel, dur) — 音符
        extra: List[Tuple[int, int, int, str, Optional[int]]] = []
        all_notes: List[Tuple[int, int, int, int, int, int]] = []

        for part_idx, part in enumerate(parts):
            for measure_idx, measure in enumerate(part.getElementsByClass('Measure')):
                measure_dur = measure.duration.quarterLength
                if measure_dur <= 0:
                    continue

                positions_in_measure = max(1, int(measure_dur / self.quarter_per_position))
                flat = measure.flatten()

                # ── 非音符标记提取 ──────────────────────────
                for elem in flat:
                    pos = min(positions_in_measure - 1,
                              int(round(elem.offset / self.quarter_per_position)))

                    # 谱号
                    if isinstance(elem, clef.Clef):
                        extra.append((measure_idx, pos, part_idx,
                                      REMITokenizer.CLEF, self._clef_name(elem)))

                    # 力度记号 (pp, ff 等)
                    elif isinstance(elem, dynamics.Dynamic):
                        extra.append((measure_idx, pos, part_idx,
                                      REMITokenizer.DYNAMIC, elem.value))

                    # 渐强/渐弱记号 (hairpin)
                    elif isinstance(elem, dynamics.Crescendo):
                        extra.append((measure_idx, pos, part_idx,
                                      REMITokenizer.HAIRPIN, 'cresc'))
                    elif isinstance(elem, dynamics.Diminuendo):
                        extra.append((measure_idx, pos, part_idx,
                                      REMITokenizer.HAIRPIN, 'dim'))

                    # 速度标记
                    elif isinstance(elem, tempo.MetronomeMark):
                        if elem.number is None:
                            continue  # 跳过文字型速度标记 (如 "Andante")
                        bpm = int(round(max(30, min(240, elem.number))))
                        bpm = (bpm // 10) * 10  # 量化到 10 的倍数
                        extra.append((measure_idx, 0, part_idx,
                                      REMITokenizer.TEMPO, bpm))

                    # 反复小节线 (bar.Repeat 是 Barline 子类)
                    elif isinstance(elem, bar.Repeat):
                        extra.append((measure_idx, 0, part_idx,
                                      REMITokenizer.REPEAT, elem.direction))

                    # 普通小节线（跳过）
                    elif isinstance(elem, bar.Barline):
                        pass

                    # 跳转标记 (Da Capo / Dal Segno / Segno / Coda / Fine)
                    elif isinstance(elem, repeat.DaCapo):
                        extra.append((measure_idx, 0, part_idx, REMITokenizer.JUMP, 'da_capo'))
                    elif isinstance(elem, repeat.DalSegno):
                        extra.append((measure_idx, 0, part_idx, REMITokenizer.JUMP, 'dal_segno'))
                    elif isinstance(elem, repeat.Segno):
                        extra.append((measure_idx, 0, part_idx, REMITokenizer.JUMP, 'segno'))
                    elif isinstance(elem, repeat.Coda):
                        extra.append((measure_idx, 0, part_idx, REMITokenizer.JUMP, 'coda'))
                    elif isinstance(elem, repeat.Fine):
                        extra.append((measure_idx, 0, part_idx, REMITokenizer.JUMP, 'fine'))

                    # 反复括号 (Volta)
                    elif isinstance(elem, spanner.RepeatBracket):
                        extra.append((measure_idx, 0, part_idx,
                                      REMITokenizer.REPEAT, f'volta_{elem.number}'))

                    # 踏板
                    elif isinstance(elem, expressions.PedalMark):
                        ped_type = 'start' if getattr(elem, 'type', 0) in (0, 'start') else 'end'
                        extra.append((measure_idx, pos, part_idx,
                                      REMITokenizer.PEDAL, ped_type))

                    # 连奏线
                    elif isinstance(elem, spanner.Slur):
                        slur_type = 'start' if getattr(elem, 'type', 'start') == 'start' else 'end'
                        extra.append((measure_idx, pos, part_idx,
                                      REMITokenizer.SLUR, slur_type))

                    # 延长记号
                    elif isinstance(elem, expressions.Fermata):
                        extra.append((measure_idx, pos, part_idx,
                                      REMITokenizer.ARTIC, 'fermata'))

                # ── 音符提取 ────────────────────────────────
                for elem in flat.notesAndRests:
                    if isinstance(elem, note.Rest):
                        continue

                    offset = elem.offset
                    pos = min(positions_in_measure - 1,
                              int(round(offset / self.quarter_per_position)))

                    # 音符上的演奏法标记
                    for art in elem.articulations:
                        art_name = art.__class__.__name__.lower()
                        if art_name in ('staccato', 'accent', 'tenuto', 'marcato', 'pizzicato'):
                            extra.append((measure_idx, pos, part_idx,
                                          REMITokenizer.ARTIC, art_name))

                    # 音符上的装饰音
                    for expr in getattr(elem, 'expressions', []):
                        exp_name = expr.__class__.__name__.lower()
                        if exp_name in ('trill', 'mordent', 'turn', 'tremolo'):
                            extra.append((measure_idx, pos, part_idx,
                                          REMITokenizer.ORNAMENT, exp_name))

                    # 获取音高
                    if isinstance(elem, note.Note):
                        pitches = [elem.pitch.midi]
                    elif isinstance(elem, chord.Chord):
                        pitches = [n.pitch.midi for n in elem.notes]
                    else:
                        continue

                    vel = int(elem.volume.velocity) if elem.volume.velocity else 64
                    vel_level = min(self.velocity_levels - 1,
                                    vel // (128 // self.velocity_levels))

                    dur_positions = max(1, min(self.grid_size,
                                               int(round(elem.quarterLength / self.quarter_per_position))))

                    for p in pitches:
                        all_notes.append((measure_idx, pos, part_idx, p, vel_level, dur_positions))

        if not all_notes and not extra:
            return []

        # 合并两类事件，按 (measure, position, part_idx) 排序
        merged: List[Tuple[int, int, int, str, tuple]] = []
        for m_idx, pos, p_idx, ttype, val in extra:
            merged.append((m_idx, pos, p_idx, 'x', (ttype, val)))
        for m_idx, pos, p_idx, p, vl, dr in all_notes:
            merged.append((m_idx, pos, p_idx, 'n', (p, vl, dr)))

        merged.sort(key=lambda x: (x[0], x[1], x[2]))

        # 组装事件序列
        events: List[Tuple[str, Optional[int]]] = []
        cur_measure = -1
        cur_pos = -1
        cur_part = -1

        for m, pos, p_idx, kind, data in merged:
            if m != cur_measure:
                events.append((REMITokenizer.BAR, None))
                cur_measure = m
                cur_pos = -1
                cur_part = -1

            if pos != cur_pos:
                events.append((REMITokenizer.POSITION, pos))
                cur_pos = pos
                cur_part = -1

            if p_idx != cur_part:
                track = REMITokenizer.TRACK_R if p_idx == right_idx else REMITokenizer.TRACK_L
                events.append((track, None))
                cur_part = p_idx

            if kind == 'x':
                ttype, val = data
                events.append((ttype, val))
            else:  # 'n' — note
                pitch, vel_level, dur = data
                events.append((REMITokenizer.NOTE_ON, pitch))
                events.append((REMITokenizer.VELOCITY, vel_level))
                events.append((REMITokenizer.DURATION, dur))

        return events

    def _identify_hands(self, parts) -> Tuple[int, int]:
        """返回 (left_hand_idx, right_hand_idx)。

        优先根据谱号判断，回退为 first=right, second=left。
        """
        right_idx = 0
        left_idx = 1 if len(parts) > 1 else 0

        for i, part in enumerate(parts):
            try:
                for c in part.flatten().getElementsByClass(clef.Clef):
                    if 'treble' in c.sign.lower():
                        right_idx = i
                    elif 'bass' in c.sign.lower():
                        left_idx = i
            except Exception:
                continue

        return left_idx, right_idx
