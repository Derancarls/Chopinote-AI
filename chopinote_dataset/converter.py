"""
MusicXML → REMI 序列转换器
将双轨钢琴谱解析、按16分格对齐、转换为REMI token序列
"""
import os
from typing import List, Tuple, Optional
import logging

from music21 import converter, stream, note, chord, clef

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

    def _score_to_events(self, score) -> List[Tuple[str, Optional[int]]]:
        """music21 Score → REMI 事件列表。"""
        parts = list(score.parts)
        if not parts:
            return []

        left_idx, right_idx = self._identify_hands(parts)

        # 收集所有音符信息: (measure_idx, position, part_idx, pitch, vel_level, dur)
        all_notes: List[Tuple[int, int, int, int, int, int]] = []

        for part_idx, part in enumerate(parts):
            for measure_idx, measure in enumerate(part.getElementsByClass('Measure')):
                measure_dur = measure.duration.quarterLength
                if measure_dur <= 0:
                    continue

                positions_in_measure = max(1, int(measure_dur / self.quarter_per_position))

                for elem in measure.flatten().notesAndRests:
                    if isinstance(elem, note.Rest):
                        continue

                    offset = elem.offset  # quarter length from measure start
                    pos = min(positions_in_measure - 1,
                              int(round(offset / self.quarter_per_position)))

                    # 获取音符集合
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

        if not all_notes:
            return []

        # 按 (measure, position, part_idx) 排序 → 保证输出顺序
        all_notes.sort(key=lambda x: (x[0], x[1], x[2]))

        # 组装事件序列
        events: List[Tuple[str, Optional[int]]] = []
        cur_measure = -1
        cur_pos = -1
        cur_part = -1

        for measure_idx, pos, part_idx, pitch, vel_level, dur in all_notes:
            if measure_idx != cur_measure:
                events.append((REMITokenizer.BAR, None))
                cur_measure = measure_idx
                cur_pos = -1
                cur_part = -1

            if pos != cur_pos:
                events.append((REMITokenizer.POSITION, pos))
                cur_pos = pos
                cur_part = -1

            # 同一 position 内手切换时才 track token
            if part_idx != cur_part:
                track = REMITokenizer.TRACK_R if part_idx == right_idx else REMITokenizer.TRACK_L
                events.append((track, None))
                cur_part = part_idx

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
