"""
MusicXML / PDMX → REMI 序列转换器
将乐谱解析、按 grid 对齐、转换为 REMI token 序列
"""
import os
import json
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict
import logging

from music21 import (
    converter, stream, note, chord, clef, meter,
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

        # ── 调号提取 ──────────────────────────────────────────
        from music21 import key
        key_name: Optional[str] = None
        key_objs = score.flatten().getElementsByClass(key.Key)
        if key_objs:
            k = key_objs[0]
            key_name = k.tonic.name + ('m' if k.mode == 'minor' else '')
        # ──────────────────────────────────────────────────────

        left_idx, right_idx = self._identify_hands(parts)

        # 收集两类事件:
        #   events: (measure, pos, part_idx, token_type, value) — 非音符标记
        #   notes:  (measure, pos, part_idx, pitch, vel, dur) — 音符
        extra: List[Tuple[int, int, int, str, Optional[int]]] = []
        all_notes: List[Tuple[int, int, int, int, int, int]] = []
        all_rests: List[Tuple[int, int, int, int]] = []       # (m, pos, part, dur)
        all_grace_notes: List[Tuple[int, int, int, str, int, int, int]] = []  # (m, pos, part, gtype, pitch, vel, dur)
        last_timesig: Optional[str] = None

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

                    # 拍号
                    elif isinstance(elem, meter.TimeSignature):
                        ts_str = f'{elem.numerator}/{elem.denominator}'
                        if ts_str in REMITokenizer.TIME_SIGNATURES and ts_str != last_timesig:
                            extra.append((measure_idx, 0, part_idx,
                                          REMITokenizer.TIMESIG, ts_str))
                            last_timesig = ts_str

                # ── 音符提取 ────────────────────────────────
                current_tuplet: Optional[str] = None
                last_tuplet_pos: int = 0
                for elem in flat.notesAndRests:
                    if isinstance(elem, note.Rest):
                        pos = min(positions_in_measure - 1,
                                  int(round(elem.offset / self.quarter_per_position)))
                        dur_positions = max(1, min(self.grid_size,
                                                   int(round(elem.quarterLength / self.quarter_per_position))))
                        all_rests.append((measure_idx, pos, part_idx, dur_positions))
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

                    # 倚音/装饰音检测：elem.duration.isGrace 为 True
                    is_grace = getattr(elem.duration, 'isGrace', False)
                    if is_grace:
                        gn_type = 'acciaccatura' if getattr(elem.duration, 'slash', False) else 'appoggiatura'
                        for p in pitches:
                            all_grace_notes.append((measure_idx, pos, part_idx, gn_type,
                                                    p, vel_level, 1))
                        continue  # 倚音不进入 all_notes

                    for p in pitches:
                        all_notes.append((measure_idx, pos, part_idx, p, vel_level, dur_positions))

                    # Tuplet 检测
                    tuplet_key = None
                    if elem.duration.tuplets:
                        t = elem.duration.tuplets[0]
                        tuplet_key = f'{t.numberNotesActual}:{t.numberNotesNormal}'
                        if tuplet_key not in REMITokenizer.TUPLET_RATIOS:
                            tuplet_key = None

                    if tuplet_key != current_tuplet:
                        if current_tuplet is not None:
                            extra.append((measure_idx, last_tuplet_pos, part_idx,
                                          REMITokenizer.TUPLET_END, None))
                        if tuplet_key is not None:
                            extra.append((measure_idx, pos, part_idx,
                                          REMITokenizer.TUPLET_START, tuplet_key))
                        current_tuplet = tuplet_key

                    if tuplet_key is not None:
                        last_tuplet_pos = pos

                if current_tuplet is not None:
                    extra.append((measure_idx, last_tuplet_pos, part_idx,
                                  REMITokenizer.TUPLET_END, None))

        if not all_notes and not extra and not all_rests and not all_grace_notes:
            return []

        # 合并各类事件，按 (measure, position, program, subtrack, priority) 排序
        # priority: 0=TimeSig, 1=extra, 1.5=GraceNote, 2=note, 2.5=Rest, 3=TupletEnd
        merged: List[Tuple[int, int, int, int, float, str, tuple]] = []

        def _hand_to_ps(p: int) -> Tuple[int, int]:
            return (0, 0) if p == right_idx else (0, 1)

        for m_idx, pos, p_idx, ttype, val in extra:
            if ttype == REMITokenizer.TUPLET_END:
                priority = 3
            elif ttype == REMITokenizer.TIMESIG:
                priority = 0
            else:
                priority = 1
            prog, sub = _hand_to_ps(p_idx)
            merged.append((m_idx, pos, prog, sub, priority, 'x', (ttype, val)))
        for m_idx, pos, p_idx, p, vl, dr in all_notes:
            prog, sub = _hand_to_ps(p_idx)
            merged.append((m_idx, pos, prog, sub, 2, 'n', (p, vl, dr)))
        for m_idx, pos, p_idx, dr in all_rests:
            prog, sub = _hand_to_ps(p_idx)
            merged.append((m_idx, pos, prog, sub, 2.5, 'r', (dr,)))
        for m_idx, pos, p_idx, gn_type, gn_pitch, gn_vel, gn_dur in all_grace_notes:
            prog, sub = _hand_to_ps(p_idx)
            merged.append((m_idx, pos, prog, sub, 1.5, 'g', (gn_type, gn_pitch, gn_vel, gn_dur)))

        merged.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))

        # 组装事件序列
        events: List[Tuple[str, Optional[int]]] = []
        cur_measure = -1
        cur_pos = -1
        cur_program = -1
        cur_subtrack = 0

        for m, pos, prog, sub, _priority, kind, data in merged:
            if m != cur_measure:
                events.append((REMITokenizer.BAR, None))
                cur_measure = m
                cur_pos = -1
                cur_program = -1

            if pos != cur_pos:
                events.append((REMITokenizer.POSITION, pos))
                cur_pos = pos
                cur_program = -1

            if prog != cur_program or sub != cur_subtrack:
                if sub == 0:
                    events.append((REMITokenizer.PROGRAM, str(prog)))
                else:
                    events.append((REMITokenizer.PROGRAM, f'{prog}_{sub}'))
                cur_program = prog
                cur_subtrack = sub

            if kind == 'x':
                ttype, val = data
                events.append((ttype, val))
            elif kind == 'g':  # grace note
                gn_type, pitch, vel_level, dur = data
                events.append((REMITokenizer.GRACE_NOTE, gn_type))
                events.append((REMITokenizer.NOTE_ON, pitch))
                events.append((REMITokenizer.VELOCITY, vel_level))
                events.append((REMITokenizer.DURATION, dur))
            elif kind == 'r':  # rest
                (dur,) = data
                events.append((REMITokenizer.REST, None))
                events.append((REMITokenizer.DURATION, dur))
            else:  # 'n' — note
                pitch, vel_level, dur = data
                events.append((REMITokenizer.NOTE_ON, pitch))
                events.append((REMITokenizer.VELOCITY, vel_level))
                events.append((REMITokenizer.DURATION, dur))

        # Key token 作为第一个内容事件（在 _convert_score 中紧跟 BOS）
        if key_name:
            events.insert(0, (REMITokenizer.KEY, key_name))
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


class PDMXToREMI:
    """PDMX JSON → REMI 转换器

    读取 PDMX (MusicRender) JSON 格式，输出与 MusicXMLToREMI 相同的
    REMI 事件序列，兼容下游 tokenizer 和训练管线。
    """

    def __init__(self, grid_size: int = 16, velocity_levels: int = 8):
        self.grid_size = grid_size
        self.velocity_levels = velocity_levels
        self.quarter_per_position = 4.0 / grid_size
        self.tokenizer = REMITokenizer(grid_size, velocity_levels)

    def convert(self, file_path: str, collect_metadata: bool = False
                ) -> Tuple[List[int], dict]:
        """加载 PDMX JSON 文件并转换为 token ID 序列。"""
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return [], {}

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                pdmx_data = json.load(f)
        except Exception as e:
            logger.error(f"解析 PDMX JSON 失败 {file_path}: {e}")
            return [], {}

        return self._convert_pdmx(pdmx_data, file_path, collect_metadata)

    def convert_pdmx(self, pdmx_data: dict, collect_metadata: bool = False
                     ) -> Tuple[List[int], dict]:
        """直接转换已解析的 PDMX dict 对象。"""
        return self._convert_pdmx(pdmx_data, None, collect_metadata)

    # ── internal ──────────────────────────────────────────────

    def _convert_pdmx(self, data: dict, file_path: str | None = None,
                      collect_metadata: bool = False) -> Tuple[List[int], dict]:
        events = self._pdmx_to_events(data)
        if not events:
            return [], {}

        full_events = [(REMITokenizer.BOS, None)] + events + [(REMITokenizer.EOS, None)]
        token_ids = self.tokenizer.tokenize(full_events)

        metadata = {}
        if collect_metadata:
            md = data.get('metadata', {})
            metadata = {
                'grid_size': self.grid_size,
                'velocity_levels': self.velocity_levels,
                'num_events': len(events),
                'num_measures': sum(1 for e in events if e[0] == REMITokenizer.BAR),
                'title': md.get('title', ''),
                'creators': md.get('creators', []),
                'source_format': md.get('source_format', ''),
                'num_tracks': len(data.get('tracks', [])),
            }
        return token_ids, metadata

    def _pdmx_to_events(self, data: dict) -> List[Tuple[str, Optional[int]]]:
        """PDMX MusicRender dict → REMI 事件列表。"""
        resolution = data.get('resolution', 480)
        ticks_per_pos = resolution * self.quarter_per_position  # 120 @ 480/res

        # ── 调号提取 ──────────────────────────────────────────
        key_name: Optional[str] = None
        ks_events = data.get('key_signatures', [])
        if ks_events:
            first = ks_events[0]
            key_name = first.get('root_str', 'C') + ('m' if first.get('mode') == 'minor' else '')
        # ──────────────────────────────────────────────────────

        # ── 1. 构建小节起止映射 ──────────────────────────────
        barlines = data.get('barlines', [])
        if not barlines:
            logger.warning("PDMX 数据缺少 barlines，无法转换")
            return []

        # measure_0based -> start_time
        measure_starts: Dict[int, int] = {}
        for bl in barlines:
            measure_starts[bl['measure'] - 1] = bl['time']

        # ── 2. 构建拍号映射 ──────────────────────────────────
        ts_map: Dict[int, Tuple[int, int]] = {}
        for ts in data.get('time_signatures', []):
            ts_map[ts['measure'] - 1] = (ts['numerator'], ts['denominator'])

        # 确定总小节数
        max_measure = max(
            max(measure_starts.keys()),
            max((n['measure'] - 1 for t in data.get('tracks', [])
                 for n in t.get('notes', [])), default=0),
        )

        # 向前传播拍号
        measure_ts: Dict[int, str] = {}
        cur_num, cur_den = 4, 4
        for m in range(max_measure + 1):
            if m in ts_map:
                cur_num, cur_den = ts_map[m]
            measure_ts[m] = f'{cur_num}/{cur_den}'

        # ── 3. 收集各声部事件 ────────────────────────────────
        # 构建 part_idx → (program, subtrack) 映射
        program_counts: Dict[int, int] = {}
        part_program_map: Dict[int, Tuple[int, int]] = {}
        for p_idx, trk in enumerate(data.get('tracks', [])):
            prog = trk.get('program', 0)
            sub_cnt = program_counts.get(prog, 0)
            sub = sub_cnt if sub_cnt < self.tokenizer.MAX_SUBTRACKS else 0
            part_program_map[p_idx] = (prog, sub)
            program_counts[prog] = sub_cnt + 1

        # merged: (measure, position, program, subtrack, priority, kind, data)
        merged: List[Tuple[int, int, int, int, float, str, tuple]] = []

        for part_idx, track in enumerate(data.get('tracks', [])):
            prog, sub = part_program_map[part_idx]
            for n in track.get('notes', []):
                m = n['measure'] - 1
                if m not in measure_starts:
                    continue

                start = measure_starts[m]
                pos = max(0, min(self.grid_size - 1,
                          int(round((n['time'] - start) / ticks_per_pos))))

                # 力度量化
                vel = n.get('velocity', 64)
                vel_level = min(self.velocity_levels - 1,
                                vel // (128 // self.velocity_levels))

                is_grace = n.get('is_grace', False)
                if is_grace:
                    dur = 1  # 倚音占 1 格
                    # PDMX 不区分 acciaccatura / appoggiatura，统一标记 grace
                    merged.append((m, pos, prog, sub, 1.5, 'g',
                                   ('grace', n['pitch'], vel_level, dur)))
                else:
                    dur_ticks = n.get('duration', ticks_per_pos)
                    dur = max(1, min(self.grid_size,
                              int(round(dur_ticks / ticks_per_pos))))
                    merged.append((m, pos, prog, sub, 2, 'n',
                                   (n['pitch'], vel_level, dur)))

        # ── 4. 拍号事件 ────────────────────────────────────
        last_ts = None
        for m in range(max_measure + 1):
            ts_str = measure_ts[m]
            if ts_str in REMITokenizer.TIME_SIGNATURES and ts_str != last_ts:
                if m in measure_starts:
                    merged.append((m, 0, 0, 0, 0, 'x',
                                   (REMITokenizer.TIMESIG, ts_str)))
                    last_ts = ts_str

        # ── 5. 速度事件 ──────────────────────────────────────
        # 只保留实际变化的非重复 tempo
        last_qpm = None
        for tempo_ev in data.get('tempos', []):
            t = tempo_ev['time']
            qpm_raw = round(tempo_ev['qpm'])
            qpm = max(30, min(240, (qpm_raw // 10) * 10))
            if qpm == last_qpm:
                continue
            last_qpm = qpm

            # 找到该时间对应的小节
            t_measure = 0
            for m_idx in sorted(measure_starts.keys(), reverse=True):
                if t >= measure_starts[m_idx]:
                    t_measure = m_idx
                    break
            merged.append((t_measure, 0, 0, 0, 0.5, 'x',
                           (REMITokenizer.TEMPO, qpm)))

        # ── 6. 排序 ──────────────────────────────────────────
        merged.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))

        # ── 7. 组装 REMI 事件序列 ──────────────────────────
        events: List[Tuple[str, Optional[int]]] = []
        cur_pos = -1
        cur_program = -1
        cur_subtrack = 0

        # 按小节分组，确保空小节也被遍历
        events_by_measure: Dict[int, list] = defaultdict(list)
        for item in merged:
            events_by_measure[item[0]].append(item)

        for m in range(max_measure + 1):
            events.append((REMITokenizer.BAR, None))
            cur_pos = -1
            cur_program = -1

            for item in events_by_measure.get(m, []):
                _m, pos, prog, sub, _priority, kind, data = item

                if pos != cur_pos:
                    events.append((REMITokenizer.POSITION, pos))
                    cur_pos = pos
                    cur_program = -1

                if prog != cur_program or sub != cur_subtrack:
                    if sub == 0:
                        events.append((REMITokenizer.PROGRAM, str(prog)))
                    else:
                        events.append((REMITokenizer.PROGRAM, f'{prog}_{sub}'))
                    cur_program = prog
                    cur_subtrack = sub

                if kind == 'x':
                    ttype, val = data
                    events.append((ttype, val))
                elif kind == 'g':  # grace note
                    gn_type, pitch, vel_level, dur = data
                    events.append((REMITokenizer.GRACE_NOTE, gn_type))
                    events.append((REMITokenizer.NOTE_ON, pitch))
                    events.append((REMITokenizer.VELOCITY, vel_level))
                    events.append((REMITokenizer.DURATION, dur))
                else:  # 'n' — normal note
                    pitch, vel_level, dur = data
                    events.append((REMITokenizer.NOTE_ON, pitch))
                    events.append((REMITokenizer.VELOCITY, vel_level))
                    events.append((REMITokenizer.DURATION, dur))

        # Key token 作为第一个内容事件
        if key_name:
            events.insert(0, (REMITokenizer.KEY, key_name))
        return events
