"""
REMI (REvamped MIDI-derived) Tokenizer for piano scores.
Manages vocabulary and conversion between REMI events and token IDs.
"""
from typing import List, Tuple, Optional


# ── 调号名 → 主音 MIDI 音高映射 ──────────────────────────────
_KEY_PC_MAP = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
    'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11, 'Cb': 11,
}


def key_name_to_tonic_midi(key_name: str | None) -> int:
    """将调号名转为 MIDI 主音音高（八度 4），无调号时默认 C（60）。

    >>> key_name_to_tonic_midi('C')
    60
    >>> key_name_to_tonic_midi('Am')
    69
    >>> key_name_to_tonic_midi('F#')
    66
    >>> key_name_to_tonic_midi(None)
    60
    """
    if not key_name:
        return 60
    root = key_name[:-1] if key_name.endswith('m') else key_name
    pc = _KEY_PC_MAP.get(root, 0)
    return pc + 60


class REMITokenizer:
    """REMI tokenizer for piano score tokenization.

    Token types:
        <PAD>, <BOS>, <EOS>, <MASK>  — Special tokens
        <Bar>                         — Bar line
        <Position N>                  — Position in measure (0 to grid_size-1)
        <Program N> / <Program N_M>   — Program change with subtrack
        <Note_ON I>                   — Note on with semitone interval from tonic (-60 to +60)
        <Velocity V>                  — Velocity level (0 to velocity_levels-1)
        <Duration D>                  — Duration in position steps (1 to grid_size)
        <Clef type>                   — Clef (treble/bass/alto/tenor)
        <Dynamic val>                 — Dynamic marking (ppp..fp)
        <Hairpin val>                 — Crescendo/diminuendo
        <Artic val>                   — Articulation (staccato..fermata)
        <Ornament val>                — Ornament (trill..tremolo)
        <Pedal val>                   — Pedal (start/end)
        <Slur val>                    — Slur (start/end)
        <Repeat val>                  — Repeat (start/end/volta)
        <Jump val>                    — Jump (da_capo..fine)
        <Tempo BPM>                   — Tempo (30-240, step 10)
        <TupletStart N:M>             — Tuplet start with ratio
        <TupletEnd>                   — Tuplet end
        <TimeSig N/M>                 — Time signature
        <Rest>                        — Rest
        <GraceNote type>              — Grace note (acciaccatura/appoggiatura/grace)
        <Key NAME>                    — Key signature (C, G, D, ..., Abm)
        <Beat N>                      — Beat position (1-16)
        <Octave val>                  — Octave shift (8va/8vb/15ma/15mb/end)
        <Arpeggio>                    — Arpeggio mark
        <Bass N>                      — Bass note pitch class (0-11, C-B)
        <Anticipate Key NAME>         — Anticipated key change target

    Vocabulary is built dynamically from grid_size and velocity_levels.
    """

    PAD = '<PAD>'
    BOS = '<BOS>'
    EOS = '<EOS>'
    MASK = '<MASK>'
    BAR = '<Bar>'
    POSITION = '<Position'
    PROGRAM = '<Program'
    NOTE_ON = '<Note_ON'
    VELOCITY = '<Velocity'
    DURATION = '<Duration'

    MAX_SUBTRACKS = 4   # 每乐器最多子轨数（_0 ~ _3）

    # --- 扩展 token 类型 ---
    CLEF = '<Clef'              # 谱号: treble, bass, alto, tenor
    DYNAMIC = '<Dynamic'        # 力度记号: ppp, pp, p, mp, mf, f, ff, fff, sfz, fp
    HAIRPIN = '<Hairpin'        # 渐强/渐弱: cresc, dim
    ARTIC = '<Artic'            # 演奏法: staccato, accent, tenuto, marcato, pizzicato, fermata
    ORNAMENT = '<Ornament'      # 装饰音: trill, mordent, turn, tremolo
    PEDAL = '<Pedal'            # 踏板: start, end
    SLUR = '<Slur'              # 连奏线: start, end
    REPEAT = '<Repeat'          # 反复: start, end, volta_1, volta_2
    JUMP = '<Jump'              # 跳转: da_capo, dal_segno, segno, coda, fine
    TEMPO = '<Tempo'            # 速度: 30-240, 步长 10
    TUPLET_START = '<TupletStart'  # 连音开始: ratio e.g. 3:2
    TUPLET_END = '<TupletEnd>'     # 连音结束
    TIMESIG = '<TimeSig'           # 拍号: e.g. 4/4, 3/4, 6/8
    REST = '<Rest>'                # 休止符
    GRACE_NOTE = '<GraceNote'      # 倚音: acciaccatura, appoggiatura, grace
    KEY = '<Key'                   # 调号: C, G, D, A, E, B, F#, C#, F, Bb, Eb, Ab, Db, Gb, Cb (major) / Am, Em, ... (minor)
    BEAT = '<Beat'                 # 拍位: 1 起，如 <Beat 1>（=强拍）、<Beat 2> 等
    OCTAVE = '<Octave'             # 八度记号: 8va, 8vb, 15ma, 15mb, end
    ARPEGGIO = '<Arpeggio>'        # 琶音记号（自闭合，无参数）
    BASS = '<Bass'                 # 低音音级: 0~11（C~B）
    ANTICIPATE = '<Anticipate'     # 预期调性变更目标: 如 Key C, Key G
    SECTION = '<Section'           # 段落类型标记 (paragraph-aware)
    SEC_SUM = '<SecSum>'           # 段落 summary token（每段一个）
    CHORD = '<Chord'               # 和弦功能标记 (functional harmony)
    CHORD7 = '<Chord 7>'           # 七和弦扩展标记
    INV = '<Inv'                   # 和弦转位标记

    # 段落类型名称（用于 Section token 构建）
    SECTION_NAMES = [
        'exposition', 'development', 'recapitulation',
        'theme1', 'theme2', 'themen', 'intro', 'coda',
        'bridge', 'cadenza', 'transition', 'variation', 'episode',
        '0', '1', '2', '3', '4', '5', '6', '7',
    ]

    # 和弦功能名称（16 个罗马数字标记）
    CHORD_FUNCTIONS = [
        'I', 'i', 'ii', 'ii°', 'iii', 'III',
        'IV', 'iv', 'V', 'vi', 'VI', 'vii°',
        'N', 'It6', 'Fr6', 'Ger6',
    ]

    # 和弦转位名称（4 个）
    CHORD_INVERSIONS = ['Root', '1st', '2nd', '3rd']

    # 最多支持的拍数（与 grid_size 16 对齐）
    MAX_BEATS = 16

    # 30 个标准调号
    KEY_NAMES = [
        'C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'F', 'Bb', 'Eb', 'Ab', 'Db', 'Gb', 'Cb',
        'Am', 'Em', 'Bm', 'F#m', 'C#m', 'G#m', 'D#m', 'A#m', 'Dm', 'Gm', 'Cm', 'Fm', 'Bbm', 'Ebm', 'Abm',
    ]

    # 预定义 tuplet 比率
    TUPLET_RATIOS = [
        '3:2', '5:4', '6:4', '7:4', '7:8', '5:6', '9:8', '10:8',
        '11:8', '13:8', '14:8', '15:8', '17:8', '19:8', '21:8',
        '22:8', '2:3', '4:3', '4:5', '4:6',
    ]

    # 预定义拍号
    TIME_SIGNATURES = [
        '2/4', '3/4', '4/4', '5/4', '6/4', '2/2', '3/2', '4/2',
        '3/8', '6/8', '9/8', '12/8', '5/8', '7/8',
    ]

    def __init__(self, grid_size: int = 16, velocity_levels: int = 8):
        self.grid_size = grid_size
        self.velocity_levels = velocity_levels
        self._token_to_id: dict = {}
        self._id_to_token: dict = {}
        self.build_vocab()

    def build_vocab(self):
        """Build the vocabulary: token string → integer ID mapping."""
        idx = 0

        # Special tokens
        for token in [self.PAD, self.BOS, self.EOS, self.MASK]:
            self._token_to_id[token] = idx
            self._id_to_token[idx] = token
            idx += 1

        # <Bar>
        self._token_to_id[self.BAR] = idx
        self._id_to_token[idx] = self.BAR
        idx += 1

        # <Position 0> .. <Position grid_size-1>
        for i in range(self.grid_size):
            t = f'{self.POSITION} {i}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1

        # <Program 0> .. <Program 127>（子轨 0 隐含）
        # <Program 0_1> .. <Program 127_3>（显式子轨）
        for prog in range(128):
            t = f'{self.PROGRAM} {prog}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1
            for sub in range(1, self.MAX_SUBTRACKS):
                t = f'{self.PROGRAM} {prog}_{sub}>'
                self._token_to_id[t] = idx
                self._id_to_token[idx] = t
                idx += 1

        # <Note_ON -60> .. <Note_ON +60>（半音程，相对主调主音）
        for interval in range(-60, 61):
            t = f'{self.NOTE_ON} {interval}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1

        # <Velocity 0> .. <Velocity velocity_levels-1>
        for level in range(self.velocity_levels):
            t = f'{self.VELOCITY} {level}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1

        # <Duration 1> .. <Duration grid_size>
        for d in range(1, self.grid_size + 1):
            t = f'{self.DURATION} {d}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1

        # ── 扩展 token 类型 ──────────────────────────────────

        # <Clef treble>, <Clef bass>, <Clef alto>, <Clef tenor>, <Clef soprano>,
        # <Clef c_1>, <Clef c_2>, <Clef c_5>, <Clef percussion>
        for clef_name in ('treble', 'bass', 'alto', 'tenor', 'soprano', 'c_1', 'c_2', 'c_5', 'percussion'):
            t = f'{self.CLEF} {clef_name}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1

        # <Dynamic pppp> .. <Dynamic sfpp>
        for dyn in ('pppp', 'ppp', 'pp', 'p', 'mp', 'mf', 'f', 'ff', 'fff', 'ffff',
                    'sfz', 'sfp', 'sf', 'fz', 'fp', 'rf', 'rfz', 'sffz', 'sfpp'):
            t = f'{self.DYNAMIC} {dyn}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1

        # <Hairpin cresc>, <Hairpin dim>
        for hp in ('cresc', 'dim'):
            t = f'{self.HAIRPIN} {hp}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1

        # <Artic staccato> .. <Artic fermata>
        for art in ('staccato', 'accent', 'tenuto', 'marcato', 'pizzicato', 'fermata'):
            t = f'{self.ARTIC} {art}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1

        # <Ornament trill>, <Ornament mordent>, <Ornament turn>, <Ornament tremolo>
        for orn in ('trill', 'mordent', 'turn', 'tremolo'):
            t = f'{self.ORNAMENT} {orn}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1

        # <Pedal start>, <Pedal end>
        for ped in ('start', 'end'):
            t = f'{self.PEDAL} {ped}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1

        # <Slur start>, <Slur end>
        for sl in ('start', 'end'):
            t = f'{self.SLUR} {sl}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1

        # <Repeat start>, <Repeat end>, <Repeat volta_1>, <Repeat volta_2>
        for rpt in ('start', 'end', 'volta_1', 'volta_2'):
            t = f'{self.REPEAT} {rpt}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1

        # <Jump da_capo>, <Jump dal_segno>, <Jump segno>, <Jump coda>, <Jump fine>
        for jmp in ('da_capo', 'dal_segno', 'segno', 'coda', 'fine'):
            t = f'{self.JUMP} {jmp}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1

        # <Tempo 30> .. <Tempo 240> (step 10)
        for bpm in range(30, 241, 10):
            t = f'{self.TEMPO} {bpm}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1

        # <TupletStart 3:2> .. tuplet ratio tokens
        for ratio in self.TUPLET_RATIOS:
            t = f'{self.TUPLET_START} {ratio}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1

        # <TupletEnd>
        self._token_to_id[self.TUPLET_END] = idx
        self._id_to_token[idx] = self.TUPLET_END
        idx += 1

        # <TimeSig 4/4> .. time signature tokens
        for ts in self.TIME_SIGNATURES:
            t = f'{self.TIMESIG} {ts}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1

        # <Rest>
        self._token_to_id[self.REST] = idx
        self._id_to_token[idx] = self.REST
        idx += 1

        # <GraceNote acciaccatura>, <GraceNote appoggiatura>, <GraceNote grace>
        for gn in ('acciaccatura', 'appoggiatura', 'grace'):
            t = f'{self.GRACE_NOTE} {gn}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1

        # <Key C> .. <Key Abm>（30 个标准调号）
        for key_name in self.KEY_NAMES:
            t = f'{self.KEY} {key_name}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1

        # <Beat 1> .. <Beat 16>（拍位标记，1=强拍）
        for beat_num in range(1, self.MAX_BEATS + 1):
            t = f'{self.BEAT} {beat_num}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1

        # <Octave 8va>, <Octave 8vb>, <Octave 15ma>, <Octave 15mb>, <Octave end>
        for oct_val in ('8va', '8vb', '15ma', '15mb', 'end'):
            t = f'{self.OCTAVE} {oct_val}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1

        # <Arpeggio>（自闭合）
        self._token_to_id[self.ARPEGGIO] = idx
        self._id_to_token[idx] = self.ARPEGGIO
        idx += 1

        # <Bass 0> .. <Bass 11>（低音音级，C~B）
        for bass_pc in range(12):
            t = f'{self.BASS} {bass_pc}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1

        # <Anticipate Key C> .. <Anticipate Key Abm>（30 个，复用 KEY_NAMES）
        for key_name in self.KEY_NAMES:
            t = f'{self.ANTICIPATE} {key_name}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1

        # ── 段落感知 token（paragraph-aware）───────────────────

        # <Section exposition> .. <Section 7>（21 个）
        for sec_name in self.SECTION_NAMES:
            t = f'{self.SECTION} {sec_name}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1

        # <SecSum>（段落 summary，自闭合）
        self._token_to_id[self.SEC_SUM] = idx
        self._id_to_token[idx] = self.SEC_SUM
        idx += 1

        # ── 功能和弦 token（functional harmony）─────────────────

        # <Chord I> .. <Chord Ger6>（16 个和弦功能）
        for func_name in self.CHORD_FUNCTIONS:
            t = f'{self.CHORD} {func_name}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1

        # <Chord 7>（七和弦扩展，自闭合）
        self._token_to_id[self.CHORD7] = idx
        self._id_to_token[idx] = self.CHORD7
        idx += 1

        # <Inv Root> .. <Inv 3rd>（4 个转位）
        for inv_name in self.CHORD_INVERSIONS:
            t = f'{self.INV} {inv_name}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1

    @property
    def vocab_size(self) -> int:
        return len(self._token_to_id)

    @property
    def pad_token_id(self) -> int:
        return self._token_to_id[self.PAD]

    @property
    def bos_token_id(self) -> int:
        return self._token_to_id[self.BOS]

    @property
    def eos_token_id(self) -> int:
        return self._token_to_id[self.EOS]

    @property
    def bar_token_id(self) -> int:
        return self._token_to_id[self.BAR]

    def encode_token(self, token: str) -> int:
        """Convert a token string to its ID (returns MASK ID for unknown)."""
        tid = self._token_to_id.get(token)
        if tid is None:
            if not hasattr(self, '_warned_unknown'):
                self._warned_unknown: set = set()
            if token not in self._warned_unknown:
                import logging
                logging.warning(f'未知 token "{token}" 被映射为 MASK ({self.MASK})')
                self._warned_unknown.add(token)
            return self._token_to_id[self.MASK]
        return tid

    def decode_token(self, token_id: int) -> str:
        """Convert a token ID back to its string form."""
        return self._id_to_token.get(token_id, self.MASK)

    def tokenize(self, events: List[Tuple[str, Optional[int]]]) -> List[int]:
        """Convert list of (token_type, value) event tuples to token IDs."""
        ids = []
        for token_type, value in events:
            if value is not None:
                token = f'{token_type} {value}>'
            else:
                token = token_type
            ids.append(self.encode_token(token))
        return ids

    def detokenize(self, token_ids: List[int]) -> List[Tuple[str, Optional[int]]]:
        """Convert token IDs back to list of (token_type, value) event tuples."""
        events = []
        for tid in token_ids:
            token = self.decode_token(tid)
            if token in (self.PAD, self.BOS, self.EOS, self.MASK, self.BAR):
                events.append((token, None))
            elif token.startswith(self.PROGRAM):
                val = token[len(self.PROGRAM) + 1:-1]  # e.g. "0" or "0_1"
                events.append((self.PROGRAM, val))
            elif token.startswith(self.POSITION):
                val = int(token[len(self.POSITION) + 1:-1])
                events.append((self.POSITION, val))
            elif token.startswith(self.NOTE_ON):
                val = int(token[len(self.NOTE_ON) + 1:-1])
                events.append((self.NOTE_ON, val))
            elif token.startswith(self.VELOCITY):
                val = int(token[len(self.VELOCITY) + 1:-1])
                events.append((self.VELOCITY, val))
            elif token.startswith(self.DURATION):
                val = int(token[len(self.DURATION) + 1:-1])
                events.append((self.DURATION, val))
            elif token.startswith(self.CLEF):
                val = token[len(self.CLEF) + 1:-1]
                events.append((self.CLEF, val))
            elif token.startswith(self.DYNAMIC):
                val = token[len(self.DYNAMIC) + 1:-1]
                events.append((self.DYNAMIC, val))
            elif token.startswith(self.HAIRPIN):
                val = token[len(self.HAIRPIN) + 1:-1]
                events.append((self.HAIRPIN, val))
            elif token.startswith(self.ARTIC):
                val = token[len(self.ARTIC) + 1:-1]
                events.append((self.ARTIC, val))
            elif token.startswith(self.ORNAMENT):
                val = token[len(self.ORNAMENT) + 1:-1]
                events.append((self.ORNAMENT, val))
            elif token.startswith(self.PEDAL):
                val = token[len(self.PEDAL) + 1:-1]
                events.append((self.PEDAL, val))
            elif token.startswith(self.SLUR):
                val = token[len(self.SLUR) + 1:-1]
                events.append((self.SLUR, val))
            elif token.startswith(self.REPEAT):
                val = token[len(self.REPEAT) + 1:-1]
                events.append((self.REPEAT, val))
            elif token.startswith(self.JUMP):
                val = token[len(self.JUMP) + 1:-1]
                events.append((self.JUMP, val))
            elif token.startswith(self.TEMPO):
                val = int(token[len(self.TEMPO) + 1:-1])
                events.append((self.TEMPO, val))
            elif token.startswith(self.TUPLET_START):
                val = token[len(self.TUPLET_START) + 1:-1]  # e.g. '3:2'
                events.append((self.TUPLET_START, val))
            elif token == self.TUPLET_END:
                events.append((self.TUPLET_END, None))
            elif token.startswith(self.TIMESIG):
                val = token[len(self.TIMESIG) + 1:-1]  # e.g. '4/4'
                events.append((self.TIMESIG, val))
            elif token == self.REST:
                events.append((self.REST, None))
            elif token.startswith(self.GRACE_NOTE):
                val = token[len(self.GRACE_NOTE) + 1:-1]  # e.g. 'acciaccatura'
                events.append((self.GRACE_NOTE, val))
            elif token.startswith(self.KEY):
                val = token[len(self.KEY) + 1:-1]  # e.g. 'C', 'Am'
                events.append((self.KEY, val))
            elif token.startswith(self.BEAT):
                val = int(token[len(self.BEAT) + 1:-1])  # e.g. 1, 2, 3, 4
                events.append((self.BEAT, val))
            elif token.startswith(self.OCTAVE):
                val = token[len(self.OCTAVE) + 1:-1]  # e.g. '8va', '8vb', 'end'
                events.append((self.OCTAVE, val))
            elif token == self.ARPEGGIO:
                events.append((self.ARPEGGIO, None))
            elif token.startswith(self.BASS):
                val = int(token[len(self.BASS) + 1:-1])  # 0~11
                events.append((self.BASS, val))
            elif token.startswith(self.ANTICIPATE):
                val = token[len(self.ANTICIPATE) + 1:-1]  # e.g. 'C', 'Am'
                events.append((self.ANTICIPATE, val))
            elif token.startswith(self.SECTION):
                val = token[len(self.SECTION) + 1:-1]  # e.g. 'theme1', '0'
                events.append((self.SECTION, val))
            elif token == self.SEC_SUM:
                events.append((self.SEC_SUM, None))
            elif token == self.CHORD7:
                events.append((self.CHORD7, None))
            elif token.startswith(self.CHORD):
                val = token[len(self.CHORD) + 1:-1]  # e.g. 'I', 'V', 'vi'
                events.append((self.CHORD, val))
            elif token.startswith(self.INV):
                val = token[len(self.INV) + 1:-1]  # e.g. 'Root', '1st', '2nd', '3rd'
                events.append((self.INV, val))
        return events
