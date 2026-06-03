"""
REMI (REvamped MIDI-derived) Tokenizer for piano scores — v0.3.0.
Manages vocabulary and conversion between REMI events and token IDs.
"""
from typing import List, Tuple, Optional


# ── 主音名 → MIDI 音高映射 ──────────────────────────────────
_TONIC_PC_MAP = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
    'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11, 'Cb': 11,
}

# ── 降号→升号规范化 (词表只含升号名) ─────────────────────
_FLAT_TO_SHARP: dict[str, str] = {
    'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#',
    'Cb': 'B', 'Fb': 'E',
}

# ── MIDI Program 0-127 → 保留的 43 个乐器映射 ────────────
# 映射原则: 同乐器族内最近的高频乐器; 无同族则映射到相近族
_PROGRAM_FALLBACK: dict[int, int] = {
    # Piano (0-7): retained 0,1,4,5
    2: 1,    # Bright Acoustic → Acoustic Grand
    3: 4,    # Electric Grand → Electric Piano 1
    6: 1,    # Harpsichord → Acoustic Grand (closest keyboard)
    7: 5,    # Clavinet → Electric Piano 2
    # Chromatic Percussion (8-15): retained 11
    8: 11, 9: 11, 10: 11, 12: 11, 13: 11, 14: 11, 15: 11,
    # Organ (16-23): retained 18,21
    16: 18, 17: 18,  # Drawbar/Percussive → Rock Organ
    19: 18,           # Rock → Rock
    20: 21, 22: 21, 23: 21,  # Church/Reed/Accordion → Accordion
    # Guitar (24-31): retained 24,25,26,27,28,29,30
    31: 30,  # Overdriven → Distortion
    # Bass (32-39): retained 32,33,34,35,38
    36: 35, 37: 35,  # Fretless/Slap → Picked Bass
    39: 38,           # Synth Bass 2 → Synth Bass 1
    # Strings (40-47): retained 40,42,45,46,47
    41: 40,  # Viola → Violin
    43: 42,  # Contrabass → Cello (bowed strings)
    44: 40,  # Tremolo → Violin
    # Ensemble (48-55): retained 48,49,50,52,53
    51: 50,  # Synth Strings 2 → Synth Strings 1
    54: 53,  # Synth Voice → Voice Oohs
    55: 62,  # Orchestra Hit → Brass Section (bright orchestral)
    # Brass (56-63): retained 56,57,60,61,62
    58: 57,  # Tuba → Trombone (low brass)
    59: 56,  # Muted Trumpet → Trumpet
    63: 62,  # Synth Brass 2 → Synth Brass 1
    # Reed (64-71): retained 65,66,68,71
    64: 65,  # Soprano Sax → Alto Sax
    67: 66,  # Baritone Sax → Tenor Sax
    69: 68,  # English Horn → Oboe
    70: 71,  # Bassoon → Clarinet
    # Pipe (72-79): retained 73
    72: 73, 74: 73, 75: 73, 76: 73, 77: 73, 78: 73, 79: 73,
    # Synth Lead (80-87): retained 80,81,87
    82: 81, 83: 81, 84: 81, 85: 81, 86: 81,
    # Synth Pad (88-95): retained 89
    88: 89, 90: 89, 91: 89, 92: 89, 93: 89, 94: 89, 95: 89,
    # Synth Effects (96-103): no retained → Synth Lead 81
    96: 81, 97: 81, 98: 81, 99: 81, 100: 81, 101: 81, 102: 81, 103: 81,
    # Ethnic (104-111): no retained → Flute 73
    104: 73, 105: 73, 106: 73, 107: 73, 108: 73, 109: 73, 110: 73, 111: 73,
    # Percussive (112-119): drum kits → Piano 0
    112: 0, 113: 0, 114: 0, 115: 0, 116: 0, 117: 0, 118: 0, 119: 0,
    # Sound Effects (120-127): → Piano 0
    120: 0, 121: 0, 122: 0, 123: 0, 124: 0, 125: 0, 126: 0, 127: 0,
}

# ── Dynamic 特殊值规范化 ───────────────────────────────
def _normalize_dynamic(value: str) -> str:
    """将非标准动态记号映射到词表中的 19 个。

    规则:
      - p (n≥5) → pppp, f (n≥5) → ffff
      - 显式映射表中未覆盖的 → 默认 mf
    """
    # DYN_FALLBACK: non-standard → standard
    _DYN_FALLBACK: dict[str, str] = {
        'ppppp': 'pppp', 'pppppp': 'pppp',
        'fffff': 'ffff', 'ffffff': 'ffff',
        'n': 'pppp', 'pf': 'mp',
    }
    if value in _DYN_FALLBACK:
        return _DYN_FALLBACK[value]
    # General rule: extended p/f sequences
    if all(c == 'p' for c in value) and len(value) >= 5:
        return 'pppp'
    if all(c == 'f' for c in value) and len(value) >= 5:
        return 'ffff'
    return 'mf'


def tonic_name_to_midi(tonic_name: str | None) -> int:
    """将主音名转为 MIDI 主音音高（八度 4），无时默认 C（60）。"""
    if not tonic_name:
        return 60
    pc = _TONIC_PC_MAP.get(tonic_name, 0)
    return pc + 60


# ── v0.2.x 兼容别名 ──
key_name_to_tonic_midi = tonic_name_to_midi


class REMITokenizer:
    """REMI tokenizer — v0.3.0 vocabulary.

    Token types (v0.3.0 changes):
        <Tonic X>          — Tonic pitch class (12), replaces <Key> (30)
        <Program N>        — Instrument program (43 retained, ×4 subtracks)
        <Voice N>          — Voice ID (4: SATB), replaces subtrack-only marking
        <Fig X>            — Figuration type (12)
        <Cad X>            — Cadence type (5)
        (removed)          — <Anticipate>, <Chord>, <Chord 7>, <Inv>
    """

    # ── Token prefix constants ──────────────────────────────
    PAD = '<PAD>'
    BOS = '<BOS>'
    EOS = '<EOS>'
    MASK = '<MASK>'
    BAR = '<Bar>'
    POSITION = '<Position'
    PROGRAM = '<Program'
    VOICE = '<Voice'
    NOTE_ON = '<Note_ON'
    VELOCITY = '<Velocity'
    DURATION = '<Duration'
    CLEF = '<Clef'
    DYNAMIC = '<Dynamic'
    HAIRPIN = '<Hairpin'
    ARTIC = '<Artic'
    ORNAMENT = '<Ornament'
    PEDAL = '<Pedal'
    SLUR = '<Slur'
    REPEAT = '<Repeat'
    JUMP = '<Jump'
    TEMPO = '<Tempo'
    TUPLET_START = '<TupletStart'
    TUPLET_END = '<TupletEnd>'
    TIMESIG = '<TimeSig'
    REST = '<Rest>'
    GRACE_NOTE = '<GraceNote'
    TONIC = '<Tonic'
    BEAT = '<Beat'
    OCTAVE = '<Octave'
    ARPEGGIO = '<Arpeggio>'
    BASS = '<Bass'
    SECTION = '<Section'
    SEC_SUM = '<SecSum>'
    FIGV = '<FigV'
    CADENCE = '<Cad'

    MAX_SUBTRACKS = 4
    MAX_BEATS = 16

    # ── 常量列表 ────────────────────────────────────────────

    SECTION_NAMES = [
        'exposition', 'development', 'recapitulation',
        'theme1', 'theme2', 'themen', 'intro', 'coda',
        'bridge', 'cadenza', 'transition', 'variation', 'episode',
        '0', '1', '2', '3', '4', '5', '6', '7',
    ]

    # 43 个保留乐器 (>1% 文件覆盖率)
    PROGRAM_NAMES = [
        0, 1, 4, 5, 11,
        18, 21,
        24, 25, 26, 27, 28, 29, 30,
        32, 33, 34, 35, 38,
        40, 42, 45, 46, 47,
        48, 49, 50, 52, 53,
        56, 57, 60, 61, 62,
        65, 66, 68, 71, 73,
        80, 81, 87, 89,
    ]

    # 4 个声部 (SATB)
    VOICE_NAMES = ['0', '1', '2', '3']

    # 12 个织体类型
    FIGURATION_NAMES = [
        'none', 'block', 'alberti', 'arpeggio', 'stride',
        'octave_tremolo', 'walking_bass', 'countermelody', 'pedal',
        'waltz', 'broken_octave', 'tremolo',
    ]

    # 5 个终止式类型
    CADENCE_NAMES = ['none', 'PAC', 'IAC', 'HC', 'DC', 'PC']

    # 12 个主音
    TONIC_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    TUPLET_RATIOS = [
        '3:2', '5:4', '6:4', '7:4', '7:8', '5:6', '9:8', '10:8',
        '11:8', '13:8', '14:8', '15:8', '17:8', '19:8', '21:8',
        '22:8', '2:3', '4:3', '4:5', '4:6',
    ]

    TIME_SIGNATURES = [
        '2/4', '3/4', '4/4', '5/4', '6/4', '2/2', '3/2', '4/2',
        '3/8', '6/8', '9/8', '12/8', '5/8', '7/8',
    ]

    def __init__(self, grid_size: int = 16, velocity_levels: int = 8):
        self.grid_size = grid_size
        self.velocity_levels = velocity_levels
        self._token_to_id: dict = {}
        self._id_to_token: dict = {}
        self._prog_index: dict = {}  # original MIDI program → vocab index
        self.build_vocab()

    def build_vocab(self):
        """Build the v0.3.0 vocabulary."""
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

        # <Position 0> .. <Position 15>
        for i in range(self.grid_size):
            t = f'{self.POSITION} {i}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1

        # ── Program (43 instruments × 4 subtracks) ──
        for prog in self.PROGRAM_NAMES:
            self._prog_index[prog] = idx
            t = f'{self.PROGRAM} {prog}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1
            for sub in range(1, self.MAX_SUBTRACKS):
                t = f'{self.PROGRAM} {prog}_{sub}>'
                self._token_to_id[t] = idx
                self._id_to_token[idx] = t
                idx += 1

        # ── Voice (4 SATB) ──
        for vname in self.VOICE_NAMES:
            t = f'{self.VOICE} {vname}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1

        # <Note_ON -60> .. <Note_ON +60>
        for interval in range(-60, 61):
            t = f'{self.NOTE_ON} {interval}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1

        # <Velocity 0> .. <Velocity 7>
        for level in range(self.velocity_levels):
            t = f'{self.VELOCITY} {level}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1

        # <Duration 1> .. <Duration 16>
        for d in range(1, self.grid_size + 1):
            t = f'{self.DURATION} {d}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1

        # ── Extended token types ────────────────────────────

        for clef_name in ('treble', 'bass', 'alto', 'tenor', 'soprano',
                          'c_1', 'c_2', 'c_5', 'percussion'):
            t = f'{self.CLEF} {clef_name}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1

        for dyn in ('pppp', 'ppp', 'pp', 'p', 'mp', 'mf', 'f', 'ff', 'fff', 'ffff',
                    'sfz', 'sfp', 'sf', 'fz', 'fp', 'rf', 'rfz', 'sffz', 'sfpp'):
            t = f'{self.DYNAMIC} {dyn}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1

        for hp in ('cresc', 'dim'):
            t = f'{self.HAIRPIN} {hp}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1

        for art in ('staccato', 'accent', 'tenuto', 'marcato', 'pizzicato', 'fermata'):
            t = f'{self.ARTIC} {art}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1

        for orn in ('trill', 'mordent', 'turn', 'tremolo'):
            t = f'{self.ORNAMENT} {orn}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1

        for ped in ('start', 'end'):
            t = f'{self.PEDAL} {ped}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1

        for sl in ('start', 'end'):
            t = f'{self.SLUR} {sl}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1

        for rpt in ('start', 'end', 'volta_1', 'volta_2'):
            t = f'{self.REPEAT} {rpt}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1

        for jmp in ('da_capo', 'dal_segno', 'segno', 'coda', 'fine'):
            t = f'{self.JUMP} {jmp}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1

        for bpm in range(30, 241, 10):
            t = f'{self.TEMPO} {bpm}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1

        for ratio in self.TUPLET_RATIOS:
            t = f'{self.TUPLET_START} {ratio}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1

        self._token_to_id[self.TUPLET_END] = idx
        self._id_to_token[idx] = self.TUPLET_END
        idx += 1

        for ts in self.TIME_SIGNATURES:
            t = f'{self.TIMESIG} {ts}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1

        self._token_to_id[self.REST] = idx
        self._id_to_token[idx] = self.REST
        idx += 1

        for gn in ('acciaccatura', 'appoggiatura', 'grace'):
            t = f'{self.GRACE_NOTE} {gn}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1

        # ── Tonic (12, replaces old Key 30) ──
        for tname in self.TONIC_NAMES:
            t = f'{self.TONIC} {tname}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1

        # <Beat 1> .. <Beat 16>
        for beat_num in range(1, self.MAX_BEATS + 1):
            t = f'{self.BEAT} {beat_num}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1

        for oct_val in ('8va', '8vb', '15ma', '15mb', 'end'):
            t = f'{self.OCTAVE} {oct_val}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1

        self._token_to_id[self.ARPEGGIO] = idx
        self._id_to_token[idx] = self.ARPEGGIO
        idx += 1

        for bass_pc in range(12):
            t = f'{self.BASS} {bass_pc}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1

        # ── Section (paragraph-aware) ──
        for sec_name in self.SECTION_NAMES:
            t = f'{self.SECTION} {sec_name}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1

        self._token_to_id[self.SEC_SUM] = idx
        self._id_to_token[idx] = self.SEC_SUM
        idx += 1

        # ── v0.3.2 gen5: Per-voice Figuration (4 voices × 11 non-none types = 44) ──
        # "none" is implicit (no token emitted), only non-none types get tokens
        for v in self.VOICE_NAMES:
            for fname in self.FIGURATION_NAMES:
                if fname == 'none':
                    continue  # implicit default, no token needed
                t = f'{self.FIGV}{v} {fname}>'
                self._token_to_id[t] = idx
                self._id_to_token[idx] = t
                idx += 1

        # ── Cadence (5 types) ──
        for cname in self.CADENCE_NAMES:
            t = f'{self.CADENCE} {cname}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1

    # ── Properties ──────────────────────────────────────────

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

    # ── Encode / Decode ─────────────────────────────────────

    def encode_token(self, token: str) -> int:
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
        return self._id_to_token.get(token_id, self.MASK)

    def tokenize(self, events: List[Tuple[str, Optional[int | str]]]) -> List[int]:
        """Convert list of (token_type, value) event tuples to token IDs."""
        ids = []
        for token_type, value in events:
            if value is not None:
                v = value
                # v0.3.2: normalize flat tonic names (Bb→A#) to match vocab
                if token_type.startswith(self.TONIC):
                    v = _FLAT_TO_SHARP.get(str(value), str(value))
                # v0.3.2: map rare MIDI programs to nearest retained instrument
                elif token_type.startswith(self.PROGRAM):
                    prog_num = int(value)
                    if prog_num not in self._prog_index:
                        prog_num = _PROGRAM_FALLBACK.get(prog_num, 0)
                    v = prog_num
                # v0.3.2: normalize rare dynamics (ppppp→pppp etc.)
                elif token_type.startswith(self.DYNAMIC):
                    v = _normalize_dynamic(str(value))
                token = f'{token_type} {v}>'
            else:
                token = token_type
            ids.append(self.encode_token(token))
        return ids

    def detokenize(self, token_ids: List[int]) -> List[Tuple[str, Optional[str]]]:
        """Convert token IDs back to list of (token_type, value) event tuples."""
        events = []
        for tid in token_ids:
            token = self.decode_token(tid)
            if token in (self.PAD, self.BOS, self.EOS, self.MASK, self.BAR):
                events.append((token, None))
            elif token.startswith(self.PROGRAM):
                val = token[len(self.PROGRAM) + 1:-1]
                events.append((self.PROGRAM, val))
            elif token.startswith(self.VOICE):
                val = token[len(self.VOICE) + 1:-1]
                events.append((self.VOICE, val))
            elif token.startswith(self.POSITION):
                val = token[len(self.POSITION) + 1:-1]
                events.append((self.POSITION, val))
            elif token.startswith(self.NOTE_ON):
                val = token[len(self.NOTE_ON) + 1:-1]
                events.append((self.NOTE_ON, val))
            elif token.startswith(self.VELOCITY):
                val = token[len(self.VELOCITY) + 1:-1]
                events.append((self.VELOCITY, val))
            elif token.startswith(self.DURATION):
                val = token[len(self.DURATION) + 1:-1]
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
                val = token[len(self.TEMPO) + 1:-1]
                events.append((self.TEMPO, val))
            elif token.startswith(self.TUPLET_START):
                val = token[len(self.TUPLET_START) + 1:-1]
                events.append((self.TUPLET_START, val))
            elif token == self.TUPLET_END:
                events.append((self.TUPLET_END, None))
            elif token.startswith(self.TIMESIG):
                val = token[len(self.TIMESIG) + 1:-1]
                events.append((self.TIMESIG, val))
            elif token == self.REST:
                events.append((self.REST, None))
            elif token.startswith(self.GRACE_NOTE):
                val = token[len(self.GRACE_NOTE) + 1:-1]
                events.append((self.GRACE_NOTE, val))
            elif token.startswith(self.TONIC):
                val = token[len(self.TONIC) + 1:-1]
                events.append((self.TONIC, val))
            elif token.startswith(self.BEAT):
                val = token[len(self.BEAT) + 1:-1]
                events.append((self.BEAT, val))
            elif token.startswith(self.OCTAVE):
                val = token[len(self.OCTAVE) + 1:-1]
                events.append((self.OCTAVE, val))
            elif token == self.ARPEGGIO:
                events.append((self.ARPEGGIO, None))
            elif token.startswith(self.BASS):
                val = token[len(self.BASS) + 1:-1]
                events.append((self.BASS, val))
            elif token.startswith(self.SECTION):
                val = token[len(self.SECTION) + 1:-1]
                events.append((self.SECTION, val))
            elif token == self.SEC_SUM:
                events.append((self.SEC_SUM, None))
            elif token.startswith(self.FIGV):
                # <FigV0 block> → val = "0 block"
                val = token[len(self.FIGV):-1]
                events.append((self.FIGV, val))
            elif token.startswith(self.CADENCE):
                val = token[len(self.CADENCE) + 1:-1]
                events.append((self.CADENCE, val))
        return events

    # ── Helpers ────────────────────────────────────────────

    def get_tonic_id(self, tonic_name: str) -> int:
        tonic_name = _FLAT_TO_SHARP.get(tonic_name, tonic_name)
        t = f'{self.TONIC} {tonic_name}>'
        return self.encode_token(t)

    def get_voice_id(self, voice_idx: int) -> int:
        t = f'{self.VOICE} {voice_idx}>'
        return self.encode_token(t)

    def get_program_id(self, prog: int, sub: int = 0) -> int:
        if sub == 0:
            t = f'{self.PROGRAM} {prog}>'
        else:
            t = f'{self.PROGRAM} {prog}_{sub}>'
        return self.encode_token(t)

    @property
    def tonic_token_ids(self) -> List[int]:
        return [self.encode_token(f'{self.TONIC} {t}>') for t in self.TONIC_NAMES]

    @property
    def voice_token_ids(self) -> List[int]:
        return [self.encode_token(f'{self.VOICE} {v}>') for v in self.VOICE_NAMES]

    @property
    def figv_token_ids(self) -> List[int]:
        """All <FigV V X> token IDs (44: 4 voices × 11 non-none types)."""
        ids = []
        for v in self.VOICE_NAMES:
            for f in self.FIGURATION_NAMES:
                if f == 'none':
                    continue
                ids.append(self.encode_token(f'{self.FIGV}{v} {f}>'))
        return ids

    def get_figv_id(self, voice: int, fig_name: str) -> int:
        """获取指定声部和织体类型的 token ID。fig_name='none' 返回 -1 (无 token)。"""
        if fig_name == 'none':
            return -1
        return self.encode_token(f'{self.FIGV}{voice} {fig_name}>')

    @property
    def cadence_token_ids(self) -> List[int]:
        return [self.encode_token(f'{self.CADENCE} {c}>') for c in self.CADENCE_NAMES]

    @property
    def framework_token_ids(self) -> set[int]:
        """All framework tokens that should NOT be sampled during generation."""
        ids = set()
        ids.add(self.bar_token_id)
        ids.update(self.tonic_token_ids)
        for ts in self.TIME_SIGNATURES:
            ids.add(self.encode_token(f'{self.TIMESIG} {ts}>'))
        for bpm in range(30, 241, 10):
            ids.add(self.encode_token(f'{self.TEMPO} {bpm}>'))
        for clef_name in ('treble', 'bass', 'alto', 'tenor', 'soprano',
                          'c_1', 'c_2', 'c_5', 'percussion'):
            ids.add(self.encode_token(f'{self.CLEF} {clef_name}>'))
        for i in range(self.grid_size):
            ids.add(self.encode_token(f'{self.POSITION} {i}>'))
        for sec_name in self.SECTION_NAMES:
            ids.add(self.encode_token(f'{self.SECTION} {sec_name}>'))
        ids.add(self.encode_token(self.SEC_SUM))
        ids.update(self.voice_token_ids)
        ids.update(self.figv_token_ids)
        ids.update(self.cadence_token_ids)
        for beat_num in range(1, self.MAX_BEATS + 1):
            ids.add(self.encode_token(f'{self.BEAT} {beat_num}>'))
        return ids
