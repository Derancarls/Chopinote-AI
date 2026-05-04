"""
REMI (REvamped MIDI-derived) Tokenizer for piano scores.
Manages vocabulary and conversion between REMI events and token IDs.
"""
from typing import List, Tuple, Optional


class REMITokenizer:
    """REMI tokenizer for piano score tokenization.

    Token types:
        <PAD>, <BOS>, <EOS>, <MASK>  — Special tokens
        <Bar>                         — Bar line
        <Position N>                  — Position in measure (0 to grid_size-1)
        <Track_L>, <Track_R>          — Left/right hand markers
        <Note_ON P>                   — Note on with MIDI pitch (0-127)
        <Velocity V>                  — Velocity level (0 to velocity_levels-1)
        <Duration D>                  — Duration in position steps (1 to grid_size)

    Vocabulary is built dynamically from grid_size and velocity_levels.
    """

    PAD = '<PAD>'
    BOS = '<BOS>'
    EOS = '<EOS>'
    MASK = '<MASK>'
    BAR = '<Bar>'
    POSITION = '<Position'
    TRACK_L = '<Track_L>'
    TRACK_R = '<Track_R>'
    NOTE_ON = '<Note_ON'
    VELOCITY = '<Velocity'
    DURATION = '<Duration'

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

        # <Track_L>, <Track_R>
        for t in [self.TRACK_L, self.TRACK_R]:
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1

        # <Note_ON 0> .. <Note_ON 127>
        for pitch in range(128):
            t = f'{self.NOTE_ON} {pitch}>'
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

        # <Clef treble>, <Clef bass>, <Clef alto>, <Clef tenor>
        for clef_name in ('treble', 'bass', 'alto', 'tenor'):
            t = f'{self.CLEF} {clef_name}>'
            self._token_to_id[t] = idx
            self._id_to_token[idx] = t
            idx += 1

        # <Dynamic ppp> .. <Dynamic fp>
        for dyn in ('ppp', 'pp', 'p', 'mp', 'mf', 'f', 'ff', 'fff', 'sfz', 'fp'):
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
        return self._token_to_id.get(token, self._token_to_id[self.MASK])

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
            elif token in (self.TRACK_L, self.TRACK_R):
                events.append((token, None))
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
        return events
