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
        return events
