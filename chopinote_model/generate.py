"""推理生成模块：自回归采样 → MusicXML 导出。"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from music21 import stream, note, chord, meter, clef

from .model import MusicTransformer
from .config import ModelConfig
from chopinote_dataset.tokenizer import REMITokenizer, key_name_to_tonic_midi


@dataclass
class SeedProfile:
    """A 阶段输出：seed 的结构画像。"""
    n_bars: int
    bar_density: float
    tonic_key: str | None = None
    tonic_midi: int = 60
    key_pitch_classes: frozenset | None = None
    tempo: int = 120
    programs: list = field(default_factory=list)
    pitch_class_dist: list = field(default_factory=lambda: [1/12]*12)
    interval_dist: list = field(default_factory=lambda: [0.04]*25)
    velocity_mean: float = 64.0
    rest_ratio: float = 0.1
    density_series: list = field(default_factory=list)
    voice_count: int = 1
    time_sig: str = '4/4'


@dataclass
class GenerationParams:
    """生成参数，A 设定初值，B 实时调整。"""
    temperature: float = 1.0
    top_k: int = 20
    rest_penalty: float = 0.0
    key_bias_strength: float = 2.0
    max_polyphony: int = 10
    complexity: float = 5.0
    lock_key: bool = True
    lock_time: bool = True
    lock_tempo: bool = True
    lock_program: bool = True
    prog_switch_strength: float = 1.0
    prog_switch_interval: int = 12

    def copy(self) -> GenerationParams:
        return GenerationParams(**self.__dict__)

    def apply_adjustments(self, adjustments: dict[str, float]) -> GenerationParams:
        """应用参数调整，按边界裁剪。"""
        CLAMPS = {
            'temperature': (0.3, 2.5),
            'rest_penalty': (0.0, 10.0),
            'key_bias_strength': (0.0, 5.0),
            'complexity': (1.0, 10.0),
            'top_k': (1, 100),
            'prog_switch_strength': (0.0, 5.0),
        }
        for k, delta in adjustments.items():
            if hasattr(self, k):
                old = getattr(self, k)
                new = old + delta
                if k in CLAMPS:
                    lo, hi = CLAMPS[k]
                    new = max(lo, min(hi, new))
                setattr(self, k, new)
        return self

logger = logging.getLogger(__name__)

# GM (General MIDI) 乐器名称映射
GM_INSTRUMENT_NAMES: dict[int, str] = {
    # Piano
    0: 'Acoustic Grand Piano', 1: 'Bright Acoustic Piano', 2: 'Electric Grand Piano',
    3: 'Honky-tonk Piano', 4: 'Electric Piano 1', 5: 'Electric Piano 2',
    6: 'Harpsichord', 7: 'Clavinet',
    # Chromatic Percussion
    8: 'Celesta', 9: 'Glockenspiel', 10: 'Music Box', 11: 'Vibraphone',
    12: 'Marimba', 13: 'Xylophone', 14: 'Tubular Bells', 15: 'Dulcimer',
    # Organ
    16: 'Drawbar Organ', 17: 'Percussive Organ', 18: 'Rock Organ',
    19: 'Church Organ', 20: 'Reed Organ', 21: 'Accordion', 22: 'Harmonica',
    23: 'Tango Accordion',
    # Guitar
    24: 'Acoustic Guitar (nylon)', 25: 'Acoustic Guitar (steel)',
    26: 'Electric Guitar (jazz)', 27: 'Electric Guitar (clean)',
    28: 'Electric Guitar (muted)', 29: 'Overdriven Guitar',
    30: 'Distortion Guitar', 31: 'Guitar Harmonics',
    # Bass
    32: 'Acoustic Bass', 33: 'Electric Bass (finger)', 34: 'Electric Bass (pick)',
    35: 'Fretless Bass', 36: 'Slap Bass 1', 37: 'Slap Bass 2',
    38: 'Synth Bass 1', 39: 'Synth Bass 2',
    # Strings
    40: 'Violin', 41: 'Viola', 42: 'Cello', 43: 'Contrabass',
    44: 'Tremolo Strings', 45: 'Pizzicato Strings', 46: 'Orchestral Harp',
    47: 'Timpani',
    # Ensemble
    48: 'String Ensemble 1', 49: 'String Ensemble 2',
    50: 'Synth Strings 1', 51: 'Synth Strings 2', 52: 'Choir Aahs',
    53: 'Voice Oohs', 54: 'Synth Choir', 55: 'Orchestral Hit',
    # Brass
    56: 'Trumpet', 57: 'Trombone', 58: 'Tuba', 59: 'Muted Trumpet',
    60: 'French Horn', 61: 'Brass Section', 62: 'Synth Brass 1',
    63: 'Synth Brass 2',
    # Reed
    64: 'Soprano Sax', 65: 'Alto Sax', 66: 'Tenor Sax', 67: 'Baritone Sax',
    68: 'Oboe', 69: 'English Horn', 70: 'Bassoon', 71: 'Clarinet',
    # Pipe
    72: 'Piccolo', 73: 'Flute', 74: 'Recorder', 75: 'Pan Flute',
    76: 'Blown Bottle', 77: 'Shakuhachi', 78: 'Whistle', 79: 'Ocarina',
    # Synth Lead
    80: 'Lead 1 (square)', 81: 'Lead 2 (sawtooth)', 82: 'Lead 3 (calliope)',
    83: 'Lead 4 (chiff)', 84: 'Lead 5 (charang)', 85: 'Lead 6 (voice)',
    86: 'Lead 7 (fifths)', 87: 'Lead 8 (bass + lead)',
    # Synth Pad
    88: 'Pad 1 (new age)', 89: 'Pad 2 (warm)', 90: 'Pad 3 (polysynth)',
    91: 'Pad 4 (choir)', 92: 'Pad 5 (bowed)', 93: 'Pad 6 (metallic)',
    94: 'Pad 7 (halo)', 95: 'Pad 8 (sweep)',
    # Synth Effects
    96: 'FX 1 (rain)', 97: 'FX 2 (soundtrack)', 98: 'FX 3 (crystal)',
    99: 'FX 4 (atmosphere)', 100: 'FX 5 (brightness)', 101: 'FX 6 (goblins)',
    102: 'FX 7 (echoes)', 103: 'FX 8 (sci-fi)',
    # Ethnic
    104: 'Sitar', 105: 'Banjo', 106: 'Shamisen', 107: 'Koto',
    108: 'Kalimba', 109: 'Bagpipe', 110: 'Fiddle', 111: 'Shanai',
    # Percussive
    112: 'Tinkle Bell', 113: 'Agogo', 114: 'Steel Drums',
    115: 'Woodblock', 116: 'Taiko Drum', 117: 'Melodic Tom',
    118: 'Synth Drum', 119: 'Reverse Cymbal',
    # Sound Effects
    120: 'Guitar Fret Noise', 121: 'Breath Noise', 122: 'Seashore',
    123: 'Bird Tweet', 124: 'Telephone Ring', 125: 'Helicopter',
    126: 'Applause', 127: 'Gunshot',
}

# 鼓组通常使用 program 114-119 以外的通道 10（GM 标准）
DRUM_KIT_NAME = 'Drums'

# GM 乐器音高范围 (MIDI note min, max)
# Program 112-127（打击乐/效果器）不受限制，MIDI note number 指代鼓件/音色而非音高
# 按 subtrack 细分音域（hand 分离用）
# 对钢琴类乐器：subtrack 0 = 右手（中高音区），subtrack 1 = 左手（低音区）
SUBTRACK_RANGES: dict[int, dict[int, tuple[int, int]]] = {
    p: {0: (48, 96), 1: (28, 72)} for p in range(8)  # piano 0-7
}
# ── 弦乐器 subtrack 级音域 — subtrack 0 = 一提/高音区, subtrack 1 = 二提/低音区
SUBTRACK_RANGES.update({
    40: {0: (62, 103), 1: (55, 88)},   # Violin
    41: {0: (55, 96), 1: (48, 84)},     # Viola
    42: {0: (42, 76), 1: (36, 67)},     # Cello
    43: {0: (32, 64), 1: (28, 55)},     # Contrabass
    44: {0: (62, 103), 1: (55, 88)},    # Tremolo Strings
    45: {0: (62, 103), 1: (55, 88)},    # Pizzicato Strings
})
# ── 弦乐合奏
for p in range(48, 52):
    SUBTRACK_RANGES[p] = {0: (55, 96), 1: (48, 84)}

# 30 个调名 → 7 个自然音级 (pitch class 0-11) 的映射
# 大调 = Ionian, 小调 = Aeolian (自然小调)
# key 格式与 REMITokenizer.KEY_NAMES 一致
KEY_TO_DIATONIC_PITCHES: dict[str, frozenset[int]] = {
    # ── 大调 ──
    'C':   frozenset({0, 2, 4, 5, 7, 9, 11}),
    'G':   frozenset({0, 2, 4, 6, 7, 9, 11}),
    'D':   frozenset({1, 2, 4, 6, 7, 9, 11}),
    'A':   frozenset({1, 2, 4, 6, 8, 9, 11}),
    'E':   frozenset({1, 3, 4, 6, 8, 9, 11}),
    'B':   frozenset({1, 3, 4, 6, 8, 10, 11}),
    'F#':  frozenset({1, 3, 5, 6, 8, 10, 11}),
    'C#':  frozenset({0, 1, 3, 5, 6, 8, 10}),
    'F':   frozenset({0, 2, 4, 5, 7, 9, 10}),
    'Bb':  frozenset({0, 2, 3, 5, 7, 9, 10}),
    'Eb':  frozenset({0, 2, 3, 5, 7, 8, 10}),
    'Ab':  frozenset({0, 1, 3, 5, 7, 8, 10}),
    'Db':  frozenset({0, 1, 3, 5, 6, 8, 10}),
    'Gb':  frozenset({1, 3, 5, 6, 8, 10, 11}),
    'Cb':  frozenset({1, 3, 4, 6, 8, 10, 11}),
    # ── 小调（自然小调） ──
    'Am':  frozenset({0, 2, 4, 5, 7, 9, 11}),
    'Em':  frozenset({0, 2, 4, 6, 7, 9, 11}),
    'Bm':  frozenset({1, 2, 4, 6, 7, 9, 11}),
    'F#m': frozenset({1, 2, 4, 6, 8, 9, 11}),
    'C#m': frozenset({1, 3, 4, 6, 8, 9, 11}),
    'G#m': frozenset({1, 3, 4, 6, 8, 10, 11}),
    'D#m': frozenset({1, 3, 5, 6, 8, 10, 11}),
    'A#m': frozenset({0, 1, 3, 5, 6, 8, 10}),
    'Dm':  frozenset({0, 2, 4, 5, 7, 9, 10}),
    'Gm':  frozenset({0, 2, 3, 5, 7, 9, 10}),
    'Cm':  frozenset({0, 2, 3, 5, 7, 8, 10}),
    'Fm':  frozenset({0, 1, 3, 5, 7, 8, 10}),
    'Bbm': frozenset({0, 1, 3, 5, 6, 8, 10}),
    'Ebm': frozenset({1, 3, 5, 6, 8, 10, 11}),
    'Abm': frozenset({1, 3, 4, 6, 8, 10, 11}),
}

GM_INSTRUMENT_RANGES: dict[int, tuple[int, int]] = {
    # ── Piano ──
    0: (21, 108), 1: (21, 108), 2: (21, 108), 3: (21, 108),
    4: (21, 108), 5: (21, 108), 6: (21, 108), 7: (21, 108),
    # ── Chromatic Percussion ──
    8: (60, 108), 9: (79, 108), 10: (60, 108), 11: (53, 108),
    12: (45, 99), 13: (65, 108), 14: (55, 79), 15: (48, 96),
    # ── Organ ──
    16: (36, 96), 17: (36, 96), 18: (36, 96), 19: (36, 96),
    20: (36, 96), 21: (48, 108), 22: (60, 84), 23: (48, 96),
    # ── Guitar ──
    24: (40, 88), 25: (40, 88), 26: (40, 88), 27: (40, 88),
    28: (40, 88), 29: (40, 88), 30: (40, 88), 31: (40, 88),
    # ── Bass ──
    32: (28, 60), 33: (28, 60), 34: (28, 60), 35: (28, 60),
    36: (28, 60), 37: (28, 60), 38: (24, 72), 39: (24, 72),
    # ── Strings ──
    40: (55, 103), 41: (48, 96), 42: (36, 76), 43: (28, 64),
    44: (55, 103), 45: (55, 103), 46: (35, 104), 47: (36, 62),
    # ── Ensemble ──
    48: (48, 96), 49: (48, 96), 50: (36, 96), 51: (36, 96),
    52: (48, 84), 53: (48, 84), 54: (36, 96), 55: (48, 96),
    # ── Brass ──
    56: (54, 89), 57: (40, 72), 58: (25, 52), 59: (54, 89),
    60: (48, 77), 61: (40, 89), 62: (36, 96), 63: (36, 96),
    # ── Reed ──
    64: (52, 82), 65: (49, 76), 66: (42, 69), 67: (34, 58),
    68: (58, 89), 69: (52, 77), 70: (34, 72), 71: (50, 96),
    # ── Pipe ──
    72: (74, 108), 73: (59, 96), 74: (60, 96), 75: (60, 96),
    76: (60, 96), 77: (60, 96), 78: (72, 100), 79: (60, 84),
    # ── Synth Lead ──
    80: (36, 96), 81: (36, 96), 82: (48, 96), 83: (48, 96),
    84: (48, 96), 85: (48, 96), 86: (48, 96), 87: (36, 96),
    # ── Synth Pad ──
    88: (36, 96), 89: (36, 96), 90: (36, 96), 91: (36, 96),
    92: (36, 96), 93: (36, 96), 94: (36, 96), 95: (36, 96),
    # ── Synth Effects ──
    96: (24, 96), 97: (24, 96), 98: (24, 96), 99: (24, 96),
    100: (24, 96), 101: (24, 96), 102: (24, 96), 103: (24, 96),
    # ── Ethnic ──
    104: (55, 96), 105: (55, 96), 106: (55, 96), 107: (48, 96),
    108: (48, 84), 109: (58, 79), 110: (55, 103), 111: (55, 96),
    # ── Percussive (112-119) + Sound Effects (120-127): 无音高限制 ──
}


# ── 乐器级复音上限 ─────────────────────────────────────────────
# 同 position 每 track 最大同时发音数，key = (program_start, program_end)
INSTRUMENT_POLYPHONY_CAP: dict[tuple[int, int], int] = {
    (0, 7): 10,      # Piano — full chords
    (8, 15): 6,      # Chromatic Percussion
    (16, 23): 6,     # Organ
    (24, 31): 4,     # Guitar
    (32, 39): 2,     # Bass — monophonic
    (40, 47): 2,     # Strings — monophonic
    (48, 51): 4,     # Ensemble strings
    (52, 55): 3,     # Vocal
    (56, 63): 2,     # Brass — monophonic
    (64, 71): 2,     # Reed/Woodwind — monophonic
    (72, 79): 2,     # Pipe/Flute — monophonic
    (80, 103): 6,    # Synth
    (104, 111): 2,   # Ethnic
    (112, 127): 8,   # Percussion/Effects
}


def get_polyphony_cap(program: int) -> int:
    """根据 program 返回乐器级复音上限。"""
    for (start, end), cap in INSTRUMENT_POLYPHONY_CAP.items():
        if start <= program <= end:
            return cap
    return 10  # fallback


def _parse_program(token_str: str) -> int:
    """从 '<Program N_M>' 或 '<Program N>' 提取 program number N。"""
    # token_str 形如 '<Program 0>' 或 '<Program 0_1>'
    val = token_str[len('<Program') + 1:-1]  # 去掉 '<Program ' 和 '>'
    return int(val.split('_')[0])


def _parse_subtrack(token_str: str) -> int:
    """从 '<Program N_M>' 提取 subtrack M，无后缀时返回 0。"""
    val = token_str[len('<Program') + 1:-1]
    parts = val.split('_')
    return int(parts[1]) if len(parts) > 1 else 0


@torch.no_grad()
def generate(model: MusicTransformer, tokenizer: REMITokenizer,
             seed_tokens: list[int], max_bars: int = 32,
             max_new_tokens: int = 4096, temperature: float = 1.0,
             top_k: int = 20) -> list[int]:
    """自回归生成 token 序列 (使用 KV cache)。

    Args:
        model: 训练好的模型
        tokenizer: REMI tokenizer
        seed_tokens: 种子 token ID 列表（前缀）
        max_bars: 最多生成多少个小节
        max_new_tokens: 最多生成多少新 token
        temperature: 采样温度
        top_k: top-k 采样

    Returns:
        完整的 token ID 序列（seed + generated）
    """
    model.eval()
    device = next(model.parameters()).device

    seed = torch.tensor([seed_tokens], dtype=torch.long, device=device)
    bar_id = tokenizer.bar_token_id
    eos_id = tokenizer.eos_token_id

    # ── 音高限制准备 ──────────────────────────────────────────
    _OFFSET = 60
    # 预计算每个半音程的 NOTE_ON token ID（-60 .. +60）
    note_on_ids = [tokenizer.encode_token(f'<Note_ON {i - _OFFSET}>')
                   for i in range(121)]
    # Program token 前缀（不含末尾 >，匹配用 startswith）
    _prog_prefix = '<Program'
    _key_prefix = '<Key'
    # 从 seed 中查找当前 program 和 key
    cur_program: int | None = None
    current_key: str | None = None
    for tid in reversed(seed_tokens):
        ts = tokenizer.decode_token(tid)
        if ts.startswith(_prog_prefix):
            cur_program = _parse_program(ts)
            break
    for tid in reversed(seed_tokens):
        ts = tokenizer.decode_token(tid)
        if ts.startswith(_key_prefix):
            current_key = ts[len(_key_prefix) + 1:-1]
            break
    tonic_midi = key_name_to_tonic_midi(current_key)
    # ──────────────────────────────────────────────────────────

    # 初始化 KV cache
    kv_caches = [[None, None] for _ in range(model.config.n_layers)]

    generated = seed.clone()
    bar_count = seed_tokens.count(bar_id)

    # measure_ids 追踪（KV cache 模式下只有最新 token 输入模型，需外部维护）
    cached_measure_ids = torch.cumsum((seed[0] == bar_id).int(), dim=0)  # (T_seed,)
    seed_measure_count = cached_measure_ids[-1].item()  # 累计小节数

    # 首轮 forward 处理全部 seed
    next_token = seed
    ctx_len = seed.size(1)

    for _ in range(max_new_tokens):
        if ctx_len > model.config.max_seq_len:
            next_token = generated[:, -1:]

        logits = model.forward(next_token, kv_caches=kv_caches,
                               measure_ids=cached_measure_ids)  # (1, T', V)
        logits = logits[:, -1, :] / temperature  # (1, V)

        # ── 音高限制：遮盖当前乐器音域外的 NOTE_ON token ──
        if cur_program is not None and cur_program < 112:
            rmin, rmax = GM_INSTRUMENT_RANGES.get(cur_program, (0, 127))
            for i in range(121):
                midi_pitch = tonic_midi + (i - _OFFSET)
                if midi_pitch < rmin or midi_pitch > rmax:
                    logits[0, note_on_ids[i]] = float('-inf')
        # ──────────────────────────────────────────────────

        # top-k
        if top_k > 0:
            vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < vals[:, -1:]] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

        generated = torch.cat([generated, next_token], dim=1)
        ctx_len = generated.size(1)
        token_id = next_token.item()

        # 更新 measure_ids 追踪
        is_bar = (token_id == bar_id)
        new_measure = seed_measure_count + bar_count + (1 if is_bar else 0)
        cached_measure_ids = torch.cat(
            [cached_measure_ids, torch.tensor([new_measure], device=device)]
        )

        # 追踪 Program / Key 变化
        ts = tokenizer.decode_token(token_id)
        if ts.startswith(_prog_prefix):
            cur_program = _parse_program(ts)
        elif ts.startswith(_key_prefix):
            current_key = ts[len(_key_prefix) + 1:-1]
            tonic_midi = key_name_to_tonic_midi(current_key)

        if token_id == bar_id:
            bar_count += 1
            if bar_count >= max_bars:
                break

        if token_id == eos_id:
            break

    return generated[0].tolist()


def tokens_to_notes(token_ids: list[int],
                    tokenizer: REMITokenizer) -> list[dict]:
    """将 token 序列解析为音符列表。"""
    events = tokenizer.detokenize(token_ids)

    notes_list = []
    cur_bar = 0
    cur_pos = 0
    cur_program = 0
    cur_subtrack = 0
    cur_tuplet = None       # (actual, normal) or None
    cur_grace_type = None   # 'acciaccatura', 'appoggiatura', 'grace' or None
    current_tonic = 60      # 默认 C4，遇到 KEY token 时更新
    cur_timesig = '4/4'     # 默认拍号
    i = 0

    while i < len(events):
        etype, evalue = events[i]

        if etype == REMITokenizer.BOS:
            i += 1
            continue
        elif etype == REMITokenizer.EOS:
            break
        elif etype == REMITokenizer.BAR:
            cur_bar += 1
            i += 1
            continue
        elif etype == REMITokenizer.POSITION:
            cur_pos = evalue
            i += 1
            continue
        elif etype == REMITokenizer.PROGRAM:
            val_parts = evalue.split('_')
            cur_program = int(val_parts[0])
            cur_subtrack = int(val_parts[1]) if len(val_parts) > 1 else 0
            i += 1
            continue
        elif etype == REMITokenizer.TUPLET_START:
            # evalue is a string like '3:2'
            parts = evalue.split(':')
            cur_tuplet = (int(parts[0]), int(parts[1]))
            i += 1
            continue
        elif etype == REMITokenizer.TUPLET_END:
            cur_tuplet = None
            i += 1
            continue
        elif etype == REMITokenizer.GRACE_NOTE:
            cur_grace_type = evalue
            i += 1
            continue
        elif etype == REMITokenizer.REST:
            dur = 4  # default
            if i + 1 < len(events) and events[i + 1][0] == REMITokenizer.DURATION:
                dur = events[i + 1][1]
                i += 1
            notes_list.append({
                'bar': cur_bar,
                'position': cur_pos,
                'program': cur_program,
                'subtrack': cur_subtrack,
                'type': 'rest',
                'pitch': -1,
                'velocity': 0,
                'duration': dur,
                'tuplet': cur_tuplet,
            })
            i += 1
        elif etype == REMITokenizer.NOTE_ON:
            pitch = current_tonic + evalue  # interval → 绝对 MIDI 音高
            # 接下来应该是 Velocity 和 Duration
            vel = 6  # default
            dur = 4  # default
            if i + 1 < len(events) and events[i + 1][0] == REMITokenizer.VELOCITY:
                vel = events[i + 1][1]
                i += 1
            if i + 1 < len(events) and events[i + 1][0] == REMITokenizer.DURATION:
                dur = events[i + 1][1]
                i += 1

            note_entry = {
                'bar': cur_bar,
                'position': cur_pos,
                'program': cur_program,
                'subtrack': cur_subtrack,
                'pitch': pitch,
                'velocity': vel,
                'duration': dur,
                'tuplet': cur_tuplet,
            }
            if cur_grace_type is not None:
                note_entry['grace_type'] = cur_grace_type
                cur_grace_type = None
            notes_list.append(note_entry)
            i += 1
        # ── 表情记号（力度/踏板/演奏法等） ────────────────
        elif etype == REMITokenizer.DYNAMIC:
            notes_list.append({
                'bar': cur_bar, 'position': cur_pos,
                'program': cur_program, 'subtrack': cur_subtrack,
                'type': 'dynamic', 'value': evalue,
            })
            i += 1
            continue
        elif etype == REMITokenizer.HAIRPIN:
            notes_list.append({
                'bar': cur_bar, 'position': cur_pos,
                'program': cur_program, 'subtrack': cur_subtrack,
                'type': 'hairpin', 'value': evalue,
            })
            i += 1
            continue
        elif etype == REMITokenizer.PEDAL:
            notes_list.append({
                'bar': cur_bar, 'position': cur_pos,
                'program': cur_program, 'subtrack': cur_subtrack,
                'type': 'pedal', 'value': evalue,
            })
            i += 1
            continue
        elif etype == REMITokenizer.ARTIC:
            notes_list.append({
                'bar': cur_bar, 'position': cur_pos,
                'program': cur_program, 'subtrack': cur_subtrack,
                'type': 'artic', 'value': evalue,
            })
            i += 1
            continue
        elif etype == REMITokenizer.ORNAMENT:
            notes_list.append({
                'bar': cur_bar, 'position': cur_pos,
                'program': cur_program, 'subtrack': cur_subtrack,
                'type': 'ornament', 'value': evalue,
            })
            i += 1
            continue
        elif etype == REMITokenizer.SLUR:
            notes_list.append({
                'bar': cur_bar, 'position': cur_pos,
                'program': cur_program, 'subtrack': cur_subtrack,
                'type': 'slur', 'value': evalue,
            })
            i += 1
            continue
        elif etype == REMITokenizer.OCTAVE:
            notes_list.append({
                'bar': cur_bar, 'position': cur_pos,
                'program': cur_program, 'subtrack': cur_subtrack,
                'type': 'octave', 'value': evalue,
            })
            i += 1
            continue
        elif etype == REMITokenizer.ARPEGGIO:
            notes_list.append({
                'bar': cur_bar, 'position': cur_pos,
                'program': cur_program, 'subtrack': cur_subtrack,
                'type': 'arpeggio',
            })
            i += 1
            continue
        # ── KEY — 追踪主音，用于 interval → 绝对音高转换 ──
        elif etype == REMITokenizer.KEY:
            current_tonic = key_name_to_tonic_midi(evalue)
            i += 1
            continue
        elif etype == REMITokenizer.TIMESIG:
            cur_timesig = evalue
            notes_list.append({
                'bar': cur_bar, 'position': cur_pos,
                'program': cur_program, 'subtrack': cur_subtrack,
                'type': 'timesig', 'value': evalue,
            })
            i += 1
            continue
        # ── 其他暂不支持的标记 ──────────────────────────
        elif etype in (REMITokenizer.CLEF,
                       REMITokenizer.REPEAT, REMITokenizer.JUMP, REMITokenizer.TEMPO,
                       REMITokenizer.BEAT,
                       REMITokenizer.BASS, REMITokenizer.ANTICIPATE):
            i += 1
            continue
        else:
            i += 1

    return notes_list


def notes_to_score(notes_list: list[dict],
                   grid_size: int = 16,
                   max_bars: int = 256) -> stream.Score:
    """将音符列表重建为 music21 Score（含力度、踏板等表情记号）。"""
    from music21 import note as note21, duration as dur21
    from music21 import dynamics, expressions, articulations as art21

    quarter_per_pos = 4.0 / grid_size

    # 按小节分组，key = (program, subtrack)
    bars: dict[int, dict[Tuple[int, int], list]] = {}
    for n in notes_list:
        b = n['bar']
        key = (n['program'], n['subtrack'])
        bars.setdefault(b, {}).setdefault(key, []).append(n)

    if not bars:
        return stream.Score()

    # 收集所有 (program, subtrack) 组合
    all_keys: set[Tuple[int, int]] = set()
    for bar_data in bars.values():
        all_keys.update(bar_data.keys())
    all_keys_sorted = sorted(all_keys, key=lambda k: (k[0], k[1]))

    # 动态创建分部
    parts: dict[Tuple[int, int], stream.Part] = {}
    for key in all_keys_sorted:
        part = stream.Part()
        prog, sub = key
        instr_name = GM_INSTRUMENT_NAMES.get(prog, f'Program {prog}')
        if prog == 0:
            part.partName = f'{instr_name} (Right)' if sub == 0 else f'{instr_name} (Left)'
            part.insert(clef.TrebleClef() if sub == 0 else clef.BassClef())
        else:
            part.partName = instr_name if sub == 0 else f'{instr_name} {sub+1}'
            part.insert(clef.TrebleClef())
        parts[key] = part

    max_existing_bar = max(bars.keys())
    cur_timesig = '4/4'     # 默认拍号，遇 TimeSig event 更新

    # 收集踏板/渐强渐弱/八度记号事件（music21 Spanner 序列化有缺陷，后处理注入 XML）
    _pedal_events: list[dict] = []
    _hairpin_events: list[dict] = []
    _octave_events: list[dict] = []

    for bar_idx in range(1, max_existing_bar + 1):
        if bar_idx > max_bars:
            break

        # 查找本小节是否有拍号变更
        for key in all_keys_sorted:
            for entry in bars.get(bar_idx, {}).get(key, []):
                if entry.get('type') == 'timesig':
                    cur_timesig = entry['value']
                    break

        meas_by_key: dict[Tuple[int, int], stream.Measure] = {}
        for key in all_keys_sorted:
            m = stream.Measure()
            m.number = bar_idx
            m.timeSignature = meter.TimeSignature(cur_timesig)
            meas_by_key[key] = m

        for key in all_keys_sorted:
            all_entries = bars.get(bar_idx, {}).get(key, [])
            meas = meas_by_key[key]
            # 分离音符和表情记号
            hand_notes = [n for n in all_entries if n.get('type') in (None, 'rest')]
            markings = [n for n in all_entries if n not in hand_notes]
            # Pre-compute tuplet-adjusted offsets and duration scales
            hand_sorted = sorted(hand_notes, key=lambda n: (n['position'], n.get('pitch', -1)))
            for n in hand_sorted:
                n['_adj_offset'] = n['position'] * quarter_per_pos
                n['_adj_dur_scale'] = 1.0

            i = 0
            while i < len(hand_sorted):
                n = hand_sorted[i]
                if n.get('tuplet'):
                    actual, normal = n['tuplet']
                    j = i
                    while j < len(hand_sorted) and hand_sorted[j].get('tuplet') == (actual, normal):
                        j += 1
                    start_pos = hand_sorted[i]['position']
                    # unique positions in this tuplet group, in order
                    seen_pos = []
                    for k in range(i, j):
                        p = hand_sorted[k]['position']
                        if p not in seen_pos:
                            seen_pos.append(p)
                    for k in range(i, j):
                        note = hand_sorted[k]
                        pos_rank = seen_pos.index(note['position'])
                        adj_pos = start_pos + pos_rank * (normal / actual)
                        note['_adj_offset'] = adj_pos * quarter_per_pos
                        note['_adj_dur_scale'] = normal / actual
                    i = j
                else:
                    i += 1

            # 按 position 分组
            pos_groups: dict[int, list] = {}
            for n in hand_notes:
                pos_groups.setdefault(n['position'], []).append(n)

            # 倚音辅助函数
            def make_grace_note(gn_entry):
                gt = gn_entry.get('grace_type', 'grace')
                gn_obj = note21.Note(gn_entry['pitch'])
                gn_obj.duration = dur21.GraceDuration('eighth')
                gn_obj.duration.slash = (gt == 'acciaccatura')
                gn_obj.volume.velocity = min(127, gn_entry['velocity'] * 16)
                return gn_obj

            for pos in sorted(pos_groups.keys()):
                notes_at_pos = pos_groups[pos]

                # 分类：倚音 / 休止符 / 主音符
                grace_notes = [n for n in notes_at_pos if n.get('grace_type')]
                rest_items = [n for n in notes_at_pos if n.get('type') == 'rest']
                regular_notes = [n for n in notes_at_pos
                                 if not n.get('grace_type') and n.get('type') != 'rest']

                # 休止符
                for r in rest_items:
                    dur_clamped = min(r['duration'], grid_size - r['position'])
                    dur_clamped = max(1, dur_clamped)
                    rst = note21.Rest()
                    rst.duration = dur21.Duration(dur_clamped * r['_adj_dur_scale'] * quarter_per_pos)
                    meas.insert(r['_adj_offset'], rst)

                if not regular_notes:
                    if grace_notes:
                        logger.warning("Grace notes without main note at bar %d pos %d", bar_idx, pos)
                    continue

                # 倚音 — 插入 measure 中，位于主音之前（同 offset，靠插入顺序区分）
                for gn in grace_notes:
                    meas.insert(gn['_adj_offset'], make_grace_note(gn))

                # 主音符 — 单音或和弦
                if len(regular_notes) == 1:
                    n = regular_notes[0]
                    dur_clamped = min(n['duration'], grid_size - n['position'])
                    dur_clamped = max(1, dur_clamped)
                    nt = note21.Note(n['pitch'])
                    nt.duration = dur21.Duration(dur_clamped * n['_adj_dur_scale'] * quarter_per_pos)
                    nt.volume.velocity = min(127, n['velocity'] * 16)
                    meas.insert(n['_adj_offset'], nt)
                    if n.get('tuplet'):
                        actual, normal = n['tuplet']
                        t = dur21.Tuplet()
                        t.numberNotesActual = actual
                        t.numberNotesNormal = normal
                        nt.duration.tuplets = (t,)
                else:
                    chord_notes = []
                    for n in regular_notes:
                        dur_clamped = min(n['duration'], grid_size - n['position'])
                        dur_clamped = max(1, dur_clamped)
                        nt = note21.Note(n['pitch'])
                        nt.duration = dur21.Duration(dur_clamped * n['_adj_dur_scale'] * quarter_per_pos)
                        chord_notes.append(nt)
                    ch = chord.Chord(chord_notes)
                    meas.insert(regular_notes[0]['_adj_offset'], ch)
                    if regular_notes[0].get('tuplet'):
                        actual, normal = regular_notes[0]['tuplet']
                        t = dur21.Tuplet()
                        t.numberNotesActual = actual
                        t.numberNotesNormal = normal
                        ch.duration.tuplets = (t,)

            # ── 插入表情记号（力度/踏板/渐强渐弱/演奏法） ──
            for m in markings:
                offset = m['position'] * quarter_per_pos
                if m['type'] == 'dynamic':
                    meas.insert(offset, dynamics.Dynamic(m['value']))
                elif m['type'] == 'pedal':
                    _pedal_events.append({
                        'bar': bar_idx, 'key': key,
                        'pos': m['position'], 'value': m['value'],
                    })
                elif m['type'] == 'hairpin':
                    _hairpin_events.append({
                        'bar': bar_idx, 'key': key,
                        'pos': m['position'], 'value': m['value'],
                    })
                elif m['type'] == 'artic':
                    _ARTIC_MAP = {
                        'staccato': art21.Staccato(), 'accent': art21.Accent(),
                        'tenuto': art21.Tenuto(), 'marcato': art21.StrongAccent(),
                        'pizzicato': art21.Pizzicato(),
                    }
                    # fermata 位于 expressions 而非 articulations
                    if m['value'] == 'fermata':
                        for el in meas.notesAndRests:
                            if abs(el.offset - offset) < 1e-6:
                                if isinstance(el, note21.Note):
                                    el.expressions.append(expressions.Fermata())
                                elif isinstance(el, chord.Chord):
                                    for sub_n in el.notes:
                                        sub_n.expressions.append(expressions.Fermata())
                                break
                    art_obj = _ARTIC_MAP.get(m['value'])
                    if art_obj:
                        for el in meas.notesAndRests:
                            if abs(el.offset - offset) < 1e-6:
                                if isinstance(el, note21.Note):
                                    el.articulations.append(art_obj)
                                elif isinstance(el, chord.Chord):
                                    for sub_n in el.notes:
                                        sub_n.articulations.append(art_obj)
                                break
                elif m['type'] == 'ornament':
                    _ORN_MAP = {
                        'trill': expressions.Trill(),
                        'mordent': expressions.Mordent(),
                        'turn': expressions.Turn(),
                        'tremolo': expressions.Tremolo(),
                    }
                    orn_obj = _ORN_MAP.get(m['value'])
                    if orn_obj:
                        for el in meas.notesAndRests:
                            if abs(el.offset - offset) < 1e-6:
                                if isinstance(el, note21.Note):
                                    el.expressions.append(orn_obj)
                                elif isinstance(el, chord.Chord):
                                    for sub_n in el.notes:
                                        sub_n.expressions.append(orn_obj)
                                break
                elif m['type'] == 'arpeggio':
                    for el in meas.notesAndRests:
                        if abs(el.offset - offset) < 1e-6:
                            if isinstance(el, chord.Chord):
                                el.expressions.append(expressions.ArpeggioMark())
                            break
                elif m['type'] == 'octave':
                    _octave_events.append({
                        'bar': bar_idx, 'key': key,
                        'pos': m['position'], 'value': m['value'],
                    })

        for key in all_keys_sorted:
            parts[key].append(meas_by_key[key])

    score = stream.Score(list(parts.values()))
    if _pedal_events:
        score._pedal_events = _pedal_events
    if _hairpin_events:
        score._hairpin_events = _hairpin_events
    if _octave_events:
        score._octave_events = _octave_events
    score._all_keys = all_keys_sorted
    return score


# ── 方向标记后处理 ─────────────────────────────────────────

def _inject_directions_in_musicxml(filepath: str,
                                   pedal_events: list[dict],
                                   hairpin_events: list[dict],
                                   octave_events: list[dict] = None,
                                   all_keys: list = None):
    """MusicXML 后处理：注入踏板、渐强/渐弱、八度方向标记。

    music21 的 PedalMark/Crescendo/Diminuendo/Ottava 都是 Spanner 子类，
    直接插入 measure 后无法正确序列化，因此改为在 XML 文件中直接注入。

    多声部时同一个小节号会出现在多个 <part> 中，通过 all_keys 映射
    event['key'] → target part 来确定正确的出现位置。
    """
    import re
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # key → part occurrence index (1-based)
    key_to_occurrence = {}
    if all_keys:
        key_to_occurrence = {tuple(k): i + 1 for i, k in enumerate(all_keys)}

    def _inject_into_measure(content, bar_num, xml_snippet, target_occurrence):
        """将 xml_snippet 注入到第 target_occurrence 个 <measure number="bar_num"> 内。"""
        pattern = rf'(<measure[^>]*?number="{bar_num}"[^>]*>.*?)(</measure>)'
        matches = list(re.finditer(pattern, content, re.DOTALL))
        if target_occurrence > len(matches):
            return content
        m = matches[target_occurrence - 1]
        return content[:m.start()] + m.group(1) + '\n' + xml_snippet + '\n    ' + m.group(2) + content[m.end():]

    def _inject(events, xml_template):
        nonlocal content
        for ev in events:
            bar_num = ev['bar']
            target = key_to_occurrence.get(tuple(ev.get('key', ())), 1)
            xml_snippet = xml_template(ev['value'])
            content = _inject_into_measure(content, bar_num, xml_snippet, target)

    # 踏板
    _inject(pedal_events,
            lambda v: (
                '      <direction placement="below">'
                '\n        <direction-type>'
                f'\n          <pedal type="{v}" line="yes" sign="yes"/>'
                '\n        </direction-type>'
                '\n      </direction>'
            ))
    # 渐强/渐弱（wedge）
    _inject(hairpin_events,
            lambda v: (
                '      <direction placement="below">'
                '\n        <direction-type>'
                f'\n          <wedge type="{"crescendo" if v == "cresc" else "diminuendo"}"'
                ' number="1" spread="0"/>'
                '\n        </direction-type>'
                '\n      </direction>'
            ))

    # 八度记号（octave-shift）
    if octave_events:
        for ev in octave_events:
            bar_num = ev['bar']
            target = key_to_occurrence.get(tuple(ev.get('key', ())), 1)
            val = ev['value']
            if val == 'end':
                xml_snippet = (
                    '      <direction placement="below">'
                    '\n        <direction-type>'
                    '\n          <octave-shift type="stop" size="8" number="1"/>'
                    '\n        </direction-type>'
                    '\n      </direction>'
                )
            else:
                is_down = 'b' in val
                size = val.replace('va', '').replace('vb', '').replace('ma', '').replace('mb', '')
                shift_type = 'down' if is_down else 'up'
                xml_snippet = (
                    '      <direction placement="below">'
                    '\n        <direction-type>'
                    f'\n          <octave-shift type="{shift_type}" size="{size}" number="1"/>'
                    '\n        </direction-type>'
                    '\n      </direction>'
                )
            content = _inject_into_measure(content, bar_num, xml_snippet, target)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)


def generate_to_musicxml(model: MusicTransformer, tokenizer: REMITokenizer,
                         seed_tokens: list[int], output_path: str,
                         max_bars: int = 32, temperature: float = 1.0,
                         top_k: int = 20) -> str:
    """生成并导出为 MusicXML 文件。"""
    full_tokens = generate(
        model, tokenizer, seed_tokens,
        max_bars=max_bars, temperature=temperature, top_k=top_k,
    )
    notes = tokens_to_notes(full_tokens, tokenizer)
    score = notes_to_score(notes, grid_size=tokenizer.grid_size, max_bars=max_bars)
    score.write('musicxml', fp=output_path)
    _inject_directions_in_musicxml(
        output_path,
        pedal_events=getattr(score, '_pedal_events', []),
        hairpin_events=getattr(score, '_hairpin_events', []),
        octave_events=getattr(score, '_octave_events', []),
        all_keys=getattr(score, '_all_keys', None),
    )
    _cleanup_accidentals(output_path)
    logger.info(f'生成完成: {output_path}')
    return output_path


# ── 还原记号后处理 ─────────────────────────────────────────

# 调号 fifths → 各 step 的默认 alter 值
_KEY_SIG_ALTER: dict[int, dict[str, int]] = {
    -7: {'C':-1,'D':-1,'E':-1,'F':-1,'G':-1,'A':-1,'B':-1},
    -6: {'C':-1,'D':-1,'E':-1,'F':-1,'G':-1,'A':-1,'B': 0},
    -5: {'C':-1,'D': 0,'E':-1,'F':-1,'G':-1,'A':-1,'B': 0},
    -4: {'C':-1,'D': 0,'E':-1,'F':-1,'G': 0,'A':-1,'B': 0},
    -3: {'C': 0,'D': 0,'E':-1,'F':-1,'G': 0,'A': 0,'B': 0},
    -2: {'C': 0,'D': 0,'E':-1,'F': 0,'G': 0,'A': 0,'B': 0},
    -1: {'C': 0,'D': 0,'E': 0,'F':-1,'G': 0,'A': 0,'B': 0},
     0: {'C': 0,'D': 0,'E': 0,'F': 0,'G': 0,'A': 0,'B': 0},
     1: {'C': 0,'D': 0,'E': 0,'F': 1,'G': 0,'A': 0,'B': 0},
     2: {'C': 0,'D': 0,'E': 0,'F': 1,'G': 1,'A': 0,'B': 0},
     3: {'C': 0,'D': 0,'E': 0,'F': 1,'G': 1,'A': 1,'B': 0},
     4: {'C': 0,'D': 0,'E': 1,'F': 1,'G': 1,'A': 1,'B': 0},
     5: {'C': 0,'D': 1,'E': 1,'F': 1,'G': 1,'A': 1,'B': 0},
     6: {'C': 0,'D': 1,'E': 1,'F': 1,'G': 1,'A': 1,'B': 1},
     7: {'C': 1,'D': 1,'E': 1,'F': 1,'G': 1,'A': 1,'B': 1},
}


def _cleanup_accidentals(path: str):
    """清理 MusicXML 中多余的还原记号。

    逐小节扫描，只保留确实在"取消"某个升降号的还原号：
    - 如果本小节内该音级有前置升降号（alter≠0）→ 保留
    - 如果调号对该音级有默认升降 → 保留
    - 否则 → 去掉
    """
    import re
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    measure_pattern = re.compile(r'(<measure[^>]*>.*?</measure>)', re.DOTALL)
    key_fifths_pattern = re.compile(r'<fifths>(-?\d+)</fifths>')
    step_pattern = re.compile(r'<step>([A-G])</step>')
    alter_pattern = re.compile(r'<alter>(-?\d+)</alter>')
    note_pattern = re.compile(r'<note[^>]*>.*?</note>', re.DOTALL)

    def _get_default_alter(step: str, fifths: int) -> int:
        return _KEY_SIG_ALTER.get(fifths, {}).get(step, 0)

    def _process_measure(measure_xml: str) -> str:
        fifths = 0
        m = key_fifths_pattern.search(measure_xml)
        if m:
            fifths = int(m.group(1))
        altered_steps: set[str] = set()

        for nb in note_pattern.finditer(measure_xml):
            note_xml = nb.group(0)
            step_m = step_pattern.search(note_xml)
            if not step_m:
                continue
            step = step_m.group(1)
            alter_m = alter_pattern.search(note_xml)
            alter = int(alter_m.group(1)) if alter_m else 0
            default = _get_default_alter(step, fifths)

            if alter != default:
                altered_steps.add(step)

            if '<accidental>natural</accidental>' in note_xml:
                needed = (step in altered_steps) or (default != 0 and alter == 0)
                if not needed:
                    cleaned_note = re.sub(
                        r'\s*<accidental>natural</accidental>\s*\n?', '\n', note_xml, count=1
                    )
                    measure_xml = measure_xml.replace(note_xml, cleaned_note, 1)
        return measure_xml

    result = measure_pattern.sub(lambda m: _process_measure(m.group(1)), content)

    if result != content:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(result)
