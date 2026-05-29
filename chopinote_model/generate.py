"""推理生成模块：自回归采样 → MusicXML 导出。"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .model import MusicTransformer
from .config import ModelConfig, NO_SECTION_ID, NO_SECTION_TYPE_ID
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
            'temperature': (0.55, 2.5),
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
    generated_bars = 0  # 仅计数新生成的小节（不含 seed）

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
        logits = logits[:, -1, :] / max(temperature, 1e-8)  # (1, V)

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
        new_measure = seed_measure_count + generated_bars + (1 if is_bar else 0)
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
            generated_bars += 1
            if bar_count >= max_bars:
                break

        if token_id == eos_id:
            break

    return generated[0].tolist()


def generate_to_musicxml(model: MusicTransformer, tokenizer: REMITokenizer,
                         seed_tokens: list[int], output_path: str,
                         max_bars: int = 32, temperature: float = 1.0,
                         top_k: int = 20) -> str:
    """生成并导出为 MusicXML 文件。"""
    from chopinote_dataset.renderer import REMIToMusicXML

    full_tokens = generate(
        model, tokenizer, seed_tokens,
        max_bars=max_bars, temperature=temperature, top_k=top_k,
    )
    renderer = REMIToMusicXML(grid_size=tokenizer.grid_size,
                               velocity_levels=tokenizer.velocity_levels)
    renderer.render_from_tokens(full_tokens, output_path)
    logger.info(f'生成完成: {output_path}')
    return output_path


# ═══════════════════════════════════════════════════════════════════
#  段落感知生成（两层生成：结构规划 → 细节填充）
# ═══════════════════════════════════════════════════════════════════

SECTION_PARAMS: dict[str, dict] = {
    'theme1':       {'temperature': 0.9,  'key_bias_strength': 2.0, 'complexity': 5.0},
    'theme2':       {'temperature': 1.0,  'key_bias_strength': 1.5, 'complexity': 4.0},
    'development':  {'temperature': 1.3,  'key_bias_strength': 0.5, 'complexity': 7.0},
    'bridge':       {'temperature': 1.1,  'key_bias_strength': 1.0, 'complexity': 3.0},
    'cadenza':      {'temperature': 1.4,  'key_bias_strength': 0.0, 'complexity': 8.0},
    'recapitulation': {'temperature': 0.8, 'key_bias_strength': 2.5, 'complexity': 5.0},
    'coda':         {'temperature': 0.9,  'key_bias_strength': 2.0, 'complexity': 4.0},
    'exposition':   {'temperature': 0.9,  'key_bias_strength': 2.0, 'complexity': 5.0},
    'development_s': {'temperature': 1.3, 'key_bias_strength': 0.5, 'complexity': 7.0},
    'intro':        {'temperature': 0.8,  'key_bias_strength': 1.5, 'complexity': 3.0},
    'transition':   {'temperature': 1.0,  'key_bias_strength': 0.5, 'complexity': 4.0},
    'variation':    {'temperature': 1.1,  'key_bias_strength': 1.5, 'complexity': 5.0},
    'episode':      {'temperature': 1.2,  'key_bias_strength': 0.0, 'complexity': 6.0},
}

# ── 段落-和弦参数联动（13 种） ─────────────────────────────────
SECTION_HARMONY_PARAMS: dict[str, dict] = {
    'exposition':       {'complexity': 5.0, 'cadence_strength': 2.0, 'density': 0.5},
    'development':      {'complexity': 7.0, 'cadence_strength': 0.5, 'density': 0.8},
    'development_s':    {'complexity': 7.0, 'cadence_strength': 0.5, 'density': 0.8},
    'recapitulation':   {'complexity': 5.0, 'cadence_strength': 2.5, 'density': 0.5},
    'theme1':           {'complexity': 5.0, 'cadence_strength': 2.0, 'density': 0.5},
    'theme2':           {'complexity': 4.0, 'cadence_strength': 1.5, 'density': 0.4},
    'bridge':           {'complexity': 4.0, 'cadence_strength': 0.8, 'density': 0.3},
    'transition':       {'complexity': 4.0, 'cadence_strength': 0.5, 'density': 0.3},
    'intro':            {'complexity': 3.0, 'cadence_strength': 1.0, 'density': 0.3},
    'coda':             {'complexity': 3.0, 'cadence_strength': 3.0, 'density': 0.2},
    'cadenza':          {'complexity': 8.0, 'cadence_strength': 0.0, 'density': 0.9},
    'variation':        {'complexity': 5.0, 'cadence_strength': 1.5, 'density': 0.5},
    'episode':          {'complexity': 6.0, 'cadence_strength': 0.0, 'density': 0.7},
}


# Section 名 → token 前缀字符串（从 tokenizer SECTION_NAMES 自动生成，消除手工同步）
def _build_section_token_map(tokenizer):
    return {name: f'Section {name}' for name in tokenizer.SECTION_NAMES}


@torch.no_grad()
def generate_structure_plan(model: MusicTransformer, tokenizer: REMITokenizer,
                            seed_tokens: list[int],
                            form_constraint: Optional[dict] = None,
                            max_new_tokens: int = 32,
                            temperature: float = 1.0,
                            top_k: int = 20) -> list[int]:
    """Stage 1: 结构规划 — 只生成结构 token 的段落规划。

    Args:
        model: MusicTransformer
        tokenizer: REMI tokenizer
        seed_tokens: 种子 token 序列（前缀）
        form_constraint: 可选，如 {"form": "sonata", "total_bars": 64}
        max_new_tokens: 最多生成 token 数
        temperature: 采样温度
        top_k: top-k 采样

    Returns:
        结构 token 序列（包含 Section / Key / Anticipate 等结构 token）
    """
    model.eval()
    device = next(model.parameters()).device
    eos_id = tokenizer.eos_token_id

    # 从 tokenizer 动态构建 section token 名→字符串映射（零手工同步）
    section_token_map = _build_section_token_map(tokenizer)

    # 计算结构 token 的 ID 集合（Section, Key, Bar, Anticipate, Tempo, TimeSig 等）
    section_ids_set = set()
    for name in section_token_map.values():
        tid = tokenizer.encode_token(f'<{name}>')
        if tid != tokenizer.encode_token('<MASK>'):
            section_ids_set.add(tid)

    # 额外允许的结构 token
    structure_prefixes = [
        tokenizer.KEY, tokenizer.BAR, tokenizer.ANTICIPATE,
        tokenizer.TEMPO, tokenizer.TIMESIG, tokenizer.SEC_SUM,
    ]
    for tid in range(tokenizer.vocab_size):
        ts = tokenizer.decode_token(tid)
        if any(ts.startswith(p) for p in structure_prefixes):
            section_ids_set.add(tid)

    structure_vocab = torch.tensor(list(section_ids_set), device=device)

    # 确保 EOS 始终在结构词表中（避免永不终止）
    if eos_id not in section_ids_set:
        structure_vocab = torch.cat([
            structure_vocab, torch.tensor([eos_id], device=device)
        ])

    seed = torch.tensor([seed_tokens], dtype=torch.long, device=device)

    kv_caches = [[None, None] for _ in range(model.config.n_layers)]
    generated = seed.clone()
    next_token = seed
    ctx_len = seed.size(1)

    for _ in range(max_new_tokens):
        if ctx_len > model.config.max_seq_len:
            next_token = generated[:, -1:]

        logits = model.forward(next_token, kv_caches=kv_caches)
        logits = logits[:, -1, :] / max(temperature, 1e-8)

        # 只允许结构 token
        structure_mask = torch.full_like(logits, float('-inf'))
        structure_mask[:, structure_vocab] = 0.0
        logits = logits + structure_mask

        # top-k
        if top_k > 0:
            vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < vals[:, -1:]] = float('-inf')

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)
        ctx_len = generated.size(1)

        if next_token.item() == eos_id:
            break

    # 移除 seed 部分，只返回结构 token
    return generated[0, len(seed_tokens):].tolist()


@torch.no_grad()
def generate_harmony_skeleton(model: MusicTransformer, tokenizer: REMITokenizer,
                               seed_tokens: list[int],
                               structure_plan_tokens: list[int],
                               harmony_constraint: Optional[dict] = None,
                               max_new_tokens: int = 512,
                               temperature: float = 1.0,
                               top_k: int = 20) -> list[int]:
    """Stage 2: 和声骨架生成 — 在结构规划基础上生成和弦进行。

    使用 Chord→Inv 状态机强制 token 配对。

    Args:
        model: MusicTransformer
        tokenizer: REMI tokenizer
        seed_tokens: 原始种子 token 序列
        structure_plan_tokens: Stage 1 生成的结构 token
        harmony_constraint: 可选和声约束
        max_new_tokens: 最多生成 token 数
        temperature: 采样温度
        top_k: top-k 采样

    Returns:
        完整的 token 序列（seed + structure + harmony skeleton）
    """
    model.eval()
    device = next(model.parameters()).device

    # 前缀 = seed + structure plan
    prefix = seed_tokens + structure_plan_tokens
    prefix_tensor = torch.tensor([prefix], dtype=torch.long, device=device)

    eos_id = tokenizer.eos_token_id
    bar_id = tokenizer.bar_token_id

    # ── 构建和声词表（只允许 Chord/Inv/Chord7/Bar/Key/Section/EOS）──
    harmony_ids_set = set()
    # Chord func tokens
    for func_name in tokenizer.CHORD_FUNCTIONS:
        tid = tokenizer.encode_token(f'<Chord {func_name}>')
        harmony_ids_set.add(tid)
    # Chord7
    harmony_ids_set.add(tokenizer.encode_token('<Chord 7>'))
    # Inv tokens
    for inv_name in tokenizer.CHORD_INVERSIONS:
        tid = tokenizer.encode_token(f'<Inv {inv_name}>')
        harmony_ids_set.add(tid)
    # Bar / Key / Section
    for tid in range(tokenizer.vocab_size):
        ts = tokenizer.decode_token(tid)
        if ts.startswith(tokenizer.KEY) or ts.startswith(tokenizer.SECTION) or \
           ts == tokenizer.BAR or ts == tokenizer.SEC_SUM:
            harmony_ids_set.add(tid)
    # EOS
    harmony_ids_set.add(eos_id)

    harmony_vocab = torch.tensor(list(harmony_ids_set), device=device)

    # 预计算 token ID 集合用于状态机
    inv_ids = [tokenizer.encode_token(f'<Inv {n}>') for n in tokenizer.CHORD_INVERSIONS]
    chord7_id = tokenizer.encode_token('<Chord 7>')
    chord_func_ids = [tokenizer.encode_token(f'<Chord {n}>') for n in tokenizer.CHORD_FUNCTIONS]

    # ── KV cache 生成 ──
    kv_caches = [[None, None] for _ in range(model.config.n_layers)]
    generated = prefix_tensor.clone()
    next_token = prefix_tensor
    ctx_len = generated.size(1)

    state = 'chord'  # 'chord' | 'inv_after_chord' | 'inv_after_chord7'

    for _ in range(max_new_tokens):
        if ctx_len > model.config.max_seq_len:
            next_token = generated[:, -1:]

        logits = model.forward(next_token, kv_caches=kv_caches)
        logits = logits[:, -1, :] / max(temperature, 1e-8)

        # ── 状态机约束 ──
        if state == 'chord':
            # 不允许 Inv/Chord7 在无 Chord func 时单独出现
            for tid in inv_ids:
                logits[0, tid] = float('-inf')
            logits[0, chord7_id] = float('-inf')
        elif state == 'inv_after_chord':
            # 只允许 Inv 或 Chord7
            full_mask = torch.full_like(logits, float('-inf'))
            for tid in inv_ids:
                full_mask[0, tid] = 0.0
            full_mask[0, chord7_id] = 0.0
            logits = logits + full_mask
        elif state == 'inv_after_chord7':
            # 只允许 Inv
            full_mask = torch.full_like(logits, float('-inf'))
            for tid in inv_ids:
                full_mask[0, tid] = 0.0
            logits = logits + full_mask

        # 和声词表约束
        harmony_mask = torch.full_like(logits, float('-inf'))
        harmony_mask[:, harmony_vocab] = 0.0
        logits = logits + harmony_mask

        # top-k
        if top_k > 0:
            vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < vals[:, -1:]] = float('-inf')

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)
        ctx_len = generated.size(1)

        tid = next_token.item()

        # 更新状态机
        if tid in chord_func_ids:
            state = 'inv_after_chord'
        elif tid == chord7_id:
            state = 'inv_after_chord7'
        elif tid in inv_ids:
            state = 'chord'
        # Bar / Key / Section 不改变 state

        if tid == eos_id:
            break

    return generated[0].tolist()


@torch.no_grad()
def section_aware_generate(model: MusicTransformer, tokenizer: REMITokenizer,
                            seed_tokens: list[int],
                            section_plan: Optional[list[dict]] = None,
                            harmony_skeleton: Optional[list[int]] = None,
                            max_bars: int = 64,
                            max_new_tokens: int = 4096,
                            temperature: float = 1.0,
                            top_k: int = 20) -> list[int]:
    """Stage 3: 段落条件生成 — 按段落规划生成完整曲谱。

    支持段落感知注意力 + 和弦偏置（如果提供 harmony_skeleton）。

    Args:
        model: MusicTransformer
        tokenizer: REMI tokenizer
        seed_tokens: 种子 token 序列（前缀含结构规划）
        section_plan: 段落规划
        harmony_skeleton: Stage 2 生成的和声骨架 token 列表
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

    # ── 初始化 KV cache ──
    kv_caches = [[None, None] for _ in range(model.config.n_layers)]

    generated = seed.clone()
    bar_count = seed_tokens.count(bar_id)
    generated_bars = 0

    # 追踪 measure_ids (KV cache 模式下需外部维护)
    cached_measure_ids = torch.cumsum((seed[0] == bar_id).int(), dim=0)
    seed_measure_count = cached_measure_ids[-1].item() if cached_measure_ids.numel() else 0

    # ── 段落追踪 ──
    # 根据 section_plan 预构建段落时间线
    section_token_map = _build_section_token_map(tokenizer)
    # section type name → section type index (0-based, used for section_type_embedding)
    _sec_name_to_type_idx = {name: i for i, name in enumerate(tokenizer.SECTION_NAMES)}
    section_schedule = []  # [(start_bars, type_idx, type_token_id, key), ...]
    if section_plan:
        planned_bars = 0
        for sec in section_plan:
            sec_type_name = sec.get('type', 'theme1')
            n_bars = sec.get('bars', 8)
            sec_key = sec.get('key', 'C')
            sec_type_idx = _sec_name_to_type_idx.get(sec_type_name, 0)
            sec_token_str = section_token_map.get(sec_type_name, 'Section 0')
            sec_type_token_id = tokenizer.encode_token(f'<{sec_token_str}>')
            section_schedule.append((planned_bars, sec_type_idx, sec_type_token_id, sec_key))
            planned_bars += n_bars

    def _get_section_id(bar_idx: int) -> int:
        """返回给定 bar 位置的 section instance ID。"""
        if not section_schedule:
            return NO_SECTION_ID
        for i, (start_bar, _, _, _) in enumerate(section_schedule):
            if i + 1 < len(section_schedule):
                if start_bar <= bar_idx < section_schedule[i + 1][0]:
                    return i + 1
            else:
                if bar_idx >= start_bar:
                    return i + 1
        return NO_SECTION_ID

    def _get_section_type_id(bar_idx: int) -> int:
        """返回给定 bar 位置的 section type index (0-based, for section_type_embedding)。"""
        if not section_schedule:
            return NO_SECTION_TYPE_ID
        for i, (start_bar, type_idx, _, _) in enumerate(section_schedule):
            if i + 1 < len(section_schedule):
                if start_bar <= bar_idx < section_schedule[i + 1][0]:
                    return type_idx
            else:
                if bar_idx >= start_bar:
                    return type_idx
        return NO_SECTION_TYPE_ID

    cur_params = GenerationParams()
    cur_params.temperature = temperature
    cur_params.top_k = top_k

    next_token = seed
    ctx_len = seed.size(1)

    # ── 和弦上下文追踪 ─────────────────────────────────────
    # 从 harmony_skeleton 解析和弦上下文
    chord_func_ids_list = []
    current_chord_func = 0  # 0 = no chord
    current_chord_inv = 0

    chord_func_token_ids = {tokenizer.encode_token(f'<Chord {n}>'): i + 1
                           for i, n in enumerate(tokenizer.CHORD_FUNCTIONS)}
    chord_inv_token_ids = {tokenizer.encode_token(f'<Inv {n}>'): i + 1
                          for i, n in enumerate(tokenizer.CHORD_INVERSIONS)}

    # 解析 harmony_skeleton 中的和弦上下文
    if harmony_skeleton:
        for tid in harmony_skeleton:
            if tid in chord_func_token_ids:
                current_chord_func = chord_func_token_ids[tid]
                current_chord_inv = 0
            elif tid in chord_inv_token_ids:
                current_chord_inv = chord_inv_token_ids[tid]

    # 对 seed_tokens 也追溯初始和弦上下文
    for tid in seed_tokens:
        if tid in chord_func_token_ids:
            current_chord_func = chord_func_token_ids[tid]
            current_chord_inv = 0
        elif tid in chord_inv_token_ids:
            current_chord_inv = chord_inv_token_ids[tid]

    # 构建 seed 段的 section_ids 和 section_types（完整列表）
    section_ids_list = []
    section_types_list = []
    chord_func_ids_list = []
    # 重新遍历 seed_tokens 构建初始列表
    chord_func = 0
    for tid in seed_tokens:
        if tid in chord_func_token_ids:
            chord_func = chord_func_token_ids[tid]
        chord_func_ids_list.append(chord_func)
    current_bar_idx = 0
    section_name_map = {v: k for k, v in section_token_map.items()} if section_plan else {}

    for tid in seed_tokens:
        if section_plan is not None:
            sec_id = _get_section_id(current_bar_idx)
            sec_type_id = _get_section_type_id(current_bar_idx)
        else:
            sec_id = NO_SECTION_ID
            sec_type_id = NO_SECTION_TYPE_ID
        section_ids_list.append(sec_id)
        section_types_list.append(sec_type_id)
        if tid == bar_id:
            current_bar_idx += 1

    for _ in range(max_new_tokens):
        if ctx_len > model.config.max_seq_len:
            next_token = generated[:, -1:]

        # ── 动态段落参数切换（P5） ──
        if section_plan is not None and section_schedule:
            current_section_idx = _get_section_id(current_bar_idx) - 1
            if 0 <= current_section_idx < len(section_schedule):
                sec_type_token_id = section_schedule[current_section_idx][2]
                sec_type_str = tokenizer.decode_token(sec_type_token_id)
                sec_name = sec_type_str.split(' ')[-1].rstrip('>') if ' ' in sec_type_str else ''
                sec_params = SECTION_PARAMS.get(sec_name)
                if sec_params:
                    changed = False
                    if sec_params.get('temperature') is not None:
                        cur_params.temperature = sec_params['temperature']
                        changed = True
                    if sec_params.get('key_bias_strength') is not None:
                        cur_params.key_bias_strength = sec_params['key_bias_strength']
                        changed = True
                    if sec_params.get('complexity') is not None:
                        cur_params.complexity = sec_params['complexity']
                        changed = True
        # ───────────────────────────────────────────────

        # ── 构建 section/chord 输入（传递完整历史，P0） ──
        sec_kwargs = {}
        if model.config.use_section_attention and section_plan is not None:
            sec_ids_tensor = torch.tensor([section_ids_list], dtype=torch.long, device=device)
            sec_types_tensor = torch.tensor([section_types_list], dtype=torch.long, device=device)
            sec_kwargs['section_ids'] = sec_ids_tensor
            sec_kwargs['section_types'] = sec_types_tensor

        # 和弦上下文（chord_bias 在 Stage 3 生效）
        if model.config.use_chord_attention and harmony_skeleton is not None:
            chord_ids_tensor = torch.tensor([chord_func_ids_list], dtype=torch.long, device=device)
            sec_kwargs['chord_func_ids'] = chord_ids_tensor

        logits = model.forward(
            next_token, kv_caches=kv_caches,
            measure_ids=cached_measure_ids, **sec_kwargs,
        )
        logits = logits[:, -1, :] / cur_params.temperature

        # top-k
        if cur_params.top_k > 0:
            vals, _ = torch.topk(logits, min(cur_params.top_k, logits.size(-1)))
            logits[logits < vals[:, -1:]] = float('-inf')

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)
        ctx_len = generated.size(1)
        token_id = next_token.item()

        # 更新 measure_ids
        is_bar = (token_id == bar_id)
        new_measure = seed_measure_count + generated_bars + (1 if is_bar else 0)
        cached_measure_ids = torch.cat(
            [cached_measure_ids, torch.tensor([new_measure], device=device)]
        )

        # 更新和弦追踪
        if token_id in chord_func_token_ids:
            current_chord_func = chord_func_token_ids[token_id]
        chord_func_ids_list.append(current_chord_func)

        # 更新段落追踪
        if section_plan is not None:
            if is_bar:
                current_bar_idx += 1
            sec_id = _get_section_id(current_bar_idx)
            sec_type_id = _get_section_type_id(current_bar_idx)
            section_ids_list.append(sec_id)
            section_types_list.append(sec_type_id)
        else:
            section_ids_list.append(NO_SECTION_ID)
            section_types_list.append(NO_SECTION_TYPE_ID)

        if token_id == bar_id:
            bar_count += 1
            generated_bars += 1
            if bar_count >= max_bars:
                break

        if token_id == eos_id:
            break

    return generated[0].tolist()


# ═══════════════════════════════════════════════════════════════
#  Stage 3 — 逐段迭代生成 (ABC Engine v2, Phase 1)
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def stage3_iterative_generate(
    model: MusicTransformer,
    tokenizer,
    seed_tokens: list[int],
    max_bars: int = 64,
    form: str = 'free',
    max_retries: int = 2,
    base_temperature: float = 1.0,
    top_k: int = 20,
) -> tuple[list[int], dict | None]:
    """Stage 3 逐段迭代生成 + ABC Engine 闭环。"""
    from chopinote_abc.database import (A1DB, A2DB, A3DB, SeedContext)
    from chopinote_abc.planner import plan_structure, plan_harmony

    device = next(model.parameters()).device
    bar_id = tokenizer.bar_token_id

    a3 = A3DB()
    a3.set_baseline(seed_tokens, tokenizer)

    seed_bar_count = sum(1 for t in seed_tokens if t == bar_id)
    a1 = A1DB(sections=plan_structure(
        seed_tokens, tokenizer, max_bars, form, seed_bar_count))
    a1.seed_context = SeedContext(
        final_key=_detect_seed_key(seed_tokens, tokenizer),
        bar_count=seed_bar_count,
    )
    a1.harmony = plan_harmony(a1, seed_tokens, tokenizer)

    a2 = A2DB()
    a2.from_seed(seed_tokens, a3, tokenizer)

    all_tokens, report = _stage3_generate_once(
        model, tokenizer, seed_tokens, a1, a2, a3,
        bar_id, base_temperature, top_k,
    )

    for _ in range(max_retries):
        if not report.get('structural_fixes'):
            break
        for fix in report['structural_fixes']:
            a1.apply_fix(fix)
        a1.reset_overrides()
        all_tokens, report = _stage3_generate_once(
            model, tokenizer, seed_tokens, a1, a2, a3,
            bar_id, base_temperature, top_k,
        )

    return all_tokens, report


@torch.no_grad()
def _stage3_generate_once(
    model, tokenizer, seed_tokens, a1, a2, a3,
    bar_id, base_temperature, top_k,
) -> tuple[list[int], dict]:
    """单次逐段生成。"""
    from chopinote_abc.planner import reharmonize_from_bar
    from chopinote_abc.decision import BHardBans, apply_zone_temperature

    device = next(model.parameters()).device
    eos_id = tokenizer.eos_token_id
    all_tokens = list(seed_tokens)
    cumulative_bar_count = a1.seed_context.bar_count if a1.seed_context else 0

    for section_idx, section in enumerate(a1.sections):
        prefix_tokens = []
        prefix_tokens.extend(a1.build_structure_tokens(tokenizer))
        prefix_tokens.extend(a1.build_harmony_tokens(tokenizer))

        if section_idx > 0:
            for label in a2.records:
                if label.startswith('theme1_') or label.startswith('seed_'):
                    prefix_tokens.extend(a2.get_purified_tokens(label))
            prefix_tokens.extend(
                _prev_section_tail(all_tokens, tokenizer, last_n_bars=1))

        kv_caches = [[None, None] for _ in range(model.config.n_layers)]
        if prefix_tokens:
            prefix_tensor = torch.tensor(
                [prefix_tokens], dtype=torch.long, device=device)
            model.forward(prefix_tensor, kv_caches=kv_caches)

        section_tokens = []
        b1_low_streak = 0

        for bar_idx in range(section.bars):
            gen_params = GenerationParams(
                temperature=apply_zone_temperature(
                    section, bar_idx, base_temperature),
                top_k=top_k,
            )
            hard_bans = BHardBans()

            bar_tokens = []
            done = False

            for _ in range(256):
                if section_tokens:
                    last_tok = section_tokens[-1]
                elif prefix_tokens:
                    last_tok = prefix_tokens[-1]
                else:
                    last_tok = seed_tokens[-1]

                next_input = torch.tensor(
                    [[last_tok]], dtype=torch.long, device=device)

                logits = model.forward(next_input, kv_caches=kv_caches)
                logits = logits[:, -1, :] / gen_params.temperature

                if hard_bans.has_bans():
                    for bid in hard_bans.merge_all():
                        if 0 <= bid < logits.size(-1):
                            logits[0, bid] = float('-inf')

                if gen_params.top_k > 0:
                    vals, _ = torch.topk(
                        logits, min(gen_params.top_k, logits.size(-1)))
                    logits[logits < vals[:, -1:]] = float('-inf')

                probs = torch.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
                tid = next_tok.item()
                bar_tokens.append(tid)

                if tid == bar_id:
                    break
                if tid == eos_id:
                    done = True
                    break

            section_tokens.extend(bar_tokens)
            cumulative_bar_count += 1

            a3.record_bar(cumulative_bar_count, bar_tokens, tokenizer)

            last = a3.get_last_bar()
            if last.b1_score is not None and last.b1_score < 0.2:
                b1_low_streak += 1
            else:
                b1_low_streak = 0

            if b1_low_streak >= 3:
                new_harmony = reharmonize_from_bar(
                    a1, from_bar=cumulative_bar_count)
                a1.override_harmony(cumulative_bar_count, new_harmony)
                section_tokens = []
                b1_low_streak = 0
                break

            if done:
                break

        a3.snapshot_section(section_idx, section_tokens, tokenizer, a1)

        if section.type.startswith('theme') or section.type == 'exposition':
            a2.from_section(section_idx, section_tokens, a3, tokenizer, a1)

        all_tokens.extend(section_tokens)

    report = _c_evaluate(all_tokens, a1, a2, a3, tokenizer)
    return all_tokens, report


def _c_evaluate(all_tokens, a1, a2, a3, tokenizer) -> dict:
    """C 复盘 — Phase 1 规则版。"""
    fixes = []

    recap_idx = a1.find_section('recapitulation')
    theme_idx = a1.find_section('theme1')
    if recap_idx is not None and theme_idx is not None:
        sim = a3.compare_sections(theme_idx, recap_idx)
        if sim.get('pitch_class_dist', 1.0) < 0.7:
            fixes.append({'type': 'tighten_recap', 'section': recap_idx})

    for i, sec in enumerate(a1.sections):
        section_bars = [b for b in a3.bar_log
                        if sec.start_bar <= b.bar < sec.start_bar + sec.bars]
        if len(section_bars) < sec.bars * 0.6:
            fixes.append({
                'type': 'extend_section', 'section': i,
                'target_bars': int(sec.bars * 0.8),
            })

    for i, sec in enumerate(a1.sections):
        end_bar = sec.start_bar + sec.bars - 1
        if end_bar not in a1.cadence_markers:
            fixes.append({
                'type': 'add_cadence', 'bar': end_bar,
                'cadence': sec.cadence,
            })

    archive_cmds = []
    for i, sec in enumerate(a1.sections):
        if sec.type == 'development':
            bars = [b for b in a3.bar_log
                    if sec.start_bar <= b.bar < sec.start_bar + sec.bars]
            if bars and any(b.density > 3 for b in bars):
                archive_cmds.append(
                    {'section_idx': i, 'label': 'development_motif'})

    return {
        'structural_fixes': fixes,
        'archive_commands': archive_cmds,
        'total_bars_generated': len(a3.bar_log),
    }


def _detect_seed_key(tokens, tokenizer) -> str:
    for tid in reversed(tokens):
        ts = tokenizer.decode_token(tid)
        if ts.startswith('<Key ') and ts.endswith('>'):
            return ts[5:-1]
    return 'C'


def _prev_section_tail(all_tokens, tokenizer, last_n_bars=1) -> list[int]:
    bar_id = tokenizer.bar_token_id
    bar_positions = [i for i, t in enumerate(all_tokens) if t == bar_id]
    if len(bar_positions) <= last_n_bars:
        return all_tokens[-64:]
    start = bar_positions[-(last_n_bars + 1)]
    return all_tokens[start:]


