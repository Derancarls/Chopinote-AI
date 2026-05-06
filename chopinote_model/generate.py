"""推理生成模块：自回归采样 → MusicXML 导出。"""
import logging
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from music21 import stream, note, chord, meter, clef

from .model import MusicTransformer
from .config import ModelConfig
from chopinote_dataset.tokenizer import REMITokenizer

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


def _parse_program(token_str: str) -> int:
    """从 '<Program N_M>' 或 '<Program N>' 提取 program number N。"""
    # token_str 形如 '<Program 0>' 或 '<Program 0_1>'
    val = token_str[len('<Program') + 1:-1]  # 去掉 '<Program ' 和 '>'
    return int(val.split('_')[0])


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
    # 预计算每个 MIDI 音高的 NOTE_ON token ID
    note_on_ids = [tokenizer.encode_token(f'<Note_ON {p}>') for p in range(128)]
    # Program token 前缀（不含末尾 >，匹配用 startswith）
    _prog_prefix = '<Program'
    # 从 seed 中查找当前 program
    cur_program: int | None = None
    for tid in reversed(seed_tokens):
        ts = tokenizer.decode_token(tid)
        if ts.startswith(_prog_prefix):
            cur_program = _parse_program(ts)
            break
    # ──────────────────────────────────────────────────────────

    # 初始化 KV cache
    kv_caches = [[None, None] for _ in range(model.config.n_layers)]

    generated = seed.clone()
    bar_count = seed_tokens.count(bar_id)

    # 首轮 forward 处理全部 seed
    next_token = seed
    ctx_len = seed.size(1)

    for _ in range(max_new_tokens):
        if ctx_len > model.config.max_seq_len:
            next_token = generated[:, -1:]

        logits = model.forward(next_token, kv_caches=kv_caches)  # (1, T', V)
        logits = logits[:, -1, :] / temperature  # (1, V)

        # ── 音高限制：遮盖当前乐器音域外的 NOTE_ON token ──
        if cur_program is not None and cur_program < 112:
            rmin, rmax = GM_INSTRUMENT_RANGES.get(cur_program, (0, 127))
            for pitch in range(rmin):
                logits[0, note_on_ids[pitch]] = float('-inf')
            for pitch in range(rmax + 1, 128):
                logits[0, note_on_ids[pitch]] = float('-inf')
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

        # 追踪 Program 变化
        ts = tokenizer.decode_token(token_id)
        if ts.startswith(_prog_prefix):
            cur_program = _parse_program(ts)

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
            pitch = evalue
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
        elif etype in (REMITokenizer.CLEF, REMITokenizer.DYNAMIC, REMITokenizer.HAIRPIN,
                       REMITokenizer.ARTIC, REMITokenizer.ORNAMENT,
                       REMITokenizer.PEDAL, REMITokenizer.SLUR,
                       REMITokenizer.REPEAT, REMITokenizer.JUMP, REMITokenizer.TEMPO,
                       REMITokenizer.TIMESIG, REMITokenizer.KEY, REMITokenizer.BEAT):
            i += 1
            continue
        else:
            i += 1

    return notes_list


def notes_to_score(notes_list: list[dict],
                   grid_size: int = 16,
                   max_bars: int = 256) -> stream.Score:
    """将音符列表重建为 music21 Score。"""
    from music21 import note as note21, duration as dur21

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

    for bar_idx in range(1, max_existing_bar + 1):
        if bar_idx > max_bars:
            break

        meas_by_key: dict[Tuple[int, int], stream.Measure] = {}
        for key in all_keys_sorted:
            m = stream.Measure()
            m.number = bar_idx
            m.timeSignature = meter.TimeSignature('4/4')
            meas_by_key[key] = m

        for key in all_keys_sorted:
            hand_notes = bars.get(bar_idx, {}).get(key, [])
            meas = meas_by_key[key]
            # Pre-compute tuplet-adjusted offsets and duration scales
            hand_sorted = sorted(hand_notes, key=lambda n: (n['position'], n['pitch']))
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

        for key in all_keys_sorted:
            parts[key].append(meas_by_key[key])

    score = stream.Score(list(parts.values()))
    return score


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
    logger.info(f'生成完成: {output_path}')
    return output_path
