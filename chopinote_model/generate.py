"""推理生成模块：自回归采样 → MusicXML 导出。"""
import logging
from typing import Optional

import torch
import torch.nn.functional as F

from music21 import stream, note, chord, meter, key as key21, clef, bar

from .model import MusicTransformer
from .config import ModelConfig
from chopinote_dataset.tokenizer import REMITokenizer

logger = logging.getLogger(__name__)


@torch.no_grad()
def generate(model: MusicTransformer, tokenizer: REMITokenizer,
             seed_tokens: list[int], max_bars: int = 32,
             max_new_tokens: int = 4096, temperature: float = 1.0,
             top_k: int = 20) -> list[int]:
    """自回归生成 token 序列。

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

    generated = seed.clone()
    bar_count = seed_tokens.count(bar_id)

    for _ in range(max_new_tokens):
        # 如果序列太长，取最后 max_seq_len 个 token
        ctx = generated[:, -model.config.max_seq_len:]

        logits = model(ctx)  # (1, T, V)
        next_logits = logits[:, -1, :] / temperature  # (1, V)

        # top-k
        if top_k > 0:
            vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < vals[:, -1:]] = float('-inf')

        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

        generated = torch.cat([generated, next_token], dim=1)
        token_id = next_token.item()

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
    cur_track = 'L'
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
        elif etype in (REMITokenizer.TRACK_L, REMITokenizer.TRACK_R):
            cur_track = 'L' if etype == REMITokenizer.TRACK_L else 'R'
            i += 1
            continue
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

            notes_list.append({
                'bar': cur_bar,
                'position': cur_pos,
                'track': cur_track,
                'pitch': pitch,
                'velocity': vel,
                'duration': dur,
            })
            i += 1
        else:
            i += 1

    return notes_list


def notes_to_score(notes_list: list[dict],
                   grid_size: int = 16,
                   max_bars: int = 256) -> stream.Score:
    """将音符列表重建为 music21 Score。"""
    from music21 import note as note21, duration as dur21

    quarter_per_pos = 4.0 / grid_size

    # 按小节分组
    bars: dict[int, dict[str, list]] = {}
    for n in notes_list:
        b = n['bar']
        if b not in bars:
            bars[b] = {'L': [], 'R': []}
        bars[b][n['track']].append(n)

    # 创建左右手分部
    right_part = stream.Part()
    left_part = stream.Part()
    right_part.partName = 'Piano (Right Hand)'
    left_part.partName = 'Piano (Left Hand)'
    right_part.insert(clef.TrebleClef())
    left_part.insert(clef.BassClef())

    if not bars:
        return stream.Score([right_part, left_part])

    max_existing_bar = max(bars.keys())

    for bar_idx in range(1, max_existing_bar + 1):
        if bar_idx > max_bars:
            break

        m_right = stream.Measure()
        m_left = stream.Measure()
        m_right.number = bar_idx
        m_left.number = bar_idx
        m_right.timeSignature = meter.TimeSignature('4/4')
        m_left.timeSignature = meter.TimeSignature('4/4')

        r_notes = bars.get(bar_idx, {}).get('R', [])
        l_notes = bars.get(bar_idx, {}).get('L', [])

        for hand_notes, meas in [(r_notes, m_right), (l_notes, m_left)]:
            # 按 position 分组
            pos_groups: dict[int, list] = {}
            for n in hand_notes:
                pos_groups.setdefault(n['position'], []).append(n)

            for pos in sorted(pos_groups.keys()):
                notes_at_pos = pos_groups[pos]
                offset = pos * quarter_per_pos

                if len(notes_at_pos) == 1:
                    n = notes_at_pos[0]
                    nt = note21.Note(n['pitch'])
                    nt.duration = dur21.Duration(n['duration'] * quarter_per_pos)
                    nt.volume.velocity = min(127, n['velocity'] * 16)
                    meas.insert(offset, nt)
                else:
                    # 和弦
                    chord_notes = []
                    for n in notes_at_pos:
                        nt = note21.Note(n['pitch'])
                        nt.duration = dur21.Duration(n['duration'] * quarter_per_pos)
                        chord_notes.append(nt)
                    ch = chord.Chord(chord_notes)
                    meas.insert(offset, ch)

        right_part.append(m_right)
        left_part.append(m_left)

    score = stream.Score([right_part, left_part])
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
