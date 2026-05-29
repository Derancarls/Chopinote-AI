"""REMI 语法约束 — 生成时屏蔽非法 token，生成后补全遗漏。

目标：让模型产出尽可能多的内容，而非追求极致语法正确。
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ── Token 类型分类 ──────────────────────────────────────

def _classify_token(token_str: str) -> str:
    """将解码后的 token 字符串映射到类型标签。"""
    if token_str.startswith('<Note_ON'):
        return 'note_on'
    if token_str.startswith('<Velocity'):
        return 'velocity'
    if token_str.startswith('<Duration'):
        return 'duration'
    if token_str == '<Rest>':
        return 'rest'
    if token_str.startswith('<Position'):
        return 'position'
    if token_str.startswith('<Bar'):
        return 'bar'
    if token_str.startswith('<Program'):
        return 'program'
    if token_str.startswith('<Key'):
        return 'key'
    if token_str.startswith('<TimeSig'):
        return 'timesig'
    if token_str.startswith('<Tempo'):
        return 'tempo'
    if token_str.startswith('<Clef'):
        return 'clef'
    if token_str.startswith('<Beat'):
        return 'beat'
    if token_str.startswith('<Bass'):
        return 'bass'
    if token_str.startswith('<Chord '):
        return 'chord'
    if token_str.startswith('<Inv '):
        return 'inv'
    if token_str.startswith('<Section'):
        return 'section'
    if token_str.startswith('<Dynamic'):
        return 'dynamic'
    if token_str.startswith('<Artic'):
        return 'artic'
    if token_str.startswith('<Ornament'):
        return 'ornament'
    if token_str.startswith('<Pedal'):
        return 'pedal'
    if token_str.startswith('<Hairpin'):
        return 'hairpin'
    if token_str.startswith('<Octave'):
        return 'octave'
    if token_str.startswith('<Slur'):
        return 'slur'
    if token_str.startswith('<Tuplet'):
        return 'tuplet'
    if token_str.startswith('<GraceNote'):
        return 'grace'
    if token_str.startswith('<Arpeggio'):
        return 'arpeggio'
    if token_str.startswith('<Tie'):
        return 'tie'
    if token_str.startswith('<Anticipate'):
        return 'anticipate'
    if token_str in ('<PAD>', '<BOS>', '<EOS>', '<MASK>', '<SecSum>'):
        return 'special'
    return 'unknown'


# ── 合法后继 token 类型表 ────────────────────────────────

# 在生成模式下（非结构规划/和声骨架），关键语法：
# Note_ON → Velocity | Note_ON(chord) | Duration
# Velocity → Note_ON(chord) | Duration
# Duration → Position | Bar | Note_ON | Rest | EOS | Program | Key | TimeSig | Tempo
# Rest → Duration
# Bar → Note_ON | Rest | Key | TimeSig | Tempo | Program | Clef | Position
# Position → Note_ON | Rest | Beat
# Program → Clef | Position | Bar | Note_ON | Rest | Beat

_ALWAYS_ALLOWED = {'bar', 'eos', 'pad', 'bos', 'mask', 'unknown',
                   'dynamic', 'artic', 'ornament', 'pedal', 'hairpin',
                   'octave', 'slur', 'tuplet', 'grace', 'arpeggio', 'tie',
                   'beat', 'bass', 'anticipate', 'section', 'inv'}

# 这些 token 几乎任何位置都可以出现（表情记号等）
_GRAMMAR_NEXT: dict[Optional[str], set[str]] = {
    # 刚生成 Note_ON → 必须有 Velocity 或 Duration，也可以继续堆 Note_ON（和弦）
    'note_on': {'note_on', 'velocity', 'duration'},

    # 刚生成 Velocity → 必须有 Duration，或者继续加 Note_ON（和弦）
    'velocity': {'note_on', 'duration', 'artic', 'ornament', 'grace'},

    # 刚生成 Duration → 可以是新位置、新音符、新小节、结束等
    'duration': {'note_on', 'rest', 'position', 'bar', 'program',
                 'key', 'timesig', 'tempo', 'clef', 'dynamic',
                 'pedal', 'hairpin', 'octave', 'slur', 'arpeggio',
                 'anticipate', 'eos'},

    # 刚生成 Rest → 必须有 Duration
    'rest': {'duration'},

    # 刚生成 Bar → 新小节的开始
    'bar': {'note_on', 'rest', 'position', 'program', 'key',
            'timesig', 'tempo', 'clef', 'beat', 'bass', 'anticipate',
            'section', 'chord', 'inv'},

    # 刚生成 Position → 该位置的内容
    'position': {'note_on', 'rest', 'beat'},

    # 刚生成 Program → 通常是 Clef 或直接开始内容
    'program': {'clef', 'position', 'bar', 'note_on', 'rest', 'beat'},

    'key': {'note_on', 'rest', 'position', 'bar', 'program', 'timesig',
            'tempo', 'clef', 'beat', 'anticipate', 'section', 'chord', 'inv'},
    'timesig': {'note_on', 'rest', 'position', 'bar', 'program', 'key',
                'tempo', 'clef', 'beat', 'section'},
    'tempo': {'note_on', 'rest', 'position', 'bar', 'program', 'key',
              'timesig', 'clef', 'beat'},
    'clef': {'note_on', 'rest', 'position', 'bar', 'program', 'key',
             'timesig', 'tempo', 'beat'},

    # Section/Chord/Inv token (结构/和声规划用)
    'section': {'section', 'chord', 'inv', 'note_on', 'rest', 'bar',
                'program', 'key', 'timesig', 'tempo', 'position'},
    'chord': {'inv', 'chord', 'section', 'bar'},
    'inv': {'chord', 'section', 'bar', 'note_on', 'rest'},

    # 起始：生成从 seed 后面开始，seed 最后可能是 Bar 或 Duration
    None: {'note_on', 'rest', 'position', 'bar', 'key', 'timesig',
           'tempo', 'program', 'beat', 'section', 'chord'},
}


def _legal_next_types(prev_type: Optional[str]) -> set[str]:
    """获取当前 token 类型之后合法的下一 token 类型集合。"""
    allowed = _GRAMMAR_NEXT.get(prev_type, set())
    return allowed | _ALWAYS_ALLOWED


def apply_grammar_mask(logits, token_ids, tokenizer,
                       prev_token_str: Optional[str] = None,
                       pending_note_on: bool = False,
                       pending_rest: bool = False):
    """在 softmax 前屏蔽非法 token，返回被屏蔽的 token 数。"""
    vocab_size = logits.size(-1)
    masked = 0

    # 确定上一步 token 类型
    if prev_token_str is not None:
        prev_type = _classify_token(prev_token_str)
    else:
        prev_type = None

    # pending state 覆盖：如果还有未处理的 Note_ON，必须给 Duration
    if pending_note_on:
        # 只允许: Velocity(more chord), Note_ON(more chord), Duration
        allowed_types = {'velocity', 'note_on', 'duration', 'artic', 'ornament', 'grace'}
    elif pending_rest:
        allowed_types = {'duration'}
    else:
        allowed_types = _legal_next_types(prev_type)

    # 允许 special token 和 always_allowed
    allowed_types |= _ALWAYS_ALLOWED

    for tid in range(vocab_size):
        token_str = tokenizer.decode_token(tid)
        token_type = _classify_token(token_str)
        if token_type not in allowed_types:
            logits[0, tid] = float('-inf')
            masked += 1

    if masked > 0:
        logger.debug(f'Grammar: masked {masked}/{vocab_size} tokens, '
                     f'prev={prev_type}, allowed={sorted(allowed_types)}')

    return masked


# ── 生成后补全 ──────────────────────────────────────────


def patch_token_sequence(token_ids: list[int], tokenizer) -> list[int]:
    """生成后修复：补全遗漏的 Duration/Velocity，移除孤立 token。

    返回修补后的 token 序列。
    """
    patched = []
    pending_notes = 0     # 等 Duration 的 Note_ON 数
    pending_vel = False   # Note_ON 存在但还没有 Velocity
    last_vel = 4          # 默认力度（mf）
    last_pos = 0
    bar_id = tokenizer.bar_token_id
    default_dur = 4       # 默认时值 = 1 拍 (grid=16, dur=4)

    for tid in token_ids:
        ts = tokenizer.decode_token(tid)
        ttype = _classify_token(ts)

        if ttype == 'note_on':
            if pending_notes > 0 and not pending_vel:
                # 前一个 Note_ON 还没有 Velocity → 补一个
                vel_token = _make_velocity_token(tokenizer, last_vel)
                if vel_token >= 0:
                    patched.append(vel_token)
            pending_notes += 1
            pending_vel = True
            patched.append(tid)

        elif ttype == 'velocity':
            pending_vel = False
            try:
                last_vel = int(ts.split(' ')[-1].rstrip('>'))
            except (ValueError, IndexError):
                pass
            patched.append(tid)

        elif ttype == 'duration':
            if pending_notes > 0 and pending_vel:
                # Note_ON 没有 Velocity → 补一个
                vel_token = _make_velocity_token(tokenizer, last_vel)
                if vel_token >= 0:
                    patched.insert(-pending_notes if pending_notes < len(patched) else len(patched), vel_token)
            # 处理所有 pending notes
            dur_val = _parse_duration_value(ts)
            for _ in range(pending_notes):
                patched.append(tid)
            pending_notes = 0
            pending_vel = False
            patched.append(tid)

        elif ttype == 'rest':
            # 先清空 pending notes（罕见但可能）
            if pending_notes > 0:
                _flush_pending_notes(patched, pending_notes, last_vel, tokenizer)
                pending_notes = 0
                pending_vel = False
            patched.append(tid)

        elif ttype == 'bar':
            if pending_notes > 0:
                _flush_pending_notes(patched, pending_notes, last_vel, tokenizer)
                pending_notes = 0
                pending_vel = False
            patched.append(tid)

        elif ttype == 'position':
            if pending_notes > 0:
                _flush_pending_notes(patched, pending_notes, last_vel, tokenizer)
                pending_notes = 0
                pending_vel = False
            patched.append(tid)

        else:
            patched.append(tid)

    # 结尾仍有 pending notes
    if pending_notes > 0:
        _flush_pending_notes(patched, pending_notes, last_vel, tokenizer)

    if len(patched) != len(token_ids):
        logger.info(f'Grammar patch: {len(token_ids)} → {len(patched)} tokens '
                    f'(+{len(patched) - len(token_ids)})')

    return patched


def _make_velocity_token(tokenizer, vel: int) -> int:
    try:
        return tokenizer.encode_token(f'<Velocity {max(1, min(8, vel))}>')
    except Exception:
        return -1


def _parse_duration_value(ts: str) -> int:
    try:
        return int(ts.split(' ')[-1].rstrip('>'))
    except (ValueError, IndexError):
        return 4


def _flush_pending_notes(buf: list[int], count: int, vel: int, tokenizer):
    """为 pending 的 Note_ON 补上 Velocity + Duration。"""
    vel_token = _make_velocity_token(tokenizer, vel)
    dur_token = tokenizer.encode_token('<Duration 4>')  # 1 拍默认

    for _ in range(count):
        if vel_token >= 0:
            buf.append(vel_token)
        buf.append(dur_token)
