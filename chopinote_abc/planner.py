"""Stage 1/2 规则规划器 — Phase 1 纯规则驱动，零模型依赖。

provide:
  plan_structure()     → list[SectionPlan]    # Stage 1
  plan_harmony()       → list[ChordAtBar]      # Stage 2
  reharmonize_from_bar() → list[ChordAtBar]    # 和声回退
  tonal_progression_template() → list[ChordAtBar]  # 模板
"""

from __future__ import annotations

from .database import SectionPlan, ChordAtBar, A1DB, DramaticCurve


# ═══════════════════════════════════════════════════════════════
#  曲式模板 — 段落分配比例
# ═══════════════════════════════════════════════════════════════

FORM_TEMPLATES: dict[str, list[tuple[str, float, list[str], float | None]]] = {
    # (section_type, bar_ratio, [sub_types], innovation_budget)
    'sonata': [
        ('exposition',      0.35, ['theme1', 'theme2'], 0.10),
        ('development',     0.35, [], 0.35),
        ('recapitulation',  0.30, ['theme1'], 0.08),
    ],
    'binary': [
        ('section_a', 0.50, [], 0.10),
        ('section_b', 0.50, [], 0.15),
    ],
    'theme_variations': [
        ('theme1',    0.20, [], 0.05),
        ('variation', 0.20, [], 0.25),
        ('variation', 0.20, [], 0.25),
        ('variation', 0.20, [], 0.25),
        ('coda',      0.20, [], 0.05),
    ],
    'free': [
        ('theme1',      0.25, [], 0.08),
        ('development', 0.50, [], 0.30),
        ('coda',        0.25, [], 0.05),
    ],
}


# ═══════════════════════════════════════════════════════════════
#  v0.3.2: VoicePlan — 声部配置
# ═══════════════════════════════════════════════════════════════

def detect_active_voices(seed_tokens: list[int], tokenizer) -> list[int]:
    """扫描 seed，返回活跃声部列表。

    统计 seed 中各 Voice token 出现次数，>=3 次视为活跃。
    检测不到时默认返回 [0,1,2,3] (SATB 全开)。

    Returns:
        活跃声部索引列表，如 [0, 3] (两声部) 或 [0, 1, 2, 3] (四声部)
    """
    voice_counts: dict[int, int] = {}
    _voice_prefix = tokenizer.VOICE
    for tid in seed_tokens:
        ts = tokenizer.decode_token(tid)
        if ts.startswith(_voice_prefix):
            try:
                v = int(ts.split(' ')[1].rstrip('>'))
            except (IndexError, ValueError):
                continue
            voice_counts[v] = voice_counts.get(v, 0) + 1

    active = sorted([v for v, c in voice_counts.items() if c >= 3])
    if not active:
        return [0, 1, 2, 3]  # 检测不到默认 SATB
    return active


def voice_count_to_plan(count: int) -> list[int]:
    """用户指定的声部数 → 活跃声部列表。

    2 → [0, 3]  (主高音 + 主低音 = 旋律+贝斯骨架)
    3 → [0, 1, 3] (主高+次高+主低)
    4 → [0, 1, 2, 3] (SATB 全开)
    """
    if count >= 4:
        return [0, 1, 2, 3]
    if count == 3:
        return [0, 1, 3]
    if count == 2:
        return [0, 3]
    return [0, 1, 2, 3]


def plan_structure(seed_tokens: list[int], tokenizer,
                   target_bars: int = 64,
                   form: str = 'free',
                   seed_bar_count: int | None = None) -> list[SectionPlan]:
    """Stage 1 规则规划器 — 分配段落。

    Args:
        seed_tokens: 用户输入的 seed
        tokenizer: REMI tokenizer
        target_bars: 目标总小节数（含 seed）
        form: 曲式名，对应 FORM_TEMPLATES 的 key
        seed_bar_count: seed 已有小节数，None=自动统计

    Returns:
        SectionPlan 列表
    """
    # 1. 统计 seed 小节数
    if seed_bar_count is None:
        bar_id = tokenizer.bar_token_id
        seed_bar_count = sum(1 for t in seed_tokens if t == bar_id)

    # 2. 检测 seed 的调性（取最后一个 Tonic token，v0.3.0 废弃 Key）
    seed_key = 'C'
    for tid in reversed(seed_tokens):
        ts = tokenizer.decode_token(tid)
        if ts.startswith(tokenizer.TONIC) and ts.endswith('>'):
            seed_key = ts[len(tokenizer.TONIC) + 1:-1]
            break

    # 3. 计算需要生成的小节数
    remaining_bars = max(1, target_bars - seed_bar_count)

    # 4. 取模板
    template = FORM_TEMPLATES.get(form, FORM_TEMPLATES['free'])

    # 5. 按比例分配
    sections = []
    total_ratio = sum(r for _, r, _, _ in template)
    bar_allocated = 0

    for i, (sec_type, ratio, sub_types, innov_budget) in enumerate(template):
        if i == len(template) - 1:
            # 最后一段拿剩余 bar
            n_bars = remaining_bars - bar_allocated
        else:
            n_bars = max(1, int(remaining_bars * ratio / total_ratio))
        bar_allocated += n_bars

        # 调性：第一段跟随 seed，之后按规则转调
        if i == 0:
            key = seed_key
        elif sec_type in ('development', 'variation', 'bridge', 'episode'):
            key = _relative_key(seed_key)  # 转关系调
        else:
            key = seed_key  # 再现/尾声回主调

        # 终止式
        if sec_type in ('coda',) or (sec_type == 'recapitulation' and i == len(template) - 1):
            cadence = 'PAC'
        elif sec_type in ('development', 'bridge', 'transition'):
            cadence = 'HC'
        else:
            cadence = 'IAC'

        # 发展配方
        dev_ops = None
        if sec_type == 'development':
            dev_ops = ['invert']  # Phase 1 默认倒影
        elif sec_type == 'variation':
            dev_ops = ['fragment']

        sections.append(SectionPlan(
            type=sec_type,
            bars=max(1, n_bars),
            key=key,
            cadence=cadence,
            innovation_budget=innov_budget or 0.1,
            development_ops=dev_ops,
        ))

    return sections


def _relative_key(key_str: str) -> str:
    """返回关系调（大调→属调，小调→关系大调）。"""
    major_keys = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#',
                  'F', 'Bb', 'Eb', 'Ab', 'Db', 'Gb', 'Cb']
    minor_keys = ['Am', 'Em', 'Bm', 'F#m', 'C#m', 'G#m', 'D#m', 'A#m',
                  'Dm', 'Gm', 'Cm', 'Fm', 'Bbm', 'Ebm', 'Abm']

    if key_str in major_keys:
        idx = major_keys.index(key_str)
        return major_keys[(idx + 1) % len(major_keys)]  # 属方向
    elif key_str in minor_keys:
        # 小调 → 关系大调
        idx = minor_keys.index(key_str)
        return major_keys[idx % len(major_keys)] if idx < len(major_keys) else 'C'
    return 'C'


# ═══════════════════════════════════════════════════════════════
#  和声规划
# ═══════════════════════════════════════════════════════════════

# 功能和声模板：段落类型 → 默认进行序列
_HARMONY_TEMPLATES: dict[str, list[str]] = {
    'theme1':       ['I', 'IV', 'V', 'I', 'vi', 'ii', 'V', 'I'],
    'theme2':       ['I', 'ii', 'V', 'I', 'IV', 'V', 'I'],
    'exposition':   ['I', 'IV', 'V', 'I', 'vi', 'IV', 'V', 'I'],
    'development':  ['vi', 'ii', 'V', 'iii', 'IV', 'ii', 'V', 'I'],
    'recapitulation': ['I', 'IV', 'V', 'I', 'IV', 'V', 'I'],
    'coda':         ['I', 'IV', 'I', 'V', 'I'],
    'bridge':       ['V', 'V/V', 'V', 'I'],
    'transition':   ['V', 'ii', 'V', 'I'],
    'variation':    ['I', 'IV', 'V', 'I', 'vi', 'V', 'I'],
    'cadenza':      ['V', 'I', 'IV', 'V', 'I'],
    'intro':        ['I', 'IV', 'I'],
    'episode':      ['vi', 'V/vi', 'vi', 'IV', 'V', 'I'],
    'section_a':    ['I', 'IV', 'V', 'I'],
    'section_b':    ['V', 'I', 'IV', 'V', 'I'],
}

# 终止式 → 最后两和弦
_CADENCE_CHORDS: dict[str, list[str]] = {
    'PAC':  ['V', 'I'],
    'IAC':  ['V', 'I'],       # 同样的和弦，但 I 可能不是根音位置
    'HC':   ['ii', 'V'],
    'DC':   ['V', 'vi'],
}


def tonal_progression_template(
    section_type: str, n_bars: int, cadence: str = 'IAC',
) -> list[ChordAtBar]:
    """给定段落类型 + 小节数 + 终止式 → 和声进行序列。

    原理：取模板进行，循环填充到 n_bars，最后 2 bar 替换为终止式和弦。
    """
    base = _HARMONY_TEMPLATES.get(section_type,
                                   _HARMONY_TEMPLATES['theme1'])
    cadence_pair = _CADENCE_CHORDS.get(cadence, ['V', 'I'])

    chords: list[ChordAtBar] = []
    base_idx = 0

    for bar in range(n_bars):
        if bar >= n_bars - 2 and len(cadence_pair) > 0:
            # 终止式
            func = cadence_pair[bar - (n_bars - len(cadence_pair))]
            # 终止式用 root 位
            chords.append(ChordAtBar(bar=bar, func=func, inv='root'))
        else:
            func = base[base_idx % len(base)]
            # 非终止式偶尔用转位
            inv = '1st' if (bar % 4 == 2) else 'root'
            chords.append(ChordAtBar(bar=bar, func=func, inv=inv))
            base_idx += 1

    return chords


def plan_harmony(A1: A1DB, seed_tokens: list[int],
                 tokenizer) -> list[ChordAtBar]:
    """Stage 2 规则和声规划器。

    1. 从 seed 末尾提取当前调性
    2. 按段落类型套用模板进行 + 终止式约束
    3. 段间转调自动插 pivot chord
    """
    # 从 seed 推测当前和声上下文
    seed_key = A1.seed_context.final_key if A1.seed_context else 'C'
    current_key = seed_key

    harmony: list[ChordAtBar] = []
    global_bar = 0

    for section in A1.sections:
        # 转调：在段首插入 pivot chord
        if section.key != current_key:
            pivot = _pivot_chord(current_key, section.key)
            if pivot:
                harmony.append(ChordAtBar(bar=global_bar, func=pivot,
                                          inv='root'))
            current_key = section.key

        # 模板进行
        sec_chords = tonal_progression_template(
            section.type, section.bars, section.cadence)

        for c in sec_chords:
            c.bar = global_bar
            global_bar += 1
            harmony.append(c)

    return harmony


def _pivot_chord(from_key: str, to_key: str) -> str | None:
    """找两调之间的 pivot chord（两调的 I/IV/V 中的交集）。"""
    # 简化版：大调属性方向 pivot = V7 of new key
    # 实际上真正的 pivot chord 需要更复杂的分析，这里用属准备
    if from_key != to_key:
        return 'V'
    return None


def reharmonize_from_bar(A1: A1DB, from_bar: int, seed_bar_offset: int = 0) -> list[ChordAtBar]:
    """B 触发和声回退时调用。仅重规划 from_bar 及之后的段落。

    Phase 1 实现（纯规则）:
    1. 找到 from_bar 所属段落和后续段落
    2. 从当前调性出发，套用和声模板重新生成
    3. 保留 from_bar 前的和声不变
    4. 后续段如果调性不同，自动插 pivot chord
    """
    # 转为 0-based 段落内偏移（A1 的 start_bar 从 0 开始）
    local_bar = from_bar - seed_bar_offset

    section = A1.get_section(local_bar)
    if not section:
        return []

    section_idx = -1
    for i, sec in enumerate(A1.sections):
        if sec is section:
            section_idx = i
            break
    if section_idx < 0:
        return []

    remaining_bars = section.bars - (local_bar - section.start_bar)
    new_chords = tonal_progression_template(
        section.type, remaining_bars, section.cadence)

    # 调整 bar offset
    for c in new_chords:
        c.bar = local_bar + c.bar

    # 后续段重新规划
    bar_offset = local_bar + remaining_bars
    current_key = section.key
    for sec in A1.sections[section_idx + 1:]:
        if sec.key != current_key:
            pivot = _pivot_chord(current_key, sec.key)
            if pivot:
                new_chords.append(ChordAtBar(bar=bar_offset, func=pivot,
                                             inv='root'))
                bar_offset += 1
            current_key = sec.key
        sec_chords = tonal_progression_template(
            sec.type, sec.bars, sec.cadence)
        for c in sec_chords:
            c.bar = bar_offset
            bar_offset += 1
            new_chords.append(c)

    return new_chords


# ═══════════════════════════════════════════════════════════════
# SSF 和弦 → 12 维向量转换 (推理用)
# ═══════════════════════════════════════════════════════════════

# 功能和声 → 距主音半音数 (scale degree → semitones from tonic, 大调参照)
_DEG_ROOT_SEMITONES: dict[str, int] = {
    'I': 0,  'i': 0,
    'II': 2, 'ii': 2, 'ii°': 2,
    'III': 4, 'iii': 4,
    'IV': 5, 'iv': 5,
    'V': 7,  'v': 7, 'V7': 7, 'V/V': 2, 'V/vi': 11, 'V/ii': 9,
    'VI': 9, 'vi': 9,
    'VII': 11, 'vii': 11, 'vii°': 11, 'vii°7': 11,
    'N': 1, 'It6': 6, 'Fr6': 6, 'Ger6': 6,
}

# 和弦性质 → 根音上方的半音间隔
_QUALITY_INTERVALS: dict[str, list[int]] = {
    'major':      [0, 4, 7],
    'minor':      [0, 3, 7],
    'dim':        [0, 3, 6],
    'dim7':       [0, 3, 6, 9],
    'hdim7':      [0, 3, 6, 10],
    'aug':        [0, 4, 8],
    'dom7':       [0, 4, 7, 10],
    'min7':       [0, 3, 7, 10],
    'maj7':       [0, 4, 7, 11],
}

# 功能 → 性质
def _func_quality(func: str) -> str:
    if '°7' in func or func == 'vii°7':
        return 'dim7'
    if '°' in func or func.startswith('vii'):
        return 'dim'
    if '7' in func:
        if func[0].islower() or func.startswith('ii') or func.startswith('vi'):
            return 'min7'
        return 'dom7'
    if func[0].islower() or func == 'N':
        return 'minor' if func != 'N' else 'major'
    if func in ('It6', 'Fr6', 'Ger6'):
        return 'aug'
    return 'major'


def chord_func_to_ssf(func: str, tonic_name: str = 'C') -> list[float]:
    """和弦功能名 → 12 维 SSF 向量 (主音锚定, tonic 在 pos 0)。

    Examples:
        chord_func_to_ssf('I', 'C')  → [1,0,1,0,1,0,0,1,0,0,0,0]  (C E G)
        chord_func_to_ssf('V', 'C')  → [0,0,1,0,0,0,0,1,0,0,0,1]  (G B D)
        chord_func_to_ssf('vi', 'C') → [1,0,0,0,1,0,0,0,1,0,0,0]  (A C E)
        chord_func_to_ssf('V7', 'C') → [0,0,1,0,1,0,0,1,0,0,1,0]  (G B D F)
    """
    from chopinote_dataset.tokenizer import _TONIC_PC_MAP

    root_offset = _DEG_ROOT_SEMITONES.get(func)
    if root_offset is None:
        return [0.5] * 12

    quality = _func_quality(func)
    intervals = _QUALITY_INTERVALS.get(quality, [0, 4, 7])

    ssf = [0.0] * 12
    for interval in intervals:
        pos = (root_offset + interval) % 12
        ssf[pos] = 1.0
    return ssf


def harmony_to_ssf(
    chords: list, tonic_name: str = 'C'
) -> list[list[float]]:
    """ChordAtBar 列表 → per-bar SSF 向量序列。

    Args:
        chords: ChordAtBar 列表 (.func 字段)
        tonic_name: 主音名

    Returns:
        list[list[float]]: 每个和弦一个 12 维向量
    """
    return [chord_func_to_ssf(c.func, tonic_name) for c in chords]


# ═══════════════════════════════════════════════════════════════
#  v0.3.2: 乐句规划
# ═══════════════════════════════════════════════════════════════

def plan_phrases_for_section(
    section_type: str, n_bars: int, cadence: str = 'PAC',
    phrase_length: int = 8,
) -> list:
    """为段规划乐句结构。

    根据 section.type 自动选择模板:
      - theme1/theme2/exposition/recapitulation → period (平行乐段)
      - development → sentence 或自由模进
      - transition/bridge → 不规划乐句 (自由过渡)
      - coda/closing → closing phrases

    Returns:
        list[PhrasePlan]
    """
    from .database import PhrasePlan

    if n_bars < 4:
        return []

    if section_type in ('development', 'variation'):
        return _expand_sentence_phrases(n_bars, cadence, phrase_length)
    elif section_type in ('transition', 'bridge', 'intro', 'cadenza', 'episode'):
        return []  # 自由过渡，不规划乐句
    else:
        # theme1/theme2/exposition/recapitulation/coda/section_a/section_b
        return _expand_period_phrases(n_bars, cadence, phrase_length)


def _expand_period_phrases(n_bars: int, final_cadence: str = 'PAC',
                           phrase_length: int = 8) -> list:
    """平行乐段: antecedent(HC) + consequent(PAC/IAC)。

    32 bars, phrase_length=8:
      phrase 0: antecedent, bars 0-7, HC
      phrase 1: consequent, bars 8-15, PAC
      phrase 2: antecedent, bars 16-23, HC
      phrase 3: consequent, bars 24-31, PAC
    """
    from .database import PhrasePlan

    phrases = []
    n_pairs = n_bars // (phrase_length * 2)
    remainder = n_bars % (phrase_length * 2)

    for pair_idx in range(n_pairs):
        base = pair_idx * phrase_length * 2
        # Antecedent → HC
        phrases.append(PhrasePlan(
            phrase_idx=len(phrases),
            phrase_type='antecedent',
            bar_start=base,
            bar_end=base + phrase_length,
            cadence_target='HC',
            contour_shape='ascending' if pair_idx % 2 == 0 else 'arch',
            relation_to_prev='parallel' if pair_idx > 0 else None,
        ))
        # Consequent → PAC (or final_cadence for last pair)
        cad = final_cadence if pair_idx == n_pairs - 1 else 'IAC'
        phrases.append(PhrasePlan(
            phrase_idx=len(phrases),
            phrase_type='consequent',
            bar_start=base + phrase_length,
            bar_end=base + phrase_length * 2,
            cadence_target=cad,
            contour_shape='arch',
            relation_to_prev='answering',
        ))

    # 剩余小节 → extension/closing
    if remainder >= 4:
        phrases.append(PhrasePlan(
            phrase_idx=len(phrases),
            phrase_type='closing',
            bar_start=n_bars - remainder,
            bar_end=n_bars,
            cadence_target=final_cadence,
            contour_shape='descending',
        ))

    return phrases


def _expand_sentence_phrases(n_bars: int, cadence: str = 'HC',
                             phrase_length: int = 8) -> list:
    """乐句组: presentation(4) + continuation(4) 交替。

    development, 24 bars, phrase_length=8:
      phrase 0: presentation, bars 0-3
      phrase 1: continuation, bars 4-7
      phrase 2: presentation, bars 8-11 (模进上移)
      phrase 3: continuation, bars 12-15 (碎片化)
      phrase 4: transition, bars 16-19
      phrase 5: closing, bars 20-23
    """
    from .database import PhrasePlan

    phrases = []
    half = max(2, phrase_length // 2)
    n_units = n_bars // phrase_length
    remainder = n_bars % phrase_length

    for unit_idx in range(n_units):
        base = unit_idx * phrase_length
        is_last = (unit_idx == n_units - 1 and remainder < half)

        if is_last:
            # 最后一组 → closing
            phrases.append(PhrasePlan(
                phrase_idx=len(phrases),
                phrase_type='closing',
                bar_start=base,
                bar_end=base + phrase_length,
                cadence_target=cadence,
                contour_shape='descending',
            ))
        else:
            # Presentation
            phrases.append(PhrasePlan(
                phrase_idx=len(phrases),
                phrase_type='antecedent',
                bar_start=base,
                bar_end=base + half,
                cadence_target='HC' if unit_idx > 0 else 'IAC',
                motif_variant='fragmented' if unit_idx > 0 else 'original',
            ))
            # Continuation
            phrases.append(PhrasePlan(
                phrase_idx=len(phrases),
                phrase_type='consequent',
                bar_start=base + half,
                bar_end=base + phrase_length,
                cadence_target='HC' if not is_last else cadence,
                motif_variant='fragmented',
            ))

    # 剩余小节
    if remainder >= 4:
        phrases.append(PhrasePlan(
            phrase_idx=len(phrases),
            phrase_type='closing',
            bar_start=n_bars - remainder,
            bar_end=n_bars,
            cadence_target=cadence,
        ))

    return phrases


def cadence_ssf_boost(
    cadence_type: str,
    ssf_field: list[float],
    strength: float = 0.2,
) -> list[float]:
    """终止区 SSF LocalField 增强 — 强化终止式和弦 chroma。

    训练时不调用——让模型从数据中学。推理时可选增强。

    Args:
        cadence_type: 终止式类型 ('PAC', 'IAC', 'HC', 'DC', 'PC')
        ssf_field: 12 维 SSF 向量 (TonicField 或 LocalField delta)
        strength: boost 强度乘数 (默认 0.2)

    Returns:
        boosted 12 维向量 (不修改输入)

    Boost 规则:
        PAC/IAC: pos 7 (属音) +strength, pos 11 (导音) +0.75*strength
        PC:      pos 5 (下属音) +strength
        HC:      pos 7 (属音) +strength
        DC:      不 boost (意外终止，保留原场)
    """
    import copy
    result = copy.copy(ssf_field) if isinstance(ssf_field, list) else ssf_field.copy()
    if cadence_type in ('PAC', 'IAC'):
        result[7] = min(1.0, result[7] + strength)       # 属音
        result[11] = min(1.0, result[11] + 0.75 * strength)  # 导音
    elif cadence_type == 'HC':
        result[7] = min(1.0, result[7] + strength)       # 属音
    elif cadence_type == 'PC':
        result[5] = min(1.0, result[5] + strength)       # 下属音
    # DC: no boost — 意外终止保留原场
    return result


# ═══════════════════════════════════════════════════════════════
#  长程张力曲线 (v0.3.3-opt3)
# ═══════════════════════════════════════════════════════════════

DRAMATIC_TEMPLATES: dict[str, dict] = {
    'sonata': {
        'description': '呈示部低→发展部上升→高潮→再现部回落→尾声溶解',
        'section_tensions': {
            'intro':          (0.15, 0.30),
            'exposition':     (0.25, 0.50),
            'development':    (0.45, 0.95),
            'recapitulation': (0.55, 0.40),
            'coda':           (0.35, 0.10),
        },
        'peak_section': 'development',
        'peak_position': 0.75,
        'smoothness': 0.3,
    },
    'theme_variations': {
        'description': '主题陈述→逐变奏递增紧张→最后变奏高潮→回归平息',
        'section_tensions': {
            'theme':      (0.30, 0.35),
            'variation1': (0.30, 0.40),
            'variation2': (0.35, 0.50),
            'variation3': (0.45, 0.65),
            'variation4': (0.55, 0.75),
            'finale':     (0.60, 0.20),
        },
        'peak_section': 'finale',
        'peak_position': 0.15,
        'smoothness': 0.5,
    },
    'binary': {
        'description': 'A→B 两部对比, B 部略高',
        'section_tensions': {
            'A': (0.25, 0.45),
            'B': (0.40, 0.30),
        },
        'peak_section': 'B',
        'peak_position': 0.4,
        'smoothness': 0.5,
    },
    'rondo': {
        'description': 'ABACA 回旋, A 每次回归略降, 插部递增',
        'section_tensions': {
            'A1': (0.30, 0.35),
            'B1': (0.30, 0.55),
            'A2': (0.40, 0.40),
            'C1': (0.35, 0.70),
            'A3': (0.50, 0.20),
        },
        'peak_section': 'C1',
        'peak_position': 0.6,
        'smoothness': 0.4,
    },
    'free': {
        'description': '自由曲式, 线性递增→高潮→回落',
        'section_tensions': None,
        'peak_section': None,
        'peak_position': 0.7,
        'smoothness': 0.5,
    },
}


def _hermite(t: float, p0: float, p1: float, smoothness: float = 0.5) -> float:
    """Cubic Hermite 插值: t ∈ [0,1], p0→p1 平滑过渡。"""
    t2 = t * t
    t3 = t2 * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    m0 = (p1 - p0) * smoothness
    m1 = (p1 - p0) * smoothness
    return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1


def _crossfade_boundaries(curve: list[float], section_plans: list,
                          fade_bars: int = 2) -> list[float]:
    """段边界 crossfade: 相邻两段最后/最前 fade_bars 做加权叠加。"""
    result = list(curve)
    bar = 0
    for i in range(len(section_plans) - 1):
        bar += section_plans[i].bars
        for j in range(fade_bars):
            if bar - j - 1 >= 0 and bar + j < len(result):
                w0 = 1.0 - (j + 1) / (fade_bars + 1)
                w1 = (j + 1) / (fade_bars + 1)
                result[bar - j - 1] = w0 * curve[bar - j - 1] + w1 * curve[bar + j]
                result[bar + j] = w1 * curve[bar - j - 1] + w0 * curve[bar + j]
    return result


def _get_section_tensions(template: dict, sec_type: str, idx: int,
                          n_sections: int) -> tuple[float, float]:
    """获取某段的 (t_start, t_end)。"""
    st = template.get('section_tensions')
    if st is None:
        # free: 自动分配
        t = idx / max(1, n_sections - 1)
        if t < template.get('peak_position', 0.7):
            return (0.2 + t * 0.7, 0.25 + t * 0.7)
        else:
            return (0.5 + (1 - t) * 0.4, 0.3 + (1 - t) * 0.3)
    if sec_type in st:
        return st[sec_type]
    # fallback: 通用段类型匹配
    for key in ('exposition', 'theme1', 'A', 'A1'):
        if key in st:
            return st[key]
    return (0.3, 0.4)


def build_dramatic_curve(section_plans: list, template: dict) -> 'DramaticCurve':
    """从模板和段落规划生成 bar 级精度的平滑张力曲线。

    Args:
        section_plans: list[SectionPlan]
        template: DRAMATIC_TEMPLATES 条目

    Returns:
        DramaticCurve
    """
    from .database import DramaticCurve

    total_bars = sum(s.bars for s in section_plans)
    if total_bars <= 0:
        return DramaticCurve(curve=[0.5], global_peak_bar=0)

    smoothness = template.get('smoothness', 0.5)
    curve = [0.5] * total_bars
    bar = 0

    for i, sec in enumerate(section_plans):
        n = sec.bars
        t_start, t_end = _get_section_tensions(
            template, sec.type, i, len(section_plans))

        for j in range(n):
            t = j / max(1, n - 1)
            curve[bar + j] = _hermite(t, t_start, t_end, smoothness)
        bar += n

    # Segment boundary crossfade
    curve = _crossfade_boundaries(curve, section_plans, fade_bars=2)

    # Clamp to [0, 1]
    curve = [max(0.0, min(1.0, c)) for c in curve]

    # 确定全局高潮位置
    peak_section = template.get('peak_section')
    peak_pos = template.get('peak_position', 0.75)
    global_peak_bar = int(total_bars * 0.5)

    if peak_section:
        bar = 0
        for i, sec in enumerate(section_plans):
            if sec.type == peak_section:
                global_peak_bar = bar + int(sec.bars * peak_pos)
                break
            bar += sec.bars
    else:
        global_peak_bar = int(total_bars * peak_pos)

    # 确定每段局部峰值
    section_peaks: dict[int, int] = {}
    bar = 0
    for i, sec in enumerate(section_plans):
        best_j = 0
        best_v = -1.0
        for j in range(sec.bars):
            v = curve[bar + j] if bar + j < len(curve) else 0.0
            if v > best_v:
                best_v = v
                best_j = j
        section_peaks[i] = best_j
        bar += sec.bars

    return DramaticCurve(
        curve=curve,
        global_peak_bar=global_peak_bar,
        section_peaks=section_peaks,
    )
