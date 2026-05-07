"""
生成结果交叉验证脚本。

检测生成的 MusicXML 是否有严重错误（空输出、音高崩溃、调性跑偏、
重复循环、token 结构异常），但不追求 token 级无损 round-trip。

用法:
    python scripts/validate_generation.py output.musicxml
    python scripts/validate_generation.py output.musicxml --seed-tokens seed.json
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chopinote_dataset.tokenizer import REMITokenizer
from chopinote_dataset.converter import MusicXMLToREMI


def _get_key_from_tokens(tokens: list[int], tokenizer: REMITokenizer) -> Optional[str]:
    """从 token 序列中提取第一个 Key 调性标记。"""
    for tid in tokens:
        t = tokenizer.decode_token(tid)
        if t.startswith(tokenizer.KEY):
            return t[len(tokenizer.KEY) + 1:-1]
    return None


def _count_bars(tokens: list[int], tokenizer: REMITokenizer) -> int:
    return sum(1 for t in tokens if t == tokenizer.bar_token_id)


def _extract_notes(tokens: list[int], tokenizer: REMITokenizer) -> list[dict]:
    """从 token 序列中提取所有 NOTE_ON 事件的位置和音高。"""
    notes = []
    events = tokenizer.detokenize(tokens)
    cur_bar = 0
    for etype, evalue in events:
        if etype == tokenizer.BAR:
            if cur_bar > 0:
                notes.append({'bar': cur_bar, 'type': 'bar_end'})
            cur_bar += 1
        elif etype == tokenizer.NOTE_ON:
            notes.append({'bar': cur_bar, 'pitch': evalue, 'type': 'note'})
    return notes


def _count_token_types(tokens: list[int], tokenizer: REMITokenizer) -> dict:
    """统计每种 token 类型的出现次数。"""
    counts = {}
    events = tokenizer.detokenize(tokens)
    for etype, _ in events:
        counts[etype] = counts.get(etype, 0) + 1
    return counts


def _positions_for_time_sig(time_sig, grid_size: int = 16) -> int:
    """根据拍号 (num, den) 推算小节总位置数。"""
    if time_sig is None:
        return grid_size  # 默认 4/4
    num, den = time_sig
    positions = num * grid_size // den  # 如 3/4 → 3*16/4 = 12
    return max(1, min(positions, grid_size))


def _validate_notation_legality(
    roundtrip_tokens: list[int],
    tokenizer: REMITokenizer,
) -> dict:
    """乐谱乐理合法性验证。

    检查项：
      1. 同音重复 — 同声部同 position 同 pitch
      2. 时值溢出 — position + duration > 小节总位置数
      3. 零/负时值 — duration <= 0
      4. 连音不配对 — TupletStart/TupletEnd 不匹配
      5. 空内容小节 — 有 Bar 但无任何 Note_ON 或 Rest

    Returns:
        {'passed': bool, 'checks': dict, 'detail': str, 'violations': dict}
    """
    events = tokenizer.detokenize(roundtrip_tokens)

    cur_bar = 0
    cur_pos = 0
    cur_program = 0
    cur_subtrack = 0
    tuplet_stack = 0
    last_time_sig = None  # (num, den)
    bar_has_content: dict[int, bool] = {}
    # (bar, program, subtrack, position) -> set[pitch]
    note_positions: dict[tuple, set] = {}

    violations = {
        'duplicate_pitch': [],
        'duration_overflow': [],
        'zero_duration': [],
        'tuplet_mismatch': [],
        'empty_measure': [],
    }

    i = 0
    while i < len(events):
        etype, evalue = events[i]

        if etype in (tokenizer.BOS, tokenizer.EOS):
            if etype == tokenizer.EOS:
                break
            i += 1
            continue
        elif etype == tokenizer.BAR:
            cur_bar += 1
            i += 1
            continue
        elif etype == tokenizer.POSITION:
            cur_pos = evalue
            i += 1
            continue
        elif etype == tokenizer.PROGRAM:
            val_parts = evalue.split('_')
            cur_program = int(val_parts[0])
            cur_subtrack = int(val_parts[1]) if len(val_parts) > 1 else 0
            i += 1
            continue
        elif etype == tokenizer.TIMESIG:
            parts = evalue.split('/')
            last_time_sig = (int(parts[0]), int(parts[1]))
            i += 1
            continue
        elif etype == tokenizer.TUPLET_START:
            tuplet_stack += 1
            i += 1
            continue
        elif etype == tokenizer.TUPLET_END:
            tuplet_stack -= 1
            if tuplet_stack < 0:
                violations['tuplet_mismatch'].append(
                    f'小节 {cur_bar}: 多余的 TupletEnd（无匹配的 TupletStart）'
                )
                tuplet_stack = 0
            i += 1
            continue
        elif etype == tokenizer.NOTE_ON:
            pitch = evalue
            bar_has_content[cur_bar] = True

            # 同音重复
            dk = (cur_bar, cur_program, cur_subtrack, cur_pos)
            note_positions.setdefault(dk, set())
            if pitch in note_positions[dk]:
                violations['duplicate_pitch'].append(
                    f'小节 {cur_bar}, 位置 {cur_pos}, 声部 {cur_program}/{cur_subtrack}: 音高 {pitch} 重复'
                )
            else:
                note_positions[dk].add(pitch)

            # 跳过后面的 Velocity
            if i + 1 < len(events) and events[i + 1][0] == tokenizer.VELOCITY:
                i += 1
            # 读取 Duration
            dur = 4
            if i + 1 < len(events) and events[i + 1][0] == tokenizer.DURATION:
                dur = events[i + 1][1]
                i += 1

            # 零时值
            if dur <= 0:
                violations['zero_duration'].append(
                    f'小节 {cur_bar}, 位置 {cur_pos}, 音高 {pitch}: 时值 {dur} <= 0'
                )

            # 时值溢出
            total_pos = _positions_for_time_sig(last_time_sig, tokenizer.grid_size)
            if cur_pos + dur > total_pos:
                violations['duration_overflow'].append(
                    f'小节 {cur_bar}, 位置 {cur_pos}, 音高 {pitch}: '
                    f'pos({cur_pos}) + dur({dur}) = {cur_pos + dur} > {total_pos}'
                )

            i += 1
            continue
        elif etype == tokenizer.REST:
            bar_has_content[cur_bar] = True

            dur = 4
            if i + 1 < len(events) and events[i + 1][0] == tokenizer.DURATION:
                dur = events[i + 1][1]
                i += 1

            if dur <= 0:
                violations['zero_duration'].append(
                    f'小节 {cur_bar}, 位置 {cur_pos}: 休止符时值 {dur} <= 0'
                )

            total_pos = _positions_for_time_sig(last_time_sig, tokenizer.grid_size)
            if cur_pos + dur > total_pos:
                violations['duration_overflow'].append(
                    f'小节 {cur_bar}, 位置 {cur_pos}: 休止符 pos({cur_pos}) + '
                    f'dur({dur}) = {cur_pos + dur} > {total_pos}'
                )

            i += 1
            continue
        else:
            i += 1

    # 结尾未闭合的连音
    if tuplet_stack > 0:
        violations['tuplet_mismatch'].append(
            f'结尾: {tuplet_stack} 个 TupletStart 未闭合'
        )

    # 空内容小节（跳过 Bar 0）
    for bar_num in range(1, cur_bar + 1):
        if bar_num not in bar_has_content:
            violations['empty_measure'].append(
                f'小节 {bar_num}: 无音符或休止符'
            )

    all_passed = all(len(v) == 0 for v in violations.values())
    failed_items = [k for k, v in violations.items() if v]
    detail_parts = [f'{k}({len(v)}处)' for k, v in violations.items() if v]
    detail = ', '.join(detail_parts) if detail_parts else '全部检查通过'

    return {
        'passed': all_passed,
        'checks': violations,
        'detail': detail,
        'failed_items': failed_items,
    }


def validate_generated_xml(
    musicxml_path: str,
    seed_tokens: Optional[list[int]] = None,
    tokenizer: Optional[REMITokenizer] = None,
    lock_key: Optional[bool] = None,
    model_vocab_size: Optional[int] = None,
) -> dict:
    """对生成的 MusicXML 执行交叉验证。

    Args:
        musicxml_path: 生成输出的 MusicXML 文件路径
        seed_tokens: 原始的种子 token（用于对比调性等）
        tokenizer: REMI tokenizer 实例（不传则新建默认）
        lock_key: 若为 True，检查生成的 key 是否与 seed 一致
        model_vocab_size: 若指定，检测 OOB token

    Returns:
        {"passed": bool, "checks": dict, "summary": str}
    """
    if tokenizer is None:
        tokenizer = REMITokenizer(grid_size=16, velocity_levels=8)

    if not os.path.isfile(musicxml_path):
        return {
            'passed': False,
            'checks': {'file_exists': False},
            'summary': f'文件不存在: {musicxml_path}',
        }

    checks = {}
    all_passed = True

    # ── 1. 空检查 ──────────────────────────────────────────────
    conv = MusicXMLToREMI(grid_size=tokenizer.grid_size,
                          velocity_levels=tokenizer.velocity_levels)
    roundtrip_tokens, _ = conv.convert(musicxml_path, collect_metadata=False)

    if not roundtrip_tokens:
        checks['empty'] = {'passed': False, 'detail': 'round-trip 解析失败或 token 序列为空'}
        all_passed = False
    else:
        checks['empty'] = {'passed': True, 'detail': f'{len(roundtrip_tokens)} tokens'}

    if not roundtrip_tokens:
        checks['summary'] = '严重错误：无法解析生成的 MusicXML'
        return {'passed': False, 'checks': checks, 'summary': checks['summary']}

    # ── 2. 小节数检查 ──────────────────────────────────────────
    bar_count = _count_bars(roundtrip_tokens, tokenizer)
    if bar_count == 0:
        checks['bar_count'] = {'passed': False, 'detail': '小节数为 0'}
        all_passed = False
    elif bar_count > 500:
        checks['bar_count'] = {'passed': False, 'detail': f'小节数异常过多: {bar_count}'}
        all_passed = False
    else:
        checks['bar_count'] = {'passed': True, 'detail': f'{bar_count} 小节'}

    # ── 3. 调性一致性 ──────────────────────────────────────────
    roundtrip_key = _get_key_from_tokens(roundtrip_tokens, tokenizer)
    checks['key'] = {'passed': True, 'detail': roundtrip_key or '未检测到调性标记'}

    if lock_key and seed_tokens is not None:
        seed_key = _get_key_from_tokens(seed_tokens, tokenizer)
        if seed_key and roundtrip_key and roundtrip_key != seed_key:
            checks['key'] = {
                'passed': False,
                'detail': f'seed 调性 {seed_key}，生成调性 {roundtrip_key}（锁定时应一致）',
            }
            all_passed = False

    # ── 4. 音高范围检查 ────────────────────────────────────────
    events = tokenizer.detokenize(roundtrip_tokens)
    pitches = [v for e, v in events if e == tokenizer.NOTE_ON]
    invalid_pitches = [p for p in pitches if not (0 <= p <= 127)]
    if invalid_pitches:
        checks['pitch_range'] = {
            'passed': False,
            'detail': f'{len(invalid_pitches)} 个音高越界: {set(invalid_pitches)}',
        }
        all_passed = False
    else:
        pitch_min = min(pitches) if pitches else 'N/A'
        pitch_max = max(pitches) if pitches else 'N/A'
        checks['pitch_range'] = {
            'passed': True,
            'detail': f'所有音高在 0-127 范围内 ({pitch_min}~{pitch_max})',
        }

    # ── 4.5 调性内部一致性 ─────────────────────────────────────
    if roundtrip_key and pitches:
        from chopinote_model.generate import KEY_TO_DIATONIC_PITCHES
        diatonic = KEY_TO_DIATONIC_PITCHES.get(roundtrip_key)
        if diatonic:
            in_key = sum(1 for p in pitches if (p % 12) in diatonic)
            ratio = in_key / len(pitches)
            checks['key_consistency'] = {
                'passed': ratio > 0.6,
                'detail': f'{ratio:.1%} 音符在调内 ({in_key}/{len(pitches)})',
            }
            if ratio <= 0.6:
                all_passed = False

    # ── 5. 音符密度检查 ────────────────────────────────────────
    note_count = len(pitches)
    if bar_count > 0:
        density = note_count / bar_count
        if density == 0:
            checks['density'] = {'passed': False, 'detail': '全曲无音符（全休止）'}
            all_passed = False
        elif density > 200:
            checks['density'] = {'passed': False, 'detail': f'音符密度异常: {density:.1f} note/bar'}
            all_passed = False
        else:
            checks['density'] = {'passed': True, 'detail': f'{density:.1f} note/bar ({note_count} notes)'}
    else:
        checks['density'] = {'passed': True, 'detail': '跳过（小节数为 0）'}

    # ── 6. OOB token 检测 ──────────────────────────────────────
    if model_vocab_size is not None:
        oob = [t for t in roundtrip_tokens if t >= model_vocab_size]
        if oob:
            checks['oob'] = {'passed': False, 'detail': f'{len(oob)} 个 token 超出词表 {model_vocab_size}'}
            all_passed = False
        else:
            checks['oob'] = {'passed': True, 'detail': '无越界 token'}

    # ── 7. 重复模式检测 ────────────────────────────────────────
    # 按小节分割 token
    bars_tokens = []
    current_bar = []
    for tid in roundtrip_tokens:
        if tid == tokenizer.bar_token_id and current_bar:
            bars_tokens.append(current_bar)
            current_bar = []
        current_bar.append(tid)
    if current_bar:
        bars_tokens.append(current_bar)

    repeat_detected = False
    if len(bars_tokens) >= 8:
        for n in range(4, 9):  # 检查 4~8 小节的重复
            last_n = bars_tokens[-n:]
            if all(a == b for a, b in zip(last_n, last_n[-1:] * n)):
                repeat_detected = True
                break
    if repeat_detected:
        checks['repetition'] = {'passed': False, 'detail': f'检测到最后 {n} 个小节完全重复'}
        all_passed = False
    else:
        checks['repetition'] = {'passed': True, 'detail': '无重复循环'}

    # ── 8. 乐谱乐理合法性验证 ─────────────────────────────────────
    if roundtrip_tokens:
        notation_result = _validate_notation_legality(roundtrip_tokens, tokenizer)
        if notation_result['passed']:
            checks['notation'] = {'passed': True, 'detail': notation_result['detail']}
        else:
            failed_str = ', '.join(notation_result['failed_items'])
            checks['notation'] = {
                'passed': False,
                'detail': f'{notation_result["detail"]}',
            }
            all_passed = False

    # ── 9. 损失报告（仅信息，不判定失败） ─────────────────────
    if seed_tokens:
        orig_counts = _count_token_types(seed_tokens, tokenizer)
        rt_counts = _count_token_types(roundtrip_tokens, tokenizer)
        lost_types = []
        preserved_types = []
        for ttype in orig_counts:
            if ttype in (tokenizer.BOS, tokenizer.EOS, tokenizer.MASK, tokenizer.PAD):
                continue
            orig_n = orig_counts[ttype]
            rt_n = rt_counts.get(ttype, 0)
            if rt_n == 0:
                lost_types.append(f'{ttype} ({orig_n}→0)')
            elif rt_n < orig_n * 0.5:
                lost_types.append(f'{ttype} ({orig_n}→{rt_n})')
            else:
                preserved_types.append(f'{ttype} ({orig_n}→{rt_n})')
        checks['loss_report'] = {
            'passed': True,
            'detail': f'丢失: {", ".join(lost_types[:5]) if lost_types else "无"} | 保留: {len(preserved_types)} 类',
            'lost_types': lost_types,
            'preserved_types': preserved_types,
        }

    # ── 汇总 ────────────────────────────────────────────────────
    if all_passed:
        summary = f'验证通过 | {bar_count} 小节 | {len(pitches)} 音符 | {len(roundtrip_tokens)} tokens'
    else:
        failed = [k for k, v in checks.items() if isinstance(v, dict) and not v.get('passed', True)]
        summary = f'{len(failed)} 项检查失败: {", ".join(failed)}'

    return {'passed': all_passed, 'checks': checks, 'summary': summary}


def main():
    parser = argparse.ArgumentParser(
        description='验证生成的 MusicXML 文件质量'
    )
    parser.add_argument('input', help='生成的 MusicXML 文件路径')
    parser.add_argument('--seed-tokens', type=str, default=None,
                        help='原始 seed token JSON 文件（用于对比调性等）')
    parser.add_argument('--lock-key', action='store_true',
                        help='检查是否存在 key 偏移')
    parser.add_argument('--model-vocab-size', type=int, default=None,
                        help='模型词表大小（检测 OOB）')
    args = parser.parse_args()

    seed = None
    if args.seed_tokens:
        with open(args.seed_tokens, 'r', encoding='utf-8') as f:
            seed = json.load(f)

    result = validate_generated_xml(
        args.input,
        seed_tokens=seed,
        lock_key=args.lock_key,
        model_vocab_size=args.model_vocab_size,
    )

    print()
    print('  [验证结果]')
    for name, check in result['checks'].items():
        status = '[OK]' if check.get('passed', True) else '[!!]'
        print(f'    {status} {name}: {check.get("detail", "")}')
    print()
    if result['passed']:
        print(f'  {result["summary"]}')
    else:
        print(f'  {result["summary"]}')
        sys.exit(1)


if __name__ == '__main__':
    main()
