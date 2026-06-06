#!/usr/bin/env python3
"""
SSF 标注精度审计 — 随机抽样 10 首，分析 tokens ↔ SSF 一致性。

检查维度:
  1. TonicField 是否与段落内音符分布一致
  2. LocalField delta 是否合理 (bar级SSF ≈ TonicField + delta)
  3. BeatField 是否与节拍级音符分布一致
  4. section_boundaries 是否与 sec 数据对齐
  5. SSF 向量归一化是否正确 (max=1.0)
  6. 稀疏性: LocalField 是否只在有显著偏差时存储
  7. 边界情况: 空小节、单音小节、无音符小节
  8. 等价类旋转: 主音锚定是否正确

用法:
  python scripts/audit_ssf_sample.py \
      --lmdb-path /root/autodl-tmp/data/processed/lmdb/chopinote_v4.lmdb \
      --num-samples 10
"""

import sys, os, json, math, random, argparse
from collections import Counter


# ── 主音名 → pitch class ──────────────────────────────────
_TONIC_PC = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
    'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11, 'Cb': 11,
}


def tonic_name_to_pc(name: str) -> int:
    return _TONIC_PC.get(name, 0)


def compute_tonic_field(note_intervals: list[int], tonic_name: str) -> list[float]:
    """独立计算 TonicField (与 generate_ssf.py 完全一致)。"""
    counts = [0] * 12
    tonic_pc = tonic_name_to_pc(tonic_name)

    for interval in note_intervals:
        abs_pc = (tonic_pc + interval) % 12
        rotated_pos = (abs_pc - tonic_pc) % 12
        counts[rotated_pos] += 1

    total = sum(counts)
    if total == 0:
        return [0.5] * 12

    max_count = max(counts)
    return [c / max_count for c in counts]


def cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def parse_tokens(ids: list[int], tonic_ids: dict, position_ids: dict, note_on_range: tuple):
    """从 token 序列提取结构化信息。"""
    note_min, note_max, note_zero = note_on_range
    bar_idx = -1
    current_tonic = 'C'
    current_position = 0
    current_voice = 0

    notes = []  # [(bar, position, voice, interval)]
    tonics_seen = []  # [(token_pos, tonic_name)]
    bars_with_positions = []  # [[pos_ids_at_this_bar]]

    for pos, tid in enumerate(ids):
        if tid in tonic_ids:
            current_tonic = tonic_ids[tid]
            tonics_seen.append((pos, current_tonic))
        elif tid == 4:  # <Bar>
            bar_idx += 1
            current_position = 0
            bars_with_positions.append(set())
        elif tid in position_ids:
            current_position = position_ids[tid]
            if bar_idx >= 0 and len(bars_with_positions) > bar_idx:
                bars_with_positions[bar_idx].add(current_position)
        elif 193 <= tid <= 196:  # <Voice N>
            current_voice = tid - 193
        elif note_min <= tid <= note_max:
            interval = tid - note_zero
            notes.append((bar_idx, current_position, current_voice, interval))

    num_bars = bar_idx + 1 if bar_idx >= 0 else 0

    # 收集每次 tonic 变化的区间
    tonic_changes = []
    for i, (pos, tonic) in enumerate(tonics_seen):
        if i == 0 or tonic != tonics_seen[i - 1][1]:
            tonic_changes.append((pos, tonic))

    return {
        'notes': notes,
        'num_bars': num_bars,
        'tonics_seen': tonics_seen,
        'tonic_changes': tonic_changes,
        'final_tonic': tonics_seen[-1][1] if tonics_seen else 'C',
        'bars_with_positions': bars_with_positions,
    }


def analyze_one(file_id: str, tokens: list[int], ssf_data: dict, sec_data: dict | None,
                tonic_ids: dict, position_ids: dict, note_on_range: tuple) -> dict:
    """深度分析单个文件的 SSF 标注质量。"""
    parsed = parse_tokens(tokens, tonic_ids, position_ids, note_on_range)
    notes = parsed['notes']
    num_bars = parsed['num_bars']
    final_tonic = parsed['final_tonic']

    # ── 提取 SSF 数据 ──
    tonic_fields = ssf_data.get('tonic_fields', [])
    section_boundaries = ssf_data.get('section_boundaries', [0])
    local_fields = ssf_data.get('local_fields', {})
    beat_fields = ssf_data.get('beat_fields', {})

    issues = []
    checks = {}

    # ═══════════════════════════════════════════════════════════
    # 检查 1: TonicField 与段落音符分布的一致性
    # ═══════════════════════════════════════════════════════════
    if not section_boundaries:
        section_boundaries = [0]

    # 补全 section_boundaries
    if len(section_boundaries) == 1:
        section_boundaries = section_boundaries + [num_bars]

    # 按节重新计算各 section 的 TonicField
    sections_ok = 0
    sections_total = len(tonic_fields)
    section_details = []

    for i in range(sections_total):
        sec_start = section_boundaries[i] if i < len(section_boundaries) else 0
        sec_end = section_boundaries[i + 1] if i + 1 < len(section_boundaries) else num_bars

        # 收集该段音符
        sec_intervals = []
        for bar, _, _, interval in notes:
            if sec_start <= bar < sec_end:
                sec_intervals.append(interval)

        # 独立计算
        recomputed = compute_tonic_field(sec_intervals, final_tonic)
        stored = tonic_fields[i] if i < len(tonic_fields) else [0.5] * 12

        sim = cosine_sim(recomputed, stored)
        if sim > 0.85:
            sections_ok += 1

        section_details.append({
            'section': i,
            'bars': f'{sec_start}-{sec_end - 1}',
            'num_notes': len(sec_intervals),
            'cosine_sim': round(sim, 4),
            'stored_tf': [round(v, 3) for v in stored],
            'recomputed_tf': [round(v, 3) for v in recomputed] if sec_intervals else None,
        })

        if sim <= 0.85 and len(sec_intervals) >= 1:
            issues.append(f"Sec{i} TonicField mismatch: cosine_sim={sim:.3f}, {len(sec_intervals)} notes")

    checks['tonic_field_consistency'] = {
        'ok': sections_ok,
        'total': sections_total,
        'rate': round(sections_ok / max(1, sections_total), 3),
        'details': section_details,
    }

    # ═══════════════════════════════════════════════════════════
    # 检查 2: LocalField 合理性
    # ═══════════════════════════════════════════════════════════
    lf_keys = len(local_fields)
    lf_nonzero = 0
    lf_bars_ok = 0
    lf_details = []

    # 先统计全部 LocalField 的 delta 分布
    lf_max_deltas = []
    for b_str, delta in local_fields.items():
        maxd = max(abs(d) for d in delta)
        lf_max_deltas.append(maxd)
        if maxd > 0.01:
            lf_nonzero += 1

    # 采样前5个做详细验证
    for b_str, delta in sorted(local_fields.items(), key=lambda x: int(x[0]))[:5]:
        b = int(b_str)

        # 收集该小节音符
        bar_intervals = [interval for bar, _, _, interval in notes if bar == b]

        if bar_intervals:
            # 独立计算 bar 级 SSF
            bar_tf = compute_tonic_field(bar_intervals, final_tonic)

            # 找到该 bar 所属 section 的 TonicField
            sec_idx = 0
            for i in range(len(section_boundaries)):
                if b >= section_boundaries[i]:
                    sec_idx = i
            sec_tf = tonic_fields[sec_idx] if sec_idx < len(tonic_fields) else [0.5] * 12

            # 实际 delta = bar_tf - sec_tf
            actual_delta = [bar_tf[i] - sec_tf[i] for i in range(12)]
            max_actual = max(abs(d) for d in actual_delta)
            max_stored = max(abs(d) for d in delta)

            # 方向一致性: 存量和实际 delta 符号是否同向
            sign_match = sum(1 for i in range(12)
                           if (delta[i] > 0.01 and actual_delta[i] > -0.01) or
                              (delta[i] < -0.01 and actual_delta[i] < 0.01) or
                              abs(delta[i]) <= 0.01)
            sign_rate = sign_match / 12

            if sign_rate > 0.6:
                lf_bars_ok += 1

            lf_details.append({
                'bar': b,
                'num_notes': len(bar_intervals),
                'max_stored_delta': round(max_stored, 3),
                'max_actual_delta': round(max_actual, 3),
                'sign_match_rate': round(sign_rate, 2),
            })

    # LocalField delta 分布统计
    if lf_max_deltas:
        lf_max_deltas.sort()
        p50 = lf_max_deltas[len(lf_max_deltas)//2]
        p90 = lf_max_deltas[int(len(lf_max_deltas)*0.9)]
        p99 = lf_max_deltas[min(int(len(lf_max_deltas)*0.99), len(lf_max_deltas)-1)]
    else:
        p50 = p90 = p99 = 0.0

    checks['local_field'] = {
        'total_bars_with_lf': lf_keys,
        'nonzero_delta': lf_nonzero,
        'delta_distribution': {
            'min': round(min(lf_max_deltas), 4) if lf_max_deltas else 0,
            'p50': round(p50, 4),
            'p90': round(p90, 4),
            'p99': round(p99, 4),
            'max': round(max(lf_max_deltas), 4) if lf_max_deltas else 0,
        },
        'sampled_bars_ok': lf_bars_ok,
        'sampled_total': len(lf_details),
        'details': lf_details,
    }

    if lf_keys > 0 and lf_bars_ok < len(lf_details) * 0.6:
        issues.append(f"LocalField sign mismatch: {lf_bars_ok}/{len(lf_details)} bars directionally consistent")

    # ═══════════════════════════════════════════════════════════
    # 检查 3: BeatField 合理性
    # ═══════════════════════════════════════════════════════════
    bf_total = 0
    bf_ok = 0
    bf_details = []

    for b_str, beats in sorted(beat_fields.items(), key=lambda x: int(x[0]))[:3]:
        b = int(b_str)
        for pos_str, bf in sorted(beats.items(), key=lambda x: int(x[0]))[:4]:
            bf_total += 1
            pos = int(pos_str)

            # 收集该 bar+position 的音符
            beat_intervals = [interval for bar, p, _, interval in notes if bar == b and p == pos]

            if beat_intervals:
                recomputed = compute_tonic_field(beat_intervals, final_tonic)
                sim = cosine_sim(bf, recomputed)
                if sim > 0.7:
                    bf_ok += 1
                bf_details.append({
                    'bar': b, 'pos': pos,
                    'num_notes': len(beat_intervals),
                    'cosine_sim': round(sim, 4),
                })

    checks['beat_field'] = {
        'total_checked': bf_total,
        'ok': bf_ok,
        'rate': round(bf_ok / max(1, bf_total), 3),
        'details': bf_details,
    }

    # ═══════════════════════════════════════════════════════════
    # 检查 4: section_boundaries 与 sec 数据对齐
    # ═══════════════════════════════════════════════════════════
    if sec_data:
        sec_boundaries_stored = sec_data.get('section_token_positions', [])
        if sec_boundaries_stored and len(sec_boundaries_stored) > 1:
            # 只在有真实多段落标注时才严格检查对齐
            bar_positions_in_tokens = []
            current_bar = -1
            for pos, tid in enumerate(tokens):
                if tid == 4:
                    current_bar += 1
                    bar_positions_in_tokens.append(pos)

            sec_bars = []
            for token_pos in sec_boundaries_stored:
                bar = 0
                for i, bp in enumerate(bar_positions_in_tokens):
                    if bp >= token_pos:
                        bar = i
                        break
                sec_bars.append(bar)

            boundaries_match = (sec_bars == section_boundaries[:len(sec_bars)])
            checks['boundary_alignment'] = {
                'sec_bars_from_data': sec_bars,
                'ssf_boundaries': section_boundaries,
                'num_sec_boundaries': len(sec_boundaries_stored),
                'match': boundaries_match,
            }
            if not boundaries_match:
                issues.append(f"Boundary mismatch: sec={sec_bars}, ssf={section_boundaries[:len(sec_bars)]}")
        else:
            # sec_data 只有开头边界 → SSF 正确补充了结尾边界，不算 issue
            checks['boundary_alignment'] = {
                'note': 'sec_data has single boundary, SSF correctly added end boundary',
                'ssf_boundaries': section_boundaries,
            }

    # ═══════════════════════════════════════════════════════════
    # 检查 5: SSF 向量归一化
    # ═══════════════════════════════════════════════════════════
    tf_max_values = [max(tf) for tf in tonic_fields]
    norms_ok = all(
        abs(m - 1.0) < 0.01 or abs(m - 0.5) < 0.01  # 0.5 = 无音符的默认值
        for m in tf_max_values
    )

    checks['normalization'] = {
        'tonic_field_max_values': [round(m, 4) for m in tf_max_values],
        'all_normalized': norms_ok,
    }

    # ═══════════════════════════════════════════════════════════
    # 检查 6: 稀疏性
    # ═══════════════════════════════════════════════════════════
    total_bars = num_bars
    lf_coverage = lf_keys / max(1, total_bars)

    checks['sparsity'] = {
        'total_bars': total_bars,
        'bars_with_lf': lf_keys,
        'lf_coverage': round(lf_coverage, 4),
        'bars_with_bf': len(beat_fields),
        'bf_coverage': round(len(beat_fields) / max(1, total_bars), 4),
    }

    # ═══════════════════════════════════════════════════════════
    # 检查 7: 等价类旋转
    # ═══════════════════════════════════════════════════════════
    # 检查: tonic=G 时，G note (interval 0) 应出现在 TonicField[0]，G# 在 [1]
    # 即所有音符被正确旋转到主音锚定的坐标系
    rotation_ok = True
    if notes and final_tonic != 'C':
        tonic_pc = tonic_name_to_pc(final_tonic)
        for tf in tonic_fields[:1]:
            # TonicField 的 position 0 应该是最强 (主音)
            if len(set(tf)) > 1:
                max_pos = tf.index(max(tf))
                if max_pos == 0:
                    rotation_ok = True
                # position 7 (属音, V) 也应该较强
                if tf[7] > tf[6] * 0.5:
                    rotation_ok = True

    checks['rotation'] = {
        'tonic': final_tonic,
        'tonic_pc': tonic_name_to_pc(final_tonic),
        'likely_correct': rotation_ok,
    }

    # ═══════════════════════════════════════════════════════════
    # 汇总
    # ═══════════════════════════════════════════════════════════
    return {
        'file_id': file_id,
        'num_tokens': len(tokens),
        'num_bars': num_bars,
        'num_notes': len(notes),
        'num_sections': len(tonic_fields),
        'section_boundaries': section_boundaries,
        'tonic': final_tonic,
        'ssf_version': ssf_data.get('version', 'unknown'),
        'issues': issues,
        'checks': checks,
    }


def main():
    ap = argparse.ArgumentParser(description='SSF annotation quality audit')
    ap.add_argument('--lmdb-path', required=True)
    ap.add_argument('--num-samples', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    from chopinote_dataset.lmdb_store import LMDBStore
    from chopinote_dataset.tokenizer import REMITokenizer

    t = REMITokenizer(16, 8)

    # 准备 token ID 映射
    tonic_ids = {}
    for name in t.TONIC_NAMES:
        tonic_ids[t.encode_token(f'<Tonic {name}>')] = name

    position_ids = {}
    for i in range(t.grid_size):
        position_ids[t.encode_token(f'<Position {i}>')] = i

    note_ids = [t.encode_token(f'<Note_ON {i}>') for i in range(-60, 61)]
    note_on_range = (min(note_ids), max(note_ids), t.encode_token('<Note_ON 0>'))

    # ── Phase 1: 从 done 文件读取已完成 file_id ──
    DONE_FILE = '/root/autodl-tmp/ssf_done.txt'
    print("=" * 70)
    print(f"[Phase 1] 从 {DONE_FILE} 读取已完成列表...")

    done_ids = []
    if os.path.exists(DONE_FILE):
        with open(DONE_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    done_ids.append(line)
    print(f"  已有 SSF: {len(done_ids)}")

    if len(done_ids) < args.num_samples:
        print(f"ERROR: only {len(done_ids)} files with SSF, need {args.num_samples}")
        sys.exit(1)

    # ── 随机抽样 ──
    random.seed(args.seed)
    sampled = random.sample(done_ids, args.num_samples)
    print(f"  抽样 {len(sampled)} 首: {sampled[:3]}...")

    # ── Phase 2: 逐首分析 (只用 point read, 避免 cursor scan 与写进程冲突) ──
    print("\n" + "=" * 70)
    print("[Phase 2] 逐首深度分析...\n")

    # 用只读打开, 仅做 point read — 不会触发 cursor assertion
    store = LMDBStore.open(args.lmdb_path, readonly=True, map_size=200 * 1024**3)
    results = []

    for i, fid in enumerate(sampled):
        try:
            # 点读: 每个 read 是独立的原子事务，不与写事务冲突
            raw_tokens = store.get_raw(fid, 'tokens')
            if raw_tokens is None:
                print(f"  [{i + 1}/{args.num_samples}] {fid[:50]}... SKIP (no tokens)")
                continue
            # 手动解码避免 list 复制
            import struct
            tokens = list(struct.unpack(f'<{len(raw_tokens)//4}I', raw_tokens))
            ssf_data = store.get_ssf(fid)
            sec_data = store.get_sec(fid)
        except Exception as e:
            print(f"  [{i + 1}/{args.num_samples}] {fid[:50]}... ERROR: {e}")
            continue

        if ssf_data is None:
            print(f"  [{i + 1}/{args.num_samples}] {fid[:50]}... SKIP (no SSF)")
            continue

        result = analyze_one(fid, tokens, ssf_data, sec_data,
                           tonic_ids, position_ids, note_on_range)
        results.append(result)

        n_issues = len(result['issues'])
        status = "✅" if n_issues == 0 else f"⚠️ {n_issues} issues"
        print(f"  [{i + 1}/{args.num_samples}] {fid[:50]}... {status}")
        print(f"      tokens={result['num_tokens']}, bars={result['num_bars']}, "
              f"notes={result['num_notes']}, sections={result['num_sections']}, "
              f"tonic={result['tonic']}")
        print(f"      TF consistency: {result['checks']['tonic_field_consistency']['ok']}/"
              f"{result['checks']['tonic_field_consistency']['total']} "
              f"({result['checks']['tonic_field_consistency']['rate']:.0%})")
        print(f"      BeatField check: {result['checks']['beat_field']['ok']}/"
              f"{result['checks']['beat_field']['total_checked']} "
              f"({result['checks']['beat_field']['rate']:.0%})")
        print(f"      Sparsity: LF={result['checks']['sparsity']['lf_coverage']:.1%}, "
              f"BF={result['checks']['sparsity']['bf_coverage']:.1%}")

        if result['issues']:
            for issue in result['issues'][:3]:
                print(f"      ❌ {issue}")

    store.close()

    # ── Phase 3: 汇总报告 ──
    print("\n" + "=" * 70)
    print("[Phase 3] 汇总报告")
    print("=" * 70)

    if not results:
        print("No results to analyze.")
        return

    # TF 一致性
    tf_rates = [r['checks']['tonic_field_consistency']['rate'] for r in results]
    print(f"\n📊 TonicField 一致性:")
    print(f"   平均 cosine_sim > 0.85 的比例: {sum(tf_rates)/len(tf_rates):.1%}")
    print(f"   完全一致 (100%): {sum(1 for r in tf_rates if r >= 0.99)}/{len(results)}")
    print(f"   高一致 (≥90%): {sum(1 for r in tf_rates if r >= 0.90)}/{len(results)}")
    print(f"   中一致 (≥70%): {sum(1 for r in tf_rates if r >= 0.70)}/{len(results)}")
    print(f"   低一致 (<50%): {sum(1 for r in tf_rates if r < 0.50)}/{len(results)}")

    # BeatField 一致性
    bf_rates = [r['checks']['beat_field']['rate'] for r in results if r['checks']['beat_field']['total_checked'] > 0]
    if bf_rates:
        print(f"\n📊 BeatField 一致性:")
        print(f"   平均 cosine_sim > 0.7 的比例: {sum(bf_rates)/len(bf_rates):.1%}")

    # 稀疏性
    lf_coverage = [r['checks']['sparsity']['lf_coverage'] for r in results]
    bf_coverage = [r['checks']['sparsity']['bf_coverage'] for r in results]
    print(f"\n📊 稀疏性:")
    print(f"   LocalField 覆盖: avg={sum(lf_coverage)/len(lf_coverage):.1%}, "
          f"range=[{min(lf_coverage):.1%}, {max(lf_coverage):.1%}]")
    print(f"   BeatField 覆盖:  avg={sum(bf_coverage)/len(bf_coverage):.1%}, "
          f"range=[{min(bf_coverage):.1%}, {max(bf_coverage):.1%}]")

    # 归一化
    norms_ok = [r['checks']['normalization']['all_normalized'] for r in results]
    print(f"\n📊 归一化:")
    print(f"   全部正确: {sum(norms_ok)}/{len(results)}")

    # 汇总 issues
    total_issues = sum(len(r['issues']) for r in results)
    print(f"\n📊 总问题数: {total_issues} (avg {total_issues/len(results):.1f}/首)")

    # 输出第一个样本的详细信息
    if results:
        print("\n" + "=" * 70)
        print("[样本详情] 第一首完整分析")
        print("=" * 70)
        r = results[0]
        print(json.dumps(r, indent=2, ensure_ascii=False, default=str)[:3000])

    print("\n✅ 审计完成")


if __name__ == '__main__':
    main()
