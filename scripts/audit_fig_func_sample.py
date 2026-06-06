#!/usr/bin/env python3
"""
Fig + Func 标注精度审计 — 随机抽样 10 首，独立重算后对比。

用法:
  python scripts/audit_fig_func_sample.py \
      --lmdb-path /root/autodl-tmp/data/processed/lmdb/chopinote_v4.lmdb \
      --num-samples 10
"""
import sys, os, json, math, random, argparse, struct
from collections import Counter

# ── Tokenizer constants ─────────────────────────────────────
def _get_constants():
    from chopinote_dataset.tokenizer import REMITokenizer
    t = REMITokenizer(16, 8)

    tonic_ids = {t.encode_token(f'<Tonic {n}>'): n for n in t.TONIC_NAMES}
    voice_ids = {t.encode_token(f'<Voice {v}>'): v for v in range(4)}
    note_ids = [t.encode_token(f'<Note_ON {i}>') for i in range(-60, 61)]
    pos_ids = [t.encode_token(f'<Position {i}>') for i in range(16)]
    bar_id = t.bar_token_id

    return {
        'tonic_ids': tonic_ids,
        'voice_ids': voice_ids,
        'note_min': min(note_ids),
        'note_max': max(note_ids),
        'note_zero': t.encode_token('<Note_ON 0>'),
        'pos_min': min(pos_ids),
        'pos_max': max(pos_ids),
        'bar_id': bar_id,
    }


# ═══════════════════════════════════════════════════════════════
# Fig 独立重算
# ═══════════════════════════════════════════════════════════════

FIG_NONE, FIG_BLOCK, FIG_ALBERTI, FIG_ARPEGGIO = 0, 1, 2, 3
FIG_STRIDE, FIG_OCTAVE_TREMOLO, FIG_WALKING_BASS = 4, 5, 6
FIG_COUNTERMELODY, FIG_PEDAL, FIG_WALTZ = 7, 8, 9
FIG_BROKEN_OCTAVE, FIG_TREMOLO = 10, 11

FIG_NAMES = {
    1:'block',2:'alberti',3:'arpeggio',4:'stride',5:'octave_tremolo',
    6:'walking_bass',7:'countermelody',8:'pedal',9:'waltz',
    10:'broken_octave',11:'tremolo'
}


def _is_alberti(intervals):
    if len(intervals) < 6:
        return False
    for period in (3, 4):
        for offset in range(period):
            errors = 0
            for i in range(offset, len(intervals) - period, period):
                if abs(intervals[i] - intervals[i+period]) > 1:
                    errors += 1
            if errors <= len(intervals) // (period * 3):
                return True
    return False


def _is_stride(notes):
    strong = [n for n in notes if n[1] % 4 == 0]
    if len(strong) < 2:
        return False
    strong_pitches = [s[0] for s in strong]
    jumps = [abs(strong_pitches[i+1] - strong_pitches[i]) for i in range(len(strong_pitches)-1)]
    return any(j > 12 for j in jumps)


def _is_waltz(notes):
    pos_groups = {}
    for n in notes:
        pos_groups.setdefault(n[1], []).append(n[0])
    if len(pos_groups) < 2:
        return False
    return max(len(v) for v in pos_groups.values()) >= 2


def _pitch_variance(pitches):
    mean = sum(pitches) / len(pitches)
    return sum((p - mean) ** 2 for p in pitches) / len(pitches)


def classify_window(notes):
    """与 generate_fig.py 完全一致的分类逻辑。"""
    if len(notes) < 4:
        return FIG_NONE
    pitches = [n[0] for n in notes]
    intervals = [pitches[i+1] - pitches[i] for i in range(len(pitches)-1)]

    pos_counts = Counter(n[1] for n in notes)
    if max(pos_counts.values()) >= 3:
        return FIG_BLOCK

    if len(intervals) >= 4 and all(abs(i) in (0, 12) for i in intervals):
        if any(abs(i) == 12 for i in intervals):
            return FIG_OCTAVE_TREMOLO

    if len(intervals) >= 4 and sum(1 for i in intervals if abs(i) <= 1) >= len(intervals)*0.7:
        return FIG_TREMOLO

    if _is_alberti(intervals):
        return FIG_ALBERTI

    if all(abs(i) in (0, 12) for i in intervals):
        return FIG_BROKEN_OCTAVE

    if _is_stride(notes):
        return FIG_STRIDE

    if _is_waltz(notes):
        return FIG_WALTZ

    if len(intervals) >= 4 and all(abs(i) <= 2 for i in intervals):
        return FIG_WALKING_BASS

    if len(set(pitches)) <= 2:
        return FIG_PEDAL

    if len(pitches) >= 8 and _pitch_variance(pitches) > 30:
        return FIG_COUNTERMELODY

    return FIG_ARPEGGIO


def recompute_fig(ids, C):
    """独立重算 fig，与 generate_fig.py 完全一致。"""
    per_bar_voice = {}
    bar_idx = -1
    current_voice = 0
    current_pos = 0

    for tid in ids:
        if tid == C['bar_id']:
            bar_idx += 1
            per_bar_voice.setdefault(bar_idx, {})
        elif tid in C['voice_ids']:
            current_voice = C['voice_ids'][tid]
        elif C['pos_min'] <= tid <= C['pos_max']:
            current_pos = tid - C['pos_min']
        elif C['note_min'] <= tid <= C['note_max']:
            interval = tid - C['note_zero']
            per_bar_voice.setdefault(bar_idx, {}).setdefault(current_voice, []).append(
                (interval, current_pos))

    if bar_idx < 0:
        return {}

    WINDOW_BARS = 4
    bar_figs = {}
    all_voices = set()
    for bar in per_bar_voice.values():
        all_voices.update(bar.keys())

    for voice in all_voices:
        voice_bars = {}
        for b in range(bar_idx + 1):
            notes = per_bar_voice.get(b, {}).get(voice, [])
            if notes:
                voice_bars[b] = notes

        if len(voice_bars) < WINDOW_BARS:
            continue

        sorted_bars = sorted(voice_bars.keys())
        for i in range(0, len(sorted_bars) - WINDOW_BARS + 1, max(1, WINDOW_BARS // 2)):
            window_bars = sorted_bars[i:i+WINDOW_BARS]
            window_notes = []
            for b in window_bars:
                window_notes.extend(voice_bars[b])
            if len(window_notes) < 4:
                continue
            fig = classify_window(window_notes)
            if fig == FIG_NONE:
                continue
            for b in window_bars:
                bar_figs.setdefault(b, {})[voice] = fig

    return {str(b): {str(v): f for v, f in vf.items()} for b, vf in bar_figs.items()}


def audit_fig(file_id, tokens, stored_fig, C):
    """对比存储的 fig 与独立重算结果。"""
    recomputed = recompute_fig(tokens, C)
    stored_bar_figs = stored_fig.get('bar_figs', {}) if stored_fig else {}

    all_bars = set(recomputed.keys()) | set(stored_bar_figs.keys())
    if not all_bars:
        return {'status': 'empty', 'match_rate': 1.0, 'details': [], 'issues': []}

    matches = 0
    mismatches = 0
    missing_in_stored = 0
    missing_in_recomputed = 0
    details = []
    issues = []

    for b_str in sorted(all_bars, key=int)[:10]:
        b = int(b_str)
        r_voices = recomputed.get(b_str, {})
        s_voices = stored_bar_figs.get(b_str, {})

        for v_str in set(r_voices.keys()) | set(s_voices.keys()):
            r_fig = r_voices.get(v_str)
            s_fig = s_voices.get(v_str)

            if r_fig is not None and s_fig is not None:
                if r_fig == s_fig:
                    matches += 1
                else:
                    mismatches += 1
                    details.append({
                        'bar': b, 'voice': int(v_str),
                        'stored': FIG_NAMES.get(s_fig, str(s_fig)),
                        'recomputed': FIG_NAMES.get(r_fig, str(r_fig)),
                    })
                    issues.append(f"Bar{b} V{v_str}: stored={FIG_NAMES.get(s_fig,'?')} recomputed={FIG_NAMES.get(r_fig,'?')}")
            elif r_fig is None:
                missing_in_recomputed += 1
            else:
                missing_in_stored += 1

    total_compared = matches + mismatches
    match_rate = matches / max(1, total_compared)

    return {
        'status': 'ok',
        'total_bars_compared': total_compared,
        'matches': matches,
        'mismatches': mismatches,
        'match_rate': round(match_rate, 3),
        'missing_in_stored': missing_in_stored,
        'missing_in_recomputed': missing_in_recomputed,
        'details': details[:5],
        'issues': issues[:5],
    }


# ═══════════════════════════════════════════════════════════════
# Func 独立重算
# ═══════════════════════════════════════════════════════════════

FUNCTION_TEMPLATES = {
    'T':  [1.0, 0.1, 0.2, 0.1, 0.7, 0.1, 0.1, 0.7, 0.1, 0.3, 0.1, 0.2],
    'SD': [0.5, 0.1, 0.3, 0.1, 0.3, 0.8, 0.1, 0.3, 0.6, 0.2, 0.2, 0.1],
    'D':  [0.5, 0.1, 0.4, 0.1, 0.3, 0.3, 0.1, 0.9, 0.1, 0.2, 0.6, 0.4],
    'SDom': [0.3, 0.1, 0.8, 0.1, 0.2, 0.3, 0.7, 0.3, 0.1, 0.1, 0.1, 0.1],
}
FUNC_NAMES = ['T', 'SD', 'D', 'SDom']
CLASSIFY_THRESHOLD = 0.50
CONFIDENCE_FLOOR = 0.55
MARKOV_TRANSITION = {
    'T':    {'T': 0.3, 'SD': 0.4, 'D': 0.25, 'SDom': 0.05},
    'SD':   {'T': 0.15, 'SD': 0.2, 'D': 0.55, 'SDom': 0.1},
    'D':    {'T': 0.7, 'SD': 0.1, 'D': 0.15, 'SDom': 0.05},
    'SDom': {'T': 0.15, 'SD': 0.0, 'D': 0.8, 'SDom': 0.05},
}


def cosine_sim(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    return dot/(na*nb) if na and nb else 0.0


def classify_ssf_vector(ssf_vec, prev_func=None, use_markov=True):
    """与 annotate_function.py 完全一致。"""
    if all(v == 0.5 for v in ssf_vec):
        return (None, 0.0)

    sim_scores = {f: cosine_sim(ssf_vec, FUNCTION_TEMPLATES[f]) for f in FUNC_NAMES}
    best_func = max(sim_scores, key=sim_scores.get)
    best_sim = sim_scores[best_func]

    if best_sim < CLASSIFY_THRESHOLD:
        return (None, 0.0)

    if use_markov and prev_func is not None:
        posterior = {}
        for f in FUNC_NAMES:
            prior = MARKOV_TRANSITION.get(prev_func, {}).get(f, 0.25)
            posterior[f] = sim_scores[f] * (0.3 + 0.7 * prior)
        best_func = max(posterior, key=posterior.get)
        best_score = posterior[best_func]
        total = sum(posterior.values())
        confidence = best_score / total if total > 0 else 0.0
    else:
        total = sum(sim_scores.values())
        confidence = best_sim / total if total > 0 else 0.0

    if confidence < CONFIDENCE_FLOOR:
        return (None, 0.0)
    return (best_func, round(confidence, 3))


def recompute_func(ssf_data):
    """独立重算 func，与 annotate_function.py 完全一致。"""
    tonic_fields = ssf_data.get('tonic_fields', [])
    local_fields = ssf_data.get('local_fields', {})
    beat_fields = ssf_data.get('beat_fields', {})
    section_boundaries = ssf_data.get('section_boundaries', [0])

    # Section-level
    section_funcs = []
    for i, tf in enumerate(tonic_fields):
        func, conf = classify_ssf_vector(tf, prev_func=None, use_markov=False)
        section_funcs.append({'section': i, 'func': func or 'none', 'confidence': conf})

    # Bar-level
    bar_funcs = []
    prev = 'T'
    if beat_fields:
        num_bars = max(int(k) for k in beat_fields.keys()) + 1
    elif local_fields:
        num_bars = max(int(k) for k in local_fields.keys()) + 1
    else:
        num_bars = len(tonic_fields) * 8

    for b in range(num_bars):
        sec_idx = 0
        for i in range(len(section_boundaries)):
            if b >= section_boundaries[i]:
                sec_idx = i
        base_tf = tonic_fields[sec_idx] if sec_idx < len(tonic_fields) else [0.5]*12
        bar_ssf = list(base_tf)
        b_str = str(b)
        if b_str in local_fields:
            delta = local_fields[b_str]
            for j in range(12):
                bar_ssf[j] = max(0.0, min(1.0, bar_ssf[j] + delta[j]))
        func, conf = classify_ssf_vector(bar_ssf, prev_func=prev, use_markov=True)
        bar_funcs.append({'bar': b, 'func': func or 'none', 'confidence': conf})
        if func is not None:
            prev = func

    # Beat-level
    beat_funcs = []
    for b_str, beats in sorted(beat_fields.items(), key=lambda x: int(x[0])):
        b = int(b_str)
        prev_beat = 'T'
        for pos_str, bf in sorted(beats.items(), key=lambda x: int(x[0])):
            pos = int(pos_str)
            func, conf = classify_ssf_vector(bf, prev_func=prev_beat, use_markov=True)
            beat_funcs.append({'bar': b, 'pos': pos, 'func': func or 'none', 'confidence': conf})
            if func is not None:
                prev_beat = func

    return {'section_funcs': section_funcs, 'bar_funcs': bar_funcs, 'beat_funcs': beat_funcs}


def audit_func(file_id, ssf_data, stored_func):
    """对比存储的 func 与独立重算结果。"""
    if ssf_data is None:
        return {'status': 'skip', 'reason': 'no_ssf', 'issues': []}

    recomputed = recompute_func(ssf_data)

    # 对比 section_funcs
    s_secs = {(e['section'], e['func']) for e in stored_func.get('section_funcs', [])}
    r_secs = {(e['section'], e['func']) for e in recomputed['section_funcs']}
    sec_match = len(s_secs & r_secs) / max(1, len(s_secs | r_secs))

    # 对比 bar_funcs (只看 func 标签)
    s_bars = {e['bar']: e['func'] for e in stored_func.get('bar_funcs', [])}
    r_bars = {e['bar']: e['func'] for e in recomputed['bar_funcs']}
    bar_matches = sum(1 for b in s_bars if s_bars.get(b) == r_bars.get(b))
    bar_total = len(s_bars)
    bar_match_rate = bar_matches / max(1, bar_total)

    # 对比 beat_funcs
    s_beats = {(e['bar'], e['pos']): e['func'] for e in stored_func.get('beat_funcs', [])}
    r_beats = {(e['bar'], e['pos']): e['func'] for e in recomputed['beat_funcs']}
    beat_matches = sum(1 for k in s_beats if s_beats.get(k) == r_beats.get(k))
    beat_total = len(s_beats)
    beat_match_rate = beat_matches / max(1, beat_total)

    issues = []
    if bar_match_rate < 0.9:
        issues.append(f"Bar func mismatch: {bar_matches}/{bar_total} ({bar_match_rate:.1%})")
    if beat_match_rate < 0.9:
        issues.append(f"Beat func mismatch: {beat_matches}/{beat_total} ({beat_match_rate:.1%})")

    # 检查 func 分布是否合理
    bar_func_dist = Counter(e['func'] for e in stored_func.get('bar_funcs', []))
    all_none = bar_func_dist.get('none', 0) / max(1, bar_total)

    # 检查 Markov 合理性 (T→SD→D→T 循环应最常见)
    transitions = Counter()
    bar_list = stored_func.get('bar_funcs', [])
    for i in range(len(bar_list)-1):
        f1 = bar_list[i]['func']
        f2 = bar_list[i+1]['func']
        if f1 != 'none' and f2 != 'none':
            transitions[f'{f1}→{f2}'] += 1

    return {
        'status': 'ok',
        'section_match': round(sec_match, 3),
        'bar_match_rate': round(bar_match_rate, 3),
        'beat_match_rate': round(beat_match_rate, 3),
        'bar_total': bar_total,
        'beat_total': beat_total,
        'all_none_rate': round(all_none, 3),
        'bar_func_distribution': dict(bar_func_dist.most_common()),
        'top_transitions': dict(transitions.most_common(5)),
        'issues': issues,
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lmdb-path', required=True)
    ap.add_argument('--num-samples', type=int, default=10)
    ap.add_argument('--seed', type=int, default=123)
    args = ap.parse_args()

    C = _get_constants()

    from chopinote_dataset.lmdb_store import LMDBStore

    # Phase 1: 抽样
    print("=" * 60)
    print("[Phase 1] 抽样...")

    DONE_FILE = '/root/autodl-tmp/func_done.txt'
    done_ids = []
    with open(DONE_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                done_ids.append(line)
    print(f"  已完成 func: {len(done_ids)}")

    random.seed(args.seed)
    sampled = random.sample(done_ids, args.num_samples)
    print(f"  抽样 {len(sampled)} 首")

    # Phase 2: 分析
    print("\n" + "=" * 60)
    print("[Phase 2] 逐首分析 Fig + Func...\n")

    store = LMDBStore.open(args.lmdb_path, readonly=True, map_size=350 * 1024**3)
    fig_results = []
    func_results = []

    for i, fid in enumerate(sampled):
        try:
            raw_tokens = store.get_raw(fid, 'tokens')
            tokens = list(struct.unpack(f'<{len(raw_tokens)//4}I', raw_tokens))
            fig_data = store.get(fid, 'fig')
            ssf_data = store.get_ssf(fid)
            func_data = store.get_func(fid)
        except Exception as e:
            print(f"  [{i+1}/{args.num_samples}] {fid[:50]}... READ ERROR: {e}")
            continue

        num_bars = sum(1 for t in tokens if t == C['bar_id'])
        num_notes = sum(1 for t in tokens if C['note_min'] <= t <= C['note_max'])

        # Fig audit
        fig_r = audit_fig(fid, tokens, fig_data, C)
        fig_results.append(fig_r)

        # Func audit
        func_r = audit_func(fid, ssf_data, func_data)
        func_results.append(func_r)

        # Print summary
        fig_ok = "✅" if fig_r.get('match_rate', 0) >= 0.95 else ("⚠️" if fig_r.get('match_rate', 0) > 0 else "❌")
        func_ok = "✅" if func_r.get('bar_match_rate', 0) >= 0.95 else ("⚠️" if func_r.get('bar_match_rate', 0) > 0 else "❌")

        print(f"  [{i+1}/{args.num_samples}] {fid[:50]}...")
        print(f"      tokens={len(tokens)}, bars={num_bars}, notes={num_notes}")
        print(f"      Fig  {fig_ok} match={fig_r.get('match_rate',0):.1%}, "
              f"matches={fig_r.get('matches',0)}, mismatches={fig_r.get('mismatches',0)}")
        print(f"      Func {func_ok} bar={func_r.get('bar_match_rate',0):.1%}, "
              f"beat={func_r.get('beat_match_rate',0):.1%}, "
              f"section={func_r.get('section_match',0):.1%}")
        if func_r.get('bar_func_distribution'):
            print(f"      Func dist: {func_r['bar_func_distribution']}")
        if fig_r.get('issues'):
            for iss in fig_r['issues'][:2]:
                print(f"      ❌ Fig: {iss}")
        if func_r.get('issues'):
            for iss in func_r['issues'][:2]:
                print(f"      ❌ Func: {iss}")

    store.close()

    # Phase 3: Summary
    print("\n" + "=" * 60)
    print("[Phase 3] 汇总报告")
    print("=" * 60)

    # Fig
    fig_rates = [r.get('match_rate', 0) for r in fig_results]
    fig_mismatches = [r.get('mismatches', 0) for r in fig_results]
    print(f"\n📊 Fig 织体标注:")
    print(f"   平均匹配率: {sum(fig_rates)/len(fig_rates):.1%}")
    print(f"   完全一致 (100%): {sum(1 for r in fig_rates if r >= 0.99)}/{len(fig_rates)}")
    print(f"   高一致 (≥95%):  {sum(1 for r in fig_rates if r >= 0.95)}/{len(fig_rates)}")
    print(f"   总 mismatch:    {sum(fig_mismatches)}")
    print(f"   典型错误: ")
    for r in fig_results:
        for d in r.get('details', [])[:2]:
            print(f"      bar{d['bar']} V{d['voice']}: stored={d['stored']} recomputed={d['recomputed']}")

    # Func
    func_bar_rates = [r.get('bar_match_rate', 0) for r in func_results]
    func_beat_rates = [r.get('beat_match_rate', 0) for r in func_results]
    all_none_rates = [r.get('all_none_rate', 0) for r in func_results]

    print(f"\n📊 Func 功能和声标注:")
    print(f"   平均 bar 匹配率:  {sum(func_bar_rates)/len(func_bar_rates):.1%}")
    print(f"   完全一致 (100%):   {sum(1 for r in func_bar_rates if r >= 0.99)}/{len(func_results)}")
    print(f"   高一致 (≥95%):    {sum(1 for r in func_bar_rates if r >= 0.95)}/{len(func_results)}")
    print(f"   平均 beat 匹配率: {sum(func_beat_rates)/len(func_beat_rates):.1%}")
    print(f"   平均 'none' 比例: {sum(all_none_rates)/len(all_none_rates):.1%}")

    # 打印第一首详细信息
    if func_results:
        print(f"\n{'='*60}")
        print("[样本详情] 第一首 Func 详细信息")
        print("=" * 60)
        r = func_results[0]
        print(f"  bar_match: {r['bar_match_rate']}, beat_match: {r['beat_match_rate']}")
        print(f"  func distribution: {r['bar_func_distribution']}")
        print(f"  top transitions: {r['top_transitions']}")

    print("\n✅ 审计完成")


if __name__ == '__main__':
    main()
