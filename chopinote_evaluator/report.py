"""报告生成 — JSON 和可读文本格式。"""

from __future__ import annotations

import json
from typing import Any


def report_to_dict(report: Any) -> dict:
    """将 EvaluationReport 转为纯 dict。"""
    result = {
        'total_score': round(report.total_score, 4),
        'alpha': report.alpha,
        'legality': {
            'passed': report.legality.passed,
            'issues': [
                {
                    'rule': i.rule,
                    'measure': i.measure,
                    'severity': i.severity,
                    'message': i.message,
                }
                for i in report.legality.issues
            ],
        },
        'general': report.general,
        'specific': report.specific,
    }

    if report.details:
        result['details'] = report.details

    return result


def report_to_json(report: Any, indent: int = 2) -> str:
    """生成 JSON 报告字符串。"""
    return json.dumps(report_to_dict(report), indent=indent, ensure_ascii=False)


def report_to_text(report: Any, color: bool = True) -> str:
    """生成可读的文本报告（支持 ANSI 颜色）。"""
    lines = []

    # 颜色相关
    if color:
        GREEN = '\033[92m'
        RED = '\033[91m'
        YELLOW = '\033[93m'
        CYAN = '\033[96m'
        BOLD = '\033[1m'
        DIM = '\033[2m'
        RESET = '\033[0m'
        CHECK = '✅'
        CROSS = '❌'
        WARN = '⚠️'
    else:
        GREEN = RED = YELLOW = CYAN = BOLD = DIM = RESET = ''
        CHECK = '✓'
        CROSS = '✗'
        WARN = '!'

    # 总分
    score = report.total_score
    score_color = GREEN if score >= 0.7 else (YELLOW if score >= 0.4 else RED)
    lines.append(f'  {BOLD}总音乐性: {score_color}{score:.4f} / 1.00  (α={report.alpha}){RESET}')
    lines.append('')

    # 合法性
    legal = report.legality
    if legal.passed:
        lines.append(f'  合法性     {GREEN}{CHECK} 通过{RESET}')
    else:
        lines.append(f'  合法性     {RED}{CROSS} 不通过 (有 error 级别问题){RESET}')

    if legal.issues:
        for issue in legal.issues:
            sev_icon = RED + CROSS if issue.severity == 'error' else YELLOW + WARN
            lines.append(f'    {sev_icon}{RESET} m.{issue.measure}: {issue.message}')
    lines.append('')

    # 广义评价
    general = report.general
    general_score = _get_nested(general, 'score', 0.0)
    gs_color = GREEN if general_score >= 0.7 else (YELLOW if general_score >= 0.4 else RED)
    lines.append(f'  {BOLD}广义评价  {gs_color}{general_score:.4f}{RESET}')

    # 统计分布
    stat_scores = _get_nested(general, 'statistical', {})
    if stat_scores:
        # 计算统计平均分
        if isinstance(stat_scores, dict) and stat_scores:
            stat_avg = sum(v for v in stat_scores.values() if isinstance(v, (int, float))) / \
                       max(sum(1 for v in stat_scores.values() if isinstance(v, (int, float))), 1)
            lines.append(f'  ├─ 统计分布    {_colorize_score(stat_avg, color)}')
            for k, v in sorted(stat_scores.items()):
                if isinstance(v, (int, float)):
                    label = _metric_label(k)
                    lines.append(f'  │  ├─ {label}  {_colorize_score(v, color, short=True)}')

    # 理论规则
    theory = _get_nested(general, 'theory', {})
    theory_score = theory.get('score', 0.0) if isinstance(theory, dict) else 0.0
    lines.append(f'  ├─ 理论规则    {_colorize_score(theory_score, color)}')
    if isinstance(theory, dict):
        for v in theory.get('violations', []):
            if isinstance(v, dict):
                sev = v.get('severity', '')
                meas = v.get('measure', '?')
                desc = v.get('description', '')
            else:
                sev = getattr(v, 'severity', '')
                meas = getattr(v, 'measure', '?')
                desc = getattr(v, 'description', '')
            sev_icon = YELLOW + WARN if sev == 'warning' else YELLOW + '!'
            lines.append(f'  │  └─ {sev_icon}{RESET} m.{meas} {desc}')

    # 狭义评价
    specific = report.specific
    if specific:
        specific_score = _get_nested(specific, 'score', 0.0)
        sc_color = GREEN if specific_score >= 0.7 else (YELLOW if specific_score >= 0.4 else RED)
        lines.append(f'')
        lines.append(f'  {BOLD}狭义评价  {sc_color}{specific_score:.4f}{RESET}')

        # 风格一致性
        cons = _get_nested(specific, 'consistency', {})
        if cons:
            cons_avg = sum(v for v in cons.values() if isinstance(v, (int, float))) / \
                       max(sum(1 for v in cons.values() if isinstance(v, (int, float))), 1)
            lines.append(f'  ├─ 风格一致    {_colorize_score(cons_avg, color)}')
            for k, v in sorted(cons.items()):
                if isinstance(v, (int, float)):
                    lines.append(f'  │  ├─ {_metric_label(k)}  {_colorize_score(v, color, short=True)}')

        # 结构衔接
        coh = _get_nested(specific, 'coherence', {})
        if coh:
            coh_avg = sum(v for v in coh.values() if isinstance(v, (int, float))) / \
                      max(sum(1 for v in coh.values() if isinstance(v, (int, float))), 1)
            lines.append(f'  ├─ 结构衔接    {_colorize_score(coh_avg, color)}')
            for k, v in sorted(coh.items()):
                if isinstance(v, (int, float)):
                    lines.append(f'  │  ├─ {_metric_label(k)}  {_colorize_score(v, color, short=True)}')

        # 模型自洽
        model_info = _get_nested(specific, 'model', {})
        if model_info:
            ppl = model_info.get('perplexity', 0)
            spike = model_info.get('boundary_spike', False)
            model_score = 1.0 - min(ppl / 20, 1.0)
            if spike:
                model_score *= 0.7
            lines.append(f'  └─ 模型自洽    {_colorize_score(model_score, color)}')
            lines.append(f'     ├─ Perplexity: {ppl:.2f}')
            lines.append(f'     └─ 边界突增:   {"有" if spike else "无"}')

    return '\n'.join(lines)


# ── 颜色辅助 ────────────────────────────────────────────────


def _colorize_score(score: float, color: bool = True, short: bool = False) -> str:
    """根据分数返回带颜色的字符串。"""
    if not color:
        return f'{score:.4f}'

    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'

    text = f'{score:.4f}' if not short else f'{score:.2f}'
    label = ' (优)' if not short and score >= 0.85 else \
            ' (良)' if not short and score >= 0.6 else \
            ' (差)' if not short else ''

    if score >= 0.7:
        return f'{GREEN}{text}{label}{RESET}'
    elif score >= 0.4:
        return f'{YELLOW}{text}{label}{RESET}'
    else:
        return f'{RED}{text}{label}{RESET}'


def _metric_label(key: str) -> str:
    """指标名 → 可读标签。"""
    labels = {
        'pitch_class_kl': '音高分布 KL',
        'interval_dist_kl': '音程分布 KL',
        'density_z': '音符密度',
        'key_consistency': '调性稳定',
        'self_similarity': '自相似性',
        'token_type_kl': 'Token 类型',
        'duration_entropy': '时值熵',
        'pitch_entropy': '音高熵',
        'dissonance_ratio': '不协和比例',
        'chromaticism_index': '半音化程度',
        'register_span': '音域跨度',
        'syncopation_ratio': '切分比例',
        'dynamic_variance': '力度变化',
        'rest_ratio': '休止比例',
        'melodic_direction': '旋律方向',
        'contour_arc': '拱形结构',
        'polyphony_mean': '织体厚度',
        'texture_variance': '织体变化',
        'harmonic_rhythm': '和声节奏',
        'density_delta': '密度差',
        'interval_shift': '音程偏移',
        'velocity_delta': '力度差',
        'articulation_delta': '演奏法差',
        'key_match': '调性匹配',
        'voice_count_delta': '声部数差',
        'tempo_continuity': '速度连续',
        'cadence_match': '终止式匹配',
    }
    return labels.get(key, key)


def _get_nested(d: dict, key: str, default: Any = None) -> Any:
    """从嵌套 dict 安全取值。"""
    if isinstance(d, dict):
        return d.get(key, default)
    return default
