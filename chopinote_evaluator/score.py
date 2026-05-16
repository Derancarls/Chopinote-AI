"""综合评价入口 — Evaluator 类 + EvaluationReport。

整合合法性检查、广义评价（统计+理论）、狭义评价（风格+结构+模型），
输出统一音乐性打分。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from chopinote_evaluator.parser import Score, parse_musicxml
from chopinote_evaluator.report import report_to_text, report_to_json
from chopinote_evaluator.general.legality import LegalityResult, check_legality
from chopinote_evaluator.general.statistics import StatisticalEvaluator
from chopinote_evaluator.general.theory import TheoryEvaluator
from chopinote_evaluator.benchmarks.build_benchmarks import load_benchmarks


@dataclass
class EvaluationReport:
    """综合评价报告。"""
    total_score: float
    alpha: float
    legality: LegalityResult
    general: dict
    specific: dict | None
    details: dict = field(default_factory=dict)

    def to_text(self, color: bool = True) -> str:
        return report_to_text(self, color=color)

    def to_json(self) -> str:
        return report_to_json(self)

    def to_dict(self) -> dict:
        from chopinote_evaluator.report import report_to_dict
        return report_to_dict(self)


class Evaluator:
    """综合评价入口。

    用法:
        # 基本使用：仅广义评价
        eval = Evaluator(benchmarks_path='...')
        report = eval.evaluate('input.musicxml')

        # 续写评价（需要种子）
        report = eval.evaluate('gen.musicxml', seed_path='seed.musicxml')

        # 带模型 scorer
        report = eval.evaluate('gen.musicxml', checkpoint='step_2000.pt')
    """

    # 场景权重 α
    ALPHA_GENERATION = 0.3    # 生成后评估：更看重续写衔接
    ALPHA_CORPUS = 0.7        # 语料筛选：更看重曲谱本身
    ALPHA_VALIDATION = 0.5    # 训练验证：均衡
    ALPHA_PURE = 1.0          # 纯广义评价：无续写场景

    def __init__(self,
                 benchmarks: dict | None = None,
                 benchmarks_path: str | None = None,
                 alpha: float = 0.3,
                 group: str = 'all'):
        """
        参数:
            benchmarks: 预加载的基准数据 dict
            benchmarks_path: 基准数据路径（与 benchmarks 二选一）
            alpha: 场景权重（0~1，越高越看重广义评价）
            group: 对比锚点组名
        """
        if benchmarks is None:
            benchmarks = load_benchmarks(benchmarks_path)
        self.benchmarks = benchmarks
        self.alpha = alpha
        self.group = group

        self.stat = StatisticalEvaluator(benchmarks, group)
        self.theory = TheoryEvaluator()

    def evaluate(self,
                 score_path: str | Path | None = None,
                 score: Score | None = None,
                 seed_path: str | Path | None = None,
                 seed: Score | None = None,
                 checkpoint: str | None = None) -> EvaluationReport:
        """主评价入口。

        参数:
            score_path: 待评价乐谱文件路径
            score: 已解析的 Score 对象（与 score_path 二选一）
            seed_path: 种子乐谱文件路径（续写场景）
            seed: 已解析的种子 Score 对象
            checkpoint: 模型 checkpoint 路径（启用 model scorer）

        返回:
            EvaluationReport
        """
        # 1. 解析
        if score_path:
            score = parse_musicxml(str(score_path))
        if seed_path:
            seed = parse_musicxml(str(seed_path))
        if score is None:
            raise ValueError('必须提供 score_path 或 score')

        # 2. 合法性门禁
        legality = check_legality(score)
        if legality.has_error:
            return EvaluationReport(
                total_score=0.0,
                alpha=self.alpha,
                legality=legality,
                general={'error': '合法性检查未通过'},
                specific=None,
            )

        # 3. 广义评价
        stat_scores = self.stat.evaluate_global(score)
        stat_total = self.stat.aggregate_score(stat_scores)
        theory_result = self.theory.evaluate(score)

        general = {
            'score': stat_total * 0.6 + theory_result['score'] * 0.4,
            'statistical': stat_scores,
            'statistical_score': stat_total,
            'theory': theory_result,
        }

        # 4. 狭义评价（有种子时才做）
        specific = None
        specific_score = 0.0

        if seed is not None:
            from chopinote_evaluator.specific.consistency import ConsistencyEvaluator
            from chopinote_evaluator.specific.coherence import CoherenceEvaluator

            cons_eval = ConsistencyEvaluator()
            coh_eval = CoherenceEvaluator()

            cons_scores = cons_eval.evaluate(seed, score)
            coh_scores = coh_eval.evaluate(seed, score)

            cons_total = cons_eval.aggregate(cons_scores)
            coh_total = coh_eval.aggregate(coh_scores)

            specific = {
                'score': cons_total * 0.5 + coh_total * 0.5,
                'consistency': cons_scores,
                'consistency_score': cons_total,
                'coherence': coh_scores,
                'coherence_score': coh_total,
            }

            # 模型 scorer（可选）
            if checkpoint:
                try:
                    from chopinote_evaluator.specific.model_scorer import ModelScorer

                    scorer = ModelScorer(checkpoint)
                    seed_tokens = scorer.score_to_tokens(seed)
                    gen_tokens = scorer.score_to_tokens(score)

                    ppl = scorer.perplexity(gen_tokens)
                    boundary = scorer.boundary_test(seed_tokens, gen_tokens)

                    model_score = 1.0 - min(ppl.overall_ppl / 20, 1.0)
                    if boundary.spike_detected:
                        model_score *= 0.7

                    specific['model'] = {
                        'perplexity': ppl.overall_ppl,
                        'per_type_ppl': ppl.per_type,
                        'boundary_spike': boundary.spike_detected,
                        'spike_magnitude': boundary.spike_magnitude,
                        'score': model_score,
                    }
                    specific['score'] = specific['score'] * 0.7 + model_score * 0.3
                except Exception as e:
                    specific['model'] = {'error': str(e)}

            specific_score = specific.get('score', 0.0)

        # 5. 综合打分
        general_score = general.get('score', 0.0)
        if specific is None:
            # 无种子时不混合狭义分
            total_score = general_score
        else:
            total_score = general_score * self.alpha + specific_score * (1 - self.alpha)

        return EvaluationReport(
            total_score=total_score,
            alpha=self.alpha,
            legality=legality,
            general=general,
            specific=specific,
        )
