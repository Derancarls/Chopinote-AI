"""模型自洽性打分 — Perplexity 计算 + 跨边界突增检测。

加载已训练的 checkpoint，计算生成结果的 perplexity 以及
种子/生成边界的 loss 异常检测。
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.cuda.amp import autocast

from chopinote_model.model import MusicTransformer
from chopinote_model.config import ModelConfig
from chopinote_dataset.tokenizer import REMITokenizer
from chopinote_evaluator.parser import Score


@dataclass
class PerplexityReport:
    """Perplexity 计算结果。"""
    overall_ppl: float = 0.0
    avg_loss: float = 0.0
    per_type: dict[str, float] = field(default_factory=dict)
    token_losses: list[float] = field(default_factory=list)
    n_tokens: int = 0


@dataclass
class BoundaryReport:
    """边界检测结果。"""
    boundary_loss: float = 0.0
    side_loss: float = 0.0
    spike_detected: bool = False
    spike_magnitude: float = 0.0
    n_boundary_tokens: int = 0
    n_side_tokens: int = 0


class ModelScorer:
    """模型自洽性评分器。

    用法:
        scorer = ModelScorer('checkpoints/step_2000.pt')
        ppl = scorer.perplexity(token_ids)
        boundary = scorer.boundary_test(seed_tokens, gen_tokens)
    """

    def __init__(self, checkpoint_path: str,
                 device: str | None = None,
                 tokenizer: REMITokenizer | None = None):
        """
        参数:
            checkpoint_path: 模型 checkpoint 路径
            device: 设备 ('cuda', 'cpu', None=自动选择)
            tokenizer: 分词器，None 则新建默认
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = tokenizer or REMITokenizer(grid_size=16, velocity_levels=8)

        print(f'  加载模型: {checkpoint_path}')
        self.model, self.config, self.ckpt_step, self.ckpt_loss = \
            self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path: str):
        """加载 checkpoint 并返回模型。"""
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f'checkpoint 不存在: {checkpoint_path}')

        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        saved_config = ckpt.get('config')
        if isinstance(saved_config, dict):
            config = ModelConfig(**saved_config)
        elif saved_config is not None:
            config = saved_config
        else:
            config = ModelConfig()

        model = MusicTransformer(config)
        state_dict = ckpt['model_state_dict']
        model_state = model.state_dict()

        for k, v in state_dict.items():
            if k in model_state and v.shape == model_state[k].shape:
                model_state[k] = v

        model.load_state_dict(model_state)
        model.to(self.device)
        model.eval()

        step = ckpt.get('step', 0)
        loss = ckpt.get('loss', None)
        return model, config, step, loss

    def score_to_tokens(self, score: Score) -> list[int]:
        """将 Score 转为 token 序列。

        通过 MusicXML 中间文件桥接 tokenizer 的转换能力。
        注意: 这个 token 序列不一定与训练时的 token 序列完全一致
        （因为跳过了 converter 的 event 构建）。
        更好的方式：直接用 converter 的 MusicXMLToREMI。

        返回:
            token ID 列表
        """
        # 从 Score → 临时 MusicXML → 重新解析
        import tempfile
        import music21

        # 重建 music21 Score（简化版）
        m21_score = music21.stream.Score()
        part = music21.stream.Part()
        for m in score.measures:
            m21_m = music21.stream.Measure(number=m.number)
            m21_m.timeSignature = music21.meter.TimeSignature(f'{m.time_signature[0]}/{m.time_signature[1]}')
            if m.key_signature and m.key_signature != 'unknown':
                try:
                    ks_name = m.key_signature.replace('_', ' ')
                    ks = music21.key.Key(ks_name)
                    m21_m.keySignature = ks
                except Exception:
                    pass
            for n in m.notes:
                if n.is_rest:
                    m21_n = music21.note.Rest(quarterLength=n.duration)
                elif n.pitch is not None:
                    m21_n = music21.note.Note(n.pitch, quarterLength=n.duration)
                    m21_n.volume.velocity = n.velocity
                else:
                    continue
                m21_m.append(m21_n)
            part.append(m21_m)
        m21_score.append(part)

        with tempfile.NamedTemporaryFile(suffix='.musicxml', delete=False) as f:
            tmp_path = f.name
        try:
            m21_score.write('musicxml', fp=tmp_path)
            from chopinote_dataset.converter import MusicXMLToREMI
            conv = MusicXMLToREMI(grid_size=16, velocity_levels=8)
            tokens, _ = conv.convert(tmp_path, collect_metadata=True)
            return tokens
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def perplexity(self, token_ids: list[int]) -> PerplexityReport:
        """计算 token 序列的 perplexity。

        参数:
            token_ids: token ID 序列

        返回:
            PerplexityReport
        """
        if len(token_ids) < 2:
            return PerplexityReport()

        losses = self._compute_token_losses(token_ids)

        # 过滤 nan/inf
        valid = [(loss, tid) for loss, tid in zip(losses, token_ids[1:])
                 if not math.isnan(loss) and not math.isinf(loss)]
        if not valid:
            return PerplexityReport()

        valid_losses = [v[0] for v in valid]
        valid_tokens = [v[1] for v in valid]

        avg_loss = sum(valid_losses) / len(valid_losses)
        ppl = math.exp(min(avg_loss, 20))  # cap at exp(20) ≈ 4.8e8

        # 按 token 类型分拆
        type_losses: dict[str, list[float]] = {}
        for loss, tid in valid:
            ttype = self._token_type_name(tid)
            type_losses.setdefault(ttype, []).append(loss)

        per_type = {}
        for ttype, tl in type_losses.items():
            if tl:
                per_type[ttype] = math.exp(sum(tl) / len(tl))

        return PerplexityReport(
            overall_ppl=ppl,
            avg_loss=avg_loss,
            per_type=per_type,
            token_losses=valid_losses,
            n_tokens=len(valid_losses),
        )

    def boundary_test(self, seed_tokens: list[int],
                      gen_tokens: list[int],
                      window: int = 32) -> BoundaryReport:
        """检测续写边界处 loss 突增。

        方法:
            1. 拼接 seed_tokens + gen_tokens
            2. 在边界前后各取 window 个 token 计算平均 loss
            3. 与两侧更远处的平均 loss 对比
            4. 如果边界 loss > 侧边 loss + 2σ，标记为突增

        参数:
            seed_tokens: 种子部分的 token 序列
            gen_tokens: 生成部分的 token 序列
            window: 边界两侧检测窗口大小

        返回:
            BoundaryReport
        """
        if not gen_tokens:
            return BoundaryReport()

        all_tokens = seed_tokens + gen_tokens
        boundary_pos = len(seed_tokens)

        if len(all_tokens) < window * 2 + 2:
            return BoundaryReport()

        losses = self._compute_token_losses(all_tokens)

        # 边界区域
        b_left = max(0, boundary_pos - window)
        b_right = min(len(losses), boundary_pos + window)
        boundary_losses = losses[b_left:b_right]

        # 两侧区域
        far_left = losses[max(0, boundary_pos - window * 2):max(0, boundary_pos - window)]
        far_right = losses[min(len(losses), boundary_pos + window):min(len(losses), boundary_pos + window * 2)]
        side_losses = far_left + far_right

        if not boundary_losses or not side_losses:
            return BoundaryReport()

        boundary_mean = sum(boundary_losses) / len(boundary_losses)
        side_mean = sum(side_losses) / len(side_losses)
        side_var = sum((l - side_mean) ** 2 for l in side_losses) / len(side_losses)
        side_std = math.sqrt(side_var)

        has_spike = boundary_mean > side_mean + 2 * max(side_std, 0.1)
        magnitude = (boundary_mean - side_mean) / max(side_std, 0.01)

        return BoundaryReport(
            boundary_loss=boundary_mean,
            side_loss=side_mean,
            spike_detected=has_spike,
            spike_magnitude=magnitude,
            n_boundary_tokens=len(boundary_losses),
            n_side_tokens=len(side_losses),
        )

    def _compute_token_losses(self, token_ids: list[int]) -> list[float]:
        """逐 token 计算 loss，返回长度为 len(token_ids) - 1 的列表。"""
        if len(token_ids) < 2:
            return []

        model = self.model
        device = self.device

        # 分块处理，避免 OOM
        chunk_size = 1024
        all_losses = []

        with torch.no_grad():
            for start in range(0, len(token_ids) - 1, chunk_size):
                end = min(start + chunk_size + 1, len(token_ids))
                chunk = token_ids[start:end]

                # 构造输入
                input_ids = torch.tensor([chunk], device=device)

                with autocast('cuda', enabled=(device == 'cuda'),
                              dtype=torch.bfloat16):
                    logits = model(input_ids)  # (1, T, vocab)

                log_probs = torch.log_softmax(logits[0], dim=-1)  # (T, vocab)

                for t in range(len(chunk) - 1):
                    target = chunk[t + 1]
                    nll = -log_probs[t, target].item()
                    all_losses.append(nll)

        return all_losses

    def _token_type_name(self, token_id: int) -> str:
        """获取 token ID 的类型名称。"""
        token_str = self.tokenizer.decode_token(token_id)

        type_map = {
            'Note_ON': 'note',
            'Note_OFF': 'note_off',
            'Velocity': 'velocity',
            'Duration': 'duration',
            'Position': 'position',
            'Beat': 'beat',
            'Bar': 'bar',
            'Tempo': 'tempo',
            'TimeSig': 'timesig',
            'Key': 'key',
            'Program': 'program',
            'Rest': 'rest',
            'Pedal': 'pedal',
            'Articulation': 'articulation',
            'Ornament': 'ornament',
            'Dynamic': 'dynamic',
            'Tuplet': 'tuplet',
        }
        for prefix, name in type_map.items():
            if token_str.startswith(prefix):
                return name
        return 'other'
