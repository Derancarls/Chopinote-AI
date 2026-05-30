"""ABC Engine 生成日志系统 — 每次大循环一个独立日志文件。

记录引擎五层之间的完整信息流:
  A1(框架记忆) → A2(动机提取) → A3(统计画像) → B(决策层) → C(评价层)

用法:
    logger = ABCGenerationLogger(
        log_dir='logs/generate',
        form='sonata', max_bars=48, seed_name='seed_piano_4bars',
    )
    logger.session_start()
    # ... 生成过程 ...
    logger.a1_structure(sections)
    logger.a2_motif(label, dna)
    logger.a3_bar(bar_idx, stats)
    logger.b_decision(bar_idx, temperature, bans, feedback)
    logger.c_evaluation(report)
    logger.session_end(summary)
"""

from __future__ import annotations

import json
import logging
import os
import queue
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO


# ═══════════════════════════════════════════════════════════════
#  格式化器 — 人类可读 + JSON 结构化双输出
# ═══════════════════════════════════════════════════════════════

_COLOR_MAP = {
    'DEBUG': '\033[36m',     # cyan
    'INFO': '\033[32m',      # green
    'WARNING': '\033[33m',   # yellow
    'ERROR': '\033[31m',     # red
    'RESET': '\033[0m',
}

_COMPONENT_COLORS = {
    'A1': '\033[34m',    # blue
    'A2': '\033[35m',    # magenta
    'A3': '\033[33m',    # yellow
    'B1': '\033[31m',    # red — 硬约束
    'B2': '\033[91m',    # bright red — 决策调参
    'C':  '\033[32m',    # green
    'SYS': '\033[36m',   # cyan
    'RESET': '\033[0m',
}


class ABCPlainFormatter(logging.Formatter):
    """文件用格式化器 — 纯文本，无 ANSI 转义码。"""

    def format(self, record: logging.LogRecord) -> str:
        level = record.levelname
        component = getattr(record, 'component', 'SYS')
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S.%f')[:-3]
        header = f'{timestamp} {level:<7} [{component}]'
        msg = record.getMessage()

        data = getattr(record, 'abc_data', None)
        if data is not None:
            data_str = json.dumps(data, ensure_ascii=False, default=str)
            if len(data_str) > 500:
                data_str = data_str[:500] + '…'
            return f'{header} {msg} | {data_str}'

        return f'{header} {msg}'


class ABCColorFormatter(logging.Formatter):
    """控制台用格式化器 — 带 ANSI 颜色码。"""

    def format(self, record: logging.LogRecord) -> str:
        level = record.levelname
        level_color = _COLOR_MAP.get(level, '')
        reset = _COLOR_MAP['RESET']

        component = getattr(record, 'component', 'SYS')
        comp_color = _COMPONENT_COLORS.get(component, '')
        comp_reset = _COMPONENT_COLORS['RESET']

        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S.%f')[:-3]
        header = f'{timestamp} {level_color}{level:<7}{reset} {comp_color}[{component}]{comp_reset}'
        msg = record.getMessage()

        data = getattr(record, 'abc_data', None)
        if data is not None:
            data_str = json.dumps(data, ensure_ascii=False, default=str)
            if len(data_str) > 500:
                data_str = data_str[:500] + '…'
            return f'{header} {msg} | {data_str}'

        return f'{header} {msg}'


class AsyncFileHandler(logging.Handler):
    """异步文件 Handler — 用独立线程写日志，不阻塞生成主线程。

    内部维护一个 queue.Queue，emit() 将 record 入队，
    daemon 线程从队列取 record 写入实际文件。
    """

    def __init__(self, filepath: str, encoding: str = 'utf-8',
                 fmt: logging.Formatter | None = None, level: int = logging.DEBUG):
        super().__init__(level=level)
        self._queue: queue.Queue[logging.LogRecord | None] = queue.Queue(maxsize=5000)
        self._real_handler = logging.FileHandler(filepath, encoding=encoding)
        if fmt:
            self._real_handler.setFormatter(fmt)
        self._thread = threading.Thread(target=self._write_loop, daemon=True,
                                        name=f'abc-log-{os.path.basename(filepath)}')
        self._running = True
        self._thread.start()

    def _write_loop(self):
        while self._running:
            try:
                record = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if record is None:
                break
            try:
                self._real_handler.emit(record)
            except Exception:
                pass

    def emit(self, record: logging.LogRecord):
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            pass  # 队列满时丢弃（避免阻塞生成）

    def close(self):
        self._running = False
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=5)
        self._real_handler.close()
        super().close()


# ═══════════════════════════════════════════════════════════════
#  主日志器
# ═══════════════════════════════════════════════════════════════

@dataclass
class GenerationSummary:
    """单次生成会话的汇总信息。"""
    session_id: str
    started_at: str
    ended_at: str = ''
    form: str = 'free'
    max_bars: int = 64
    seed_name: str = ''
    seed_bars: int = 0
    seed_tokens: int = 0
    # 结果
    total_bars: int = 0
    total_tokens: int = 0
    generated_bars: int = 0
    generated_tokens: int = 0
    elapsed_seconds: float = 0.0
    # A1
    sections_planned: int = 0
    harmony_chords: int = 0
    # A2
    motifs_extracted: int = 0
    # A3
    bars_logged: int = 0
    # B1 — 硬约束
    b1_hard_bans_applied: int = 0          # 禁令触发次数
    b1_context_ban_count: int = 0          # 上下文禁令 token 数
    b1_total_bans: int = 0                 # 累计被禁 token 数
    # B2 — 决策调参
    b2_fatal_signals: int = 0              # 致命信号次数
    b2_innovations_admitted: int = 0       # 创新承认次数
    b2_temperature_adjustments: int = 0    # 温度调节次数
    b2_reharmonize_count: int = 0          # 和声回退次数
    # C
    total_score: float = 0.0
    novelty_score: float = 0.0
    diversity_score: float = 0.0
    structural_fixes: int = 0
    xml_review_bars: int = 0
    xml_fidelity: float = 1.0
    # 错误
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class ABCGenerationLogger:
    """ABC Engine 生成会话日志器。

    每次 stage3_iterative_generate() 调用创建一个实例，
    产生一个独立的日志文件。

    特性:
      - 双输出: 日志文件 (完整) + 控制台 (WARNING+)
      - 结构化数据: 每个 log 方法接受 data dict，记录组件间传输的关键值
      - 会话汇总: session_end() 输出 JSON 格式汇总
    """

    def __init__(
        self,
        log_dir: str = 'logs/generate',
        form: str = 'free',
        max_bars: int = 64,
        seed_name: str = '',
        console_level: int = logging.WARNING,
        file_level: int = logging.DEBUG,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.form = form
        self.max_bars = max_bars
        self.seed_name = seed_name
        self.console_level = console_level
        self.file_level = file_level

        # 会话 ID: 时间戳 + 曲式
        self.session_id = (
            f'{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            f'_{form}_{max_bars}bar'
        )
        self.log_path = self.log_dir / f'{self.session_id}.log'

        # Python logger
        self._logger = logging.getLogger(f'abc_gen.{self.session_id}')
        self._logger.setLevel(logging.DEBUG)
        self._logger.handlers.clear()
        self._logger.propagate = False

        # 文件 handler — 异步写，纯文本（无 ESC 码）
        self._file_handler = AsyncFileHandler(
            str(self.log_path), encoding='utf-8',
            fmt=ABCPlainFormatter(), level=file_level,
        )
        self._logger.addHandler(self._file_handler)

        # 控制台 handler — 同步，带颜色
        self._console_handler = logging.StreamHandler(sys.stderr)
        self._console_handler.setLevel(console_level)
        self._console_handler.setFormatter(ABCColorFormatter())
        self._logger.addHandler(self._console_handler)

        # 会话状态
        self.summary = GenerationSummary(
            session_id=self.session_id,
            started_at=datetime.now().isoformat(),
            form=form,
            max_bars=max_bars,
            seed_name=seed_name,
        )
        self._t0 = time.time()
        self._section_idx = -1
        self._bar_idx = -1

    # ── 日志方法 ──────────────────────────────────────────

    def _log(self, level: int, component: str, msg: str, data: dict | None = None):
        """底层日志调用。"""
        record = self._logger.makeRecord(
            self._logger.name, level, '', 0, msg, (), None)
        record.component = component
        record.abc_data = data
        self._logger.handle(record)

    def debug(self, component: str, msg: str, data: dict | None = None):
        self._log(logging.DEBUG, component, msg, data)

    def info(self, component: str, msg: str, data: dict | None = None):
        self._log(logging.INFO, component, msg, data)

    def warning(self, component: str, msg: str, data: dict | None = None):
        self._log(logging.WARNING, component, msg, data)
        self.summary.warnings.append(f'[{component}] {msg}')

    def error(self, component: str, msg: str, data: dict | None = None):
        self._log(logging.ERROR, component, msg, data)
        self.summary.errors.append(f'[{component}] {msg}')

    # ── 会话生命周期 ─────────────────────────────────────

    def session_start(self, seed_tokens_count: int = 0, seed_bars: int = 0,
                      checkpoint_step: int = 0, checkpoint_loss: float = 0.0):
        """生成会话开始。"""
        self.summary.seed_tokens = seed_tokens_count
        self.summary.seed_bars = seed_bars
        self._t0 = time.time()

        self.info('SYS', '═' * 60)
        self.info('SYS', f'ABC Engine 生成会话开始: {self.session_id}')
        self.info('SYS', f'曲式={self.form}  目标={self.max_bars}bar  '
                         f'Seed={self.seed_name}({seed_bars}bar/{seed_tokens_count}tok)')
        if checkpoint_step > 0:
            self.info('SYS', f'Checkpoint: step_{checkpoint_step}  loss={checkpoint_loss:.4f}')
        self.info('SYS', f'日志文件: {self.log_path}')
        self.info('SYS', '─' * 60)
        self._log_file_header()

    def session_end(self, all_tokens_count: int = 0, total_bars: int = 0):
        """生成会话结束，输出汇总。"""
        elapsed = time.time() - self._t0
        self.summary.ended_at = datetime.now().isoformat()
        self.summary.elapsed_seconds = elapsed
        self.summary.total_tokens = all_tokens_count
        self.summary.total_bars = total_bars
        self.summary.generated_bars = total_bars - self.summary.seed_bars
        self.summary.generated_tokens = all_tokens_count - self.summary.seed_tokens

        self.info('SYS', '─' * 60)
        self.info('SYS', '生成会话汇总')
        self.info('SYS', f'耗时: {elapsed:.1f}s  '
                         f'生成: {self.summary.generated_bars}bar/{self.summary.generated_tokens}tok  '
                         f'速度: {self.summary.generated_tokens/max(elapsed,0.1):.1f} tok/s')
        self.info('SYS', f'A1: {self.summary.sections_planned}段  '
                         f'{self.summary.harmony_chords}和弦')
        self.info('SYS', f'A2: {self.summary.motifs_extracted}动机')
        self.info('SYS', f'A3: {self.summary.bars_logged}bar统计')
        self.info('SYS', f'B1: {self.summary.b1_hard_bans_applied}次禁令({self.summary.b1_total_bans}tok)  '
                         f'上下文禁{self.summary.b1_context_ban_count}tok')
        self.info('SYS', f'B2: {self.summary.b2_fatal_signals}次致命信号  '
                         f'{self.summary.b2_innovations_admitted}次创新  '
                         f'{self.summary.b2_temperature_adjustments}次调温  '
                         f'回退{self.summary.b2_reharmonize_count}次')
        self.info('SYS', f'C: 总分={self.summary.total_score:.4f}  '
                         f'新颖={self.summary.novelty_score:.4f}  '
                         f'多样={self.summary.diversity_score:.4f}  '
                         f'修复={self.summary.structural_fixes}  '
                         f'XML保真={self.summary.xml_fidelity:.3f}')

        if self.summary.errors:
            self.warning('SYS', f'错误: {len(self.summary.errors)}条')
            for e in self.summary.errors:
                self.warning('SYS', f'  ✗ {e}')
        if self.summary.warnings:
            self.info('SYS', f'警告: {len(self.summary.warnings)}条')

        self.info('SYS', f'日志文件: {self.log_path}')
        self.info('SYS', '═' * 60)

        # 写入 JSON 汇总文件
        self._write_summary_json()
        # 关闭异步文件 handler，确保所有日志写入完毕
        self.close()

    def close(self):
        """关闭日志器，等待异步写入完成。"""
        self._file_handler.close()

    # ── Token 级详细日志 ─────────────────────────────────────

    def log_token_sample(self, bar_idx: int, token_idx: int, tid: int,
                         token_str: str, token_type: str,
                         temperature: float, top_k_threshold: float,
                         prob: float, logit_min: float, logit_max: float):
        """记录单个 token 的采样详情（仅 DEBUG 级别，仅文件）。"""
        if self.file_level > logging.DEBUG:
            return
        self._log(logging.DEBUG, 'TOK',
            f'Bar {bar_idx} tok[{token_idx}] '
            f'id={tid} type={token_type} '
            f'p={prob:.4f} T={temperature:.2f} topK_thr={top_k_threshold:.2f} '
            f'logit_range=[{logit_min:.2f}, {logit_max:.2f}]',
            {'bar': bar_idx, 'idx': token_idx, 'token_id': tid,
             'token': token_str, 'type': token_type,
             'prob': round(prob, 6), 'temperature': temperature,
             'top_k_threshold': round(top_k_threshold, 2),
             'logit_min': round(logit_min, 2), 'logit_max': round(logit_max, 2)})

    def log_forward_pass(self, bar_idx: int, ms_elapsed: float,
                         kv_cache_len: int, seq_len: int):
        """记录模型前向传播耗时（DEBUG 级别，仅文件）。"""
        if self.file_level > logging.DEBUG:
            return
        self._log(logging.DEBUG, 'FWD',
            f'Bar {bar_idx} forward: {ms_elapsed:.1f}ms '
            f'kv_len={kv_cache_len} seq={seq_len}',
            {'bar': bar_idx, 'ms': round(ms_elapsed, 1),
             'kv_cache_len': kv_cache_len, 'seq_len': seq_len})

    def log_bar_tokens(self, bar_idx: int, bar_tokens: list[int],
                       tokenizer=None):
        """记录整个 bar 的 token 序列（DEBUG 级别，仅文件）。"""
        if self.file_level > logging.DEBUG:
            return
        if tokenizer is not None:
            decoded = [tokenizer.decode_token(t) for t in bar_tokens]
        else:
            decoded = [str(t) for t in bar_tokens]
        tok_summary = ', '.join(decoded[:32])
        if len(decoded) > 32:
            tok_summary += f' … (+{len(decoded)-32})'
        self._log(logging.DEBUG, 'TOK',
            f'Bar {bar_idx}: {len(bar_tokens)} tokens [{tok_summary}]',
            {'bar': bar_idx, 'count': len(bar_tokens),
             'token_ids': bar_tokens[:64], 'decoded': decoded[:64]})

        if self.summary.errors:
            self.warning('SYS', f'错误: {len(self.summary.errors)}条')
            for e in self.summary.errors:
                self.warning('SYS', f'  ✗ {e}')
        if self.summary.warnings:
            self.info('SYS', f'警告: {len(self.summary.warnings)}条')

        self.info('SYS', f'日志文件: {self.log_path}')
        self.info('SYS', '═' * 60)

        # 写入 JSON 汇总文件
        self._write_summary_json()

    def _log_file_header(self):
        """写入日志文件头（元信息）。"""
        header = {
            'session_id': self.session_id,
            'started_at': self.summary.started_at,
            'form': self.form,
            'max_bars': self.max_bars,
            'seed_name': self.seed_name,
            'seed_bars': self.summary.seed_bars,
            'seed_tokens': self.summary.seed_tokens,
        }
        self.debug('SYS', '会话元信息', header)

    def _write_summary_json(self):
        """写 JSON 格式汇总文件。"""
        summary_path = self.log_path.with_suffix('.summary.json')
        try:
            data = {
                'session_id': self.summary.session_id,
                'started_at': self.summary.started_at,
                'ended_at': self.summary.ended_at,
                'elapsed_seconds': round(self.summary.elapsed_seconds, 1),
                'form': self.summary.form,
                'max_bars': self.summary.max_bars,
                'seed': {
                    'name': self.summary.seed_name,
                    'bars': self.summary.seed_bars,
                    'tokens': self.summary.seed_tokens,
                },
                'results': {
                    'total_bars': self.summary.total_bars,
                    'total_tokens': self.summary.total_tokens,
                    'generated_bars': self.summary.generated_bars,
                    'generated_tokens': self.summary.generated_tokens,
                    'tokens_per_second': round(
                        self.summary.generated_tokens / max(self.summary.elapsed_seconds, 0.1), 1),
                },
                'components': {
                    'A1': {
                        'sections_planned': self.summary.sections_planned,
                        'harmony_chords': self.summary.harmony_chords,
                    },
                    'A2': {
                        'motifs_extracted': self.summary.motifs_extracted,
                    },
                    'A3': {
                        'bars_logged': self.summary.bars_logged,
                    },
                    'B1': {
                        'hard_bans_applied': self.summary.b1_hard_bans_applied,
                        'context_ban_count': self.summary.b1_context_ban_count,
                        'total_bans': self.summary.b1_total_bans,
                    },
                    'B2': {
                        'fatal_signals': self.summary.b2_fatal_signals,
                        'innovations_admitted': self.summary.b2_innovations_admitted,
                        'temperature_adjustments': self.summary.b2_temperature_adjustments,
                        'reharmonize_count': self.summary.b2_reharmonize_count,
                    },
                    'C': {
                        'total_score': round(self.summary.total_score, 4),
                        'novelty_score': round(self.summary.novelty_score, 4),
                        'diversity_score': round(self.summary.diversity_score, 4),
                        'structural_fixes': self.summary.structural_fixes,
                        'xml_fidelity': round(self.summary.xml_fidelity, 3),
                    },
                },
                'errors': self.summary.errors,
                'warnings': self.summary.warnings,
            }
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def set_section(self, section_idx: int, section_type: str):
        """标记当前段落。"""
        self._section_idx = section_idx
        self._bar_idx = -1
        self.debug('SYS', f'进入段落 {section_idx}: {section_type}',
                   {'section_idx': section_idx, 'type': section_type})

    def set_bar(self, bar_idx: int, global_bar: int):
        """标记当前 bar。"""
        self._bar_idx = bar_idx
        self.debug('SYS', f'Bar {bar_idx} (全局 {global_bar})',
                   {'bar_idx': bar_idx, 'global_bar': global_bar})

    # ── A1 框架记忆 ──────────────────────────────────────

    def a1_structure(self, sections: list):
        """A1 段落规划结果。"""
        self.summary.sections_planned = len(sections)
        data = []
        for i, sec in enumerate(sections):
            sec_data = {
                'idx': i, 'type': sec.type, 'bars': sec.bars,
                'key': sec.key, 'cadence': sec.cadence,
                'innovation_budget': getattr(sec, 'innovation_budget', 0),
                'development_ops': getattr(sec, 'development_ops', None),
            }
            data.append(sec_data)
        self.info('A1', f'段落规划: {len(sections)}段', {'sections': data})

    def a1_harmony(self, chords: list, total_chords: int):
        """A1 和声规划结果。"""
        self.summary.harmony_chords = total_chords
        # 摘要: 每段的和声进行
        by_section: dict[int, list[str]] = {}
        for c in chords:
            sid = getattr(c, 'section_idx', 0)
            if sid not in by_section:
                by_section[sid] = []
            by_section[sid].append(f'{c.func}({c.inv})')
        for sid, prog in by_section.items():
            self.debug('A1', f'和声进行 S{sid}: {"→".join(prog[:12])}',
                       {'section': sid, 'progression': prog})

        self.info('A1', f'和声规划: {total_chords}和弦',
                  {'total_chords': total_chords, 'sections': len(by_section)})

    def a1_reharmonize(self, from_bar: int, new_chords_count: int):
        """A1 和声回退（由 B2 致命信号触发）。"""
        self.summary.b2_reharmonize_count += 1
        self.warning('A1', f'和声回退 from bar {from_bar}, {new_chords_count}新和弦',
                     {'from_bar': from_bar, 'new_chords': new_chords_count})

    # ── A2 动机提取 ──────────────────────────────────────

    def a2_seed_analysis(self, landmarks: list, dna: dict | None = None):
        """A2 种子分析结果。"""
        self.info('A2', f'种子分析: {len(landmarks)}个地标',
                  {'landmarks': landmarks, 'dna_contour': dna.get('contour', []) if dna else None})

    def a2_motif_extracted(self, label: str, dna):
        """A2 动机提取成功。"""
        self.summary.motifs_extracted += 1
        self.debug('A2', f'动机提取: {label}',
                   {'label': label,
                    'contour_len': len(dna.contour) if dna.contour else 0,
                    'ambitus': dna.ambitus if hasattr(dna, 'ambitus') else None,
                    'register_centroid': dna.register_centroid if hasattr(dna, 'register_centroid') else None})

    # ── A3 统计画像 ──────────────────────────────────────

    def a3_baseline(self, baselines: dict):
        """A3 基线统计。"""
        self.info('A3', '基线统计已建立',
                  {'density': baselines.get('density', 0),
                   'rest_ratio': baselines.get('rest_ratio', 0),
                   'velocity_mean': baselines.get('velocity_mean', 0),
                   'voice_count': baselines.get('voice_count', 0)})

    def a3_bar_record(self, bar: int, stats, b1_score: float | None = None):
        """A3 单 bar 统计记录。"""
        self.summary.bars_logged += 1
        density = getattr(stats, 'density', 0) if stats else 0
        rest_ratio = getattr(stats, 'rest_ratio', 0) if stats else 0
        note_count = getattr(stats, 'note_count', 0) if stats else 0

        self.debug('A3', f'Bar {bar}: density={density:.1f} notes={note_count} '
                          f'rest={rest_ratio:.2f} b1={b1_score:.3f}' if b1_score else f'Bar {bar}: density={density:.1f} notes={note_count}',
                   {'bar': bar, 'density': density, 'rest_ratio': rest_ratio,
                    'note_count': note_count, 'b1_score': b1_score})

    def a3_section_snapshot(self, section_idx: int, section_type: str,
                            bar_count: int, avg_density: float):
        """A3 段快照。"""
        self.info('A3', f'段快照 S{section_idx}({section_type}): '
                         f'{bar_count}bar avg_density={avg_density:.1f}',
                  {'section_idx': section_idx, 'type': section_type,
                   'bar_count': bar_count, 'avg_density': avg_density})

    # ── B1 硬约束层 ──────────────────────────────────────
    # 负责: token 屏蔽 (context bans + dynamic bans)、声部音域、平行禁止

    def b1_context_bans(self, ban_count: int, prefixes: list[str]):
        """B1 上下文禁令初始化（Program/Tempo/TimeSig/Tuplet/GraceNote）。"""
        self.summary.b1_context_ban_count = ban_count
        self.info('B1', f'上下文禁令: {ban_count}tok ({", ".join(prefixes)})',
                  {'ban_count': ban_count, 'prefixes': prefixes})

    def b1_bar_bans(self, bar_idx: int, context_count: int,
                     dynamic_count: int, ban_items: dict[str, int] | None = None):
        """B1 逐 bar 禁令汇总。"""
        total = context_count + dynamic_count
        if total > 0:
            self.summary.b1_hard_bans_applied += 1
            self.summary.b1_total_bans += dynamic_count
            self.debug('B1', f'Bar {bar_idx}: 禁{total}tok (上下文{context_count}+动态{dynamic_count})',
                       {'bar_idx': bar_idx, 'context': context_count,
                        'dynamic': dynamic_count, 'items': ban_items})
        else:
            self.debug('B1', f'Bar {bar_idx}: 无禁令', {'bar_idx': bar_idx})

    def b1_bar_result(self, bar_idx: int, bar_tokens: int,
                       note_count: int, b1_score: float | None):
        """B1 bar 完成 — 硬约束通过状态。"""
        level = logging.DEBUG
        if b1_score is not None and b1_score < 0.3:
            level = logging.WARNING
        self._log(level, 'B1',
                  f'Bar {bar_idx}: {bar_tokens}tok {note_count}notes b1={b1_score:.3f}'
                  if b1_score else f'Bar {bar_idx}: {bar_tokens}tok {note_count}notes',
                  {'bar_idx': bar_idx, 'tokens': bar_tokens,
                   'notes': note_count, 'b1_score': b1_score})

    # ── B2 决策调参层 ─────────────────────────────────────
    # 负责: 温区退火、创新预算、致命信号、参数调节、发展配方

    def b2_section_start(self, section_idx: int, section_type: str,
                          bars: int, innovation_budget: float,
                          development_ops: list | None = None):
        """B2 段开始 — 创新预算 + 发展配方。"""
        self.info('B2', f'段开始 S{section_idx}({section_type}): '
                        f'{bars}bar innov={innovation_budget:.2f} '
                        f'ops={development_ops}',
                  {'section_idx': section_idx, 'type': section_type,
                   'bars': bars, 'innovation_budget': innovation_budget,
                   'development_ops': development_ops})

    def b2_zone_temperature(self, bar_idx: int, base_temp: float,
                             adjusted_temp: float, zone: str):
        """B2 温区退火 — 冷→热→冷曲线。"""
        self.summary.b2_temperature_adjustments += 1
        self.debug('B2', f'Bar {bar_idx}: T={adjusted_temp:.3f} '
                         f'(base={base_temp:.2f} zone={zone})',
                   {'bar_idx': bar_idx, 'base': base_temp,
                    'adjusted': adjusted_temp, 'zone': zone})

    def b2_fatal_signal(self, signal: str, bar_idx: int, reason: str = ''):
        """B2 致命信号 — reharmonize / abort。"""
        self.summary.b2_fatal_signals += 1
        if signal == 'reharmonize':
            self.summary.b2_reharmonize_count += 1
        self.error('B2', f'致命信号: {signal} at bar {bar_idx} — {reason}',
                   {'signal': signal, 'bar': bar_idx, 'reason': reason})

    def b2_innovation_admitted(self, bar_idx: int, innovation_type: str,
                                surprise: float):
        """B2 创新承认 — 偏离基线但硬约束通过。"""
        self.summary.b2_innovations_admitted += 1
        self.info('B2', f'Bar {bar_idx}: 创新 type={innovation_type} surprise={surprise:.3f}',
                  {'bar': bar_idx, 'type': innovation_type, 'surprise': surprise})

    def b2_parameter_adjustment(self, adjustments: dict):
        """B2 参数调节 — temperature/complexity/rest_penalty 变更。"""
        if adjustments:
            self.debug('B2', f'参数调整: {adjustments}', {'adjustments': adjustments})

    # ── C 评价层 ─────────────────────────────────────────

    def c_section_compare(self, theme_idx: int, recap_idx: int,
                           pc_similarity: float, density_similarity: float):
        """C 段落对比。"""
        level = logging.WARNING if pc_similarity < 0.7 else logging.INFO
        self._log(level, 'C',
                  f'再现部自检: S{theme_idx}↔S{recap_idx} '
                  f'PC_sim={pc_similarity:.3f} density_sim={density_similarity:.3f}',
                  {'theme': theme_idx, 'recap': recap_idx,
                   'pc_sim': pc_similarity, 'density_sim': density_similarity})

    def c_structural_fix(self, fix_type: str, section: int | None,
                          detail: str = ''):
        """C 结构修复。"""
        self.summary.structural_fixes += 1
        self.warning('C', f'结构修复: {fix_type} S{section} {detail}',
                     {'type': fix_type, 'section': section, 'detail': detail})

    def c_xml_review(self, inspections_count: int, issues: dict):
        """C MusicXML 审查结果。"""
        self.summary.xml_review_bars = inspections_count
        total_warnings = sum(len(v) for v in issues.values()) if isinstance(issues, dict) else 0
        level = logging.WARNING if total_warnings > 5 else logging.INFO
        self._log(level, 'C',
                  f'XML审查: {inspections_count}bar {total_warnings}条问题',
                  {'bars': inspections_count, 'issues': issues})

    def c_xml_comparison(self, fidelity: float, ok: bool, mismatches: list):
        """C Token↔XML 对比结果。"""
        self.summary.xml_fidelity = fidelity
        level = logging.WARNING if fidelity < 0.7 else logging.INFO
        self._log(level, 'C',
                  f'Token↔XML保真度: {fidelity:.3f} {"✓" if ok else "✗"}',
                  {'fidelity': fidelity, 'ok': ok, 'mismatches': mismatches})

    def c_feedback_to_b(self, section_idx: int, alerts: list,
                         temperature_delta: float, complexity_delta: float,
                         fatal: str | None):
        """C→B 反馈。"""
        if alerts:
            self.info('C', f'C→B反馈 S{section_idx}→{section_idx+1}: '
                           f'{"; ".join(alerts[:3])}',
                      {'from_section': section_idx, 'alerts': alerts,
                       'T_delta': temperature_delta,
                       'C_delta': complexity_delta,
                       'fatal': fatal})

    def c_scores(self, novelty: float, diversity: float,
                  total_score: float = 0.0):
        """C 最终评分。"""
        self.summary.novelty_score = novelty
        self.summary.diversity_score = diversity
        self.summary.total_score = total_score
        self.info('C', f'评分: 新颖={novelty:.4f} 多样={diversity:.4f} 综合={total_score:.4f}',
                  {'novelty': novelty, 'diversity': diversity, 'total': total_score})

    def c_archive(self, archive_commands: list):
        """C 归档指令。"""
        if archive_commands:
            self.info('C', f'归档: {len(archive_commands)}条',
                      {'commands': archive_commands})
