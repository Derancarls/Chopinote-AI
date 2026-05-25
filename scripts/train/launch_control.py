#!/usr/bin/env python3
"""Chopinote-AI 点火控制台。

预检 → 点火 → 监视 → 中止，类似火箭发射控制面板。

用法:
    python scripts/launch_control.py check          # 预检清单
    python scripts/launch_control.py launch         # 全流程点火
    python scripts/launch_control.py monitor        # 实时仪表盘
    python scripts/launch_control.py abort          # 中止一切
    python scripts/launch_control.py status         # 快照
"""
import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# 确保项目根目录在 sys.path 中
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import psutil

# ── ANSI 终端渲染 ────────────────────────────────────────────
# 从共享的 terminal 模块导入，额外定义 launch 专用的 emoji 组合
from chopinote_cli.terminal import (
    RED as _RED, GREEN as _GREEN, YELLOW as _YELLOW, CYAN as _CYAN,
    BOLD as _BOLD, DIM as _DIM, RESET as _RESET,
    CLR as _CLR, CLR_EOL as _CLR_EOL,
    visible_len as _visible_len, pad_line as _pad_line,
    TerminalBox,
)
_CHECK_PASS = f'{_GREEN}✅{_RESET}'
_CHECK_FAIL = f'{_RED}❌{_RESET}'
_CHECK_WARN = f'{_YELLOW}⚠{_RESET}'
_BOX_WIDTH = 62


# ── 训练配置 ──────────────────────────────────────────────────
@dataclass
class TrainingConfig:
    """所有训练参数集中配置。支持 YAML 文件和 CLI 覆盖。"""
    # 路径（可通过环境变量覆盖）
    project_dir: str = field(default_factory=lambda: os.environ.get(
        'CHOPINOTE_PROJECT_DIR', '/root/Chopinote-AI'))
    data_dir: str = field(default_factory=lambda: os.environ.get(
        'CHOPINOTE_DATA_DIR', '/root/autodl-tmp/data/processed'))
    checkpoint_dir: str = field(default_factory=lambda: os.environ.get(
        'CHOPINOTE_OUTPUT_DIR', '/root/autodl-tmp/chopinote/checkpoints'))
    log_dir: str = field(default_factory=lambda: os.environ.get(
        'CHOPINOTE_LOG_DIR', '/root/autodl-tmp/chopinote/logs'))
    tb_dir: str = field(default_factory=lambda: os.environ.get(
        'CHOPINOTE_TB_DIR', '/root/autodl-tmp/chopinote/tensorboard'))

    # 超参数
    batch_size: int = 8
    grad_accum: int = 4
    phase1_steps: int = 120000
    phase1_lr: float = 1.5e-4
    phase1_warmup: int = 4000
    phase2_steps: int = 50000
    phase2_lr: float = 1.0e-4
    phase2_warmup: int = 2000

    # 看门狗
    max_restarts: int = 5
    restart_delay: int = 30

    # 恢复训练
    resume_checkpoint: Optional[str] = None

    # 数据文件
    train_list: str = 'train.txt'
    val_list: str = 'val.txt'

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum

    @property
    def train_list_path(self) -> str:
        return os.path.join(self.data_dir, self.train_list)

    @property
    def val_list_path(self) -> str:
        return os.path.join(self.data_dir, self.val_list)

    @property
    def lock_file(self) -> str:
        return '/tmp/chopinote-train.lock'

    @property
    def pid_file(self) -> str:
        return '/tmp/chopinote-train.pid'

    @property
    def watchdog_path(self) -> str:
        return '/tmp/chopinote-watchdog.sh'

    @property
    def state_file(self) -> str:
        return os.path.join(self.log_dir, '.launch_state.json')

    def apply_args(self, args: argparse.Namespace) -> 'TrainingConfig':
        for key in ('batch_size', 'grad_accum',
                     'phase1_steps', 'phase1_lr', 'phase1_warmup',
                     'phase2_steps', 'phase2_lr', 'phase2_warmup',
                     'max_restarts', 'restart_delay'):
            val = getattr(args, key, None)
            if val is not None:
                setattr(self, key, val)
        # 路径覆盖
        for key in ('project_dir', 'data_dir', 'checkpoint_dir',
                     'log_dir', 'tb_dir'):
            val = getattr(args, key, None)
            if val is not None:
                setattr(self, key, val)
        # 恢复训练
        val = getattr(args, 'resume', None)
        if val is not None:
            self.resume_checkpoint = val
        return self


# ── 启动状态文件 ──────────────────────────────────────────────
@dataclass
class LaunchState:
    pid: int
    launched_at: str  # ISO timestamp

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f)

    @classmethod
    def load(cls, path: str) -> Optional['LaunchState']:
        try:
            with open(path) as f:
                data = json.load(f)
            return cls(**data)
        except (FileNotFoundError, json.JSONDecodeError, TypeError):
            return None


# ── 辅助函数 ──────────────────────────────────────────────────
def _run(cmd: list[str], timeout: int = 10) -> subprocess.CompletedProcess:
    """运行 shell 命令并返回结果。"""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return subprocess.CompletedProcess(args=cmd, returncode=127, stdout='', stderr='')


def _gpu_info() -> dict:
    """解析 nvidia-smi --query-gpu 输出，含功率、时钟。"""
    r = _run(['nvidia-smi',
              '--query-gpu=index,name,memory.used,memory.total,'
              'utilization.gpu,temperature.gpu,power.draw,power.limit,'
              'clocks.current.sm,clocks.current.memory',
              '--format=csv,noheader'])
    if r.returncode != 0:
        return {}
    fields = [f.strip() for f in r.stdout.strip().split(', ')]
    if len(fields) < 10:
        return {}

    def _safe_int(v: str, default: int = 0) -> int:
        try:
            return int(v.replace(' MiB', '').replace(' %', '').replace(' MHz', ''))
        except (ValueError, AttributeError):
            return default

    def _safe_float(v: str, default: float = 0.0) -> float:
        try:
            return float(v.replace(' W', ''))
        except (ValueError, AttributeError):
            return default

    return {
        'index': fields[0],
        'name': fields[1],
        'mem_used_mib': _safe_int(fields[2]),
        'mem_total_mib': _safe_int(fields[3]),
        'util_pct': _safe_int(fields[4]),
        'temp_c': _safe_int(fields[5]),
        'power_draw_w': _safe_float(fields[6]),
        'power_limit_w': _safe_float(fields[7]),
        'sm_clock_mhz': _safe_int(fields[8]),
        'mem_clock_mhz': _safe_int(fields[9]),
    }


def _gpu_processes() -> list[dict]:
    """列出 nvidia-smi 报告的 GPU 进程。"""
    r = _run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
              '--format=csv,noheader'])
    if r.returncode != 0:
        return []
    procs = []
    for line in r.stdout.strip().split('\n'):
        if not line.strip():
            continue
        parts = line.split(', ')
        if len(parts) >= 2:
            procs.append({'pid': int(parts[0]), 'name': parts[1],
                          'memory': parts[2] if len(parts) > 2 else '?'})
    return procs


def _progress_bar(filled: int, total: int, width: int = 30) -> str:
    """生成进度条：▓▓▓▓▓░░░░░ 30/50"""
    if total <= 0:
        return '[?' * width + ']'
    filled_w = int(width * filled / total)
    filled_w = min(filled_w, width)
    bar = '▓' * filled_w + '░' * (width - filled_w)
    pct = 100.0 * filled / total if total > 0 else 0
    return f'{bar} {filled:,}/{total:,} ({pct:.1f}%)'


def _eta_str(seconds: float) -> str:
    """将秒转为人类可读的 ETA 字符串。"""
    if seconds <= 0 or not math.isfinite(seconds):
        return '--'
    if seconds < 60:
        return f'{int(seconds)}s'
    if seconds < 3600:
        return f'{int(seconds / 60)}m {int(seconds % 60)}s'
    return f'{int(seconds / 3600)}h {int((seconds % 3600) / 60)}m'


def _fmt_size(gib: float) -> str:
    """格式化 GiB 为简洁字符串。"""
    if gib < 1:
        return f'{int(gib * 1024)} MiB'
    return f'{gib:.1f} GiB'


def _fmt_time(ts: str) -> str:
    """解析 ISO 时间戳为本地时间字符串。"""
    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except (ValueError, TypeError):
        return ts or '--'


def _tail_lines(path: str, n: int = 50) -> list[str]:
    """读取文件最后 n 行（高效，从文件尾开始读）。"""
    try:
        with open(path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            chunk = min(size, 65536)
            f.seek(size - chunk)
            data = f.read(chunk).decode('utf-8', errors='replace')
            lines = data.split('\n')
            return lines[-n:]
    except (FileNotFoundError, OSError):
        return []


# ── 日志解析 ──────────────────────────────────────────────────
_STEP_RE = re.compile(
    r'(?:\[.+\]\s+)?Step (\d+)/(\d+)(?:\s*\(global (\d+)\))? \| '
    r'Loss: ([\d.]+)(?: \| Sec: ([\d.]+))?(?: \| Chord: ([\d.]+))? \| '
    r'LR: ([\de.\-+]+)(?: \| GN: ([\d.]+))?'
)
_PHASE_RE = re.compile(
    r'\[(\w[\w ]+)\] Step \d+/\d+'
)
_VAL_LOSS_RE = re.compile(
    r'(?:\[.+\]\s+)?Val loss: ([\d.]+)'
)
_VAL_ACC_RE = re.compile(
    r'acc/(\w+)=([\d.]+)'
)


def _parse_training_progress(log_path: str) -> dict:
    """解析 crashes.log 获取最新训练进度。"""
    lines = _tail_lines(log_path, 300)
    result = {
        'phase': None, 'phase_step': 0, 'phase_total': 0,
        'global_step': 0, 'loss': None, 'lr': None, 'grad_norm': None,
        'sec_loss': None, 'chord_loss': None,
        'val_loss': None, 'val_loss_step': 0, 'val_acc': {},
        'step_time_s': None,
        'loss_history': [],  # last N losses for trend
    }

    last_line_time = None
    for line in reversed(lines):
        ts_match = re.match(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
        if ts_match:
            last_line_time = ts_match.group(1)

        # val loss + accuracy (earliest recent entry wins)
        val_match = _VAL_LOSS_RE.search(line)
        if val_match and result['val_loss'] is None:
            result['val_loss'] = float(val_match.group(1))
            result['val_loss_step'] = result['global_step']
            # Parse accuracy metrics from same line
            for type_name, acc_val in _VAL_ACC_RE.findall(line):
                result['val_acc'][type_name] = float(acc_val)

        # step progress (first, i.e. most recent, entry wins)
        step_match = _STEP_RE.search(line)
        if step_match and result['loss'] is None:
            phase_step_val = int(step_match.group(1))
            result['phase_step'] = phase_step_val
            result['phase_total'] = int(step_match.group(2))
            global_str = step_match.group(3)
            result['global_step'] = int(global_str) if global_str else phase_step_val
            result['loss'] = float(step_match.group(4))
            sec_str = step_match.group(5)
            result['sec_loss'] = float(sec_str) if sec_str else None
            chord_str = step_match.group(6)
            result['chord_loss'] = float(chord_str) if chord_str else None
            result['lr'] = float(step_match.group(7))
            gn_str = step_match.group(8)
            result['grad_norm'] = float(gn_str) if gn_str else None

            # Parse time from log: "... | Time: 123.4s"
            time_m = re.search(r'\| Time: ([\d.]+)s', line)
            if time_m:
                result['step_time_s'] = float(time_m.group(1))
                # step_time_s is cumulative elapsed, not per-step

        # collect loss history for trend (up to 10)
        step_any = _STEP_RE.search(line)
        if step_any and len(result['loss_history']) < 10:
            result['loss_history'].append(float(step_any.group(4)))

        phase_match = _PHASE_RE.search(line)
        if phase_match and result['phase'] is None:
            result['phase'] = phase_match.group(1)

    # loss_history collected in reversed order → reverse back to chronological
    result['loss_history'].reverse()
    return result


def _parse_watchdog_status(log_path: str) -> dict:
    """解析 watchdog.log 获取看门狗状态。"""
    lines = _tail_lines(log_path, 100)
    restarts = 0
    last_launch_time = None
    last_crash_time = None
    is_active = False
    maxed_out = False

    # 正向扫描日志，按时间顺序累积状态
    for line in lines:
        ts_match = re.match(r'^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]', line)
        ts = ts_match.group(1) if ts_match else None

        if '看门狗启动' in line:
            restarts = 0
            is_active = True
            maxed_out = False
        elif '训练崩溃' in line:
            last_crash_time = ts or last_crash_time
            restarts += 1
        elif '训练启动' in line:
            last_launch_time = ts or last_launch_time
        elif '最大重启次数' in line:
            maxed_out = True
            is_active = False
        elif '训练正常完成' in line:
            is_active = False
            maxed_out = False

    uptime = 0
    if is_active and last_launch_time:
        try:
            dt = datetime.strptime(last_launch_time, '%Y-%m-%d %H:%M:%S')
            uptime = (datetime.now() - dt).total_seconds()
        except ValueError:
            pass

    return {
        'restarts': restarts,
        'last_launch_time': last_launch_time,
        'last_crash_time': last_crash_time,
        'uptime_seconds': uptime,
        'maxed_out': maxed_out,
        'is_active': is_active,
    }


# ── 预检 ──────────────────────────────────────────────────────
@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str = ''
    is_warning: bool = False


class PreflightChecker:
    """19 项预检。"""

    def __init__(self, config: TrainingConfig):
        self.config = config

    def check_pytorch_cuda(self) -> CheckResult:
        try:
            import torch
            py_ver = torch.__version__
            cuda_ver = torch.version.cuda or 'N/A'
            cc = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
            cc_str = f'{cc[0]}.{cc[1]}' if cc else 'N/A'
            return CheckResult('PyTorch/CUDA', True,
                               f'PT {py_ver} CUDA {cuda_ver} CC {cc_str}')
        except Exception as e:
            return CheckResult('PyTorch/CUDA', False, str(e))

    def check_gpu_available(self) -> CheckResult:
        try:
            import torch
            if torch.cuda.is_available():
                count = torch.cuda.device_count()
                name = torch.cuda.get_device_name(0)
                return CheckResult('GPU 可用', True, f'{name} x{count}')
            return CheckResult('GPU 可用', False, 'CUDA 不可用')
        except (ImportError, RuntimeError) as e:
            return CheckResult('GPU 可用', False, str(e))

    def check_gpu_memory(self) -> CheckResult:
        info = _gpu_info()
        if not info:
            return CheckResult('GPU 显存', False, '无法读取 nvidia-smi')
        free_mib = info['mem_total_mib'] - info['mem_used_mib']
        free_gib = free_mib / 1024
        if free_gib < 4:
            return CheckResult('GPU 显存', False,
                               f'仅剩 {free_gib:.1f} GiB（需要 ≥4 GiB）')
        return CheckResult('GPU 显存', True,
                           f'空余 {free_gib:.1f} GiB / {info["mem_total_mib"]/1024:.0f} GiB')

    def check_zombie_processes(self) -> CheckResult:
        procs = _gpu_processes()
        own_pid = os.getpid()
        zombies = [p for p in procs if p['pid'] != own_pid and 'python' in p['name'].lower()]
        if zombies:
            names = ', '.join(f'PID {p["pid"]}' for p in zombies)
            mems = sum(int(p.get('memory', '0').split()[0]) for p in zombies if 'MiB' in p.get('memory', ''))
            return CheckResult('僵尸进程', False,
                               f'{len(zombies)} 个残留: {names} ({mems} MiB)')
        return CheckResult('僵尸进程', True, '无残留进程')

    def check_tmux_session(self) -> CheckResult:
        r = _run(['tmux', 'has-session', '-t', 'chopinote'])
        if r.returncode == 0:
            return CheckResult('tmux 会话', False, '已有 chopinote 会话在运行', is_warning=True)
        return CheckResult('tmux 会话', True, '无冲突会话')

    def check_stale_locks(self) -> CheckResult:
        lock = self.config.lock_file
        if not os.path.exists(lock):
            return CheckResult('锁文件', True, '无锁文件')
        try:
            with open(lock) as f:
                pid = int(f.read().strip())
            os.kill(pid, 0)  # 检查进程是否存活
            return CheckResult('锁文件', False,
                               f'锁文件存在且 PID {pid} 仍在运行', is_warning=True)
        except (ValueError, ProcessLookupError):
            return CheckResult('锁文件', False, '锁文件过时（PID 已死）')

    def check_data_files(self) -> CheckResult:
        missing = []
        for name in [self.config.train_list, self.config.val_list, 'token_lengths.json']:
            path = os.path.join(self.config.data_dir, name)
            if not os.access(path, os.R_OK):
                missing.append(name)
        if missing:
            return CheckResult('数据文件', False,
                               f'缺少: {", ".join(missing)}')
        return CheckResult('数据文件', True, f'train/val/token_lengths.json 均存在')

    def check_disk_space(self) -> CheckResult:
        try:
            usage = psutil.disk_usage(self.config.data_dir)
            free_pct = usage.free / usage.total * 100
            free_gib = usage.free / 1024**3
            if free_pct < 10:
                return CheckResult('磁盘空间', False,
                                   f'仅剩 {free_gib:.0f} GiB ({free_pct:.0f}%)')
            return CheckResult('磁盘空间', True,
                               f'空余 {free_gib:.0f} GiB / {usage.total/1024**3:.0f} GiB ({free_pct:.0f}%)')
        except OSError as e:
            return CheckResult('磁盘空间', False, str(e))

    def check_model_import(self) -> CheckResult:
        try:
            sys.path.insert(0, self.config.project_dir)
            from chopinote_model.config import ModelConfig, TrainingConfig as TC
            from chopinote_model.model import MusicTransformer
            cfg = ModelConfig()
            total = cfg.vocab_size * cfg.d_model  # embedding only, not full model
            return CheckResult('模型导入', True,
                               f'MusicTransformer {cfg.n_layers}L/{cfg.n_heads}H/{cfg.d_model}d')
        except Exception as e:
            return CheckResult('模型导入', False, str(e))

    def check_watchdog_script(self) -> CheckResult:
        path = self.config.watchdog_path
        if os.access(path, os.X_OK):
            return CheckResult('看门狗脚本', True, path)
        return CheckResult('看门狗脚本', True, '将重新生成', is_warning=True)

    # ── 新增高级检查 ──────────────────────────────────────

    def check_python_deps(self) -> CheckResult:
        missing = []
        for pkg in ['torch', 'liger_kernel', 'psutil', 'tensorboard']:
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg)
        if missing:
            return CheckResult('Python 依赖', False, f'缺少: {", ".join(missing)}')
        return CheckResult('Python 依赖', True, '核心依赖均已安装')

    def check_config_model(self) -> CheckResult:
        try:
            from chopinote_model.config import ModelConfig
            cfg = ModelConfig()
            params = (cfg.vocab_size * cfg.d_model  # embedding
                      + 2 * cfg.d_model * cfg.d_ff * cfg.n_layers  # MLP W1+W2
                      + cfg.n_layers * (
                          4 * cfg.d_model * (cfg.d_model // cfg.n_heads) * cfg.n_heads  # QKV+O
                          + 4 * cfg.d_model  # LN
                      )
                      + cfg.d_model * cfg.vocab_size)  # lm_head
            return CheckResult('模型配置', True,
                               f'{cfg.n_layers}L/{cfg.n_heads}H/{cfg.d_model}d/'
                               f'd_ff={cfg.d_ff} (~{params/1e6:.0f}M params)')
        except Exception as e:
            return CheckResult('模型配置', False, str(e))

    def check_data_integrity(self) -> CheckResult:
        """采样检查 token 文件是否能被正确加载。"""
        train_list = self.config.train_list_path
        if not os.path.exists(train_list):
            return CheckResult('数据完整性', False, f'训练列表不存在: {train_list}')
        try:
            with open(train_list) as f:
                files = [line.strip() for line in f if line.strip()]
            if not files:
                return CheckResult('数据完整性', False, '训练列表为空')
            # 随机采样 3 个文件检查
            import random
            samples = random.sample(files, min(3, len(files)))
            for fpath in samples:
                # 匹配 TokenDataset._resolve_path 逻辑：优先 data_dir/tokens_v3/<basename>
                full = os.path.join(self.config.data_dir, 'tokens_v3', os.path.basename(fpath))
                if not os.path.exists(full):
                    full = fpath if os.path.isabs(fpath) else os.path.join(self.config.data_dir, fpath)
                if not os.path.exists(full):
                    return CheckResult('数据完整性', False, f'文件缺失: {full}')
                size = os.path.getsize(full)
                if size == 0:
                    return CheckResult('数据完整性', False, f'空文件: {full}')
                # 尝试加载 token 文件 (JSON lines 格式)
                try:
                    with open(full) as fh:
                        first_line = fh.readline().strip()
                        if first_line:
                            import json
                            json.loads(first_line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    return CheckResult('数据完整性', False, f'无效 token 文件: {full}')
            return CheckResult('数据完整性', True,
                               f'{len(files)} 个 token 文件，采样 {len(samples)} 个正常')
        except Exception as e:
            return CheckResult('数据完整性', False, str(e))

    def check_gpu_power_cap(self) -> CheckResult:
        info = _gpu_info()
        if not info:
            return CheckResult('GPU 功率', False, '无法读取 nvidia-smi')
        pct = info['power_draw_w'] / info['power_limit_w'] * 100 if info['power_limit_w'] > 0 else 0
        return CheckResult('GPU 功率', True,
                           f'{info["power_draw_w"]:.0f}W / {info["power_limit_w"]:.0f}W ({pct:.0f}%)')

    def check_env_vars(self) -> CheckResult:
        needed = ['CHOPINOTE_DATA_DIR', 'CHOPINOTE_OUTPUT_DIR']
        found = []
        missing = []
        for var in needed:
            if os.environ.get(var):
                found.append(var)
            else:
                missing.append(var)
        if missing:
            return CheckResult('环境变量', False,
                               f'未设置: {", ".join(missing)}（使用默认值）', is_warning=True)
        return CheckResult('环境变量', True, f'已设置: {", ".join(found)}')

    def check_disk_io(self) -> CheckResult:
        """快速磁盘写入速度测试。"""
        test_file = os.path.join(self.config.log_dir, '.io_test')
        try:
            os.makedirs(self.config.log_dir, exist_ok=True)
            import tempfile
            data = b'x' * 64 * 1024  # 64KB
            t0 = time.time()
            for _ in range(128):  # 8MB total
                with open(test_file, 'ab') as f:
                    f.write(data)
            t1 = time.time()
            os.remove(test_file)
            speed = 8 / (t1 - t0) if (t1 - t0) > 0 else 0
            if speed < 10:
                return CheckResult('磁盘 I/O', False,
                                   f'{speed:.0f} MB/s (过慢，建议 ≥50 MB/s)', is_warning=True)
            return CheckResult('磁盘 I/O', True, f'{speed:.0f} MB/s')
        except Exception as e:
            return CheckResult('磁盘 I/O', False, str(e))

    def check_checkpoint_dir(self) -> CheckResult:
        ckpt_dir = self.config.checkpoint_dir
        if not os.path.exists(ckpt_dir):
            return CheckResult('Checkpoint 目录', True, f'{ckpt_dir} (将创建)')
        ckpts = sorted(Path(ckpt_dir).glob('step_*.pt'))
        latest = str(ckpts[-1].name) if ckpts else '无'
        try:
            usage = psutil.disk_usage(ckpt_dir)
            free_gib = usage.free / 1024**3
            return CheckResult('Checkpoint 目录', True,
                               f'{len(ckpts)} 个存档, 最新: {latest}, 空闲 {free_gib:.0f} GiB')
        except OSError:
            return CheckResult('Checkpoint 目录', True, f'{len(ckpts)} 个存档')

    def run(self) -> list[CheckResult]:
        return [
            self.check_pytorch_cuda(),
            self.check_gpu_available(),
            self.check_gpu_memory(),
            self.check_gpu_power_cap(),
            self.check_zombie_processes(),
            self.check_tmux_session(),
            self.check_stale_locks(),
            self.check_data_files(),
            self.check_data_integrity(),
            self.check_checkpoint_dir(),
            self.check_disk_io(),
            self.check_disk_space(),
            self.check_env_vars(),
            self.check_model_import(),
            self.check_config_model(),
            self.check_python_deps(),
            self.check_watchdog_script(),
        ]


def _print_checks(results: list[CheckResult]):
    """打印预检结果表格。"""
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f'  {_BOLD}预检清单{_RESET}')
    for r in results:
        icon = _CHECK_PASS if r.passed else (_CHECK_WARN if r.is_warning else _CHECK_FAIL)
        print(f'  {icon} {r.name}: {r.message}')
    print(f'  {_BOLD}{passed}/{total}{_RESET} 项通过')
    return passed == total


# ── 看门狗模板 ────────────────────────────────────────────────
_WATCHDOG_TEMPLATE = r"""#!/bin/bash
set -uo pipefail

PROJECT_DIR="{project_dir}"
DATA_DIR="{data_dir}"
CHECKPOINT_DIR="{checkpoint_dir}"
LOG_DIR="{log_dir}"
TB_DIR="{tb_dir}"
CRASH_LOG="$LOG_DIR/crashes.log"
WATCHDOG_LOG="$LOG_DIR/watchdog.log"
MAX_RESTARTS={max_restarts}
RESTART_DELAY={restart_delay}
RESUME_ARG="{resume_arg}"

cd "$PROJECT_DIR"

echo "[$(date "+%Y-%m-%d %H:%M:%S")] ⚙️  看门狗启动 (最多重启 $MAX_RESTARTS 次)" >> "$WATCHDOG_LOG"

RESTART_COUNT=0
while true; do
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] 🚀 训练启动 (重启 #$RESTART_COUNT)" >> "$WATCHDOG_LOG"

    python3 -c "
import torch
torch.cuda.empty_cache()
print(f'CUDA cleared | GPU: {{torch.cuda.get_device_name(0)}} | '
      f'显存: {{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}}GB')
" 2>> "$CRASH_LOG"

    python3 scripts/train/run_curriculum_training.py \
        --midi-train-list "$DATA_DIR/train.txt" \
        --musicxml-train-list "$DATA_DIR/train.txt" \
        --val-list "$DATA_DIR/val.txt" \
        --data-dir "$DATA_DIR" \
        --phase1-steps {phase1_steps} \
        --phase1-lr {phase1_lr} \
        --phase1-warmup {phase1_warmup} \
        --phase2-steps {phase2_steps} \
        --phase2-lr {phase2_lr} \
        --phase2-warmup {phase2_warmup} \
        --batch-size {batch_size} \
        --output-dir "$CHECKPOINT_DIR" \
        --log-dir "$TB_DIR" \
        $RESUME_ARG \
        2>> "$CRASH_LOG"

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date "+%Y-%m-%d %H:%M:%S")] ✅ 训练正常完成" >> "$WATCHDOG_LOG"
        break
    fi

    RESTART_COUNT=$((RESTART_COUNT + 1))

    {{
        echo "═══════════════════════════════════════════════"
        echo "[$(date "+%Y-%m-%d %H:%M:%S")] ❌ 训练崩溃 (exit=$EXIT_CODE, 重启 #$RESTART_COUNT)"
        echo "  GPU 状态:"
        nvidia-smi --query-gpu=index,name,temperature.gpu,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/    /'
        echo "  系统内存:"
        free -h | grep Mem | sed 's/^/    /'
        echo "═══════════════════════════════════════════════"
    }} >> "$CRASH_LOG"
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] ❌ 训练崩溃 (exit=$EXIT_CODE, 重启 #$RESTART_COUNT)" >> "$WATCHDOG_LOG"

    if [ $RESTART_COUNT -ge $MAX_RESTARTS ]; then
        echo "[$(date "+%Y-%m-%d %H:%M:%S")] ⛔ 已达最大重启次数 ($MAX_RESTARTS)，停止" >> "$WATCHDOG_LOG"
        break
    fi

    echo "[$(date "+%Y-%m-%d %H:%M:%S")] 等待 ${{RESTART_DELAY}}s 后重启..." >> "$WATCHDOG_LOG"
    sleep $RESTART_DELAY
done
"""


def _find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """扫描 checkpoint 目录，返回最新 step_*.pt 的路径。"""
    try:
        ckpts = sorted(
            Path(checkpoint_dir).glob('step_*.pt'),
            key=lambda p: int(p.stem.split('_')[1]),
        )
        return str(ckpts[-1]) if ckpts else None
    except (ValueError, IndexError, OSError):
        return None


# ── LaunchController ──────────────────────────────────────────
class LaunchController:
    """点火控制器。"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self._step = 0

    def _log(self, msg: str, end='\n'):
        icon = ['🟢', '🔄', '⏳', '✅', 'ℹ️ '][
            min(self._step, 4)
        ] if self._step <= 4 else '✅'
        print(f'  [{self._step}/7] {msg}', end=end, flush=True)

    def cleanup(self) -> int:
        """清理残留进程和文件。返回清理的进程数。"""
        # tmux kill
        _run(['tmux', 'kill-session', '-t', 'chopinote'], timeout=5)

        # 杀掉残留 GPU Python 进程
        killed = 0
        for proc in _gpu_processes():
            if proc['pid'] != os.getpid() and 'python' in proc['name'].lower():
                try:
                    os.kill(proc['pid'], signal.SIGTERM)
                    killed += 1
                except (ProcessLookupError, PermissionError):
                    pass
        if killed:
            time.sleep(2)  # 等进程退出

        # 清理临时文件
        for f in [self.config.lock_file, self.config.pid_file,
                  self.config.watchdog_path]:
            try:
                os.remove(f)
            except FileNotFoundError:
                pass

        # CUDA 缓存
        try:
            import torch
            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass

        return killed

    def generate_watchdog(self) -> str:
        """生成看门狗脚本。"""
        if self.config.resume_checkpoint:
            latest_ckpt = self.config.resume_checkpoint
        else:
            latest_ckpt = _find_latest_checkpoint(self.config.checkpoint_dir)
        resume_arg = f'--resume {latest_ckpt}' if latest_ckpt else ''

        script = _WATCHDOG_TEMPLATE.format(
            project_dir=self.config.project_dir,
            data_dir=self.config.data_dir,
            checkpoint_dir=self.config.checkpoint_dir,
            log_dir=self.config.log_dir,
            tb_dir=self.config.tb_dir,
            max_restarts=self.config.max_restarts,
            restart_delay=self.config.restart_delay,
            phase1_steps=self.config.phase1_steps,
            phase1_lr=self.config.phase1_lr,
            phase1_warmup=self.config.phase1_warmup,
            phase2_steps=self.config.phase2_steps,
            phase2_lr=self.config.phase2_lr,
            phase2_warmup=self.config.phase2_warmup,
            batch_size=self.config.batch_size,
            resume_arg=resume_arg,
        )

        path = self.config.watchdog_path
        with open(path, 'w') as f:
            f.write(script)
        os.chmod(path, 0o755)
        return path

    def start_tmux(self) -> bool:
        """创建 tmux 会话并启动看门狗和 TensorBoard。"""
        # 创建 tmux 会话
        r = _run(['tmux', 'new-session', '-d', '-s', 'chopinote',
                   '-x', '160', '-y', '50'])
        if r.returncode != 0:
            return False

        # 窗口 0: 训练
        env_cmds = (
            'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 '
            'NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1; '
            'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,roundup_power2_divisions:16,garbage_collection_threshold:0.6; '
            'export TORCH_CUDNN_V8_API_ENABLED=1 TOKENIZERS_PARALLELISM=false; '
            f'export CHOPINOTE_DATA_DIR={self.config.data_dir} '
            f'export CHOPINOTE_OUTPUT_DIR={self.config.checkpoint_dir} '
            f'export CHOPINOTE_LOG_DIR={self.config.log_dir} '
            f'export CHOPINOTE_TB_DIR={self.config.tb_dir}; '
            f'cd {self.config.project_dir}'
        )
        _run(['tmux', 'send-keys', '-t', 'chopinote:0', env_cmds, 'Enter'])
        time.sleep(0.3)
        _run(['tmux', 'send-keys', '-t', 'chopinote:0',
              f'bash {self.config.watchdog_path}', 'Enter'])

        # 窗口 1: TensorBoard
        _run(['tmux', 'new-window', '-t', 'chopinote', '-n', 'tensorboard'])
        time.sleep(0.3)
        _run(['tmux', 'send-keys', '-t', 'chopinote:1',
              f'tensorboard --logdir {self.config.tb_dir} --port 6006 --bind_all 2>&1 | '
              f'tee {self.config.log_dir}/tensorboard.log', 'Enter'])

        return True

    def wait_for_first_step(self, timeout: int = 300) -> bool:
        """等待训练输出第一步日志。"""
        crash_log = os.path.join(self.config.log_dir, 'crashes.log')
        # 先等模型加载（日志出现 "屏蔽 token 数" 或 "Step"）
        start = time.time()
        spinner = '◐◓◑◒'
        si = 0
        while time.time() - start < timeout:
            lines = _tail_lines(crash_log, 20)
            text = '\n'.join(lines)
            if 'Step ' in text and '| Loss:' in text:
                return True
            if '屏蔽 token 数' in text or '训练完成' in text:
                # 模型已加载但还没出 step log
                pass
            # 检查进程是否崩溃
            if 'Traceback' in text:
                return False
            si = (si + 1) % len(spinner)
            print(f'\r   等待第一步... {spinner[si]}  ({int(time.time()-start)}s)',
                  end='', flush=True)
            time.sleep(1)
        print()
        return False

    def launch(self, enter_monitor: bool = True) -> bool:
        """执行完整点火序列。"""
        self._step = 1

        # [1] 预检
        self._log('运行预检...', end='')
        checker = PreflightChecker(self.config)
        results = checker.run()
        all_pass = _print_checks(results)
        print()
        if not all_pass:
            fatal = [r for r in results if not r.passed and not r.is_warning]
            if fatal:
                self._log(f'{_RED}预检失败（{len(fatal)} 项严重错误），中止点火{_RESET}')
                return False

        # [2] 清理
        self._step = 2
        self._log('清理残留进程...', end='')
        killed = self.cleanup()
        print(f'\r  [2/7] 清理残留进程... ✅ 已清理 {killed} 个残留进程')

        # [3] 生成看门狗
        self._step = 3
        self._log('生成看门狗脚本...', end='')
        wd_path = self.generate_watchdog()
        print(f'\r  [3/7] 生成看门狗脚本... ✅ {wd_path}')

        # [4] 启动 tmux
        self._step = 4
        self._log('启动 tmux 会话...', end='')
        if not self.start_tmux():
            print(f'\r  [4/7] 启动 tmux 会话... {_RED}❌ 失败（tmux 是否已安装？）{_RESET}')
            return False
        print(f'\r  [4/7] 启动 tmux 会话... ✅')

        # 保存启动状态
        state = LaunchState(pid=os.getpid(),
                            launched_at=datetime.now().isoformat())
        state.save(self.config.state_file)

        # [5] 等待模型加载
        self._step = 5
        self._log('等待模型加载...', end='')
        crash_log = os.path.join(self.config.log_dir, 'crashes.log')
        start = time.time()
        found_model = False
        while time.time() - start < 60:
            for line in _tail_lines(crash_log, 20):
                if '参数量' in line or '屏蔽 token 数' in line:
                    found_model = True
                    break
            if found_model:
                break
            time.sleep(1)
        if found_model:
            print(f'\r  [5/7] 等待模型加载... ✅ ({int(time.time()-start)}s)')
        else:
            print(f'\r  [5/7] 等待模型加载... {_YELLOW}⚠️  超时（继续等待）{_RESET}')

        # [6] 等待第一步
        self._step = 6
        has_step = self.wait_for_first_step()
        print(f'\r  [6/7] 等待第一步... {"✅" if has_step else "⚠️  "}')

        # [7] 完成
        self._step = 7
        print(f'  [7/7] 点火完成！{_GREEN}训练已启动{_RESET}')
        print(f'  tmux: tmux attach -t chopinote:0')
        print(f'  TensorBoard: {self.config.tb_dir}')
        print(f'  Monitor: python scripts/launch_control.py monitor')

        if enter_monitor:
            print(f'\n{_CYAN}进入实时监控模式（Ctrl+C 退出）{_RESET}\n')
            try:
                Monitor(self.config).start()
            except KeyboardInterrupt:
                print(f'\n{_DIM}监控已退出，训练仍在后台运行{_RESET}')

        return True


# ── Monitor ───────────────────────────────────────────────────
class Monitor:
    """实时训练仪表盘。"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self._crash_log = os.path.join(config.log_dir, 'crashes.log')
        self._wd_log = os.path.join(config.log_dir, 'watchdog.log')
        self._prev_step = 0
        self._prev_time = time.time()
        self._cached_steps_pm = 0.0
        self._cached_tokens_ps = 0.0
        self._cached_eta_s = 0.0
        self._dashboard_height = 0

    def _collect(self) -> dict:
        """收集所有指标。"""
        gpu = _gpu_info()
        procs = _gpu_processes()
        train = _parse_training_progress(self._crash_log)
        wd = _parse_watchdog_status(self._wd_log)

        # 吞吐量计算（step 未推进时复用缓存值避免闪烁）
        steps_pm = self._cached_steps_pm
        tokens_ps = self._cached_tokens_ps
        eta_s = self._cached_eta_s
        if train['loss'] is not None and train['global_step'] > 0:
            now = time.time()
            if self._prev_step > 0 and train['global_step'] > self._prev_step:
                t_delta = now - self._prev_time
                s_delta = train['global_step'] - self._prev_step
                if t_delta > 0 and s_delta > 0:
                    steps_pm = s_delta / t_delta * 60
                    tokens_ps = s_delta * self.config.effective_batch_size * 4096 / t_delta
                    remaining = train['phase_total'] - train['global_step']
                    if steps_pm > 0:
                        eta_s = remaining / steps_pm * 60
                    self._cached_steps_pm = steps_pm
                    self._cached_tokens_ps = tokens_ps
                    self._cached_eta_s = eta_s
            self._prev_step = train['global_step']
            self._prev_time = now

        # Loss 趋势: 最近 5 个 loss 是上升还是下降
        trend = '—'
        history = train.get('loss_history', [])
        if len(history) >= 5:
            recent = history[-5:]
            if recent[-1] < recent[0] * 0.99:
                trend = f'{_GREEN}↓{_RESET}'
            elif recent[-1] > recent[0] * 1.01:
                trend = f'{_RED}↑{_RESET}'
            else:
                trend = f'{_DIM}→{_RESET}'

        # tmux 状态
        tmux_r = _run(['tmux', 'has-session', '-t', 'chopinote'])
        tmux_alive = tmux_r.returncode == 0

        # 磁盘
        disk = psutil.disk_usage(self.config.data_dir)
        mem = psutil.virtual_memory()

        return {
            'gpu': gpu,
            'gpu_processes': procs,
            'train': train,
            'watchdog': wd,
            'tmux_alive': tmux_alive,
            'throughput': {
                'steps_per_min': steps_pm,
                'tokens_per_sec': tokens_ps,
                'eta_seconds': eta_s,
            },
            'disk': {
                'used_gib': disk.used / 1024**3,
                'total_gib': disk.total / 1024**3,
                'percent': disk.percent,
            },
            'memory': {
                'used_gib': mem.used / 1024**3,
                'total_gib': mem.total / 1024**3,
                'percent': mem.percent,
            },
            'loss_trend': trend,
        }

    def _render(self, d: dict) -> str:
        """渲染仪表盘。"""
        g = d['gpu']
        t = d['train']
        w = d['watchdog']
        tp = d['throughput']
        disk = d['disk']
        mem = d['memory']

        box = TerminalBox(_BOX_WIDTH)

        # ── 标题行 ──────────────────────────────────
        status = (_GREEN + 'RUNNING' + _RESET) if d['tmux_alive'] else \
                 (_RED + 'STOPPED' + _RESET) if w['maxed_out'] else \
                 (_YELLOW + 'CRASHED' + _RESET)
        title = f'{_BOLD}CHOPINOTE-AI LAUNCH CONTROL ─── Status: {status}{_RESET}'
        lines = [box.top(title)]

        # ── 训练进度 ──────────────────────────────────
        phase_display = {'pretrain': 'Phase 1 预训练', 'finetune': 'Phase 2 微调'}.get(
            t['phase'], t['phase'] or '加载中...')

        if t['global_step'] > 0 and t['phase_total'] > 0:
            bar = _progress_bar(t['global_step'], t['phase_total'])
            lines.append(box.row(f'{_BOLD}{phase_display}{_RESET}'))
            lines.append(box.row(f'{bar}'))
        else:
            lines.append(box.row(f'{_BOLD}{phase_display}{_RESET}  {"◌ 等待第一步..." if d["tmux_alive"] else "—"}'))

        # ── 指标行 ──────────────────────────────────
        loss_str = f'{t["loss"]:.4f}' if t['loss'] is not None else '--'
        lr_str = f'{t["lr"]:.2e}' if t['lr'] is not None else '--'
        gn_str = f'{t["grad_norm"]:.2f}' if t['grad_norm'] is not None else '--'
        trend_str = d.get('loss_trend', '—')
        sec_str = f'  Sec: {t["sec_loss"]:.4f}' if t.get("sec_loss") else ''
        chord_str = f'  Chord: {t["chord_loss"]:.4f}' if t.get("chord_loss") else ''
        lines.append(box.row(f'Loss: {loss_str} {trend_str}{sec_str}{chord_str}    LR: {lr_str}    GN: {gn_str}'))

        # ── Validation ──────────────────────────
        if t['val_loss'] is not None:
            acc_parts = []
            for key in ('overall', 'note', 'duration', 'bar', 'dynamic', 'velocity', 'key', 'tempo',
                         'sec_bars', 'sec_keys', 'sec_types', 'chord_func', 'chord_inv'):
                val = t.get('val_acc', {}).get(key)
                if val is not None:
                    display_key = {'overall': 'all', 'duration': 'dur', 'velocity': 'vel',
                                   'sec_bars': 'secᴮ', 'sec_keys': 'secᴷ', 'sec_types': 'secᵀ',
                                   'chord_func': 'chordᶠ', 'chord_inv': 'chordᴵ'}.get(key, key)
                    acc_parts.append(f'{display_key} {val*100:.1f}%')
            acc_str = f'  ▏{"  ".join(acc_parts)}' if acc_parts else ''
            lines.append(box.row(f'Val Loss: {t["val_loss"]:.4f}  (step {t["val_loss_step"]:,}){acc_str}'))

        # ── GPU ──────────────────────────────────────
        if g:
            mem_str = f'{_fmt_size(g["mem_used_mib"]/1024)} / {_fmt_size(g["mem_total_mib"]/1024)}'
            power_str = f'{g["power_draw_w"]:.0f}/{g["power_limit_w"]:.0f}W'
            clock_str = f'{g["sm_clock_mhz"]}/{g["mem_clock_mhz"]}MHz'
            lines.append(box.hline('GPU'))
            lines.append(box.row(f'{g["name"]}  {mem_str}  {g["util_pct"]}%  {g["temp_c"]}°C'))
            lines.append(box.row(f'{power_str}  {clock_str}'))
        else:
            lines.append(box.hline('GPU'))
            lines.append(box.row('GPU: N/A'))

        # ── 吞吐量 ────────────────────────────────────
        lines.append(box.hline('Throughput'))
        if tp['steps_per_min'] > 0:
            step_str = f'{tp["steps_per_min"]:.1f} step/min'
            tok_str = f'{tp["tokens_per_sec"]:.0f} tok/s'
            eta = _eta_str(tp['eta_seconds'])
            # step time from log if available
            if t.get('step_time_s'):
                cumulative_h = t['step_time_s'] / 3600
                lines.append(box.row(f'{step_str}  {tok_str}  Elapsed: {cumulative_h:.1f}h  ETA: {eta}'))
            else:
                lines.append(box.row(f'{step_str}  {tok_str}  ETA: {eta}'))
        else:
            lines.append(box.row(f'{"等待数据..." if d["tmux_alive"] else "—"}'))

        # ── 看门狗 ────────────────────────────────────
        lines.append(box.hline('Watchdog'))
        wd_status = (_GREEN + 'ACTIVE' + _RESET) if d['tmux_alive'] and not w['maxed_out'] else \
                    (_RED + 'STOPPED' + _RESET) if w['maxed_out'] else \
                    (_YELLOW + 'WAITING' + _RESET)
        uptime_str = str(timedelta(seconds=int(w['uptime_seconds']))) if w['uptime_seconds'] > 0 else '--'
        crash_str = f'  Last crash: {w["last_crash_time"]}' if w['last_crash_time'] else ''
        lines.append(box.row(f'Status: {wd_status}  Restarts: {w["restarts"]}  Uptime: {uptime_str}{crash_str}'))

        # ── 系统 ──────────────────────────────────────
        lines.append(box.hline('System'))
        lines.append(box.row(f'Disk: {disk["used_gib"]:.0f}/{disk["total_gib"]:.0f}G ({disk["percent"]:.0f}%)  '
                             f'RAM: {mem["used_gib"]:.0f}/{mem["total_gib"]:.0f}G ({mem["percent"]:.0f}%)'))

        # ── 底栏 ──────────────────────────────────────
        now = datetime.now().strftime('%H:%M:%S')
        lines.append(box.row(f'{_DIM}[Ctrl+C detach]  Last: {now}{_RESET}'))
        lines.append(box.bottom())

        return '\n'.join(lines)

    def start(self):
        """进入监控主循环。"""
        # 确保终端可用
        if not sys.stdout.isatty():
            print('Monitor 模式需要终端')
            return

        print(_CLR, end='')
        no_data_cycles = 0

        while True:
            data = self._collect()
            rendered = self._render(data)

            # 首次渲染或更新
            if self._dashboard_height == 0:
                print(rendered)
                self._dashboard_height = rendered.count('\n') + 1
            else:
                # 光标回到顶部
                print(f'\033[{self._dashboard_height}A', end='')
                print(rendered)

            # 检查超时退出（无训练时）
            tmux_alive = data['tmux_alive']
            has_steps = data['train']['global_step'] > 0
            if not tmux_alive and not has_steps:
                no_data_cycles += 1
            else:
                no_data_cycles = 0

            if no_data_cycles > 30:  # 60s 无训练
                print(f'\n{_YELLOW}未检测到训练活动，监控退出。{_RESET}')
                break

            time.sleep(2)


# ── AbortController ──────────────────────────────────────────
class AbortController:
    """中止控制器。"""

    def __init__(self, config: TrainingConfig):
        self.config = config

    def abort(self, force: bool = False) -> bool:
        """中止所有训练相关进程。返回 True 如果有任何进程被清理。"""
        killed_anything = False

        if not force:
            print(f'{_YELLOW}将停止所有训练相关进程及 tmux 会话。{_RESET}')
            try:
                resp = input(f'{_BOLD}确认? [y/N] {_RESET}')
            except EOFError:
                return False
            resp = resp.lower().strip()
            if resp not in ('y', 'yes'):
                print('已取消')
                return False

        # 1. tmux 优雅停止
        print('  Stopping tmux...', end=' ', flush=True)
        _run(['tmux', 'send-keys', '-t', 'chopinote:0', 'C-c'], timeout=3)
        time.sleep(2)
        _run(['tmux', 'kill-session', '-t', 'chopinote'], timeout=5)
        print('✅')

        # 2. 杀掉残留 GPU 进程
        killed = 0
        for proc in _gpu_processes():
            if 'python' in proc['name'].lower():
                try:
                    os.kill(proc['pid'], signal.SIGKILL)
                    killed += 1
                except (ProcessLookupError, PermissionError):
                    pass
        if killed:
            killed_anything = True
            print(f'  Killed {killed} GPU process(es)')
        time.sleep(1)

        # 3. 清理临时文件
        for path in [self.config.lock_file, self.config.pid_file,
                     self.config.watchdog_path, self.config.state_file]:
            try:
                os.remove(path)
                print(f'  Removed {path}')
            except FileNotFoundError:
                pass

        # 4. CUDA 缓存
        try:
            import torch
            torch.cuda.empty_cache()
            print('  CUDA cache cleared')
        except (ImportError, RuntimeError):
            pass

        print(f'{_GREEN}已完成中止。{_RESET}')
        return killed_anything


# ── StatusReporter ────────────────────────────────────────────
class StatusReporter:
    """状态快照。"""

    def __init__(self, config: TrainingConfig):
        self.config = config

    def report(self):
        """打印单次状态报告。"""
        crash_log = os.path.join(self.config.log_dir, 'crashes.log')
        wd_log = os.path.join(self.config.log_dir, 'watchdog.log')

        tmux_r = _run(['tmux', 'has-session', '-t', 'chopinote'])
        tmux_alive = tmux_r.returncode == 0

        gpu = _gpu_info()
        train = _parse_training_progress(crash_log)
        wd = _parse_watchdog_status(wd_log)

        print()
        print(f'{_BOLD}Chopinote-AI 训练状态{_RESET}')
        print(f'{"═" * 50}')

        # 状态
        if tmux_alive:
            print(f'  Status:   {_GREEN}RUNNING{_RESET}  (tmux: chopinote)')
        elif wd['maxed_out']:
            print(f'  Status:   {_RED}STOPPED (看门狗已达最大重启次数){_RESET}')
        else:
            print(f'  Status:   {_YELLOW}STOPPED{_RESET}')

        # 训练
        if train['global_step'] > 0:
            phase_name = train['phase'] or '?'
            print(f'  Phase:    {phase_name}  (step {train["global_step"]:,} / {train["phase_total"]:,})')
            gn_str = f'  GN: {train["grad_norm"]:.2f}' if train['grad_norm'] else ''
            sec_str = f'  Sec: {train["sec_loss"]:.4f}' if train.get("sec_loss") else ''
            chord_str = f'  Chord: {train["chord_loss"]:.4f}' if train.get("chord_loss") else ''
            print(f'  Loss:     {train["loss"]:.4f}    LR: {train["lr"]:.2e}{sec_str}{chord_str}{gn_str}')
            if train['val_loss'] is not None:
                acc_parts = []
                for key in ('overall', 'note', 'duration', 'bar', 'dynamic', 'velocity', 'key', 'tempo',
                         'sec_bars', 'sec_keys', 'sec_types', 'chord_func', 'chord_inv'):
                    val = train.get('val_acc', {}).get(key)
                    if val is not None:
                        display_key = {'overall': 'all', 'duration': 'dur', 'velocity': 'vel',
                                       'sec_bars': 'secB', 'sec_keys': 'secK', 'sec_types': 'secT',
                                       'chord_func': 'chordF', 'chord_inv': 'chordI'}.get(key, key)
                        acc_parts.append(f'{display_key}={val*100:.1f}%')
                acc_str = f'  {"  ".join(acc_parts)}' if acc_parts else ''
                print(f'  Val Loss: {train["val_loss"]:.4f}  (step {train["val_loss_step"]:,})')
                if acc_parts:
                    print(f'  Val Acc:  {acc_str}')
        else:
            print(f'  Phase:    {"—" if not tmux_alive else "加载中..."}')

        # GPU
        if gpu:
            mem_str = f'{_fmt_size(gpu["mem_used_mib"]/1024)} / {_fmt_size(gpu["mem_total_mib"]/1024)}'
            power_str = f'{gpu["power_draw_w"]:.0f}/{gpu["power_limit_w"]:.0f}W'
            clock_str = f'{gpu["sm_clock_mhz"]}/{gpu["mem_clock_mhz"]}MHz'
            print(f'  GPU:      {gpu["name"]}  |  {mem_str}  |  {gpu["util_pct"]}%  |  {gpu["temp_c"]}°C')
            print(f'  GPU Power:{power_str}  Clocks: {clock_str}')

        # 看门狗
        wd_status = 'ACTIVE' if tmux_alive else 'STOPPED'
        uptime = str(timedelta(seconds=int(wd['uptime_seconds']))) if wd['uptime_seconds'] > 0 else '--'
        print(f'  Watchdog: {wd_status}  |  Restarts: {wd["restarts"]}  |  Uptime: {uptime}')
        if wd['last_crash_time']:
            print(f'  Last crash: {wd["last_crash_time"]}')

        # 磁盘/内存
        try:
            disk = psutil.disk_usage(self.config.data_dir)
            mem = psutil.virtual_memory()
            print(f'  Disk:     {disk.used/1024**3:.0f}/{disk.total/1024**3:.0f} GiB ({disk.percent:.0f}%)')
            print(f'  RAM:      {mem.used/1024**3:.0f}/{mem.total/1024**3:.0f} GiB ({mem.percent:.0f}%)')
        except OSError:
            pass

        # 最近的日志行
        for line in _tail_lines(crash_log, 5):
            if 'Step ' in line and '| Loss:' in line:
                print(f'  Recent:   {line.strip()}')

        print()
        print(f'{_DIM}命令:{_RESET}')
        print(f'  python scripts/launch_control.py monitor  — 实时仪表盘')
        print(f'  python scripts/launch_control.py abort     — 停止训练')
        print(f'  tmux attach -t chopinote:0                — 训练窗口')
        print(f'  tmux attach -t chopinote:1                — TensorBoard')
        print()


# ── 主入口 ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Chopinote-AI 点火控制台',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    sub = parser.add_subparsers(dest='command', required=True)

    # check
    sub.add_parser('check', help='运行预检清单')

    # launch
    p_launch = sub.add_parser('launch', help='全流程点火')
    p_launch.add_argument('--no-monitor', action='store_true',
                          help='点火后不进入监控模式')
    p_launch.add_argument('--phase1-steps', type=int)
    p_launch.add_argument('--phase1-lr', type=float)
    p_launch.add_argument('--phase1-warmup', type=int)
    p_launch.add_argument('--phase2-steps', type=int)
    p_launch.add_argument('--phase2-lr', type=float)
    p_launch.add_argument('--phase2-warmup', type=int)
    p_launch.add_argument('--batch-size', type=int)
    p_launch.add_argument('--grad_accum', type=int)
    p_launch.add_argument('--resume', type=str, default=None,
                          help='从指定 checkpoint 恢复（默认自动检测最新）')

    # monitor
    sub.add_parser('monitor', help='实时训练仪表盘')

    # abort
    p_abort = sub.add_parser('abort', help='中止一切')
    p_abort.add_argument('--force', '-f', action='store_true',
                         help='跳过确认')

    # status
    p_status = sub.add_parser('status', help='训练状态快照')
    p_status.add_argument('--watch', '-w', action='store_true',
                          help='持续监控（类似 top）')

    args = parser.parse_args()

    # 加载配置
    config = TrainingConfig()
    config.apply_args(args)

    if args.command == 'check':
        checker = PreflightChecker(config)
        results = checker.run()
        print()
        _print_checks(results)

    elif args.command == 'launch':
        # 先检查是否有已在运行的训练
        tmux_r = _run(['tmux', 'has-session', '-t', 'chopinote'])
        if tmux_r.returncode == 0:
            resp = input('已有 tmux 会话在运行。重启将中止当前训练。确认? [y/N] ').lower().strip()
            if resp not in ('y', 'yes'):
                print('已取消')
                return

        controller = LaunchController(config)
        controller.launch(enter_monitor=not args.no_monitor)

    elif args.command == 'monitor':
        if not sys.stdout.isatty():
            print('Monitor 模式需要终端')
            return
        print(f'{_CLR}', end='')
        try:
            Monitor(config).start()
        except KeyboardInterrupt:
            print(f'\n{_DIM}监控已退出{_RESET}')

    elif args.command == 'abort':
        AbortController(config).abort(force=args.force)

    elif args.command == 'status':
        reporter = StatusReporter(config)
        reporter.report()
        if args.watch:
            print(f'{_DIM}按 Ctrl+C 退出监控模式{_RESET}')
            try:
                while True:
                    time.sleep(5)
                    reporter.report()
            except KeyboardInterrupt:
                pass


if __name__ == '__main__':
    main()
