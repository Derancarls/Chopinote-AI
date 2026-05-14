"""ANSI 颜色和终端渲染工具。"""
import re
from typing import Optional

# ── ANSI 颜色 ─────────────────────────────────────────────────
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
BOLD = '\033[1m'
DIM = '\033[2m'
RESET = '\033[0m'

# ── ANSI 控制序列 ───────────────────────────────────────────
CLR = '\033[H\033[J'
CLR_EOL = '\033[K'
CURSOR_UP = '\033[1A'
SAVE = '\033[s'
RESTORE = '\033[u'

# ── 组合 ────────────────────────────────────────────────────
CHECK_PASS = f'{GREEN}OK{RESET}'
CHECK_FAIL = f'{RED}FAIL{RESET}'
CHECK_WARN = f'{YELLOW}WARN{RESET}'

_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')


def visible_len(s: str) -> int:
    """返回去除 ANSI 转义码后的可见字符数。"""
    return len(_ANSI_RE.sub('', s))


def pad_line(content: str, width: int = 62) -> str:
    """在内容后补空格至目标可见宽度，确保右侧边框对齐。"""
    vlen = visible_len(content)
    return content + ' ' * max(0, width - vlen)


class TerminalBox:
    """ASCII 边框渲染器。"""

    def __init__(self, width: int = 62):
        self.width = width

    def row(self, inner: str) -> str:
        return pad_line(f'│ {inner}', self.width) + ' │'

    def hline(self, label: str = '') -> str:
        if label:
            return self.row(f'{DIM}── {label} ──{RESET}')
        return self.row('─' * self.width)

    def top(self, title: str) -> str:
        return pad_line(f'┌─ {title}', self.width) + ' ─┐'

    def bottom(self) -> str:
        return pad_line('└' + '─' * (self.width + 2), self.width) + '┘'

    @staticmethod
    def progress_bar(filled: int, total: int, width: int = 30) -> str:
        """绘制进度条 [████░░░░] 形式。"""
        if total <= 0:
            return '[░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]'
        ratio = min(1.0, max(0.0, filled / total))
        done = int(round(ratio * width))
        return '[' + '█' * done + '░' * (width - done) + ']'
