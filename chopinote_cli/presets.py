"""预设模板系统与条件注入工具。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Preset:
    """生成预设模板，捆绑所有风格相关的生成参数。"""
    name: str
    label: str
    description: str

    # 生成参数
    program: Optional[int] = None   # GM 乐器编号，None=不强制
    complexity: int = 5
    temperature: float = 1.0
    top_k: int = 20

    # 锁定模式
    lock_key: bool = True
    lock_time: bool = True
    lock_tempo: bool = True

    # 条件（以控制 token 注入 seed 前）
    condition_key: Optional[str] = None    # 'C', 'Am', 'G' ...
    condition_time: Optional[str] = None   # '4/4', '3/4', '6/8' ...
    condition_tempo: Optional[int] = None  # 30-240

    def attrs(self) -> dict:
        """返回可覆盖 CLI 默认值的生成参数。"""
        return {
            'program': self.program,
            'complexity': self.complexity,
            'temperature': self.temperature,
            'top_k': self.top_k,
            'lock_key': self.lock_key,
            'lock_time': self.lock_time,
            'lock_tempo': self.lock_tempo,
        }

    def conditions(self) -> dict:
        """返回条件字典：'key'/'time'/'tempo'/'program' → 值。"""
        d = {}
        if self.condition_key is not None:
            d['key'] = self.condition_key
        if self.condition_time is not None:
            d['time'] = self.condition_time
        if self.condition_tempo is not None:
            d['tempo'] = self.condition_tempo
        if self.program is not None:
            d['program'] = self.program
        return d


# ── 内置预设 ──────────────────────────────────────────────────

BUILTIN_PRESETS: dict[str, Preset] = {}


def _register(p: Preset):
    BUILTIN_PRESETS[p.name] = p


_register(Preset(
    'default', '默认', '默认设置',
))

_register(Preset(
    'baroque', '巴洛克', '羽管键琴音色，复调织体',
    program=6, complexity=6, temperature=0.9, top_k=25,
    lock_key=True, lock_time=True, lock_tempo=True,
))

_register(Preset(
    'romantic', '浪漫派', '钢琴，丰富表情，自由速度',
    program=0, complexity=7, temperature=1.1, top_k=30,
    lock_key=True, lock_time=True, lock_tempo=False,
))

_register(Preset(
    'classical', '古典派', '钢琴，均衡结构',
    program=0, complexity=5, temperature=1.0, top_k=20,
    lock_key=True, lock_time=True, lock_tempo=True,
))

_register(Preset(
    'minimal', '极简', '稀疏织体，慢速',
    program=0, complexity=2, temperature=0.8, top_k=15,
    lock_key=True, lock_time=True, lock_tempo=True,
    condition_tempo=60,
))

_register(Preset(
    'jazz', '爵士', '电钢琴，即兴风格',
    program=5, complexity=6, temperature=1.2, top_k=35,
    lock_key=False, lock_time=True, lock_tempo=False,
))

_register(Preset(
    'church', '管风琴', '管风琴音色，庄重',
    program=19, complexity=4, temperature=0.9, top_k=20,
    lock_key=True, lock_time=True, lock_tempo=True,
))


def get_preset(name: str) -> Optional[Preset]:
    return BUILTIN_PRESETS.get(name)


def list_presets() -> list[Preset]:
    return list(BUILTIN_PRESETS.values())
