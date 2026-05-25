"""
配置文件加载与校验模块。

提供 Config dataclass + load_config() 自动搜索 + validate() 范围检查。

优先级链: CLI 参数 > --preset > 配置文件 > 内置默认值
"""

from dataclasses import dataclass, fields, asdict
from pathlib import Path
from typing import Optional

import types
import typing
import yaml


@dataclass
class Config:
    """所有可配置的生成参数。默认值从 YAML 文件读取，dataclass 字段为内置后备。"""

    # 核心采样
    temperature: float = 1.0
    top_k: int = 20
    max_bars: int = 32
    complexity: int = 5

    # 锁定模式
    lock_key: bool = True
    lock_time: bool = True
    lock_tempo: bool = True
    lock_program: bool = True

    # 高级约束
    rest_penalty: float = 0.0
    max_polyphony: int = 10
    key_bias_strength: float = 2.0
    prog_switch_strength: float = 1.0
    prog_switch_interval: int = 12

    # 段落感知
    section_aware: bool = False
    section_form: str = 'sonata'
    section_total_bars: int = 64

    # 其他
    seed_bars: int = 16
    num_samples: int = 1
    target_key: Optional[str] = None
    random_seed: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'Config':
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    def validate(self) -> list[str]:
        """校验数值范围与字段类型，返回错误列表（空列表 = 全部通过）。"""
        errors: list[str] = []

        # (min, max) 范围约束
        ranges: dict[str, tuple[float, float]] = {
            'temperature': (0.1, 2.5),
            'top_k': (1, 100),
            'max_bars': (1, 256),
            'complexity': (0, 10),
            'rest_penalty': (0.0, 10.0),
            'max_polyphony': (1, 20),
            'key_bias_strength': (0.0, 5.0),
            'prog_switch_strength': (0.0, 5.0),
            'prog_switch_interval': (1, 128),
            'seed_bars': (1, 128),
            'num_samples': (1, 10),
            'section_total_bars': (16, 256),
        }
        valid_forms = {'sonata', 'rondo', 'aba', 'theme-variations', 'binary'}

        for field in fields(self.__class__):
            val = getattr(self, field.name)
            if val is None:
                continue

            # ── 类型校验 ──
            expected_type = field.type
            # 解 Optional[X] / Union[X, None] → X
            origin = getattr(expected_type, '__origin__', None)
            if origin is typing.Union or origin is types.UnionType:
                inner_types = expected_type.__args__
                if not any(isinstance(val, t) for t in inner_types if t is not type(None)):
                    type_names = ' / '.join(
                        t.__name__ for t in inner_types if t is not type(None)
                    )
                    errors.append(
                        f'{field.name}: 应为 {type_names}，收到 {type(val).__name__} ({val!r})'
                    )
            elif not isinstance(val, expected_type):
                errors.append(
                    f'{field.name}: 应为 {expected_type.__name__}，'
                    f'收到 {type(val).__name__} ({val!r})'
                )

            # ── 范围校验 ──
            if field.name in ranges and isinstance(val, (int, float)):
                lo, hi = ranges[field.name]
                if not (lo <= val <= hi):
                    errors.append(
                        f'{field.name}: 值 {val} 超出范围 [{lo}, {hi}]'
                    )

            # ── 曲式校验 ──
            if field.name == 'section_form' and isinstance(val, str):
                if val not in valid_forms:
                    errors.append(
                        f'section_form: "{val}" 不是有效曲式'
                        f' ({", ".join(sorted(valid_forms))})'
                    )

            # ── 目标调性校验 ──
            if field.name == 'target_key' and isinstance(val, str):
                from chopinote_dataset.tokenizer import REMITokenizer

                if val not in REMITokenizer.KEY_NAMES:
                    errors.append(
                        f'target_key: "{val}" 不是有效调性'
                    )

        return errors


def find_config(path: str | None = None) -> str | None:
    """自动检测配置文件路径。

    搜索优先级：
        1. 用户显式指定的 path
        2. 当前目录 ./chopinote_config.{yaml,yml}
        3. ~/.chopinote/config.{yaml,yml}
        4. 包内置 generation_config.yaml
    """
    if path:
        return path if Path(path).is_file() else None

    candidates = [
        Path.cwd() / 'chopinote_config.yaml',
        Path.cwd() / 'chopinote_config.yml',
        Path.home() / '.chopinote' / 'config.yaml',
        Path.home() / '.chopinote' / 'config.yml',
    ]
    for c in candidates:
        if c.is_file():
            return str(c)

    # 内置默认配置（包目录下）
    builtin = Path(__file__).resolve().parent / 'generation_config.yaml'
    return str(builtin) if builtin.is_file() else None


def load_config(path: str | None = None) -> Config:
    """加载配置文件 → Config。

    找不到文件时返回全默认 Config，不会报错。

    Raises:
        ValueError: 配置文件存在但校验失败（类型错误 / 范围越界）
    """
    resolved = find_config(path)
    if resolved is None:
        return Config()

    with open(resolved, encoding='utf-8') as f:
        content = f.read()

    data = yaml.safe_load(content) or {}
    config = Config.from_dict(data)

    errors = config.validate()
    if errors:
        raise ValueError(
            f'配置文件 {resolved} 校验失败:\n'
            + '\n'.join(f'  - {e}' for e in errors)
        )

    return config
