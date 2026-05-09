"""共享 fixtures。"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chopinote_dataset.tokenizer import REMITokenizer


@pytest.fixture(scope="module")
def tokenizer():
    return REMITokenizer(grid_size=16, velocity_levels=8)
