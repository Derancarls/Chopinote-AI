#!/usr/bin/env python3
"""三格式预处理流水线 (v3)：PDMX → MusicXML → MIDI → 重建分词

后台运行: nohup python scripts/rerun_all_v3.py > /root/autodl-tmp/preprocess_v3.log 2>&1 &
"""
import sys, os, time, logging, subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LOG_FILE = '/root/autodl-tmp/preprocess_v3.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

STAGE_SCRIPTS = [
    ('PDMX',    'scripts/rerun_pdmx.py'),
    ('MusicXML','scripts/rerun_musicxml.py'),
    ('MIDI',    'scripts/run_fast_preprocess.py'),
]

START = time.time()

for name, script in STAGE_SCRIPTS:
    logger.info(f"{'='*60}")
    logger.info(f"  阶段: {name} ({script})")
    logger.info(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script],
        cwd='/root/Chopinote-AI',
        capture_output=True, text=True,
        timeout=86400,  # 24h
    )
    elapsed = time.time() - t0
    logger.info(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.returncode != 0:
        logger.error(f"{name} 失败 (rc={result.returncode}): {result.stderr[-1000:]}")
    else:
        logger.info(f"{name} 完成 ({elapsed/60:.0f}min)")

TOTAL = time.time() - START
logger.info(f"\n{'='*60}")
logger.info(f"  全流程完成 ({TOTAL/60:.0f}min)")
logger.info(f"{'='*60}")
