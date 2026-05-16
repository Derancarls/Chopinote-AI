"""
MIDI 跑完后执行：
1. 删除旧的 PDMX tokens (815 vocab，Qm 前缀)
2. 用当前 tokenizer 重新处理 PDMX，输出到 autodl-tmp
3. 合并划分数据集
"""
import sys
import os
import time
import logging
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# 目标目录（与 MIDI 预处理一致）
DATA_DISK = '/root/autodl-tmp'
PROCESSED_DIR = os.path.join(DATA_DISK, 'data', 'processed')
TOKEN_DIR = os.path.join(PROCESSED_DIR, 'tokens')
META_DIR = os.path.join(PROCESSED_DIR, 'metadata')

# 旧 PDMX 目录
OLD_TOKEN_DIR = '/root/Chopinote-AI/data/processed/tokens'
OLD_META_DIR = '/root/Chopinote-AI/data/processed/metadata'


def step1_clean_old_pdmx():
    """删除旧的 PDMX token 文件（Qm 前缀）和 metadata。"""
    logger.info("=== Step 1: 清理旧 PDMX tokens ===")

    if os.path.isdir(OLD_TOKEN_DIR):
        old_tokens = [f for f in os.listdir(OLD_TOKEN_DIR) if f.startswith('Qm')]
        for f in old_tokens:
            os.remove(os.path.join(OLD_TOKEN_DIR, f))
        logger.info(f"  删除 {len(old_tokens)} 个旧 token 文件")

    if os.path.isdir(OLD_META_DIR):
        old_meta = [f for f in os.listdir(OLD_META_DIR) if f.startswith('Qm')]
        for f in old_meta:
            os.remove(os.path.join(OLD_META_DIR, f))
        logger.info(f"  删除 {len(old_meta)} 个旧 meta 文件")


def step2_reprocess_pdmx():
    """用当前 tokenizer 重新处理 PDMX。"""
    logger.info("=== Step 2: 重新处理 PDMX → REMI tokens ===")
    from chopinote_dataset.processor import PDMXPreprocessor

    input_dir = '/root/autodl-tmp/chopinote_data/data/raw/PDMX'
    preprocessor = PDMXPreprocessor(config_path='/root/Chopinote-AI/config.yaml')
    processed, failed = preprocessor.process_directory(
        input_dir=input_dir,
        output_dir=PROCESSED_DIR,
    )
    logger.info(f"  PDMX 处理完成: {len(processed)} 成功, {len(failed)} 失败")
    return processed, failed


def step3_regenerate_splits():
    """合并所有 tokens，重新生成 train/val/test 划分。"""
    logger.info("=== Step 3: 重新生成数据集划分 ===")

    token_files = sorted(f for f in os.listdir(TOKEN_DIR) if f.endswith('.tokens'))
    all_files = [os.path.join(TOKEN_DIR, f) for f in token_files]
    logger.info(f"  总 token 文件数: {len(all_files)}")

    import random
    random.seed(42)
    shuffled = list(all_files)
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    splits = {
        'train.txt': shuffled[:n_train],
        'val.txt': shuffled[n_train:n_train + n_val],
        'test.txt': shuffled[n_train + n_val:],
    }

    for name, files in splits.items():
        path = os.path.join(PROCESSED_DIR, name)
        with open(path, 'w', encoding='utf-8') as f:
            for fp in files:
                rel = os.path.relpath(fp, os.path.join(DATA_DISK, 'data'))
                f.write(f'data/processed/{rel}\n')
        logger.info(f"  {name}: {len(files)} 文件")

    # 更新 processing_stats.json
    stats_path = os.path.join(PROCESSED_DIR, 'processing_stats.json')
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
    else:
        stats = {}
    stats['total_files'] = n
    stats['train'] = n_train
    stats['val'] = n_val
    stats['test'] = n - n_train - n_val
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"  划分完成: train={n_train}, val={n_val}, test={n - n_train - n_val}")


def step4_sync_project():
    """确保项目目录能访问到数据（软链）。"""
    logger.info("=== Step 4: 同步项目数据路径 ===")
    project_data = '/root/Chopinote-AI/data/processed'

    # 如果项目目录目前指向 autodl-tmp 或者为空，就做软链
    if os.path.islink(project_data):
        os.unlink(project_data)
        logger.info(f"  移除旧软链 {project_data}")

    # 创建软链 project/data/processed → autodl-tmp/data/processed
    target = os.path.relpath(PROCESSED_DIR, os.path.dirname(project_data))
    # 不能用相对路径跨文件系统，用绝对路径
    os.symlink(PROCESSED_DIR, project_data)
    logger.info(f"  创建软链: {project_data} → {PROCESSED_DIR}")


def wait_for_midi_done():
    """等待 MIDI 预处理完成。"""
    lock_path = os.path.join(PROCESSED_DIR, 'preprocess.lock')
    while True:
        if not os.path.exists(lock_path):
            logger.info("  MIDI 预处理锁已释放，进程已结束")
            return
        try:
            with open(lock_path) as f:
                pid = int(f.read().strip())
            # 检查进程是否存活
            os.kill(pid, 0)
            logger.info(f"  MIDI 预处理仍在运行 (PID {pid})，等待 60 秒...")
            time.sleep(60)
        except (ValueError, OSError, ProcessLookupError):
            logger.info("  MIDI 预处理已结束")
            return


if __name__ == '__main__':
    start = time.time()

    # 等待 MIDI 跑完
    # wait_for_midi_done()

    step1_clean_old_pdmx()
    step2_reprocess_pdmx()
    step3_regenerate_splits()
    step4_sync_project()

    elapsed = time.time() - start
    logger.info(f"全部完成！耗时 {elapsed:.1f} 秒")
