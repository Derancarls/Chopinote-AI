"""
PDMX JSON → REMI tokens 预处理 + 数据集划分 CLI。

用法:
    # 完整处理所有 PDMX 数据
    python scripts/preprocess_pdmx.py

    # 指定输入/输出目录
    python scripts/preprocess_pdmx.py \
        --input-dir data/raw/pdmx_extracted/PDMX \
        --output-dir data/processed

    # 小批量测试（仅处理前 N 个文件）
    python scripts/preprocess_pdmx.py --max-files 100
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chopinote_dataset.processor import PDMXPreprocessor
from chopinote_dataset.splitter import split_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
)
logger = logging.getLogger(__name__)


def generate_all_files(token_dir: str, output_dir: str) -> str:
    """遍历 token 目录，生成 all_files.txt 文件列表。"""
    token_path = Path(token_dir)
    files = sorted(token_path.glob("*.tokens"))
    all_files_path = Path(output_dir) / "all_files.txt"
    with open(all_files_path, 'w', encoding='utf-8') as f:
        for fp in files:
            f.write(str(fp) + '\n')
    logger.info(f"生成文件列表: {all_files_path} ({len(files)} 个文件)")
    return str(all_files_path)


def main():
    parser = argparse.ArgumentParser(description='PDMX 数据预处理 + 数据集划分')
    parser.add_argument('--input-dir', default='data/raw/pdmx_extracted/PDMX',
                        help='PDMX JSON 数据目录')
    parser.add_argument('--output-dir', default='data/processed',
                        help='输出目录（processed_dir）')
    parser.add_argument('--config', default='config.yaml',
                        help='配置文件路径')
    parser.add_argument('--max-files', type=int, default=None,
                        help='最大处理文件数（测试用）')
    args = parser.parse_args()

    start_time = time.time()
    logger.info(f"PDMX 输入目录: {args.input_dir}")
    logger.info(f"输出目录: {args.output_dir}")

    # ── 1. PDMX → REMI tokens ──────────────────────────────
    preprocessor = PDMXPreprocessor(config_path=args.config)
    processed, failed = preprocessor.process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )

    if not processed:
        logger.error("没有成功处理任何文件，请检查数据目录和配置")
        sys.exit(1)

    logger.info(f"处理完成: {len(processed)} 成功, {len(failed)} 失败")

    # ── 2. 限制数量后的截断 ──────────────────────────────
    if args.max_files and len(processed) > args.max_files:
        token_dir = Path(args.output_dir) / "tokens"
        kept = set()
        for p in processed[:args.max_files]:
            kept.add(p['file_id'])
        # 删除超出 max_files 的 token 文件
        for p in processed[args.max_files:]:
            tp = Path(p['token_path'])
            if tp.exists():
                tp.unlink()
            mp = Path(p['metadata_path'])
            if mp.exists():
                mp.unlink()
        processed = processed[:args.max_files]
        logger.info(f"已截断为前 {args.max_files} 个文件")

    # ── 3. 生成 all_files.txt ──────────────────────────────
    token_dir = str(Path(args.output_dir) / "tokens")
    all_files_path = generate_all_files(token_dir, args.output_dir)

    # ── 4. 数据集划分 ──────────────────────────────────────
    split_dataset(
        file_list_path=all_files_path,
        output_dir=args.output_dir,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=42,
    )

    elapsed = time.time() - start_time
    logger.info(f"全部完成！耗时 {elapsed:.1f} 秒")
    logger.info(f"  train.txt/val.txt/test.txt 已生成至 {args.output_dir}/")


if __name__ == '__main__':
    main()
