"""
MusicXML/MXL → REMI tokens 预处理 CLI。

处理 ASAP、ATEPP、openscore 等 MusicXML 数据集，
输出与 MIDI/PDMX 兼容的 token 和元数据文件。

用法:
    python scripts/preprocess_musicxml.py \\
        --input-dirs /root/autodl-tmp/asap /root/autodl-tmp/ATEPP-1.2 \\
        --output-dir /root/autodl-tmp/data/processed
"""
import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chopinote_dataset.processor import MusicXMLPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='MusicXML 数据预处理')
    parser.add_argument('--input-dirs', nargs='+', required=True,
                        help='MusicXML 数据目录列表')
    parser.add_argument('--output-dir', default='/root/autodl-tmp/data/processed',
                        help='输出目录（tokens/ 和 metadata/ 所在目录）')
    parser.add_argument('--config', default='config.yaml',
                        help='配置文件路径')
    args = parser.parse_args()

    start_time = time.time()
    logger.info(f"MusicXML 输入目录: {args.input_dirs}")
    logger.info(f"输出目录: {args.output_dir}")

    preprocessor = MusicXMLPreprocessor(config_path=args.config)

    total_processed = 0
    total_failed = 0

    for input_dir in args.input_dirs:
        if not Path(input_dir).is_dir():
            logger.warning(f"目录不存在，跳过: {input_dir}")
            continue

        logger.info(f"正在处理: {input_dir}")
        t0 = time.time()
        processed, failed = preprocessor.process_directory(
            input_dir=input_dir,
            output_dir=args.output_dir,
        )
        elapsed = time.time() - t0
        total_processed += len(processed)
        total_failed += len(failed)
        logger.info(f"  {elapsed:.1f}s — {len(processed)} 成功, {len(failed)} 失败")

    elapsed = time.time() - start_time
    logger.info(f"MusicXML 全部完成！耗时 {elapsed:.1f}s")
    logger.info(f"总计: {total_processed} 成功, {total_failed} 失败")


if __name__ == '__main__':
    main()
