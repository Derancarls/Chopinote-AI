#!/usr/bin/env python3
"""Single worker: processes a batch of token files for section annotation."""
import sys, os, argparse, logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-list', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    with open(args.file_list) as f:
        files = [l.strip() for l in f if l.strip()]

    from scripts.structure_annotator import annotate_file

    done = 0
    for fp in files:
        try:
            r = annotate_file(fp, args.output_dir)
            done += 1
        except Exception as e:
            pass
    logger.info(f"Worker done: {done}/{len(files)}")


if __name__ == '__main__':
    main()
