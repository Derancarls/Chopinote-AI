"""Generate token_lengths.json from metadata_v2 files — required by TokenDataset."""
import json
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = '/root/autodl-tmp/data/processed'
META_DIR = os.path.join(DATA_DIR, 'metadata_v3')
OUTPUT = os.path.join(DATA_DIR, 'token_lengths.json')


def main():
    logger.info("Scanning %s ...", META_DIR)
    lengths = {}
    total = 0
    for fname in os.listdir(META_DIR):
        if not fname.endswith('.meta.json'):
            continue
        stem = fname[:-10]  # remove ".meta.json"
        try:
            with open(os.path.join(META_DIR, fname)) as f:
                md = json.load(f)
            lengths[stem] = md.get('num_tokens', 0)
            total += 1
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read %s: %s", fname, e)

    logger.info("Writing %d entries to %s", total, OUTPUT)
    with open(OUTPUT, 'w') as f:
        json.dump(lengths, f)
    logger.info("Done. Total tokens across all files: %d", sum(lengths.values()))


if __name__ == '__main__':
    main()
