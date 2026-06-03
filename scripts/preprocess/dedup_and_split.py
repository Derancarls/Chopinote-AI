"""
去重 + 重新划分 train/val/test
按 source hash_md5 去重，保留 token 数最大的副本。
"""
import json, os, logging, time, random
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = '/root/autodl-tmp/data/processed'
TOKEN_DIR = f'{DATA_DIR}/tokens_v4'
META_DIR = f'{DATA_DIR}/metadata_v4'

# ── 1. 按 hash_md5 分组 ────────────────────────────────────
logger.info("扫描 metadata 文件...")
hash_groups = defaultdict(list)  # hash -> [(meta_path, token_path, num_tokens)]

total = 0
for fname in os.listdir(META_DIR):
    if not fname.endswith('.meta.json'):
        continue
    meta_path = os.path.join(META_DIR, fname)
    with open(meta_path) as f:
        md = json.load(f)
    h = md.get('hash_md5', '')
    tid = md.get('file_id', '')
    token_path = os.path.join(TOKEN_DIR, f'{tid}.tokens')
    if not os.path.exists(token_path):
        continue
    hash_groups[h].append((meta_path, token_path, md.get('num_tokens', 0)))
    total += 1

logger.info(f"总文件: {total}, 唯一 hash: {len(hash_groups)}")

# ── 2. 删除重复 ────────────────────────────────────────────
dup_groups = {h: v for h, v in hash_groups.items() if len(v) > 1}
logger.info(f"重复 hash 组: {len(dup_groups)}")

removed_meta = 0
removed_token = 0
for h, entries in dup_groups.items():
    # 按 num_tokens 降序，保留第一个（最长）
    entries.sort(key=lambda x: -x[2])
    keep = entries[0]
    for meta_path, token_path, _ in entries[1:]:
        try:
            os.remove(meta_path)
            removed_meta += 1
        except OSError:
            pass
        try:
            os.remove(token_path)
            removed_token += 1
        except OSError:
            pass

logger.info(f"删除: {removed_meta} meta, {removed_token} token")

# ── 3. 统计最终文件数 ──────────────────────────────────────
remaining_tokens = sorted(f for f in os.listdir(TOKEN_DIR) if f.endswith('.tokens'))
remaining_meta = sorted(f for f in os.listdir(META_DIR) if f.endswith('.meta.json'))
logger.info(f"去重后: {len(remaining_tokens)} tokens, {len(remaining_meta)} metadata")

# ── 4. 重新划分 train/val/test ─────────────────────────────
logger.info("重新生成划分...")
random.seed(42)
shuffled = [os.path.join(TOKEN_DIR, f) for f in remaining_tokens]
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
    path = os.path.join(DATA_DIR, name)
    with open(path, 'w', encoding='utf-8') as f:
        for fp in files:
            f.write(fp + '\n')
    logger.info(f"  {name}: {len(files)} 文件")

logger.info(f"划分完成: train={n_train}, val={n_val}, test={n - n_train - n_val}")
