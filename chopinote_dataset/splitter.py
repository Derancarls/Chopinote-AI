"""80/10/10 数据集划分工具。"""
import random
from pathlib import Path


def split_dataset(file_list_path: str, output_dir: str,
                  train_ratio: float = 0.8, val_ratio: float = 0.1,
                  seed: int = 42):
    """读取 token 文件列表，按比例 shuffle 划分并保存。

    Args:
        file_list_path: 由 prepare_corpus.py 生成的 all_files.txt 路径
        output_dir: 保存 train.txt / val.txt / test.txt 的目录
    """
    with open(file_list_path, 'r', encoding='utf-8') as f:
        files = [line.strip() for line in f if line.strip()]

    if not files:
        print("错误: 文件列表为空")
        return

    random.seed(seed)
    random.shuffle(files)

    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        'train.txt': train_files,
        'val.txt': val_files,
        'test.txt': test_files,
    }

    for name, file_list in splits.items():
        path = output_dir / name
        with open(path, 'w', encoding='utf-8') as f:
            for fp in file_list:
                f.write(fp + '\n')
        print(f"  {name}: {len(file_list)} 文件 -> {path}")

    print(f"\n划分完成: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")


def load_split(file_path: str) -> list[str]:
    """读取划分文件，返回 token JSON 路径列表。"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


if __name__ == '__main__':
    import sys
    file_list = sys.argv[1] if len(sys.argv) > 1 else 'data/processed/all_files.txt'
    output = sys.argv[2] if len(sys.argv) > 2 else 'data/processed'
    split_dataset(file_list, output)
