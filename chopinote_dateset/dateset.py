"""
数据集加载与管理模块
"""
import os
from typing import List, Dict, Any

class MusicDataset:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.file_list = self._scan_files()

    def _scan_files(self) -> List[str]:
        files = []
        for root, _, filenames in os.walk(self.data_dir):
            for fname in filenames:
                if fname.endswith('.pkl') or fname.endswith('.json'):
                    files.append(os.path.join(root, fname))
        return files

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx) -> Any:
        file_path = self.file_list[idx]
        if file_path.endswith('.pkl'):
            import pickle
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        elif file_path.endswith('.json'):
            import json
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            raise ValueError('Unsupported file type')

    def get_all(self) -> List[Any]:
        return [self[i] for i in range(len(self))]
