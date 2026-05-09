import tempfile
from pathlib import Path

from chopinote_dataset.splitter import split_dataset, load_split


class TestSplitDataset:
    def test_split_ratios(self, tmp_path):
        file_list = tmp_path / 'all_files.txt'
        files = [f'data/tokens/file_{i:03d}.json' for i in range(100)]
        file_list.write_text('\n'.join(files))

        split_dataset(str(file_list), str(tmp_path), train_ratio=0.8, val_ratio=0.1, seed=42)

        train = load_split(str(tmp_path / 'train.txt'))
        val = load_split(str(tmp_path / 'val.txt'))
        test = load_split(str(tmp_path / 'test.txt'))

        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

        all_files = set(train + val + test)
        assert len(all_files) == 100

    def test_deterministic(self, tmp_path):
        files = [f'data/tokens/file_{i:03d}.json' for i in range(20)]
        file_list1 = tmp_path / 'all1.txt'
        file_list2 = tmp_path / 'all2.txt'
        file_list1.write_text('\n'.join(files))
        file_list2.write_text('\n'.join(files))

        split_dataset(str(file_list1), str(tmp_path / 'out1'), seed=42)
        split_dataset(str(file_list2), str(tmp_path / 'out2'), seed=42)

        train1 = load_split(str(tmp_path / 'out1' / 'train.txt'))
        train2 = load_split(str(tmp_path / 'out2' / 'train.txt'))
        assert train1 == train2

    def test_empty_input(self, tmp_path):
        file_list = tmp_path / 'empty.txt'
        file_list.write_text('')
        # should not crash
        split_dataset(str(file_list), str(tmp_path / 'out'))
