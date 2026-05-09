import torch
from chopinote_model.dataset import collate_fn


class TestCollateFn:
    def test_equal_length(self):
        batch = [
            {
                'input_ids': torch.tensor([1, 2, 3]),
                'labels': torch.tensor([2, 3, 4]),
                'attention_mask': torch.tensor([1, 1, 1]),
            },
            {
                'input_ids': torch.tensor([5, 6, 7]),
                'labels': torch.tensor([6, 7, 8]),
                'attention_mask': torch.tensor([1, 1, 1]),
            },
        ]
        result = collate_fn(batch)
        assert result['input_ids'].shape == (2, 3)
        assert result['labels'].shape == (2, 3)
        assert result['attention_mask'].shape == (2, 3)

    def test_variable_length_padding(self):
        batch = [
            {
                'input_ids': torch.tensor([1, 2]),
                'labels': torch.tensor([2, 3]),
                'attention_mask': torch.tensor([1, 1]),
            },
            {
                'input_ids': torch.tensor([5, 6, 7, 8]),
                'labels': torch.tensor([6, 7, 8, 9]),
                'attention_mask': torch.tensor([1, 1, 1, 1]),
            },
        ]
        result = collate_fn(batch)
        assert result['input_ids'].shape == (2, 4)
        assert result['labels'].shape == (2, 4)
        assert result['attention_mask'].shape == (2, 4)

    def test_label_padding_is_neg_100(self):
        batch = [
            {
                'input_ids': torch.tensor([1]),
                'labels': torch.tensor([2]),
                'attention_mask': torch.tensor([1]),
            },
            {
                'input_ids': torch.tensor([5, 6, 7]),
                'labels': torch.tensor([6, 7, 8]),
                'attention_mask': torch.tensor([1, 1, 1]),
            },
        ]
        result = collate_fn(batch)
        assert result['labels'][0, 1] == -100
        assert result['labels'][0, 2] == -100
        assert result['labels'][0, 0] == 2

    def test_attention_mask_correct(self):
        batch = [
            {
                'input_ids': torch.tensor([1, 2]),
                'labels': torch.tensor([2, 3]),
                'attention_mask': torch.tensor([1, 1]),
            },
            {
                'input_ids': torch.tensor([5, 6, 7, 8]),
                'labels': torch.tensor([6, 7, 8, 9]),
                'attention_mask': torch.tensor([1, 1, 1, 1]),
            },
        ]
        result = collate_fn(batch)
        assert (result['attention_mask'][0] == torch.tensor([1, 1, 0, 0])).all()
        assert (result['attention_mask'][1] == torch.tensor([1, 1, 1, 1])).all()
