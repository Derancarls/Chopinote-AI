import pytest
from chopinote_model.config import ModelConfig, TrainingConfig


class TestModelConfig:
    def test_head_dim(self):
        config = ModelConfig()
        assert config.head_dim == 64

    def test_vocab_default(self):
        config = ModelConfig()
        assert config.vocab_size == 872

    def test_d_model_divisible(self):
        config = ModelConfig(d_model=768, n_heads=12)

    def test_d_model_indivisible_raises(self):
        with pytest.raises(AssertionError):
            ModelConfig(d_model=769, n_heads=12)

    def test_override_fields(self):
        config = ModelConfig(vocab_size=815, n_layers=8)
        assert config.vocab_size == 815
        assert config.n_layers == 8


class TestTrainingConfig:
    def test_effective_batch_size(self):
        config = TrainingConfig(batch_size=8, grad_accum_steps=4)
        assert config.effective_batch_size == 32

    def test_defaults(self):
        config = TrainingConfig()
        assert config.lr == 2e-4
        assert config.warmup_steps == 2000
        assert config.total_steps == 100000

    def test_override(self):
        config = TrainingConfig(batch_size=16, total_steps=50000)
        assert config.batch_size == 16
        assert config.total_steps == 50000
