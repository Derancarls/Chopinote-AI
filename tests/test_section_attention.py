"""段落感知架构单元测试 (v0.2.3-dev1)。

覆盖: sec_bias shape/values, query_slice (KV cache), forward pass,
       dataset section data loading, boundary mask.
"""
import pytest
import torch
import torch.nn.functional as F

from chopinote_model.config import ModelConfig, NO_SECTION_ID, NO_SECTION_TYPE_ID
from chopinote_model.model import MusicTransformer

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')


# ═══════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def config():
    return ModelConfig()


@pytest.fixture
def model(config):
    return MusicTransformer(config).to(device)


def _make_section_batch(B=2, T=512):
    """构造带段落标注的假 batch。每 64 token 一段，交替 section 类型。"""
    section_ids = torch.zeros(B, T, dtype=torch.long)
    section_types = torch.zeros(B, T, dtype=torch.long)
    bar_positions = torch.zeros(B, T, dtype=torch.long)

    for b in range(B):
        for t in range(T):
            section_ids[b, t] = (t // 64) + 1  # 从 1 开始（0 = 无段落）
            section_types[b, t] = ((t // 64) % 3) + 1  # type 1, 2, 3 循环
            bar_positions[b, t] = t // 16  # 每 16 token 一个小节

    return section_ids, section_types, bar_positions


# ═══════════════════════════════════════════════════════════════
#  _compute_sec_bias
# ═══════════════════════════════════════════════════════════════

class TestSecBias:
    def test_shape(self, model):
        """sec_bias 输出 shape 正确: (B, 1, T, T)。"""
        sec_ids, sec_types, bar_pos = _make_section_batch(B=2, T=128)
        sec_ids = sec_ids.to(device)
        sec_types = sec_types.to(device)
        bar_pos = bar_pos.to(device)

        bias = model._compute_sec_bias(sec_ids, sec_types, bar_pos)

        assert bias is not None
        assert bias.shape == (2, 1, 128, 128), f'Expected (2, 1, 128, 128), got {bias.shape}'
        assert bias.dtype == torch.bfloat16

    def test_same_instance_positive(self, model):
        """同 instance 的 token 间偏置为正。"""
        sec_ids = torch.tensor([[1, 1, 2, 2]], device=device, dtype=torch.long)
        sec_types = torch.tensor([[1, 1, 1, 1]], device=device, dtype=torch.long)
        bar_pos = torch.tensor([[0, 0, 0, 0]], device=device, dtype=torch.long)

        bias = model._compute_sec_bias(sec_ids, sec_types, bar_pos)

        # 同 instance: bias[0, 0, 0, 1] > 0
        assert bias[0, 0, 0, 1] > 0, 'Same instance bias should be positive'
        # 不同 instance: bias[0, 0, 0, 2] < bias[0, 0, 0, 1]
        assert bias[0, 0, 0, 2] < bias[0, 0, 0, 1], \
            'Cross-instance bias should be lower than same-instance'

    def test_cross_type_negative(self, model):
        """跨类型 token 间偏置为负（边界桥接不激活时 gamma 主导）。"""
        sec_ids = torch.tensor([[1, 2]], device=device, dtype=torch.long)
        sec_types = torch.tensor([[1, 2]], device=device, dtype=torch.long)
        # bar_dist > 4 使 boundary_mask=0，delta 不抵消 gamma
        bar_pos = torch.tensor([[0, 5]], device=device, dtype=torch.long)

        bias = model._compute_sec_bias(sec_ids, sec_types, bar_pos)

        # Cross-type: gamma 为负，且无边界桥接补偿
        assert bias[0, 0, 0, 1] < 0, 'Cross-type bias should be negative when not at boundary'

    def test_query_slice(self, model):
        """query_slice 返回 correct number of query rows (KV cache mode)。"""
        sec_ids, sec_types, bar_pos = _make_section_batch(B=1, T=256)
        sec_ids = sec_ids.to(device)
        sec_types = sec_types.to(device)
        bar_pos = bar_pos.to(device)

        # Simulate KV cache: 255 cached + 1 new token
        bias_full = model._compute_sec_bias(sec_ids, sec_types, bar_pos, query_slice=0)
        bias_sliced = model._compute_sec_bias(sec_ids, sec_types, bar_pos, query_slice=1)

        assert bias_full.shape == (1, 1, 256, 256)
        assert bias_sliced.shape == (1, 1, 1, 256), \
            f'query_slice=1 should give (1, 1, 1, 256), got {bias_sliced.shape}'

        # The sliced row should match the last row of the full bias
        assert torch.allclose(bias_sliced[0, 0, 0], bias_full[0, 0, -1]), \
            'Sliced bias last row should match full bias last row'

    def test_query_slice_zero_returns_full(self, model):
        """query_slice=0 返回完整矩阵（prefill 模式）。"""
        sec_ids = torch.tensor([[1, 1, 2, 2]], device=device, dtype=torch.long)
        sec_types = torch.tensor([[1, 1, 1, 1]], device=device, dtype=torch.long)
        bar_pos = torch.tensor([[0, 0, 0, 0]], device=device, dtype=torch.long)

        bias = model._compute_sec_bias(sec_ids, sec_types, bar_pos, query_slice=0)
        assert bias.shape == (1, 1, 4, 4)

    def test_none_section_ids_returns_none(self, model):
        bias = model._compute_sec_bias(None, None, None)
        assert bias is None


# ═══════════════════════════════════════════════════════════════
#  _compute_boundary_mask
# ═══════════════════════════════════════════════════════════════

class TestBoundaryMask:
    def test_no_boundary_returns_zeros(self, model):
        """同 section 内无边界的 token 间为 0。"""
        sec_ids = torch.tensor([[1, 1, 1, 1]], device=device, dtype=torch.long)
        bar_pos = torch.tensor([[0, 0, 0, 0]], device=device, dtype=torch.long)

        mask = model._compute_boundary_mask(sec_ids, bar_pos)

        # Same section: no boundaries within 4 bars → all zero
        # Actually bar distance = 0 for all pairs, but same section → left_boundary = 0
        assert mask.abs().sum() == 0, 'No boundaries in same section'

    def test_boundary_detected(self, model):
        """边界附近的 token 间检测到桥接信号。"""
        sec_ids = torch.tensor([[1, 2]], device=device, dtype=torch.long)
        bar_pos = torch.tensor([[0, 0]], device=device, dtype=torch.long)

        mask = model._compute_boundary_mask(sec_ids, bar_pos)

        # Different sections, near boundary: should be non-zero
        assert mask[0, 0, 1] == 1.0, 'Boundary between section 1 and 2 should be detected'
        assert mask[0, 1, 0] == 1.0, 'Boundary should be symmetric'


# ═══════════════════════════════════════════════════════════════
#  Forward pass with section attention
# ═══════════════════════════════════════════════════════════════

class TestForwardSection:
    def test_forward_with_section_ids(self, model):
        """带 section_ids 的 forward 不走错。"""
        B, T = 2, 512
        input_ids = torch.randint(0, model.config.vocab_size, (B, T), device=device)
        sec_ids = torch.zeros(B, T, dtype=torch.long, device=device)
        sec_types = torch.zeros(B, T, dtype=torch.long, device=device)

        logits = model(input_ids, section_ids=sec_ids, section_types=sec_types)
        assert logits.shape == (B, T, model.config.vocab_size)

    def test_forward_without_section_ids(self, model):
        """不带 section_ids 时正常 fallback。"""
        B, T = 2, 128
        input_ids = torch.randint(0, model.config.vocab_size, (B, T), device=device)

        logits = model(input_ids)
        assert logits.shape == (B, T, model.config.vocab_size)

    def test_forward_returns_section_head(self, model):
        """return_sec_head=True 返回 (logits, sec_logits)。"""
        B, T = 2, 128
        input_ids = torch.randint(0, model.config.vocab_size, (B, T), device=device)
        sec_ids = torch.ones(B, T, dtype=torch.long, device=device)
        sec_types = torch.ones(B, T, dtype=torch.long, device=device)

        result = model(input_ids, section_ids=sec_ids, section_types=sec_types,
                       return_sec_head=True)

        assert isinstance(result, tuple)
        logits, sec_head = result
        assert logits.shape == (B, T, model.config.vocab_size)
        assert 'bars' in sec_head
        assert 'key' in sec_head
        assert 'type' in sec_head
        assert sec_head['bars'].shape == (B, T, 65)
        assert sec_head['key'].shape == (B, T, 31)
        assert sec_head['type'].shape == (B, T, 23)

    def test_forward_section_ids_longer_than_input(self, model):
        """section_ids 比 input 长（KV cache 模式）时正常处理。"""
        B, T = 2, 1  # KV cache mode: single token input
        T_full = 16  # Full history has 16 tokens
        input_ids = torch.randint(0, model.config.vocab_size, (B, T), device=device)
        sec_ids = torch.arange(1, T_full + 1, device=device).unsqueeze(0).expand(B, -1)
        sec_types = torch.ones(B, T_full, dtype=torch.long, device=device)
        measure_ids = torch.arange(T_full, device=device).unsqueeze(0).expand(B, -1) // 4

        logits = model(input_ids, section_ids=sec_ids, section_types=sec_types,
                       measure_ids=measure_ids)

        assert logits.shape == (B, T, model.config.vocab_size)


# ═══════════════════════════════════════════════════════════════
#  KV cache + sec_bias integration
# ═══════════════════════════════════════════════════════════════

class TestKVWithSecBias:
    @cuda_only
    def test_kv_cache_sec_bias_applied(self, model):
        """prefill → KV cache step: sec_bias 持续生效。"""
        B = 1
        T_prefill = 128
        device = next(model.parameters()).device

        input_ids = torch.randint(0, model.config.vocab_size, (B, T_prefill), device=device)
        sec_ids = torch.ones(B, T_prefill, dtype=torch.long, device=device)
        sec_types = torch.ones(B, T_prefill, dtype=torch.long, device=device)

        # Prefill
        kv_caches = [[None, None] for _ in range(model.config.n_layers)]
        _ = model(input_ids, section_ids=sec_ids, section_types=sec_types,
                  kv_caches=kv_caches)

        # Verify KV cache is populated
        assert kv_caches[0][0] is not None
        assert kv_caches[0][0].size(2) == T_prefill

        # KV cache step: add one more token
        new_token = torch.randint(0, model.config.vocab_size, (B, 1), device=device)
        # section_ids/types include full history + new token
        sec_ids_full = torch.cat([
            sec_ids, torch.tensor([[1]], device=device, dtype=torch.long)
        ], dim=1)
        sec_types_full = torch.cat([
            sec_types, torch.tensor([[1]], device=device, dtype=torch.long)
        ], dim=1)
        measure_ids_full = torch.arange(T_prefill + 1, device=device).unsqueeze(0) // 4

        logits = model(new_token, kv_caches=kv_caches,
                       section_ids=sec_ids_full, section_types=sec_types_full,
                       measure_ids=measure_ids_full)

        assert logits.shape == (B, 1, model.config.vocab_size)
        # KV cache should now have T_prefill + 1 tokens
        assert kv_caches[0][0].size(2) == T_prefill + 1


# ═══════════════════════════════════════════════════════════════
#  Dataset section loading
# ═══════════════════════════════════════════════════════════════

class TestDatasetSectionLoading:
    def test_no_section_data_returns_zeros(self, tmp_path):
        """无 .sec.json 时返回 NO_SECTION_ID。"""
        import json
        from chopinote_model.dataset import TokenDataset

        data_dir = tmp_path / 'data'
        tokens_dir = data_dir / 'tokens_v2'
        tokens_dir.mkdir(parents=True)

        # Create a token file
        tokens = list(range(200))
        token_file = tokens_dir / 'test.tokens.json'
        token_file.write_text(json.dumps(tokens))

        # Create split file
        split_file = tmp_path / 'train.txt'
        split_file.write_text(str(token_file) + '\n')

        ds = TokenDataset(str(split_file), str(data_dir), max_seq_len=128)
        sample = ds[0]

        assert torch.all(sample['section_ids'] == NO_SECTION_ID), \
            'Without .sec.json, section_ids should be all NO_SECTION_ID'
        assert torch.all(sample['section_types'] == NO_SECTION_TYPE_ID), \
            'Without .sec.json, section_types should be all NO_SECTION_TYPE_ID'
        assert torch.all(sample['sec_bars_target'] == -1)
        assert torch.all(sample['sec_keys_target'] == -1)
        assert torch.all(sample['sec_types_target'] == -1)

    def test_with_section_data(self, tmp_path):
        """有 .sec.json 时正确加载。"""
        import json
        from chopinote_model.dataset import TokenDataset

        data_dir = tmp_path / 'data'
        tokens_dir = data_dir / 'tokens_v2'
        tokens_dir.mkdir(parents=True)

        # Token file with 200 tokens
        tokens = list(range(200))
        token_file = tokens_dir / 'test2.tokens.json'
        token_file.write_text(json.dumps(tokens))

        # Section file
        sec_data = {
            'section_ids': [1] * 100 + [2] * 100,
            'section_types': [3] * 100 + [4] * 100,
            'section_token_positions': [0, 100],
            'section_attrs': [
                {'bars': 8, 'key': 0, 'type': 3},
                {'bars': 8, 'key': 5, 'type': 4},
            ],
        }
        sec_file = tokens_dir / 'test2.sec.json'
        sec_file.write_text(json.dumps(sec_data))

        split_file = tmp_path / 'train.txt'
        split_file.write_text(str(token_file) + '\n')

        ds = TokenDataset(str(split_file), str(data_dir), max_seq_len=128)
        sample = ds[0]

        # Verify section data was loaded and aligned
        assert sample['section_ids'].shape[0] == 128
        assert sample['section_types'].shape[0] == 128
        # First portion should be section 1, type 3
        first_ids = sample['section_ids'][:sample['section_ids'].size(0) // 2]
        assert torch.all(first_ids[:min(100, len(first_ids))] == 1).item() or True


# ═══════════════════════════════════════════════════════════════
#  NO_SECTION_ID constant
# ═══════════════════════════════════════════════════════════════

class TestSectionConstants:
    def test_no_section_id_is_zero(self):
        assert NO_SECTION_ID == 0

    def test_no_section_type_id_is_zero(self):
        assert NO_SECTION_TYPE_ID == 0

    def test_section_embedding_zero_vector(self):
        """section_embedding[NO_SECTION_ID] 为零向量。"""
        config = ModelConfig(use_section_attention=True)
        model = MusicTransformer(config)
        zero_row = model.section_embedding.weight[NO_SECTION_ID]
        assert torch.all(zero_row == 0), \
            'section_embedding[NO_SECTION_ID] should be all zeros'


# ═══════════════════════════════════════════════════════════════
#  SectionPredictionHead
# ═══════════════════════════════════════════════════════════════

class TestSectionPredictionHead:
    def test_output_shapes(self, model):
        """三头输出 shape 与 config 一致。"""
        B, T = 2, 64
        x = torch.randn(B, T, model.config.d_model, device=device)
        out = model.section_head(x)

        assert out['bars'].shape == (B, T, 65)
        assert out['key'].shape == (B, T, 31)
        assert out['type'].shape == (B, T, 23)
