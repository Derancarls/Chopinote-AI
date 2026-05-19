"""全局烟雾测试：快速验证所有核心模块能否正常加载和运行。

设计目标：
  - 在 < 30 秒内完成（纯 CPU 模型用 tiny config）
  - 覆盖全流水线：tokenizer → converter → dataset → model → training → generation
  - 不依赖外部文件（全部合成数据）
  - 硬件无关：无 GPU 时自动跳过 CUDA 测试
"""
import gc
import math
import pytest
import torch
from pathlib import Path

# ── Skip guards ──
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')


# ═══════════════════════════════════════════════════════════════
#  Tokenizer
# ═══════════════════════════════════════════════════════════════

class TestTokenizerSmoke:
    def test_import_and_init(self):
        from chopinote_dataset.tokenizer import REMITokenizer
        t = REMITokenizer()
        assert t.vocab_size == 908

    def test_tokenize_detokenize_roundtrip(self, tokenizer):
        events = [
            ('<BOS>', None), ('<Bar>', None), ('<Position', 0),
            ('<Program', '0'), ('<Note_ON', 60), ('<Velocity', 4), ('<Duration', 8),
            ('<EOS>', None),
        ]
        ids = tokenizer.tokenize(events)
        recovered = tokenizer.detokenize(ids)
        assert recovered == events

    def test_all_token_types_encodable(self, tokenizer):
        """每种 token 类型至少有一个实例可以编码且解码回原文。"""
        samples = [
            '<Bar>', '<Position 0>', '<Program 0>', '<Note_ON 60>',
            '<Velocity 4>', '<Duration 8>', '<Clef treble>', '<Dynamic f>',
            '<Hairpin cresc>', '<Artic staccato>', '<Ornament trill>',
            '<Pedal start>', '<Slur start>', '<Rest>', '<GraceNote acciaccatura>',
            '<Repeat start>', '<Jump dal_segno>', '<Tempo 120>',
            '<TupletStart 3:2>', '<TupletEnd>', '<TimeSig 4/4>', '<Key C>', '<Beat 1>',
            '<Octave 8va>', '<Arpeggio>', '<Bass 0>', '<Anticipate C>',
        ]
        for token_str in samples:
            tid = tokenizer.encode_token(token_str)
            assert tid != 3, f'{token_str} 编码为 MASK，应存在有效 ID'
            decoded = tokenizer.decode_token(tid)
            assert decoded == token_str, f'{token_str} → {tid} → {decoded}'


# ═══════════════════════════════════════════════════════════════
#  Converters
# ═══════════════════════════════════════════════════════════════

class TestConvertersSmoke:
    """三种格式转换器合成数据测试。"""

    def test_fast_converter_import(self):
        """FastMIDIToREMI 能正常导入。"""
        from chopinote_dataset.fast_converter import FastMIDIToREMI, process_midi_file_fast
        conv = FastMIDIToREMI()
        assert conv.grid_size == 16

    def test_musicxml_converter_import(self):
        from chopinote_dataset.converter import MusicXMLToREMI
        conv = MusicXMLToREMI()
        assert conv.grid_size == 16

    def test_pdmx_converter_import(self):
        from chopinote_dataset.converter import PDMXToREMI
        conv = PDMXToREMI()
        assert conv.grid_size == 16

    def test_midi_converter_import(self):
        from chopinote_dataset.converter import MIDIToREMI
        conv = MIDIToREMI()
        assert conv.grid_size == 16

    def test_fast_converter_synthetic_midi(self, tmp_path):
        """合成一个极简 MIDI 文件，走通 fast 转换管道。"""
        from chopinote_dataset.fast_converter import process_midi_file_fast
        import struct

        # 合成 1 轨 1 个音符的 MIDI 文件
        midi_path = tmp_path / 'test_smoke.mid'
        with open(midi_path, 'wb') as f:
            # MIDI header: format 0, 1 track, 96 ticks/quarter
            f.write(b'MThd' + struct.pack('>IHHH', 6, 0, 1, 96))
            # Track
            track_data = b''
            track_data += b'\x00\xff\x03\x04Test'  # track name
            track_data += b'\x00\xff\x51\x03\x07\xa1\x20'  # tempo=120bpm
            track_data += b'\x00\x90\x3c\x40'  # Note ON C4 vel=64
            track_data += b'\x60\x80\x3c\x00'  # Note OFF after 96 ticks
            track_data += b'\x00\xff\x2f\x00'  # End of Track
            track_len = len(track_data)
            f.write(b'MTrk' + struct.pack('>I', track_len) + track_data)

        result = process_midi_file_fast(
            str(midi_path), str(tmp_path),
            min_notes=1, max_notes=100,
            min_tokens=1, max_tokens=65536,
            min_size_kb=0, max_size_mb=50,
        )
        assert result is not None, 'fast converter 应成功处理合成 MIDI'
        assert result['num_tokens'] > 0, '应生成至少 1 个 token'
        # 检查 token 文件已写入
        token_path = Path(result['token_path'])
        assert token_path.exists(), 'token 文件应存在'
        content = token_path.read_text().strip()
        assert len(content) > 0, 'token 内容不为空'

    def test_pdmx_converter_synthetic(self):
        """PDMXToREMI 合成数据转换。"""
        from chopinote_dataset.converter import PDMXToREMI
        from chopinote_dataset.tokenizer import REMITokenizer

        p = PDMXToREMI()
        data = {
            'resolution': 480,
            'barlines': [{'name': 'Barline', 'time': 0, 'measure': 1},
                         {'name': 'Barline', 'time': 1920, 'measure': 2}],
            'time_signatures': [{'name': 'TimeSignature', 'time': 0,
                                 'numerator': 4, 'denominator': 4, 'measure': 1}],
            'tempos': [{'name': 'Tempo', 'time': 0, 'qpm': 120, 'measure': 1}],
            'key_signatures': [{'name': 'KeySignature', 'time': 0, 'key': 0, 'measure': 1}],
            'beats': [],
            'tracks': [{
                'program': 0, 'name': 'Piano', 'is_drum': False,
                'notes': [
                    {'name': 'Note', 'time': 0, 'pitch': 60, 'duration': 480,
                     'velocity': 80, 'pitch_str': 'C', 'measure': 1, 'is_grace': False},
                ],
                'annotations': [
                    {'name': 'Annotation', 'time': 0,
                     'annotation': {'name': 'Dynamic', 'subtype': 'mf'},
                     'group': None, 'measure': 1},
                ],
            }],
        }
        ids = p.convert_pdmx(data)
        assert len(ids) > 0
        t = REMITokenizer()
        decoded = t.detokenize(ids)
        # 解码后的 event 格式为 ('<Dynamic', 'mf')
        assert any(x[0] == '<Dynamic' for x in decoded), 'Dynamic 应出现在输出中'


# ═══════════════════════════════════════════════════════════════
#  Preprocessors & Dataset
# ═══════════════════════════════════════════════════════════════

class TestPreprocessorSmoke:
    def test_pdmx_preprocessor_init(self):
        from chopinote_dataset.processor import PDMXPreprocessor
        p = PDMXPreprocessor()
        assert p.config is not None

    def test_musicxml_preprocessor_init(self):
        from chopinote_dataset.processor import MusicXMLPreprocessor
        p = MusicXMLPreprocessor()
        assert p.config is not None

    def test_splitter_import(self):
        from chopinote_dataset.splitter import split_dataset, load_split
        assert callable(split_dataset)

    def test_collate_fn(self):
        from chopinote_model.dataset import collate_fn
        batch = [
            {'input_ids': torch.tensor([1, 2, 3]), 'labels': torch.tensor([2, 3, 4]),
             'attention_mask': torch.tensor([1, 1, 1])},
            {'input_ids': torch.tensor([5, 6]), 'labels': torch.tensor([6, 7]),
             'attention_mask': torch.tensor([1, 1])},
        ]
        result = collate_fn(batch)
        assert result['input_ids'].shape == (2, 3)
        assert result['labels'].shape == (2, 3)
        assert result['attention_mask'].shape == (2, 3)


# ═══════════════════════════════════════════════════════════════
#  Model
# ═══════════════════════════════════════════════════════════════

class TestModelSmoke:
    """Small config model forward/backward."""

    CONFIG_KWARGS = dict(d_model=128, n_heads=2, n_layers=2, d_ff=512,
                         vocab_size=100, max_seq_len=64,
                         use_section_attention=False)

    def _make_model(self):
        from chopinote_model.model import MusicTransformer
        from chopinote_model.config import ModelConfig
        return MusicTransformer(ModelConfig(**self.CONFIG_KWARGS)).to(device)

    def test_model_forward(self):
        model = self._make_model()
        x = torch.randint(0, 100, (2, 32), device=device)
        logits = model(x)
        assert logits.shape == (2, 32, 100)
        assert torch.isfinite(logits).all()

    def test_model_backward(self):
        model = self._make_model()
        x = torch.randint(0, 100, (2, 32), device=device)
        logits = model(x)
        loss = logits.mean()
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f'{name} grad is None'
            assert torch.isfinite(p.grad).all(), f'{name} grad has NaN/Inf'

    def test_model_with_mask(self):
        model = self._make_model()
        x = torch.randint(0, 100, (2, 32), device=device)
        mask = torch.zeros(2, 32, dtype=torch.bool, device=device)
        mask[:, :24] = True
        logits = model(x, attention_mask=mask)
        assert torch.isfinite(logits).all()

    def test_model_deterministic_eval(self):
        model = self._make_model().eval()
        x = torch.randint(0, 100, (1, 16), device=device)
        with torch.no_grad():
            a = model(x)
            b = model(x)
        assert torch.allclose(a, b, atol=1e-6)

    def test_model_measure_ids_from_bar_tokens(self):
        """Bar token 被自动累积为 measure_ids。"""
        from chopinote_model.config import ModelConfig
        from chopinote_model.model import MusicTransformer
        config = ModelConfig(**self.CONFIG_KWARGS)
        model = MusicTransformer(config).to(device)
        # token 7 = bar token
        bar_id = config.bar_token_id  # should be 4
        x = torch.tensor([[1, bar_id, 5, 6, bar_id, 7, bar_id, 8]], device=device)
        logits = model(x)
        assert torch.isfinite(logits).all()

    def test_model_weight_tying(self):
        """token_embedding 与 lm_head 权重共享。"""
        from chopinote_model.config import ModelConfig
        from chopinote_model.model import MusicTransformer
        config = ModelConfig(**self.CONFIG_KWARGS)
        model = MusicTransformer(config)
        assert model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()


# ═══════════════════════════════════════════════════════════════
#  KV cache generation
# ═══════════════════════════════════════════════════════════════

class TestGenerationSmoke:
    def test_kv_cache_forward(self):
        """KV cache 扩展后 forward 不崩溃。"""
        from chopinote_model.config import ModelConfig
        from chopinote_model.model import MusicTransformer

        config = ModelConfig(d_model=128, n_heads=2, n_layers=2, d_ff=512,
                             vocab_size=100, max_seq_len=64)
        model = MusicTransformer(config).to(device)
        model.eval()

        # 第一次 forward: prefilling (each cache slot is [None, None])
        x1 = torch.randint(0, 100, (1, 8), device=device)
        caches = [[None, None] for _ in range(config.n_layers)]
        with torch.no_grad():
            out1 = model(x1, kv_caches=caches)
            # caches 被 in-place 填充
            for c in caches:
                assert c[0] is not None, 'cache 应在 prefill 后被填充'

        # 第二次 forward: 1 个新 token
        x2 = torch.randint(0, 100, (1, 1), device=device)
        with torch.no_grad():
            out2 = model(x2, kv_caches=caches)

        assert out1.shape[1] == 8
        assert out2.shape[1] == 1
        assert torch.isfinite(out1).all()
        assert torch.isfinite(out2).all()
        # cache 累积长度
        assert caches[0][0].shape[2] == 9  # 8 + 1

    def test_kv_cache_prefill_matches_normal(self):
        """KV cache prefill 输出与正常 forward 一致。"""
        from chopinote_model.config import ModelConfig
        from chopinote_model.model import MusicTransformer

        config = ModelConfig(d_model=128, n_heads=2, n_layers=2, d_ff=512,
                             vocab_size=100, max_seq_len=64)
        model = MusicTransformer(config).to(device)
        model.eval()
        x = torch.randint(0, 100, (1, 8), device=device)

        with torch.no_grad():
            logits_normal = model(x)

            caches = [[None, None] for _ in range(config.n_layers)]
            logits_cached = model(x, kv_caches=caches)

        assert torch.allclose(logits_normal, logits_cached, atol=1e-6)


# ═══════════════════════════════════════════════════════════════
#  Config & CLI
# ═══════════════════════════════════════════════════════════════

class TestConfigSmoke:
    def test_model_config_default(self):
        from chopinote_model.config import ModelConfig
        c = ModelConfig()
        assert c.vocab_size == 908
        assert c.d_model == 2048
        assert c.n_layers == 24

    def test_training_config_default(self):
        from chopinote_model.config import TrainingConfig
        c = TrainingConfig()
        assert c.lr == 1.5e-4
        assert c.total_steps == 100000

    def test_phase_config(self):
        from chopinote_model.config import PhaseConfig
        p = PhaseConfig(name='test', total_steps=100,
                        data_split_file='/tmp/test.txt')
        assert p.lr == 2e-4  # default for PhaseConfig
        assert p.warmup_steps == 2000

    def test_presets_can_list(self):
        from chopinote_cli.presets import list_presets, Preset
        presets = list_presets()
        assert len(presets) == 7
        assert all(isinstance(p, Preset) for p in presets)

    def test_preset_attrs_roundtrip(self):
        from chopinote_cli.presets import get_preset
        p = get_preset('baroque')
        attrs = p.attrs()
        assert isinstance(attrs, dict)
        assert 'temperature' in attrs
        p2 = get_preset('baroque')
        assert p2.attrs() == attrs


# ═══════════════════════════════════════════════════════════════
#  Loss mask
# ═══════════════════════════════════════════════════════════════

class TestLossMaskSmoke:
    def test_default_mask(self):
        from chopinote_model.config import TokenLossMask
        from chopinote_dataset.tokenizer import REMITokenizer
        mask = TokenLossMask()
        t = REMITokenizer()
        ids = mask.get_masked_token_ids(t)
        assert len(ids) > 0
        # 结构性 token 不应该被 mask
        assert t.pad_token_id not in ids
        assert t.bar_token_id not in ids
        # 表现性 token 应该被 mask
        assert t.encode_token('<Dynamic f>') in ids
        assert t.encode_token('<Artic staccato>') in ids

    def test_grace_not_masked(self):
        from chopinote_model.config import TokenLossMask
        from chopinote_dataset.tokenizer import REMITokenizer
        mask = TokenLossMask(mask_grace_note=False)
        t = REMITokenizer()
        ids = mask.get_masked_token_ids(t)
        assert t.encode_token('<GraceNote acciaccatura>') not in ids


# ═══════════════════════════════════════════════════════════════
#  FP8 (skip if no CUDA)
# ═══════════════════════════════════════════════════════════════

@cuda_only
class TestFP8Smoke:
    def test_fp8_linear_import(self):
        from chopinote_model.fp8_linear import FP8Linear
        linear = FP8Linear(64, 128).cuda()
        x = torch.randn(2, 64, dtype=torch.bfloat16, device='cuda')
        y = linear(x)
        assert y.shape == (2, 128)
        assert torch.isfinite(y).all()

    def test_fp8_model_forward(self):
        from chopinote_model.config import ModelConfig
        from chopinote_model.model import MusicTransformer

        config = ModelConfig(d_model=128, n_heads=2, n_layers=2, d_ff=512,
                             vocab_size=100, max_seq_len=64,
                             use_section_attention=False)
        model = MusicTransformer(config).cuda()
        model.eval()
        model.set_fp8_mode(True)
        x = torch.randint(0, 100, (2, 16), device='cuda')
        _ = model(x)  # warmup
        logits = model(x)
        assert torch.isfinite(logits).all()

    def test_fp8_model_backward(self):
        from chopinote_model.config import ModelConfig
        from chopinote_model.model import MusicTransformer

        config = ModelConfig(d_model=128, n_heads=2, n_layers=2, d_ff=512,
                             vocab_size=100, max_seq_len=64,
                             use_section_attention=False)
        model = MusicTransformer(config).cuda()
        model.eval()
        model.set_fp8_mode(True)
        x = torch.randint(0, 100, (2, 8), device='cuda')
        _ = model(x)  # warmup
        logits = model(x)
        loss = logits.sum()
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f'{name} grad is None'
            assert torch.isfinite(p.grad).all(), f'{name} grad has NaN/Inf'


# ═══════════════════════════════════════════════════════════════
#  Sidecar hash
# ═══════════════════════════════════════════════════════════════

class TestHashSidecarSmoke:
    def test_compute_file_hash_sidecar(self, tmp_path):
        from chopinote_dataset.fast_converter import compute_file_hash
        f = tmp_path / 'test.mid'
        f.write_bytes(b'\x00\x01\x02\x03')
        h1 = compute_file_hash(str(f))
        assert len(h1) == 32
        # sidecar 文件已生成
        assert (tmp_path / 'test.mid.hash').exists()
        h2 = compute_file_hash(str(f))
        assert h1 == h2  # 第二次应命中缓存


# ═══════════════════════════════════════════════════════════════
#  Trainer
# ═══════════════════════════════════════════════════════════════

class TestTrainerSmoke:
    def test_trainer_import(self):
        from chopinote_model.train import Trainer
        assert Trainer is not None

    def test_grad_scaler_not_used_for_bf16(self):
        """bf16 training 不应使用 GradScaler。"""
        import inspect
        from chopinote_model.train import Trainer
        src = inspect.getsource(Trainer.__init__)
        assert 'GradScaler' not in src


# ═══════════════════════════════════════════════════════════════
#  File & import integrity
# ═══════════════════════════════════════════════════════════════

class TestImportIntegrity:
    """所有核心模块的顶层导入不报错。"""

    def test_import_chopinote_model(self):
        import chopinote_model
        assert hasattr(chopinote_model, 'MusicTransformer')

    def test_import_chopinote_dataset(self):
        import chopinote_dataset
        assert hasattr(chopinote_dataset, 'REMITokenizer')

    def test_import_chopinote_cli(self):
        """验证 chopinote_cli.main 模块可导入。"""
        from chopinote_cli import main as cli_main
        assert cli_main is not None
