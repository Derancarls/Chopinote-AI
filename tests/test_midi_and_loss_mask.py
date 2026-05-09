"""MIDI → REMI 转换器和 Loss 屏蔽测试。"""
import pytest
from chopinote_dataset.tokenizer import REMITokenizer
from chopinote_dataset.converter import MIDIToREMI
from chopinote_model.config import TokenLossMask, PhaseConfig


class TestMIDIToREMI:
    def test_init(self):
        mt = MIDIToREMI(grid_size=16, velocity_levels=8)
        assert mt.grid_size == 16
        assert mt.velocity_levels == 8
        assert mt.quarter_per_position == 0.25

    def test_convert_nonexistent_file(self):
        mt = MIDIToREMI()
        ids, meta = mt.convert("nonexistent.mid", collect_metadata=True)
        assert ids == []
        assert meta == {}

    def test_convert_score_empty(self):
        """music21 Score 无 part 时返回空列表。"""
        from music21 import stream
        mt = MIDIToREMI()
        empty = stream.Score()
        ids = mt.convert_score(empty)
        assert ids == []

    def test_drum_tracks_skipped(self):
        """鼓轨 program >= 112 应被跳过。"""
        mt = MIDIToREMI()
        assert 120 in mt._DRUM_PROGRAMS
        assert 0 not in mt._DRUM_PROGRAMS


class TestTokenLossMask:
    def test_default_masks_all_expressive(self):
        mask = TokenLossMask()
        t = REMITokenizer()
        ids = mask.get_masked_token_ids(t)
        assert len(ids) > 0
        # 核心 token 不应被屏蔽
        assert t.bar_token_id not in ids
        assert t.bos_token_id not in ids
        assert t.eos_token_id not in ids
        assert t.pad_token_id not in ids

    def test_dynamic_masked(self):
        mask = TokenLossMask()
        t = REMITokenizer()
        ids = mask.get_masked_token_ids(t)
        dyn_f = t.encode_token('<Dynamic f>')
        assert dyn_f in ids

    def test_artic_masked(self):
        mask = TokenLossMask()
        t = REMITokenizer()
        ids = mask.get_masked_token_ids(t)
        staccato = t.encode_token('<Artic staccato>')
        assert staccato in ids

    def test_note_not_masked(self):
        mask = TokenLossMask()
        t = REMITokenizer()
        ids = mask.get_masked_token_ids(t)
        note_60 = t.encode_token('<Note_ON 60>')
        assert note_60 not in ids

    def test_partial_mask(self):
        """只屏蔽部分类别。"""
        mask = TokenLossMask(mask_clef=False, mask_dynamic=True,
                             mask_hairpin=False, mask_artic=False,
                             mask_ornament=False, mask_pedal=False,
                             mask_slur=False, mask_octave=False,
                             mask_arpeggio=False, mask_grace_note=False,
                             mask_repeat=False, mask_jump=False,
                             mask_tuplet=False)
        t = REMITokenizer()
        ids = mask.get_masked_token_ids(t)
        # 只有 Dynamic 被屏蔽
        assert t.encode_token('<Dynamic f>') in ids
        assert t.encode_token('<Clef treble>') not in ids
        assert t.encode_token('<Artic staccato>') not in ids

    def test_no_mask_is_empty(self):
        """全部关闭时返回空集。"""
        mask = TokenLossMask(
            mask_clef=False, mask_dynamic=False, mask_hairpin=False,
            mask_artic=False, mask_ornament=False, mask_pedal=False,
            mask_slur=False, mask_octave=False, mask_arpeggio=False,
            mask_grace_note=False, mask_repeat=False, mask_jump=False,
            mask_tuplet=False,
        )
        t = REMITokenizer()
        ids = mask.get_masked_token_ids(t)
        assert len(ids) == 0


class TestPhaseConfig:
    def test_phase_basic(self):
        p = PhaseConfig(name="pretrain", total_steps=1000,
                        data_split_file="data/midi_train.txt")
        assert p.name == "pretrain"
        assert p.total_steps == 1000
        assert p.warmup_steps == 2000  # default
        assert p.lr == 2e-4  # default
        assert p.loss_mask is None

    def test_phase_with_mask(self):
        mask = TokenLossMask()
        p = PhaseConfig(name="pretrain", total_steps=100,
                        data_split_file="data/midi_train.txt",
                        loss_mask=mask)
        assert p.loss_mask is not None
        assert p.loss_mask.mask_dynamic is True
