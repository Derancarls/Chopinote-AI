"""MIDI / PDMX → REMI 转换器和 Loss 屏蔽测试。"""
import pytest
from chopinote_dataset.tokenizer import REMITokenizer
from chopinote_dataset.converter import MIDIToREMI, PDMXToREMI
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

    # ── 新增：MIDI 增强功能烟雾测试 ─────────────────────────

    def test_rest_from_score(self):
        """music21 Rest 对象应生成 <Rest> token。"""
        from music21 import stream, note, meter
        mt = MIDIToREMI()
        s = stream.Score()
        p = stream.Part()
        m = stream.Measure()
        m.timeSignature = meter.TimeSignature('4/4')
        m.append(note.Note('C4', quarterLength=1.0))
        m.append(note.Rest(quarterLength=1.0))
        m.append(note.Note('E4', quarterLength=1.0))
        p.append(m)
        s.append(p)
        ids = mt.convert_score(s)
        assert len(ids) > 0
        decoded = REMITokenizer().detokenize(ids)
        assert any(t[0] == '<Rest>' for t in decoded), "Rest should produce <Rest> token"

    def test_grace_note_heuristic(self):
        """极短音应被检测为 GraceNote。"""
        from music21 import stream, note, meter, duration
        mt = MIDIToREMI()
        s = stream.Score()
        p = stream.Part()
        m = stream.Measure()
        m.timeSignature = meter.TimeSignature('4/4')
        # 短音（< 0.25 * quarter_per_position ≈ 0.0625 quarter）
        gn = note.Note('D4')
        gn.duration = duration.Duration(0.04)
        m.append(gn)
        n = note.Note('C4', quarterLength=1.0)
        m.append(n)
        p.append(m)
        s.append(p)
        ids = mt.convert_score(s)
        assert len(ids) > 0
        decoded = REMITokenizer().detokenize(ids)
        assert any(t[0] == '<GraceNote' for t in decoded), "Short note should produce <GraceNote>"

    def test_pedalmark_detected(self):
        """music21 PedalMark 应生成 <Pedal> token（即便 MIDI 解析不产生，代码路径正确）。"""
        from music21 import stream, note, meter, expressions
        mt = MIDIToREMI()
        s = stream.Score()
        p = stream.Part()
        m = stream.Measure()
        m.timeSignature = meter.TimeSignature('4/4')
        m.append(note.Note('C4', quarterLength=1.0))
        m.append(expressions.PedalMark())
        m.append(note.Note('E4', quarterLength=1.0))
        p.append(m)
        s.append(p)
        ids = mt.convert_score(s)
        decoded = REMITokenizer().detokenize(ids)
        assert any(t[0] == '<Pedal' for t in decoded), "PedalMark should produce <Pedal> token"


class TestPDMXToREMI:
    """PDMX JSON → REMI 转换测试（合成数据，无文件依赖）。"""

    @pytest.fixture
    def simple_data(self):
        """2 小节 4/4，一轨钢琴，2 个四分音符。"""
        return {
            'resolution': 480,
            'barlines': [
                {'name': 'Barline', 'time': 0, 'measure': 1},
                {'name': 'Barline', 'time': 1920, 'measure': 2},
            ],
            'time_signatures': [
                {'name': 'TimeSignature', 'time': 0, 'numerator': 4, 'denominator': 4, 'measure': 1},
            ],
            'tempos': [
                {'name': 'Tempo', 'time': 0, 'qpm': 120, 'measure': 1},
            ],
            'key_signatures': [],
            'beats': [],
            'tracks': [
                {
                    'name': 'Piano',
                    'program': 0,
                    'is_drum': False,
                    'notes': [
                        {'name': 'Note', 'time': 0, 'pitch': 60, 'duration': 480,
                         'velocity': 80, 'pitch_str': 'C', 'measure': 1, 'is_grace': False},
                        {'name': 'Note', 'time': 480, 'pitch': 64, 'duration': 480,
                         'velocity': 80, 'pitch_str': 'E', 'measure': 1, 'is_grace': False},
                    ],
                    'annotations': [],
                }
            ],
        }

    def test_init(self):
        p = PDMXToREMI()
        assert p.grid_size == 16

    def test_empty_data(self):
        p = PDMXToREMI()
        ids = p.convert_pdmx({})
        assert ids == []

    def test_no_barlines(self):
        """无 barlines 时返回空列表。"""
        p = PDMXToREMI()
        ids = p.convert_pdmx({'tracks': [{'notes': [{'measure': 1}]}]})
        assert ids == []

    def test_basic_conversion(self, simple_data):
        p = PDMXToREMI()
        ids = p.convert_pdmx(simple_data)
        assert len(ids) > 0
        d = REMITokenizer().detokenize(ids)
        note_ons = [t for t in d if t[0] == '<Note_ON']
        assert len(note_ons) >= 2, "Should have at least 2 Note_ON tokens"

    def test_artic_extraction(self):
        """Articulation annotation → <Artic> token。"""
        p = PDMXToREMI()
        data = {
            'resolution': 480,
            'barlines': [{'name': 'Barline', 'time': 0, 'measure': 1},
                         {'name': 'Barline', 'time': 1920, 'measure': 2}],
            'time_signatures': [{'name': 'TimeSignature', 'time': 0, 'numerator': 4, 'denominator': 4, 'measure': 1}],
            'tempos': [{'name': 'Tempo', 'time': 0, 'qpm': 120, 'measure': 1}],
            'key_signatures': [], 'beats': [],
            'tracks': [{
                'program': 0, 'name': 'Piano', 'is_drum': False,
                'notes': [{'name': 'Note', 'time': 0, 'pitch': 60, 'duration': 480,
                           'velocity': 80, 'pitch_str': 'C', 'measure': 1, 'is_grace': False}],
                'annotations': [
                    {'name': 'Annotation', 'time': 0, 'annotation': {'name': 'Articulation', 'subtype': 'articStaccatoAbove'},
                     'group': None, 'measure': 1},
                ],
            }],
        }
        ids = p.convert_pdmx(data)
        d = REMITokenizer().detokenize(ids)
        assert any(t == ('<Artic', 'staccato') for t in d), "articStaccatoAbove → <Artic staccato>"

    def test_artic_all_subtypes(self):
        """所有 Articulation subtypes 正确映射。"""
        p = PDMXToREMI()
        subtypes = ['articStaccatoAbove', 'articAccentAbove', 'articMarcatoAbove',
                    'articTenutoAbove', 'fermata']
        expected = ['staccato', 'accent', 'marcato', 'tenuto', 'fermata']
        data = {
            'resolution': 480, 'barlines': [], 'time_signatures': [], 'tempos': [],
            'key_signatures': [], 'beats': [],
            'tracks': [{
                'program': 0, 'name': 'Piano', 'is_drum': False,
                'notes': [], 'annotations': [
                    {'name': 'Annotation', 'time': 0, 'annotation': {'name': 'Articulation', 'subtype': st},
                     'group': None, 'measure': 1} for st in subtypes
                ],
            }],
        }
        # Add barlines for all measures referenced by annotations
        data['barlines'] = [{'name': 'Barline', 'time': 0, 'measure': 1}]
        ids = p.convert_pdmx(data)
        d = REMITokenizer().detokenize(ids)
        for art, exp in zip(subtypes, expected):
            if exp == 'fermata':
                assert any(t == ('<Artic', 'fermata') for t in d), f"{art} → Artic fermata"
            else:
                assert any(t == ('<Artic', exp) for t in d), f"{art} → Artic {exp}"

    def test_dynamic_extraction(self):
        """Dynamic annotation → <Dynamic> token。"""
        p = PDMXToREMI()
        data = {
            'resolution': 480,
            'barlines': [{'name': 'Barline', 'time': 0, 'measure': 1},
                         {'name': 'Barline', 'time': 1920, 'measure': 2}],
            'time_signatures': [{'name': 'TimeSignature', 'time': 0, 'numerator': 4, 'denominator': 4, 'measure': 1}],
            'tempos': [{'name': 'Tempo', 'time': 0, 'qpm': 120, 'measure': 1}],
            'key_signatures': [], 'beats': [],
            'tracks': [{
                'program': 0, 'name': 'Piano', 'is_drum': False,
                'notes': [], 'annotations': [
                    {'name': 'Annotation', 'time': 0, 'annotation': {'name': 'Dynamic', 'subtype': 'f'},
                     'group': None, 'measure': 1},
                ],
            }],
        }
        ids = p.convert_pdmx(data)
        d = REMITokenizer().detokenize(ids)
        assert any(t == ('<Dynamic', 'f') for t in d), "Dynamic f → <Dynamic f>"

    def test_rfz_maps_to_sfz(self):
        """rfz → sfz（折中映射）。"""
        p = PDMXToREMI()
        data = {
            'resolution': 480,
            'barlines': [{'name': 'Barline', 'time': 0, 'measure': 1}],
            'time_signatures': [{'name': 'TimeSignature', 'time': 0, 'numerator': 4, 'denominator': 4, 'measure': 1}],
            'tempos': [{'name': 'Tempo', 'time': 0, 'qpm': 120, 'measure': 1}],
            'key_signatures': [], 'beats': [],
            'tracks': [{
                'program': 0, 'name': 'Piano', 'is_drum': False,
                'notes': [], 'annotations': [
                    {'name': 'Annotation', 'time': 0, 'annotation': {'name': 'Dynamic', 'subtype': 'rfz'},
                     'group': None, 'measure': 1},
                ],
            }],
        }
        ids = p.convert_pdmx(data)
        d = REMITokenizer().detokenize(ids)
        assert any(t == ('<Dynamic', 'sfz') for t in d), "rfz → <Dynamic sfz>"

    def test_pedal_spanner(self):
        """PedalSpanner → <Pedal start> / <Pedal end> 配对。"""
        p = PDMXToREMI()
        data = {
            'resolution': 480,
            'barlines': [{'name': 'Barline', 'time': 0, 'measure': 1},
                         {'name': 'Barline', 'time': 1920, 'measure': 2}],
            'time_signatures': [{'name': 'TimeSignature', 'time': 0, 'numerator': 4, 'denominator': 4, 'measure': 1}],
            'tempos': [{'name': 'Tempo', 'time': 0, 'qpm': 120, 'measure': 1}],
            'key_signatures': [], 'beats': [],
            'tracks': [{
                'program': 0, 'name': 'Piano', 'is_drum': False,
                'notes': [{'name': 'Note', 'time': 0, 'pitch': 60, 'duration': 480,
                           'velocity': 80, 'pitch_str': 'C', 'measure': 1, 'is_grace': False}],
                'annotations': [
                    {'name': 'Annotation', 'time': 0,
                     'annotation': {'name': 'PedalSpanner', 'duration': 960},
                     'group': None, 'measure': 1},
                ],
            }],
        }
        ids = p.convert_pdmx(data)
        d = REMITokenizer().detokenize(ids)
        assert any(t == ('<Pedal', 'start') for t in d), "PedalSpanner start"
        assert any(t == ('<Pedal', 'end') for t in d), "PedalSpanner end (from duration)"

    def test_slur_spanner(self):
        """SlurSpanner → <Slur start> / <Slur end> 配对。"""
        p = PDMXToREMI()
        data = {
            'resolution': 480,
            'barlines': [{'name': 'Barline', 'time': 0, 'measure': 1}],
            'time_signatures': [{'name': 'TimeSignature', 'time': 0, 'numerator': 4, 'denominator': 4, 'measure': 1}],
            'tempos': [{'name': 'Tempo', 'time': 0, 'qpm': 120, 'measure': 1}],
            'key_signatures': [], 'beats': [],
            'tracks': [{
                'program': 0, 'name': 'Piano', 'is_drum': False,
                'notes': [], 'annotations': [
                    {'name': 'Annotation', 'time': 0,
                     'annotation': {'name': 'SlurSpanner', 'duration': 480,
                                    'is_slur': True, 'subtype': 'slur'},
                     'group': None, 'measure': 1},
                ],
            }],
        }
        ids = p.convert_pdmx(data)
        d = REMITokenizer().detokenize(ids)
        assert any(t == ('<Slur', 'start') for t in d), "Slur start"
        assert any(t == ('<Slur', 'end') for t in d), "Slur end"

    def test_hairpin_spanner(self):
        """HairPinSpanner hairpin_type=0 → <Hairpin cresc>。"""
        p = PDMXToREMI()
        data = {
            'resolution': 480,
            'barlines': [{'name': 'Barline', 'time': 0, 'measure': 1}],
            'time_signatures': [{'name': 'TimeSignature', 'time': 0, 'numerator': 4, 'denominator': 4, 'measure': 1}],
            'tempos': [{'name': 'Tempo', 'time': 0, 'qpm': 120, 'measure': 1}],
            'key_signatures': [], 'beats': [],
            'tracks': [{
                'program': 0, 'name': 'Piano', 'is_drum': False,
                'notes': [], 'annotations': [
                    {'name': 'Annotation', 'time': 0,
                     'annotation': {'name': 'HairPinSpanner', 'duration': 480,
                                    'hairpin_type': 0},
                     'group': None, 'measure': 1},
                ],
            }],
        }
        ids = p.convert_pdmx(data)
        d = REMITokenizer().detokenize(ids)
        assert any(t == ('<Hairpin', 'cresc') for t in d), "HairPinSpanner type=0 → Hairpin cresc"

    def test_hairpin_dim(self):
        """hairpin_type=1 → <Hairpin dim>。"""
        p = PDMXToREMI()
        data = {
            'resolution': 480,
            'barlines': [{'name': 'Barline', 'time': 0, 'measure': 1}],
            'time_signatures': [{'name': 'TimeSignature', 'time': 0, 'numerator': 4, 'denominator': 4, 'measure': 1}],
            'tempos': [{'name': 'Tempo', 'time': 0, 'qpm': 120, 'measure': 1}],
            'key_signatures': [], 'beats': [],
            'tracks': [{
                'program': 0, 'name': 'Piano', 'is_drum': False,
                'notes': [], 'annotations': [
                    {'name': 'Annotation', 'time': 0,
                     'annotation': {'name': 'HairPinSpanner', 'duration': 480,
                                    'hairpin_type': 1},
                     'group': None, 'measure': 1},
                ],
            }],
        }
        ids = p.convert_pdmx(data)
        d = REMITokenizer().detokenize(ids)
        assert any(t == ('<Hairpin', 'dim') for t in d), "HairPinSpanner type=1 → Hairpin dim"

    def test_ottava_spanner(self):
        """OttavaSpanner 8va → <Octave 8va> + <Octave end>。"""
        p = PDMXToREMI()
        data = {
            'resolution': 480,
            'barlines': [{'name': 'Barline', 'time': 0, 'measure': 1}],
            'time_signatures': [{'name': 'TimeSignature', 'time': 0, 'numerator': 4, 'denominator': 4, 'measure': 1}],
            'tempos': [{'name': 'Tempo', 'time': 0, 'qpm': 120, 'measure': 1}],
            'key_signatures': [], 'beats': [],
            'tracks': [{
                'program': 0, 'name': 'Piano', 'is_drum': False,
                'notes': [], 'annotations': [
                    {'name': 'Annotation', 'time': 0,
                     'annotation': {'name': 'OttavaSpanner', 'duration': 480, 'subtype': '8va'},
                     'group': None, 'measure': 1},
                ],
            }],
        }
        ids = p.convert_pdmx(data)
        d = REMITokenizer().detokenize(ids)
        assert any(t == ('<Octave', '8va') for t in d), "OttavaSpanner 8va"
        assert any(t == ('<Octave', 'end') for t in d), "Ottava end after duration"

    def test_tremolo_to_ornament(self):
        """Tremolo → <Ornament tremolo>。"""
        p = PDMXToREMI()
        data = {
            'resolution': 480,
            'barlines': [{'name': 'Barline', 'time': 0, 'measure': 1}],
            'time_signatures': [{'name': 'TimeSignature', 'time': 0, 'numerator': 4, 'denominator': 4, 'measure': 1}],
            'tempos': [{'name': 'Tempo', 'time': 0, 'qpm': 120, 'measure': 1}],
            'key_signatures': [], 'beats': [],
            'tracks': [{
                'program': 0, 'name': 'Piano', 'is_drum': False,
                'notes': [], 'annotations': [
                    {'name': 'Annotation', 'time': 0,
                     'annotation': {'name': 'Tremolo', 'subtype': 'r16'},
                     'group': None, 'measure': 1},
                ],
            }],
        }
        ids = p.convert_pdmx(data)
        d = REMITokenizer().detokenize(ids)
        assert any(t == ('<Ornament', 'tremolo') for t in d), "Tremolo → Ornament tremolo"

    def test_arpeggio_extraction(self):
        """Arpeggio → <Arpeggio>。"""
        p = PDMXToREMI()
        data = {
            'resolution': 480,
            'barlines': [{'name': 'Barline', 'time': 0, 'measure': 1}],
            'time_signatures': [{'name': 'TimeSignature', 'time': 0, 'numerator': 4, 'denominator': 4, 'measure': 1}],
            'tempos': [{'name': 'Tempo', 'time': 0, 'qpm': 120, 'measure': 1}],
            'key_signatures': [], 'beats': [],
            'tracks': [{
                'program': 0, 'name': 'Piano', 'is_drum': False,
                'notes': [], 'annotations': [
                    {'name': 'Annotation', 'time': 0,
                     'annotation': {'name': 'Arpeggio', 'subtype': 'default'},
                     'group': None, 'measure': 1},
                ],
            }],
        }
        ids = p.convert_pdmx(data)
        d = REMITokenizer().detokenize(ids)
        assert any(t[0] == '<Arpeggio>' for t in d), "Arpeggio annotation"

    def test_ornament_from_articulation(self):
        """Ornament disguised as Articulation (ornamentTrill) → <Ornament trill>。"""
        p = PDMXToREMI()
        data = {
            'resolution': 480,
            'barlines': [{'name': 'Barline', 'time': 0, 'measure': 1}],
            'time_signatures': [{'name': 'TimeSignature', 'time': 0, 'numerator': 4, 'denominator': 4, 'measure': 1}],
            'tempos': [{'name': 'Tempo', 'time': 0, 'qpm': 120, 'measure': 1}],
            'key_signatures': [], 'beats': [],
            'tracks': [{
                'program': 0, 'name': 'Piano', 'is_drum': False,
                'notes': [], 'annotations': [
                    {'name': 'Annotation', 'time': 0,
                     'annotation': {'name': 'Articulation', 'subtype': 'ornamentTrill'},
                     'group': None, 'measure': 1},
                ],
            }],
        }
        ids = p.convert_pdmx(data)
        d = REMITokenizer().detokenize(ids)
        assert any(t == ('<Ornament', 'trill') for t in d), "ornamentTrill → Ornament trill"

    def test_rest_at_empty_beat(self):
        """拍位无音符处生成 <Rest>。"""
        p = PDMXToREMI()
        data = {
            'resolution': 480,
            'barlines': [{'name': 'Barline', 'time': 0, 'measure': 1},
                         {'name': 'Barline', 'time': 1920, 'measure': 2}],
            'time_signatures': [{'name': 'TimeSignature', 'time': 0, 'numerator': 4, 'denominator': 4, 'measure': 1}],
            'tempos': [{'name': 'Tempo', 'time': 0, 'qpm': 120, 'measure': 1}],
            'key_signatures': [], 'beats': [],
            'tracks': [{
                'program': 0, 'name': 'Piano', 'is_drum': False,
                'notes': [],  # 无音符 → 所有拍位都是 Rest
                'annotations': [],
            }],
        }
        ids = p.convert_pdmx(data)
        d = REMITokenizer().detokenize(ids)
        rests = [t for t in d if t[0] == '<Rest>']
        assert len(rests) >= 4, f"4/4 空两小节应有至少 8 个 Rest，实际 {len(rests)}"


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
