from chopinote_dataset.tokenizer import REMITokenizer


class TestVocabSize:
    def test_vocab_size(self, tokenizer):
        assert tokenizer.vocab_size == 837

    def test_grid16_vel8_determines_vocab(self):
        t1 = REMITokenizer(grid_size=16, velocity_levels=8)
        t2 = REMITokenizer(grid_size=16, velocity_levels=8)
        assert t1.vocab_size == t2.vocab_size


class TestSpecialTokenIDs:
    def test_special_ids(self, tokenizer):
        assert tokenizer.pad_token_id == 0
        assert tokenizer.bos_token_id == 1
        assert tokenizer.eos_token_id == 2
        assert tokenizer.bar_token_id == 4
        assert tokenizer.encode_token('<MASK>') == 3

    def test_decode_specials(self, tokenizer):
        assert tokenizer.decode_token(0) == '<PAD>'
        assert tokenizer.decode_token(1) == '<BOS>'
        assert tokenizer.decode_token(2) == '<EOS>'
        assert tokenizer.decode_token(3) == '<MASK>'
        assert tokenizer.decode_token(4) == '<Bar>'


class TestEncodeDecodeRoundtrip:
    def test_position(self, tokenizer):
        for i in range(16):
            t = f'<Position {i}>'
            assert tokenizer.decode_token(tokenizer.encode_token(t)) == t

    def test_note_on(self, tokenizer):
        for p in [0, 60, 127]:
            t = f'<Note_ON {p}>'
            assert tokenizer.decode_token(tokenizer.encode_token(t)) == t

    def test_velocity(self, tokenizer):
        for v in range(8):
            t = f'<Velocity {v}>'
            assert tokenizer.decode_token(tokenizer.encode_token(t)) == t

    def test_duration(self, tokenizer):
        for d in range(1, 17):
            t = f'<Duration {d}>'
            assert tokenizer.decode_token(tokenizer.encode_token(t)) == t

    def test_program(self, tokenizer):
        for p in [0, 40, 127]:
            t = f'<Program {p}>'
            assert tokenizer.decode_token(tokenizer.encode_token(t)) == t
            for sub in range(1, 4):
                t_sub = f'<Program {p}_{sub}>'
                assert tokenizer.decode_token(tokenizer.encode_token(t_sub)) == t_sub

    def test_key(self, tokenizer):
        for k in ['C', 'Am', 'F#', 'Bb', 'Ebm']:
            t = f'<Key {k}>'
            assert tokenizer.decode_token(tokenizer.encode_token(t)) == t

    def test_tempo(self, tokenizer):
        for bpm in [30, 60, 120, 240]:
            t = f'<Tempo {bpm}>'
            assert tokenizer.decode_token(tokenizer.encode_token(t)) == t

    def test_markings(self, tokenizer):
        for t in ['<Clef treble>', '<Clef bass>', '<Dynamic f>', '<Dynamic ppp>',
                   '<Hairpin cresc>', '<Hairpin dim>', '<Artic staccato>', '<Artic fermata>',
                   '<Ornament trill>', '<Ornament tremolo>', '<Pedal start>', '<Pedal end>',
                   '<Slur start>', '<Slur end>', '<TupletEnd>', '<Rest>', '<Arpeggio>',
                   '<Octave 8va>', '<Octave end>']:
            assert tokenizer.decode_token(tokenizer.encode_token(t)) == t

    def test_timesig(self, tokenizer):
        for ts in ['4/4', '3/4', '6/8']:
            t = f'<TimeSig {ts}>'
            assert tokenizer.decode_token(tokenizer.encode_token(t)) == t


class TestTokenizeDetokenizeRoundtrip:
    def test_simple_sequence(self, tokenizer):
        events = [
            ('<BOS>', None), ('<Key', 'C'), ('<TimeSig', '4/4'), ('<Tempo', 120),
            ('<Bar>', None), ('<Position', 0), ('<Program', '0'),
            ('<Note_ON', 60), ('<Velocity', 4), ('<Duration', 8),
            ('<EOS>', None),
        ]
        ids = tokenizer.tokenize(events)
        recovered = tokenizer.detokenize(ids)
        assert recovered == events

    def test_multitrack(self, tokenizer):
        events = [
            ('<BOS>', None), ('<Bar>', None), ('<Position', 0),
            ('<Program', '0'), ('<Note_ON', 60), ('<Velocity', 4), ('<Duration', 8),
            ('<Program', '0_1'), ('<Note_ON', 36), ('<Velocity', 3), ('<Duration', 16),
            ('<EOS>', None),
        ]
        ids = tokenizer.tokenize(events)
        recovered = tokenizer.detokenize(ids)
        assert recovered == events


class TestUnknownToken:
    def test_unknown_returns_mask(self, tokenizer):
        assert tokenizer.encode_token('<NotARealToken>') == 3

    def test_out_of_range_decode(self, tokenizer):
        assert tokenizer.decode_token(99999) == '<MASK>'


class TestAllTokenIDs:
    def test_all_ids_decode_to_nonempty(self, tokenizer):
        for tid in range(tokenizer.vocab_size):
            token = tokenizer.decode_token(tid)
            assert token and len(token) > 0
            if tid == tokenizer.encode_token('<MASK>'):
                continue  # MASK token itself decodes to MASK, by design
            assert token != '<MASK>'
