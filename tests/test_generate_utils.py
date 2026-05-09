from chopinote_model.generate import (
    _parse_program,
    _parse_subtrack,
    get_polyphony_cap,
    KEY_TO_DIATONIC_PITCHES,
    GM_INSTRUMENT_RANGES,
    INSTRUMENT_POLYPHONY_CAP,
)


class TestParseProgram:
    def test_basic(self):
        assert _parse_program('<Program 0>') == 0

    def test_large_number(self):
        assert _parse_program('<Program 127>') == 127

    def test_with_subtrack(self):
        assert _parse_program('<Program 40_2>') == 40

    def test_typical_piano(self):
        assert _parse_program('<Program 0_1>') == 0


class TestParseSubtrack:
    def test_no_subtrack_returns_0(self):
        assert _parse_subtrack('<Program 0>') == 0

    def test_subtrack_explicit(self):
        assert _parse_subtrack('<Program 0_2>') == 2

    def test_max_subtrack(self):
        assert _parse_subtrack('<Program 40_3>') == 3


class TestGetPolyphonyCap:
    def test_piano(self):
        assert get_polyphony_cap(0) == 10
        assert get_polyphony_cap(1) == 10
        assert get_polyphony_cap(7) == 10

    def test_strings(self):
        assert get_polyphony_cap(40) == 2
        assert get_polyphony_cap(41) == 2

    def test_brass(self):
        assert get_polyphony_cap(56) == 2
        assert get_polyphony_cap(57) == 2

    def test_woodwind(self):
        assert get_polyphony_cap(68) == 2
        assert get_polyphony_cap(71) == 2

    def test_bass(self):
        assert get_polyphony_cap(32) == 2

    def test_unknown_fallback(self):
        assert get_polyphony_cap(999) == 10


class TestKeyToDiatonicPitches:
    def test_c_major(self):
        cmajor = KEY_TO_DIATONIC_PITCHES['C']
        assert 0 in cmajor
        assert 1 not in cmajor
        assert 2 in cmajor
        assert 3 not in cmajor
        assert 4 in cmajor
        assert 5 in cmajor
        assert 6 not in cmajor
        assert 7 in cmajor
        assert 8 not in cmajor
        assert 9 in cmajor
        assert 10 not in cmajor
        assert 11 in cmajor

    def test_f_major_has_flat(self):
        fmajor = KEY_TO_DIATONIC_PITCHES['F']
        assert 10 in fmajor
        assert 11 not in fmajor

    def test_am_natural_minor(self):
        amin = KEY_TO_DIATONIC_PITCHES['Am']
        assert amin == KEY_TO_DIATONIC_PITCHES['C']

    def test_all_30_keys_present(self):
        expected = [
            'C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'F', 'Bb', 'Eb', 'Ab', 'Db', 'Gb', 'Cb',
            'Am', 'Em', 'Bm', 'F#m', 'C#m', 'G#m', 'D#m', 'A#m', 'Dm', 'Gm', 'Cm', 'Fm', 'Bbm', 'Ebm', 'Abm',
        ]
        for k in expected:
            assert k in KEY_TO_DIATONIC_PITCHES


class TestGMInstrumentRanges:
    def test_piano(self):
        assert GM_INSTRUMENT_RANGES[0] == (21, 108)

    def test_violin(self):
        assert GM_INSTRUMENT_RANGES[40] == (55, 103)

    def test_cello(self):
        assert GM_INSTRUMENT_RANGES[42] == (36, 76)

    def test_trumpet(self):
        assert GM_INSTRUMENT_RANGES[56] == (54, 89)

    def test_percussion_not_restricted(self):
        assert 114 not in GM_INSTRUMENT_RANGES
