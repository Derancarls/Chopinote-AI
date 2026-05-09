import copy

from chopinote_dataset.processor import PDMXPreprocessor


def _make_pdmx_data(*pitches):
    return {
        'tracks': [
            {
                'program': 0,
                'name': 'Piano',
                'notes': [
                    {'pitch': p, 'velocity': 64, 'measure': 1, 'time': 0, 'duration': 120}
                    for p in pitches
                ],
            },
        ],
    }


def _all_pitches(pdmx_data):
    result = []
    for track in pdmx_data.get('tracks', []):
        for note in track.get('notes', []):
            result.append(note['pitch'])
    return result


class TestTransposePositive:
    def test_transpose_up_5(self):
        preprocessor = PDMXPreprocessor()
        original = _make_pdmx_data(60, 64, 67)
        result = preprocessor._transpose_pdmx(original, 5)
        assert _all_pitches(result) == [65, 69, 72]

    def test_transpose_single_note_up(self):
        preprocessor = PDMXPreprocessor()
        result = preprocessor._transpose_pdmx(_make_pdmx_data(72), 12)
        assert _all_pitches(result) == [84]


class TestTransposeNegative:
    def test_transpose_down_3(self):
        preprocessor = PDMXPreprocessor()
        original = _make_pdmx_data(60, 64, 67)
        result = preprocessor._transpose_pdmx(original, -3)
        assert _all_pitches(result) == [57, 61, 64]

    def test_transpose_down_to_boundary(self):
        preprocessor = PDMXPreprocessor()
        result = preprocessor._transpose_pdmx(_make_pdmx_data(5), -5)
        assert _all_pitches(result) == [0]


class TestTransposeZero:
    def test_transpose_unchanged(self):
        preprocessor = PDMXPreprocessor()
        original = _make_pdmx_data(60, 64, 67)
        # semitone=0 is handled by caller (skipped), but method itself works
        result = preprocessor._transpose_pdmx(original, 0)
        assert _all_pitches(result) == [60, 64, 67]


class TestTransposeOutOfBounds:
    def test_oob_high_returns_none(self):
        preprocessor = PDMXPreprocessor()
        result = preprocessor._transpose_pdmx(_make_pdmx_data(120), 200)
        assert result is None

    def test_oob_low_returns_none(self):
        preprocessor = PDMXPreprocessor()
        result = preprocessor._transpose_pdmx(_make_pdmx_data(5), -200)
        assert result is None

    def test_borderline_high(self):
        preprocessor = PDMXPreprocessor()
        assert preprocessor._transpose_pdmx(_make_pdmx_data(127), 0) is not None


class TestTransposeNoMutation:
    def test_original_not_mutated(self):
        preprocessor = PDMXPreprocessor()
        original = _make_pdmx_data(60, 64)
        original_copy = copy.deepcopy(original)
        preprocessor._transpose_pdmx(original, 3)
        assert original == original_copy
