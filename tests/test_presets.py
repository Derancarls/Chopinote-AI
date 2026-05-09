from chopinote_cli.presets import get_preset, list_presets, Preset


class TestGetPreset:
    def test_exists(self):
        p = get_preset('baroque')
        assert p is not None
        assert p.name == 'baroque'

    def test_missing(self):
        assert get_preset('nonexistent') is None

    def test_default_preset(self):
        p = get_preset('default')
        assert p is not None


class TestListPresets:
    def test_count(self):
        presets = list_presets()
        assert len(presets) == 7

    def test_all_have_names(self):
        for p in list_presets():
            assert p.name
            assert p.label


class TestPresetAttrs:
    def test_returns_dict(self):
        p = get_preset('romantic')
        attrs = p.attrs()
        assert isinstance(attrs, dict)
        assert 'complexity' in attrs
        assert 'temperature' in attrs
        assert 'top_k' in attrs
        assert 'lock_key' in attrs

    def test_default_values(self):
        p = get_preset('default')
        attrs = p.attrs()
        assert attrs['complexity'] == 5
        assert attrs['temperature'] == 1.0
        assert attrs['top_k'] == 20


class TestPresetConditions:
    def test_empty_for_default(self):
        p = get_preset('default')
        conds = p.conditions()
        assert conds == {}

    def test_key_condition(self):
        p = get_preset('minimal')
        conds = p.conditions()
        assert conds['tempo'] == 60

    def test_program_condition(self):
        p = get_preset('baroque')
        conds = p.conditions()
        assert conds['program'] == 6

    def test_church_conditions(self):
        p = get_preset('church')
        conds = p.conditions()
        assert conds['program'] == 19
