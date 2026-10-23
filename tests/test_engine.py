# Tests for ScoliosisModelEngine's level names and curve apexes, which the
# app panel, the viewer labels and the exported JPG all use.

from modules.model_mock import ScoliosisModelEngine

from synthetic_spine import EXPECTED_CURVES, MARKER_INDEX, make_result


def loaded_engine():
    engine = ScoliosisModelEngine()
    engine.load_from_dict(make_result())
    return engine


def test_levels_are_counted_up_from_l5():
    engine = loaded_engine()
    last = len(engine.get_detections()) - 1
    assert engine.get_level(last) == "L5"
    assert engine.get_level(MARKER_INDEX) == "C7"
    assert engine.get_level(None) == "?"
    assert engine.get_level(last + 1) == "?"


def test_each_curve_apex_lies_inside_its_curve():
    apexes = loaded_engine().get_curve_apexes()
    assert len(apexes) == len(EXPECTED_CURVES)
    for (upper, lower, _), apex in zip(EXPECTED_CURVES, apexes):
        assert upper < apex < lower


def test_the_curve_containing_the_whole_spine_apex_uses_it():
    engine = loaded_engine()
    main_apex, _ = engine.get_apex()
    for (upper, lower, _), apex in zip(EXPECTED_CURVES, engine.get_curve_apexes()):
        if upper < main_apex < lower:
            assert apex == main_apex
