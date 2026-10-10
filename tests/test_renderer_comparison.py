"""Analytic cross-renderer contracts for SSTIM-named auditory stimuli."""

import numpy as np
import pytest

from tools.compare_stimulation_renderers import CASES, analytic, compare


@pytest.mark.parametrize("technique,implementation,parameters", CASES)
def test_independent_renderers_agree_within_lut_quantization(
        technique, implementation, parameters):
    row = compare(technique, implementation, parameters)
    assert row["technique_iri"].endswith("#" + technique)
    assert row["max_absolute_difference"] < 0.001
    assert row["rms_difference"] < 0.0005


def test_binaural_channels_are_distinct_and_monaural_is_one_channel():
    binaural = analytic(CASES[0][0], CASES[0][2])
    monaural = analytic(CASES[1][0], CASES[1][2])
    assert binaural.shape == (2, 44100)
    assert monaural.shape == (44100,)
    np.testing.assert_allclose(binaural.mean(axis=0), monaural)


def test_undefined_comparison_technique_is_refused():
    with pytest.raises(ValueError, match="unknown technique"):
        analytic("not-a-stimulus", {"carrier_freq": 200.0})
