"""Document realistic oversampling support for legacy moving-pitch APIs.

These smoke tests assert operational compatibility, not that every
dynamic-pitch path becomes alias-free or has phase-equivalent output.
"""

import numpy as np
import pytest

import music


@pytest.mark.parametrize("rate", [44100, 48000, 96000])
@pytest.mark.parametrize("name,parameters", [
    ("note_with_fm", {"freq": 8000, "fm": 900,
                      "max_fm_deviation": 7}),
    ("note_with_vibrato", {
        "freq": 8000, "vibrato_freq": 11, "max_pitch_dev": 3}),
    ("note_with_glissando", {
        "start_freq": 2000, "end_freq": 11000, "method": "lin"}),
])
def test_existing_pitch_paths_work_with_explicit_oversampling(
        rate, name, parameters):
    # Opt-in resampling does not change legacy default behavior.
    generator = getattr(music, name)
    output = music.render_oversampled(
        generator, sample_rate=rate, number_of_samples=2048,
        factor=4, **parameters)
    legacy = generator(sample_rate=rate, number_of_samples=2048,
                       **parameters)
    assert output.shape == legacy.shape == (2048,)
    assert np.isfinite(output).all()
    assert np.isfinite(legacy).all()
    assert not np.array_equal(output, legacy)
