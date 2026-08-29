"""Normalization of mono and stereo sonic vectors.

`normalize_stereo`'s four combinations of `remove_bias` and `normalize_sep`
were untested, and they behave quite differently from one another.
"""

import numpy as np
import pytest

from music.core.functions import normalize_mono, normalize_stereo


def test_mono_all_zero_is_returned_untouched():
    """There is nothing to scale by, and dividing would give NaN."""
    zeros = np.zeros(5)
    assert np.array_equal(normalize_mono(zeros), zeros)


def test_mono_centres_and_scales_by_the_larger_peak():
    assert np.allclose(normalize_mono([0.0, 1.0, 2.0]), [-1.0, 0.0, 1.0])


def test_mono_without_bias_removal_maps_the_range_onto_the_full_scale():
    """remove_bias=False is an affine map of [min, max] onto [-1, 1], which
    fills the range but stretches an asymmetric waveform."""
    out = normalize_mono([0.0, 1.0, 4.0], remove_bias=False)
    assert out.min() == pytest.approx(-1.0)
    assert out.max() == pytest.approx(1.0)


def test_stereo_all_zero_is_returned_untouched():
    zeros = np.zeros((2, 4))
    assert np.array_equal(normalize_stereo(zeros), zeros)


def _stereo():
    return np.vstack((np.array([0.0, 1.0, 2.0]),
                      np.array([0.0, 0.5, 1.0])))


def test_stereo_together_keeps_the_balance_between_channels():
    """The default scales both channels by one factor, so the quieter
    channel stays quieter."""
    out = normalize_stereo(_stereo())
    assert np.allclose(out[0], [-1.0, 0.0, 1.0])
    assert np.allclose(out[1], [-0.5, 0.0, 0.5])


def test_stereo_separately_brings_each_channel_to_full_scale():
    """normalize_sep=True discards the balance, by design."""
    out = normalize_stereo(_stereo(), normalize_sep=True)
    assert np.allclose(out[0], [-1.0, 0.0, 1.0])
    assert np.allclose(out[1], [-1.0, 0.0, 1.0])


def test_stereo_without_bias_removal_maps_the_range():
    out = normalize_stereo(_stereo(), remove_bias=False)
    assert out.min() == pytest.approx(-1.0)
    assert out.max() == pytest.approx(1.0)


def test_stereo_without_bias_removal_separately():
    out = normalize_stereo(_stereo(), remove_bias=False, normalize_sep=True)
    for channel in out:
        assert channel.min() == pytest.approx(-1.0)
        assert channel.max() == pytest.approx(1.0)


@pytest.mark.parametrize("remove_bias", [True, False])
@pytest.mark.parametrize("normalize_sep", [True, False])
def test_stereo_always_lands_inside_full_scale(remove_bias, normalize_sep):
    rng = np.random.default_rng(0)
    signal = np.vstack((rng.uniform(-3, 5, 64), rng.uniform(-1, 2, 64)))

    out = normalize_stereo(signal, remove_bias=remove_bias,
                           normalize_sep=normalize_sep)

    assert out.min() >= -1.0 - 1e-12
    assert out.max() <= 1.0 + 1e-12


@pytest.mark.parametrize("value", [1.0, -3.0, 0.0])
@pytest.mark.parametrize("remove_bias", [True, False])
def test_a_constant_mono_signal_normalizes_to_silence(value, remove_bias):
    """Regression: a constant signal has no dynamic range, so the scale
    factor is zero and dividing by it filled the result with NaN. Only the
    all-zero case was guarded."""
    out = normalize_mono(np.full(5, value), remove_bias=remove_bias)
    assert np.array_equal(out, np.zeros(5))


@pytest.mark.parametrize("normalize_sep", [True, False])
@pytest.mark.parametrize("remove_bias", [True, False])
def test_a_constant_stereo_signal_normalizes_to_silence(normalize_sep,
                                                        remove_bias):
    out = normalize_stereo(np.ones((2, 4)), remove_bias=remove_bias,
                           normalize_sep=normalize_sep)
    assert np.array_equal(out, np.zeros((2, 4)))


def test_one_flat_channel_does_not_poison_the_other():
    """Normalizing separately, a silent channel must not make the live one
    NaN."""
    signal = np.vstack((np.ones(3), np.array([0.0, 1.0, 2.0])))
    out = normalize_stereo(signal, normalize_sep=True)
    assert np.array_equal(out[0], np.zeros(3))
    assert np.allclose(out[1], [-1.0, 0.0, 1.0])


def test_normalizing_never_produces_nan():
    """The writers refuse NaN, so this is the last line before an error."""
    rng = np.random.default_rng(0)
    for signal in (np.ones(8), np.zeros(8), rng.uniform(-1, 1, 8),
                   np.full(8, -2.5)):
        assert np.isfinite(normalize_mono(signal)).all()
        stereo = np.vstack((signal, signal))
        assert np.isfinite(normalize_stereo(stereo)).all()


def test_a_mono_vector_given_to_the_stereo_normalizer_is_promoted():
    """Regression: a 1-D array had its first two *samples* read as the two
    channels. The mean of a scalar is itself, so those two samples were
    silently zeroed and the rest scaled wrongly."""
    signal = np.array([0.5, -0.3, 0.9, -0.7, 0.2])

    out = normalize_stereo(signal)

    assert out.shape == (2, len(signal))
    assert np.array_equal(out[0], out[1])
    assert np.allclose(out[0], normalize_mono(signal))


def test_the_stereo_writer_accepts_a_mono_vector(tmp_path):
    """It writes a genuine stereo file rather than a corrupted mono one."""
    import music

    tone = music.note(440, 0.05)
    path = tmp_path / "mono_in.wav"
    music.write_wav_stereo(tone, filename=str(path))

    restored = music.read_wav(str(path))
    assert restored.shape[0] == 2
    assert np.allclose(restored[0], restored[1])
