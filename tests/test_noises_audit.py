"""What the `noises` mutation audit found untested or wrong."""

import math
import re

import numpy as np
import pytest

import music


def _exactly(message):
    return "^" + re.escape(message) + "$"


def _components(signal):
    """The indices of the nonzero components below or at Nyquist."""
    spectrum = np.abs(np.fft.fft(signal))[:len(signal) // 2 + 1]
    return np.nonzero(spectrum > 1e-9 * spectrum.max())[0]


@pytest.mark.parametrize("length", [100, 101])
@pytest.mark.parametrize("low, high, first, last", [
    (10, 20, 10, 20),        # on components: both kept
    (10.5, 20.5, 11, 20),    # between: the ones inside
    (9.99, 20.01, 10, 20),
    (0, 30, 1, 30),          # never the constant
    (20, 20, 20, 20),        # one component
    (9.7, 20.7, 10, 20),     # a fraction above a half rounds nowhere
])
def test_the_band_holds_the_components_from_its_low_end_to_its_high(
        length, low, high, first, last):
    """At 1 Hz a component. The band was rounded down at both ends, which
    let in 10 Hz under a low end of 10.5 and always left out the high
    end."""
    np.random.seed(3)
    produced = music.noise("white", number_of_samples=length,
                           sample_rate=length, min_freq=low, max_freq=high)
    np.testing.assert_array_equal(_components(produced),
                                  np.arange(first, last + 1))


def test_an_edge_the_resolution_divides_is_not_lost_to_rounding():
    """0.3 * 10 / 1 is 3.0000000000000004 in floating point."""
    np.random.seed(5)
    produced = music.noise("white", number_of_samples=10, sample_rate=1,
                           min_freq=0.3, max_freq=0.3)
    np.testing.assert_array_equal(_components(produced), [3])


def test_the_band_reaches_the_nyquist_component():
    np.random.seed(7)
    produced = music.noise("white", number_of_samples=64, sample_rate=64,
                           min_freq=1, max_freq=32)
    np.testing.assert_array_equal(_components(produced), np.arange(1, 33))


@pytest.mark.parametrize("slope", [math.nan, math.inf, -math.inf])
def test_a_slope_that_is_not_a_number_of_decibels_is_refused(slope):
    """NaN rendered NaN, and infinity NaN behind a warning."""
    with pytest.raises(ValueError, match=_exactly(
            "noise_type must be a finite number of decibels per octave; "
            f"got {slope}")):
        music.noise(slope, duration=0.1)


@pytest.mark.parametrize("sample_rate", [0, -1])
@pytest.mark.parametrize("routine", [music.noise, music.gaussian_noise])
def test_a_noise_needs_a_positive_sample_rate(routine, sample_rate):
    """With number_of_samples given, a rate of zero divided by zero."""
    with pytest.raises(ValueError, match=_exactly(
            f"sample_rate must be positive; got {sample_rate}")):
        if routine is music.noise:
            routine("white", number_of_samples=100, sample_rate=sample_rate)
        else:
            routine(sample_rate=sample_rate)


def test_a_band_upside_down_is_refused():
    """It rendered silence."""
    with pytest.raises(ValueError, match=_exactly(
            "min_freq (5000 Hz) is above max_freq (100 Hz), so the band "
            "between them is empty")):
        music.noise("white", min_freq=5000, max_freq=100)


def test_a_band_above_the_nyquist_frequency_is_refused():
    with pytest.raises(ValueError, match=_exactly(
            "min_freq (600 Hz) is above the Nyquist frequency (500.0 Hz), "
            "so the band holds nothing")):
        music.noise("white", min_freq=600, max_freq=700, sample_rate=1000)


def test_a_band_starting_at_the_nyquist_frequency_is_that_component():
    np.random.seed(1)
    produced = music.noise("white", number_of_samples=10, sample_rate=10,
                           min_freq=5, max_freq=5)
    np.testing.assert_array_equal(_components(produced), [5])


def test_what_was_wrong_is_said_even_for_no_samples():
    """The colour was read after the length, so a noise of no samples
    accepted any name."""
    with pytest.raises(ValueError, match="^Set ntype"):
        music.noise("mauve", duration=0)


def test_a_noise_too_short_for_its_band_is_silence():
    produced = music.noise("white", number_of_samples=2, sample_rate=100,
                           min_freq=10, max_freq=20)
    np.testing.assert_array_equal(produced, [0.0, 0.0])


@pytest.mark.parametrize("std", [0, -0.5])
def test_a_gaussian_band_needs_a_width(std):
    with pytest.raises(ValueError, match=_exactly(
            f"std is the width of the band and must be positive; got "
            f"{std}. A band of no width holds no frequency to draw a noise "
            "from")):
        music.gaussian_noise(std=std)


@pytest.mark.parametrize("length, mean, std, first, last", [
    # At a rate of `length`, a component is 1 Hz and the length a second.
    (1000, 0.1, 0.02, 270, 330),    # 270 to 330 Hz
    (1001, 0.1, 0.02, 270, 330),
    (1000, 0.15, 0.1, 300, 499),    # to 600 Hz: stops below Nyquist's 500
    (1001, 0.15, 0.1, 300, 500),    # an odd length has 500 and no Nyquist
    (1000, 0.16, 0.1, 330, 499),
])
def test_a_gaussian_band_is_flat_and_stops_at_the_nyquist_frequency(
        length, mean, std, first, last):
    """The band was zeroed after the spectrum was mirrored, so one that
    reached past the Nyquist frequency kept mirrored components, at twice
    the level of the rest."""
    np.random.seed(11)
    produced = music.gaussian_noise(mean=mean, std=std, duration=1,
                                    sample_rate=length)
    assert len(produced) == length
    spectrum = np.abs(np.fft.fft(produced))[:length // 2 + 1]
    band = spectrum[first:last + 1]
    np.testing.assert_allclose(band, band[0], rtol=1e-9)
    outside = np.concatenate((spectrum[:first], spectrum[last + 1:]))
    assert outside.max() < 1e-9 * band[0]


def test_a_gaussian_band_entirely_past_the_nyquist_frequency_is_refused():
    with pytest.raises(ValueError, match="holds no frequency"):
        music.gaussian_noise(mean=0.5, std=0.1, duration=1,
                             sample_rate=1000)


@pytest.mark.parametrize("duration", [-1, -0.5, 0, 1e-9])
def test_a_silence_too_short_for_a_sample_is_empty(duration):
    """A negative one raised numpy's "negative dimensions"."""
    assert music.silence(duration).shape == (0,)


def test_a_silence_counts_its_samples_at_its_rate():
    assert music.silence(0.25, sample_rate=8).shape == (2,)
    assert not music.silence(0.5, sample_rate=10).any()


# --------------------------------------------------------------------------
# What the survivors showed unasserted
# --------------------------------------------------------------------------

@pytest.mark.parametrize("length", [64, 65])
def test_white_noise_is_flat_up_to_its_highest_component(length):
    """For an even length that is the Nyquist component, set on its own;
    for an odd one the last conjugate pair, which was never drawn a phase
    and so was always silent."""
    np.random.seed(2)
    produced = music.noise("white", number_of_samples=length,
                           sample_rate=length, min_freq=1,
                           max_freq=length / 2)
    spectrum = np.abs(np.fft.fft(produced))[1:length // 2 + 1]
    np.testing.assert_allclose(spectrum, spectrum[0], rtol=1e-9)


@pytest.mark.parametrize("length", [64, 65])
def test_each_noise_component_takes_its_phase_from_one_draw(length):
    """The phases are the uniform draws in order, one a component."""
    np.random.seed(29)
    draws = np.random.uniform(0, 2 * np.pi, length // 2)
    np.random.seed(29)
    produced = music.noise("pink", number_of_samples=length,
                           sample_rate=length, min_freq=1, max_freq=20)
    phases = np.angle(np.fft.fft(produced))[1:21]
    np.testing.assert_allclose(np.exp(1j * phases),
                               np.exp(1j * draws[1:21]), atol=1e-9)


@pytest.mark.parametrize("length", [64, 65])
def test_each_gaussian_component_takes_its_phase_from_one_draw(length):
    np.random.seed(31)
    draws = np.random.uniform(0, 2 * np.pi, length)
    np.random.seed(31)
    produced = music.gaussian_noise(mean=0.005, std=0.006, duration=1,
                                    sample_rate=length)
    phases = np.angle(np.fft.fft(produced))[7:25]   # 6 to 24 Hz, inside
    np.testing.assert_allclose(np.exp(1j * phases),
                               np.exp(1j * draws[7:25]), atol=1e-9)


def test_a_bare_noise_is_brown_over_its_default_band():
    """Two seconds at 44.1 kHz, 15 to 15,000 Hz at 0.5 Hz a component."""
    np.random.seed(17)
    produced = music.noise()
    assert len(produced) == 88200
    np.testing.assert_array_equal(_components(produced),
                                  np.arange(30, 30001))
    spectrum = np.abs(np.fft.fft(produced))
    # -6 dB an octave: 30 Hz, at index 60, is 10 ** (-6 / 20) of 15 Hz.
    assert spectrum[60] / spectrum[30] == pytest.approx(10 ** (-6 / 20),
                                                        rel=1e-9)


def test_a_bare_gaussian_band_is_2250_to_3750_hz():
    np.random.seed(19)
    produced = music.gaussian_noise(duration=1, sample_rate=8000)
    np.testing.assert_array_equal(_components(produced),
                                  np.arange(2250, 3751))


def test_a_gaussian_band_may_start_at_the_lowest_component():
    np.random.seed(23)
    produced = music.gaussian_noise(mean=0.05, std=0.1, duration=1,
                                    sample_rate=1000)
    np.testing.assert_array_equal(_components(produced),
                                  np.arange(1, 301))


def test_a_gaussian_noise_at_one_hertz_is_a_noise():
    produced = music.gaussian_noise(mean=1e-4, std=1e-4, duration=100,
                                    sample_rate=1)
    np.testing.assert_array_equal(_components(produced),
                                  np.arange(15, 46))


@pytest.mark.parametrize("mean, std, duration, low, high, resolution", [
    (0.09025, 0.0001, 1, 270.6, 270.9, 1.0),   # between two components
    (0.1, 0.02, 1 / 1000, 270.0, 330.0, 1000.0),  # one sample
])
def test_a_gaussian_band_that_holds_no_component_says_so(
        mean, std, duration, low, high, resolution):
    with pytest.raises(ValueError, match=_exactly(
            f"mean={mean} and std={std} give a band from {low:.1f} Hz to "
            f"{high:.1f} Hz, which holds no frequency at a resolution of "
            f"{resolution:.1f} Hz. Widen std, or lengthen the noise")):
        music.gaussian_noise(mean=mean, std=std, duration=duration,
                             sample_rate=1000)


def test_a_bare_silence_is_one_second_at_44100_hz():
    assert music.silence().shape == (44100,)
    assert music.silence(1).shape == (44100,)


def test_a_bare_gaussian_noise_lasts_two_seconds():
    np.random.seed(37)
    assert music.gaussian_noise().shape == (88200,)


def test_an_unknown_colour_is_refused_with_the_names_there_are():
    with pytest.raises(ValueError, match=_exactly(
            "Set ntype to a number or one of the following strings: "
            "'white', 'pink', 'brown', 'blue', 'violet', 'black'. "
            "Check docstring for more information.")):
        music.noise("mauve")


@pytest.mark.parametrize("length, draws", [(64, 32), (65, 33)])
def test_a_noise_takes_one_draw_a_component_from_the_random_stream(
        length, draws):
    """An even length draws what it always did, so a seeded render that
    follows one is unchanged; an odd one draws its highest component
    too."""
    np.random.seed(41)
    music.noise("white", number_of_samples=length, sample_rate=length,
                min_freq=1, max_freq=length / 2)
    after = np.random.random()
    np.random.seed(41)
    np.random.uniform(0, 2 * np.pi, draws)
    assert after == np.random.random()
