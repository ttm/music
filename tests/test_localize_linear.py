"""Localizing a sound along a straight path.

The cues are checked against the geometry they come from, and against
`localize`, which computes the same two cues for a single fixed position.
"""

import numpy as np
import pytest

import music
from music.core.filters.localization import localize_linear

SAMPLE_RATE = 44100
ZETA = 0.215
SPEED = 331.3 + .606 * 20


def _ear_distances(theta_degrees, dist=1.0, zeta=ZETA):
    """Distance from each ear to a source at that angle.

    x is the lateral axis, so 0 degrees is fully to the right and 90 is
    straight ahead. The left ear sits at -zeta/2, the right at +zeta/2.
    """
    x = np.cos(np.radians(theta_degrees)) * dist
    y = np.sin(np.radians(theta_degrees)) * dist
    return np.hypot(x + zeta / 2, y), np.hypot(x - zeta / 2, y)


def _lag(first, second):
    """How far `first` lags `second`, to a fraction of a sample.

    The correlation peak is interpolated with a parabola through its
    neighbours, since the delays here are not whole numbers of samples.
    """
    correlation = np.correlate(first - first.mean(),
                               second - second.mean(), "full")
    peak = int(np.argmax(correlation))
    if 0 < peak < len(correlation) - 1:
        before, at, after = correlation[peak - 1:peak + 2]
        peak = peak + .5 * (before - after) / (before - 2 * at + after)
    return peak - (len(second) - 1)


def _noise(length=8192, seed=0):
    """Non-periodic, so the correlation peak is unambiguous."""
    return np.random.default_rng(seed).uniform(-1, 1, length)


def _band_limited(length=8192, seed=0, cutoff=4000):
    """Noise with no energy near Nyquist.

    Gain has to be measured on band-limited content: any fractional-delay
    interpolator attenuates the top of the spectrum, so broadband noise
    would conflate the interaural gain with the interpolator's own loss.
    """
    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft(rng.uniform(-1, 1, length))
    freqs = np.fft.rfftfreq(length, 1 / SAMPLE_RATE)
    spectrum[freqs > cutoff] = 0
    signal = np.fft.irfft(spectrum, n=length)
    return signal / np.abs(signal).max()


# --------------------------------------------------------------------------
# Shape
# --------------------------------------------------------------------------

def test_the_output_is_stereo_and_keeps_the_input_length():
    signal = music.note(duration=0.1)
    out = localize_linear(signal)
    assert out.shape == (2, len(signal))
    assert np.isfinite(out).all()


def test_it_synthesizes_a_note_when_given_nothing():
    assert localize_linear().shape[0] == 2


@pytest.mark.parametrize("length", [0, 1, 2])
def test_degenerate_lengths(length):
    out = localize_linear(np.ones(length))
    assert out.shape == (2, length)


# --------------------------------------------------------------------------
# A path that stays put must reproduce the static geometry
# --------------------------------------------------------------------------

@pytest.mark.parametrize("theta", [0, 45, 90, 135, 180])
def test_a_stationary_path_matches_the_interaural_geometry(theta):
    """Both cues follow from the two ear distances: the farther ear is
    delayed by the extra distance and attenuated by the ratio."""
    signal = _band_limited()
    out = localize_linear(signal, theta1=theta, theta2=theta, dist=1.0)

    left, right = _ear_distances(theta)
    nearest = min(left, right)
    expected_delay = (max(left, right) - nearest) * SAMPLE_RATE / SPEED
    expected_gain = nearest / max(left, right)

    assert abs(abs(_lag(out[0], out[1])) - expected_delay) < 0.2

    farther, nearer = (0, 1) if left > right else (1, 0)
    measured_gain = np.sqrt((out[farther] ** 2).mean()
                            / (out[nearer] ** 2).mean())
    # A few parts per thousand of slack: the interpolator still costs a
    # little at the top of the band, even band-limited.
    assert measured_gain == pytest.approx(expected_gain, rel=5e-3)


@pytest.mark.parametrize("theta, leading", [(0, "right"), (180, "left")])
def test_the_nearer_ear_hears_it_first(theta, leading):
    out = localize_linear(_noise(), theta1=theta, theta2=theta, dist=1.0)
    lag = _lag(out[0], out[1])
    assert (lag > 0) == (leading == "right")


def test_straight_ahead_reaches_both_ears_alike():
    out = localize_linear(_noise(), theta1=90, theta2=90, dist=1.0)
    assert np.allclose(out[0], out[1])


# --------------------------------------------------------------------------
# Agreement with localize, which computes the same cues for one position
# --------------------------------------------------------------------------

@pytest.mark.parametrize("theta", [0, 30, 150, 180])
def test_a_stationary_path_agrees_with_localize(theta):
    """`localize` pads its output rather than resampling, so the arrays
    differ in length -- but the two cues it applies must be the same."""
    x = np.cos(np.radians(theta))   # localize branches on the sign of x
    left, right = _ear_distances(theta)

    # What localize does, read off its own source.
    localize_gain = right / left          # applied to the far ear
    localize_delay = int((left - right) / SPEED * SAMPLE_RATE)

    out = localize_linear(_band_limited(), theta1=theta, theta2=theta,
                          dist=1.0)
    farther, nearer = (0, 1) if left > right else (1, 0)
    measured_gain = np.sqrt((out[farther] ** 2).mean()
                            / (out[nearer] ** 2).mean())

    assert measured_gain == pytest.approx(
        localize_gain if x > 0 else 1 / localize_gain, rel=5e-3
    )
    assert abs(abs(_lag(out[0], out[1])) - abs(localize_delay)) < 1.0


# --------------------------------------------------------------------------
# Movement
# --------------------------------------------------------------------------

@pytest.mark.parametrize("progress", [0.1, 0.5, 0.9])
def test_the_cues_track_the_source_along_the_path(progress):
    """Sampled in a narrow window, the local delay must match the position
    the source has reached by then."""
    signal = _noise(44100 * 2)
    out = localize_linear(signal, theta1=90, theta2=0, dist=1.0)

    index = int(progress * len(signal))
    window = slice(index - 1024, index + 1024)

    # The straight line from (0, 1) to (1, 0).
    fraction = index / (len(signal) - 1)
    x, y = fraction, 1 - fraction
    left = np.hypot(x + ZETA / 2, y)
    right = np.hypot(x - ZETA / 2, y)
    expected = (left - right) * SAMPLE_RATE / SPEED

    assert _lag(out[0][window], out[1][window]) == pytest.approx(
        expected, abs=0.5
    )


def test_a_source_crossing_the_midline_swaps_which_ear_leads():
    signal = _noise(44100)
    out = localize_linear(signal, theta1=0, theta2=180, dist=1.0)
    quarter = len(signal) // 4

    early = _lag(out[0][:quarter], out[1][:quarter])
    late = _lag(out[0][-quarter:], out[1][-quarter:])
    assert early > 0 > late


def test_reversing_the_path_mirrors_the_channels():
    signal = _noise(8192)
    forward = localize_linear(signal, theta1=0, theta2=180, dist=1.0)
    backward = localize_linear(signal, theta1=180, theta2=0, dist=1.0)
    # Reversing the sweep swaps left for right at every point in time.
    assert np.allclose(forward[0], backward[1], atol=1e-9)
    assert np.allclose(forward[1], backward[0], atol=1e-9)


# --------------------------------------------------------------------------
# It attenuates, it never amplifies
# --------------------------------------------------------------------------

@pytest.mark.parametrize("theta1, theta2", [(0, 180), (90, 0), (45, 45)])
def test_no_channel_carries_more_energy_than_the_source(theta1, theta2):
    """Both gains are a ratio of distances to the nearer one, so neither
    can exceed unity.

    Measured as energy rather than peak: a bandlimited interpolator
    reconstructs the peaks *between* the original samples, which are
    genuinely higher than any sample, so the peak may rise a little even
    though nothing is amplified.
    """
    signal = _noise()
    out = localize_linear(signal, theta1=theta1, theta2=theta2, dist=1.0)

    source_energy = (signal ** 2).mean()
    for channel in out:
        assert (channel ** 2).mean() <= source_energy + 1e-12


def test_the_output_stays_continuous_as_the_source_moves():
    """A delay that jumped by whole samples would step audibly."""
    signal = music.note(220, 1.0)
    out = localize_linear(signal, theta1=0, theta2=180, dist=1.0)
    assert np.abs(np.diff(out[0])).max() <= 3 * np.abs(np.diff(signal)).max()
