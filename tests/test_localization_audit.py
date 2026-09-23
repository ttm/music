"""Localization cues, read sample by sample.

From the mutation audit of :mod:`music.core.filters.localization`. The
geometry the article gives -- a distance to each ear, a delay and an
intensity ratio from their difference -- is computed here independently
and compared with what each routine renders.
"""

import warnings

import numpy as np
import pytest

import music
from music.core.filters.localization import _delayed, _localize_positions

SPEED = 331.3 + 0.606 * 20
ZETA = 0.215


def _ears(x, y, zeta=ZETA):
    """Distances from a source at (x, y) to the left and right ears."""
    return np.hypot(x + zeta / 2, y), np.hypot(x - zeta / 2, y)


# --------------------------------------------------------------------------
# what the audit corrected
# --------------------------------------------------------------------------

@pytest.mark.parametrize("routine", ["localize_linear", "spatial_motion"])
@pytest.mark.parametrize("azimuth, ear", [(0, 1), (180, 0)])
def test_a_source_on_an_ear_is_heard_by_that_ear_alone(routine, azimuth,
                                                       ear):
    """At distance zero from an ear the intensity ratio was 0 / 0, and the
    channel came back NaN. Its limit is one for that ear and zero for the
    other, which is also what `localize` renders there."""
    sound = np.linspace(.1, .5, 5)
    if routine == "localize_linear":
        placed = music.localize_linear(sound, theta1=azimuth,
                                       theta2=azimuth, dist=ZETA / 2)
    else:
        placed = music.spatial_motion(sonic_vector=sound, motion_rate=0,
                                      theta1=azimuth, theta2=azimuth,
                                      dist=ZETA / 2)
    expected = np.zeros((2, 5))
    expected[ear] = sound
    np.testing.assert_allclose(placed, expected, atol=1e-15)
    side = ZETA / 2 * np.cos(np.radians(azimuth))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        still = music.localize(sound, x=side, y=0)
    np.testing.assert_allclose(still[:, :5], placed, atol=1e-15)


@pytest.mark.parametrize("ear", [0, 1])
def test_a_source_exactly_on_either_ear_is_heard_by_it_alone(ear):
    """At 180 degrees the sine of pi is not quite zero, so the routines
    never put a source exactly on the left ear; the positions can."""
    sound = np.linspace(.1, .5, 5)
    x = np.full(5, (ZETA / 2) * (1 if ear else -1))
    placed = _localize_positions(sound, x, np.zeros(5), ZETA, 20, 44100)
    expected = np.zeros((2, 5))
    expected[ear] = sound
    np.testing.assert_array_equal(placed, expected)


def test_the_far_ear_hears_nothing_until_the_sound_arrives():
    """A click at the start reached the far ear as a plateau.

    Reads before the first sample took the first sample, so the far ear
    held it for the whole interaural delay: 27 samples of a constant,
    where the sound had not yet arrived at all.
    """
    click = np.r_[1.0, np.zeros(63)]
    placed = music.localize_linear(click, theta1=0, theta2=0, dist=1.0)
    far, near = _ears(1.0, 0.0)
    delay = (far - near) * 44100 / SPEED
    np.testing.assert_array_equal(placed[1], click)
    assert not placed[0][:int(delay) - 1].any()
    assert int(np.argmax(np.abs(placed[0]))) in (int(delay),
                                                 int(delay) + 1)


def _catmull_rom(signal, position):
    """The textbook spline through a signal extended by silence."""
    padded = np.r_[np.zeros(3), signal, np.zeros(3)]
    index = np.floor(position).astype(int)
    t = position - index
    p0, p1, p2, p3 = (padded[index + 3 + offset] for offset in (-1, 0, 1, 2))
    return .5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3)
                 * t ** 2 + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3)


def test_the_delay_line_is_the_catmull_rom_spline_through_silence():
    """Every sample, the first and last included, for delays of whole and
    fractional samples: the signal is silent before and after itself."""
    rng = np.random.default_rng(9)
    signal = rng.standard_normal(40)
    delay = rng.uniform(0, 3, 40)
    delay[::7] = np.floor(delay[::7])
    # Reads within a sample of the end, whose last taps fall past it.
    delay[-3:] = .5, .25, .75
    np.testing.assert_allclose(
        _delayed(signal, delay),
        _catmull_rom(signal, np.arange(40) - delay), atol=1e-12)


def test_the_delay_line_is_silent_outside_the_signal():
    """Before the first sample and after the last, so the tail of a
    delayed read tapers rather than holding the final sample."""
    np.testing.assert_allclose(_delayed(np.ones(4), np.full(4, 2.0)),
                               [0, 0, 1, 1])
    np.testing.assert_allclose(_delayed(np.ones(4), np.full(4, 10.0)),
                               np.zeros(4))


def test_localize_places_a_list_as_it_places_an_array():
    """Documented as array_like, it multiplied a list by the intensity
    ratio and raised TypeError."""
    sound = [.1, .2, .3]
    for x in (1, -1):
        np.testing.assert_array_equal(
            music.localize(sound, x=x, y=0),
            music.localize(np.array(sound), x=x, y=0))


# --------------------------------------------------------------------------
# the fractional delay
# --------------------------------------------------------------------------

@pytest.mark.parametrize("delay", [.5, .25, 1.75])
def test_the_interpolator_reproduces_a_parabola_between_its_samples(delay):
    """Catmull-Rom is exact for a quadratic, so a fractional read of
    ``k ** 2`` is ``(k - delay) ** 2`` wherever its four taps are inside.
    Its cubic term vanishes there, and any edit to it does not."""
    k = np.arange(16.)
    read = _delayed(k ** 2, np.full(16, delay))
    inside = (k - delay >= 1) & (k - delay <= 13)
    np.testing.assert_allclose(read[inside], (k[inside] - delay) ** 2)


@pytest.mark.parametrize("air_temp", [0, 20, 35])
def test_the_far_ear_lags_by_the_extra_path_at_the_speed_of_sound(air_temp):
    """A ramp passes through the interpolator exactly, so the far ear's
    samples are the ramp shifted by the delay, to rounding."""
    ramp = np.arange(200.)
    x, y = .6, .4
    placed = _localize_positions(ramp, np.full(200, x), np.full(200, y),
                                 ZETA, air_temp, 44100)
    far, near = _ears(x, y)
    delay = (far - near) * 44100 / (331.3 + .606 * air_temp)
    k = np.arange(200.)
    inside = (k - delay >= 1) & (k - delay <= 196)
    np.testing.assert_allclose(placed[0][inside],
                               near / far * (k[inside] - delay), rtol=1e-12)
    np.testing.assert_array_equal(placed[1], ramp)


# --------------------------------------------------------------------------
# localize: one fixed position
# --------------------------------------------------------------------------

@pytest.mark.parametrize("x", [.5, -.5])
def test_localize_delays_and_attenuates_the_far_ear_on_either_side(x):
    """Nothing measured the intensity on the left, where the far ear is
    the right one and is scaled by the reciprocal of the ratio."""
    sound = np.linspace(-1, 1, 50)
    left, right = _ears(x, .2)
    shift = int(abs(left - right) * 44100 / SPEED)
    near = np.r_[sound, np.zeros(shift)]
    far = np.r_[np.zeros(shift), sound * min(left, right) / max(left, right)]
    expected = [far, near] if x > 0 else [near, far]
    np.testing.assert_allclose(music.localize(sound, x=x, y=.2), expected,
                               rtol=1e-12)


def test_localize_places_an_angle_at_the_distance_given():
    """Every test used a distance of one, where multiplying by it and
    dividing by it agree."""
    sound = np.linspace(-1, 1, 30)
    angle = np.radians(30)
    np.testing.assert_allclose(
        music.localize(sound, theta=30, distance=2),
        music.localize(sound, x=2 * np.cos(angle), y=2 * np.sin(angle)))


def test_an_angle_without_a_distance_leaves_the_source_at_the_centre():
    """As documented: `distance` must also be given for theta to act."""
    sound = np.linspace(-1, 1, 30)
    np.testing.assert_array_equal(music.localize(sound, theta=30),
                                  [sound, sound])


@pytest.mark.parametrize("whole, offset", [(19, -1e-4), (20, 1e-4)])
def test_localize_truncates_its_delay_to_whole_samples(whole, offset):
    """On the ear axis the path difference is ``zeta`` itself, so a zeta
    a hair either side of 20 samples at 44.1 kHz and 20 degrees shows
    both the default rate and the default temperature."""
    zeta = (20 + offset) * SPEED / 44100
    placed = music.localize(np.ones(10), x=1, y=0, zeta=zeta)
    assert placed.shape == (2, 10 + whole)


def test_a_bare_localize_is_the_one_its_defaults_declare():
    np.testing.assert_array_equal(
        music.localize(),
        music.localize(music.note(), x=.1, y=.01, zeta=.215, air_temp=20,
                       sample_rate=44100))


# --------------------------------------------------------------------------
# localize_linear: a straight path
# --------------------------------------------------------------------------

@pytest.mark.parametrize("count", [2, 3, 50])
def test_a_linear_path_runs_from_its_first_angle_to_its_last(count):
    """At the first sample theta1 and at the last theta2, so two samples
    are the two endpoints and nothing between."""
    sound = np.linspace(.2, 1, count)
    first, last = np.radians(150), np.radians(20)
    x = np.linspace(.7 * np.cos(first), .7 * np.cos(last), count)
    y = np.linspace(.7 * np.sin(first), .7 * np.sin(last), count)
    np.testing.assert_allclose(
        music.localize_linear(sound, theta1=150, theta2=20, dist=.7),
        _localize_positions(sound, x, y, ZETA, 20, 44100),
        rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("sound", [
    np.random.default_rng(5).standard_normal(300).astype(np.float32),
    np.arange(300) % 5 > 2,
])
def test_a_linear_path_moves_any_sound_as_its_float64_values(sound):
    """Single precision interpolates differently, and booleans cannot be
    subtracted, so the conversion is not a formality."""
    np.testing.assert_array_equal(
        music.localize_linear(sound, theta1=160, theta2=10, dist=.5),
        music.localize_linear(sound.astype(np.float64), theta1=160,
                              theta2=10, dist=.5))


def test_a_bare_linear_path_is_the_one_its_defaults_declare():
    np.testing.assert_array_equal(
        music.localize_linear(),
        music.localize_linear(music.note(), theta1=90, theta2=0, dist=.1,
                              zeta=.215, air_temp=20, sample_rate=44100))


# --------------------------------------------------------------------------
# localize_hrtf
# --------------------------------------------------------------------------

@pytest.mark.parametrize("ear", ["left", "right"])
def test_an_empty_response_is_named_by_its_ear(ear):
    responses = {"left_hrir": [1.], "right_hrir": [1.], f"{ear}_hrir": []}
    with pytest.raises(ValueError, match=f"the {ear} impulse response"):
        music.localize_hrtf(np.ones(4), **responses)


# --------------------------------------------------------------------------
# localize2: per-frequency cues
# --------------------------------------------------------------------------

def _tone(freq, count=4410, sample_rate=44100, shift=0., phase=0.):
    """A sine at an exact FFT bin when ``freq`` divides into the grid."""
    return np.sin(2 * np.pi * freq * (np.arange(count) / sample_rate
                                      - shift) + phase)


def _cues(freq, theta):
    """The delay and gain localize2 documents for one frequency."""
    lateral = abs(np.sin(np.radians(theta)))
    coefficient = .3 if freq <= 4000 else .2
    return (coefficient * ZETA * lateral / SPEED,
            1 + (freq / 1000) ** .8 * lateral)


@pytest.mark.parametrize("freq, count, sample_rate", [
    (10, 4410, 44100), (400, 4410, 44100), (4000, 4410, 44100),
    (4001, 16000, 16000), (5000, 4410, 44100), (16000, 4410, 44100),
])
@pytest.mark.parametrize("theta", [-70, 40])
def test_the_ifft_method_delays_the_far_ear_and_amplifies_the_near(
        freq, count, sample_rate, theta):
    """A tone at an exact bin, where the circular phase shift is an exact
    delay. 10 Hz is the lowest bin, 4000 and 4001 Hz straddle the
    crossover, and 16 kHz lies above a third of the spectrum."""
    itd, iid = _cues(freq, theta)
    placed = music.localize2(_tone(freq, count, sample_rate), theta=theta,
                             sample_rate=sample_rate)
    near, far = (placed[1], placed[0]) if theta < 0 else placed
    np.testing.assert_allclose(near, iid * _tone(freq, count, sample_rate),
                               atol=1e-9)
    np.testing.assert_allclose(far, _tone(freq, count, sample_rate, itd),
                               atol=1e-9)


def test_a_position_uses_the_default_coordinates_when_none_are_given():
    sound = _tone(400, 441)
    np.testing.assert_array_equal(
        music.localize2(sound, theta=None),
        music.localize2(sound, theta=None, x=.1, y=.01))


@pytest.mark.parametrize("routine, zero, position", [
    ("localize", dict(theta=0, distance=1), dict(x=1, y=0)),
    ("localize2", dict(theta=0), dict(theta=None, x=0, y=1)),
])
def test_an_angle_of_zero_is_an_angle(routine, zero, position):
    """Zero read as "no angle given" and fell back to the default
    position: for `localize` the right ear's side became (0.1, 0.01), and
    for `localize2` straight ahead became a source far to one side."""
    sound = _tone(400, 441)
    np.testing.assert_allclose(getattr(music, routine)(sound, **zero),
                               getattr(music, routine)(sound, **position),
                               atol=1e-12)


def _brute(sound, theta=-70, sample_rate=44100):
    with pytest.warns(UserWarning, match="long time"):
        return music.localize2(sound, theta=theta, method="brute",
                               sample_rate=sample_rate)


def _resynthesized(partials, count=4410, sample_rate=44100, theta=-70,
                   phase=0.):
    """The brute-force answer, from each partial's own delay and gain."""
    longest = abs(int(sample_rate * .3 * ZETA
                      * abs(np.sin(np.radians(theta))) / SPEED))
    left, right = np.zeros(count + longest), np.zeros(count + longest)
    for freq, amplitude in partials:
        itd, iid = _cues(freq, theta)
        shift = abs(int(sample_rate * itd))
        tone = amplitude * _tone(freq, count, sample_rate, phase=phase)
        left[shift:shift + count] += tone
        right[:count] += iid * tone
    return np.array([left, right])


@pytest.mark.parametrize("freq, count, sample_rate", [
    (10, 4410, 44100), (400, 4410, 44100), (4000, 4410, 44100),
    (4001, 16000, 16000), (14700, 4410, 44100), (400, 2205, 22050),
])
@pytest.mark.parametrize("phase", [0, np.pi / 2, 2.])
def test_brute_force_resynthesizes_each_partial_in_phase(freq, count,
                                                         sample_rate, phase):
    """A sine came back as minus a cosine, a quarter cycle early, because
    the FFT's angles are a cosine's and the resynthesis read a sine
    table. 14.7 kHz is a third of the spectrum, the bin a mutated Nyquist
    test would have halved.

    The table limits the waveform to a few parts in ten thousand, but the
    two ears are the same resynthesized samples, so the gain between
    them is exact."""
    expected = _resynthesized([(freq, 1)], count, sample_rate, phase=phase)
    placed = _brute(_tone(freq, count, sample_rate, phase=phase),
                    sample_rate=sample_rate)
    assert placed.shape == expected.shape
    np.testing.assert_allclose(placed, expected,
                               atol=1e-3 * np.abs(expected).max())
    itd, iid = _cues(freq, -70)
    shift = abs(int(sample_rate * itd))
    far, near = placed[0][shift:shift + count], placed[1][:count]
    np.testing.assert_allclose(near, iid * far, rtol=1e-12, atol=1e-9)


@pytest.mark.parametrize("partials, kept", [
    ([(400, 1), (410, .07)], [(400, 1)]),
    ([(400, 1), (410, .15)], [(400, 1), (410, .15)]),
    ([(390, .07), (400, 1)], [(390, .07), (400, 1)]),
])
def test_brute_force_drops_the_last_percent_of_energy_counted_upwards(
        partials, kept):
    """A partial above the rest holding half a percent of the energy is
    dropped, and one holding two percent is kept; the same weak partial
    below the strong one is kept, because the energy is counted from the
    lowest frequency up."""
    sound = sum(amplitude * _tone(freq) for freq, amplitude in partials)
    expected = _resynthesized(kept)
    np.testing.assert_allclose(_brute(sound), expected,
                               atol=1e-3 * np.abs(expected).max())


@pytest.mark.parametrize("routine", ["localize", "localize_linear",
                                     "localize2"])
def test_the_default_sound_is_rendered_at_the_rate_asked_for(routine):
    """Two seconds of the default note, at whatever rate is localized to.
    It was rendered at 44.1 kHz regardless: at 8 kHz, eleven seconds of a
    note at 40 Hz rather than 220."""
    rendered = getattr(music, routine)(sample_rate=8000)
    given = getattr(music, routine)(music.note(sample_rate=8000),
                                    sample_rate=8000)
    np.testing.assert_array_equal(rendered, given)
    assert rendered.shape[-1] < 2 * 8000 + 100


@pytest.mark.parametrize("method", ["ifft", "brute"])
def test_localize2_places_nothing_as_nothing(method):
    """As the other localizers do; its FFT refused zero points."""
    assert music.localize2(np.array([]), method=method).shape == (2, 0)


def test_localize2_places_a_single_sample_in_both_ears():
    """One sample has no frequencies but DC, which the cues leave alone."""
    np.testing.assert_allclose(music.localize2(np.array([.5]), theta=40),
                               [[.5], [.5]])
