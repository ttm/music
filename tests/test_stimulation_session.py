"""Sessions: a protocol's timing is a promise, and these tests hold it.

A stimulation protocol is written as durations -- ten minutes here, five
there -- and the person who wrote them down means them. A session that
renders 9:58 because it spent two seconds crossfading has quietly
changed the protocol. The first test in this file is the one that
matters: the session lasts exactly the sum of its phases.

The rest are about the joins. A cut between two stimuli is a step
discontinuity, and a crossfade that dips is a hole; both are audible,
and neither is in the protocol either.
"""

import numpy as np
import pytest

import music
from music.stimulation.session import _ramp_shape

SR = 44100


def constant(number_of_samples=0, sample_rate=SR, level=1.0):
    """A stimulus of a fixed level, so envelopes are readable directly."""
    return np.full(number_of_samples, level)


def stereo_constant(number_of_samples=0, sample_rate=SR, level=1.0):
    """The same, in two channels."""
    return np.full((2, number_of_samples), level)


# --------------------------------------------------------------------------
# timing -- the promise the class makes
# --------------------------------------------------------------------------

@pytest.mark.parametrize('ramp', [0.0, 0.1, 0.5])
def test_a_session_lasts_the_sum_of_its_phases(ramp):
    """Whatever the ramps, the arithmetic the author did still holds."""
    session = music.StimulationSession()
    session.add(constant, duration=1.0)
    session.add(constant, duration=2.0, ramp=ramp)
    session.add(constant, duration=0.5, ramp=ramp)
    assert session.duration == pytest.approx(3.5)
    assert len(session.render()) == int(round(3.5 * SR))


def test_the_opening_and_closing_ramps_do_not_lengthen_the_session():
    """They overlap nothing, so they are carved out rather than added."""
    session = music.StimulationSession(end_ramp=0.25)
    session.add(constant, duration=1.0, ramp=0.25)
    assert session.duration == pytest.approx(1.0)


def test_many_phases_do_not_accumulate_a_rounding_drift():
    """Durations that do not land on whole samples, summed twenty times."""
    session = music.StimulationSession()
    for _ in range(20):
        session.add(constant, duration=0.037, ramp=0.011)
    expected = int(round(20 * 0.037 * SR))
    assert len(session.render()) == expected


def test_an_empty_session_renders_nothing_rather_than_failing():
    session = music.StimulationSession()
    assert session.duration == 0.0
    assert len(session.render()) == 0


# --------------------------------------------------------------------------
# the joins
# --------------------------------------------------------------------------

def test_a_crossfade_leaves_no_gap():
    """Whatever the shape, the level never reaches zero mid-session.

    A gap would mean the two extents did not line up, which is the
    failure that a crossfade is supposed to make impossible.
    """
    session = music.StimulationSession(ramp_shape='linear')
    session.add(constant, duration=0.5)
    session.add(constant, duration=0.5, ramp=0.2)
    out = session.render()
    assert out.min() > 0.99


def test_equal_power_ramps_sum_in_quadrature():
    """The property the default exists for, exactly rather than by ear.

    Two different stimuli are uncorrelated, so their amplitudes add in
    quadrature: holding the level across a transition means the two
    gain curves square to one. A linear pair does not -- it squares to
    0.5 in the middle, which is the 3 dB hole.
    """
    rising = _ramp_shape(512, 'equal_power', True)
    falling = _ramp_shape(512, 'equal_power', False)
    assert np.allclose(rising ** 2 + falling ** 2, 1.0)

    linear_rise = _ramp_shape(512, 'linear', True)
    linear_fall = _ramp_shape(512, 'linear', False)
    assert (linear_rise ** 2 + linear_fall ** 2).min() == pytest.approx(0.5)


def test_linear_ramps_sum_in_amplitude():
    """And the property the other option exists for."""
    rising = _ramp_shape(512, 'linear', True)
    falling = _ramp_shape(512, 'linear', False)
    assert np.allclose(rising + falling, 1.0)


def test_linear_holds_the_level_for_correlated_phases():
    """The case the other option is kept for.

    Two phases that are the same sound add in amplitude rather than in
    power, and there the linear pair is the flat one while equal power
    bumps by about 3 dB.
    """
    def render(shape):
        session = music.StimulationSession(ramp_shape=shape)
        session.add(constant, duration=0.5)
        session.add(constant, duration=0.5, ramp=0.2)
        return session.render()

    assert render('linear').max() == pytest.approx(1.0)
    assert render('equal_power').max() == pytest.approx(np.sqrt(2), abs=0.01)


def test_the_session_opens_from_silence_and_closes_to_it():
    session = music.StimulationSession(end_ramp=0.1)
    session.add(constant, duration=1.0, ramp=0.1)
    out = session.render()
    assert out[0] == pytest.approx(0.0)
    assert out[-1] < 0.02
    assert out[len(out) // 2] == pytest.approx(1.0)


def test_without_a_ramp_a_phase_starts_at_full_level():
    session = music.StimulationSession()
    session.add(constant, duration=0.5)
    out = session.render()
    assert out[0] == pytest.approx(1.0)


def test_a_ramp_longer_than_its_phase_is_clipped_to_it():
    """A 1 s ramp into a 0.1 s phase cannot run past the phase.

    It gives a shape rather than an IndexError or a silent phase.
    """
    session = music.StimulationSession(end_ramp=1.0)
    session.add(constant, duration=0.1, ramp=1.0)
    out = session.render()
    assert len(out) == int(round(0.1 * SR))
    assert np.all(np.isfinite(out))
    assert out.max() > 0


@pytest.mark.parametrize('shape', ['linear', 'equal_power'])
@pytest.mark.parametrize('stimulus', [constant, stereo_constant])
def test_oversized_crossfade_preserves_the_two_phase_protocol(shape, stimulus):
    session = music.StimulationSession(sample_rate=1000, ramp_shape=shape)
    session.add(stimulus, duration=0.1)
    session.add(stimulus, duration=0.1, ramp=1.0)
    out = session.render()
    assert out.shape == ((200,) if stimulus is constant else (2, 200))
    assert session.duration == 0.2
    if shape == 'linear':
        np.testing.assert_allclose(out, 1.0, atol=1e-15)
    else:
        assert out.min() >= 1.0 - 1e-15
        assert out.max() <= np.sqrt(2) + 1e-15


@pytest.mark.parametrize('shape', ['linear', 'equal_power'])
@pytest.mark.parametrize('stimulus', [constant, stereo_constant])
@pytest.mark.parametrize('durations, ramp, rate', [
    ([1.0, 0.1, 1.0], 1.0, 1000),
    ([0.10, 0.03, 0.10], 0.09, 100),
])
def test_short_middle_phase_cannot_make_overlapping_crossfades(
        shape, stimulus, durations, ramp, rate):
    session = music.StimulationSession(sample_rate=rate, ramp_shape=shape)
    for index, duration in enumerate(durations):
        session.add(stimulus, duration=duration, ramp=ramp if index else 0)
    out = session.render()
    assert out.shape[-1] == round(sum(durations) * rate)
    if shape == 'linear':
        np.testing.assert_allclose(out, 1.0, atol=1e-15)
    else:
        assert out.min() >= 1.0 - 1e-15
        assert out.max() <= np.sqrt(2) + 1e-15


@pytest.mark.parametrize('shape', ['linear', 'equal_power'])
@pytest.mark.parametrize('as_array', [False, True])
def test_competing_outer_ramps_keep_both_ends_quiet(shape, as_array):
    session = music.StimulationSession(sample_rate=1000, end_ramp=1.0,
                                       ramp_shape=shape)
    session.add(np.ones(100) if as_array else constant,
                duration=0.1, ramp=1.0)
    out = session.render()
    assert out.shape == (100,)
    assert out[0] == 0.0
    assert out[-1] < 0.04
    assert out.max() == pytest.approx(1.0)


@pytest.mark.parametrize('shape', ['linear', 'equal_power'])
@pytest.mark.parametrize('lengths, ramp', [([10, 10], 25),
                                        ([100, 10, 100], 100)])
@pytest.mark.parametrize('stereo', [False, True])
def test_short_array_phases_keep_their_samples_and_matching_ramps(
        shape, lengths, ramp, stereo):
    session = music.StimulationSession(sample_rate=100, ramp_shape=shape)
    for index, length in enumerate(lengths):
        sound = np.ones((2, length)) if stereo else np.ones(length)
        session.add(sound, ramp=ramp / 100 if index else 0)
    out = session.render()
    # Two ten-sample arrays share ten samples; the three-array case
    # shares five on each side of its short middle array.
    expected = 10 if len(lengths) == 2 else 200
    assert out.shape == ((2, expected) if stereo else (expected,))
    assert session.duration == expected / 100
    if shape == 'linear':
        np.testing.assert_allclose(out, 1.0, atol=1e-15)
    else:
        assert out.min() >= 1.0 - 1e-15
        assert out.max() <= np.sqrt(2) + 1e-15


@pytest.mark.parametrize('rise, fall', [(0.01, 10), (10, 0.01)])
@pytest.mark.parametrize('shape', ['linear', 'equal_power'])
@pytest.mark.parametrize('as_array', [False, True])
def test_unbalanced_outer_ramps_preserve_both_requested_fades(
        rise, fall, shape, as_array):
    session = music.StimulationSession(sample_rate=100, end_ramp=fall,
                                       ramp_shape=shape)
    session.add(np.ones(100) if as_array else constant,
                duration=1, ramp=rise)
    out = session.render()
    assert out.shape == (100,)
    assert out[0] == 0
    assert out[-1] < 0.02
    assert out.max() > 0.98


def test_a_one_sample_opening_phase_keeps_its_fade_with_a_long_crossfade():
    session = music.StimulationSession(sample_rate=100, ramp_shape='linear')
    session.add(constant, duration=0.01, ramp=0.01)
    session.add(constant, duration=1, ramp=10)
    out = session.render()
    assert out.shape == (101,)
    assert out[0] == 0
    np.testing.assert_allclose(out[1:], 1, atol=1e-15)


@pytest.mark.parametrize('array_first', [False, True])
def test_oversized_mixed_crossfade_uses_the_array_length(array_first):
    session = music.StimulationSession(sample_rate=100, ramp_shape='linear')
    stimuli = ([np.ones(10), constant] if array_first
               else [constant, np.ones(10)])
    for index, stimulus in enumerate(stimuli):
        session.add(stimulus, duration=0.1, ramp=0.25 if index else 0)
    # The ten-sample array lends five samples to the centred transition;
    # the callable still contributes its requested ten-sample duration.
    out = session.render()
    assert out.shape == (15,)
    assert session.duration == 0.15
    np.testing.assert_allclose(out, 1.0, atol=1e-15)


@pytest.mark.parametrize('shape', ['linear', 'equal_power'])
@pytest.mark.parametrize('length', [1, 2])
@pytest.mark.parametrize('as_array', [False, True])
def test_one_or_two_samples_with_outer_fades_are_silent(
        shape, length, as_array):
    session = music.StimulationSession(sample_rate=100, end_ramp=1,
                                       ramp_shape=shape)
    session.add(np.ones(length) if as_array else constant,
                duration=length / 100, ramp=1)
    np.testing.assert_array_equal(session.render(), np.zeros(length))


def test_a_phase_rounded_to_no_samples_does_not_call_a_generator():
    def must_not_render(**parameters):
        raise AssertionError('a zero-length phase called its generator')

    session = music.StimulationSession(sample_rate=100, ramp_shape='linear')
    session.add(constant, duration=0.1)
    session.add(must_not_render, duration=0.001, ramp=1)
    session.add(constant, duration=0.1, ramp=1)
    assert session.duration == 0.2
    np.testing.assert_array_equal(session.render(), np.ones(20))


@pytest.mark.parametrize('shape', ['linear', 'equal_power'])
def test_valid_odd_ramps_keep_their_original_sample_positions(shape):
    session = music.StimulationSession(sample_rate=100, ramp_shape=shape)
    session.add(constant, duration=0.1, level=1.0)
    session.add(constant, duration=0.2, ramp=0.03, level=3.0)
    session.add(constant, duration=0.1, ramp=0.05, level=0.5)
    expected = np.r_[np.ones(9), np.full(19, 3.0), np.full(12, 0.5)]
    for start, count, left, right in [(9, 3, 1.0, 3.0), (28, 5, 3.0, 0.5)]:
        progress = np.arange(count) / count
        if shape == 'linear':
            expected[start:start + count] = (
                left * (1 - progress) + right * progress)
        else:
            expected[start:start + count] = (
                left * np.cos(progress * np.pi / 2)
                + right * np.sin(progress * np.pi / 2))
    np.testing.assert_allclose(session.render(), expected, atol=1e-15)


def test_ramp_shape_endpoints_are_silence_and_full():
    rising = _ramp_shape(100, 'equal_power', True)
    falling = _ramp_shape(100, 'equal_power', False)
    assert rising[0] == pytest.approx(0.0)
    assert rising[-1] > 0.99
    assert falling[0] == pytest.approx(1.0)
    assert falling[-1] < 0.02
    assert len(_ramp_shape(0, 'equal_power', True)) == 0


# --------------------------------------------------------------------------
# channels -- a protocol mixes stimuli that do not agree on them
# --------------------------------------------------------------------------

def test_a_mono_session_stays_mono():
    session = music.StimulationSession()
    session.add(constant, duration=0.2)
    assert session.render().ndim == 1


def test_one_stereo_phase_makes_the_whole_session_stereo():
    """Flattening is not an option: it is what destroys a binaural beat."""
    session = music.StimulationSession()
    session.add(constant, duration=0.2)
    session.add(stereo_constant, duration=0.2, ramp=0.05)
    out = session.render()
    assert out.shape == (2, int(round(0.4 * SR)))
    assert np.array_equal(out[0, :100], out[1, :100])


def test_a_real_protocol_of_binaural_and_isochronic_phases():
    """The case the class exists for, end to end."""
    session = music.StimulationSession(end_ramp=0.05)
    session.add(music.binaural_beats, duration=0.3, label='settle',
                carrier_freq=200, beat_freq=10)
    session.add(music.isochronic_tones, duration=0.3, ramp=0.1,
                label='descend', carrier_freq=200, pulse_rate=6)
    out = session.render()
    assert out.shape == (2, int(round(0.6 * SR)))
    assert np.abs(out).max() <= 1.0


# --------------------------------------------------------------------------
# phases given as arrays
# --------------------------------------------------------------------------

def test_an_array_phase_brings_its_own_length():
    session = music.StimulationSession()
    session.add(np.ones(SR // 2))
    assert session.duration == pytest.approx(0.5)
    assert len(session.render()) == SR // 2


def test_an_array_phase_gives_up_its_share_of_the_ramps():
    """The array is the whole extent, ramps included.

    So its nominal span is what is left after the halves it lends to
    its neighbours, and the session still lasts the sum of the spans.
    """
    session = music.StimulationSession()
    session.add(constant, duration=0.5)
    session.add(np.ones(SR // 2), ramp=0.1)
    out = session.render()
    assert len(out) == int(round(0.5 * SR)) + SR // 2 - int(round(0.05 * SR))
    assert session.duration == pytest.approx(len(out) / SR)


def test_a_stereo_array_phase_is_measured_by_its_second_axis():
    session = music.StimulationSession()
    session.add(np.ones((2, SR // 4)))
    assert session.render().shape == (2, SR // 4)


# --------------------------------------------------------------------------
# gain, labels, and refusals
# --------------------------------------------------------------------------

def test_gain_scales_a_phase_before_it_is_mixed():
    session = music.StimulationSession()
    session.add(constant, duration=0.2, gain=0.25)
    assert session.render().max() == pytest.approx(0.25)


def test_parameters_reach_the_generator():
    session = music.StimulationSession()
    session.add(constant, duration=0.2, level=0.5)
    assert session.render().max() == pytest.approx(0.5)


def test_a_callable_without_a_duration_is_refused():
    """Rendering nothing would drop a phase out of a protocol silently."""
    session = music.StimulationSession()
    with pytest.raises(ValueError, match='needs a duration'):
        session.add(constant)


@pytest.mark.parametrize('kwargs, match', [
    ({'duration': -1}, 'duration'),
    ({'duration': 1, 'ramp': -1}, 'ramp'),
])
def test_negative_times_are_refused(kwargs, match):
    session = music.StimulationSession()
    with pytest.raises(ValueError, match=match):
        session.add(constant, **kwargs)


def test_a_stimulus_that_ignores_number_of_samples_is_caught():
    """It would shift every phase after it, so it is an error here.

    `number_of_samples` was declared and ignored by two routines in
    this package until 1.3.0, which is why this is checked rather than
    assumed.
    """
    def wrong_length(number_of_samples=0, sample_rate=SR):
        return np.ones(number_of_samples + 10)

    session = music.StimulationSession()
    session.add(wrong_length, duration=0.2)
    with pytest.raises(ValueError, match='samples where'):
        session.render()


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------

@pytest.mark.parametrize('stimulus, channels', [
    (constant, 1),
    (stereo_constant, 2),
])
def test_write_produces_a_readable_wav(tmp_path, stimulus, channels):
    session = music.StimulationSession()
    session.add(stimulus, duration=0.1)
    path = tmp_path / 'session.wav'
    session.write(str(path))
    assert path.exists()
    data = np.asarray(music.read_wav(str(path)))
    assert (2 if data.ndim == 2 else 1) == channels


def test_repr_names_the_phases_for_a_reader():
    session = music.StimulationSession()
    assert repr(session) == 'StimulationSession(empty)'
    session.add(music.binaural_beats, duration=0.1)
    session.add(np.ones(10), label='bed')
    text = repr(session)
    assert 'binaural_beats' in text
    assert 'bed' in text
    assert '2 phases' in text


def test_an_unlabelled_array_phase_is_named_for_what_it_is():
    session = music.StimulationSession()
    session.add(np.ones(10))
    assert 'array' in repr(session)
