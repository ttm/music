import warnings

import numpy as np
import pytest

from music.core.filters import (
    adsr,
    fade,
    cross_fade,
    reverb,
    loud,
    louds,
)
from music.core.filters.localization import localize
import music


def test_adsr_envelope_basic():
    env = adsr(
        envelope_duration=0.1,
        attack_duration=10,
        decay_duration=20,
        sustain_level=-6,
        release_duration=10,
        transition="exp",
        sample_rate=1000,
    )
    sustain_amp = 10 ** (-6 / 20)
    assert len(env) == 100
    assert env[0] < 1e-3
    assert np.isclose(env[9], 1.0, atol=1e-6)
    assert np.allclose(env[30:90], sustain_amp)
    assert env[-1] < 1e-4


def test_fade_and_cross_fade():
    fade_out = fade(number_of_samples=5, fade_out=True, method="linear")
    fade_in = fade(number_of_samples=5, fade_out=False, method="linear")
    assert np.allclose(fade_out, np.linspace(1, 0, 5))
    assert np.allclose(fade_in, np.linspace(0, 1, 5))

    s1 = np.ones(441)
    s2 = np.ones(441) * 2
    mixed = cross_fade(s1.copy(), s2.copy(), duration=5, sample_rate=44100)
    assert len(mixed) == 661
    assert mixed[0] == 1.0
    assert np.isclose(mixed[-1], 2.0, atol=1e-6)


def test_reverb_minimal_operation():
    np.random.seed(0)
    ir = reverb(
        duration=0.02,
        first_phase_duration=0.01,
        decay=-1,
        noise_type="white",
        sample_rate=100,
    )
    assert len(ir) == 2
    assert ir[0] == 1.0

    out = reverb(
        duration=0.02,
        first_phase_duration=0.01,
        decay=-1,
        noise_type="white",
        sonic_vector=np.ones(5),
        sample_rate=100,
    )
    assert out.shape == (6,)

def test_localize_basic():
    sv = np.ones(5)
    out = localize(sonic_vector=sv, x=0.1, y=0.1, sample_rate=10)
    assert out.shape[0] == 2
    assert out.shape[1] >= 5
    assert not np.allclose(out[0], out[1])


def test_loud_ramp_start_end():
    sr = 100
    # Exponential ramp up by 6 dB
    env = loud(duration=0.1, trans_dev=6, method="exp", sample_rate=sr)
    assert len(env) == int(0.1 * sr)
    assert np.isclose(env[0], 1.0, atol=1e-6)
    assert np.isclose(env[-1], 10 ** (6 / 20), atol=1e-6)

    # Linear ramp down to zero
    lin_env = loud(duration=0.05, trans_dev=0, method="linear", sample_rate=sr)
    assert lin_env[0] == 1.0
    assert np.isclose(lin_env[-1], 0.0, atol=1e-6)


def test_louds_concatenation_and_continuity():
    sr = 100
    durations = (0.1, 0.2)
    devs = (6, -6)
    env = louds(durations=durations, trans_devs=devs, alpha=(1, 1),
                method=("exp", "exp"), sample_rate=sr)
    expected_len = int(sum(durations) * sr)
    assert len(env) == expected_len

    n1 = int(durations[0] * sr)
    assert np.isclose(env[n1 - 1], env[n1], atol=1e-6)
    assert np.isclose(env[0], 1.0, atol=1e-6)
    mid_amp = 10 ** (devs[0] / 20)
    assert np.isclose(env[n1 - 1], mid_amp, atol=1e-6)
    assert np.isclose(env[-1], 1.0, atol=1e-6)



def test_reverb_refuses_a_first_phase_longer_than_the_whole_reverb():
    """Regression: the shapes disagreed and numpy reported a broadcast
    failure naming two sample counts, which says nothing about the two
    durations that caused it. `reverb(duration=0.1)` hit it on the
    default first_phase_duration of 0.15.
    """
    with pytest.raises(ValueError, match="cannot exceed duration"):
        reverb(duration=0.1)


@pytest.mark.parametrize("sample_rate", [8000, 22050, 44100])
def test_the_reverb_tail_spans_the_band_at_any_rate(sample_rate):
    """White noise up to the Nyquist frequency of the rate asked for.

    `reverb` asked `noise` for a band up to half its own rate without
    passing the rate, so the band was built at 44.1 kHz: at 8 kHz the
    tail held 95% of its energy below 680 Hz rather than near 4 kHz.
    """
    np.random.seed(1)
    response = reverb(duration=1.0, first_phase_duration=0.1, decay=-10,
                      noise_type="white", sample_rate=sample_rate)
    tail = response[int(0.2 * sample_rate):]
    power = np.abs(np.fft.rfft(tail)) ** 2
    freqs = np.fft.rfftfreq(len(tail), 1 / sample_rate)
    edge = freqs[np.searchsorted(np.cumsum(power) / power.sum(), .95)]
    assert edge > .9 * sample_rate / 2


# The envelope contracts the mutation audit found nothing depending on.
# Each test below kills mutants that survived the first run: the defaults
# in the signatures, and the settings these routines pass on to the fades
# and loudness transitions they are built from. See MUTATION_AUDIT.md.

def test_the_adsr_defaults_are_the_ones_its_docstring_promises():
    # Two seconds at 44.1 kHz; 20 ms attack, 20 ms decay, 50 ms release;
    # sustain 5 dB down. Nothing called `adsr()` bare and looked at the
    # result, so every one of those numbers could be changed freely.
    env = adsr()
    attack, decay, release = 882, 882, 2205
    assert len(env) == 2 * 44100
    assert env[attack - 1] == pytest.approx(1.0, abs=1e-6)
    assert np.allclose(env[attack + decay:-release], 10 ** (-5 / 20))
    # `to_zero` defaults to 1 ms, so the envelope departs from zero and
    # returns to it. It did neither until the audit: the milliseconds were
    # passed on as a ratio where `fade` reads a percentage, which rounded
    # to no samples, and the envelope began and ended at `db_dev` instead.
    assert env[0] == pytest.approx(0.0, abs=1e-12)
    assert env[-1] == pytest.approx(0.0, abs=1e-12)
    assert adsr(to_zero=0)[0] == pytest.approx(10 ** (-80 / 20), rel=1e-6)


@pytest.mark.parametrize("milliseconds", [1, 3, 5])
def test_the_adsr_departs_from_zero_for_as_long_as_it_is_told_to(milliseconds):
    # `to_zero` is a duration, so the straight part is that many
    # milliseconds of samples and not some other multiple of them: the
    # whole defect was a factor of a hundred in exactly this number.
    # Every value of `to_zero` gave the same envelope before the audit.
    env = adsr(to_zero=milliseconds)
    expected = int(milliseconds * 44100 / 1000)
    assert env[0] == pytest.approx(0.0, abs=1e-12)
    # Straight for `expected` samples, and curved immediately after.
    assert np.allclose(np.diff(env[:expected], 2), 0.0, atol=1e-12)
    assert not np.allclose(np.diff(env[:expected + 2], 2), 0.0, atol=1e-12)
    # The release reaches zero over the same span, at the other end.
    assert env[-1] == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(np.diff(env[-expected:], 2), 0.0, atol=1e-14)
    assert not np.allclose(np.diff(env[-(expected + 2):], 2), 0.0,
                           atol=1e-14)
    head = expected + 40
    assert not np.allclose(env[:head], adsr(to_zero=0)[:head])


@pytest.mark.parametrize("transition", ["exp", "linear"])
def test_adsr_hands_each_stage_the_shape_it_was_given(transition):
    # `adsr` is three calls to `fade` and `loud` with a constant between
    # them. What is under test is that `transition`, `alpha` and `db_dev`
    # reach all three: each could be dropped from any of the calls and
    # every test still passed, because none used a non-default value.
    env = adsr(envelope_duration=0.2, attack_duration=30, decay_duration=40,
               sustain_level=-6, release_duration=50, transition=transition,
               alpha=2, db_dev=-40, sample_rate=1000)
    sustain = 10 ** (-6 / 20)
    assert len(env) == 200
    assert np.allclose(env[:30], fade(
        fade_out=0, method=transition, alpha=2, db=-40, perc=100 / 30,
        number_of_samples=30))
    assert np.allclose(env[30:70], loud(
        trans_dev=-6, method=transition, alpha=2, number_of_samples=40))
    assert np.allclose(env[70:150], sustain)
    assert np.allclose(env[-50:], fade(
        method=transition, alpha=2, db=-40, perc=100 / 50,
        number_of_samples=50) * sustain)


@pytest.mark.parametrize("transition", ["exp", "linear"])
def test_adsr_stereo_gives_both_channels_every_setting(transition):
    # The stereo shorthand forwards eight arguments to `adsr` twice over,
    # and any of them could be dropped from either call unnoticed.
    settings = dict(attack_duration=30, decay_duration=40, sustain_level=-6,
                    release_duration=50, transition=transition, alpha=2,
                    db_dev=-40, to_zero=3, number_of_samples=200,
                    sample_rate=1000)
    mono = adsr(envelope_duration=5, **settings)
    stereo = music.adsr_stereo(duration=5, **settings)
    assert stereo.shape == (2, 200)
    assert np.allclose(stereo[0], mono)
    assert np.allclose(stereo[1], mono)


def test_the_adsr_stereo_defaults_are_the_mono_ones():
    # Its signature repeats every default in `adsr`'s, so the two can
    # drift apart. Called bare, the two channels are that mono envelope.
    assert np.allclose(music.adsr_stereo(), np.vstack((adsr(), adsr())))


def test_adsr_vibrato_shapes_the_note_it_renders():
    # Nothing in the suite depended on what this returns: its whole body
    # could be replaced by `adsr(**adsr_dict)` -- dropping the note it
    # exists to render -- and no test went red.
    note = dict(freq=220, duration=0.5)
    envelope = dict(sustain_level=-10)
    got = music.adsr_vibrato(note_dict=note, adsr_dict=envelope)
    assert len(got) == len(music.note_with_vibrato(**note))
    assert np.allclose(got, adsr(sonic_vector=music.note_with_vibrato(**note),
                                 **envelope))
    # ...and it is the note that is shaped, not a bare envelope.
    assert not np.allclose(got[:100], adsr(**envelope)[:100])


def test_the_fade_defaults_are_the_ones_its_docstring_promises():
    envelope = fade()
    assert len(envelope) == 2 * 44100
    assert envelope[0] == pytest.approx(1.0, abs=1e-9)
    assert envelope[-1] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("fade_out", [True, False])
def test_a_fade_hands_its_curvature_to_the_transition(fade_out):
    # `alpha` bends the exponential fade. It was passed to `loud` in both
    # directions and dropping it from either was invisible.
    straight = fade(number_of_samples=200, alpha=1, fade_out=fade_out)
    curved = fade(number_of_samples=200, alpha=3, fade_out=fade_out)
    assert not np.allclose(straight, curved)
    # The curve changes the path, not where it starts and ends.
    assert curved[0] == pytest.approx(straight[0], abs=1e-9)
    assert curved[-1] == pytest.approx(straight[-1], abs=1e-9)


@pytest.mark.parametrize("fade_out", [True, False])
def test_an_exponential_fade_joins_its_linear_tail_without_a_step(fade_out):
    # An exponential fade is an exponential part and a short linear one,
    # and the linear part is scaled to start where the other ends. Scale
    # it the other way and the envelope leaps to ten thousand at the join
    # -- which nothing noticed, because nothing asserted that a fade stays
    # inside [0, 1] or moves in one direction.
    envelope = fade(duration=0.5, fade_out=fade_out, perc=10)
    assert envelope.min() >= 0.0
    assert envelope.max() <= 1.0 + 1e-12
    steps = np.diff(envelope)
    assert np.all(steps <= 1e-12) if fade_out else np.all(steps >= -1e-12)


def test_the_cross_fade_defaults_are_the_ones_its_docstring_promises():
    first, second = np.ones(44100), np.ones(44100) * 2
    joined = cross_fade(first, second)
    assert len(joined) == 44100 + 44100 - 22050      # 500 ms at 44.1 kHz


def test_a_cross_fade_overlaps_by_the_time_it_was_given():
    first, second = np.ones(44100), np.ones(44100) * 2
    joined = cross_fade(first, second, duration=200)
    assert len(joined) == 44100 + 44100 - 8820


@pytest.mark.parametrize("sample_rate", [8000, 44100, 48000])
def test_a_cross_fade_blends_rather_than_summing(sample_rate):
    # Between a steady 3 and a steady 5 there is no moment quieter than 3
    # and none louder than 5: that is what distinguishes a crossfade from
    # a sum. Replacing either sound with its envelope, or fading the
    # second one out instead of in, breaks it -- and the level-1 sounds
    # the previous test used could not tell any of those apart.
    first, second = np.ones(441) * 3, np.ones(441) * 5
    joined = cross_fade(first, second, duration=5, sample_rate=sample_rate)
    assert joined.min() >= 3.0 - 1e-9
    assert joined.max() <= 5.0 + 0.05
    assert joined[0] == pytest.approx(3.0)
    assert joined[-1] == pytest.approx(5.0)


def test_a_cross_fade_measures_the_overlap_at_the_rate_it_was_given():
    # The fades were cut at `sample_rate` and the overlap was placed at
    # 44.1 kHz, so at any other rate the two sounds met at full level.
    first, second = np.ones(441) * 3, np.ones(441) * 5
    joined = cross_fade(first, second, duration=5, sample_rate=8000)
    assert len(joined) == 441 + 441 - 40
    assert joined.max() <= 5.0 + 1e-9


def test_a_cross_fade_uses_the_curve_it_was_given():
    first, second = np.ones(441) * 3, np.ones(441) * 5
    linear = cross_fade(first.copy(), second.copy(), duration=5, method='lin')
    exponential = cross_fade(first, second, duration=5, method='exp')
    assert not np.allclose(linear, exponential)


def test_a_stereo_cross_fade_keeps_each_channel_to_itself():
    # Distinct levels in all four places, so crossing the channels over
    # shows up as a value no channel should be able to reach.
    first = np.vstack((np.ones(441) * 3, np.ones(441) * 7))
    second = np.vstack((np.ones(441) * 5, np.ones(441) * 9))
    # A rate the recursion has to carry down with it, rather than the one
    # each channel would have fallen back to on its own.
    joined = cross_fade(first, second, duration=5, sample_rate=8000)
    assert joined.shape[1] == 441 + 441 - 40
    assert joined.shape[0] == 2
    assert joined[0].min() >= 3.0 - 1e-9 and joined[0].max() <= 5.0 + 0.05
    assert joined[1].min() >= 7.0 - 1e-9 and joined[1].max() <= 9.0 + 0.05


@pytest.mark.parametrize("duration, sample_rate", [
    (0, 44100),        # `fade` reads 0 samples as "unset" and gives 2 s
    (-5, 44100),
    (500, 44100),      # the default overlap, longer than these sounds
    (5, 96000),        # 480 samples of overlap in a 441-sample sound
])
def test_a_cross_fade_refuses_an_overlap_it_cannot_make(duration, sample_rate):
    first, second = np.ones(441), np.ones(441) * 2
    with pytest.raises(ValueError, match="duration"):
        cross_fade(first, second, duration=duration, sample_rate=sample_rate)


@pytest.mark.parametrize("samples", [1, 441])
def test_a_cross_fade_allows_the_overlaps_at_the_edge_of_what_fits(samples):
    # One sample of overlap, and an overlap as long as the whole sound,
    # are both things it can make; the refusal above them must not creep
    # inward onto either.
    first, second = np.ones(441) * 3, np.ones(441) * 5
    joined = cross_fade(first, second, duration=samples * 1000 / 44100)
    assert joined.min() >= 3.0 - 1e-9
    assert joined.max() <= 5.0 + 1e-9


def test_a_reverb_of_only_its_first_period_has_no_tail():
    """`noise` reads zero samples as "not given" and returned two
    seconds of tail, which failed to broadcast against no decay."""
    np.random.seed(2)
    response = reverb(duration=.01, first_phase_duration=.01,
                      sample_rate=1000)
    assert len(response) == 10 and response[0] == 1


def test_a_one_sample_reverb_is_the_direct_sound():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        np.testing.assert_array_equal(
            reverb(duration=1 / 1000, first_phase_duration=0,
                   sample_rate=1000), [1.])


def test_a_reverb_shorter_than_a_sample_is_refused():
    """With no first period to conflict with, a zero duration failed as a
    broadcast error rather than saying what was wrong."""
    with pytest.raises(ValueError, match="at least one sample"):
        reverb(duration=0, first_phase_duration=0)


def test_louds_by_count_is_louds_by_duration():
    """The counts branch passed `trans_devs[i], alpha[i]` by position to
    `loud(duration, trans_dev, alpha, ...)`, so every alpha became a
    deviation: a 6 dB rise came out as a 1 dB one."""
    settings = dict(trans_devs=(6, -12), alpha=(1, 2),
                    method=("exp", "exp"))
    by_count = louds(number_of_samples=(100, 50), **settings)
    by_time = louds(durations=(.1, .05), sample_rate=1000, **settings)
    np.testing.assert_array_equal(by_count, by_time)
    assert 20 * np.log10(by_count[99]) == pytest.approx(6)
    assert 20 * np.log10(by_count[-1]) == pytest.approx(6 - 12)


@pytest.mark.parametrize("given", [dict(number_of_samples=(100, 0)),
                                   dict(durations=(.1, 0))])
def test_louds_refuses_a_transition_with_no_samples(given):
    """It failed on the empty transition's missing last value."""
    with pytest.raises(ValueError, match="at least one sample"):
        louds(trans_devs=(6, -6), alpha=(1, 1), method=("exp", "exp"),
              sample_rate=1000, **given)
