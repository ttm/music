"""The AM and tremolo envelopes, and the settings they pass on.

The first two tests here checked that each envelope stays inside its
documented depth. A bound is a weak thing to assert on an oscillator: it
says nothing about how fast the envelope oscillates, how long it is, or
whether it oscillates at all. The mutation audit showed it -- every
default in both signatures could be changed without a test noticing, and
`tremolos` could drop any of the five settings it forwards to `tremolo`
and still pass. The tests below pin the period and the length as well as
the depth, and compare `tremolos` against `tremolo` called directly.

See MUTATION_AUDIT.md for the run these come from.
"""

import warnings

import numpy as np
import pytest

import music
from music.utils import WAVEFORM_SINE, WAVEFORM_TRIANGULAR


# Test amplitude modulation envelope

def test_am_envelope_range_and_application():
    ns = 1000
    env = music.am(number_of_samples=ns, fm=50, max_amplitude=0.3,
                   sonic_vector=None)
    assert env.min() >= 1 - 0.3 - 1e-6
    assert env.max() <= 1 + 0.3 + 1e-6

    wave = np.ones_like(env)
    modulated = music.am(number_of_samples=ns, fm=50, max_amplitude=0.3,
                         sonic_vector=wave)
    assert np.max(modulated) > 1
    assert np.min(modulated) < 1


# Test tremolo envelope

def test_tremolo_envelope_range_and_application():
    ns = 1000
    db_dev = 6
    env = music.tremolo(number_of_samples=ns, tremolo_freq=100,
                        max_db_dev=db_dev, sonic_vector=None)
    min_val = 10 ** (-db_dev / 20)
    max_val = 10 ** (db_dev / 20)
    assert env.min() >= min_val - 1e-6
    assert env.max() <= max_val + 1e-6

    wave = np.ones_like(env)
    modulated = music.tremolo(number_of_samples=ns, tremolo_freq=100,
                              max_db_dev=db_dev, sonic_vector=wave)
    assert np.max(modulated) > 1
    assert np.min(modulated) < 1


# Test that the documented defaults are the ones in the signature

def test_the_am_defaults_are_the_ones_its_docstring_promises():
    # Two seconds at 44.1 kHz, oscillating fifty times a second, reaching
    # 0.4 either side of unity. Nothing else in the suite calls `am` with
    # no arguments and looks at the result, so until this existed each of
    # those four numbers could be changed without a test going red.
    env = music.am()
    assert len(env) == 2 * 44100
    period = 44100 // 50
    assert np.allclose(env[:period], env[period:2 * period])
    # ...and that it is that period, rather than some fraction of it.
    half = period // 2
    assert not np.allclose(env[:period], env[half:half + period])
    assert env.max() == pytest.approx(1.4, abs=1e-3)
    assert env.min() == pytest.approx(0.6, abs=1e-3)


def test_the_tremolo_defaults_are_the_ones_its_docstring_promises():
    # Two seconds at 44.1 kHz, twice a second, 10 dB either way. The depth
    # is in decibels, so the two bounds are not equidistant from unity --
    # which is the reason the article gives for measuring it that way.
    env = music.tremolo()
    assert len(env) == 2 * 44100
    period = 44100 // 2
    assert np.allclose(env[:period], env[period:2 * period])
    half = period // 2
    assert not np.allclose(env[:period], env[half:half + period])
    assert env.max() == pytest.approx(10 ** (10 / 20), rel=1e-3)
    assert env.min() == pytest.approx(10 ** (-10 / 20), rel=1e-3)


# Test that tremolos gives each tremolo the settings it was given for it

def test_tremolos_hands_each_tremolo_the_settings_it_was_given():
    # Every argument differs from `tremolo`'s own default, so dropping any
    # one of them on the way through is visible here. The comparison is
    # against `tremolo` itself rather than against recomputed samples:
    # what is under test is the forwarding, not the arithmetic.
    expected = music.tremolo(tremolo_freq=5, max_db_dev=6,
                             waveform_table=WAVEFORM_TRIANGULAR,
                             number_of_samples=500, sample_rate=8000)
    got = music.tremolos(tremolo_freqs=((5,),), max_db_devs=((6,),),
                         alpha=((1,),),
                         waveform_tables=((WAVEFORM_TRIANGULAR,),),
                         number_of_samples=((500,),), sample_rate=8000)
    assert np.allclose(got, expected)


def test_tremolos_hands_on_its_settings_when_given_durations_instead():
    # The same forwarding, down the other branch: `number_of_samples`
    # unset, so the durations decide the lengths.
    expected = music.tremolo(duration=0.25, tremolo_freq=5, max_db_dev=6,
                             waveform_table=WAVEFORM_TRIANGULAR,
                             sample_rate=8000)
    got = music.tremolos(durations=((0.25,),), tremolo_freqs=((5,),),
                         max_db_devs=((6,),), alpha=((1,),),
                         waveform_tables=((WAVEFORM_TRIANGULAR,),),
                         sample_rate=8000)
    assert np.allclose(got, expected)


def test_tremolos_holds_a_short_row_at_its_last_value():
    # Rows of different total length are squared off before they are
    # multiplied together, and the documented way is to hold the last
    # value rather than to drop to zero or to unity.
    short = music.tremolo(duration=0.01, tremolo_freq=5, max_db_dev=6,
                          waveform_table=WAVEFORM_TRIANGULAR,
                          sample_rate=8000)
    got = music.tremolos(durations=((0.01,), (0.02,)),
                         tremolo_freqs=((5,), (5,)),
                         max_db_devs=((6,), (6,)), alpha=((1,), (1,)),
                         waveform_tables=((WAVEFORM_TRIANGULAR,),
                                          (WAVEFORM_TRIANGULAR,)),
                         sample_rate=8000)
    assert len(got) == 160
    long_row = music.tremolo(duration=0.02, tremolo_freq=5, max_db_dev=6,
                             waveform_table=WAVEFORM_TRIANGULAR,
                             sample_rate=8000)
    held = np.hstack((short, np.ones(160 - len(short)) * short[-1]))
    assert np.allclose(got, held * long_row)


# Test the distortion index, which the article gives no tremolo

@pytest.mark.parametrize("alpha", [0.5, 1.5, 2, 3, 9])
def test_a_distorted_tremolo_is_finite_and_still_cuts(alpha):
    # Raising the signed oscillation straight to `alpha`, as the MASS
    # reference does, is NaN wherever the waveform is negative -- half of
    # every cycle -- and rectifies the pattern when `alpha` is even, so a
    # tremolo that could only ever boost. See DISCREPANCIES.md.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        env = music.tremolo(duration=1, tremolo_freq=1, max_db_dev=10,
                            alpha=alpha, sample_rate=1000)
    assert np.all(np.isfinite(env))
    assert env.min() < 1.0
    assert env.max() > 1.0


@pytest.mark.parametrize("alpha", [0.5, 1.5, 2, 3, 9])
def test_a_distorted_tremolo_stays_symmetric_in_decibels(alpha):
    # Keeping the sign is what makes the distortion bend the oscillation
    # rather than fold it: the depth above and below unity stays equal in
    # decibels, which is the article's reason for measuring it that way.
    env = music.tremolo(duration=1, tremolo_freq=1, max_db_dev=10,
                        alpha=alpha, sample_rate=1000)
    above = 20 * np.log10(env.max())
    below = 20 * np.log10(env.min())
    assert above == pytest.approx(-below, rel=1e-6)


def test_an_odd_distortion_index_is_the_plain_power_it_always_was():
    # sign(x)|x|**3 is x**3, so every whole odd index is bit-identical to
    # what the reference computes, and the `T` and `T_` rows of
    # RECONCILIATION.md stay sample-exact.
    table = music.utils.WAVEFORM_SINE
    length = len(table)
    indices = (np.arange(1000) * length / 1000).astype(np.int64) % length
    scaled = table[indices] * 10 / 20
    for alpha in (3, 9):
        env = music.tremolo(duration=1, tremolo_freq=1, max_db_dev=10,
                            alpha=alpha, sample_rate=1000)
        assert np.array_equal(env, 10. ** (scaled ** alpha))


def test_tremolos_hands_on_a_distortion_index_too():
    # `alpha` is the one setting `tremolos` forwards that its own default
    # matches, so dropping it went unnoticed down both branches.
    expected = music.tremolo(duration=0.25, tremolo_freq=5, max_db_dev=6,
                             alpha=2, waveform_table=WAVEFORM_TRIANGULAR,
                             sample_rate=8000)
    got = music.tremolos(durations=((0.25,),), tremolo_freqs=((5,),),
                         max_db_devs=((6,),), alpha=((2,),),
                         waveform_tables=((WAVEFORM_TRIANGULAR,),),
                         sample_rate=8000)
    assert np.allclose(got, expected)
    assert not np.allclose(got, music.tremolos(
        durations=((0.25,),), tremolo_freqs=((5,),), max_db_devs=((6,),),
        alpha=((1,),), waveform_tables=((WAVEFORM_TRIANGULAR,),),
        sample_rate=8000))


def test_tremolos_hands_on_a_distortion_index_by_sample_count_too():
    expected = music.tremolo(tremolo_freq=5, max_db_dev=6, alpha=2,
                             waveform_table=WAVEFORM_TRIANGULAR,
                             number_of_samples=500, sample_rate=8000)
    got = music.tremolos(tremolo_freqs=((5,),), max_db_devs=((6,),),
                         alpha=((2,),),
                         waveform_tables=((WAVEFORM_TRIANGULAR,),),
                         number_of_samples=((500,),), sample_rate=8000)
    assert np.allclose(got, expected)


def test_the_tremolos_defaults_are_the_ones_its_signature_promises():
    # Two rows of three and four tremolos, the longest running 16 seconds
    # at 44.1 kHz, with a distortion index of 9 on the last of them.
    envelope = music.tremolos()
    assert len(envelope) == 16 * 44100
    assert np.all(np.isfinite(envelope))
    rows = (music.tremolos(durations=((3, 4, 5),),
                           tremolo_freqs=((2, 6, 20),),
                           max_db_devs=((10, 20, 1),), alpha=((1, 1, 1),),
                           waveform_tables=((WAVEFORM_SINE,) * 3,)),
            music.tremolos(durations=((2, 3, 7, 4),),
                           tremolo_freqs=((5, 6.2, 21, 5),),
                           max_db_devs=((5, 7, 9, 2),), alpha=((1, 1, 1, 9),),
                           waveform_tables=((WAVEFORM_TRIANGULAR,) * 3
                                            + (WAVEFORM_SINE,),)))
    held = np.hstack((rows[0], np.ones(len(envelope) - len(rows[0]))
                      * rows[0][-1]))
    assert np.allclose(envelope, held * rows[1])
