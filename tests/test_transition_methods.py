"""Every linear-or-exponential transition takes the same four names.

They used to read them three ways: `loud` by exact name, `fade` by
substring, so ``"explicit"`` faded exponentially, and the glissandi by
comparison with ``"exp"`` alone, so ``"exponential"`` swept linearly and
an unknown name was linear without a word. `adsr` passes one name to both
`fade` and `loud`, so the same name could be accepted by one stage and
refused by the next.
"""

import numpy as np
import pytest

import music

_REFUSAL = "must be 'lin'/'linear' or 'exp'/'exponential'; got "
_SOUND = np.linspace(-1.0, 1.0, 400)


def _loud(method):
    return music.loud(number_of_samples=64, trans_dev=-12, method=method)


def _louds(method):
    return music.louds(number_of_samples=(32, 32), trans_devs=(6, -12),
                       alpha=(1, 2), method=(method, method))


def _fade(method):
    return music.fade(number_of_samples=64, method=method, perc=10)


def _fade_in(method):
    return music.fade(number_of_samples=64, fade_out=False, method=method,
                      perc=10)


def _adsr(method):
    return music.adsr(envelope_duration=0.2, attack_duration=20,
                      decay_duration=20, release_duration=50,
                      transition=method, sample_rate=1000)


def _cross_fade(method):
    return music.cross_fade(_SOUND.copy(), _SOUND[::-1].copy(),
                            duration=100, method=method, sample_rate=1000)


def _glissando(method):
    return music.note_with_glissando(start_freq=50, end_freq=200,
                                     duration=0.5, method=method,
                                     sample_rate=1000)


#: A path on one side of the listener, which an exponential move needs.
_PATH = dict(x=(1, 2, 4, 3), y=(1, 2, .5, .1))


def _localized(method):
    return music.note_with_vibrato_seq_localization(
        method=(method, method, method), sample_rate=200, **_PATH)


def _localized_mono(method):
    return music.note_with_vibrato_seq_localization(
        method=(method, method, method), stereo=False, sample_rate=200,
        **_PATH)


ROUTINES = [_loud, _louds, _fade, _fade_in, _adsr, _cross_fade, _glissando,
            _localized, _localized_mono]


@pytest.mark.parametrize("routine", ROUTINES)
@pytest.mark.parametrize("alias, name", [("linear", "lin"),
                                         ("exponential", "exp")])
def test_each_long_name_is_its_short_one(routine, alias, name):
    np.testing.assert_array_equal(routine(alias), routine(name))


@pytest.mark.parametrize("routine", ROUTINES)
def test_linear_and_exponential_are_different_transitions(routine):
    assert not np.array_equal(routine("lin"), routine("exp"))


@pytest.mark.parametrize("routine", ROUTINES)
@pytest.mark.parametrize("method", [
    "explicit", "linen", "linexp", "Exp", "LINEAR", " lin", "", "bogus",
])
def test_any_other_name_is_refused_the_same_way(routine, method):
    """``"explicit"`` and ``"linen"`` were a fade's exponential and linear
    shapes, ``"linexp"`` both at once, and every one of these was a linear
    glissando."""
    name = "transition" if routine is _adsr else "method"
    with pytest.raises(ValueError, match=f"^{name} {_REFUSAL}"):
        routine(method)


def test_an_exponential_glissando_by_its_long_name_rises_by_a_ratio():
    """"exponential" swept linearly: the pitch at the midpoint of 50 to
    200 Hz was their mean rather than their geometric mean."""
    rate, count = 1000, 500
    produced = music.note_with_glissando(
        start_freq=50, end_freq=200, number_of_samples=count,
        method="exponential", waveform_table=np.arange(1000) / 1000,
        sample_rate=rate)
    geometric = music.note_with_glissando(
        start_freq=50, end_freq=200, number_of_samples=count, method="exp",
        waveform_table=np.arange(1000) / 1000, sample_rate=rate)
    arithmetic = music.note_with_glissando(
        start_freq=50, end_freq=200, number_of_samples=count, method="lin",
        waveform_table=np.arange(1000) / 1000, sample_rate=rate)
    np.testing.assert_array_equal(produced, geometric)
    assert not np.array_equal(produced, arithmetic)


def test_an_envelope_of_no_stages_still_checks_its_transition():
    """With every stage empty nothing reached `fade` or `loud`, so any
    name was accepted."""
    with pytest.raises(ValueError, match=f"^transition {_REFUSAL}'bogus'"):
        music.adsr(envelope_duration=0.1, attack_duration=0,
                   decay_duration=0, release_duration=0,
                   transition="bogus", sample_rate=1000)
    np.testing.assert_array_equal(
        music.adsr(envelope_duration=0.1, attack_duration=0,
                   decay_duration=0, release_duration=0, sustain_level=0,
                   transition="linear", sample_rate=1000),
        np.ones(100))


def test_a_stereo_envelope_checks_its_transition_too():
    with pytest.raises(ValueError, match=f"^transition {_REFUSAL}"):
        music.adsr_stereo(duration=0.1, transition="explicit",
                          sample_rate=1000)


def test_a_sequence_checks_every_transition_before_rendering():
    with pytest.raises(ValueError, match=f"^method {_REFUSAL}'sideways'"):
        music.note_with_vibrato_seq_localization(
            method=("lin", "exp", "sideways"), sample_rate=200)


def test_a_stereo_fade_checks_its_method_on_each_channel():
    stereo = np.vstack((_SOUND, _SOUND[::-1]))
    np.testing.assert_array_equal(
        music.fade(sonic_vector=stereo, method="exponential"),
        music.fade(sonic_vector=stereo, method="exp"))
    with pytest.raises(ValueError, match=f"^method {_REFUSAL}'explicit'"):
        music.fade(sonic_vector=stereo, method="explicit")
