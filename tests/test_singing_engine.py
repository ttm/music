"""The eCantorix engine itself, when it is installed.

These sing, and measure what was sung. They skip unless the engine is set
up with everything it needs, as the KEMAR tests skip without the
measurements; the rest of the singing tests check what is handed to the
engine without running it.
"""

import numpy as np
import pytest

from music.singing import paths
import music.singing.perform as perform


def _ready():
    try:
        return (paths.is_engine(paths.engine_dir())
                and not paths.missing_requirements()
                and not paths.missing_perl_modules())
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _ready(), reason="the eCantorix engine is not installed with all "
    "it needs; run music.singing.setup_engine()")


def _fundamental(segment, rate=44100, low=40, high=1500):
    """The autocorrelation peak between `low` and `high` Hz."""
    segment = segment - segment.mean()
    correlation = np.correlate(segment, segment, "full")[len(segment) - 1:]
    lags = np.arange(len(correlation))
    usable = (lags >= rate / high) & (lags <= rate / low)
    lag = lags[usable][np.argmax(correlation[usable])]
    left, peak, right = correlation[lag - 1:lag + 2]
    return rate / (lag + 0.5 * (left - right) / (left - 2 * peak + right))


def _pitches(sound, notes, windows=9):
    """The pitch of each of `notes` equal-length notes.

    The median over windows spread across the note, of those at least
    half as loud as its loudest: a sung syllable can fall nearly silent
    for a moment, and a window there measures nothing.
    """
    length = len(sound) // notes
    pitches = []
    for i in range(notes):
        centres = i * length + np.linspace(0.15, 0.85, windows) * length
        segments = [sound[int(c) - 4096:int(c) + 4096] for c in centres]
        loudest = max(np.abs(segment).max() for segment in segments)
        pitches.append(np.median([
            _fundamental(segment) for segment in segments
            if np.abs(segment).max() >= loudest / 2]))
    return pitches


def _midi(hertz):
    return 69 + 12 * np.log2(hertz / 440)


@pytest.fixture(scope="module")
def octave_at_middle_c():
    return perform.sing(text="laa laa", notes=(0, 12), durs=(8, 8),
                        transpose=0)


def test_the_engine_sings_reference_plus_note_plus_transpose(
        octave_at_middle_c):
    """Middle C and the C above, at transpose=0. The engine never read
    the transposition, and the score was an octave high, so these came out
    at 130 and 261 Hz whatever was asked."""
    low, high = _pitches(octave_at_middle_c, 2)
    assert _midi(low) == pytest.approx(60, abs=0.35)
    assert _midi(high) == pytest.approx(72, abs=0.35)


def test_the_default_sings_an_octave_below_the_score():
    """Where every note used to be sung: reference + note - 12."""
    sound = perform.sing(text="laa", notes=(0,), durs=(8,))
    assert _midi(_pitches(sound, 1)[0]) == pytest.approx(48, abs=0.35)


def test_the_language_reaches_the_voice(octave_at_middle_c):
    other = perform.sing(text="laa laa", notes=(0, 12), durs=(8, 8),
                         transpose=0, lang="de")
    assert other.shape != octave_at_middle_c.shape or not np.allclose(
        other, octave_at_middle_c)


@pytest.mark.parametrize("effect, shape", [
    ("flite", 1), ("tremolo", 2), ("melt", 2)])
def test_every_effect_renders(effect, shape):
    """Their files are in the engine's examples, which were not where the
    configuration loaded them from; melt also needs a copy of espeak's
    data, which the engine's Makefile makes from a Linux path."""
    sound = perform.sing(text="laa", notes=(0,), durs=(4,), effect=effect)
    assert sound.ndim == shape
    if shape == 2:
        assert sound.shape[0] == 2
    assert np.abs(sound).max() == pytest.approx(1)
