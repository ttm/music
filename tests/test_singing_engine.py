"""The singing backends themselves, where they are installed.

These sing, and measure what was sung, with the measurement
``tools/compare_singing.py`` reports. Each backend's tests skip unless it
is set up with everything it needs, as the KEMAR tests skip without the
measurements; the rest of the singing tests check what is handed to the
engines without running them.
"""

import numpy as np
import pytest

from music.singing import paths, psola
import music.singing.perform as perform
from tools.compare_singing import (SCORES, cents, expected_pitches,
                                   expected_seconds, note_pitches)


def _ecantorix_ready():
    try:
        return (paths.is_engine(paths.engine_dir())
                and not paths.missing_requirements()
                and not paths.missing_perl_modules())
    except OSError:
        return False


ECANTORIX = pytest.param("ecantorix", marks=pytest.mark.skipif(
    not _ecantorix_ready(), reason="the eCantorix engine is not installed "
    "with all it needs; run music.singing.setup_engine()"))
PSOLA = pytest.param("psola", marks=pytest.mark.skipif(
    bool(psola.missing_requirements()), reason="the psola backend needs "
    "espeak-ng and pip install 'music[singing]'"))
BACKENDS = [ECANTORIX, PSOLA]


def _midi(hertz):
    return 69 + 12 * np.log2(hertz / 440)


@pytest.fixture(scope="module")
def sung():
    """Each score each backend has sung, rendered once."""
    cache = {}

    def render(backend, name):
        if (backend, name) not in cache:
            cache[backend, name] = perform.sing(backend=backend,
                                                **SCORES[name])
        return cache[backend, name]
    return render


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_engine_sings_reference_plus_note_plus_transpose(backend, sung):
    """Middle C up an octave at transpose=0. eCantorix never read the
    transposition, and the score was an octave high, so these came out at
    130 and 261 Hz whatever was asked."""
    score = SCORES["octave"]
    pitches = note_pitches(sung(backend, "octave"), expected_seconds(score))
    assert [round(float(_midi(p))) for p in pitches] == [60, 64, 67, 72, 60]
    assert all(abs(cents(p, e)) < 35 for p, e
               in zip(pitches, expected_pitches(score)))


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_default_sings_an_octave_below_the_score(backend):
    """Where every note used to be sung: reference + note - 12."""
    sound = perform.sing(text="laa", notes=(0,), durs=(8,), backend=backend)
    pitch, = note_pitches(sound, [4.0])
    assert _midi(pitch) == pytest.approx(48, abs=0.35)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", ["mary", "quick", "test song"])
def test_every_note_is_in_tune_and_the_line_as_long_as_its_score(
        backend, name, sung):
    score = SCORES[name]
    sound = sung(backend, name)
    seconds = expected_seconds(score)
    assert abs(len(sound) / 44100 - sum(seconds)) < 0.01
    pitches = note_pitches(sound, seconds)
    measured = [(p, e) for p, e in zip(pitches, expected_pitches(score))
                if p is not None]
    assert len(measured) >= len(pitches) - 1
    assert all(abs(cents(p, e)) < 35 for p, e in measured)


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_long_note_is_sung_to_its_end(backend):
    """Praat's overlap-add stops at three times the length of the sound it
    is given, so the psola backend sang a four-second "laa" for one
    second and left the rest silent."""
    sound = perform.sing(text="laa", notes=(9,), durs=(8,), backend=backend)

    def level(start, end):
        part = sound[int(start * 44100):int(end * 44100)]
        return float(np.sqrt(np.mean(part ** 2)))

    assert level(3.0, 3.5) > 0.1 * level(0.25, 0.75)


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_syllable_said_as_a_consonant_alone_is_sung(backend):
    """espeak says French "ques" as a bare /k/, 41 ms long: the psola
    backend stopped there, too short for Praat to find a pitch in, and
    neither backend had a vowel to sing the note on. A singer sings the
    silent e, and so do both."""
    sound = perform.sing(text="Jac-ques", notes=(4, 0), durs=(1, 1),
                         lang="fr", backend=backend)
    assert abs(len(sound) / 44100 - 1) < 0.01
    jac, ques = note_pitches(sound, [0.5, 0.5])
    assert ques is not None and abs(cents(ques, 130.81)) < 35


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("key", ["F", "Bb"])
def test_the_key_changes_no_note(backend, key):
    """E and B were written without their natural sign, so eCantorix sang
    them flat in a key with flats: B flat for B in F, and E flat too in
    B flat."""
    sound = perform.sing(text="la la la", notes=(4, 11, 5), durs=(2, 2, 2),
                         K=key, backend=backend)
    pitches = note_pitches(sound, [1.0, 1.0, 1.0])
    expected = [164.81, 246.94, 174.61]
    assert all(abs(cents(p, e)) < 35 for p, e in zip(pitches, expected))


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_language_reaches_the_voice(backend, sung):
    english = perform.sing(backend=backend, **dict(SCORES["german"],
                                                   lang="en"))
    german = sung(backend, "german")
    assert english.shape != german.shape or not np.allclose(english, german)


@pytest.mark.skipif(not _ecantorix_ready(),
                    reason="the eCantorix engine is not installed")
@pytest.mark.parametrize("effect, lang, shape", [
    pytest.param("flite", "rms", 1, marks=pytest.mark.skipif(
        not (psola.shutil.which("flite") and psola.shutil.which("bc")),
        reason="the flite effect runs flite and bc")),
    ("tremolo", "en", 2), ("melt", "en", 2)])
def test_every_effect_renders(effect, lang, shape):
    """Their files are in the engine's examples, which were not where the
    configuration loaded them from; melt also needs a copy of espeak's
    data, which the engine's Makefile made from a Linux path."""
    sound = perform.sing(text="laa", notes=(0,), durs=(4,), effect=effect,
                         lang=lang, backend="ecantorix")
    assert sound.ndim == shape
    if shape == 2:
        assert sound.shape[0] == 2
    assert np.abs(sound).max() == pytest.approx(1)


@pytest.mark.skipif(not (_ecantorix_ready()
                         and not psola.missing_requirements()),
                    reason="comparing needs both backends")
def test_the_two_backends_sing_the_same_notes(sung):
    """The comparison the psola backend exists for: the same score, the
    same pitches, the same length."""
    score = SCORES["mary"]
    seconds = expected_seconds(score)
    ecantorix = sung("ecantorix", "mary")
    ours = sung("psola", "mary")
    assert abs(len(ecantorix) - len(ours)) < 0.01 * 44100
    for one, other in zip(note_pitches(ecantorix, seconds),
                          note_pitches(ours, seconds)):
        assert abs(cents(one, other)) < 50
