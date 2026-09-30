"""The psola singing backend, with espeak and Praat stood in for.

What the backend does with a score -- which syllable each note gets, how
long each note is, what pitch each is asked for, how the notes are
joined -- is checked here with stand-ins for espeak-ng and Parselmouth,
so it runs where neither is installed. ``test_singing_engine.py`` sings
with the real ones and measures what comes out.
"""

import re
import sys
import types

import numpy as np
import pytest

import music.singing.perform as perform
import music.singing.psola as psola


def _exactly(message):
    return "^" + re.escape(message) + "$"


# --------------------------------------------------------------------------
# What it needs
# --------------------------------------------------------------------------

def test_espeak_ng_is_preferred_to_espeak(monkeypatch):
    asked = []
    monkeypatch.setattr(psola.shutil, "which", lambda name: (
        asked.append(name) or f"/bin/{name}"))
    assert psola.speaker() == "/bin/espeak-ng"
    assert asked == ["espeak-ng"]


def test_the_older_espeak_will_do(monkeypatch):
    monkeypatch.setattr(psola.shutil, "which", lambda name: (
        "/bin/espeak" if name == "espeak" else None))
    assert psola.speaker() == "/bin/espeak"


def test_without_either_there_is_no_speaker(monkeypatch):
    monkeypatch.setattr(psola.shutil, "which", lambda name: None)
    assert psola.speaker() is None


@pytest.mark.parametrize("has_speaker, has_parselmouth, missing", [
    (True, True, []),
    (False, True, ["espeak-ng"]),
    (True, False, ["praat-parselmouth"]),
    (False, False, ["espeak-ng", "praat-parselmouth"]),
])
def test_what_is_missing_is_named(monkeypatch, has_speaker, has_parselmouth,
                                  missing):
    monkeypatch.setattr(psola, "speaker",
                        lambda: "/bin/espeak-ng" if has_speaker else None)
    looked_for = []
    monkeypatch.setattr(psola.importlib.util, "find_spec", lambda name: (
        looked_for.append(name) or (object() if has_parselmouth else None)))
    assert psola.missing_requirements() == missing
    assert looked_for == ["parselmouth"]


@pytest.mark.parametrize("missing, message", [
    (["espeak-ng"],
     "the psola singing backend needs espeak-ng, which are not installed. "
     "espeak-ng: sudo apt install espeak-ng, or brew install espeak-ng."),
    (["praat-parselmouth"],
     "the psola singing backend needs praat-parselmouth, which are not "
     "installed. praat-parselmouth: pip install 'music[singing]'."),
    (["espeak-ng", "praat-parselmouth"],
     "the psola singing backend needs espeak-ng and praat-parselmouth, "
     "which are not installed. espeak-ng: sudo apt install espeak-ng, or "
     "brew install espeak-ng; praat-parselmouth: pip install "
     "'music[singing]'."),
])
def test_what_is_missing_says_how_to_install_it(monkeypatch, missing,
                                                message):
    monkeypatch.setattr(psola, "missing_requirements", lambda: missing)
    with pytest.raises(RuntimeError, match=_exactly(message)):
        psola.require()


def test_nothing_missing_is_quiet(monkeypatch):
    monkeypatch.setattr(psola, "missing_requirements", lambda: [])
    psola.require()


# --------------------------------------------------------------------------
# The score
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("Mar-ry had a litt-le lamb",
     ["Mar", "ry", "had", "a", "litt", "le", "lamb"]),
    ("  hey   ma-  -bro ", ["hey", "ma", "bro"]),
    ("", []),
])
def test_a_lyric_splits_at_spaces_and_hyphens(text, expected):
    assert psola.syllables(text) == expected


@pytest.mark.parametrize("word, said", [
    ("bro,", "bro"), ("dive?", "dive"), ("don't", "don't"), ("—", ""),
    ("Ent", "Ent")])
def test_a_syllable_is_said_without_its_punctuation(word, said):
    assert psola._clean(word) == said


@pytest.fixture
def stand_ins(monkeypatch):
    """psola with its speaker and its PSOLA recorded, not run.

    `_speak` gives back the syllable it was asked for, and `_sung` a
    constant as long as it was asked for, at the note's frequency, so a
    sung line can be read back note by note.
    """
    calls = {"speak": [], "sung": []}
    monkeypatch.setattr(psola, "require", lambda: None)
    monkeypatch.setattr(psola, "speaker", lambda: "/bin/espeak-ng")

    def speak(program, syllable, voice, path):
        calls["speak"].append((program, syllable, voice, path.name))
        return syllable

    def sung(sound, frequency, seconds):
        calls["sung"].append((sound, frequency, seconds))
        return np.full(int(round(seconds * psola.RATE)) + 7, frequency)

    monkeypatch.setattr(psola, "_speak", speak)
    monkeypatch.setattr(psola, "_sung", sung)
    return calls


def test_each_syllable_is_sung_at_its_note_for_its_length(stand_ins):
    line = psola.sing("la ti do,", notes=(0, 11, 12), durs=(1, 0.5, "3/2"),
                      L="1/8", Q="1/4=120", reference=60, lang="pt-br",
                      transpose=-12)
    assert [s[1:3] for s in stand_ins["speak"]] == [
        ("la", "pt-br"), ("ti", "pt-br"), ("do", "pt-br")]
    assert [s[3] for s in stand_ins["speak"]] == ["0.wav", "1.wav", "2.wav"]
    # An eighth is a quarter of a second at 120 quarters a minute.
    seconds = [0.25, 0.125, 0.375]
    frequencies = [440 * 2 ** ((60 + n - 12 - 69) / 12) for n in (0, 11, 12)]
    assert [s[0] for s in stand_ins["sung"]] == ["la", "ti", "do"]
    np.testing.assert_allclose([s[1] for s in stand_ins["sung"]],
                               frequencies)
    # Asked for in whole samples, each within one of the score.
    np.testing.assert_allclose([s[2] for s in stand_ins["sung"]], seconds,
                               atol=1 / psola.RATE)
    assert len(line) == round(sum(seconds) * psola.RATE)
    edges = np.round(np.cumsum([0] + seconds) * psola.RATE).astype(int)
    for (start, end), frequency in zip(zip(edges, edges[1:]), frequencies):
        middle = line[start + 300:end - 300]
        np.testing.assert_allclose(
            middle, frequency / max(frequencies), rtol=1e-12)


def test_a_syllable_of_punctuation_alone_is_a_rest(stand_ins):
    line = psola.sing("la — la", notes=(0, 0, 0), durs=(1, 1, 1),
                      L="1/4", Q=120, reference=60, lang="en", transpose=0)
    assert [s[1] for s in stand_ins["speak"]] == ["la", "la"]
    rest = line[22050:44100]
    assert not rest.any() and line[:22050].any() and line[44100:].any()


def test_notes_join_with_short_fades(stand_ins):
    line = psola.sing("la la", notes=(0, 0), durs=(1, 1), L="1/4", Q=120,
                      reference=60, lang="en", transpose=0)
    fade = int(psola.FADE * psola.RATE)
    assert line[0] == 0 and line[fade] == pytest.approx(1)
    np.testing.assert_allclose(
        line[:fade], np.linspace(0, 1, fade, endpoint=False))
    # Each note fades out to nothing on its last sample, and the next
    # fades in from nothing on its first.
    assert line[22048] == pytest.approx(1 / fade)
    assert line[22049] == 0 and line[22050] == 0


def test_a_note_too_short_for_the_fade_fades_over_a_quarter():
    fitted = psola._fit(np.ones(40), 40)
    np.testing.assert_allclose(fitted[:10], np.linspace(0, 1, 10,
                                                        endpoint=False))
    np.testing.assert_allclose(fitted[10:30], 1)
    assert psola._fit(np.ones(2), 3).tolist() == [1.0, 1.0, 0.0]


def test_a_note_of_no_samples_is_skipped(stand_ins):
    line = psola.sing("la la", notes=(0, 0), durs=(1, 1e-9), L="1/4",
                      Q=120, reference=60, lang="en", transpose=0)
    assert len(stand_ins["sung"]) == 1 and len(line) == 22050


@pytest.mark.parametrize("text, notes, durs, counts", [
    ("la la la", (0, 0), (1, 1), (3, 2, 2)),
    ("la la", (0, 0), (1,), (2, 2, 1)),
])
def test_every_note_needs_a_syllable_and_a_duration(stand_ins, text, notes,
                                                    durs, counts):
    with pytest.raises(ValueError, match=_exactly(
            f"got {counts[0]} syllables, {counts[1]} notes and {counts[2]} "
            "durations; there must be one syllable and one duration for "
            "each note")):
        psola.sing(text, notes, durs)


def test_the_backend_says_what_it_lacks_before_anything_else(monkeypatch):
    monkeypatch.setattr(psola, "missing_requirements",
                        lambda: ["espeak-ng"])
    with pytest.raises(RuntimeError, match="needs espeak-ng"):
        psola.sing("la", (0,), (1,))


# --------------------------------------------------------------------------
# Through sing()
# --------------------------------------------------------------------------

def test_sing_hands_psola_its_score(monkeypatch):
    handed = []
    monkeypatch.setattr(psola, "sing", lambda *a, **k: (
        handed.append((a, k)) or np.zeros(3)))
    perform.sing(text="la la", notes=(0, 12), durs=(1, 2), M="3/4",
                 L="1/8", Q=90, K="G", reference=48, lang="de",
                 transpose=5, backend="psola")
    assert handed == [(("la la", (0, 12), (1, 2)),
                       dict(L="1/8", Q=90, reference=48, lang="de",
                            transpose=5))]


def test_sing_refuses_an_unknown_backend():
    with pytest.raises(ValueError, match=_exactly(
            "backend must be one of ('ecantorix', 'psola'); got 'festival'")):
        perform.sing(backend="festival")


def test_sing_refuses_an_effect_with_psola():
    with pytest.raises(ValueError, match=_exactly(
            "effect 'melt' is one of eCantorix's voices; the psola backend "
            "sings the plain voice only")):
        perform.sing(effect="melt", backend="psola")


def test_sing_refuses_the_notes_ecantorix_would(monkeypatch):
    monkeypatch.setattr(psola, "sing", lambda *a, **k: np.zeros(3))
    with pytest.raises(ValueError, match="outside the 12 to 96"):
        perform.sing(text="la", notes=(40,), durs=(1,), backend="psola")
    with pytest.raises(ValueError, match="2 notes and 1 durations"):
        perform.sing(text="la la", notes=(0, 0), durs=(1,),
                     backend="psola")


# --------------------------------------------------------------------------
# Speaking and PSOLA, against a stand-in Parselmouth
# --------------------------------------------------------------------------

class _Sound:
    """Enough of parselmouth.Sound for psola."""

    def __init__(self, source, rate=22050.0, calls=None):
        self.values = np.array([source], dtype=float) \
            if not isinstance(source, str) else _Sound.written[source]
        self.sampling_frequency = rate
        self.calls = calls if calls is not None else []

    written: dict = {}

    def extract_part(self, start, end, preserve_times):
        self.calls.append(("extract_part", start, end, preserve_times))
        rate = self.sampling_frequency
        return _Sound(self.values[0][int(round(start * rate)):
                                     int(round(end * rate))], rate,
                      self.calls)


@pytest.fixture
def parselmouth(monkeypatch):
    module = types.ModuleType("parselmouth")
    module.Sound = _Sound
    praat = types.ModuleType("parselmouth.praat")
    calls = []
    module.calls = calls
    praat.call = lambda *args: calls.append(args) or module.answers.pop(0)
    module.answers = []
    module.praat = praat
    monkeypatch.setitem(sys.modules, "parselmouth", module)
    monkeypatch.setitem(sys.modules, "parselmouth.praat", praat)
    return module


def _said(monkeypatch, samples, returncode=0, stderr="", write=True):
    ran = []

    def run(command, capture_output, text):
        ran.append(command)
        path = command[command.index("-w") + 1]
        if write:
            _Sound.written[path] = np.array([samples], dtype=float)
            open(path, "w").close()
        return types.SimpleNamespace(returncode=returncode, stderr=stderr)

    monkeypatch.setattr(psola.subprocess, "run", run)
    return ran


def test_a_syllable_is_said_and_trimmed_of_its_silence(tmp_path,
                                                       monkeypatch,
                                                       parselmouth):
    samples = [0, 0.001, 0.5, -1.0, 0.3, 0.005, 0, 0]
    ran = _said(monkeypatch, samples)
    path = tmp_path / "3.wav"
    trimmed = psola._speak("/bin/espeak-ng", "la", "en", path)
    assert ran == [["/bin/espeak-ng", "-v", "en", "-w", str(path), "la"]]
    np.testing.assert_array_equal(trimmed.values[0], [0.5, -1.0, 0.3])
    assert trimmed.calls == [("extract_part", 2 / 22050, 5 / 22050, False)]


@pytest.mark.parametrize("returncode, stderr, write, said", [
    (1, "Failed to read voice 'xx'\n", False, "Failed to read voice 'xx'"),
    (0, "", False, "it wrote nothing"),
])
def test_a_syllable_the_speaker_cannot_say_says_why(tmp_path, monkeypatch,
                                                    parselmouth, returncode,
                                                    stderr, write, said):
    _said(monkeypatch, [1.0], returncode, stderr, write)
    with pytest.raises(RuntimeError, match=_exactly(
            f"espeak-ng could not say 'la' in voice 'xx': {said}")):
        psola._speak("/usr/bin/espeak-ng", "la", "xx", tmp_path / "0.wav")


def test_a_syllable_said_as_silence_says_so(tmp_path, monkeypatch,
                                            parselmouth):
    _said(monkeypatch, [0.0, 0.0])
    with pytest.raises(RuntimeError, match=_exactly(
            "espeak said 'la' in voice 'en' as silence")):
        psola._speak("/bin/espeak", "la", "en", tmp_path / "0.wav")


class _Pitch:
    def __init__(self, times, frequencies):
        self._times = np.array(times)
        self.selected_array = {"frequency": np.array(frequencies)}

    def xs(self):
        return self._times


class _Spoken:
    def __init__(self, total, times, frequencies):
        self.total, self.pitch = total, _Pitch(times, frequencies)
        self.asked = []

    def get_total_duration(self):
        return self.total

    def to_pitch(self, time_step, pitch_floor, pitch_ceiling):
        self.asked.append((time_step, pitch_floor, pitch_ceiling))
        return self.pitch


def _psola_answers(parselmouth, resampled):
    """Praat's answers, in the order _sung asks, ending with the result."""
    sung = types.SimpleNamespace(values=np.array([resampled]))
    parselmouth.answers[:] = ["manipulation", "pitch tier", None, None,
                              None, None, "duration tier"]
    return sung


def test_the_voiced_stretch_alone_is_lengthened(parselmouth):
    spoken = _Spoken(0.5, [0.1, 0.2, 0.3, 0.4], [0, 120, 125, 0])
    result = _psola_answers(parselmouth, [0.25, -0.5])
    parselmouth.answers += [None] * 5 + ["resynthesis", result]
    samples = psola._sung(spoken, 261.6, 1.3)
    np.testing.assert_array_equal(samples, [0.25, -0.5])
    assert spoken.asked == [(0.01, 60, 600)]
    # 0.4 s is unvoiced, so the 0.1 s voiced stretch becomes 0.9 s.
    calls = parselmouth.calls
    assert calls[:6] == [
        (spoken, "To Manipulation", 0.01, 60, 600),
        ("manipulation", "Extract pitch tier"),
        ("pitch tier", "Remove points between", 0, 0.5),
        ("pitch tier", "Add point", 0, 261.6),
        ("pitch tier", "Add point", 0.5, 261.6),
        (["pitch tier", "manipulation"], "Replace pitch tier")]
    assert calls[6] == ("manipulation", "Extract duration tier")
    points = [call[2:] for call in calls[7:11]]
    assert points[0] == pytest.approx((0.199, 1))
    assert points[1] == pytest.approx((0.2, 9))
    assert points[2] == pytest.approx((0.3, 9))
    assert points[3] == pytest.approx((0.301, 1))
    assert calls[11:] == [
        (["duration tier", "manipulation"], "Replace duration tier"),
        ("manipulation", "Get resynthesis (overlap-add)"),
        ("resynthesis", "Resample", 44100, 50)]


@pytest.mark.parametrize("times, frequencies, seconds", [
    ([0.1, 0.2], [0, 0], 1.0),          # nothing voiced
    ([0.1, 0.2, 0.3], [0, 120, 0], 1.0),  # one voiced frame: no stretch
    ([0.1, 0.2, 0.3], [120, 120, 120], 0.2),  # too short to keep the rest
])
def test_otherwise_the_syllable_is_scaled_whole(parselmouth, times,
                                                frequencies, seconds):
    spoken = _Spoken(0.5, times, frequencies)
    result = _psola_answers(parselmouth, [1.0])
    parselmouth.answers += [None, None, "resynthesis", result]
    psola._sung(spoken, 220.0, seconds)
    duration_points = [call for call in parselmouth.calls
                       if call[0] == "duration tier"]
    assert duration_points == [
        ("duration tier", "Add point", 0, seconds / 0.5)]


# --------------------------------------------------------------------------
# Lengths and tempi, as both backends read them
# --------------------------------------------------------------------------

from fractions import Fraction  # noqa: E402


@pytest.mark.parametrize("duration, length", [
    (1, 1), (2, 2), (0.5, Fraction(1, 2)), (-2, Fraction(1, 2)),
    (-4, Fraction(1, 4)), ("4", 4), ("1", 1), ("3/2", Fraction(3, 2)),
    ("/2", Fraction(1, 2)), ("/4", Fraction(1, 4)), ("3/", Fraction(3, 2)),
    ("/", Fraction(1, 2)), ("//", Fraction(1, 4)), ("///", Fraction(1, 8)),
    ("1-2", Fraction(1, 2)), (1 / 1001, Fraction(1, 1000)),
])
def test_a_duration_is_read_as_abc_reads_a_length(duration, length):
    """Numbers to a thousandth of a unit, and ABC's slashes that halve."""
    assert perform._note_length(duration) == length


@pytest.mark.parametrize("duration, message", [
    (0, "a note cannot last no time; got a duration of 0"),
    ("0", "a note cannot last no time; got a duration of '0'"),
    ("0/4", "a note cannot last no time; got a duration of '0/4'"),
    ("x", "'x' is not a length ABC can read, such as \"3/2\" or \"/2\""),
    ("1.5", "'1.5' is not a length ABC can read, such as \"3/2\" or "
     "\"/2\""),
])
def test_a_duration_that_is_no_length_is_refused(duration, message):
    with pytest.raises(ValueError, match=_exactly(message)):
        perform._note_length(duration)


@pytest.mark.parametrize("L, Q, seconds", [
    ("1/4", 120, Fraction(1, 2)), ("1/8", 120, Fraction(1, 2)),
    ("1/8", "1/4=120", Fraction(1, 4)), ("1/4", "3/8=40", Fraction(1)),
    ("1/4", 1, Fraction(60)), ("1/4", 90.5, Fraction(120, 181)),
    ("1/4", 1 / 1001, Fraction(60000)),
])
def test_a_unit_lasts_what_abc_s_tempo_says(L, Q, seconds):
    """A bare Q counts units of L a minute; "beat=count" counts beats."""
    assert perform.unit_seconds(L, Q) == seconds


@pytest.mark.parametrize("L, Q, message", [
    ("x", 120, "L must be a fraction such as \"1/4\"; got 'x'"),
    ("0", 120, "L must be positive; got '0'"),
    ("-1/4", 120, "L must be positive; got '-1/4'"),
    ("1/4", "fast", "Q must be a number or \"beat=count\", such as "
     "\"1/4=120\"; got 'fast'"),
    ("1/4", "x=120", "the beat in Q must be a fraction such as \"1/4\"; "
     "got 'x'"),
    ("1/4", "1/4=x", "the count in Q must be a fraction such as \"1/4\"; "
     "got 'x'"),
    ("1/4", "1/4=1=2", "the count in Q must be a fraction such as "
     "\"1/4\"; got '1=2'"),
    ("1/4", 0, "the tempo must be positive; got Q=0"),
    ("1/4", -60, "the tempo must be positive; got Q=-60"),
    ("1/4", True, "Q must be a number or \"beat=count\", such as "
     "\"1/4=120\"; got True"),
])
def test_a_tempo_abc_cannot_read_is_refused(L, Q, message):
    with pytest.raises(ValueError, match=_exactly(message)):
        perform.unit_seconds(L, Q)


def test_a_bare_unit_is_a_quarter_at_120():
    assert perform.unit_seconds() == Fraction(1, 2)


# --------------------------------------------------------------------------
# What each call to a program is given
# --------------------------------------------------------------------------

def test_the_speaker_s_output_is_captured_as_text(tmp_path, monkeypatch,
                                                  parselmouth):
    given = []

    def run(command, **kwargs):
        given.append(kwargs)
        path = command[command.index("-w") + 1]
        _Sound.written[path] = np.array([[0.5, 1.0]])
        open(path, "w").close()
        return types.SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(psola.subprocess, "run", run)
    psola._speak("/bin/espeak-ng", "la", "en", tmp_path / "0.wav")
    assert given == [{"capture_output": True, "text": True}]


def test_the_trim_threshold_is_a_hundredth_of_the_peak(tmp_path,
                                                       monkeypatch,
                                                       parselmouth):
    """At a peak of a half, 0.01 is above a hundredth of it, 0.005."""
    _said(monkeypatch, [0.004, 0.01, 0.5, 0.01, 0.004])
    trimmed = psola._speak("/bin/espeak-ng", "la", "en", tmp_path / "0.wav")
    np.testing.assert_array_equal(trimmed.values[0], [0.01, 0.5, 0.01])


def test_every_syllable_is_said_by_the_speaker_found(stand_ins):
    psola.sing("la la", notes=(0, 0), durs=(1, 1))
    assert [call[0] for call in stand_ins["speak"]] == ["/bin/espeak-ng"] * 2


def test_the_voiced_stretch_is_lengthened_by_these_exact_calls(parselmouth):
    spoken = _Spoken(0.5, [0.1, 0.2, 0.3, 0.4], [0, 120, 125, 0])
    result = _psola_answers(parselmouth, [0.0])
    parselmouth.answers += [None] * 5 + ["resynthesis", result]
    psola._sung(spoken, 261.6, 1.3)
    added = parselmouth.calls[7:11]
    assert [call[:2] for call in added] == [("duration tier", "Add point")] * 4


@pytest.mark.parametrize("times, frequencies, seconds", [
    ([0.0, 0.01], [120, 120], 1.0),        # a voiced stretch of 0.01 s
    ([0.25, 0.5], [120, 120], 0.25),       # exactly the unvoiced length
])
def test_at_the_boundaries_the_syllable_is_scaled_whole(parselmouth, times,
                                                        frequencies,
                                                        seconds):
    spoken = _Spoken(0.5, times, frequencies)
    result = _psola_answers(parselmouth, [1.0])
    parselmouth.answers += [None, None, "resynthesis", result]
    psola._sung(spoken, 220.0, seconds)
    assert [call for call in parselmouth.calls
            if call[0] == "duration tier"] == [
        ("duration tier", "Add point", 0, seconds / 0.5)]


def test_psola_s_own_defaults_are_sing_s(stand_ins):
    """sing() passes every one of them; a direct call gets the same."""
    psola.sing("la", notes=(0,), durs=(1,))
    (_, _, voice, _), = stand_ins["speak"]
    (_, frequency, seconds), = stand_ins["sung"]
    assert voice == "en"
    assert frequency == pytest.approx(440 * 2 ** ((48 - 69) / 12))
    assert seconds == pytest.approx(0.5)
