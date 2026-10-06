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
    monkeypatch.setattr(psola, "missing_requirements",
                        lambda effect=None: missing)
    with pytest.raises(RuntimeError, match=_exactly(message)):
        psola.require()


def test_nothing_missing_is_quiet(monkeypatch):
    monkeypatch.setattr(psola, "missing_requirements",
                        lambda effect=None: [])
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


class _Said(str):
    """A syllable as the stand-in speaker says it: its text, said in
    `seconds`."""

    def __new__(cls, text, seconds):
        said = super().__new__(cls, text)
        said.seconds = seconds
        return said

    def get_total_duration(self):
        return self.seconds


@pytest.fixture
def stand_ins(monkeypatch):
    """psola with its speaker and its PSOLA recorded, not run.

    `_speak` gives back the syllable it was asked for, said in
    ``calls["said in"]`` seconds at each speed -- by default longer than
    any note here, so that none is said slower or held -- and `_sung` a
    constant as long as it was asked for, at the note's frequency, so a
    sung line can be read back note by note. `_nucleus` finds
    ``calls["nucleus"]``. No syllable is said without a vowel, unless a
    test puts one in ``calls["phonemes"]``.
    """
    calls = {"speak": [], "sung": [], "transcribed": [], "phonemes": {},
             "required": [], "said in": {None: np.inf},
             "nucleus": (0.1, 0.2), "nuclei": [], "vibratos": []}
    monkeypatch.setattr(psola, "require",
                        lambda effect=None: calls["required"].append(effect))
    monkeypatch.setattr(psola, "speaker", lambda: "/bin/espeak-ng")

    def phonemes(program, voice, syllables):
        calls["transcribed"].append((program, voice, list(syllables)))
        return calls["phonemes"]

    monkeypatch.setattr(perform, "sung_phonemes", phonemes)

    def speak(program, syllable, voice, path, speed=None):
        calls["speak"].append((program, syllable, voice, path.name, speed))
        return _Said(syllable, calls["said in"][speed])

    def sung(sound, frequency, seconds, region=None, vibrato=None):
        calls["sung"].append((sound, frequency, seconds, region))
        calls["vibratos"].append(vibrato)
        return np.full(int(round(seconds * psola.RATE)) + 7, frequency)

    def nucleus(sound):
        calls["nuclei"].append(sound)
        return calls["nucleus"]

    monkeypatch.setattr(psola, "_speak", speak)
    monkeypatch.setattr(psola, "_sung", sung)
    monkeypatch.setattr(psola, "_nucleus", nucleus)
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
    # In 4/4, abc2midi accents the first note, and none of the others
    # starts on a half note: an eighth and a sixteenth in.
    levels = [frequency * accent for frequency, accent in zip(
        frequencies, [1, 126 / 165, 126 / 165])]
    for (start, end), level in zip(zip(edges, edges[1:]), levels):
        middle = line[start + 300:end - 300]
        np.testing.assert_allclose(middle, level / max(levels), rtol=1e-12)


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
                        lambda effect=None: ["espeak-ng"])
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
                       dict(M="3/4", L="1/8", Q=90, reference=48,
                            lang="de", transpose=5, effect=None))]


def test_sing_sings_with_psola_by_default(monkeypatch):
    """eCantorix was the default, and needed its engine cloned, Perl and
    four of its modules, abc2midi and sox; PSOLA needs espeak-ng and a pip
    extra."""
    handed = []
    monkeypatch.setattr(psola, "sing", lambda *a, **k: (
        handed.append((a, k)) or np.zeros(3)))
    perform.sing()
    assert handed == [(("Mar-ry had a litt-le lamb", (4, 2, 0, 2, 4, 4, 4),
                        (1, 1, 1, 1, 1, 1, 2)),
                       dict(M="4/4", L="1/4", Q=120, reference=60,
                            lang="en", transpose=-12, effect=None))]


def test_sing_refuses_an_unknown_backend():
    with pytest.raises(ValueError, match=_exactly(
            "backend must be one of ('psola', 'ecantorix'); got 'festival'")):
        perform.sing(backend="festival")


@pytest.mark.parametrize("asked, handed", [
    ("tremolo", "tremolo"), ("melt", "melt"), ("flite", "flite"),
    ("flint", "flite")])
def test_sing_hands_psola_its_effect(monkeypatch, asked, handed):
    """Both backends sing eCantorix's effects, under the same names."""
    given = []
    monkeypatch.setattr(psola, "sing", lambda *a, **k: (
        given.append(k["effect"]) or np.zeros(3)))
    perform.sing(effect=asked, lang="rms" if handed == "flite" else "en")
    assert given == [handed]


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


@pytest.mark.parametrize("program", ["/bin/espeak-ng", "/bin/espeak"])
def test_espeak_can_say_a_syllable_slower(tmp_path, monkeypatch,
                                          parselmouth, program):
    ran = _said(monkeypatch, [0, 0.5, -1.0, 0])
    path = tmp_path / "0.wav"
    trimmed = psola._speak(program, "la", "en", path, speed=95)
    assert ran == [[program, "-v", "en", "-w", str(path), "-s", "95", "la"]]
    np.testing.assert_array_equal(trimmed.values[0], [0.5, -1.0])


def test_flite_is_never_asked_for_a_speed(tmp_path):
    path = tmp_path / "0.wav"
    assert psola._command("/bin/flite", "la", "rms", path, speed=95) == [
        "/bin/flite", "-voice", "rms", "-t", "la", "-o", str(path)]


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


def _region_points(parselmouth, spoken, seconds, region):
    """Record the experimental tier without requiring a Praat install."""
    answers = {
        "To Manipulation": "manipulation",
        "Extract pitch tier": "pitch tier",
        "Extract duration tier": "duration tier",
        "Get resynthesis (overlap-add)": "resynthesis",
        "Resample": types.SimpleNamespace(values=np.array([[0.25, -0.5]])),
    }

    def call(*args):
        parselmouth.calls.append(args)
        return answers.get(args[1])

    parselmouth.praat.call = call
    samples = psola._sung(spoken, 220.0, seconds, region=region)
    np.testing.assert_array_equal(samples, [0.25, -0.5])
    assert not spoken.asked  # the supplied region needs no pitch analysis
    return [call[2:] for call in parselmouth.calls
            if call[0] == "duration tier"]


def _tier_duration(points, start, end):
    """The time occupied by a source interval under a linear duration tier."""
    times, values = np.asarray(points).T
    edges = np.r_[start, times[(times > start) & (times < end)], end]
    rates = np.interp(edges, times, values)
    return np.sum(np.diff(edges) * (rates[:-1] + rates[1:]) / 2)


@pytest.mark.parametrize("seconds", [0.9, 0.5, 0.45, 0.3995])
def test_a_selected_hold_region_accounts_for_both_ramps(parselmouth,
                                                       seconds):
    spoken = _Spoken(0.5, [0.05, 0.45], [120, 120])
    points = _region_points(parselmouth, spoken, seconds, (0.2, 0.3))
    assert [time for time, _ in points] == pytest.approx(
        [0.199, 0.2, 0.3, 0.301])
    assert all(value > 0 for _, value in points)
    assert _tier_duration(points, 0, 0.5) == pytest.approx(seconds)
    # Only the selected region and its one-millisecond ramps change;
    # consonant transitions outside them retain their original duration.
    assert _tier_duration(points, 0, 0.199) == pytest.approx(0.199)
    assert _tier_duration(points, 0.301, 0.5) == pytest.approx(0.199)
    assert points[0][1] == points[-1][1] == 1


@pytest.mark.parametrize("region, times", [
    ((0, 0.3), [0, 0.3, 0.301]),
    ((0.2, 0.5), [0.199, 0.2, 0.5]),
    ((0, 0.5), [0, 0.5]),
    ((0.0005, 0.4995), [0, 0.0005, 0.4995, 0.5]),
])
def test_hold_ramps_at_sound_boundaries_have_no_duplicate_times(
        parselmouth, region, times):
    points = _region_points(parselmouth, _Spoken(0.5, [], []), 0.8, region)
    assert [time for time, _ in points] == pytest.approx(times)
    assert np.all(np.diff([time for time, _ in points]) > 0)
    assert _tier_duration(points, 0, 0.5) == pytest.approx(0.8)


@pytest.mark.parametrize("seconds", [0.249, 0.2])
def test_compression_that_cannot_fit_the_region_scales_the_whole_syllable(
        parselmouth, seconds):
    points = _region_points(parselmouth, _Spoken(0.5, [], []), seconds,
                            (0.125, 0.375))
    assert points == [(0, seconds / 0.5)]


@pytest.mark.parametrize("region", [
    (-0.1, 0.3), (0.2, 0.6), (0.3, 0.2), (0.2, 0.2),
    (0, 0.01), (float("nan"), 0.3), (0.2, float("inf")),
])
def test_an_unusable_hold_region_scales_the_whole_syllable(parselmouth,
                                                         region):
    points = _region_points(parselmouth, _Spoken(0.5, [], []), 0.8, region)
    assert points == [(0, 0.8 / 0.5)]


# --------------------------------------------------------------------------
# A note longer than its syllable: said slower, and its vowel held
# --------------------------------------------------------------------------

def test_a_note_shorter_than_its_syllable_is_said_once_and_fitted(stand_ins):
    stand_ins["said in"] = {None: 0.6}
    psola.sing("la", notes=(0,), durs=(1,))
    assert [call[4] for call in stand_ins["speak"]] == [None]
    assert stand_ins["nuclei"] == []
    (sound, _, seconds, region), = stand_ins["sung"]
    assert (sound.seconds, seconds, region) == (0.6, 0.5, None)


def test_a_longer_note_has_its_syllable_said_slower_and_its_vowel_held(
        stand_ins):
    """eCantorix asks espeak for the speed that fits each note; the
    syllable said nearest the note's half second is kept, 88 words a
    minute here, and the middle of its vowel held for the rest."""
    stand_ins["said in"] = {None: 0.25, 88: 0.45, 80: 0.4}
    psola.sing("lamb", notes=(0,), durs=(1,))
    # 175 * 0.25 / 0.5 asks for 88; 88 * 0.45 / 0.5 for 79, which is
    # slower than SLOWEST; and 80 * 0.4 / 0.5 for 80 again, which ends it.
    assert [call[4] for call in stand_ins["speak"]] == [None, 88, 80]
    assert [call[3] for call in stand_ins["speak"]] == ["0.wav"] * 3
    (held,) = stand_ins["nuclei"]
    assert held.seconds == 0.45
    (sound, _, seconds, region), = stand_ins["sung"]
    assert sound is held and seconds == 0.5 and region == (0.1, 0.2)


def test_a_syllable_said_slower_than_its_note_is_fitted_not_held(stand_ins):
    stand_ins["said in"] = {None: 0.3, 105: 0.52}
    psola.sing("lamb", notes=(0,), durs=(1,))
    assert [call[4] for call in stand_ins["speak"]] == [None, 105]
    assert stand_ins["nuclei"] == []
    (sound, _, _, region), = stand_ins["sung"]
    assert sound.seconds == 0.52 and region is None


def test_a_vowel_with_no_nucleus_to_hold_leaves_psola_to_choose(stand_ins):
    stand_ins["said in"] = {None: 0.48}
    stand_ins["nucleus"] = None
    psola.sing("lamb", notes=(0,), durs=(1,))
    assert [call[4] for call in stand_ins["speak"]] == [None]
    (_, _, _, region), = stand_ins["sung"]
    assert region is None and len(stand_ins["nuclei"]) == 1


def test_flite_is_held_but_never_asked_to_speak_slower(stand_ins,
                                                      monkeypatch):
    monkeypatch.setattr(psola.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(psola, "_require_flite_voice", lambda lang: None)
    stand_ins["said in"] = {None: 0.25}
    psola.sing("la", notes=(0,), durs=(1,), lang="slt", effect="flite")
    assert [call[::4] for call in stand_ins["speak"]] == [("/bin/flite", None)]
    (_, _, _, region), = stand_ins["sung"]
    assert region == (0.1, 0.2)


def test_the_melted_voice_is_timed_by_the_length_it_is_sung_for(stand_ins,
                                                               monkeypatch):
    """Resampled by a quarter at 220 Hz, a half-second note is sung for
    0.625 s, which a syllable said in 0.55 s is short of."""
    monkeypatch.setattr(psola, "_resampled", lambda samples, shift: samples)
    monkeypatch.setattr(psola, "_trembling", lambda samples: samples)
    monkeypatch.setattr(psola, "_reverberated", lambda line: np.array(
        [line, line]))
    stand_ins["said in"] = {None: 0.55, 154: 0.6}
    psola.sing("la", notes=(9,), durs=(1,), effect="melt")
    assert [call[4] for call in stand_ins["speak"]] == [None, 154]
    (sound, _, seconds, region), = stand_ins["sung"]
    assert seconds == pytest.approx(0.625) and sound.seconds == 0.6
    assert region == (0.1, 0.2)


def test_every_syllable_is_sung_with_the_vibrato(stand_ins):
    psola.sing("la la", notes=(0, 7), durs=(1, 4))
    times = np.linspace(0, 2, 401)
    for vibrato in stand_ins["vibratos"]:
        np.testing.assert_array_equal(vibrato(times), psola._vibrato(times))


def test_the_melted_voice_s_vibrato_is_drawn_slower_by_its_shift(
        stand_ins, monkeypatch):
    """Resampled a quarter faster at 220 Hz, it comes out at its rate."""
    monkeypatch.setattr(psola, "_resampled", lambda samples, shift: samples)
    monkeypatch.setattr(psola, "_trembling", lambda samples: samples)
    monkeypatch.setattr(psola, "_reverberated", lambda line: np.array(
        [line, line]))
    psola.sing("la", notes=(9,), durs=(4,), effect="melt")
    (vibrato,) = stand_ins["vibratos"]
    times = np.linspace(0, 2.5, 501)
    np.testing.assert_allclose(vibrato(times), psola._vibrato(times / 1.25))


def _speeds(monkeypatch, durations):
    """`_speak` saying "lamb" in `durations[speed]` seconds."""
    asked = []

    def speak(program, syllable, voice, path, speed):
        asked.append(speed)
        return _Said(syllable, durations[speed])

    monkeypatch.setattr(psola, "_speak", speak)
    return asked


@pytest.mark.parametrize("seconds, said, durations, chosen, asked", [
    (0.6, 0.3, {88: 0.6}, 88, [88]),
    (4.0, 0.3, {80: 0.7}, 80, [80]),
    # Within 5 percent of the note: nothing more to ask.
    (0.6, 0.58, {}, None, []),
    # Three more tries at most, and the nearest kept, not the last.
    (0.6, 0.3, {88: 0.75, 110: 0.7, 128: 0.8}, 110, [88, 110, 128]),
    # Never faster than espeak's own speed: 146 words a minute comes out
    # too long, and would ask for 219.
    (0.6, 0.5, {146: 0.9}, None, [146]),
])
def test_a_syllable_is_slowed_toward_its_note_never_hurried(
        monkeypatch, seconds, said, durations, chosen, asked):
    tried = _speeds(monkeypatch, durations)
    spoken = _Said("lamb", said)
    best = psola._slowed("/bin/espeak-ng", "lamb", "en", "0.wav", seconds,
                         spoken)
    assert tried == asked
    assert best is spoken if chosen is None else \
        best.seconds == durations[chosen]


def _frames(levels, frequencies=None):
    levels = np.asarray(levels, dtype=float)
    times = np.arange(len(levels)) * 0.01 + 0.025
    if frequencies is None:
        frequencies = np.full(len(levels), 120.0)
    return times, np.asarray(frequencies), levels


def test_the_nucleus_leaves_a_weak_voiced_onset_and_coda_unheld():
    frames = _frames([0.2] * 10 + [1.0] * 10 + [0.3] * 10)
    start, end = psola._nucleus_from_frames(*frames)
    assert (start, end) == pytest.approx((0.135, 0.205))
    assert start > frames[0][9] and end < frames[0][20]


def test_a_loud_unvoiced_fricative_is_not_taken_for_the_vowel():
    frames = _frames([8.0] * 10 + [1.0] * 10 + [0.2] * 10,
                     [0] * 10 + [120] * 20)
    assert psola._nucleus_from_frames(*frames) == pytest.approx(
        (0.135, 0.205))


def test_a_voiceless_gap_between_vowels_is_not_held():
    frames = _frames([1.0] * 10 + [0.9] * 20,
                     [120] * 10 + [0] * 10 + [120] * 10)
    start, end = psola._nucleus_from_frames(*frames)
    assert end < frames[0][10]
    assert start == pytest.approx(0.035)


@pytest.mark.parametrize("levels, frequencies", [
    ([], []),
    ([1] * 20, [0] * 20),
    ([0] * 20, [120] * 20),
    ([1] * 4, [120] * 4),
    ([0.1] * 10 + [1] + [0.1] * 10, [120] * 21),
])
def test_no_hold_without_enough_contiguous_strong_voicing(levels,
                                                         frequencies):
    assert psola._nucleus_from_frames(*_frames(levels, frequencies)) is None


def test_the_shortest_nucleus_held_is_thirty_milliseconds():
    start, end = psola._nucleus_from_frames(*_frames([1] * 6))
    assert end - start == pytest.approx(0.03)


def test_a_syllable_too_short_to_have_a_pitch_has_no_nucleus():
    spoken = _Spoken(psola.SHORTEST, [], [])
    assert psola._nucleus(spoken) is None
    assert spoken.asked == []


def test_the_nucleus_reads_the_pitch_and_a_40_ms_level_around_each_frame():
    """Voiced from 0.1 s, and loud from 0.2 s to 0.3 s."""
    rate = 1000.0
    samples = np.r_[np.full(200, 0.1), np.ones(100), np.full(100, 0.1)]
    spoken = _SpokenSamples(samples, rate, np.arange(0.005, 0.4, 0.01),
                            [0] * 10 + [120] * 30)
    start, end = psola._nucleus(spoken)
    assert spoken.asked == [(0.01, 60, 600)]
    # A frame whose 40 ms are at least half loud is at least 70 percent
    # as strong as the loudest, sqrt(0.505) of it: those from 0.205 s to
    # 0.295 s, held less 10 ms at each end.
    assert (start, end) == pytest.approx((0.215, 0.285))


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
# A vibrato, drawn where Praat reads it
# --------------------------------------------------------------------------

def test_the_vibrato_sets_in_late_and_grows_to_its_depth():
    times = np.array([0.0, 0.2, 0.25, 0.4, 0.55, 0.55 + 1 / 22, 2.0])
    semitones = psola._vibrato(times)
    np.testing.assert_allclose(semitones[:3], 0, atol=1e-12)
    # Half grown, 0.15 s in, at the sine's 0.825th turn.
    assert semitones[3] == pytest.approx(
        0.35 * 0.5 * np.sin(2 * np.pi * 5.5 * 0.15))
    # Full grown: a quarter turn later, a quarter period later.
    assert semitones[5] - semitones[4] == pytest.approx(
        0.35 * (np.sin(2 * np.pi * 5.5 * (0.3 + 1 / 22))
                - np.sin(2 * np.pi * 5.5 * 0.3)))
    assert np.abs(psola._vibrato(np.linspace(0.6, 4, 9999))).max() == \
        pytest.approx(0.35, abs=1e-4)


@pytest.mark.parametrize("points, total, ends", [
    ([(0, 2.0)], 0.5, 1.0),
    # Each millisecond's ramp, between 1 and 9, is sung in 5 ms.
    ([(0.199, 1), (0.2, 9.0), (0.3, 9.0), (0.301, 1)], 0.5, 1.308),
])
def test_the_warp_is_when_each_moment_of_the_syllable_is_sung(points, total,
                                                             ends):
    said, sung = psola._warp(points, total)
    assert said[0] == 0 and said[-1] == total and sung[0] == 0
    assert sung[-1] == pytest.approx(ends)
    assert np.all(np.diff(sung) > 0)
    for time, _ in points:
        assert time in said


def test_the_held_stretch_is_sung_at_its_rate():
    said, sung = psola._warp([(0.199, 1), (0.2, 9.0), (0.3, 9.0),
                              (0.301, 1)], 0.5)
    at = np.interp([0.1, 0.2, 0.25, 0.3, 0.4], said, sung)
    # The ramp from 1 to 9 over a millisecond is sung in 5 ms.
    np.testing.assert_allclose(at, [0.1, 0.204, 0.654, 1.104, 1.208])


def test_a_vibrato_is_drawn_every_5_ms_of_the_sung_syllable(parselmouth):
    spoken = _Spoken(0.5, [], [])
    vibrato = []

    def semitones(times):
        vibrato.append(times)
        return np.where(times < 0.5, 0.0, 1.0)

    answers = {"To Manipulation": "manipulation",
               "Extract pitch tier": "pitch tier",
               "Extract duration tier": "duration tier",
               "Get resynthesis (overlap-add)": "resynthesis",
               "Resample": types.SimpleNamespace(values=np.array([[0.0]]))}

    def call(*args):
        parselmouth.calls.append(args)
        return answers.get(args[1])

    parselmouth.praat.call = call
    psola._sung(spoken, 220.0, 1.3, region=(0.2, 0.3), vibrato=semitones)
    (times,) = vibrato
    np.testing.assert_allclose(times, np.append(np.arange(0, 1.3, 0.005),
                                                1.3))
    points = [args[2:] for args in parselmouth.calls
              if args[:2] == ("pitch tier", "Add point")]
    places, hertz = np.array(points).T
    # The held stretch's rate, its ramps counted, sings it in 1.3 s.
    factor = 1 + 0.8 / 0.101
    said, sung = psola._warp([(0.199, 1), (0.2, factor), (0.3, factor),
                              (0.301, 1)], 0.5)
    assert sung[-1] == pytest.approx(1.3)
    np.testing.assert_allclose(places, np.interp(times, sung, said))
    assert places[0] == 0 and places[-1] == pytest.approx(0.5)
    np.testing.assert_allclose(hertz, np.where(times < 0.5, 220.0,
                                               220.0 * 2 ** (1 / 12)))


# --------------------------------------------------------------------------
# Accents, as abc2midi gives them and eCantorix sings them
# --------------------------------------------------------------------------

FIRST, STRONG, OTHER = 1, 150 / 165, 126 / 165


@pytest.mark.parametrize("M, L, durs, levels", [
    # Every half note in 4/4, and the first note above them all.
    ("4/4", "1/4", [1] * 5, [FIRST, OTHER, STRONG, OTHER, STRONG]),
    ("C", "1/4", [1] * 5, [FIRST, OTHER, STRONG, OTHER, STRONG]),
    ("4/4", "1/4", [0.5] * 5, [FIRST, OTHER, OTHER, OTHER, STRONG]),
    ("4/4", "1/4", ["3/2", 0.5, "3/2", 0.5, 2, 1],
     [FIRST, OTHER, STRONG, OTHER, STRONG, STRONG]),
    ("4/4", "1/8", [2, 2, 1, 1, 2], [FIRST, OTHER, STRONG, OTHER, OTHER]),
    # Once a bar in 3/4 and 2/2, and every dotted quarter in 6/8.
    ("3/4", "1/4", [1] * 4, [FIRST, OTHER, OTHER, STRONG]),
    ("2/2", "1/4", [1] * 5, [FIRST, OTHER, OTHER, OTHER, STRONG]),
    ("C|", "1/4", [1] * 5, [FIRST, OTHER, OTHER, OTHER, STRONG]),
    ("6/8", "1/8", [1] * 7, [FIRST, OTHER, OTHER, STRONG, OTHER, OTHER,
                             STRONG]),
    ("5/8", "1/8", [1] * 6, [FIRST, OTHER, OTHER, OTHER, OTHER, STRONG]),
    ("1/4", "1/8", [1] * 3, [FIRST, OTHER, STRONG]),
    ("4/4", "1/4", [], []),
])
def test_the_beats_abc2midi_accents_are_sung_louder(M, L, durs, levels):
    """abc2midi gives the first note a velocity of 105, a note on a strong
    beat 95 and any other 80, and eCantorix has espeak say them at
    amplitudes of 165, 150 and 126."""
    assert perform.ACCENTS == (165, 150, 126)
    assert perform.accents(durs, M, L) == pytest.approx(levels)


@pytest.mark.parametrize("M", ["x", "4/0", "0/4", "3+2/8", "4/4/4", "",
                               "/4", "4/", None])
def test_a_meter_that_is_not_one_is_refused(M):
    with pytest.raises(ValueError, match=_exactly(
            f'M must be a meter such as "4/4", "C" or "C|"; got {M!r}')):
        perform.accents([1], M)


def test_psola_sings_each_note_at_its_accent(stand_ins):
    line = psola.sing("la la la", notes=(0, 0, 0), durs=(2, 1, 1), M="3/4",
                      reference=60, transpose=0)
    middles = [line[start + 500:start + 1000] for start in (0, 44100, 66150)]
    np.testing.assert_allclose([m.mean() for m in middles],
                               [FIRST, OTHER, STRONG], rtol=1e-12)


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


def test_a_syllable_too_short_to_have_a_pitch_is_sung_as_said(parselmouth):
    """espeak says French "ques" as a /k/ 41 ms long. Praat looks for a
    pitch over three periods of the lowest one, 50 ms at 60 Hz, refused
    to analyse it, and sing stopped with Praat's error."""
    spoken = _Spoken(0.05, [], [])
    parselmouth.answers[:] = [types.SimpleNamespace(
        values=np.array([[0.5, -0.5]]))]
    samples = psola._sung(spoken, 220.0, 0.5)
    np.testing.assert_array_equal(samples, [0.5, -0.5])
    assert parselmouth.calls == [(spoken, "Resample", 44100, 50)]
    assert spoken.asked == []


def test_a_selected_region_cannot_make_a_short_syllable_analysable(
        parselmouth):
    spoken = _Spoken(0.05, [], [])
    assert _region_points(parselmouth, spoken, 0.5, (0.01, 0.04)) == []
    assert parselmouth.calls == [(spoken, "Resample", 44100, 50)]


def test_a_syllable_just_long_enough_is_analysed(parselmouth):
    spoken = _Spoken(0.0501, [0.02], [0])
    result = _psola_answers(parselmouth, [1.0])
    parselmouth.answers += [None, None, "resynthesis", result]
    psola._sung(spoken, 220.0, 0.5)
    assert spoken.asked == [(0.01, 60, 600)]


class _SpokenSamples(_Spoken):
    """A spoken syllable with samples, which a long note pads."""

    def __init__(self, samples, rate, times, frequencies):
        super().__init__(len(samples) / rate, times, frequencies)
        self.values = np.array([samples], dtype=float)
        self.sampling_frequency = rate


def _manipulated(parselmouth):
    """The sound _sung made its Manipulation from."""
    (sound, command, *_), = [call for call in parselmouth.calls
                             if call[1:2] == ("To Manipulation",)]
    return sound


def test_a_note_longer_than_praat_can_write_is_given_room(parselmouth):
    """Praat's overlap-add writes into three times the length of the sound
    it is given, and stops there: a 0.3 s "laa" held for four seconds was
    sung for one, and the rest left silent."""
    spoken = _SpokenSamples([0.5, -0.5, 0.25, 0.0], 8.0, [0.1, 0.3],
                            [120, 120])
    result = _psola_answers(parselmouth, [1.0])
    parselmouth.answers += [None] * 5 + ["resynthesis", result]
    psola._sung(spoken, 220.0, 1.6)
    padded = _manipulated(parselmouth)
    np.testing.assert_array_equal(padded.values[0],
                                  [0.5, -0.5, 0.25, 0.0] + [0.0] * 13)
    assert padded.sampling_frequency == 8.0
    # The tiers are the syllable's, as they would be without the room.
    assert parselmouth.calls[2] == ("pitch tier", "Remove points between",
                                    0, 0.5)
    assert [call[2] for call in parselmouth.calls[7:11]] == pytest.approx(
        [0.099, 0.1, 0.3, 0.301])


@pytest.mark.parametrize("region", [(0.1, 0.3), (0.1, 0.5), (0, 0.5)])
def test_a_long_selected_hold_keeps_the_padding_at_unit_rate(parselmouth,
                                                            region):
    spoken = _SpokenSamples([0.5, -0.5, 0.25, 0.0], 8.0, [], [])
    points = _region_points(parselmouth, spoken, 1.6, region)
    padded = _manipulated(parselmouth)
    np.testing.assert_array_equal(padded.values[0],
                                  [0.5, -0.5, 0.25, 0.0] + [0.0] * 13)
    assert _tier_duration(points, 0, 0.5) == pytest.approx(1.6)
    assert points[-1][1] == 1
    assert np.all(np.diff([time for time, _ in points]) > 0)
    assert _tier_duration(points, 0.501, 1.6) == pytest.approx(1.099)


@pytest.mark.parametrize("seconds", [1.5, 1.4])
def test_a_note_praat_can_write_is_not_padded(parselmouth, seconds):
    spoken = _SpokenSamples([0.5, -0.5, 0.25, 0.0], 8.0, [0.1, 0.3],
                            [120, 120])
    result = _psola_answers(parselmouth, [1.0])
    parselmouth.answers += [None] * 5 + ["resynthesis", result]
    psola._sung(spoken, 220.0, seconds)
    assert _manipulated(parselmouth) is spoken


@pytest.mark.parametrize("times, frequencies", [
    ([0.1, 0.3], [0, 0]),           # nothing voiced to hold
    ([0.1, 0.105], [120, 120]),     # a voiced stretch of 0.005 s
])
def test_a_syllable_with_nothing_to_hold_is_not_padded(parselmouth, times,
                                                       frequencies):
    """Scaled whole, the silence would be scaled too, and make no room."""
    spoken = _SpokenSamples([0.5, -0.5, 0.25, 0.0], 8.0, times, frequencies)
    result = _psola_answers(parselmouth, [1.0])
    parselmouth.answers += [None, None, "resynthesis", result]
    psola._sung(spoken, 220.0, 1.6)
    assert _manipulated(parselmouth) is spoken


def test_padding_keeps_a_sound_s_rate_and_adds_whole_samples(parselmouth):
    spoken = _SpokenSamples([1.0, 2.0], 10.0, [], [])
    padded = psola._padded(spoken, 0.21)
    np.testing.assert_array_equal(padded.values[0], [1.0, 2.0, 0, 0, 0])
    assert padded.sampling_frequency == 10.0


def test_a_syllable_said_without_a_vowel_is_said_with_a_schwa(stand_ins):
    """espeak says French "ques" as a bare /k/, which left its note with
    nothing to sing."""
    stand_ins["phonemes"]["ques"] = "[[k@]]"
    psola.sing("Jac-ques, !", notes=(4, 0, 2), durs=(1, 1, 1), lang="fr")
    assert stand_ins["transcribed"] == [
        ("/bin/espeak-ng", "fr", ["Jac", "ques"])]
    assert [call[1] for call in stand_ins["speak"]] == ["Jac", "[[k@]]"]


def test_psola_s_own_defaults_are_sing_s(stand_ins):
    """sing() passes every one of them; a direct call gets the same."""
    psola.sing("la", notes=(0,), durs=(1,))
    (_, _, voice, _, _), = stand_ins["speak"]
    (_, frequency, seconds, _), = stand_ins["sung"]
    assert voice == "en"
    assert frequency == pytest.approx(440 * 2 ** ((48 - 69) / 12))
    assert seconds == pytest.approx(0.5)


# --------------------------------------------------------------------------
# Effects, after eCantorix's extra voices
# --------------------------------------------------------------------------

def test_an_effect_it_does_not_have_is_refused(stand_ins):
    with pytest.raises(ValueError, match=_exactly(
            "effect must be one of ('tremolo', 'melt', 'flite'), or None; "
            "got 'flint'")):
        psola.sing("la", (0,), (1,), effect="flint")
    assert stand_ins["required"] == []


@pytest.mark.parametrize("effect, has_speaker, has_flite, missing", [
    ("flite", False, True, []),
    ("flite", True, False, ["flite"]),
    (None, False, True, ["espeak-ng"]),
    ("melt", False, True, ["espeak-ng"]),
])
def test_flite_needs_flite_rather_than_espeak(monkeypatch, effect,
                                               has_speaker, has_flite,
                                               missing):
    monkeypatch.setattr(psola, "speaker",
                        lambda: "/bin/espeak-ng" if has_speaker else None)
    monkeypatch.setattr(psola.shutil, "which", lambda name: (
        "/bin/flite" if name == "flite" and has_flite else None))
    monkeypatch.setattr(psola.importlib.util, "find_spec",
                        lambda name: object())
    assert psola.missing_requirements(effect) == missing


def test_missing_flite_says_how_to_install_it(monkeypatch):
    monkeypatch.setattr(psola, "missing_requirements", lambda effect=None: (
        ["flite", "praat-parselmouth"] if effect == "flite" else []))
    with pytest.raises(RuntimeError, match=_exactly(
            "the psola singing backend needs flite and praat-parselmouth, "
            "which are not installed. flite: sudo apt install flite, or "
            "brew install flite; praat-parselmouth: pip install "
            "'music[singing]'.")):
        psola.require("flite")


def _flite_lists(monkeypatch, voices="kal awb_time kal16 awb rms slt"):
    asked = []

    def run(command, capture_output, text):
        asked.append(command)
        return types.SimpleNamespace(
            returncode=0, stdout=f"Voices available: {voices}\n", stderr="")

    monkeypatch.setattr(psola.subprocess, "run", run)
    return asked


def test_flite_s_voices_are_those_it_lists(monkeypatch):
    asked = _flite_lists(monkeypatch)
    assert psola._flite_voices() == ["kal", "awb_time", "kal16", "awb",
                                     "rms", "slt"]
    assert asked == [["flite", "-lv"]]


def test_a_voice_flite_lacks_is_refused(monkeypatch):
    """flite says a syllable in a voice it lacks in its default one, and
    says nothing of it."""
    _flite_lists(monkeypatch, "kal rms")
    psola._require_flite_voice("rms")
    with pytest.raises(ValueError, match=_exactly(
            "with the flite effect, lang names one of flite's voices, "
            "['kal', 'rms']; got 'en'")):
        psola._require_flite_voice("en")


@pytest.mark.parametrize("program, command", [
    ("/usr/bin/flite", ["/usr/bin/flite", "-voice", "rms", "-t", "la",
                        "-o", "/tmp/0.wav"]),
    ("C:/flite/flite.exe", ["C:/flite/flite.exe", "-voice", "rms", "-t",
                            "la", "-o", "/tmp/0.wav"]),
    ("/usr/bin/espeak-ng", ["/usr/bin/espeak-ng", "-v", "rms", "-w",
                            "/tmp/0.wav", "la"]),
])
def test_flite_is_asked_in_its_own_terms(program, command):
    from pathlib import Path
    assert psola._command(program, "la", "rms", Path("/tmp/0.wav")) == \
        command


def test_flite_says_each_syllable_as_written(stand_ins, monkeypatch):
    """flite reads no espeak phonemes, so no syllable is looked up."""
    monkeypatch.setattr(psola.shutil, "which",
                        lambda name: f"/bin/{name}")
    checked = []
    monkeypatch.setattr(psola, "_require_flite_voice", checked.append)
    psola.sing("Jac-ques", (4, 0), (1, 1), lang="rms", effect="flite")
    assert stand_ins["required"] == ["flite"]
    assert checked == ["rms"]
    assert stand_ins["transcribed"] == []
    assert [call[:3] for call in stand_ins["speak"]] == [
        ("/bin/flite", "Jac", "rms"), ("/bin/flite", "ques", "rms")]


@pytest.mark.parametrize("lang, voice", [
    ("en", "en+f1"), ("pt-br", "pt-br+f1"), ("en+m3", "en+m3")])
def test_the_melted_voice_is_espeak_s_female_one(lang, voice):
    assert psola._melted(lang) == voice


def test_melt_moves_the_formants_with_the_pitch_below_its_floor(
        stand_ins, monkeypatch):
    """eCantorix's melt voice is espeak's f1 with its formants a quarter
    higher, which espeak sings down to 216 Hz; below that eCantorix
    resamples it to each note, and its formants sink with the pitch."""
    resampled = []
    monkeypatch.setattr(psola, "_resampled", lambda samples, shift: (
        resampled.append(shift) or samples[:len(samples)]))
    monkeypatch.setattr(psola, "_trembling", lambda samples: samples)
    monkeypatch.setattr(psola, "_reverberated", lambda line: np.array(
        [line, line]))
    # MIDI 57 at 220 Hz is above the floor; MIDI 45 at 110 Hz below it.
    psola.sing("la la", notes=(9, -3), durs=(1, 1), effect="melt")
    assert [call[2] for call in stand_ins["speak"]] == ["en+f1"] * 2
    assert stand_ins["transcribed"] == [("/bin/espeak-ng", "en+f1",
                                         ["la", "la"])]
    above, below = 1.25, 1.25 * 110 / 216
    assert resampled == pytest.approx([above, below])
    assert [call[0] for call in stand_ins["sung"]] == ["la", "la"]
    (_, high, high_seconds, _), (_, low, low_seconds, _) = \
        stand_ins["sung"]
    assert high == pytest.approx(220 / above)
    assert low == pytest.approx(110 / below) == pytest.approx(216 / 1.25)
    assert high_seconds == pytest.approx(0.5 * above)
    assert low_seconds == pytest.approx(0.5 * below)


@pytest.mark.parametrize("effect", ["tremolo", "melt"])
def test_tremolo_and_melt_tremble_each_note_and_reverberate_the_line(
        stand_ins, monkeypatch, effect):
    trembled, reverberated = [], []
    monkeypatch.setattr(psola, "_resampled", lambda samples, shift: samples)
    monkeypatch.setattr(psola, "_trembling", lambda samples: (
        trembled.append(len(samples)) or 2 * samples))
    monkeypatch.setattr(psola, "_reverberated", lambda line: (
        reverberated.append(line.copy()) or np.array([line, -4 * line])))
    sung = psola.sing("la la, !", notes=(0, 7, 0), durs=(1, 1, 1),
                      effect=effect)
    assert trembled == [22050, 22050]
    (line,), notes = reverberated, 2 * 22050
    assert np.all(line[notes:] == 0) and np.any(line[:notes])
    assert sung.shape == (2, 3 * 22050)
    assert np.abs(sung).max() == pytest.approx(1)
    np.testing.assert_allclose(sung[1], -4 * sung[0])


def test_flite_and_the_plain_voice_neither_tremble_nor_reverberate(
        stand_ins, monkeypatch):
    def refuse(*args):
        raise AssertionError("not for this effect")

    monkeypatch.setattr(psola, "_trembling", refuse)
    monkeypatch.setattr(psola, "_reverberated", refuse)
    monkeypatch.setattr(psola, "_require_flite_voice", lambda lang: None)
    monkeypatch.setattr(psola.shutil, "which", lambda name: "/bin/flite")
    assert psola.sing("la", (0,), (1,)).ndim == 1
    assert psola.sing("la", (0,), (1,), lang="rms", effect="flite").ndim == 1


def test_the_tremolo_is_sox_s_nine_hertz_falling_to_half():
    """eCantorix's sox ``tremolo 9 50``: the level falls to half and back,
    nine times a second, from the start of each note."""
    from music.core.synths.envelopes import tremolo

    steady = np.ones(psola.RATE)
    trembled = psola._trembling(steady)
    np.testing.assert_array_equal(trembled, tremolo(
        sonic_vector=steady, tremolo_freq=9, max_db_dev=10 * np.log10(2),
        sample_rate=psola.RATE))
    assert trembled.max() / trembled.min() == pytest.approx(2, rel=1e-3)
    assert trembled[0] == 1
    # A second of it: the spectrum's bins are a hertz apart.
    assert np.argmax(np.abs(np.fft.rfft(trembled - trembled.mean()))) == 9


def test_resampling_moves_pitch_and_formants_as_tape(parselmouth):
    resampled = types.SimpleNamespace(values=np.array([[0.5, 0.25]]))
    parselmouth.answers[:] = [resampled]
    out = psola._resampled(np.array([1.0, 2.0, 3.0]), 0.75)
    np.testing.assert_array_equal(out, [0.5, 0.25])
    (sound, command, rate, precision), = parselmouth.calls
    np.testing.assert_array_equal(sound.values[0], [1.0, 2.0, 3.0])
    assert (sound.sampling_frequency, command, rate, precision) == (
        33075.0, "Resample", 44100, 50)


def test_the_room_rings_a_second_in_stereo_ten_decibels_down():
    """sox's reverb at its defaults, which eCantorix runs, has its tail
    10 dB below the direct sound, and a tail of its own in each channel.
    The package's early reflections are all positive, so its channels are
    less independent than sox's: 0.43 correlated, against 0.09."""
    impulse = np.zeros(1000)
    impulse[0] = 1.0
    room = psola._reverberated(impulse)
    assert room.shape == (2, 1000 + 44100)
    np.testing.assert_allclose(room[:, 0], [1.0, 1.0])
    wet = np.sum(room[:, 1:] ** 2, axis=1)
    np.testing.assert_allclose(10 * np.log10(wet), [-10, -10], atol=1e-6)
    assert not np.allclose(room[0], room[1])


def test_the_room_is_the_same_every_time_and_leaves_chance_alone():
    np.random.seed(7)
    before = np.random.get_state()[1].copy()
    first = psola._reverberated(np.ones(10))
    after = np.random.get_state()[1].copy()
    np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(first, psola._reverberated(np.ones(10)))


def test_each_channel_draws_a_response_of_the_measured_room(monkeypatch):
    """A second long, decaying 40 dB over it, as sox's rings for 0.87 s."""
    drawn = []
    module = sys.modules["music.core.filters.reverb"]

    original = module.reverb

    def counted(**kwargs):
        drawn.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(module, "reverb", counted)
    psola._reverberated(np.ones(4))
    assert drawn == [dict(duration=1.0, decay=-40.0, sample_rate=44100)] * 2


@pytest.mark.parametrize("signal, response, length", [
    ([1.0, 2.0, 3.0], [1.0, 0.5], 4), ([1.0, -1.0], [0.25, 0.5, 2.0], 6),
    ([0.5], [1.0], 1), ([1.0, 2.0], [3.0], 2),
    # Cut shorter than the convolution, five long, which a transform of
    # four, or of two, would fold back onto its start.
    ([1.0, 2.0, 3.0], [1.0, -1.0, 0.5], 2)])
def test_the_convolution_is_numpy_s_padded_or_cut(signal, response, length):
    full = np.convolve(signal, response)
    expected = np.zeros(length)
    expected[:min(length, len(full))] = full[:length]
    np.testing.assert_allclose(
        psola._convolved(np.array(signal), np.array(response), length),
        expected, atol=1e-12)
