"""What the `singing` mutation audit found untested or wrong.

The engine is Perl driving espeak, and its Makefile turns the score into
MIDI with abc2midi before eCantorix sings it at
``440 * 2 ** ((note - 69 + ESPEAK_TRANSPOSE) / 12)``. Everything before
that hand-over is checked here without it.
"""

import re
import shutil
import subprocess
import types

import numpy as np
import pytest

import music.singing.bootstrap as bootstrap
import music.singing.paths as paths
import music.singing.perform as perform
from _singing_stub import fake_run


@pytest.fixture
def engine(tmp_path, monkeypatch):
    """A stand-in engine: a Makefile, a cache, and every program found."""
    root = tmp_path / "engine"
    (root / "cache").mkdir(parents=True)
    (root / "Makefile").write_text("all:\n\ttrue\n")
    monkeypatch.setenv(paths.ENV_VAR, str(root))
    monkeypatch.setattr(paths, "missing_requirements", lambda: [])
    monkeypatch.setattr(paths, "missing_perl_modules", lambda: [])
    monkeypatch.setattr(perform.subprocess, "run",
                        fake_run(root / "cache"))
    monkeypatch.setattr(perform.sf, "read",
                        lambda path, dtype=None: (np.array([0.0, 1.0]),
                                                  44100))
    return root


# --------------------------------------------------------------------------
# The octave: MIDI 60 is ABC's C
# --------------------------------------------------------------------------

@pytest.mark.parametrize("midi, name", [
    (12, "=C,,,,"), (24, "=C,,,"), (36, "=C,,"), (48, "=C,"), (60, "=C"),
    (69, "=A"), (72, "=c"), (84, "=c'"), (96, "=c''"), (61, "^C"),
    (71, "=B"), (83, "=b"), (64, "=E"), (76, "=e"),
])
def test_each_midi_note_has_its_abc_name(midi, name):
    """abc2midi reads C as middle C, MIDI 60. This named 60 c, which it
    reads as 72, so every score was an octave above its reference."""
    assert perform.converter.notes_dict[midi] == name


def test_the_default_sings_where_it_always_did(engine):
    """The engine never read the conf and sang at its own -24, a score an
    octave above its reference: reference + note - 12, which the default
    transposition now asks for."""
    perform.sing(backend="ecantorix")
    conf = (engine / "cache" / "achant.conf").read_text()
    assert "$ESPEAK_TRANSPOSE = -12;" in conf
    score = (engine / "cache" / "achant.abc").read_text()
    assert "\n=E=D=C=D=E=E=E2\nw: " in score


# --------------------------------------------------------------------------
# Durations, as ABC writes a length
# --------------------------------------------------------------------------

@pytest.mark.parametrize("duration, written", [
    (1, ""), (1.0, ""), (2, "2"), (0.5, "/2"), (0.25, "/4"), (1.5, "3/2"),
    (0.75, "3/4"), (-2, "/2"), (-3, "/3"), (np.float64(0.5), "/2"),
    ("3/2", "3/2"), ("1-2", "1/2"), ("1", ""),
])
def test_a_duration_is_written_as_abc_writes_a_length(duration, written):
    """0.5 went into the score as "0.5", which is not ABC."""
    assert perform.translate_to_abc((0,), (duration,), 60) == "=C" + written


def test_a_note_of_no_length_is_refused():
    with pytest.raises(ValueError, match="^a note cannot last no time; "
                       "got a duration of 0$"):
        perform.translate_to_abc((0,), (0,), 60)


def test_the_test_song_is_a_score_abc_can_read(engine, monkeypatch):
    """Its halves and quarters were written as 0.5 and 0.25."""
    bootstrap.make_test_song()
    body = (engine / "cache" / "achant.abc").read_text().split("\n")[-2]
    assert body == "=G/2=C/2=F/4=G/4=B=c/4=G/2"
    assert "." not in body


# --------------------------------------------------------------------------
# Effects, requirements and the engine directory
# --------------------------------------------------------------------------

def test_a_wrong_effect_is_refused_before_anything_else(tmp_path,
                                                        monkeypatch):
    """Even with no engine installed, and without touching the cache."""
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path / "nowhere"))
    with pytest.raises(ValueError, match=re.escape(
            "effect not understood: 'reverse-cathedral'; expected one of "
            "['flint', 'flite', 'melt', 'tremolo'], or None for the plain "
            "voice")):
        perform.sing(effect="reverse-cathedral", backend="ecantorix")
    assert not (tmp_path / "nowhere").exists()


@pytest.mark.parametrize("effect", [None, "", False])
def test_no_effect_sings_with_the_plain_voice(engine, effect):
    perform.sing(effect=effect, backend="ecantorix")
    conf = (engine / "cache" / "achant.conf").read_text()
    assert "extravoices" not in conf


def test_abc2midi_is_a_requirement_named_by_its_package(monkeypatch):
    """The Makefile runs it, and the check did not ask for it."""
    assert "abc2midi" in paths.SYSTEM_REQUIREMENTS
    monkeypatch.setattr(paths, "missing_requirements",
                        lambda: ["espeak", "abc2midi"])
    with pytest.raises(RuntimeError, match=re.escape(
            "the eCantorix engine needs these programs, which are not "
            "installed: espeak, abc2midi. On Debian or Ubuntu: sudo apt "
            "install espeak abcmidi. On macOS with Homebrew: brew install "
            "espeak abcmidi. sing's default backend, psola, needs none of "
            "these: only espeak-ng and pip install 'music[singing]'.")):
        paths.require_system_dependencies()


def test_missing_requirements_asks_for_each_program(monkeypatch):
    asked = []
    monkeypatch.setattr(paths.shutil, "which",
                        lambda name: asked.append(name) or
                        (None if name == "perl" else f"/bin/{name}"))
    assert paths.missing_requirements() == ["perl"]
    assert asked == list(paths.SYSTEM_REQUIREMENTS)


def test_the_engine_is_not_cloned_over_something_else(tmp_path,
                                                      monkeypatch):
    """git refused, and the error was its exit status."""
    target = tmp_path / "ecantorix"
    target.mkdir()
    (target / "notes.txt").write_text("mine")
    monkeypatch.setenv(paths.ENV_VAR, str(target))
    cloned = []
    monkeypatch.setattr(bootstrap.subprocess, "run",
                        lambda *a, **k: cloned.append(a))
    with pytest.raises(RuntimeError, match=re.escape(
            f"{target} is not empty but holds no eCantorix engine; move it "
            "aside, or set $MUSIC_ECANTORIX_DIR to install the engine "
            "somewhere else")):
        bootstrap.setup_engine()
    assert cloned == [] and (target / "notes.txt").read_text() == "mine"


def test_an_empty_directory_is_cloned_into(tmp_path, monkeypatch):
    target = tmp_path / "ecantorix"
    target.mkdir()
    monkeypatch.setenv(paths.ENV_VAR, str(target))
    monkeypatch.setattr(paths, "missing_requirements", lambda: [])
    monkeypatch.setattr(paths, "missing_perl_modules", lambda: [])
    cloned = []
    monkeypatch.setattr(bootstrap.subprocess, "run",
                        lambda *a, **k: cloned.append((a, k)))
    assert bootstrap.setup_engine() == str(target)
    assert cloned == [((["git", "clone", "--branch", "music-2",
                         "https://github.com/ttm/ecantorix", str(target)],),
                       {"check": True})]


# --------------------------------------------------------------------------
# The table, built fresh: `converter` is built when the module is imported
# --------------------------------------------------------------------------

def test_a_fresh_table_names_every_octave_with_its_marks():
    table = perform.Notes().notes_dict
    assert [table[midi] for midi in range(12, 97, 12)] == [
        "=C,,,,", "=C,,,", "=C,,", "=C,", "=C", "=c", "=c'", "=c''"]
    assert [table[midi] for midi in range(60, 72)] == [
        "=C", "^C", "=D", "^D", "=E", "=F", "^F", "=G", "^G", "=A", "^A",
        "=B"]
    assert len(table) == 85


# --------------------------------------------------------------------------
# What sing() hands the engine, recorded
# --------------------------------------------------------------------------

@pytest.fixture
def recorded(engine, monkeypatch):
    calls = {"run": [], "read": [], "copy": []}
    monkeypatch.setattr(perform.subprocess, "run",
                        fake_run(engine / "cache", calls["run"]))
    monkeypatch.setattr(perform.sf, "read", lambda path, **k: (
        calls["read"].append((path, k)) or (np.array([0.0, 1.0]), 44100)))
    original = perform.shutil.copy
    monkeypatch.setattr(perform.shutil, "copy", lambda source, target: (
        calls["copy"].append((source, target)) or original(source,
                                                           target)))
    return calls


def test_sing_writes_the_score_and_conf_the_engine_reads(engine, recorded):
    import os

    perform.sing(effect="melt", backend="ecantorix")
    cache = engine / "cache"
    assert sorted(os.listdir(cache)) == ["Makefile", "achant.abc",
                                         "achant.conf", "achant.wav"]
    assert (cache / "achant.conf").read_text() == (
        '$ESPEAK_VOICE = "en";\n$ESPEAK_TRANSPOSE = -12;\n'
        "do 'extravoices/melt.inc';")
    assert (cache / "achant.abc").read_text() == (
        "X:1\nT:Some chanting for music python package\nM:4/4\nL:1/4\n"
        "Q:120\nV:1\nK:C\n=E=D=C=D=E=E=E2\nw: Mar-ry had a litt-le lamb")
    assert recorded["copy"] == [(engine / "Makefile", cache / "Makefile")]
    assert recorded["run"] == [
        (["espeak", "-q", "--ipa", "-v", "en", syllable],
         {"capture_output": True, "text": True})
        for syllable in ["Mar", "ry", "had", "a", "litt", "le", "lamb"]] + [
        (["espeak", "--version"], {"capture_output": True, "text": True}),
        (["make", "-C", str(cache), "ECANTORIX=perl -I. ../ecantorix.pl"],
         {"check": True, "capture_output": True, "text": True})]
    assert recorded["read"] == [(str(cache / "achant.wav"),
                                 {"dtype": "float64"})]


def test_the_engine_sings_a_syllable_without_a_vowel_with_a_schwa(
        engine, recorded, monkeypatch):
    """espeak says French "ques" as a bare /k/, which left its note with
    nothing to sing."""
    asked = []
    monkeypatch.setattr(perform, "sung_phonemes", lambda *args: (
        asked.append(args) or {"ques,": "[[k@]]"}))
    perform.sing(text="Jac-ques, Jac", notes=(4, 0, 4), durs=(1, 1, 1),
                 lang="fr", effect="tremolo", backend="ecantorix")
    assert asked == [("espeak", "fr", ["Jac", "ques,", "Jac"])]
    assert (engine / "cache" / "achant.conf").read_text() == (
        '$ESPEAK_VOICE = "fr";\n$ESPEAK_TRANSPOSE = -12;\n'
        "do 'extravoices/tremolo.inc';\n"
        + perform._edit_syllables({"ques,": "[[k@]]"}))


def test_a_lyric_with_a_vowel_in_every_syllable_edits_none(engine, recorded,
                                                         monkeypatch):
    monkeypatch.setattr(perform, "sung_phonemes", lambda *args: {})
    perform.sing(backend="ecantorix")
    assert "EDIT_SYLLABLES" not in (engine / "cache" /
                                    "achant.conf").read_text()


def test_with_flite_no_syllable_is_looked_up(engine, recorded, monkeypatch):
    """flite reads no espeak phonemes, and lang names a flite voice."""
    monkeypatch.setattr(perform, "_require_flite", lambda lang: None)
    asked = []
    monkeypatch.setattr(perform, "sung_phonemes", lambda *args: (
        asked.append(args) or {"ques": "[[k@]]"}))
    perform.sing(text="Jac-ques", notes=(4, 0), durs=(1, 1), lang="rms",
                 effect="flite", backend="ecantorix")
    assert asked == []
    assert "EDIT_SYLLABLES" not in (engine / "cache" /
                                    "achant.conf").read_text()


def test_sing_passes_its_score_settings_through(engine, recorded):
    perform.sing(text="la la", notes=(0, 7), durs=(1, 1), M="3/4",
                 L="1/8", Q=90, K="G", reference=48, backend="ecantorix")
    score = (engine / "cache" / "achant.abc").read_text()
    assert "\nM:3/4\nL:1/8\nQ:90\nV:1\nK:G\n=C,=G,\nw: la la" in score


def test_the_test_song_sings_its_own_words(engine, recorded):
    bootstrap.make_test_song()
    score = (engine / "cache" / "achant.abc").read_text()
    assert score.endswith("\nw: hey ma bro, why fly while dive?")


@pytest.mark.parametrize("prepare, detail", [
    (lambda root: None, "nothing is there"),
    (lambda root: root.mkdir(), "the directory exists but has no Makefile"),
])
def test_a_missing_engine_says_what_is_there(tmp_path, monkeypatch,
                                             prepare, detail):
    root = tmp_path / "engine"
    prepare(root)
    monkeypatch.setenv(paths.ENV_VAR, str(root))
    with pytest.raises(RuntimeError, match=re.escape(
            f"no usable eCantorix engine at {root}: {detail}. Run "
            "music.singing.setup_engine() to install it.")):
        perform.sing(backend="ecantorix")
    with pytest.raises(RuntimeError, match=re.escape(
            f"no usable eCantorix engine at {root}. Run 'setup_engine()' "
            "to install it.")):
        bootstrap.get_engine()


def test_the_engine_is_cloned_into_a_directory_that_does_not_exist_yet(
        tmp_path, monkeypatch):
    target = tmp_path / "a" / "b" / "ecantorix"
    monkeypatch.setenv(paths.ENV_VAR, str(target))
    monkeypatch.setattr(paths, "missing_requirements", lambda: [])
    monkeypatch.setattr(paths, "missing_perl_modules", lambda: [])
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda *a, **k: None)
    assert bootstrap.setup_engine() == str(target)
    assert target.parent.is_dir()


def test_a_bare_score_is_common_time_in_c_at_120(engine):
    perform.write_abc("la", (0,), (1,))
    assert (engine / "cache" / "achant.abc").read_text() == (
        "X:1\nT:Some chanting for music python package\nM:4/4\nL:1/4\n"
        "Q:120\nV:1\nK:C\n=C\nw: la")


# --------------------------------------------------------------------------
# What the engine needs, and what it answers with
# --------------------------------------------------------------------------

def test_a_render_that_writes_nothing_says_what_the_engine_said(
        engine, monkeypatch):
    """The Makefile pipes the script through tee, so make succeeded
    when the script failed, and the missing file was read as libsndfile's
    "System error". A render left from an earlier call was read back as
    this one's."""
    import types

    stale = engine / "cache" / "achant.wav"
    stale.write_bytes(b"an earlier song")

    def silent_make(command, *args, **kwargs):
        return types.SimpleNamespace(
            returncode=0, stdout="abc2midi achant.abc",
            stderr="Can't locate MIDI.pm in @INC")

    monkeypatch.setattr(perform.subprocess, "run", silent_make)
    with pytest.raises(RuntimeError, match=re.escape(
            "the singing engine ran but wrote no achant.wav; what it "
            "said:\nabc2midi achant.abc\nCan't locate MIDI.pm in @INC")):
        perform.sing(backend="ecantorix")
    assert not stale.exists()


def test_a_failed_make_says_what_it_printed(engine, monkeypatch):
    import subprocess

    failure = subprocess.CalledProcessError(2, ["make"], output="made",
                                            stderr="broke")
    monkeypatch.setattr(perform.subprocess, "run",
                        fake_run(engine / "cache", make_fails=failure))
    with pytest.raises(RuntimeError, match=re.escape(
            "Failed to build singing cache: Command '['make']' returned "
            "non-zero exit status 2.\nmade\nbroke")):
        perform.sing(backend="ecantorix")


def test_a_stereo_render_comes_back_as_two_channels(engine, monkeypatch):
    """The tremolo and melt effects render in stereo, which the file holds
    as (frames, 2) and which came back so, normalized as one channel."""
    frames = np.array([[0.0, 2.0], [1.0, -2.0], [0.5, 0.0]])
    monkeypatch.setattr(perform.sf, "read",
                        lambda path, dtype=None: (frames, 44100))
    sung = perform.sing(effect="tremolo", backend="ecantorix")
    np.testing.assert_array_equal(sung, perform.normalize_stereo(frames.T))
    assert sung.shape == (2, 3)


def test_the_perl_modules_are_asked_of_the_perl_on_path(monkeypatch):
    import types

    asked = []

    def run(command, capture_output):
        asked.append((command, capture_output))
        missing = command[1] in ("-MMIDI", "-MMath::FFT")
        return types.SimpleNamespace(returncode=int(missing))

    found = []
    monkeypatch.setattr(paths.shutil, "which",
                        lambda name: found.append(name) or f"/bin/{name}")
    monkeypatch.setattr(paths.subprocess, "run", run)
    assert paths.missing_perl_modules() == ["MIDI", "Math::FFT"]
    assert found == ["perl"]
    assert asked == [(["perl", f"-M{module}", "-e1"], True)
                     for module in paths.PERL_MODULES]


def test_without_perl_the_modules_are_not_asked_about(monkeypatch):
    monkeypatch.setattr(paths.shutil, "which", lambda name: None)
    assert paths.missing_perl_modules() == []


def test_missing_modules_are_named_with_how_to_install_them(monkeypatch):
    monkeypatch.setattr(paths, "missing_requirements", lambda: [])
    monkeypatch.setattr(paths, "missing_perl_modules",
                        lambda: ["MIDI", "Math::FFT"])
    with pytest.raises(RuntimeError, match=re.escape(
            "the eCantorix engine's Perl script needs these modules, which "
            "the perl on PATH cannot load: MIDI, Math::FFT. Install them "
            "with: cpan MIDI Math::FFT. sing's default backend, psola, "
            "needs none of these: only espeak-ng and pip install "
            "'music[singing]'.")):
        paths.require_system_dependencies()


def test_sox_and_the_modules_are_requirements():
    assert "sox" in paths.SYSTEM_REQUIREMENTS
    assert paths.PERL_MODULES == ("MIDI", "Math::FFT", "URI::Escape",
                                  "Digest::SHA")


def test_espeak_s_data_is_where_its_version_line_says(monkeypatch):
    import types

    monkeypatch.setattr(perform.subprocess, "run", lambda command, **k: (
        types.SimpleNamespace(stdout="eSpeak text-to-speech: 1.48.03  "
                              "Data at: /opt/espeak/espeak-data\n")))
    assert perform._espeak_data() == perform.Path("/opt/espeak/espeak-data")
    monkeypatch.setattr(perform.subprocess, "run", lambda command, **k: (
        types.SimpleNamespace(stdout="")))
    assert perform._espeak_data() is None


def test_the_cache_gets_the_effects_and_espeak_s_data(tmp_path,
                                                      monkeypatch):
    engine, cache = tmp_path / "engine", tmp_path / "engine" / "cache"
    (engine / "examples" / "extravoices").mkdir(parents=True)
    cache.mkdir()
    (engine / "Makefile").write_text("all:\n")
    (engine / "examples" / "extravoices" / "melt.inc").write_text("melt")
    data = tmp_path / "espeak-data"
    (data / "voices").mkdir(parents=True)
    (data / "phontab").write_text("phonemes")
    monkeypatch.setattr(perform, "_espeak_data", lambda: data)

    perform._prepare_cache(engine, cache)

    assert (cache / "Makefile").read_text() == "all:\n"
    assert (cache / "extravoices" / "melt.inc").read_text() == "melt"
    assert (cache / "espeak-data" / "phontab").read_text() == "phonemes"
    assert (cache / "espeak-data" / "voices" / "!v" / "melt.inc").is_file()


def test_espeak_s_data_is_copied_once(tmp_path, monkeypatch):
    engine, cache = tmp_path / "engine", tmp_path / "engine" / "cache"
    cache.mkdir(parents=True)
    (engine / "Makefile").write_text("all:\n")
    (cache / "espeak-data").mkdir()
    (cache / "espeak-data" / "phontab").write_text("ours")
    copied = []
    monkeypatch.setattr(perform, "_espeak_data", lambda: tmp_path)
    original = perform.shutil.copytree
    monkeypatch.setattr(perform.shutil, "copytree", lambda *a, **k: (
        copied.append(a) or original(*a, **k)))
    perform._prepare_cache(engine, cache)
    assert copied == []
    assert (cache / "espeak-data" / "phontab").read_text() == "ours"


def test_only_the_last_twenty_lines_of_what_was_said_are_kept():
    printed = "\n".join(f"line {i}" for i in range(30))
    assert perform._tail(printed, "last") == "\n".join(
        [f"line {i}" for i in range(11, 30)] + ["last"])


def _engine_with_voices(tmp_path):
    engine, cache = tmp_path / "engine", tmp_path / "engine" / "cache"
    (engine / "examples" / "extravoices").mkdir(parents=True)
    cache.mkdir()
    (engine / "Makefile").write_text("all:\n")
    (engine / "examples" / "extravoices" / "melt.inc").write_text("melt")
    data = tmp_path / "espeak-data"
    (data / "voices").mkdir(parents=True)
    (data / "phontab").write_text("phonemes")
    return engine, cache, data


def test_the_cache_is_prepared_by_these_exact_paths(tmp_path, monkeypatch):
    """Recorded, not found: on a case-insensitive filesystem EXTRAVOICES
    finds extravoices."""
    import os
    engine, cache, data = _engine_with_voices(tmp_path)
    monkeypatch.setattr(perform, "_espeak_data", lambda: data)
    copied = []
    original = perform.shutil.copytree

    def copytree(*args, **kwargs):
        # copytree calls itself for each subdirectory, with more arguments.
        if len(args) == 2:
            copied.append((*args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(perform.shutil, "copytree", copytree)
    checked = []
    original_is_file = perform.Path.is_file
    monkeypatch.setattr(perform.Path, "is_file", lambda self: (
        checked.append(self) or original_is_file(self)))

    perform._prepare_cache(engine, cache)

    voices = engine / "examples" / "extravoices"
    assert copied == [
        (voices, cache / "extravoices", {"dirs_exist_ok": True}),
        (data, cache / "espeak-data", {"dirs_exist_ok": True}),
        (voices, cache / "espeak-data" / "voices" / "!v",
         {"dirs_exist_ok": True})]
    assert checked == [cache / "espeak-data" / "phontab"]
    assert sorted(os.listdir(cache)) == ["Makefile", "espeak-data",
                                         "extravoices"]
    assert os.listdir(cache / "espeak-data" / "voices") == ["!v"]


def test_the_cache_can_be_prepared_again(tmp_path, monkeypatch):
    """Over what an earlier call copied, and over an empty espeak-data,
    which is what the engine's Makefile leaves when it cannot copy."""
    engine, cache, data = _engine_with_voices(tmp_path)
    (cache / "espeak-data").mkdir()
    monkeypatch.setattr(perform, "_espeak_data", lambda: data)
    perform._prepare_cache(engine, cache)
    perform._prepare_cache(engine, cache)
    assert (cache / "espeak-data" / "phontab").read_text() == "phonemes"


@pytest.mark.parametrize("lang", [
    'en"; system("touch pwned"); "', "en\n", "", "en us", "../x", 3])
def test_a_voice_that_is_not_a_voice_name_is_refused(tmp_path, monkeypatch,
                                                     lang):
    """The configuration is Perl the engine runs; since it reads it, a
    lang with a quote in it would have been run as code."""
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path / "nowhere"))
    with pytest.raises(ValueError, match=re.escape(
            "lang must be an espeak voice name, such as 'en' or 'pt-br'; "
            f"got {lang!r}")):
        perform.sing(lang=lang, backend="ecantorix")


@pytest.mark.parametrize("lang", ["en", "pt-br", "en-us", "en+m3", "de"])
def test_voice_names_espeak_uses_are_accepted(engine, lang):
    perform.sing(lang=lang, backend="ecantorix")
    conf = (engine / "cache" / "achant.conf").read_text()
    assert conf.startswith(f'$ESPEAK_VOICE = "{lang}";\n')


@pytest.mark.parametrize("transpose", [
    float("nan"), float("inf"), "0; system('x')", None, True])
def test_a_transposition_that_is_not_a_number_is_refused(tmp_path,
                                                         monkeypatch,
                                                         transpose):
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path / "nowhere"))
    with pytest.raises(ValueError, match=re.escape(
            "transpose must be a finite number of semitones; got "
            f"{transpose!r}")):
        perform.sing(transpose=transpose, backend="ecantorix")


@pytest.mark.parametrize("transpose", [0, -12, 7.5, np.int64(-24)])
def test_any_finite_transposition_reaches_the_conf(engine, transpose):
    perform.sing(transpose=transpose, backend="ecantorix")
    conf = (engine / "cache" / "achant.conf").read_text()
    assert f"$ESPEAK_TRANSPOSE = {transpose};" in conf


# --------------------------------------------------------------------------
# The flite effect, which runs flite rather than espeak
# --------------------------------------------------------------------------

@pytest.mark.parametrize("missing, message", [
    (["flite"], "the flite effect runs flite, which is not installed. On "
     "Debian or Ubuntu: sudo apt install flite. On macOS with Homebrew: "
     "brew install flite."),
    (["flite", "bc"], "the flite effect runs flite and bc, which are not "
     "installed. On Debian or Ubuntu: sudo apt install flite bc. On macOS "
     "with Homebrew: brew install flite."),
])
def test_the_flite_effect_names_what_it_runs(engine, monkeypatch, missing,
                                             message):
    """It "rendered" before: the cache held espeak's renders of the same
    syllable, which flite's shared, so flite was never run."""
    monkeypatch.setattr(perform.shutil, "which", lambda name: (
        None if name in missing else f"/bin/{name}"))
    with pytest.raises(RuntimeError, match=re.escape(message)):
        perform.sing(effect="flite", lang="rms", backend="ecantorix")


def test_the_flite_effect_needs_one_of_flite_s_voices(engine, monkeypatch):
    import types

    monkeypatch.setattr(perform.shutil, "which", lambda name: f"/bin/{name}")
    asked = []

    def run(command, **kwargs):
        asked.append(command)
        if command[0] == "flite":
            return types.SimpleNamespace(
                stdout="Voices available: kal awb rms slt\n")
        return fake_run(engine / "cache")(command, **kwargs)

    monkeypatch.setattr(perform.subprocess, "run", run)
    with pytest.raises(ValueError, match=re.escape(
            "with the flite effect, lang names one of flite's voices, "
            "['kal', 'awb', 'rms', 'slt']; got 'en'")):
        perform.sing(effect="flite", backend="ecantorix")
    assert asked == [["flite", "-lv"]]
    perform.sing(effect="flint", lang="slt", backend="ecantorix")
    assert "flite.inc" in (engine / "cache" / "achant.conf").read_text()


def test_flite_s_voices_are_read_from_its_captured_output(engine,
                                                          monkeypatch):
    import types

    monkeypatch.setattr(perform.shutil, "which", lambda name: f"/bin/{name}")
    given = []

    def run(command, **kwargs):
        if command[0] == "flite":
            given.append(kwargs)
            return types.SimpleNamespace(stdout="Voices available: rms\n")
        return fake_run(engine / "cache")(command, **kwargs)

    monkeypatch.setattr(perform.subprocess, "run", run)
    perform.sing(effect="flite", lang="rms", backend="ecantorix")
    assert given == [{"capture_output": True, "text": True}]


@pytest.mark.parametrize("backend", ["ecantorix", "psola"])
@pytest.mark.parametrize("L, Q, message", [
    ("1/4", 0, "the tempo must be positive; got Q=0"),
    ("x", 120, "L must be a fraction such as \"1/4\"; got 'x'"),
])
def test_a_tempo_is_read_before_anything_is_sung(tmp_path, monkeypatch,
                                                 backend, L, Q, message):
    """eCantorix handed Q:0 to abc2midi, which called it malformed and
    stopped the build."""
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path / "nowhere"))
    with pytest.raises(ValueError, match=re.escape(message)):
        perform.sing(L=L, Q=Q, backend=backend)


# --------------------------------------------------------------------------
# A syllable said without a vowel, sung with a schwa
# --------------------------------------------------------------------------

def _transcriber(monkeypatch, answers, returncode=0):
    """espeak, as `answers` maps (notation, syllable) to what it prints."""
    ran = []

    def run(command, capture_output, text):
        ran.append(command)
        notation, syllable = command[2], command[-1]
        return types.SimpleNamespace(
            returncode=returncode,
            stdout=answers.get((notation, syllable), ""), stderr="")

    monkeypatch.setattr(perform.subprocess, "run", run)
    return ran


def test_a_syllable_without_a_vowel_gets_its_phonemes_and_a_schwa(
        monkeypatch):
    ran = _transcriber(monkeypatch, {
        ("--ipa", "Jac"): "ʒˈak\n", ("--ipa", "ques"): " k\n",
        ("-x", "ques"): " k\n", ("--ipa", "ble"): "bl\n",
        ("-x", "ble"): "b l\n"})
    sung = perform.sung_phonemes("/bin/espeak", "fr",
                                 ["Jac", "ques", "Jac", "ble", "ques"])
    assert sung == {"ques": "[[k@]]", "ble": "[[bl@]]"}
    assert ran == [
        ["/bin/espeak", "-q", "--ipa", "-v", "fr", "Jac"],
        ["/bin/espeak", "-q", "--ipa", "-v", "fr", "ques"],
        ["/bin/espeak", "-q", "-x", "-v", "fr", "ques"],
        ["/bin/espeak", "-q", "--ipa", "-v", "fr", "ble"],
        ["/bin/espeak", "-q", "-x", "-v", "fr", "ble"]]


@pytest.mark.parametrize("ipa", [
    "ʒˈak", "lˈɑː", "ʁˈə-", "kˈø", "fʁˈɛ", "tˈuː", "ɡˈɪv", "bˈʌt", "ɪt",
    "lˈʊk", "bˈɔːl", "mˈæn", "nˈʏ", "ʃˈœn", "vˈɐ", "hˈɜː", "ɡˈɒt",
    "lˈɪtᵻl", "fˈɚ", "ˈaɪ", "mˈe", "zˈi", "ʃˈy", "tˈo", "bˈu",
    "bˈʉ", "kˈɨ", "sˈɯ", "pˈɤ", "ɵ", "ɘ", "ɞ", "ɶ", "ɝ", "n\u0329"])
def test_a_syllable_with_a_vowel_or_a_syllabic_consonant_is_left(
        monkeypatch, ipa):
    _transcriber(monkeypatch, {("--ipa", "la"): ipa})
    assert perform.sung_phonemes("espeak", "en", ["la"]) == {}


@pytest.mark.parametrize("ipa, phonemes", [
    ("", "k"),          # said as nothing: a rest, as it was
    ("k", ""),          # no mnemonics to sing
])
def test_a_syllable_said_as_nothing_is_left(monkeypatch, ipa, phonemes):
    _transcriber(monkeypatch, {("--ipa", "-"): ipa, ("-x", "-"): phonemes})
    assert perform.sung_phonemes("espeak", "fr", ["-"]) == {}


def test_a_voice_espeak_lacks_edits_nothing(monkeypatch):
    ran = _transcriber(monkeypatch, {("--ipa", "ques"): "k",
                                     ("-x", "ques"): "k"}, returncode=1)
    assert perform.sung_phonemes("espeak", "xx", ["ques"]) == {}
    assert ran == [["espeak", "-q", "--ipa", "-v", "xx", "ques"]]


def test_an_espeak_that_cannot_be_run_edits_nothing(monkeypatch):
    """The singer's own check says what is missing; this only looks."""
    def run(command, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", command[0])

    monkeypatch.setattr(perform.subprocess, "run", run)
    assert perform._transcribe("espeak", "fr", "ques", "--ipa") is None
    assert perform.sung_phonemes("espeak", "fr", ["ques"]) == {}


def test_the_transcription_is_asked_for_as_text():
    asked = []

    def run(command, **kwargs):
        asked.append(kwargs)
        return types.SimpleNamespace(returncode=0, stdout="a", stderr="")

    original = perform.subprocess.run
    perform.subprocess.run = run
    try:
        assert perform._transcribe("espeak", "en", "a", "--ipa") == "a"
    finally:
        perform.subprocess.run = original
    assert asked == [{"capture_output": True, "text": True}]


@pytest.mark.parametrize("text, written", [
    ("ques", "'ques'"), ("qu'", "'qu\\''"), ("a\\b", "'a\\\\b'"),
    ("[[k@]]", "'[[k@]]'")])
def test_a_syllable_is_written_as_a_perl_string(text, written):
    assert perform._perl_string(text) == written


@pytest.mark.skipif(not shutil.which("perl"), reason="needs perl")
def test_the_engine_s_edit_sings_each_mapped_syllable_and_runs_nothing(
        tmp_path):
    """The configuration is run as Perl, and the syllables are the
    user's: each must be a string to Perl and never code."""
    sung = {"ques,": "[[k@]]", "qu'": "[[k@]]", "a\\b": "[[b@]]",
            "'; print 'ran'; '": "[[r@]]", "Frè": "[[f@]]",
            "$x @y": "[[d@]]"}
    script = tmp_path / "edit.pl"
    script.write_text(
        perform._edit_syllables(sung) + "\n"
        "binmode STDOUT;\n"
        "while (my $line = <STDIN>) { chomp $line; local $_ = $line; "
        "$EDIT_SYLLABLES->(); print \"$_\\n\"; }\n", encoding="utf-8")
    heard = ["\nFrè", " ques,", "qu'", " a\\b", "'; print 'ran'; '",
             "$x @y", " Jac", "ques"]
    result = subprocess.run(["perl", str(script)], capture_output=True,
                            input="\n".join(h.replace("\n", "")
                                            for h in heard).encode() + b"\n",
                            check=True)
    assert result.stdout.decode().splitlines() == [
        "Frè[[f@]]", " ques,[[k@]]", "qu'[[k@]]", " a\\b[[b@]]",
        "'; print 'ran'; '[[r@]]", "$x @y[[d@]]", " Jac", "ques"]
