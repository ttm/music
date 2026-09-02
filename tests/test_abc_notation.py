"""Turning notes and durations into ABC notation.

The singing engine is a Perl program that cannot run in CI, but everything
this package does *before* handing over to it is pure string generation.
"""

import numpy as np
import pytest

import music.singing.paths as paths
import music.singing.perform as perform


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """Point the engine at a writable scratch directory."""
    engine = tmp_path / "engine"
    (engine / "cache").mkdir(parents=True)
    monkeypatch.setenv(paths.ENV_VAR, str(engine))
    return engine / "cache"


# --------------------------------------------------------------------------
# Pitch names
# --------------------------------------------------------------------------

def test_semitones_become_abc_pitch_names():
    """Relative to a reference of 60, zero is middle c."""
    assert perform.converter.convert((0, 4, 7, 12), 60) == \
        ["=c", "e", "=g", "=c'"]


def test_octaves_change_case_and_add_marks():
    """ABC writes the octave below in upper case and the one above with a
    prime."""
    assert perform.converter.convert((-12, 0, 12), 60) == ["=C", "=c", "=c'"]


def test_the_dictionary_is_rebuilt_if_it_goes_missing():
    notes = perform.Notes()
    notes.notes_dict = None
    assert notes.convert((0,), 60) == ["=c"]


# --------------------------------------------------------------------------
# Durations
# --------------------------------------------------------------------------

def test_a_duration_of_one_is_left_implicit():
    """ABC takes the unit length as the default, so 1 is written as
    nothing."""
    assert perform.translate_to_abc((0,), (1,), 60) == "=c"


def test_other_durations_follow_their_note():
    assert perform.translate_to_abc((0, 4), (1, 2), 60) == "=ce2"


def test_a_negative_duration_becomes_a_division():
    """`-2` is written `/2`, ABC's notation for a half-length note."""
    assert perform.translate_to_abc((0,), (-2,), 60) == "=c/2"


# --------------------------------------------------------------------------
# The .abc file
# --------------------------------------------------------------------------

def test_write_abc_emits_a_well_formed_header(cache):
    perform.write_abc("la la", (0, 4), (1, 1), M="3/4", L="1/8", Q=90, K="G")

    written = (cache / "achant.abc").read_text()
    assert written.startswith("X:1\n")
    for field in ("M:3/4", "L:1/8", "Q:90", "V:1", "K:G"):
        assert field in written


def test_write_abc_carries_the_lyric_line(cache):
    perform.write_abc("hey ma bro", (0, 4, 7), (1, 1, 1))
    written = (cache / "achant.abc").read_text()
    assert written.rstrip().endswith("w: hey ma bro")
    assert "=ce=g" in written


# --------------------------------------------------------------------------
# sing()'s own branches
# --------------------------------------------------------------------------

@pytest.mark.parametrize("effect, expected", [
    ("flint", "flite.inc"),
    ("tremolo", "tremolo.inc"),
    ("melt", "melt.inc"),
])
def test_each_effect_selects_its_include(cache, effect, expected,
                                         monkeypatch):
    """The effect names an extra voice for the engine to `do`."""
    engine = cache.parent
    (engine / "Makefile").write_text("all:\n\ttrue\n")
    monkeypatch.setattr(paths, "missing_requirements", lambda: [])
    monkeypatch.setattr(perform.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(perform.sf, "read",
                        lambda path, dtype=None: (np.zeros(4), 44100))

    perform.sing(effect=effect)

    assert expected in (cache / "achant.conf").read_text()


def test_the_language_and_transposition_reach_the_conf(cache, monkeypatch):
    engine = cache.parent
    (engine / "Makefile").write_text("all:\n\ttrue\n")
    monkeypatch.setattr(paths, "missing_requirements", lambda: [])
    monkeypatch.setattr(perform.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(perform.sf, "read",
                        lambda path, dtype=None: (np.zeros(4), 44100))

    perform.sing(lang="pt", transpose=-24)

    conf = (cache / "achant.conf").read_text()
    assert '$ESPEAK_VOICE = "pt";' in conf
    assert "$ESPEAK_TRANSPOSE = -24;" in conf


def test_sing_returns_normalized_samples(cache, monkeypatch):
    engine = cache.parent
    (engine / "Makefile").write_text("all:\n\ttrue\n")
    monkeypatch.setattr(paths, "missing_requirements", lambda: [])
    monkeypatch.setattr(perform.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(perform.sf, "read",
                        lambda path, dtype=None: (np.array([0.0, 1.0, 2.0]),
                                                  44100))

    out = perform.sing()

    assert np.allclose(out, [-1.0, 0.0, 1.0])


def test_sing_rejects_a_render_at_the_wrong_sample_rate(cache, monkeypatch):
    """The engine is expected to produce 44100 Hz; anything else would be
    silently the wrong pitch."""
    engine = cache.parent
    (engine / "Makefile").write_text("all:\n\ttrue\n")
    monkeypatch.setattr(paths, "missing_requirements", lambda: [])
    monkeypatch.setattr(perform.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(perform.sf, "read",
                        lambda path, dtype=None: (np.zeros(4), 22050))

    with pytest.raises(RuntimeError, match="44100"):
        perform.sing()


def test_translate_to_abc_rejects_a_length_mismatch():
    """Regression: notes and durations were zipped, so the tail of
    whichever was longer vanished. Five notes with three durations
    produced a three-note score, and since write_abc appends the lyric
    line separately, the words then pointed at notes that were gone."""
    from music.singing.perform import translate_to_abc

    assert translate_to_abc([0, 2, 4], [1, 1, 1], reference=60) == "=c=de"

    with pytest.raises(ValueError, match="5 notes and 3 durations"):
        translate_to_abc([0, 2, 4, 5, 7], [1, 1, 1], reference=60)

    with pytest.raises(ValueError, match="2 notes and 5 durations"):
        translate_to_abc([0, 2], [1, 1, 1, 1, 1], reference=60)


def test_the_note_dictionary_covers_the_midi_range_it_claims():
    """The eight octaves of names are longer than the MIDI range the
    dictionary maps, and the surplus is sliced off deliberately."""
    from music.singing.perform import Notes

    notes = Notes()
    assert len(notes.notes_dict) == 85
    assert min(notes.notes_dict) == 12
    assert max(notes.notes_dict) == 96
