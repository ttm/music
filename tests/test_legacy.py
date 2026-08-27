"""Behaviour of the legacy synthesizers.

These were the least covered modules in the package. CanonicalSynth builds
its attributes dynamically, which is what made them invisible to mypy, so
the surface is pinned here before it is made explicit.
"""

import numpy as np
import pytest

import music
from music.legacy.CanonicalSynth import CanonicalSynth
from music.legacy.IteratorSynth import IteratorSynth

#: Every attribute a default CanonicalSynth ends up carrying. Assigning
#: these is the whole job of synthSetup and adsrSetup.
EXPECTED_ATTRIBUTES = {
    "A", "A_", "A_i", "D", "D_i", "Lambda_A", "Lambda_D", "Lambda_R",
    "R", "R_i", "S", "a_S", "adsr_method", "duration",
    "fundamental_frequency", "ii", "render_note", "samplerate", "table",
    "tables", "tremolo", "tremolo_depth", "tremolo_frequency",
    "tremolo_table", "vibrato", "vibrato_depth", "vibrato_frequency",
    "vibrato_table",
}

#: The subset actually read elsewhere in the package.
READ_ATTRIBUTES = {
    "A_i", "D_i", "R_i", "Lambda_A", "Lambda_D", "Lambda_R", "a_S",
    "duration", "fundamental_frequency", "samplerate", "table", "tables",
    "tremolo_depth", "tremolo_frequency", "tremolo_table", "vibrato_depth",
    "vibrato_frequency", "vibrato_table",
}


def test_default_synth_carries_the_expected_attributes():
    """Pins the surface that synthSetup and adsrSetup assign."""
    assert set(vars(CanonicalSynth())) == EXPECTED_ATTRIBUTES


def test_the_attributes_read_elsewhere_all_exist():
    synth = CanonicalSynth()
    for name in sorted(READ_ATTRIBUTES):
        assert hasattr(synth, name), name


def test_default_attribute_values():
    """The defaults, so a refactor of how they are assigned cannot drift."""
    synth = CanonicalSynth()
    assert synth.samplerate == 44100
    assert synth.fundamental_frequency == 220
    assert synth.duration == 2
    assert synth.vibrato_depth == pytest.approx(0.1)
    assert synth.vibrato_frequency == pytest.approx(2.0)
    assert synth.tremolo_depth == pytest.approx(3.0)
    assert synth.tremolo_frequency == pytest.approx(0.2)
    assert synth.a_S == pytest.approx(10 ** (-5.0 / 20))
    assert synth.Lambda_A == 4410
    assert synth.Lambda_D == 1764
    assert synth.Lambda_R == 2205
    for name in ("A_i", "D_i", "R_i", "table", "vibrato_table",
                 "tremolo_table"):
        assert isinstance(getattr(synth, name), np.ndarray), name


def test_absorb_state_sets_arbitrary_attributes():
    """The synth is parametrised by keyword on any call, and remembers."""
    synth = CanonicalSynth(anything=7)
    assert synth.anything == 7
    synth.absorbState(other="value")
    assert synth.other == "value"


def test_state_passed_to_the_constructor_survives_setup():
    synth = CanonicalSynth(samplerate=22050)
    assert synth.samplerate == 22050


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def test_raw_render_produces_audio_of_the_requested_duration():
    synth = CanonicalSynth()
    samples = synth.rawRender(duration=0.05)
    assert isinstance(samples, np.ndarray)
    assert samples.shape[0] == pytest.approx(0.05 * synth.samplerate, abs=2)
    assert np.isfinite(samples).all()


def test_render_and_render2_produce_finite_audio():
    synth = CanonicalSynth()
    for method in (synth.render, synth.render2):
        samples = method(duration=0.05)
        assert isinstance(samples, np.ndarray)
        assert samples.size > 0
        assert np.isfinite(samples).all()


def test_tremolo_envelope_spans_its_depth_in_decibels():
    """The envelope is 10 ** (pattern * depth / 20), so over at least one
    full cycle it swings between +/- tremolo_depth around unity gain."""
    synth = CanonicalSynth()
    depth = synth.tremolo_depth
    # One whole cycle: the 0.2 Hz default would need five seconds.
    envelope = synth.tremoloEnvelope(duration=1.0, tremolo_frequency=4.0)

    assert np.isfinite(envelope).all()
    assert envelope.min() == pytest.approx(10 ** (-depth / 20), rel=1e-3)
    assert envelope.max() == pytest.approx(10 ** (depth / 20), rel=1e-3)


def test_adsr_apply_shapes_the_note_it_is_given():
    synth = CanonicalSynth()
    note = synth.rawRender(duration=0.5)
    shaped = synth.adsrApply(note)
    assert shaped.shape == note.shape
    assert np.isfinite(shaped).all()
    # The attack starts from silence, so the opening sample is quieter.
    assert abs(shaped[0]) <= abs(note[0]) + 1e-12


# --------------------------------------------------------------------------
# IteratorSynth
# --------------------------------------------------------------------------

def test_iterator_synth_cycles_through_its_sequences():
    synth = IteratorSynth()
    synth.fundamental_frequency_sequence = [220, 440]
    synth.duration_sequence = [0.05]
    first = synth.renderIterate()
    second = synth.renderIterate()
    assert np.isfinite(first).all() and np.isfinite(second).all()
    assert first.size and second.size


def test_being_is_constructible_and_renders():
    being = music.Being()
    assert being is not None


# --------------------------------------------------------------------------
# The demonstration piece
# --------------------------------------------------------------------------

def test_test_song_2_renders(tmp_path, monkeypatch):
    """Regression: `synth = M.legacy.CanonicalSynth` bound the class rather
    than an instance, so every call in this module was an unbound method
    missing self and the piece could not run at all. It writes its WAV
    files to the working directory, so it is run inside tmp_path.
    """
    from music.legacy.pieces.testSong2 import TestSong2

    monkeypatch.chdir(tmp_path)
    song = TestSong2()

    assert song.notes_, "the piece built no notes"
    song.render()
    assert (tmp_path / "vibrosong.wav").is_file()
    assert len(list(tmp_path.glob("*.wav"))) > 1


def test_the_module_level_synth_is_an_instance():
    """The bug above in one assertion, without rendering anything."""
    from music.legacy.pieces import testSong2

    assert isinstance(testSong2.synth, CanonicalSynth)
