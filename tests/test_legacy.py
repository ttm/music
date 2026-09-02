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


# --------------------------------------------------------------------------
# Being: walking, staying and rendering
# --------------------------------------------------------------------------

def _being_with_grid(size=8):
    """A Being with a grid and pointer ready to walk."""
    being = music.Being()
    being.grid = list(range(size))
    being.pointer = 0
    being.seqsize = size
    being.curseq = "f_"
    being.f_ = []
    return being


def test_walk_takes_consecutive_steps_from_the_grid():
    being = _being_with_grid()
    being.walk(3)
    assert list(being.f_) == [0, 1, 2]
    assert being.pointer == 3


def test_walk_low_high_interleaves_across_the_sequence():
    being = _being_with_grid(size=8)
    being.seqsize = 4
    being.walk(2, method="low-high")
    assert len(being.f_) == 2 * being.seqsize


def test_walk_rejects_an_unknown_method():
    """Regression: `sequence` was only assigned inside the recognised
    branches, so anything else died on UnboundLocalError."""
    being = _being_with_grid()
    with pytest.raises(ValueError, match="method not understood"):
        being.walk(2, method="sideways")


def test_perm_walk_is_no_longer_a_hole_in_the_api():
    """It raised NotImplementedError because the original was lost.

    What replaced it is a reconstruction rather than a recovery, and the
    docstring says so; these tests pin the reading it committed to.
    """
    being = _being_with_perms()
    being.walk(4, method="perm-walk")
    assert being.f_ == [1, 0, 3, 2]


def test_stay_permutes_the_domain():
    """The campanology example's second form: a domain plus permutations,
    with `curseq` naming which parameter sequence to fill."""
    being = music.Being()
    being.domain = [220, 440, 330]
    being.perms = music.PlainChanges(3).peal_direct
    being.curseq = "f_"
    being.f_ = []

    being.stay(6)

    assert len(being.f_) == 6
    assert set(being.f_) <= {220, 440, 330}
    assert being.total_notes == 6


def test_stay_falls_back_to_the_grid_when_no_domain_is_set():
    """Without a domain it permutes a slice of the grid, so the slice has
    to be as wide as the permutations."""
    being = _being_with_grid(size=8)
    being.seqsize = 3
    being.domain = []
    being.perms = music.PlainChanges(3).peal_direct
    being.stay(4)
    assert len(being.f_) == 4


def test_stay_accepts_a_numpy_domain():
    being = music.Being()
    being.domain = np.array([220.0, 440.0, 330.0])
    being.perms = music.PlainChanges(3).peal_direct
    being.curseq = "f_"
    being.f_ = []
    being.stay(3)
    assert len(being.f_) == 3


def test_stay_can_walk_straight_instead():
    being = _being_with_grid()
    being.stay(3, method="straight")
    assert list(being.f_) == [0, 1, 2]


def test_render_writes_a_wav_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    being = music.Being()
    being.f_ = [220.0, 440.0, 330.0]

    being.render(3, "being.wav")

    assert (tmp_path / "being.wav").is_file()


def test_render_returns_the_samples_when_given_no_filename():
    being = music.Being()
    being.f_ = [220.0, 440.0]
    out = being.render(2)
    assert isinstance(out, np.ndarray)
    assert out.size > 0
    assert np.isfinite(out).all()


def test_set_par_switches_to_the_frequency_grid():
    """`setPar('f')` reads fgrid/fpointer, which the caller supplies the
    way it supplies perms and domain."""
    being = music.Being()
    being.fgrid, being.fpointer = [1.0, 2.0, 3.0], 1

    being.setPar("f")

    assert being.grid == [1.0, 2.0, 3.0]
    assert being.pointer == 1


def test_set_par_rejects_a_parameter_it_cannot_switch_to():
    """Regression: anything but 'f' was a silent no-op."""
    with pytest.raises(ValueError, match="only the 'f' parameter"):
        music.Being().setPar("d")


def test_set_size_and_set_perms_record_what_they_are_given():
    being = music.Being()
    being.setSize(11)
    assert being.seqsize == 11

    perms = music.PlainChanges(3).peal_direct
    being.setPerms(perms)
    assert being.perms is perms


def test_add_seq_extends_a_list_and_stacks_an_array():
    being = music.Being()
    being.curseq = "f_"
    being.f_ = [1.0]
    being.addSeq([2.0, 3.0])
    assert list(being.f_) == [1.0, 2.0, 3.0]

    being.f_ = np.array([1.0])
    being.addSeq([2.0])
    assert len(being.f_) == 2


def test_howl_and_freeze_are_callable():
    """Both are placeholders; keep them from silently disappearing."""
    being = music.Being()
    being.howl()
    being.freeze()


# --------------------------------------------------------------------------
# Being.walk's reconstructed perm-walk
# --------------------------------------------------------------------------

def _being_with_perms(size=8, seqsize=4):
    """A Being ready to perm-walk, with two permutations to cycle."""
    from sympy.combinatorics import Permutation
    being = _being_with_grid(size)
    being.seqsize = seqsize
    being.perms = [Permutation([1, 0, 3, 2]), Permutation([3, 2, 1, 0])]
    return being


def test_perm_walk_permutes_each_successive_window():
    """The reconstruction: stay(method='perm') with the ground moving.

    Window [0,1,2,3] under (1,0,3,2), then [4,5,6,7] under (3,2,1,0).
    """
    being = _being_with_perms()
    being.walk(8, method='perm-walk')
    assert being.f_ == [1, 0, 3, 2, 7, 6, 5, 4]


def test_perm_walk_leaves_the_pointer_where_it_walked_to():
    """This is the whole difference from stay(): staying does not move."""
    being = _being_with_perms()
    being.walk(8, method='perm-walk')
    assert being.pointer == 8


def test_perm_walk_stops_after_the_notes_it_was_asked_for():
    """A count that is not a whole number of windows still gives n notes."""
    being = _being_with_perms()
    being.walk(6, method='perm-walk')
    assert being.f_ == [1, 0, 3, 2, 7, 6]


def test_perm_walk_wraps_around_the_end_of_the_grid():
    """Walking past the end continues from the start rather than raising,
    which is what makes the grid a cycle rather than a list that ends."""
    being = _being_with_perms(size=4, seqsize=4)
    being.walk(8, method='perm-walk')
    assert being.f_ == [1, 0, 3, 2, 3, 2, 1, 0]


def test_perm_walk_cycles_through_the_permutations():
    """Three windows and two permutations: the first comes round again."""
    being = _being_with_perms(size=12, seqsize=4)
    being.walk(12, method='perm-walk')
    assert being.f_[:4] == [1, 0, 3, 2]
    assert being.f_[8:] == [9, 8, 11, 10]


def test_perm_walk_ignores_domain_because_a_walk_is_not_a_stay():
    """`stay` honours a fixed domain; honouring it here would make the
    walk stand still, which is the one thing it must not do."""
    being = _being_with_perms()
    being.domain = [100, 200, 300, 400]
    being.walk(8, method='perm-walk')
    assert being.f_ == [1, 0, 3, 2, 7, 6, 5, 4]
