"""The alternative branches of parametrised routines.

Most of this package's functions take a `method`, an `alpha`, a `stereo`
flag or a choice of duration parameter, and the suite reached only the
default of each. Two of the defects found earlier -- `fir` and the mono
path of `note_with_vibrato_seq_localization` -- were hiding exactly here.
"""

import itertools

import numpy as np
import pytest
from sympy.combinatorics import Permutation

import music
from music.core.filters.fade import cross_fade
from music.utils import mix_with_offset, rhythm_to_durations


def _finite(array):
    array = np.asarray(array)
    return array.size > 0 and np.isfinite(array).all()


# --------------------------------------------------------------------------
# Pitch transitions: alpha != 1 takes a different curve
# --------------------------------------------------------------------------

@pytest.mark.parametrize("alpha", [1, 2.0, 0.5])
@pytest.mark.parametrize("method", ["exp", "lin"])
def test_glissando_curves(alpha, method):
    out = music.note_with_glissando(start_freq=220, end_freq=440,
                                    duration=0.05, alpha=alpha, method=method)
    assert _finite(out)


@pytest.mark.parametrize("alpha", [1, 2.0])
def test_glissando_with_vibrato_curves(alpha):
    out = music.note_with_glissando_vibrato(
        start_freq=220, end_freq=440, duration=0.05, alpha=alpha
    )
    assert _finite(out)


@pytest.mark.parametrize("alpha", [1, 2.0])
def test_note_with_vibrato_curves(alpha):
    assert _finite(music.note_with_vibrato(duration=0.05, alpha=alpha))


@pytest.mark.parametrize("alphav1", [1, 2.0])
def test_two_vibratos_curves(alphav1):
    assert _finite(music.note_with_two_vibratos(duration=0.05,
                                                alphav1=alphav1))


@pytest.mark.parametrize("alpha", [1, 2.0])
def test_two_vibratos_glissando_curves(alpha):
    assert _finite(music.note_with_two_vibratos_glissando(
        duration=0.05, alpha=alpha))


def test_vibratos_glissandos_renders():
    assert _finite(music.note_with_vibratos_glissandos())


# --------------------------------------------------------------------------
# Localization: the sign of x picks which ear leads
# --------------------------------------------------------------------------

@pytest.mark.parametrize("x", [0.5, -0.5])
def test_localize_from_either_side(x):
    out = music.localize(sonic_vector=music.note(duration=0.05), x=x, y=0.1)
    assert out.shape[0] == 2 and _finite(out)


def test_localize_by_angle_and_distance():
    """theta and distance are an alternative to x and y."""
    out = music.localize(sonic_vector=music.note(duration=0.05),
                         theta=45, distance=1.0)
    assert out.shape[0] == 2 and _finite(out)


@pytest.mark.parametrize("theta", [0, -70, 70])
def test_localize2_across_angles(theta):
    with _nullcontext():
        out = music.localize2(sonic_vector=np.ones(2048), theta=theta)
    assert out.shape[0] == 2 and _finite(out)


@pytest.mark.parametrize("length", [2048, 2049, 1101, 999])
def test_localize2_handles_an_odd_number_of_samples(length):
    """Regression: the conjugate mirror ran to max_coef, which only
    balances for an even length. Every odd-length input raised."""
    with _nullcontext():
        out = music.localize2(sonic_vector=np.ones(length))
    assert out.shape == (2, length)
    assert _finite(out)


@pytest.mark.parametrize("x", [0.1, -0.1])
def test_note_with_doppler_from_either_side(x):
    out = music.note_with_doppler(x=(x, -x), number_of_samples=2000)
    assert _finite(out)


class _nullcontext:
    def __enter__(self):
        import warnings
        self._cm = warnings.catch_warnings()
        self._cm.__enter__()
        warnings.simplefilter("ignore")
        return self

    def __exit__(self, *exc):
        return self._cm.__exit__(*exc)


# --------------------------------------------------------------------------
# Envelopes and fades
# --------------------------------------------------------------------------

@pytest.mark.parametrize("to", [True, False])
@pytest.mark.parametrize("alpha", [1, 2.0])
def test_loud_directions_and_curves(to, alpha):
    assert _finite(music.loud(duration=0.05, to=to, alpha=alpha))


def test_louds_can_be_driven_by_sample_counts():
    """number_of_samples is the alternative to durations."""
    out = music.louds(number_of_samples=(500, 500), trans_devs=(6, -6),
                      alpha=(1, 1), method=("exp", "exp"))
    assert _finite(out)


@pytest.mark.parametrize("fade_out", [True, False])
@pytest.mark.parametrize("method", ["exp", "lin"])
def test_fade_directions_and_methods(fade_out, method):
    assert _finite(music.fade(duration=0.05, fade_out=fade_out,
                              method=method))


def test_fade_resolves_a_stereo_vector_per_channel():
    stereo = np.vstack((music.note(duration=0.05),) * 2)
    out = music.fade(sonic_vector=stereo)
    assert out.shape[0] == 2 and _finite(out)


def test_cross_fade_handles_stereo_pairs():
    stereo = np.vstack((music.note(duration=0.1),) * 2)
    out = cross_fade(stereo, stereo * 0.5, duration=10)
    assert out.shape[0] == 2 and _finite(out)


def test_cross_fade_rejects_mismatched_shapes():
    mono = music.note(duration=0.05)
    stereo = np.vstack((mono, mono))
    with pytest.raises(ValueError, match="same shape"):
        cross_fade(mono, stereo)


def test_tremolos_pads_a_shorter_sonic_vector():
    out = music.tremolos(sonic_vector=music.note(duration=0.05))
    assert _finite(out)


# --------------------------------------------------------------------------
# Rhythm
# --------------------------------------------------------------------------

def test_rhythm_from_beats_per_minute():
    assert _finite(rhythm_to_durations(durations=[4, 2, 2], bpm=120))


def test_rhythm_from_a_total_duration():
    out = rhythm_to_durations(durations=[4, 2, 2], total_duration=4)
    assert sum(out) == pytest.approx(4)


def test_rhythm_from_frequencies_and_a_total_duration():
    out = rhythm_to_durations(freqs=[4, 8, 8], total_duration=2)
    assert sum(out) == pytest.approx(2)


def test_rhythm_with_nested_tuplets():
    """A nested iterable is a tuplet: its first value is the cell's own
    duration and the rest divide it."""
    out = rhythm_to_durations(durations=[4, [2, 1, 1], 2], duration=0.5)
    assert _finite(out)


def test_rhythm_frequencies_with_nested_tuplets():
    out = rhythm_to_durations(freqs=[4, [2, 3, 3], 4], duration=4)
    assert _finite(out)


# --------------------------------------------------------------------------
# Mixing offsets
# --------------------------------------------------------------------------

def test_mix_with_offset_places_the_second_vector_later():
    out = mix_with_offset(np.ones(100), np.ones(50) * 2, duration=0.001)
    assert _finite(out)


def test_mix_with_offset_accepts_a_number_of_samples():
    out = mix_with_offset(np.ones(100), np.ones(50) * 2,
                          number_of_samples=20)
    assert len(out) == 100


@pytest.mark.parametrize("stereo", [True, False])
def test_pan_transitions_accepts_a_sonic_vector(stereo):
    """Regression: `if sonic_vector:` truth-tested the array, so passing
    one raised 'truth value of an array is ambiguous'."""
    tone = music.note(duration=0.3)
    signal = np.vstack((tone, tone)) if stereo else tone

    out = music.pan_transitions(d=(0.1, 0.1, 0.1), sonic_vector=signal)

    assert out.shape[0] == 2 and _finite(out)


@pytest.mark.parametrize("first_stereo", [True, False])
@pytest.mark.parametrize("second_stereo", [True, False])
def test_mix_with_offset_across_channel_combinations(first_stereo,
                                                     second_stereo):
    """Regression: the stereo path passed ['s1', 's2'] to resolve_stereo,
    parameter names that had been renamed, and raised KeyError."""
    tone = music.note(duration=0.05)
    first = np.vstack((tone, tone)) if first_stereo else tone
    second = np.vstack((tone, tone)) if second_stereo else tone

    out = music.mix_with_offset(first, second)

    expected_dims = 2 if (first_stereo or second_stereo) else 1
    assert out.ndim == expected_dims
    assert _finite(out)


# --------------------------------------------------------------------------
# Permutations and peals
# --------------------------------------------------------------------------

def test_even_odd_agrees_with_sympy_for_every_permutation():
    """The parity of a permutation is the parity of its transposition
    count; checked exhaustively rather than by example."""
    perms = music.InterestingPermutations(nelements=4)
    for sequence in itertools.permutations(range(4)):
        expected = "odd" if Permutation(list(sequence)).parity() else "even"
        assert perms.even_odd(list(sequence)) == expected


def test_interesting_permutations_of_a_pair():
    """Two elements is the degenerate dihedral case."""
    perms = music.InterestingPermutations(nelements=2)
    assert perms.dihedral


def test_plain_changes_acts_on_its_own_domain_by_default():
    peal = music.PlainChanges(3)
    rows = peal.act()
    assert len(rows) == len(peal.peal_direct)
    for row in rows:
        assert sorted(row) == [0, 1, 2]


def test_plain_changes_acts_on_a_given_domain():
    rows = music.PlainChanges(3).act(domain=[220, 440, 330])
    assert all(sorted(row) == [220, 330, 440] for row in rows)


def test_plain_changes_act_all_records_every_peal():
    """Regression: `peals` was left as None, so act_all could never run on
    a PlainChanges even though the class builds two of them."""
    peal = music.PlainChanges(3)
    assert set(peal.peals) == {"peal_direct", "peal_sequence"}

    peal.act_all()

    assert set(peal.acted_peals) == {"peal_direct_acted",
                                     "peal_sequence_acted"}
    assert peal.domain == [0, 1, 2]


# --------------------------------------------------------------------------
# Error branches
# --------------------------------------------------------------------------

def test_noise_rejects_an_unknown_colour():
    with pytest.raises(ValueError, match="Set ntype"):
        music.noise("chartreuse")


def test_stretches_rejects_a_non_positive_duration():
    with pytest.raises(ValueError, match="must be positive"):
        music.stretches(music.note(duration=0.05), durations=(1, 0))


def test_writing_nan_is_refused(tmp_path):
    with pytest.raises(ValueError, match="NaN or infinity"):
        music.write_wav_mono(np.array([0.0, np.nan, 1.0]),
                             filename=str(tmp_path / "x.wav"))


def test_generic_peal_needs_nelements_for_a_default_domain():
    holder = music.GenericPeal()
    holder.peals = {"p": [Permutation([1, 0, 2])]}
    with pytest.raises(ValueError, match="nelements has not been set"):
        holder.act("p")
    with pytest.raises(ValueError, match="nelements has not been set"):
        holder.act_all()


# ---------------------------------------------------------------------
# The last twelve, found by measuring branches rather than lines
# ---------------------------------------------------------------------
#
# The suite reports 100% coverage, which is 100% of *lines*. Turning on
# `--cov-branch` showed twelve conditions that had only ever gone one
# way. One of them turned out to be dead code -- `even_odd` guarded a
# cycle length that cannot be zero, since the loop it counts is entered
# only when the point is unvisited and its body runs before the condition
# is tested again, which an exhaustive check over all 720 permutations of
# six elements confirmed. That guard is gone. These cover the other
# eleven.

def test_tremolos_pads_a_sound_shorter_than_its_envelope():
    """Its envelope is as long as the longest tremolo in the sequence, and
    a shorter sound is zero-padded to meet it rather than truncating the
    envelope."""
    short = music.note(440, 0.5)
    out = np.asarray(music.tremolos(sonic_vector=short), dtype=float)

    assert out.size > short.size
    assert _finite(out)
    # What was padded is silent, since the padding is zeros and the
    # envelope multiplies into it.
    assert np.abs(out[short.size:]).max() == 0.0


def test_localize2_takes_the_brute_force_path():
    """`method="brute"` computes the filter coefficients one at a time
    instead of by inverse transform, and warns that it is slow. Nothing
    had run it."""
    with pytest.warns(UserWarning, match="long time"):
        placed = np.asarray(music.localize2(
            sonic_vector=music.note(440, 0.05), method="brute"),
            dtype=float)

    assert placed.shape[0] == 2
    assert _finite(placed)


def test_profile_calls_a_short_off_centre_array_a_parametrisation():
    """`profile` guesses what an array is for. A long array centred on
    zero is samples; a short one with an offset mean is a list of
    parameters, and that second guess had never been made."""
    from music.utils import profile

    guesses = profile({"tuning": np.array([5.0, 5.1, 5.2])})["guesses"]
    assert any(kind == "parametrisation"
               for kind, _why in guesses["tuning"])


def test_a_peal_acts_on_the_domain_and_peal_it_is_given():
    """Both arguments default to None and both defaults were all that had
    ever been used, so passing either was untested."""
    changes = music.PlainChanges(4)

    named = changes.act(domain=list("abcd"))
    assert named[0] == list("abcd")
    assert all(sorted(row) == sorted("abcd") for row in named)

    one_peal = changes.act(peal=changes.peal_direct[:3])
    assert len(one_peal) == 3


def test_a_peal_acts_all_of_them_on_a_domain_it_is_given():
    """`act_all` records the result on the object rather than returning
    it, which is worth pinning as well as reaching."""
    changes = music.PlainChanges(4)
    changes.act_all(domain=list("wxyz"))

    assert changes.domain == list("wxyz")
    assert changes.acted_peals
    for rows in changes.acted_peals.values():
        assert all(sorted(row) == sorted("wxyz") for row in rows)


def test_a_generic_peal_acts_all_on_a_domain_it_is_given():
    """The same on the base class, which `PlainChanges` does not use."""
    peal = music.GenericPeal()
    peal.peals = {"p": [Permutation([1, 0, 2]), Permutation([0, 2, 1])]}
    peal.act_all(domain=list("abc"))

    assert peal.acted_peals
    for rows in peal.acted_peals.values():
        assert all(sorted(row) == sorted("abc") for row in rows)


def test_a_canonical_synth_sets_up_with_the_tables_it_is_given():
    """Three `if not table` guards in `synthSetup`, one per table, and
    every one had only ever filled in a default."""
    synth = music.CanonicalSynth()
    synth.synthSetup(table=synth.tables.sine,
                     vibrato_table=synth.tables.triangle,
                     tremolo_table=synth.tables.square)

    assert synth.table is synth.tables.sine
    assert _finite(synth.render(duration=0.05))


def test_a_canonical_synth_keeps_tables_it_was_constructed_with():
    """`__init__` fills in tables and a sample rate when the state it
    absorbed did not bring them. Given them, it must leave them alone --
    the branch nothing had taken, since nobody had passed any."""
    tables = music.legacy.tables.Basic()
    synth = music.CanonicalSynth(tables=tables, samplerate=22050)

    assert synth.tables is tables
    assert synth.samplerate == 22050


def test_a_being_can_sequence_by_permutation():
    """`stay` has a `method` and only 'straight' had been used. 'perm'
    walks the permutations in `perms` over the domain instead of the grid
    in order."""
    being = music.Being()
    being.curseq = "f_"
    being.f_ = []
    being.domain = [220.0, 330.0, 440.0]
    being.perms = [Permutation([1, 0, 2]), Permutation([2, 1, 0])]

    being.stay(6, method="perm")

    assert being.total_notes == 6
    assert len(being.f_) == 6
    # Every frequency comes from the domain, permuted rather than walked
    # in order, so the set is the domain and the order is not.
    assert set(being.f_) == set(being.domain)
    assert being.f_[:3] != list(being.domain)


def test_a_being_refuses_a_way_of_sequencing_it_does_not_have():
    """It used to leave `sequence` unbound and reach `addSeq` as an
    UnboundLocalError, which is the rough edge `fade` had: a name the
    routine never bound, reported from the line that tried to use it
    rather than from the argument that was wrong."""
    being = music.Being()
    being.curseq = "f_"
    being.f_ = []

    with pytest.raises(ValueError, match="straight"):
        being.stay(2, method="nonsense")


def test_tremolos_leaves_a_sound_at_least_as_long_as_its_envelope_alone():
    """The other side of the padding: a sound already long enough is
    multiplied in as it is."""
    envelope = np.asarray(music.tremolos(), dtype=float)
    long_enough = music.note(440, len(envelope) / 44100)

    out = np.asarray(music.tremolos(sonic_vector=long_enough), dtype=float)

    assert out.size == len(envelope)
    assert _finite(out)


def test_profile_does_not_call_a_long_off_centre_array_a_parametrisation():
    """The false side of that guess: long enough to be samples, but with
    an offset mean, so neither guess above it applies."""
    from music.utils import profile

    offset = np.linspace(0.0, 1.0, 5000) + 3.0
    guesses = profile({"ramp": offset})["guesses"]

    assert not any(kind == "parametrisation"
                   for kind, _why in guesses["ramp"])
