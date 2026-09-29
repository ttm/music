"""What the `legacy` mutation audit found untested or wrong."""

import numpy as np
import pytest

import music
from music.legacy import CanonicalSynth
from music.legacy.classes import ADV, V_, Being
from music.utils import WAVEFORM_SINE, WAVEFORM_TRIANGULAR


def test_v_passes_every_argument_to_the_note():
    """Only st and freq reached note_with_vibrato; the rest were two
    seconds, 2 Hz, two semitones and a triangle whatever was asked."""
    np.testing.assert_array_equal(
        V_(st=12, freq=220, duration=0.25, vibrato_freq=5, max_pitch_dev=1,
           waveform_table=WAVEFORM_SINE,
           vibrato_waveform_table=WAVEFORM_TRIANGULAR),
        music.note_with_vibrato(freq=440, duration=0.25, vibrato_freq=5,
                                max_pitch_dev=1,
                                waveform_table=WAVEFORM_SINE,
                                vibrato_waveform_table=WAVEFORM_TRIANGULAR))


def test_a_bare_v_is_two_seconds_of_220_hz():
    np.testing.assert_array_equal(
        V_(), music.note_with_vibrato(freq=220, duration=2, vibrato_freq=2,
                                      max_pitch_dev=2))


def test_adv_is_the_envelope_over_the_note():
    note = dict(freq=330, duration=0.2, vibrato_freq=6)
    envelope = dict(attack_duration=10, release_duration=30)
    np.testing.assert_array_equal(
        ADV(note, envelope),
        music.adsr(sonic_vector=V_(**note), **envelope))


def test_a_being_plays_each_note_for_its_own_duration():
    """Every note lasted two seconds, whatever d_ said."""
    being = Being()
    being.f_ = [220, 330]
    being.d_ = [0.3, 0.2]
    being.fv_ = [5, 7]
    being.nu_ = [1, 3]
    being.A_ = [10, 5]
    being.D_ = [10, 5]
    being.S_ = [-3, -6]
    being.R_ = [20, 30]
    rendered = being.render(3)
    assert len(rendered) == int(0.3 * 44100) * 2 + int(0.2 * 44100)
    expected = np.concatenate([
        music.adsr(sonic_vector=music.note_with_vibrato(
            freq=freq, duration=duration, vibrato_freq=rate,
            max_pitch_dev=depth, waveform_table=being.tab_[0]),
            attack_duration=attack, decay_duration=decay,
            sustain_level=sustain, release_duration=release)
        for freq, duration, rate, depth, attack, decay, sustain, release in [
            (220, 0.3, 5, 1, 10, 10, -3, 20),
            (330, 0.2, 7, 3, 5, 5, -6, 30),
            (220, 0.3, 5, 1, 10, 10, -3, 20)]])
    np.testing.assert_array_equal(rendered, expected)


def test_render2_reads_the_table_it_was_given_at_its_own_length():
    """It took the length from tables.size, 2048, so this 1000-sample
    table played at twice its rate and read half of itself."""
    synth = CanonicalSynth()
    table = music.waveform_table("sawtooth", 1000)
    np.testing.assert_array_equal(
        synth.render2(table=table, duration=0.1, vibrato_depth=0),
        synth.adsrApply(synth.rawRender(table=table, duration=0.1,
                                        vibrato_depth=0)))


@pytest.mark.parametrize("stage", ["A", "D"])
def test_a_one_sample_stage_holds_its_first_value(stage):
    """One sample divided zero by zero into NaN."""
    synth = CanonicalSynth()
    one_sample = 1000 / synth.samplerate
    synth.adsrSetup(**{stage: one_sample})
    envelope = synth.A_i if stage == "A" else synth.D_i
    np.testing.assert_array_equal(envelope, [0.0] if stage == "A" else [1.0])


def test_the_test_song_s_first_two_notes_sound_the_same(tmp_path,
                                                        monkeypatch):
    """Its comment said so, and the first was the tremolo envelope alone:
    `sounduration` and `tre_freq` were names nothing read."""
    soundfile = pytest.importorskip("soundfile")
    from music.legacy.pieces import testSong2

    monkeypatch.chdir(tmp_path)
    testSong2.TestSong2()
    samples, rate = soundfile.read(str(tmp_path / "TV.wav"))
    note = 4 * rate
    np.testing.assert_array_equal(samples[:note], samples[note:2 * note])
    stray = {"sounduration", "tre_freq"} & set(vars(testSong2.synth))
    assert not stray


# --------------------------------------------------------------------------
# CanonicalSynth, against the equations it implements
# --------------------------------------------------------------------------

RATE = 1000


def _synth(**state):
    return CanonicalSynth(samplerate=RATE, **state)


def test_raw_render_is_the_vibrato_and_phase_equations():
    """F_i = f 2^(T_v nu / 12), and the table read at the accumulated
    phase, as the article writes them."""
    table = np.arange(64.0)
    vibrato = np.linspace(-1, 1, 32)
    synth = _synth()
    produced = synth.rawRender(table=table, vibrato_table=vibrato,
                               fundamental_frequency=30, duration=0.25,
                               vibrato_frequency=3, vibrato_depth=5)
    samples = np.arange(250)
    vibrato_at = vibrato[np.floor(samples * 3 * 32 / RATE).astype(int) % 32]
    freqs = 30 * 2 ** (vibrato_at * 5 / 12)
    phase = np.floor(np.cumsum(freqs * 64 / RATE)).astype(int)
    np.testing.assert_array_equal(produced, table[phase % 64])


def test_render_is_the_envelope_over_the_tremolo_over_the_note():
    synth = _synth(duration=0.3, tremolo_frequency=4, tremolo_depth=6)
    expected = synth.adsrApply(synth.rawRender() * synth.tremoloEnvelope())
    np.testing.assert_array_equal(synth.render(), expected)


def test_the_adsr_stages_are_the_ramps_they_describe():
    synth = _synth()
    synth.adsrSetup(A=10, D=5, S=-6, R=4)
    sustain = 10 ** (-6 / 20)
    np.testing.assert_allclose(synth.A_i, np.arange(10) / 9)
    np.testing.assert_allclose(synth.D_i,
                               1 - (1 - sustain) * np.arange(5) / 4)
    np.testing.assert_allclose(synth.R_i, sustain * np.linspace(1, 0, 4))
    assert synth.a_S == sustain and synth.adsr_method == "absolute"
    assert synth.render_note is False
    np.testing.assert_allclose(
        synth.adsrApply(np.full(30, 2.0)),
        2 * np.concatenate((synth.A_i, synth.D_i, np.full(11, sustain),
                            synth.R_i)))


@pytest.mark.parametrize("count, indexes", [
    (4, [0, 3, 6, 9]), (1, [0]), (0, []), (10, list(range(10)))])
def test_fit_resamples_a_stage_keeping_its_ends(count, indexes):
    from music.legacy.CanonicalSynth import _fit
    np.testing.assert_array_equal(_fit(np.arange(10.0) * 2, count),
                                  np.array(indexes, dtype=float) * 2)


def test_iterating_cycles_every_sequence_by_its_own_length():
    from music.legacy import IteratorSynth
    synth = IteratorSynth()
    synth.fundamental_frequency_sequence = [220, 330, 440]
    synth.duration_sequence = [0.1, 0.2]
    seen = []
    for _ in range(4):
        synth.iterateElements()
        seen.append((synth.fundamental_frequency, synth.duration))
    assert seen == [(220, 0.1), (330, 0.2), (440, 0.1), (220, 0.2)]
    assert synth.fundamental_frequency_sequence_position == 1
    assert synth.duration_sequence_position == 0


# --------------------------------------------------------------------------
# Being
# --------------------------------------------------------------------------

def _being(grid=20, pointer=0, seqsize=3):
    being = Being()
    being.grid = list(range(grid))
    being.pointer = pointer
    being.seqsize = seqsize
    being.curseq = "f_"
    being.f_ = []
    return being


def test_a_bare_being_s_note_is_the_one_its_defaults_describe():
    being = Being()
    assert [list(values) for values in (
        being.d_, being.f_, being.fv_, being.nu_, being.A_, being.D_,
        being.S_, being.R_)] == [[1], [220], [3], [1], [20], [20], [-5],
                                 [50]]
    np.testing.assert_array_equal(
        being.render(1),
        music.adsr(sonic_vector=music.note_with_vibrato(
            freq=220, duration=1, vibrato_freq=3, max_pitch_dev=1,
            waveform_table=being.tab_[0]),
            attack_duration=20, decay_duration=20, sustain_level=-5,
            release_duration=50))


def test_dscale_scales_every_duration():
    being = Being()
    being.d_ = [0.4]
    being.dscale = 0.5
    assert len(being.render(2)) == 2 * int(0.2 * 44100)


def test_a_rendered_file_is_the_render_with_wav_added(tmp_path,
                                                      monkeypatch):
    import os
    monkeypatch.chdir(tmp_path)
    being = Being()
    being.d_ = [0.1]
    being.render(2, "piece")
    assert os.listdir(tmp_path) == ["piece.wav"]
    np.testing.assert_allclose(music.read_audio("piece.wav"),
                               music.normalize_mono(being.render(2)),
                               atol=2 ** -14)


def test_walking_moves_the_pointer_on_from_where_it_was():
    being = _being()
    being.walk(3)
    being.walk(2)
    assert being.f_ == [0, 1, 2, 3, 4] and being.pointer == 5


def test_a_low_high_walk_is_the_interleaving_it_computes():
    being = _being(pointer=2)
    being.walk(2, method="low-high")
    assert being.f_ == [2, 3, 4, 6, 3, 4]


def test_a_perm_walk_moves_the_pointer_on_by_every_window():
    from sympy.combinatorics import Permutation
    being = _being(seqsize=2)
    being.perms = [Permutation([1, 0])]
    being.walk(4, method="perm-walk")
    being.walk(2, method="perm-walk")
    assert being.f_ == [1, 0, 3, 2, 5, 4] and being.pointer == 6


def test_staying_cycles_the_permutations_in_order_and_counts_notes():
    from sympy.combinatorics import Permutation
    being = _being(seqsize=3)
    being.domain = None
    being.perms = [Permutation([0, 1, 2]), Permutation([1, 0, 2]),
                   Permutation([2, 1, 0])]
    being.stay(9)
    being.stay(3)
    assert being.f_ == [0, 1, 2, 1, 0, 2, 2, 1, 0, 0, 1, 2]
    assert being.total_notes == 12


def test_a_sequence_held_as_an_array_is_extended_as_one():
    being = _being()
    being.f_ = np.array([7.0, 8.0])
    being.walk(2)
    np.testing.assert_array_equal(being.f_, [7.0, 8.0, 0.0, 1.0])


@pytest.mark.parametrize("call, message", [
    (lambda being: being.walk(1, method="sideways"),
     "method not understood: 'sideways'; expected 'straight', 'low-high' "
     "or 'perm-walk'"),
    (lambda being: being.setPar("d"),
     "only the 'f' parameter has a grid to switch to; got 'd'. Set curseq "
     "directly to choose which sequence walk() and stay() fill."),
])
def test_the_being_s_refusals_say_what_to_do(call, message):
    import re
    with pytest.raises(ValueError, match="^" + re.escape(message) + "$"):
        call(_being())


def test_every_rhythm_a_being_keeps_fills_one_second():
    """rhythm4 was [1/4, 1/4, 1/3]."""
    resources = Being().resources
    for name in ("rhythm", "rhythm2", "rhythm3", "rhythm4"):
        assert sum(resources[name]) == pytest.approx(1), name
    assert all(sum(rhythm) == pytest.approx(1)
               for rhythm in resources["rhythmic_spectrum"])


def test_a_bare_set_par_switches_to_the_frequency_grid():
    being = _being()
    being.fgrid, being.fpointer = [5, 6, 7], 1
    being.setPar()
    assert being.grid == [5, 6, 7] and being.pointer == 1


def test_a_two_sample_stage_runs_from_one_end_to_the_other():
    synth = _synth()
    synth.adsrSetup(A=2, D=2, S=-6, R=2)
    np.testing.assert_allclose(synth.A_i, [0.0, 1.0])
    np.testing.assert_allclose(synth.D_i, [1.0, 10 ** (-6 / 20)])


@pytest.mark.parametrize("method", ["straight", "perm"])
def test_staying_stays_on_the_window_at_the_pointer(method):
    """'straight' read the window at the start of the grid, and 'perm' at
    the end of it came up short and was refused."""
    from sympy.combinatorics import Permutation
    being = _being(grid=10, pointer=8, seqsize=3)
    being.domain = None
    being.perms = [Permutation([0, 1, 2])]
    being.stay(6, method=method)
    assert being.f_ == [8, 9, 0, 8, 9, 0]
    assert being.pointer == 8


def test_each_symmetric_scale_is_one_octave_in_equal_steps():
    """freq_sym took j notes of each step j: two of the whole-tone scale,
    six tritones over two and a half octaves."""
    resources = Being().resources
    for row, extended, step in zip(resources["freq_sym"],
                                   resources["freq_sym_"], (2, 3, 4, 6)):
        assert len(row) == 12 // step
        assert row == extended[:12 // step]
        np.testing.assert_allclose(np.diff(np.log2(row)), step / 12)
        assert extended[12 // step] == pytest.approx(2 * row[0])


def test_each_diatonic_row_is_one_degree_in_every_octave():
    """Every row was the octaves of 110 Hz: the degree added to each was
    a whole rotation's sum, which is always 12."""
    resources = Being().resources
    assert resources["notes_diatonic_"] == [0, 2, 4, 5, 7, 9, 11]
    for degree, row in zip([0, 2, 4, 5, 7, 9, 11],
                           resources["freq_diatonic"]):
        assert row[:3] == pytest.approx(
            [110 * 2 ** ((12 * octave + degree) / 12) for octave in range(3)])
