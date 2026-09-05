"""Scales, chords and the harmonic series, against the article that gives them.

`notesInMusic.tex` is the MASS companion paper. Its equations ``eq:escalas``,
``eq:relacaoDia``, ``eq:escalasMenores``, ``eq:serieHarmonica`` and its
``triades`` are what `music.theory` implements, and these are the checks.

References
----------
.. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time
       representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)
"""

import math

import numpy as np
import pytest

import music
from music.theory import scales


# --------------------------------------------------------------------------
# eq:escalas and eq:escalasMenores -- the scales, as written
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name, degrees", [
    ("aeolian", (0, 2, 3, 5, 7, 8, 10)),
    ("locrian", (0, 1, 3, 5, 6, 8, 10)),
    ("ionian", (0, 2, 4, 5, 7, 9, 11)),
    ("dorian", (0, 2, 3, 5, 7, 9, 10)),
    ("phrygian", (0, 1, 3, 5, 7, 8, 10)),
    ("lydian", (0, 2, 4, 6, 7, 9, 11)),
    ("mixolydian", (0, 2, 4, 5, 7, 9, 10)),
])
def test_the_modes_are_the_degrees_equation_escalas_lists(name, degrees):
    assert music.scale(name) == degrees


@pytest.mark.parametrize("name, degrees", [
    ("natural minor", (0, 2, 3, 5, 7, 8, 10)),
    ("harmonic minor", (0, 2, 3, 5, 7, 8, 11)),
    ("melodic minor", (0, 2, 3, 5, 7, 9, 11, 12, 10, 8, 7, 5, 3, 2, 0)),
])
def test_the_minor_scales_are_equation_escalasmenores(name, degrees):
    assert music.scale(name) == degrees


def test_the_two_scales_the_article_names_twice_are_one_scale_each():
    """"aeolian = natural minor scale" and "ionian = major scale"."""
    assert music.scale("major") == music.scale("ionian")
    assert music.scale("minor") == music.scale("aeolian")
    assert music.scale("minor") == music.scale("natural minor")


def test_the_melodic_minor_rises_one_way_and_falls_another():
    """Fifteen degrees, not seven: the article writes both directions."""
    melodic = music.scale("melodic minor")
    assert len(melodic) == 15
    rising, falling = melodic[:8], melodic[7:]
    assert list(rising) == sorted(rising)
    assert list(falling) == sorted(falling, reverse=True)
    # It rises through the major sixth and seventh and falls through the
    # minor ones, which is the whole point of it.
    assert 9 in rising and 11 in rising
    assert 8 in falling and 10 in falling


# --------------------------------------------------------------------------
# eq:relacaoDia -- one step pattern, rotated
# --------------------------------------------------------------------------

def test_the_steps_are_the_pattern_equation_relacaodia_gives():
    assert scales.DIATONIC_STEPS == (2, 2, 1, 2, 2, 2, 1)
    assert sum(scales.DIATONIC_STEPS) == 12       # they close an octave


@pytest.mark.parametrize("kappa, name", [
    (0, "dorian"), (1, "phrygian"), (2, "lydian"), (3, "mixolydian"),
    (4, "aeolian"), (5, "locrian"), (6, "ionian"),
])
def test_every_mode_is_one_rotation_of_that_pattern(kappa, name):
    """e_0 = 0, e_i = d[(i + kappa) % 7] + e_(i-1), for each of the seven.

    This is the content of the equation: the seven modes are not seven
    facts but one pattern read from seven places.
    """
    assert music.mode_by_rotation(kappa) == music.scale(name)


def test_the_rotations_are_exactly_the_seven_modes_and_nothing_else():
    generated = {music.mode_by_rotation(k) for k in range(7)}
    assert generated == set(scales.MODES.values())
    assert len(generated) == 7


def test_the_rotation_wraps_so_any_integer_names_a_mode():
    for kappa in (-8, -1, 7, 15):
        assert music.mode_by_rotation(kappa) == music.mode_by_rotation(
            kappa % 7)


# --------------------------------------------------------------------------
# eq:serieHarmonica -- the harmonic series
# --------------------------------------------------------------------------

@pytest.mark.parametrize("partial", range(1, 21))
def test_the_harmonic_series_is_twelve_log_two_of_the_partial(partial):
    """The n-th partial is 12 log2(n) semitones above the fundamental."""
    assert music.harmonic_series(partial)[-1] == pytest.approx(
        12 * math.log2(partial))


def test_the_octaves_of_the_series_are_exact_and_nothing_else_is():
    series = music.harmonic_series(20)
    for partial in (1, 2, 4, 8, 16):
        assert series[partial - 1] == pytest.approx(
            12 * math.log2(partial), abs=1e-12)
        assert series[partial - 1] == round(series[partial - 1])
    for partial in (3, 5, 6, 7, 9, 11, 13):
        assert series[partial - 1] != round(series[partial - 1])


def test_the_series_disagrees_with_equal_temperament_where_it_is_known_to():
    """Two cents sharp at the fifth, thirty-one flat at the seventh."""
    series = music.harmonic_series(8)
    assert series[2] - 19 == pytest.approx(0.02, abs=0.005)   # the fifth
    assert series[6] - 34 == pytest.approx(-0.31, abs=0.005)  # the seventh


def test_the_printed_table_is_the_computed_series_but_for_one_digit():
    """Nineteen of twenty agree; the sixth is a typo. See DISCREPANCIES.md.

    ``eq:serieHarmonica`` prints the sixth partial as ``31 + 0.2``. The
    exact value is 31.02, which ``31 + 0.02`` gives, and the same ``+0.02``
    appears at the third partial, an octave below it.
    """
    printed = scales.HARMONIC_SERIES_AS_PRINTED
    computed = music.harmonic_series(len(printed))
    assert len(printed) == 20

    off = [n for n, (a, b) in enumerate(zip(printed, computed), start=1)
           if abs(a - b) > 0.006]
    assert off == [6]
    assert printed[5] == pytest.approx(31.2)
    assert computed[5] == pytest.approx(31.02, abs=0.005)
    assert 31 + 0.02 == pytest.approx(computed[5], abs=0.005)


def test_the_series_is_not_capped_at_the_twenty_the_article_tabulates():
    assert len(music.harmonic_series(64)) == 64
    assert music.harmonic_series(64)[-1] == pytest.approx(12 * math.log2(64))


def test_a_partial_below_the_first_is_refused():
    with pytest.raises(ValueError, match="at least 1"):
        music.harmonic_series(0)


# --------------------------------------------------------------------------
# triades -- the chords
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name, notes", [
    ("major", (0, 4, 7)),
    ("minor", (0, 3, 7)),
    ("diminished", (0, 3, 6)),
    ("augmented", (0, 4, 8)),
])
def test_the_triads_are_the_sets_the_article_writes(name, notes):
    assert music.chord(name) == notes


def test_a_seventh_is_one_more_third_on_top():
    """"include 10 ... for a minor seventh, or 11 ... for a major one"."""
    assert music.add_seventh(music.chord("major")) == (0, 4, 7, 10)
    assert music.add_seventh(music.chord("major"), major=True) == (0, 4, 7, 11)
    assert music.add_seventh(music.chord("major")) == music.chord(
        "dominant seventh")
    assert music.add_seventh(music.chord("minor")) == music.chord(
        "minor seventh")


def test_every_triad_has_a_third_and_a_fifth_of_some_kind():
    for _name, notes in music.TRIADS.items():
        assert len(notes) == 3
        assert notes[0] == 0
        assert notes[1] in (3, 4)          # minor or major third
        assert notes[2] in (6, 7, 8)       # diminished, perfect, augmented


def test_an_inversion_moves_one_note_by_an_octave():
    """"+/- 12 to the selected note", and the chord read from the bottom."""
    major = music.chord("major")
    assert music.invert(major, degree=0) == (4, 7, 12)
    assert music.invert(major, degree=1) == (0, 7, 16)
    assert music.invert(major, degree=2, octaves=-1) == (-5, 0, 4)

    # The pitch classes are unchanged, which is what makes it an inversion.
    for degree in range(3):
        inverted = music.invert(major, degree=degree)
        assert sorted(n % 12 for n in inverted) == sorted(
            n % 12 for n in major)


def test_an_open_position_is_the_same_move_upward():
    assert music.invert(music.chord("major"), degree=2, octaves=1) == \
        (0, 4, 19)


def test_a_chord_that_is_not_one_of_these_is_refused():
    with pytest.raises(ValueError, match="not a chord here"):
        music.chord("neapolitan sixth")


def test_a_scale_that_is_not_one_of_these_is_refused():
    with pytest.raises(ValueError, match="not a scale here"):
        music.scale("bebop dominant")


# --------------------------------------------------------------------------
# The point of any of it: they make sounds
# --------------------------------------------------------------------------

def test_a_scale_becomes_frequencies_by_the_tuning_equation():
    """eq:micro applied to eq:escalas: f_i = f 2 ** (s_i / 12)."""
    freqs = music.pitch_to_freq(start_freq=220.0,
                                semitones=music.scale("major"))
    expected = [220.0 * 2 ** (s / 12) for s in music.scale("major")]
    assert freqs == pytest.approx(expected)
    assert freqs[0] == pytest.approx(220.0)
    assert freqs[4] == pytest.approx(220.0 * 2 ** (7 / 12))   # the fifth


def test_a_chord_renders_as_the_sum_of_its_notes():
    freqs = music.pitch_to_freq(start_freq=220.0,
                                semitones=music.chord("major"))
    notes = [music.note(freq=f, duration=0.1) for f in freqs]
    sounded = music.mix_many(notes)
    assert len(sounded) == len(notes[0])
    assert np.allclose(sounded, sum(notes))
