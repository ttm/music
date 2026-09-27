"""What the `music.theory` mutation audit found untested or wrong.

The interval notation is checked against an oracle built independently of
the module: a degree's size is read off the major scale and then raised by
octaves, rather than through `SIMPLE_INTERVALS` and `_simple`. The audit's
survivors sat on the degrees the article's examples do not use: 6, 7, 8
and 13 through the parser, and the minor ninth.
"""

import math

import numpy as np
import pytest

import music

#: The major scale gives each degree its major or perfect size.
_MAJOR = (0, 2, 4, 5, 7, 9, 11)
#: Unison, fourth and fifth, as indices into it.
_PERFECT = (0, 3, 4)


def _expected(quality, degree):
    """Semitones for `quality` and `degree`, or None where there are none."""
    octaves, index = divmod(degree - 1, 7)
    size = _MAJOR[index] + 12 * octaves
    perfect = index in _PERFECT
    if quality == "P":
        return size if perfect else None
    if quality == "M":
        return None if perfect else size
    if quality == "m":
        return None if perfect else size - 1
    if quality in ("aug", "A"):
        return size + 1
    return size - 1 if perfect else size - 2


@pytest.mark.parametrize("degree", range(1, 23))
@pytest.mark.parametrize("quality", ["P", "M", "m", "aug", "A", "dim", "d"])
def test_every_quality_and_degree_to_three_octaves_is_its_size(
        quality, degree):
    expected = _expected(quality, degree)
    if expected is None:
        with pytest.raises(ValueError, match="perfect"):
            music.interval(f"{quality}{degree}")
    else:
        assert music.interval(f"{quality}{degree}") == expected


@pytest.mark.parametrize("semitones", range(61))
def test_every_size_to_five_octaves_is_named_and_reads_back(semitones):
    names = music.interval_names(semitones)
    tritone = semitones % 12 == 6
    assert len(names) == (3 if semitones == 6 else 2 if tritone else 1)
    for name in names:
        assert music.interval(name) == semitones


@pytest.mark.parametrize("semitones", range(61))
def test_a_compound_interval_is_classified_as_the_simple_one(semitones):
    expected = music.CONSONANCE[semitones % 12]
    assert music.consonance(semitones) == expected
    assert music.consonance(music.interval_names(semitones)[0]) == expected


def test_a_minor_ninth_is_the_strong_dissonance_of_its_second():
    assert music.consonance("m9") == "strong dissonance"
    assert music.interval_names(13) == ("m9",)


@pytest.mark.parametrize("name, semitones", [
    ("TT ", 6), (" tri", 6), (" M3 ", 4), ("\tP5\n", 7), (" M9", 14),
])
def test_surrounding_whitespace_is_ignored_for_every_name(name, semitones):
    """A table name and a parsed one were treated differently: " M3" was
    read and "TT " refused."""
    assert music.interval(name) == semitones


@pytest.mark.parametrize("semitones, names", [
    (16.0, ("M10",)),
    (np.float64(18), ("aug11", "dim12")),
    (np.int64(4), ("M3",)),
])
def test_a_whole_number_of_any_type_is_named_as_an_integer(semitones, names):
    """16.0 was named "M10.0", which `interval` cannot read back."""
    assert music.interval_names(semitones) == names


@pytest.mark.parametrize("semitones", [4.5, -0.5, np.float64(13.25)])
def test_a_fraction_of_a_semitone_has_no_name(semitones):
    with pytest.raises(ValueError, match="^an interval is a whole number"):
        music.interval_names(semitones)


@pytest.mark.parametrize("semitones", [4.5, 4.7, np.float64(-0.5)])
def test_a_fraction_of_a_semitone_has_no_classification(semitones):
    """4.7 was truncated to 4 and classified as a major third."""
    with pytest.raises(ValueError, match="^an interval is a whole number"):
        music.consonance(semitones)


def test_a_whole_float_is_classified_as_its_integer():
    assert music.consonance(np.float64(13.0)) == "strong dissonance"
    assert music.consonance(7.0) == "perfect consonance"


@pytest.mark.parametrize("lower, upper", [
    (0.0, 1.0), (1.0, 0.0), (-1.0, 1.0), (1.0, -1.0),
    (math.nan, 1.0), (1.0, math.nan),
    (math.inf, 1.0), (1.0, math.inf), (math.inf, math.inf),
])
def test_a_frequency_that_is_not_positive_and_finite_is_refused(lower, upper):
    """NaN and infinity failed inside round(), with a message about
    converting to an integer."""
    with pytest.raises(ValueError,
                       match="^frequencies must be positive and finite"):
        music.interval_between(lower, upper)


@pytest.mark.parametrize("lower, upper, semitones", [
    (0.25, 0.5, 12), (0.5, 0.75, 7), (0.5, 1.0, 12), (1.0, 1.0, 0),
])
def test_frequencies_below_one_hertz_are_frequencies(lower, upper, semitones):
    assert music.interval_between(lower, upper) == semitones


def test_an_upper_frequency_below_the_lower_is_named_as_that():
    with pytest.raises(ValueError, match="^0.5 is below 1.0"):
        music.interval_between(1.0, 0.5)


def test_the_defaults_are_the_documented_ones():
    assert music.invert(music.chord("major")) == (4, 7, 12)
    assert music.mode_by_rotation() == music.scale("ionian")
    assert music.harmonic_series() == tuple(
        12 * math.log2(n) for n in range(1, 21))


@pytest.mark.parametrize("degree, octaves, notes", [
    (2, 2, (0, 4, 31)),
    (0, -2, (-24, 4, 7)),
    (1, 3, (0, 7, 40)),
])
def test_an_inversion_moves_by_as_many_octaves_as_asked(degree, octaves,
                                                        notes):
    moved = music.invert((0, 4, 7), degree=degree, octaves=octaves)
    assert moved == notes
    assert all(isinstance(note, int) for note in moved)


@pytest.mark.parametrize("negative, positive", [(-1, 2), (-2, 1), (-3, 0)])
def test_a_negative_degree_counts_from_the_top(negative, positive):
    triad = music.chord("minor")
    assert (music.invert(triad, degree=negative)
            == music.invert(triad, degree=positive))


@pytest.mark.parametrize("degree", [3, -4])
def test_a_degree_outside_the_chord_is_refused(degree):
    with pytest.raises(IndexError):
        music.invert(music.chord("major"), degree=degree)


def test_a_diminished_unison_is_the_rule_applied_below_zero():
    """The article's rule lowers a perfect interval by a semitone, which
    takes the unison to -1. The routines that measure upward refuse it."""
    assert music.interval("dim1") == -1
    with pytest.raises(ValueError, match="not negative"):
        music.consonance("dim1")
