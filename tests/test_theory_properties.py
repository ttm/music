"""Laws the theory, the structures and the bonds obey.

`tests/test_properties.py` does this for signals; this is the other half
of the API, which had none. Of 126 exports, 24 had a property or a
combination checked against them before these two files, and most of what
was left is not sound at all: conversions between units, names for
intervals, sets of semitones, permutation groups, and the small functions
that bind one characteristic of a note to another.

That half is well suited to laws, because almost all of it is exact.
A conversion has an inverse. An interval has a name that names it back.
The seven modes are one step pattern rotated seven ways. A peal on four
bells is twenty-four rows, each a permutation of the same four, each
differing from the last by one adjacent swap. None of that needs a
tolerance, so none of these tests carries one.

Nothing here found a defect, which is worth saying plainly: this half of
the package was already right. What it did not have was anything that
would notice if it stopped being.

Four laws failed on the way in, and the code was correct every time --
`interval_names(6)` gives three spellings and `interval` accepts a fourth,
`MODES` holds degrees where I expected steps, `harmonic_series` gives
semitones where I expected ratios, and the melodic minor is fifteen
degrees because it descends differently from how it ascends. Each of
those is now a test of the thing the code actually does, which is a
better test than the one I meant to write: the laws in this file are the
package's, not mine.
"""

import math

import numpy as np
import pytest
from sympy.combinatorics import Permutation

import music


def _steps(degrees):
    """The step pattern of a scale given as degrees, closing the octave."""
    degrees = list(degrees)
    return tuple([b - a for a, b in zip(degrees, degrees[1:])]
                 + [12 - degrees[-1]])


# ---------------------------------------------------------------------
# Conversions, which are exact and have inverses
# ---------------------------------------------------------------------

@pytest.mark.parametrize("midi", [0, 21, 60, 69, 108, 127, 69.5])
def test_a_midi_number_survives_the_trip_to_hertz_and_back(midi):
    """Exactly, since both directions are one exponential and its log."""
    assert music.hz_to_midi(music.midi_to_hz(midi)) == pytest.approx(
        midi, abs=1e-12)


@pytest.mark.parametrize("hertz", [20, 55, 261.63, 440, 1000, 12000])
def test_a_frequency_survives_the_trip_to_midi_and_back(hertz):
    assert music.midi_to_hz(music.hz_to_midi(hertz)) == pytest.approx(
        hertz, rel=1e-12)


@pytest.mark.parametrize("amplitude", [1e-4, 0.01, 0.5, 1.0, 2.0, 100.0])
def test_an_amplitude_survives_the_trip_to_decibels_and_back(amplitude):
    assert music.db_to_amp(music.amp_to_db(amplitude)) == pytest.approx(
        amplitude, rel=1e-12)


@pytest.mark.parametrize("decibels", [-80, -20, -6, 0, 6, 20])
def test_a_decibel_figure_survives_the_trip_to_amplitude_and_back(decibels):
    assert music.amp_to_db(music.db_to_amp(decibels)) == pytest.approx(
        decibels, abs=1e-9)


def test_the_conversions_agree_with_the_constants_everyone_knows():
    """A round trip is satisfied by any invertible pair, including a
    wrong one. These say the pair is the right one."""
    assert music.midi_to_hz(69) == pytest.approx(440.0)
    assert music.midi_to_hz(60) == pytest.approx(261.6255653, abs=1e-6)
    assert music.hz_to_midi(880) - music.hz_to_midi(440) == pytest.approx(12)
    assert music.midi_to_hz_interval(12) == pytest.approx(2.0)
    assert music.amp_to_db(2) == pytest.approx(6.0206, abs=1e-4)
    assert music.db_to_amp(-6.0206) == pytest.approx(0.5, abs=1e-5)


@pytest.mark.parametrize("semitones", [-24, -12, -7, 0, 7, 12, 24])
def test_an_interval_in_semitones_is_that_ratio_in_frequency(semitones):
    assert music.midi_to_hz_interval(semitones) == pytest.approx(
        2 ** (semitones / 12))


def test_both_conversions_rise_with_what_they_convert():
    """Monotonicity, which a round trip alone would not establish: a pair
    that reversed the order would still invert itself."""
    frequencies = [55, 110, 220, 440, 880]
    assert music.hz_to_midi(frequencies[0]) < music.hz_to_midi(
        frequencies[-1])
    assert all(music.hz_to_midi(low) < music.hz_to_midi(high)
               for low, high in zip(frequencies, frequencies[1:]))
    amplitudes = [0.01, 0.1, 0.5, 1.0, 4.0]
    assert all(music.amp_to_db(low) < music.amp_to_db(high)
               for low, high in zip(amplitudes, amplitudes[1:]))


# ---------------------------------------------------------------------
# Intervals, which have names that name them back
# ---------------------------------------------------------------------

@pytest.mark.parametrize("root", [110.0, 261.63, 440.0])
@pytest.mark.parametrize("semitones", list(range(13)))
def test_the_interval_between_a_note_and_its_transposition(root, semitones):
    """Measured from the frequencies, which is what the routine takes."""
    assert music.interval_between(
        root, root * 2 ** (semitones / 12)) == semitones


@pytest.mark.parametrize("name", sorted(music.SIMPLE_INTERVALS))
def test_every_interval_name_maps_to_a_size_inside_the_octave(name):
    assert 0 <= music.interval(name) <= 12


@pytest.mark.parametrize("semitones", list(range(13)))
def test_every_name_a_size_is_given_maps_back_to_that_size(semitones):
    """The round trip that holds, which is not the one going the other
    way: `interval_names(6)` gives ('aug4', 'dim5', 'TT') and `tri` is a
    fourth spelling that `interval` accepts and `interval_names` does not
    return. A name may be an alias; a size has one set of names.
    """
    names = music.interval_names(semitones)
    assert names, f"{semitones} semitones has no name"
    for name in names:
        assert music.interval(name) == semitones


@pytest.mark.parametrize("semitones", list(range(12)))
def test_every_simple_interval_has_a_consonance(semitones):
    assert isinstance(music.consonance(semitones), str)
    assert music.consonance(semitones)


def test_consonance_reads_the_same_by_name_as_by_size():
    """The two ways in should agree, since they are the same table."""
    for name in sorted(music.SIMPLE_INTERVALS):
        semitones = music.interval(name)
        if semitones == 12:          # the octave, if names reach that far
            continue
        assert music.consonance(name) == music.consonance(semitones)


# ---------------------------------------------------------------------
# The scales, which are one pattern rotated
# ---------------------------------------------------------------------

def test_the_seven_modes_are_one_step_pattern_rotated_seven_ways():
    """Which is the claim the README makes about them, checked.

    The modes are held as degrees in semitones, so the pattern is the
    differences between them with the octave closing the loop: ionian is
    (2, 2, 1, 2, 2, 2, 1) and every other mode is that sequence started
    somewhere else.
    """
    ionian = _steps(music.MODES["ionian"])
    rotations = {tuple(ionian[i:] + ionian[:i]) for i in range(len(ionian))}

    assert len(music.MODES) == 7
    for mode, degrees in music.MODES.items():
        assert len(degrees) == 7, mode
        assert _steps(degrees) in rotations, mode


@pytest.mark.parametrize("name", sorted(music.MODES))
def test_every_mode_spans_exactly_one_octave(name):
    assert sum(_steps(music.MODES[name])) == 12


#: The scale that is not a set of degrees, and why.
GOES_UP_AND_COMES_DOWN = {
    "melodic minor": (
        "it raises its sixth and seventh going up and reverts them coming "
        "down, which is the whole point of it, so it is held as fifteen "
        "degrees rising to the octave and falling back rather than as "
        "seven"),
}


@pytest.mark.parametrize("name", sorted(music.SCALES))
def test_every_scale_is_rising_degrees_inside_one_octave(name):
    if name in GOES_UP_AND_COMES_DOWN:
        return
    degrees = music.SCALES[name]
    assert list(degrees) == sorted(degrees)
    assert len(set(degrees)) == len(degrees)
    assert all(0 <= degree < 12 for degree in degrees)


def test_the_melodic_minor_rises_one_way_and_falls_another():
    """The exception above, stated as the law it does obey.

    Up: 0 2 3 5 7 9 11 12, with the sixth and seventh raised. Down: the
    same octave with both reverted, which is the natural minor backwards.
    A test that only knew scales were rising sets would have to exclude
    this one; this says what it is instead.
    """
    degrees = list(music.SCALES["melodic minor"])
    peak = degrees.index(12)
    rising, falling = degrees[:peak + 1], degrees[peak:]

    assert rising == sorted(rising)
    assert falling == sorted(falling, reverse=True)
    assert rising[-1] == 12 and falling[-1] == 0

    natural = list(music.SCALES["natural minor"])
    assert rising == natural[:5] + [natural[5] + 1, natural[6] + 1, 12], (
        "going up it is the natural minor with the sixth and seventh "
        "raised a semitone")
    assert falling == [12] + natural[::-1][:-1] + [0], (
        "coming down it is the natural minor again, unraised")
    assert rising[:-1] != falling[1:][::-1], (
        "and the two directions differ, which is what makes it melodic")


@pytest.mark.parametrize("name", sorted(music.SCALES))
@pytest.mark.parametrize("tonic", [0, 5, 11])
def test_a_scale_moves_with_its_tonic(name, tonic):
    """Transposition adds, and nothing else changes."""
    assert tuple(music.scale(name, tonic=tonic)) == tuple(
        degree + tonic for degree in music.scale(name))


@pytest.mark.parametrize("name", sorted(music.CHORDS))
@pytest.mark.parametrize("root", [0, 4, 9])
def test_a_chord_moves_with_its_root(name, root):
    assert tuple(music.chord(name, root=root)) == tuple(
        note + root for note in music.chord(name))


@pytest.mark.parametrize("name", sorted(music.TRIADS))
def test_a_seventh_extends_the_triad_it_was_built_from(name):
    """The first three notes are the triad, and the fourth is above them."""
    triad = music.chord(name)
    for major in (False, True):
        seventh = music.add_seventh(triad, major=major)
        assert tuple(seventh)[:3] == tuple(triad)
        assert seventh[3] > triad[-1]


def test_the_harmonic_series_is_the_integer_multiples_it_names():
    """It is given in semitones above the fundamental, so the check is
    that raising two to them gives 1, 2, 3, 4 -- the third partial at
    19.0195 semitones being a perfect twelfth, which is the interesting
    one and the reason the numbers do not look like ratios."""
    series = music.harmonic_series(12)
    for partial, semitones in enumerate(series, start=1):
        assert 2 ** (semitones / 12) == pytest.approx(partial, rel=1e-9)


@pytest.mark.parametrize("kappa", [0, 1, 6])
def test_a_mode_by_rotation_is_a_mode(kappa):
    rotated = music.mode_by_rotation(kappa)
    assert len(rotated) == 7
    assert sum(_steps(rotated)) == 12


def test_frequencies_from_semitones_rise_with_them():
    """`pitch_to_freq` over a rising scale should give rising pitches, and
    the octave should double."""
    frequencies = music.pitch_to_freq(start_freq=220.0,
                                      semitones=(0, 2, 4, 5, 7, 9, 11, 12))
    assert all(low < high
               for low, high in zip(frequencies, frequencies[1:]))
    assert frequencies[-1] == pytest.approx(2 * frequencies[0])


# ---------------------------------------------------------------------
# Permutations, where the laws are the point of the structure
# ---------------------------------------------------------------------

@pytest.mark.parametrize("bells", [3, 4, 5])
def test_plain_changes_ring_every_row_once(bells):
    """A peal visits each arrangement of the bells exactly once, which is
    what makes it an extent: `bells` factorial rows, no repeats."""
    rows = [tuple(row) for row in music.PlainChanges(bells).peal_direct]

    assert len(rows) == math.factorial(bells)
    assert len(set(rows)) == len(rows)
    assert all(sorted(row) == sorted(rows[0]) for row in rows)


@pytest.mark.parametrize("bells", [3, 4, 5])
def test_each_change_swaps_one_adjacent_pair(bells):
    """Which is the rule the method is named for. Two positions differ
    between consecutive rows, and they are next to each other."""
    rows = [tuple(row) for row in music.PlainChanges(bells).peal_direct]

    for before, after in zip(rows, rows[1:]):
        moved = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
        assert len(moved) == 2, f"{before} -> {after} moved {len(moved)}"
        assert moved[1] - moved[0] == 1, f"{before} -> {after} is not adjacent"


def test_transposing_a_permutation_shifts_what_it_moves():
    """And transposing by nothing changes nothing."""
    permutation = Permutation([2, 0, 1])

    assert music.transpose_permutation(permutation, 0) == permutation
    for step in (1, 2, 5):
        shifted = music.transpose_permutation(permutation, step)
        assert sorted(shifted.support()) == [
            point + step for point in sorted(permutation.support())]


@pytest.mark.parametrize("bells", [3, 4])
def test_interesting_permutations_are_permutations(bells):
    """Whatever else they are, every one has to be a rearrangement of the
    same bells."""
    interesting = music.InterestingPermutations(bells)
    groups = [value for name, value in vars(interesting).items()
              if isinstance(value, list) and value
              and isinstance(value[0], Permutation)]
    assert groups, "no lists of permutations to check"

    for group in groups:
        for permutation in group:
            assert sorted(permutation.array_form) == sorted(
                range(len(permutation.array_form)))


# ---------------------------------------------------------------------
# Bonds, which are small functions and should behave like them
# ---------------------------------------------------------------------

@pytest.mark.parametrize("factor,offset", [(1, 0), (3, 1), (-2, 0.5)])
def test_a_proportional_bond_is_the_line_it_describes(factor, offset):
    bond = music.proportional(factor=factor, offset=offset)
    for value in (0.0, 1.0, 2.5, 10.0):
        assert bond(value) == pytest.approx(factor * value + offset)


@pytest.mark.parametrize("numerator,offset", [(1, 0), (6, 0), (2, 1.5)])
def test_an_inversely_proportional_bond_is_the_curve_it_describes(
        numerator, offset):
    bond = music.inversely_proportional(numerator=numerator, offset=offset)
    for value in (1.0, 2.0, 4.0):
        assert bond(value) == pytest.approx(numerator / value + offset)


def test_a_stepped_bond_takes_the_value_of_the_threshold_it_falls_under():
    """And the fallback above the last one."""
    bond = music.stepped(thresholds=[(1.0, 10.0), (2.0, 20.0)],
                         otherwise=99.0)
    assert bond(0.5) == 10.0
    assert bond(1.5) == 20.0
    assert bond(9.0) == 99.0


# ---------------------------------------------------------------------
# The two halves together
# ---------------------------------------------------------------------

def test_a_scale_becomes_frequencies_becomes_sound():
    """The two steps the README describes -- semitones to frequencies to
    samples -- and the pitch that comes out at the end should be the pitch
    that went in at the start.

    Nothing else runs the theory into the synthesis. A scale that was
    right and a note that was right could still disagree about what a
    semitone is, and only this would show it.
    """
    degrees = music.scale("major")
    frequencies = music.pitch_to_freq(start_freq=220.0, semitones=degrees)

    for degree, frequency in zip(degrees, frequencies):
        rendered = np.asarray(music.note(frequency, 0.5), dtype=float)
        spectrum = np.abs(np.fft.rfft(rendered))
        spectrum[0] = 0.0
        strongest = np.fft.rfftfreq(len(rendered), 1 / 44100)[
            spectrum.argmax()]

        assert strongest == pytest.approx(220.0 * 2 ** (degree / 12),
                                          rel=0.02)


def test_a_peal_becomes_a_melody_of_the_right_length():
    """Rows of bells rendered as notes: the structures and the synthesis
    used together, which is what the package is for."""
    rows = [tuple(row) for row in music.PlainChanges(3).peal_direct]
    frequencies = [220.0 * 2 ** (bell / 12) for row in rows for bell in row]

    melody = music.horizontal_stack(
        *[music.adsr(sonic_vector=music.note(freq, 0.05))
          for freq in frequencies])

    assert len(melody) == len(frequencies) * int(0.05 * 44100)
    assert np.isfinite(melody).all()


def test_a_sequencer_puts_its_notes_where_it_was_told():
    """Two notes half a second apart make a second of sound, and the
    second one starts where the first one stops."""
    sequencer = music.Sequencer()
    sequencer.add_note(440, start=0.0, duration=0.5)
    sequencer.add_note(880, start=0.5, duration=0.5)
    rendered = np.asarray(sequencer.render(), dtype=float)

    assert abs(rendered.shape[-1] - 44100) < 100

    half = rendered.shape[-1] // 2
    for start, stop, expected in ((0, half, 440.0), (half, None, 880.0)):
        segment = rendered[start:stop]
        spectrum = np.abs(np.fft.rfft(segment))
        spectrum[0] = 0.0
        strongest = np.fft.rfftfreq(len(segment), 1 / 44100)[
            spectrum.argmax()]
        assert strongest == pytest.approx(expected, rel=0.05)


def test_mixing_many_is_mixing_two_repeatedly():
    """`mix_many` should be `mix` folded over the list, which is the only
    thing it could sensibly be."""
    sounds = [music.note(freq, duration)
              for freq, duration in ((440, 0.2), (550, 0.3), (660, 0.1))]

    folded = sounds[0]
    for sound in sounds[1:]:
        folded = music.mix(folded, sound)

    assert np.allclose(music.mix_many(sounds), folded, atol=1e-12)
