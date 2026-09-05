"""Bonds between a note's characteristics, for `eq:vinculos`.

The article's equation is a schema: it says a vibrato rate may be a
function of the note's frequency without naming the function, and adds that
such functions "are arbitrary and dependent on musical intentions". So
these do not check a formula. They check that the place a formula goes
behaves: that a bond is applied to every note, that it is applied by
frequency, and that a note with no bonds is the plain note.

References
----------
.. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time
       representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)
"""

import numpy as np
import pytest

import music
from music.bonds import BINDABLE


def test_the_ready_made_bonds_are_the_two_relations_the_article_names():
    """"a vibrato frequency proportional to note pitch", and the inverse."""
    rising = music.proportional(1 / 100)
    assert rising(220) == pytest.approx(2.2)
    assert rising(440) == pytest.approx(4.4)
    assert rising(440) == pytest.approx(2 * rising(220))

    falling = music.inversely_proportional(1000)
    assert falling(200) == pytest.approx(5.0)
    assert falling(400) == pytest.approx(2.5)
    assert falling(400) == pytest.approx(falling(200) / 2)


def test_a_proportional_bond_takes_an_offset():
    bond = music.proportional(1 / 100, offset=1.0)
    assert bond(0) == pytest.approx(1.0)
    assert bond(200) == pytest.approx(3.0)


def test_an_inverse_bond_has_no_value_at_zero_and_says_so():
    with pytest.raises(ValueError, match="frequency of zero"):
        music.inversely_proportional(1000)(0)


def test_a_stepped_bond_changes_by_register():
    register = music.stepped([(262, 3.0), (523, 6.0)], otherwise=12.0)
    assert register(220) == 3.0
    assert register(261) == 3.0
    assert register(262) == 6.0
    assert register(880) == 12.0


# --------------------------------------------------------------------------
# The bonds themselves
# --------------------------------------------------------------------------

def test_a_bond_is_evaluated_at_the_frequency_of_the_note():
    """func_a(f), func_b(f), func_c(f): each reads the note's own pitch."""
    bonds = music.Bonds(vibrato_freq=music.proportional(1 / 40),
                        max_pitch_dev=music.inversely_proportional(500))
    assert bonds.characteristics(440) == {
        "vibrato_freq": pytest.approx(11.0),
        "max_pitch_dev": pytest.approx(500 / 440),
    }
    assert bonds.characteristics(220)["vibrato_freq"] == pytest.approx(5.5)


def test_a_constant_bond_is_the_same_at_every_frequency():
    """A characteristic that does not vary is still a bond, just a flat one."""
    bonds = music.Bonds(max_pitch_dev=2.0)
    for freq in (110, 440, 1760):
        assert bonds.characteristics(freq) == {"max_pitch_dev": 2.0}


def test_binding_something_that_is_not_a_characteristic_is_refused():
    with pytest.raises(ValueError, match="cannot be bound"):
        music.Bonds(loudness=music.proportional(1))


def test_the_bindable_characteristics_are_the_ones_the_equation_names():
    """f_vbr and f_tr, nu, and V_dB -- the four of eq:vinculos."""
    assert set(BINDABLE) == {
        "vibrato_freq", "max_pitch_dev", "tremolo_freq", "max_db_dev"}


def test_bonds_with_nothing_bound_renders_the_plain_note():
    """The identity case, so an empty set of bonds is not a special case."""
    plain = music.Bonds().note(freq=440, duration=0.1)
    assert np.array_equal(plain, music.note(freq=440, duration=0.1))


def test_a_vibrato_bond_renders_through_note_with_vibrato():
    bonds = music.Bonds(vibrato_freq=music.proportional(1 / 40),
                        max_pitch_dev=1.5)
    produced = bonds.note(freq=440, duration=0.2)
    expected = music.note_with_vibrato(freq=440, duration=0.2,
                                       vibrato_freq=11.0, max_pitch_dev=1.5)
    assert np.array_equal(produced, expected)


def test_a_tremolo_bond_is_an_envelope_over_whatever_is_underneath():
    bonds = music.Bonds(tremolo_freq=6.0, max_db_dev=12.0)
    produced = bonds.note(freq=440, duration=0.2)
    expected = music.tremolo(duration=0.2, tremolo_freq=6.0, max_db_dev=12.0,
                             sonic_vector=music.note(freq=440, duration=0.2))
    assert np.array_equal(produced, expected)


def test_both_kinds_of_bond_compose():
    bonds = music.Bonds(vibrato_freq=3.0, max_pitch_dev=1.0,
                        tremolo_freq=5.0, max_db_dev=6.0)
    produced = bonds.note(freq=330, duration=0.2)
    underneath = music.note_with_vibrato(freq=330, duration=0.2,
                                         vibrato_freq=3.0, max_pitch_dev=1.0)
    expected = music.tremolo(duration=0.2, tremolo_freq=5.0, max_db_dev=6.0,
                             sonic_vector=underneath)
    assert np.array_equal(produced, expected)


# --------------------------------------------------------------------------
# Rendering a sequence
# --------------------------------------------------------------------------

def test_every_note_of_a_render_carries_the_bond_at_its_own_frequency():
    """What makes it a bond and not a parameter: the piece decides once."""
    bonds = music.Bonds(vibrato_freq=music.proportional(1 / 40))
    freqs, duration = [220.0, 440.0, 330.0], 0.1
    rendered = bonds.render(freqs, duration=duration)

    length = int(duration * 44100)
    assert len(rendered) == length * len(freqs)
    for i, freq in enumerate(freqs):
        expected = music.note_with_vibrato(
            freq=freq, duration=duration, vibrato_freq=freq / 40,
            max_pitch_dev=2)
        assert np.array_equal(rendered[i * length:(i + 1) * length], expected)


def test_a_render_takes_one_duration_for_all_or_one_for_each():
    bonds = music.Bonds(max_pitch_dev=1.0, vibrato_freq=4.0)
    same = bonds.render([220, 440], duration=0.1)
    each = bonds.render([220, 440], duration=[0.1, 0.1])
    assert np.array_equal(same, each)

    uneven = bonds.render([220, 440], duration=[0.1, 0.3])
    assert len(uneven) == int(0.4 * 44100)


def test_a_render_of_nothing_is_refused_rather_than_returning_nothing():
    with pytest.raises(ValueError, match="at least one frequency"):
        music.Bonds().render([])


def test_the_durations_must_match_the_frequencies_they_are_for():
    with pytest.raises(ValueError, match="3 durations for 2 frequencies"):
        music.Bonds().render([220, 440], duration=[0.1, 0.2, 0.3])


def test_bonds_say_what_they_have_bound():
    assert repr(music.Bonds()) == "Bonds()"
    two = music.Bonds(vibrato_freq=3.0, max_db_dev=1.0)
    assert repr(two) == "Bonds(max_db_dev, vibrato_freq)"


def test_a_bound_render_is_finite_and_peaks_where_the_deepest_bond_says():
    """It is still a sound, whatever the bonds say.

    A tremolo of V_dB spans that many decibels either way, so it lifts the
    carrier as well as cutting it and the render passes unity by design.
    The deepest tremolo here is 8.8 dB, on the highest note, and that is
    what the peak follows -- which is also the reason a bound piece needs
    normalising before it is written.
    """
    bonds = music.Bonds(vibrato_freq=music.proportional(1 / 20),
                        max_pitch_dev=music.inversely_proportional(200),
                        tremolo_freq=music.stepped([(300, 2.0)],
                                                   otherwise=8.0),
                        max_db_dev=music.proportional(1 / 100))
    freqs = [110, 220, 440, 880]
    rendered = bonds.render(freqs, duration=0.1)

    assert np.isfinite(rendered).all()
    deepest = max(bonds.characteristics(f)["max_db_dev"] for f in freqs)
    assert np.abs(rendered).max() == pytest.approx(
        music.db_to_amp(deepest), rel=0.02)
    assert np.abs(music.normalize_mono(rendered)).max() == pytest.approx(1.0)
