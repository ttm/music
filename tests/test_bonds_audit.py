"""What the `bonds` mutation audit found untested or wrong."""

import numpy as np
import pytest

import music
from music.bonds import Bonds, stepped


def test_a_stepped_bond_reads_its_thresholds_once():
    """A generator was read by each call where the last left off, so 220
    Hz fell in the first step and then, a note later, in the second."""
    bond = stepped(((below, value) for below, value
                    in [(262, 3.0), (523, 6.0)]), otherwise=12.0)
    assert [bond(220), bond(220), bond(440), bond(880)] == [
        3.0, 3.0, 6.0, 12.0]


@pytest.mark.parametrize("bound, routine_defaults", [
    ({"vibrato_freq": 7}, {"vibrato_freq": 7}),
    ({"max_pitch_dev": 3}, {"max_pitch_dev": 3}),
])
def test_an_unbound_vibrato_value_is_the_routine_s_own_default(
        monkeypatch, bound, routine_defaults):
    """Only what is bound is passed; the rest is note_with_vibrato's."""
    calls = []
    module = __import__("music.bonds", fromlist=["note_with_vibrato"])
    monkeypatch.setattr(module, "note_with_vibrato",
                        lambda **kwargs: calls.append(kwargs) or
                        np.zeros(4))
    Bonds(**bound).note(freq=330, duration=0.25, sample_rate=16)
    assert calls == [dict(freq=330, duration=0.25, sample_rate=16,
                          **routine_defaults)]


@pytest.mark.parametrize("bound", [{"tremolo_freq": 5}, {"max_db_dev": 3}])
def test_an_unbound_tremolo_value_is_the_routine_s_own_default(
        monkeypatch, bound):
    calls = []
    module = __import__("music.bonds", fromlist=["tremolo"])
    monkeypatch.setattr(module, "tremolo",
                        lambda **kwargs: calls.append(kwargs) or
                        kwargs["sonic_vector"])
    Bonds(**bound).note(freq=330, duration=0.25, sample_rate=16)
    assert len(calls) == 1
    sound = calls[0].pop("sonic_vector")
    assert calls[0] == dict(sample_rate=16, **bound)
    np.testing.assert_array_equal(
        sound, music.note(freq=330, duration=0.25, sample_rate=16))


def test_an_inversely_proportional_bond_at_zero_says_why():
    bond = music.inversely_proportional(1000)
    with pytest.raises(ValueError, match="^an inversely proportional bond "
                       "has no value at a frequency of zero$"):
        bond(0)


def test_a_stepped_bond_above_every_step_is_zero_by_default():
    assert stepped([(262, 3.0)])(880) == 0.0


def test_a_bare_render_gives_each_note_two_seconds():
    assert len(Bonds().render([220, 330])) == 2 * 2 * 44100


def test_a_render_is_at_the_rate_it_is_given():
    rendered = Bonds(vibrato_freq=5).render([220, 330], duration=0.5,
                                            sample_rate=8000)
    np.testing.assert_array_equal(rendered, np.concatenate([
        Bonds(vibrato_freq=5).note(freq, duration=0.5, sample_rate=8000)
        for freq in (220, 330)]))


def test_a_render_of_nothing_says_so():
    with pytest.raises(ValueError, match="^render needs at least one "
                       "frequency$"):
        Bonds().render([])


def test_the_bare_bonds_are_the_documented_ones():
    assert music.inversely_proportional()(500) == 2.0
    np.testing.assert_array_equal(Bonds().note(), music.note(220, 2))
