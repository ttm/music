"""The two named peals, against the book they come from.

`Peals` cites Tintinnalogia (1668) as its core reference and then raised
NotImplementedError for the two peals it names. Both are implemented as
the rules the book states, and both are checked here against the tables
the book prints -- row for row, in order, rather than by counting rows
or asserting that nothing crashed.

The tables below are transcribed from Project Gutenberg's edition of
Duckworth and Stedman, *Tintinnalogia, or, The Art of Ringing*:
https://www.gutenberg.org/files/18567/18567-h/18567-h.htm
"""

import pytest

import music

#: The Twenty all over, as printed: twenty changes and back into rounds.
TWENTY_ALL_OVER = """
12345 21345 23145 23415 23451
32451 34251 34521 34512
43512 45312 45132 45123
54123 51423 51243 51234
15234 12534 12354 12345
""".split()

#: An Eight and Forty, as printed.
EIGHT_AND_FORTY = """
12345 12354 12534 15234 51234 52134 25134 21534 21354 21345
21435 24135 42135 42315 24315 23415 23145 23154 23514 25314
52314 53214 35214 32514 32154 32145 32415 34215 43215 43125
34125 31425 31245 31254 31524 35124 53124 51324 15324 13524
13254 13245 13425 14325 41325 41235 14235 12435 12345
""".split()


def rung(peal_name, nelements=5):
    """The peal as rows of bell numbers, the way the book prints them."""
    peals = music.Peals(nelements=nelements)
    getattr(peals, peal_name)()
    return [''.join(str(bell + 1) for bell in row)
            for row in peals.act(peal_name)]


def test_twenty_all_over_is_the_table_in_the_book():
    assert rung('twenty_all_over') == TWENTY_ALL_OVER[:-1]


def test_twenty_all_over_comes_round_again():
    """The last change of the peal restores rounds, which is what makes
    it a peal rather than a sequence that stops."""
    rows = rung('twenty_all_over')
    assert rows[0] == '12345'
    assert TWENTY_ALL_OVER[-1] == '12345'
    assert len(rows) == 20


def test_an_eight_and_forty_is_the_table_in_the_book():
    assert rung('an_eight_and_forty') == EIGHT_AND_FORTY[:-1]


def test_an_eight_and_forty_is_true():
    """No row is rung twice, which is the condition for a true peal."""
    rows = rung('an_eight_and_forty')
    assert len(rows) == 48
    assert len(set(rows)) == 48


def test_every_change_swaps_one_adjacent_pair():
    """A change exchanges bells that are next to each other. Two rows
    differing anywhere else would not be ringable."""
    for name in ('twenty_all_over', 'an_eight_and_forty'):
        rows = rung(name)
        for before, after in zip(rows, rows[1:]):
            differing = [i for i in range(5) if before[i] != after[i]]
            assert len(differing) == 2, (name, before, after)
            assert differing[1] - differing[0] == 1, (name, before, after)


@pytest.mark.parametrize('nelements, changes', [(3, 6), (4, 12), (6, 30)])
def test_twenty_all_over_is_a_rule_not_a_table(nelements, changes):
    """The rule holds for any number of bells; twenty is the five-bell
    case. Each of n bells takes n-1 changes to hunt to the back."""
    rows = rung('twenty_all_over', nelements=nelements)
    assert len(rows) == changes
    assert len(set(rows)) == changes


def test_an_eight_and_forty_refuses_any_other_number_of_bells():
    """It is a composition for five, not a rule that generalizes: two
    whole hunts plus the three ringing the six changes are five."""
    with pytest.raises(ValueError, match='peal on five bells'):
        music.Peals(nelements=4).an_eight_and_forty()


def test_the_three_free_bells_ring_the_plain_changes_on_three():
    """The book calls them "the six changes", and they are the same six
    PlainChanges produces for three elements."""
    rows = rung('an_eight_and_forty')
    seen, order = set(), []
    for row in rows:
        trio = tuple(int(c) for c in row if c in '123')
        if trio not in seen:
            seen.add(trio)
            order.append(trio)
    assert len(order) == 6

    plain = music.PlainChanges(nelements=3)
    expected = {tuple(bell + 1 for bell in row) for row in plain.act()}
    assert set(order) == expected
