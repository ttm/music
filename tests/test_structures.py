"""Group-theoretic invariants for the permutation and peal structures.

``music.structures`` is the part of this package with the fewest peers — there
is little else in Python doing change ringing — and it was the least covered.
The properties below are the ones campanology and group theory actually
guarantee, so they pin the behaviour rather than merely exercising it.

A plain-changes peal on ``n`` bells must:

* traverse the whole symmetric group, visiting each of the ``n!``
  permutations exactly once, and
* move by a single *adjacent* transposition between consecutive rows, since a
  ringer can only swap with a neighbour.

References
----------
.. [1] Stedman's *Campanalogia*, and the change-ringing survey at
       http://www.gutenberg.org/files/18567/18567-h/18567-h.htm
"""

import math
import warnings

import pytest
from sympy.combinatorics import Permutation

import music
from music.structures.peals.plain_changes import PlainChanges

# Beyond eight bells the peal is 40320 rows and the test turns into a
# benchmark; the invariants do not become more convincing with size.
SIZES = [2, 3, 4, 5, 6, 7]


def _rows(peal):
    """The peal's permutations as plain tuples, for set membership."""
    return [tuple(row) for row in peal]


def _is_adjacent_transposition(before, after):
    """True when `after` is `before` with one neighbouring pair swapped."""
    moved = [i for i, (x, y) in enumerate(zip(before, after)) if x != y]
    if len(moved) != 2:
        return False
    first, second = moved
    return (second - first == 1
            and before[first] == after[second]
            and before[second] == after[first])


@pytest.mark.parametrize("nelements", SIZES)
def test_peal_traverses_the_whole_symmetric_group(nelements):
    """Regression: the default hunt count was hardcoded to 2 for n > 4, so
    the peal covered a fraction of the group — 120 of 720 rows at n=6, and
    224 of 40320 at n=8 — with nothing to indicate it."""
    rows = _rows(PlainChanges(nelements).peal_direct)

    assert len(rows) == math.factorial(nelements)
    assert len(set(rows)) == len(rows), "a peal must not repeat a row"


@pytest.mark.parametrize("nelements", SIZES)
def test_every_change_swaps_neighbours(nelements):
    """Consecutive rows differ by one adjacent transposition — the physical
    constraint that makes a peal ringable."""
    rows = _rows(PlainChanges(nelements).peal_direct)

    for index, (before, after) in enumerate(zip(rows, rows[1:])):
        assert _is_adjacent_transposition(before, after), (
            f"row {index} -> {index + 1} is not an adjacent swap: "
            f"{before} -> {after}"
        )


@pytest.mark.parametrize("nelements", SIZES)
def test_peal_starts_from_rounds(nelements):
    """A peal opens on rounds, the identity permutation."""
    rows = _rows(PlainChanges(nelements).peal_direct)
    assert rows[0] == tuple(range(nelements))


@pytest.mark.parametrize("nelements", [4, 5, 6, 7])
def test_saturating_hunts_is_the_least_that_completes_the_group(nelements):
    """`saturating_hunts` claims to be the number of hunts at which the peal
    becomes complete. One fewer must therefore fall short."""
    saturating = PlainChanges.saturating_hunts(nelements)
    assert saturating == max(1, nelements - 3)

    complete = _rows(PlainChanges(nelements, nhunts=saturating).peal_direct)
    assert len(complete) == math.factorial(nelements)

    if saturating > 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fewer = _rows(
                PlainChanges(nelements, nhunts=saturating - 1).peal_direct
            )
        assert len(fewer) < math.factorial(nelements)


def test_more_hunts_than_saturating_warns_but_still_completes():
    """Above the saturating count nothing is gained, which the code says out
    loud. The peal must still be a valid traversal."""
    with pytest.warns(UserWarning, match="hunts less"):
        peal = PlainChanges(5, nhunts=4).peal_direct
    rows = _rows(peal)
    assert len(set(rows)) == len(rows)


def test_more_hunts_than_elements_is_rejected():
    with pytest.raises(ValueError, match="more hunts than elements"):
        PlainChanges(4, nhunts=5)


@pytest.mark.parametrize("nelements", [2, 3, 4])
def test_small_peals_do_not_warn_about_negative_hunts(nelements):
    """Regression: the warning threshold was `nelements - 3`, which goes
    negative below four bells, so PlainChanges(2) advised removing two of
    its one hunt."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        PlainChanges(nelements)


# --------------------------------------------------------------------------
# Permutation helpers
# --------------------------------------------------------------------------

def test_dist_counts_displaced_elements():
    """dist reports how far a permutation moves things."""
    assert music.dist(Permutation([0, 1, 2])) == 0
    assert music.dist(Permutation([1, 0, 2])) == 1


def test_transpose_permutation_shifts_the_support():
    """Transposing by one moves every moved index up by one."""
    transposed = music.transpose_permutation(Permutation([1, 0, 2]))
    assert transposed.support() == [1, 2]


def test_interesting_permutations_are_permutations_of_the_domain():
    """Whatever families it builds, each must be a genuine permutation."""
    perms = music.InterestingPermutations(nelements=4)
    families = [value for name, value in vars(perms).items()
                if isinstance(value, list) and value
                and isinstance(value[0], Permutation)]
    assert families, "expected at least one family of permutations"

    for family in families:
        for permutation in family:
            assert max(permutation.array_form or [0]) < 4


# --------------------------------------------------------------------------
# Peals: acting permutations on a domain
# --------------------------------------------------------------------------

def test_transpositions_peal_decomposes_into_real_transpositions():
    """Regression: this built each transposition with `Permutation(pair)`,
    reading a cycle as an array form. `Permutation((0, 1))` is the identity
    and `Permutation((0, 2))` raises outright, so the method could not
    produce a correct peal for any permutation."""
    from functools import reduce

    peals = music.Peals(5)
    permutation = Permutation([2, 0, 1, 4, 3])
    built = peals.transpositions_peal(permutation)

    assert built, "expected at least one transposition"
    for transposition in built:
        assert transposition.size == permutation.size
        assert len(transposition.support()) == 2, "not a transposition"

    # sympy composes transpositions right to left.
    product = reduce(lambda a, b: a * b, reversed(built),
                     Permutation(size=permutation.size))
    assert product == permutation


def test_transpositions_peal_is_stored_under_its_name():
    """Regression: `peals` was initialised to a list, so assigning by name
    raised TypeError. Everything else treats it as a mapping."""
    peals = music.Peals(3)
    assert isinstance(peals.peals, dict)

    peals.transpositions_peal(Permutation([1, 0, 2]), peal_name="mine")
    assert "mine" in peals.peals


def _generic_peal_with(peals_by_name, nelements):
    """A GenericPeal populated by hand, to exercise the base directly."""
    holder = music.GenericPeal()
    holder.peals = peals_by_name
    holder.nelements = nelements
    return holder


def test_acting_a_peal_permutes_the_domain():
    """A peal acted on a domain yields one arrangement of it per row."""
    permutation = Permutation([2, 0, 1])
    source = music.Peals(3)
    source.transpositions_peal(permutation, peal_name="p")
    holder = _generic_peal_with(source.peals, permutation.size)

    rows = holder.act("p")
    assert len(rows) == len(source.peals["p"])
    for row in rows:
        assert sorted(row) == list(range(permutation.size))


def test_act_all_records_every_peal():
    source = music.Peals(3)
    source.transpositions_peal(Permutation([2, 0, 1]), peal_name="one")
    source.transpositions_peal(Permutation([1, 0, 2]), peal_name="two")
    holder = _generic_peal_with(source.peals, 3)

    holder.act_all()
    assert set(holder.acted_peals) == {"one_acted", "two_acted"}
    assert holder.domain == [0, 1, 2]


def test_acting_before_any_peal_exists_says_so():
    """Regression: this surfaced as `'NoneType' object cannot be interpreted
    as an integer`, which says nothing about what went wrong."""
    empty = music.GenericPeal()
    with pytest.raises(ValueError, match="no peals have been defined"):
        empty.act("anything")
    with pytest.raises(ValueError, match="no peals have been defined"):
        empty.act_all()


def test_acting_an_unknown_peal_lists_the_known_ones():
    source = music.Peals(3)
    source.transpositions_peal(Permutation([2, 0, 1]), peal_name="known")
    holder = _generic_peal_with(source.peals, 3)
    with pytest.raises(KeyError, match="known"):
        holder.act("unknown")


def test_the_named_peals_are_implemented_now():
    """They were honest placeholders raising NotImplementedError; both
    are rung against Tintinnalogia's own tables in
    tests/test_peals_named.py, so this only pins that they exist and
    populate the peal collection they belong to."""
    peals = music.Peals(nelements=5)
    assert len(peals.twenty_all_over()) == 20
    assert len(peals.an_eight_and_forty()) == 48
    assert set(peals.peals) >= {"twenty_all_over", "an_eight_and_forty"}


def test_print_peal_writes_a_coloured_row_per_permutation(capsys):
    """It colours the hunted positions and prints one line per row."""
    peal = [[0, 1, 2], [1, 0, 2], [1, 2, 0]]

    music.print_peal(peal, hunts=[0])

    printed = capsys.readouterr().out
    # one line per row, plus print()'s own trailing newline
    assert printed.count("\n") == len(peal) + 1
    for row in peal:
        for element in row:
            assert str(element) in printed


def test_print_peal_defaults_to_hunting_the_first_two(capsys):
    music.print_peal([[0, 1, 2]])
    assert capsys.readouterr().out.strip()


def test_peals_can_act_its_own_named_peals():
    """Regression: Peals held a dict of named peals -- exactly what
    GenericPeal.act is for -- but did not inherit it, so it could build
    peals and then had no way to act them."""
    peals = music.Peals(3)
    peals.transpositions_peal(Permutation([2, 0, 1]), peal_name="mine")

    assert isinstance(peals, music.GenericPeal)

    rows = peals.act("mine")
    assert len(rows) == len(peals.peals["mine"])
    for row in rows:
        assert sorted(row) == [0, 1, 2]

    peals.act_all()
    assert set(peals.acted_peals) == {"mine_acted"}


def test_peals_acts_on_a_domain_it_is_given():
    peals = music.Peals(3)
    peals.transpositions_peal(Permutation([2, 0, 1]), peal_name="p")
    for row in peals.act("p", domain=[220, 440, 330]):
        assert sorted(row) == [220, 330, 440]


@pytest.mark.parametrize("nelements", [3, 4, 5])
def test_peals_can_be_built_for_any_size(nelements):
    """It used to call InterestingPermutations with no arguments, so it was
    always four elements and the inherited act() could only ever build a
    four-element default domain."""
    assert music.Peals(nelements).nelements == nelements


def test_a_peal_must_match_the_size_it_will_be_acted_at():
    """Otherwise the mismatch surfaces later as a sympy TypeError about
    lengths, from inside act()."""
    with pytest.raises(ValueError, match="acts on 3 elements"):
        music.Peals(4).transpositions_peal(Permutation([2, 0, 1]))


def test_plain_changes_keeps_its_own_act():
    """PlainChanges is built from one peal, not a mapping of them, so its
    act takes the domain first. The examples call it that way."""
    import inspect

    parameters = list(inspect.signature(music.PlainChanges.act).parameters)
    assert parameters[1] == "domain"
    assert not isinstance(music.PlainChanges(3), music.GenericPeal)
