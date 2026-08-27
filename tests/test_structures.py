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
