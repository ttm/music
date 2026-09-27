"""What the `music.structures` mutation audit found untested or wrong."""

import functools
from importlib import import_module
import itertools
import re
import warnings

import pytest
from sympy.combinatorics import Permutation
import termcolor

import music
from music.structures.permutations import InterestingPermutations


def _all_permutations(size):
    return [Permutation(list(order))
            for order in itertools.permutations(range(size))]


@pytest.mark.parametrize("step", [1, 2, 3])
@pytest.mark.parametrize("permutation", _all_permutations(4))
def test_a_transposed_permutation_sends_each_shifted_point_where_it_went(
        permutation, step):
    """Conjugating by the shift: i + step goes to p(i) + step, and the
    points below the shift stay put. A three-cycle came back running the
    other way, and two swaps as one four-cycle."""
    shifted = music.transpose_permutation(permutation, step)
    moved = permutation.support()
    if not moved:
        assert shifted == permutation
        return
    assert shifted.size == max(permutation.size, moved[-1] + step + 1)
    for point in range(shifted.size):
        if point - step in moved:
            assert shifted(point) == permutation(point - step) + step
        else:
            assert shifted(point) == point
    assert sorted(len(cycle) for cycle in shifted.cyclic_form) == sorted(
        len(cycle) for cycle in permutation.cyclic_form)


def test_a_shifted_three_cycle_keeps_its_direction():
    assert music.transpose_permutation(
        Permutation([2, 0, 1])).cyclic_form == [[1, 3, 2]]


def test_two_shifted_swaps_stay_two_swaps():
    assert music.transpose_permutation(
        Permutation([1, 0, 3, 2])).cyclic_form == [[1, 2], [3, 4]]


def test_a_shifted_swap_still_acts_on_its_row():
    """It was sized to its highest point, so a swap of four bells moved up
    one could not act on four bells."""
    swap = Permutation(0, 1, size=4)
    shifted = music.transpose_permutation(swap)
    assert shifted.size == 4
    assert shifted(["a", "b", "c", "d"]) == ["a", "c", "b", "d"]


def test_a_permutation_can_be_shifted_down():
    shifted = music.transpose_permutation(Permutation(2, 3, size=5), -2)
    assert shifted.cyclic_form == [[0, 1]] and shifted.size == 5


@pytest.mark.parametrize("step", [-2, -3])
def test_a_shift_below_zero_is_refused(step):
    with pytest.raises(ValueError, match="^a step of .* below zero"):
        music.transpose_permutation(Permutation(1, 2, size=4), step)


def test_a_shift_down_to_zero_is_allowed():
    assert music.transpose_permutation(
        Permutation(1, 2, size=4), -1).cyclic_form == [[0, 1]]


@pytest.mark.parametrize("step", [0, 1, -1])
def test_shifting_rounds_changes_nothing(step):
    rounds = Permutation([0, 1, 2, 3])
    assert music.transpose_permutation(rounds, step) is rounds


@pytest.mark.parametrize("nelements", [1, 0, -1])
@pytest.mark.parametrize("build", [InterestingPermutations, music.Peals])
def test_one_element_has_no_structures_to_build(build, nelements):
    with pytest.raises(ValueError, match="^nelements must be at least 2"):
        build(nelements=nelements)


def test_a_generation_method_sympy_lacks_is_refused():
    """sympy raised NotImplementedError for it."""
    with pytest.raises(ValueError, match="^method must be 'dimino' or"):
        InterestingPermutations(4, method="schreier_sims")


def test_both_generation_methods_give_the_same_permutations():
    dimino = InterestingPermutations(4, method="dimino")
    coset = InterestingPermutations(4, method="coset")
    assert set(dimino.permutations) == set(coset.permutations)
    assert set(dimino.alternations) == set(coset.alternations)


@pytest.mark.parametrize("sequence", [[0, 0], [1, 1], [1, 2], [0, 1, 1]])
def test_parity_is_refused_for_what_is_not_a_permutation(sequence):
    with pytest.raises(ValueError, match="^sequence must hold each of"):
        InterestingPermutations(3).even_odd(sequence)


@pytest.mark.parametrize("nelements", [1, 0])
def test_plain_changes_need_two_bells(nelements):
    with pytest.raises(ValueError, match="^nelements must be at least 2"):
        music.PlainChanges(nelements)


def test_a_negative_number_of_hunts_is_refused():
    with pytest.raises(ValueError, match="^nhunts cannot be negative"):
        music.PlainChanges(5, nhunts=-1)


def test_the_unread_hunts_argument_warns_and_changes_nothing():
    with pytest.warns(UserWarning, match="does not read hunts"):
        given = music.PlainChanges(4, hunts={"hunt0": {}})
    assert given.peal_direct == music.PlainChanges(4).peal_direct


def test_no_warning_without_the_hunts_argument():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        music.PlainChanges(4)


def test_a_peal_on_nine_or_more_bells_prints(capsys):
    """The eight colours ran out at the ninth bell."""
    rows = [list(range(10)), [1, 0] + list(range(2, 10))]
    music.print_peal(rows, hunts=(0, 9))
    printed = capsys.readouterr().out
    for digit in "0123456789":
        assert digit in printed
    assert printed.count("\n") == 3


# --------------------------------------------------------------------------
# The families InterestingPermutations builds, each against a definition
# that does not go through sympy's groups
# --------------------------------------------------------------------------

def _exactly(message):
    return "^" + re.escape(message) + "$"


def _moving(size, count):
    """Every permutation of `size` elements that moves `count` of them."""
    return {permutation for permutation in _all_permutations(size)
            if permutation.length() == count}


def _around_the_circle(size):
    """The n-gon's rotations and reflections, as maps of its corners."""
    rotations = {Permutation([(i + k) % size for i in range(size)])
                 for k in range(size)}
    reflections = {Permutation([(k - i) % size for i in range(size)])
                   for k in range(size)}
    return rotations, reflections


@pytest.mark.parametrize("size", range(2, 9))
def test_the_distance_of_a_swap_is_the_shorter_way_round(size):
    for low, high in itertools.combinations(range(size), 2):
        swap = Permutation(low, high, size=size)
        assert music.dist(swap) == min(high - low, size - (high - low))
    assert music.dist(Permutation(size=size)) == 0


@pytest.mark.parametrize("size", range(3, 8))
def test_rotations_and_mirrors_are_the_polygon_s(size):
    structures = InterestingPermutations(size)
    rotations, reflections = _around_the_circle(size)
    assert set(structures.rotations) == rotations
    assert set(structures.dihedral) == rotations | reflections
    assert len(structures.dihedral) == 2 * size
    assert set(structures.mirrors) == reflections - rotations
    assert structures.neutral_perm == Permutation(size=size)


@pytest.mark.parametrize("size", [4, 6])
def test_an_even_polygon_has_edge_and_vertex_mirrors(size):
    """An edge mirror moves every corner; a vertex mirror fixes two."""
    structures = InterestingPermutations(size)
    assert len(structures.edge_mirrors) == size // 2
    assert len(structures.vertex_mirrors) == size // 2
    assert all(mirror.length() == size
               for mirror in structures.edge_mirrors)
    assert all(mirror.length() == size - 2
               for mirror in structures.vertex_mirrors)
    assert set(structures.edge_mirrors + structures.vertex_mirrors) == set(
        structures.mirrors)


@pytest.mark.parametrize("size", [3, 5])
def test_an_odd_polygon_has_neither_kind_of_mirror(size):
    structures = InterestingPermutations(size)
    assert structures.edge_mirrors is None
    assert structures.vertex_mirrors is None


def test_two_elements_are_rounds_and_the_swap():
    structures = InterestingPermutations(2)
    assert structures.dihedral == [Permutation([0, 1]), Permutation([1, 0])]
    assert structures.mirrors == []
    assert structures.edge_mirrors == [] and structures.vertex_mirrors == []


@pytest.mark.parametrize("size", range(2, 6))
def test_every_permutation_is_grouped_by_how_many_it_moves(size):
    structures = InterestingPermutations(size)
    assert set(structures.permutations) == set(_all_permutations(size))
    assert len(structures.permutations_by_sizes) == size - 1
    for index, group in enumerate(structures.permutations_by_sizes):
        assert group == [permutation for permutation
                         in structures.permutations
                         if permutation.length() == index + 2]
        assert set(group) == _moving(size, index + 2)


@pytest.mark.parametrize("size", range(2, 7))
def test_the_swaps_are_every_transposition_in_three_orders(size):
    structures = InterestingPermutations(size)
    transpositions = _moving(size, 2)
    assert structures.swaps_as_comes == [
        permutation for permutation in structures.permutations
        if permutation.length() == 2]
    assert set(structures.swaps) == transpositions
    ranks = [swap.rank() for swap in structures.swaps]
    assert ranks == sorted(ranks, reverse=True)
    assert len(set(ranks)) == len(ranks)
    assert structures.neighbor_swaps == [
        Permutation(i, i + 1, size=size) for i in range(size - 1)]
    assert len(structures.swaps_by_stepsizes) == size // 2
    for step, group in enumerate(structures.swaps_by_stepsizes, start=1):
        assert group == [
            swap for swap in structures.swaps
            if min(swap.support()[1] - swap.support()[0],
                   size - swap.support()[1] + swap.support()[0]) == step]


@pytest.mark.parametrize("size", range(2, 7))
def test_the_alternations_outside_the_polygon_are_grouped_by_size(size):
    structures = InterestingPermutations(size)
    even = {permutation for permutation in _all_permutations(size)
            if permutation.is_even}
    assert set(structures.alternations) == even
    assert structures.alternations_complement == [
        permutation for permutation in structures.alternations
        if permutation not in structures.dihedral]
    assert not set(structures.alternations_complement) & set(
        structures.dihedral)
    assert len(structures.alternations_by_sizes) == size + 1
    for count, group in enumerate(structures.alternations_by_sizes):
        assert group == [permutation for permutation
                         in structures.alternations_complement
                         if permutation.length() == count]


@pytest.mark.parametrize("size", range(1, 6))
def test_parity_agrees_with_sympy_at_every_size(size):
    """Every size, odd ones included: counting each cycle's length twice
    changes the parity only when the size is odd."""
    structures = InterestingPermutations(2)
    for permutation in _all_permutations(size):
        expected = "even" if permutation.is_even else "odd"
        assert structures.even_odd(permutation.array_form) == expected


def test_parity_names_the_range_a_permutation_must_hold():
    with pytest.raises(ValueError, match=_exactly(
            "sequence must hold each of 0 to 2 once; got [0, 1, 1]")):
        InterestingPermutations(3).even_odd([0, 1, 1])


def test_the_refusals_say_exactly_what_was_wrong():
    with pytest.raises(ValueError, match=_exactly(
            "nelements must be at least 2; got 1. One element has no swap, "
            "mirror or second rotation")):
        InterestingPermutations(1)
    with pytest.raises(ValueError, match=_exactly(
            "a step of -2 moves point 1 below zero")):
        music.transpose_permutation(Permutation(1, 2, size=4), -2)


def test_peals_generate_in_the_order_their_method_gives():
    coset = music.Peals(4, method="coset")
    dimino = music.Peals(4, method="dimino")
    assert coset.permutations == InterestingPermutations(
        4, method="coset").permutations
    assert coset.permutations != dimino.permutations


# --------------------------------------------------------------------------
# Peals
# --------------------------------------------------------------------------

def test_a_new_generic_peal_holds_nothing_yet():
    peal = music.GenericPeal()
    assert peal.peals == {} and peal.acted_peals == {}
    assert peal.domain is None and peal.nelements is None


@pytest.mark.parametrize("method", ["act", "act_all"])
def test_a_generic_peal_says_why_it_cannot_act(method):
    peal = music.GenericPeal()
    call = (lambda: peal.act("rows")) if method == "act" else peal.act_all
    with pytest.raises(ValueError, match=_exactly(
            "no peals have been defined on this object yet")):
        call()
    peal.peals = {"rows": [Permutation([1, 0])]}
    with pytest.raises(ValueError, match=_exactly(
            "nelements has not been set, so no default domain can be "
            "built; pass domain explicitly")):
        call()


def test_the_named_peals_refuse_by_the_count_they_were_built_for():
    with pytest.raises(ValueError, match=_exactly(
            "the permutation acts on 3 elements but this Peals was built "
            "for 4; the default domain act() builds would not fit it")):
        music.Peals(4).transpositions_peal(Permutation([1, 0, 2]))
    with pytest.raises(ValueError, match=_exactly(
            "an_eight_and_forty is a peal on five bells, and this Peals "
            "was built for 4; the two whole hunts and the three bells "
            "that ring the six changes are five")):
        music.Peals(4).an_eight_and_forty()


@pytest.mark.parametrize("size", range(2, 7))
def test_each_row_is_the_change_before_it_applied_to_the_last(size):
    """peal_sequence holds the changes: each takes one row to the next,
    and the last brings the bells back into rounds."""
    peal = music.PlainChanges(size)
    rounds = Permutation(size=size)
    assert peal.neutral_perm == rounds
    assert peal.neighbor_swaps == [
        Permutation(i, i + 1, size=size) for i in range(size - 1)]
    assert len(peal.peal_sequence) == len(peal.peal_direct)
    assert all(change in peal.neighbor_swaps
               for change in peal.peal_sequence)
    rows = peal.peal_direct + [rounds]
    for before, change, after in zip(rows, peal.peal_sequence, rows[1:]):
        assert change * before == after
    assert peal.peals == {"peal_direct": peal.peal_direct,
                          "peal_sequence": peal.peal_sequence}


def test_a_new_plain_change_has_acted_on_nothing():
    peal = music.PlainChanges(4)
    assert peal.domain is None and peal.acted_peals is None


@pytest.mark.parametrize("size, count", [(4, 1), (5, 2), (6, 3)])
def test_the_hunts_start_at_the_lead_going_up(size, count):
    names = [f"hunt{i}" for i in range(count)]
    expected = {
        name: dict(level=i, position=i, status="started", direction="up",
                   next_=names[i + 1] if i + 1 < count else None)
        for i, name in enumerate(names)}
    assert music.PlainChanges(size).hunts == expected


@pytest.mark.parametrize("size", range(3, 8))
def test_every_extra_hunt_the_warning_allows_rings_the_same_peal(size):
    """Above the saturating count and below the number of bells."""
    saturating = music.PlainChanges.saturating_hunts(size)
    expected = music.PlainChanges(size).peal_direct
    for nhunts in range(saturating + 1, size):
        with pytest.warns(UserWarning, match=_exactly(
                f"peals are the same if there are {nhunts - saturating} "
                "hunts less")):
            crowded = music.PlainChanges(size, nhunts=nhunts)
        assert crowded.peal_direct == expected


@pytest.mark.parametrize("size", range(2, 8))
def test_as_many_hunts_as_bells_are_refused(size):
    """They passed the check and failed with an IndexError."""
    with pytest.raises(ValueError, match=_exactly(
            f"there must be fewer hunts than elements; got {size} for "
            f"{size}. The last hunt needs a bell to pass")):
        music.PlainChanges(size, nhunts=size)


def test_the_plain_changes_refusals_say_what_was_wrong():
    with pytest.raises(ValueError, match=_exactly(
            "nelements must be at least 2; got 1. A change swaps two "
            "bells")):
        music.PlainChanges(1)


def test_the_unread_hunts_warning_points_at_the_caller():
    with pytest.warns(UserWarning, match=_exactly(
            "PlainChanges does not read hunts; the hunts are laid out by "
            "nhunts")) as caught:
        music.PlainChanges(4, hunts={})
    # Not at plain_changes.py, where it would say nothing about the call.
    # The exact frame is not asserted: the mutation audit calls through a
    # trampoline, one frame deeper than a caller does.
    assert not caught[0].filename.endswith("plain_changes.py")


# --------------------------------------------------------------------------
# print_peal, with the colours on and off
# --------------------------------------------------------------------------

_COLOURS = ("yellow", "magenta", "green", "red", "blue", "white", "grey",
            "cyan")
_BACKGROUNDS = ("on_white", "on_blue", "on_red", "on_grey", "on_yellow",
                "on_magenta", "on_green", "on_cyan")


def test_a_peal_prints_one_row_a_line_without_colour(monkeypatch, capsys):
    module = import_module("music.structures.peals.peals")
    monkeypatch.setattr(module, "colored",
                        functools.partial(termcolor.colored, no_color=True))
    music.print_peal([[0, 1, 2, 3], [1, 0, 2, 3]])
    assert capsys.readouterr().out == "0123\n1023\n\n"


def test_each_bell_has_its_colour_and_a_hunt_its_background(monkeypatch,
                                                            capsys):
    """Bell i is drawn in the i-th colour, bold on white; a hunt is not
    bold, and has the i-th background counted from the end. The colours
    repeat from the ninth bell."""
    module = import_module("music.structures.peals.peals")
    forced = functools.partial(termcolor.colored, force_color=True)
    monkeypatch.setattr(module, "colored", forced)
    row = list(range(10))
    # Every background once: 9 takes the second's, 1 and 8 are not hunts.
    hunts = (0, 2, 3, 4, 5, 6, 7, 9)
    music.print_peal([row], hunts=hunts)
    expected = "".join(
        forced(i, _COLOURS[i % 8], _BACKGROUNDS[-(i % 8 + 1)])
        if i in hunts else
        forced(i, _COLOURS[i % 8], "on_white", ["bold"])
        for i in row) + "\n"
    assert capsys.readouterr().out == expected + "\n"


def test_the_structures_default_to_four_elements():
    assert InterestingPermutations().nelements == 4
    assert len(InterestingPermutations().permutations) == 24
    assert music.Peals().nelements == 4
    assert len(music.PlainChanges().peal_direct) == 24
    assert music.PlainChanges(6).initialize_hunts() == (
        music.PlainChanges(4).hunts)


def test_a_transposition_peal_is_kept_under_its_default_name():
    peals = music.Peals(4)
    changes = peals.transpositions_peal(Permutation([1, 2, 0, 3]))
    assert peals.peals == {"transposition_peal": changes}
