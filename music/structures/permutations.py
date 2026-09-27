"""Provides tools for working with interesting permutations.

This module defines the `InterestingPermutations` class, which facilitates the
generation and manipulation of permutations with specific properties. It also
includes utility functions for permutation operations.

Classes in this module:

* ``InterestingPermutations`` -- Provides tools for generating and manipulating
* permutations with specific properties.

Functions in this module:

* ``dist`` -- Calculates the distance between elements in a swap permutation.
* ``transpose_permutation`` -- Transposes a permutation by a specified step.

Examples
--------
To work with interesting permutations:

>>> from sympy.combinatorics import Permutation
>>> from sympy.combinatorics.named_groups import AlternatingGroup
>>> interesting_perms = InterestingPermutations(nelements=4,
...                                             method="dimino")
>>> len(interesting_perms.alternations)
12

"""
from sympy.combinatorics import Permutation
from sympy.combinatorics.named_groups import AlternatingGroup
import sympy


class InterestingPermutations:
    """Get permutations of n elements in meaningful sequences.
    Mirrors are ordered by swaps (0,n-1...).

    Parameters
    ----------
    nelements : integer
        How many elements are permuted. Every group generated here --
        symmetric, alternating, cyclic, dihedral -- is the group on this
        many elements, so it also sets how many permutations there are.
    method : string
        The generation method handed to sympy's group ``generate()``:
        ``"dimino"`` or ``"coset"``. It changes the order the
        permutations come out in, which is the sequence this class
        exists to make meaningful, and not which permutations they are.

    Raises
    ------
    ValueError
        If `nelements` is less than two, or `method` is neither of the
        two sympy generates by. One element has no swaps, mirrors or
        second rotation, and failed inside sympy; an unknown method
        failed there as ``NotImplementedError``.

    Methods
    -------
    get_alternating
        Generates permutations in the alternating group.
    get_rotations
        Generates rotations of permutations.
    get_mirrors
        Generates mirror permutations.
    get_swaps
        Generates swap permutations.
    even_odd
        Determines if a permutation is even or odd.
    get_full_symmetry
        Generates permutations with full symmetry.


    Examples
    --------
    >>> structures = InterestingPermutations(nelements=4)
    >>> len(structures.rotations), len(structures.dihedral)
    (4, 8)
    >>> rows = [p(list("abcd")) for p in structures.rotations]
    >>> ["".join(row) for row in rows]
    ['abcd', 'bcda', 'cdab', 'dabc']
    """
    # Populated by the get_* methods that __init__ calls. vertex_mirrors and
    # edge_mirrors stay None for an odd number of elements, which have
    # neither, so those keep their placeholder.
    permutations_by_sizes: list
    permutations: list
    alternations: list
    alternations_complement: list
    alternations_by_sizes: list
    neighbor_swaps: list
    swaps_by_stepsizes: list
    swaps_as_comes: list
    swaps: list
    rotations: list
    mirrors: list
    dihedral: list

    def __init__(self, nelements=4, method="dimino"):
        if nelements < 2:
            raise ValueError(
                f"nelements must be at least 2; got {nelements}. One "
                "element has no swap, mirror or second rotation")
        if method not in ("dimino", "coset"):
            raise ValueError(
                f"method must be 'dimino' or 'coset'; got {method!r}")
        self.vertex_mirrors = None
        self.edge_mirrors = None
        self.nelements = nelements
        self.neutral_perm = Permutation(size=nelements)
        self.method = method
        self.get_rotations()
        self.get_mirrors()
        self.get_alternating()
        self.get_full_symmetry()
        self.get_swaps()

    def get_alternating(self):
        """Generates permutations in the alternating group.

        This method generates permutations in the alternating group of the
        specified size using the provided generation method.
        """
        # sympy builds the alternating group of two elements on one point,
        # so its rounds matched nothing in the dihedral group and was
        # counted outside it. Rounds is the one even permutation of two.
        if self.nelements > 2:
            self.alternations = list(AlternatingGroup(self.nelements).
                                     generate(method=self.method))
        else:
            self.alternations = [Permutation([0, 1])]
        self.alternations_complement = [i for i in self.alternations
                                        if i not in self.dihedral]
        length_max = self.nelements
        self.alternations_by_sizes = []
        for length in range(length_max + 1):
            # while length in [i.length()
            #                  for i in self.alternations_complement]:
            self.alternations_by_sizes.append(
                [i for i in self.alternations_complement
                 if i.length() == length])

        assert len(self.alternations_complement) ==\
            sum([len(i)for i in self.alternations_by_sizes])

    def get_rotations(self):
        """Generates rotations of permutations.

        This method generates rotations of permutations of the specified size
        using the provided generation method.
        """
        self.rotations = list(sympy.combinatorics.named_groups.
                              CyclicGroup(self.nelements).
                              generate(method=self.method))

    def get_mirrors(self):
        """Generates mirror permutations.

        This method generates mirror permutations of the specified size using
        the provided generation method.
        """
        # sympy builds the dihedral group of two elements on four points,
        # as the Klein four-group, rather than on two, so the pair is
        # written out: rounds and the swap.
        if self.nelements > 2:
            self.dihedral = list(sympy.combinatorics.named_groups.
                                 DihedralGroup(self.nelements).
                                 generate(method=self.method))
        else:
            self.dihedral = [Permutation([0, 1]), Permutation([1, 0])]
        self.mirrors = [i for i in self.dihedral if i not in self.rotations]
        # even elements have edge and vertex mirrors
        if self.nelements % 2 == 0:
            self.edge_mirrors = [i for i in self.mirrors
                                 if i.length() == self.nelements]
            self.vertex_mirrors = [i for i in self.mirrors
                                   if i.length() == self.nelements - 2]
            assert len(self.edge_mirrors + self.vertex_mirrors) ==\
                len(self.mirrors)

    def get_swaps(self):
        """Generates swap permutations.

        This method generates swap permutations of the specified size using
        the provided generation method.
        """
        self.swaps = sorted(self.permutations_by_sizes[0],
                            key=lambda x: -x.rank())
        self.swaps_as_comes = self.permutations_by_sizes[0]
        self.swaps_by_stepsizes = []
        self.neighbor_swaps = [sympy.combinatorics.
                               Permutation(i, i + 1, size=self.nelements)
                               for i in range(self.nelements - 1)]
        dist_ = 1
        while dist_ in [dist(i) for i in self.swaps]:
            self.swaps_by_stepsizes += [[i for i in self.swaps
                                         if dist(i) == dist_]]
            dist_ += 1

    def even_odd(self, sequence):
        """Determines if a permutation is even or odd.

        This method determines if a given permutation is even or odd based on
        its sequence of elements.

        Parameters
        ----------
        sequence : list
            The sequence of elements representing the
            permutation.

        Returns
        -------
        str
            Either 'even' or 'odd' indicating the parity of the
            permutation.

        Raises
        ------
        ValueError
            If `sequence` is not a permutation of ``0`` to ``n - 1``. A
            repeated entry used to be read as a cycle, so ``[1, 1]`` was
            'odd'.

        """
        n = len(sequence)
        if sorted(sequence) != list(range(n)):
            raise ValueError(
                f"sequence must hold each of 0 to {n - 1} once; got "
                f"{list(sequence)}")
        visited = [False] * n
        parity = 0

        for i in range(n):
            if not visited[i]:
                cycle_length = 0
                x = i

                while not visited[x]:
                    visited[x] = True
                    x = sequence[x]
                    cycle_length += 1

                # At least one, always: the loop above is entered only
                # when visited[x] is False and its body runs before the
                # condition is tested again. The guard that used to be
                # here could not be False, which is why coverage never
                # reached its other side.
                parity += cycle_length - 1

        return 'even' if parity % 2 == 0 else 'odd'

    def get_full_symmetry(self):
        """Generates permutations with full symmetry.

        This method generates permutations with full symmetry of the specified
        size using the provided generation method.
        """
        self.permutations = list(sympy.combinatorics.named_groups.
                                 SymmetricGroup(self.nelements).
                                 generate(method=self.method))
        # sympy.combinatorics.generators.symmetric(self.nelements)
        self.permutations_by_sizes = []
        length = 2
        while length in [i.length() for i in self.permutations]:
            self.permutations_by_sizes += [[i for i in self.permutations
                                            if i.length() == length]]
            length += 1


def dist(swap):
    """
    Computes the cyclic distance between the two elements of a permutation.

    Parameters
    ----------
    swap : sympy.combinatorics.Permutation
        A permutation object with exactly two elements in its support.

    Returns
    -------
    int
        The cyclic distance between the two elements.

    Notes
    -----
    The elements are read as points on a circle of ``swap.size``, so the
    distance is the shorter way round: the difference between the two, or
    the size less it, whichever is smaller.

    This measures the two lowest displaced positions, so it is meaningful for
    a transposition. For a permutation with a larger support the remaining
    displaced positions are ignored. The identity displaces nothing and gives
    zero.

    Examples
    --------
    >>> from sympy.combinatorics import Permutation
    >>> perm = Permutation([1, 0, 2])
    >>> dist(perm)
    1
    >>> perm = Permutation([2, 0, 1])
    >>> dist(perm)
    1
    >>> dist(Permutation([0, 1, 2]))  # the identity displaces nothing
    0
    """
    support = swap.support()
    if len(support) < 2:
        # The identity moves nothing, so there is no pair to measure
        # between. Any other permutation displaces at least two elements.
        return 0
    # The support is sorted, so this is the difference going up.
    diff = support[1] - support[0]
    return min(diff, swap.size - diff)


def transpose_permutation(permutation, step=1):
    """
    Shifts a permutation up or down by `step` positions.

    Where `permutation` sends ``i`` to ``j``, the result sends
    ``i + step`` to ``j + step``: every cycle keeps its order and its
    direction, only moved along. A swap of neighbours becomes the swap of
    the next pair, which is how a change moves along a row of bells.

    Parameters
    ----------
    permutation : sympy.combinatorics.Permutation
        The permutation to be transposed.
    step : int, optional
        The number of positions to shift each element of the permutation,
        by default 1. Negative shifts it down.

    Returns
    -------
    sympy.combinatorics.Permutation
        The shifted permutation, on as many elements as `permutation` or
        more if the shifted points need them, so that it still acts on
        the domain the original did.

    Raises
    ------
    ValueError
        If `step` would move a point below zero.

    Notes
    -----
    If `step` is 0, the function returns the original permutation.

    It used to return one cycle through the moved points in ascending
    order, which is right only for a single swap: ``(0 2 1)``
    came back as ``(1 2 3)``, running the other way, and two swaps
    ``(0 1)(2 3)`` as the four-cycle ``(1 2 3 4)``. It was also sized to
    the highest point, so a shifted swap of four bells could not act on a
    row of four.

    Examples
    --------
    >>> from sympy.combinatorics import Permutation
    >>> perm = Permutation([2, 0, 1])
    >>> transpose_permutation(perm, 1).array_form
    [0, 3, 1, 2]
    >>> transpose_permutation(perm, 0).array_form
    [2, 0, 1]
    >>> transpose_permutation(Permutation(0, 1, size=4)).array_form
    [0, 2, 1, 3]
    """
    if not step:
        return permutation
    support = permutation.support()
    if not support:
        return permutation
    if support[0] + step < 0:
        raise ValueError(
            f"a step of {step} moves point {support[0]} below zero")
    cycles = [[point + step for point in cycle]
              for cycle in permutation.cyclic_form]
    # sympy grows the size to fit the highest point, and keeps this one
    # when the shifted points fit inside it.
    return sympy.combinatorics.Permutation(cycles, size=permutation.size)
