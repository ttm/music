"""Provides tools for working with interesting permutations.

This module defines the `InterestingPermutations` class, which facilitates the
generation and manipulation of permutations with specific properties. It also
includes utility functions for permutation operations.

Classes:
    - InterestingPermutations: Provides tools for generating and manipulating
    permutations with specific properties.

Functions:
    - dist: Calculates the distance between elements in a swap permutation.
    - transpose_permutation: Transposes a permutation by a specified step.

Example:
    To work with interesting permutations:

    >>> from sympy.combinatorics import Permutation
    >>> from sympy.combinatorics.named_groups import AlternatingGroup
    >>> interesting_perms = InterestingPermutations(nelements=4,
    >>>                                             method="dimino")
    >>> print(interesting_perms.alternations)

"""
from sympy.combinatorics import Permutation
from sympy.combinatorics.named_groups import AlternatingGroup
import sympy


class InterestingPermutations:
    """Get permutations of n elements in meaningful sequences.
    Mirrors are ordered by swaps (0,n-1...).

    Methods:
        - get_alternating: Generates permutations in the alternating group.
        - get_rotations: Generates rotations of permutations.
        - get_mirrors: Generates mirror permutations.
        - get_swaps: Generates swap permutations.
        - even_odd: Determines if a permutation is even or odd.
        - get_full_symmetry: Generates permutations with full symmetry.
    """
    def __init__(self, nelements=4, method="dimino"):
        self.permutations_by_sizes = None
        self.permutations = None
        self.neighbor_swaps = None
        self.swaps_by_stepsizes = None
        self.swaps_as_comes = None
        self.vertex_mirrors = None
        self.edge_mirrors = None
        self.swaps = None
        self.rotations = None
        self.mirrors = None
        self.dihedral = None
        self.alternations_by_sizes = None
        self.alternations_complement = None
        self.alternations = None
        self.nelements = nelements
        self.neutral_perm = Permutation([0], size=nelements)
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
        self.alternations = list(AlternatingGroup(self.nelements).
                                 generate(method=self.method))
        self.alternations_complement = [i for i in self.alternations
                                        if i not in self.dihedral]
        length_max = self.nelements
        self.alternations_by_sizes = []
        for length in range(0, 1 + length_max):
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
        if self.nelements > 2:  # bug in sympy?
            self.dihedral = list(sympy.combinatorics.named_groups.
                                 DihedralGroup(self.nelements).
                                 generate(method=self.method))
        else:
            self.dihedral = [Permutation([0], size=self.nelements),
                             Permutation([1, 0], size=self.nelements)]
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

        Parameters:
            sequence (list): The sequence of elements representing the
                             permutation.

        Returns:
            str: Either 'even' or 'odd' indicating the parity of the
                 permutation.
        """
        n = len(sequence)
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

                if cycle_length > 0:
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
    The distance is adjusted to account for the circular nature of the
    permutation. If the difference is greater than or equal to half the size
    of the permutation, the distance is calculated as the size of the
    permutation minus the difference.

    Examples
    --------
    >>> from sympy.combinatorics import Permutation
    >>> perm = Permutation([1, 0, 2])
    >>> dist(perm)
    1
    >>> perm = Permutation([2, 0, 1])
    >>> dist(perm)
    1
    """
    if swap.size % 2 == 0:
        half = swap.size / 2
    else:
        half = swap.size // 2 + 1
    diff = abs(swap.support()[1] - swap.support()[0])
    if diff >= half:
        diff = swap.size - diff
    return diff


def transpose_permutation(permutation, step=1):
    """
    Transposes (shifts) the elements of a permutation by a given step.

    Parameters
    ----------
    permutation : sympy.combinatorics.Permutation
        The permutation to be transposed.
    step : int, optional
        The number of positions to shift each element of the permutation,
        by default 1.

    Returns
    -------
    sympy.combinatorics.Permutation
        A new permutation with elements shifted by the specified step.

    Notes
    -----
    If `step` is 0, the function returns the original permutation.

    Examples
    --------
    >>> from sympy.combinatorics import Permutation
    >>> perm = Permutation([2, 0, 1])
    >>> transpose_permutation(perm, 1)
    Permutation([3, 1, 2])
    >>> transpose_permutation(perm, 0)
    Permutation([2, 0, 1])
    """
    if not step:
        return permutation
    new_indexes = (i + step for i in permutation.support())
    return sympy.combinatorics.Permutation(*new_indexes)
