from sympy.combinatorics import Permutation
from sympy.combinatorics.named_groups import AlternatingGroup
import sympy


class InterestingPermutations:
    """Get permutations of n elements in meaningfull sequences.
    mirrors are ordered by swaps (0,n-1...)"""

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
        self.alternations = list(AlternatingGroup(self.nelements).generate(method=self.method))
        self.alternations_complement = [i for i in self.alternations if i not in self.dihedral]
        length_max = self.nelements
        self.alternations_by_sizes = []
        for length in range(0, 1 + length_max):
            # while length in [i.length() for i in self.alternations_complement]:
            self.alternations_by_sizes.append(
                [i for i in self.alternations_complement if i.length() == length])

        assert len(self.alternations_complement) == sum([len(i) for i in self.alternations_by_sizes])

    def get_rotations(self):
        """method dimino or coset"""
        self.rotations = list(sympy.combinatorics.named_groups.CyclicGroup(self.nelements).generate(method=self.method))

    def get_mirrors(self):
        if self.nelements > 2:  # bug in sympy?
            self.dihedral = list(
                sympy.combinatorics.named_groups.DihedralGroup(self.nelements).generate(method=self.method))
        else:
            self.dihedral = [Permutation([0], size=self.nelements),
                             Permutation([1, 0], size=self.nelements)]
        self.mirrors = [i for i in self.dihedral if i not in self.rotations]
        if self.nelements % 2 == 0:  # even elements have edge and vertex mirrors
            self.edge_mirrors = [i for i in self.mirrors if i.length() == self.nelements]
            self.vertex_mirrors = [i for i in self.mirrors if i.length() == self.nelements - 2]
            assert len(self.edge_mirrors + self.vertex_mirrors) == len(self.mirrors)

    def get_swaps(self):
        # contiguous swaps
        # swaps by distance between the indexes
        self.swaps = sorted(self.permutations_by_sizes[0], key=lambda x: -x.rank())
        self.swaps_as_comes = self.permutations_by_sizes[0]
        self.swaps_by_stepsizes = []
        self.neighbor_swaps = [sympy.combinatorics.Permutation(i, i + 1, size=self.nelements) for i in
                               range(self.nelements - 1)]
        dist_ = 1
        while dist_ in [dist(i) for i in self.swaps]:
            self.swaps_by_stepsizes += [[i for i in self.swaps if dist(i) == dist_]]
            dist_ += 1

    def even_odd(self, sequence):
        # get even and odd permutations ?
        pass

    def get_full_symmetry(self):
        self.permutations = list(
            sympy.combinatorics.named_groups.SymmetricGroup(self.nelements).generate(method=self.method))
        # sympy.combinatorics.generators.symmetric(self.nelements)
        self.permutations_by_sizes = []
        length = 2
        while length in [i.length() for i in self.permutations]:
            self.permutations_by_sizes += [[i for i in self.permutations if i.length() == length]]
            length += 1


def dist(swap):
    if swap.size % 2 == 0:
        half = swap.size / 2
    else:
        half = swap.size // 2 + 1
    diff = abs(swap.support()[1] - swap.support()[0])
    if diff >= half:
        diff = swap.size - diff
    return diff


def transpose_permutation(permutation, step=1):
    if not step:
        return permutation
    new_indexes = (i + step for i in permutation.support())
    return sympy.combinatorics.Permutation(*new_indexes)
