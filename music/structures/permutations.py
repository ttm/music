from percolation.rdf import c
import sympy
from sympy import combinatorics

class InterestingPermutations:
    """

    mirrors are ordered by swaps (0,n-1..)
    """
    def __init__(self,n_elements=4,method="dimino"):
        c("started permutations with",n_elements,"elements")
        self.n_elements=n_elements
        self.method=method
        self.getRotations()
        self.getMirrors()
        self.getAlternating()
        self.getFullSymmetry()
        self.getSwaps()
        c("finished permutations with",n_elements,"elements")

    def getAlternating(self):
        self.alternations=list(sympy.combinatorics.named_groups.AlternatingGroup(self.n_elements).generate(method=self.method))
        self.alternations_complement=[i for i in self.alternations if i not in self.dihedral]
        length=3
        self.alternations_by_sizes=[]
        while length in [i.length() for i in self.alternations_complement]:
            self.alternations_by_sizes+=[[i for i in self.alternations_complement if i.length()==length]]
            length+=1
        assert len(self.alternations_complement)==sum([len(i) for i in self.alternations_by_sizes])

    def getRotations(self):
        """method dimino or coset"""
        self.rotations=list(sympy.combinatorics.named_groups.CyclicGroup(self.n_elements).generate(method=self.method))
    def getMirrors(self):
        self.dihedral=list(sympy.combinatorics.named_groups.DihedralGroup(self.n_elements).generate(method=self.method))
        self.mirrors=[i for i in self.dihedral if i not in self.rotations]
        if self.n_elements%2==0: # even elements have edge and vertex mirrors
            self.edge_mirrors=[i for i in   self.mirrors if i.length()==self.n_elements]
            self.vertex_mirrors=[i for i in self.mirrors if i.length()==self.n_elements-2]
            assert len(self.edge_mirrors+self.vertex_mirrors)==len(self.mirrors)
    def getSwaps(self):
        # contiguous swaps
        # swaps by distance between the indexes
        # indicate ordering of swaps to make peals
        # ascents, descents
        self.swaps=sorted(self.permutations_by_sizes[0], key=lambda x: -x.rank())
        self.swaps_as_comes=self.permutations_by_sizes[0]
        self.ascent_swaps=[i for i in self.swaps if i.support()[1]>i.support()[0]]
        self.descent_swaps=[i for i in self.swaps if i.support()[1]<i.support()[0]]
        self.swaps_by_stepsizes=[]
        dist_=1
        while dist_ in [dist(i) for i in self.swaps]:
            self.swaps_by_stepsizes+=[[i for i in self.swaps if dist(i)==dist_]]
            dist_+=1
    def evenOdd(self):
        # get even and odd permutations
        pass
    def getFullSymmetry(self):
        self.permutations=list(sympy.combinatorics.named_groups.SymmetricGroup(self.n_elements).generate(method=self.method))
        # sympy.combinatorics.generators.symmetric(self.n_elements)
        self.permutations_by_sizes=[]
        length=2
        while length in [i.length() for i in self.permutations]:
            self.permutations_by_sizes+=[[i for i in self.permutations if i.length()==length]]
            length+=1
def dist(swap):
    if swap.size%2==0:
        half=swap.size/2
    else:
        half=swap.size//2+1
    diff=abs(swap.support()[1]-swap.support()[0])
    if diff>=half:
        diff=swap.size-diff
    return diff


