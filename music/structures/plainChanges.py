from . permutations import InterestingPermutations, transposePermutation
from percolation.rdf import c
from sympy.combinatorics import Permutation as p_

class PlainChanges(InterestingPermutations):
    """Present plain changes as swaps and act in domains to make peals
    
    http://www.gutenberg.org/files/18567/18567-h/18567-h.htm"""
    def __init__(self,nelements=4,hunts=None):
        InterestingPermutations.__init__(self,nelements)
        hunts=self.initializeHunts(nelements)
        hunts_=self.performPeal(nelements,dict(hunts)) # with the hunts, etc. 
        self.hunts=hunts
        #self.hunts_=hunts_
    def initializeHunts(self,nelements=4,nhunts=None):
        if not nhunts:
            if nelements>4:
                nhunts=2
            else:
                nhunts=1
        assert nelements>0
        if nhunts>nelements:
            raise ValueError("There cannot be more hunts than elements")
        elif nhunts>nelements-3:
            c("peals are the same if there are", nhunts-(nelements-3), "hunts less")
        hunts_dict={}
        for hunt in range(nhunts):
            # implement different starting settings here
            if hunt==nhunts-1:
                next_=None
            else:
                next_="hunt"+str(hunt+1)
            hunts_dict["hunt"+str(hunt)]=dict(level=hunt,position=hunt,status="started",direction="up",next_=next_)
        return hunts_dict
    def performPeal(self,nelements,hunts=None):
        if hunts==None:
            hunts=self.initializeHunts(nelements)
        permutation,hunts=self.performChange(nelements,hunts)
        #peal_sequence=[permutation]
        total_perm=permutation
        peal_direct=[self.neutral_perm]
        peal_sequence=[permutation]
        #c("prompt",permutation,total_perm)
        while total_perm!=self.neutral_perm:
            peal_direct+=[total_perm]
            permutation,hunts=self.performChange(nelements,hunts)
            #total_perm*=permutation
            total_perm=permutation*total_perm
            peal_sequence+=[permutation]
            #c("prompt",permutation,total_perm)
        self.peal_direct=peal_direct
        self.peal_sequence=peal_sequence
        return hunts
    def performChange(self,nelements,hunts,hunt=None):
        """Perform change procedure from 'hunt' on to subsequent hunts.
        
        Return permutation of the change and the hunts dictionary.
        Peals should be classified by restrictions satisfied by permutations
        between changes:
            1) canonical peal: only adjacent swaps allowed. E.g. plain changes, twenty all over.
            2) semi-canonical peal: only adjacent chunks are displaced, at least one permutation needs more than one swap. E.g.: rotations, mirrors.
            3) free peal: at least one permutation displaces non-adjacent indexes. E.g. paradox peal, phoenix peal, any nondihedral? """
        if hunt==None:
            hunt="hunt0"
        hunt_=hunts[hunt]
        direction=hunt_["direction"]
        assert direction in {"up","down"}
        position=hunt_["position"]
        swap_with=(position-1,position+1)[direction=="up"]
        c(position,swap_with,direction,nelements)
        if swap_with>=0 and swap_with<nelements: # move
            c("move")
            swap=self.neighbor_swaps[(position-1,position)[direction=="up"]]
            hunts[hunt]["position"]=swap_with
        else: # invert direction and move to next hunt and one less element
            c("invert")
            new_direction=("up","down")[direction=="up"]
            hunts[hunt]["direction"]=new_direction
            if hunt_["next_"]==None:
                c("there is no subsequent hunt")
                swap=self.neighbor_swaps[(0,-1)[new_direction=="up"]]
            else: # 
                c("there is a subsequent hunt")
                subsequent_hunt=hunt["next_"]
                swap,hunts=performChange(nelements-1,hunts,subsequent_hunt)
        return swap, hunts
    def act(self,domain=None):
        if domain==None:
            domain=list(range(self.nelements))
        return [i(domain) for i in self.peal_direct]
    def actAll(self,domain=None):
        if domain==None:
            domain=list(range(self.nelements))
        acted_peals={}
        for peal in self.peals:
            acted_peals[peal+"_acted"]=[i(domain) for i in self.peals[peal]]
        self.domain=domain
        self.acted_peals=acted_peals
    def transpositionsPeal(permutation,peal_name="transposition_peal"):
        self.peals[peal_name]=[sympy.combinatorics.Permutation(i) for i in permutation.transpositions()]


