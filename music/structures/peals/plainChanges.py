from ..permutations import InterestingPermutations, transposePermutation
import sympy
from sympy.combinatorics import Permutation

#class PlainChanges(InterestingPermutations):
class PlainChanges:
    """Present plain changes as swaps and act in domains to make peals

    http://www.gutenberg.org/files/18567/18567-h/18567-h.htm"""
    def __init__(self,nelements=4,nhunts=None,hunts=None):
        # InterestingPermutations.__init__(self,nelements)
        hunts = self.initializeHunts(nelements, nhunts)
        self.neutral_perm=sympy.combinatorics.Permutation([0],size=nelements)
        self.neighbor_swaps=[
            sympy.combinatorics.Permutation(
                i,i+1,size=nelements)
            for i in range(nelements-1)]
        self.domains=[]
        hunts_=self.performPeal(nelements,dict(hunts)) # with the hunts, etc.
        self.hunts=hunts
        self.nelements = nelements
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
            print("peals are the same if there are", nhunts-(nelements-3), "hunts less")
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
        while total_perm!=self.neutral_perm:
            peal_direct+=[total_perm]
            permutation,hunts=self.performChange(nelements,hunts)
            #total_perm*=permutation
            total_perm=permutation*total_perm
            peal_sequence+=[permutation]
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
        position_=position
        swap_with=(position-1,position+1)[direction=="up"]
        # find domain by iterating upper hunts
        cut_bellow=sum([hunts["hunt"+str(i)]["direction"]=="up" for i in range(hunt_["level"])])
        cut_above=nelements-(hunt_["level"]-cut_bellow)
        #cut_above=nelements-cut_bellow
        domain=list(range(nelements))[cut_bellow:cut_above]
        self.domains+=[(domain,cut_bellow,cut_above,hunt_["level"],hunt,position,swap_with)]
        if swap_with in domain: # move
            swap=self.neighbor_swaps[(position-1,position)[direction=="up"]]
            for ahunt in hunts:
                if hunts[ahunt]["position"]==swap_with:
                    hunts[ahunt]["position"]=position
            hunts[hunt]["position"]=swap_with
        else: # invert direction and move to next hunt and one less element
            new_direction=("up","down")[direction=="up"]
            hunts[hunt]["direction"]=new_direction
            self.domains+=["invert",new_direction, hunt]
            if hunt_["next_"]==None:
                #swap=self.neighbor_swaps[(domain[0],domain[-2+cut_bellow])[new_direction=="up"]]
                swap=self.neighbor_swaps[(domain[0],domain[-2])[new_direction=="up"]]
            else:
                subsequent_hunt=hunt_["next_"]
                swap,hunts=self.performChange(nelements,hunts,subsequent_hunt)
                #swap=transposePermutation(swap,(0,1)[new_direction=="up"])
        self.domains+=[swap]
        return swap, hunts
    def act(self,domain=None,peal=None):
        if domain==None:
            domain=list(range(self.nelements))
        if peal==None:
            peal=self.peal_direct
        return [i(domain) for i in peal]
    def actAll(self,domain=None):
        if domain==None:
            domain=list(range(self.nelements))
        acted_peals={}
        for peal in self.peals:
            acted_peals[peal+"_acted"]=[i(domain) for i in self.peals[peal]]
        self.domain=domain
        self.acted_peals=acted_peals
