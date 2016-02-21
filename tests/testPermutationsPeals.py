import sys
keys=tuple(sys.modules.keys())
for key in keys:
    if "music" in key:
        del sys.modules[key]
import music as M
ip3=M.structures.symmetry.InterestingPermutations(3)
ip4=M.structures.symmetry.InterestingPermutations(4)
#ip5=M.structures.symmetry.InterestingPermutations(5)
#ip6=M.structures.symmetry.InterestingPermutations(6)
#ip7=M.structures.symmetry.InterestingPermutations(7)
#ip8=M.structures.symmetry.InterestingPermutations(8)
pe3=M.structures.symmetry.PlainChanges(3)
pe4=M.structures.symmetry.PlainChanges(4)
pe4=M.structures.symmetry.PlainChanges(5)


