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
pe5=M.structures.symmetry.PlainChanges(5)
M.structures.symmetry.printPeal(pe4.act(),[0])
pe6=M.structures.symmetry.PlainChanges(6,3)
pe7=M.structures.symmetry.PlainChanges(7,4)
pe8=M.structures.symmetry.PlainChanges(8,5)
pe9=M.structures.symmetry.PlainChanges(9,6)
#pe10=M.structures.symmetry.PlainChanges(10,7) # might take too long and halt system
#pe10=M.structures.symmetry.PlainChanges(11,7) # might take too long and halt system
#pe10=M.structures.symmetry.PlainChanges(12,7) # might take too long and halt system


