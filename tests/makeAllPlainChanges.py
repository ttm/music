import sys
keys = tuple(sys.modules.keys())
for key in keys:
    if "music" in key:
        del sys.modules[key]
import music as M
from percolation.rdf import c

def fact(x):
    if x == 1:
        return 1
    return x*fact(x-1)


nelements = 0
while nelements not in range(3, 13):
    nelements_maximum = input("make changes until maximum number of elements:\
                    (min=3,,max=12,default=5) ")
    try:
        nelements = int(nelements_maximum)
    except:
        pass
    if not nelements_maximum:
        nelements_maximum = 5
# generate peals with elements in numbers of 3 to 12
peals = {}
for nelements in range(3, int(nelements_maximum)+1):
    key = "peal_with_" + str(nelements) + "_elements"
    nhunts=nelements-3
    peal = M.structures.symmetry.PlainChanges(nelements,nhunts)
    peals[key] = peal
    c(len(peal.peal_direct), fact(nelements))
    assert len(peal.peal_direct) == fact(nelements)
