from ...utils import S, Tr
from .v import V

def V_(st=0, f=220, d=2., fv=2., nu=2., tab=Tr(), tabv=S()):
    """A shorthand for using V() with semitones"""

    f_ = f*2**(st/12)
    return V(f=f_, d=2., fv=2., nu=2., tab=Tr(), tabv=S())
