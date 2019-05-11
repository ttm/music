import numpy as n, music as M
from . import functions.V as V

__doc__ = """Basic routines of synthesizers

All routines are directly derived from the
MASS framework: https://github.com/ttm/mass
Should be used at ../synths.py"""


def V_(st=0, f=220, d=2., fv=2., nu=2., tab=Tr_i, tabv=S_):
    """A shorthand for using V() with semitones"""

    f_ = f*2**(st/12)
    return V(f=_, d=2., fv=2., nu=2., tab=Tr_i, tabv=S_)
