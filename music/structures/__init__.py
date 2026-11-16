"""Permutations, the groups they form, and the peals of change ringing.

Built from sympy's permutation groups, which ``pip install
'music[structures]'`` installs. The names import without it; building a
structure says what to install.
"""

from .symmetry import (
    dist,
    GenericPeal,
    InterestingPermutations,
    Peals,
    PlainChanges,
    print_peal,
    transpose_permutation
)
__all__ = [
    'dist',
    'InterestingPermutations',
    'transpose_permutation',
    'GenericPeal',
    'Peals',
    'PlainChanges',
    'print_peal',
]
