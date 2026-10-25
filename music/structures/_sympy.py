"""sympy, which ``pip install music`` leaves out.

The permutation groups and change-ringing peals here are built from
sympy's, and sympy, with the mpmath it needs, is about 74 MB that most of
this package's users never touch: it was more than half of what
``pip install music`` installed. So it comes with the ``structures``
extra, and these modules import it when a structure is built rather than
when they are, which keeps every name importable -- ``from music import
*`` included -- on an install without it.
"""

from types import ModuleType


def combinatorics() -> ModuleType:
    """`sympy.combinatorics`, its named groups loaded.

    Raises
    ------
    ImportError
        If sympy is not installed, saying how to install it. Anything
        else that goes wrong importing it is raised as it is: a sympy
        that is there and broken is not fixed by installing it.
    """
    try:
        import sympy.combinatorics.named_groups
    except ModuleNotFoundError as exc:
        # The module Python could not find: `sympy` when it is not
        # installed, or one of its submodules when `sympy` is there but is
        # not a package -- what a None in sys.modules makes of it. Any
        # other name is a dependency of sympy's.
        if (exc.name or "").partition(".")[0] != "sympy":
            raise
        raise ImportError(
            "the permutations and peals are built from sympy's permutation "
            "groups: pip install 'music[structures]'") from exc
    return sympy.combinatorics
