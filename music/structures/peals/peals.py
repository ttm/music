"""
Provides functions for generating and representing peals using permutations.
"""

from sympy.combinatorics import Permutation
from termcolor import colored
from colorama import init
from ..permutations import InterestingPermutations

init()


def print_peal(peal, hunts=[0, 1]):
    """
    Prints a peal with colored numbers. Hunts have also colored background.

    Parameters:
        peal (list): The peal to print.
        hunts (list, optional): The indices of hunted elements. Defaults to
                                [0, 1].
    """
    colors = 'yellow', 'magenta', 'green', 'red', 'blue', 'white', 'grey', \
        'cyan'
    hcolors = 'on_white', 'on_blue', 'on_red', 'on_grey', 'on_yellow', \
        'on_magenta', 'on_green', 'on_cyan'
    final_string = ''
    for sequence in peal:
        final_string += ''.join(
            colored(i, colors[i], hcolors[-(i + 1)]) if i in hunts else
            colored(i, colors[i], "on_white", ["bold"]) for i in sequence) + \
            '\n'
    print(final_string)


class Peals(InterestingPermutations):
    """
    Uses permutations to make peals and represents peals as permutations.

    Notes:
        Core reference:
        - http://www.gutenberg.org/files/18567/18567-h/18567-h.htm

        Also check peal rules, such as conditions for trueness.
        - Wikipedia seemed ok last time.
    """

    def __init__(self):
        """
        Initializes a Peals object.
        """
        InterestingPermutations.__init__(self)
        self.peals = []
        self.transpositions_peal(self.peals["rotation_peal"][1])
        self.twenty_all_over()  # TODO
        self.an_eight_and_forty()  # TODO

    def transpositions_peal(self, permutation, peal_name="transposition_peal"):
        """Generates a peal from transpositions of a permutation.

        Parameters:
            permutation (Permutation): The permutation to generate
                                       transpositions from.
            peal_name (str, optional): The name of the peal. Defaults to
                                       "transposition_peal".
        """
        self.peals[peal_name] = [Permutation(i)
                                 for i in permutation.transpositions()]

    def twenty_all_over(self):
        """
        TODO
        """
        pass

    def an_eight_and_forty(self):
        """
        TODO
        """
        pass
