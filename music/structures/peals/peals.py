from sympy.combinatorics import Permutation
from termcolor import colored
from colorama import init
from ..permutations import InterestingPermutations

init()


def print_peal(peal, hunts=[0, 1]):
    """
    Print peal with colored numbers. Hunt have also colored background

    TODO: documentation
    
    """
    reset = 'reset'
    # colors='black',
    # colors='redasd','green','yellow','blue','magenta','cyan'#,'white'
    colors = 'yellow', 'magenta', 'green', 'red', 'blue', 'white', 'grey', 'cyan'
    hcolors = 'on_white', 'on_blue', 'on_red', 'on_grey', 'on_yellow', 'on_magenta', 'on_green', 'on_cyan'
    final_string = ''
    for sequence in peal:
        final_string += ''.join(
            colored(i, colors[i], hcolors[-(i + 1)]) if i in hunts else colored(i, colors[i], "on_white", ["bold"]) for
            i in sequence) + '\n'
    print(final_string)


class Peals(InterestingPermutations):  # TODO
    """
    Use permutations to make peals and represent peals as permutations.

    Notes
    -----
    Core reference:
      - http://www.gutenberg.org/files/18567/18567-h/18567-h.htm

    Also check peal rules, such as conditions for trueness.
      - Wikipedia seemed ok last time.
    
    """

    def __init__(self):
        InterestingPermutations.__init__(self)
        self.peals = []
        self.transpositions_peal(self.peals["rotation_peal"][1])
        self.twenty_all_over()  # TODO
        self.an_eight_and_forty()  # TODO

    def transpositions_peal(self, permutation, peal_name="transposition_peal"):
        self.peals[peal_name] = [Permutation(i) for i in permutation.transpositions()]

    def twenty_all_over(self):
        pass

    def an_eight_and_forty(self):
        pass
