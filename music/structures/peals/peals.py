"""
Provides functions for generating and representing peals using permutations.
"""

from sympy.combinatorics import Permutation
from termcolor import colored
from colorama import init
from .base import GenericPeal
from ..permutations import InterestingPermutations

init()


def print_peal(peal, hunts=(0, 1)):
    """
    Prints a peal with colored numbers. Hunts have also colored background.

    Parameters
    ----------
    peal : list
        The peal to print.
    hunts : list, optional
        The indices of hunted elements. Defaults to
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


class Peals(InterestingPermutations, GenericPeal):
    """
    Uses permutations to make peals and represents peals as permutations.

    Holds peals by name, which is the model :class:`GenericPeal` provides
    ``act`` and ``act_all`` for -- ``act("some_peal", domain)``. That is a
    different operation from :meth:`PlainChanges.act`, which acts the one
    peal the object was built from and takes the domain first.

    Notes
    -----
    Core reference:
    - http://www.gutenberg.org/files/18567/18567-h/18567-h.htm

    Also check peal rules, such as conditions for trueness.
    - Wikipedia seemed ok last time.

    """

    def __init__(self, nelements=4, method="dimino"):
        """
        Initializes a Peals object.

        Parameters
        ----------
        nelements : int, optional
            How many elements the permutations act on, by default 4. This
            also sizes the default domain that `act` and `act_all` build.
        method : str, optional
            The generation method passed to InterestingPermutations.
        """
        InterestingPermutations.__init__(self, nelements=nelements,
                                         method=method)
        # A mapping of name -> list of permutations, which is what
        # GenericPeal.act and act_all index into.
        self.peals = {}
        # Base peals can be created here when implementations become available
        # self.transpositions_peal(self.peals["rotation_peal"][1])

    def transpositions_peal(self, permutation, peal_name="transposition_peal"):
        """Generates a peal from transpositions of a permutation.

        Parameters
        ----------
        permutation : Permutation
            The permutation to generate
            transpositions from.
        peal_name : str, optional
            The name of the peal. Defaults to
            "transposition_peal".

        Returns
        -------
        list
            The transpositions, as permutations over the same domain.

        Notes
        -----
        sympy's transpositions() yields index pairs, which are cycle
        notation rather than array form: Permutation((0, 1)) is the
        identity and Permutation((0, 2)) raises. The pairs are expanded
        with the original size, so composing them in reverse rebuilds
        the permutation they came from.

        """
        if permutation.size != self.nelements:
            raise ValueError(
                f"the permutation acts on {permutation.size} elements but "
                f"this Peals was built for {self.nelements}; the default "
                "domain act() builds would not fit it"
            )
        self.peals[peal_name] = [
            Permutation(*pair, size=permutation.size)
            for pair in permutation.transpositions()
        ]
        return self.peals[peal_name]

    def _rows_to_peal(self, rows, peal_name):
        """Store `rows` as a named peal of cumulative permutations.

        ``act`` applies each permutation to the domain, so a peal is held
        as the permutation that reaches each row from rounds rather than
        as the change that gets there from the row before.
        """
        self.peals[peal_name] = [Permutation(list(row)) for row in rows]
        return self.peals[peal_name]

    def twenty_all_over(self, peal_name="twenty_all_over"):
        """Ring the Twenty All Over.

        Every bell hunts up in turn, from the lead to behind the others,
        and the bell that inherits the lead hunts next. On five bells
        that is the twenty changes the peal is named for, and they bring
        the bells back into rounds [1]_:

            "every Bell hunts in order once through the Bells, until it
            comes behind them; and first the Treble hunts up, next the
            Second, and then the 3, 4 and 5, which brings the Bells
            round in their right places again, at the end of the Twenty
            Changes."

        Parameters
        ----------
        peal_name : string
            The key to store the peal under in ``peals``.

        Returns
        -------
        list of Permutation
            One permutation per row, from rounds onwards. The closing
            change back into rounds is implied rather than stored, as it
            is for :class:`music.PlainChanges`.

        Notes
        -----
        The rule holds for any number of bells, and the twenty is what
        it comes to on five: each of the ``n`` bells takes ``n - 1``
        changes to hunt from the lead to the back, so the peal is
        ``n * (n - 1)`` changes long and every row of it is distinct.
        The name is the five-bell case, which is the one Tintinnalogia
        prints and the one the tests check row for row.

        References
        ----------
        .. [1] Duckworth, Richard, and Fabian Stedman. *Tintinnalogia,
               or, The Art of Ringing*, 1668.
               https://www.gutenberg.org/files/18567/18567-h/18567-h.htm

        Examples
        --------
        >>> peal = Peals(nelements=5).twenty_all_over()
        >>> len(peal)
        20

        """
        count = self.nelements
        state = list(range(count))
        rows = [tuple(state)]
        for _ in range(count):
            lead = state[0]
            while state.index(lead) < count - 1:
                i = state.index(lead)
                state[i], state[i + 1] = state[i + 1], state[i]
                rows.append(tuple(state))
        # The last row is rounds again, which is where the peal started.
        return self._rows_to_peal(rows[:-1], peal_name)

    def an_eight_and_forty(self, peal_name="an_eight_and_forty"):
        """Ring An Eight and Forty, on five bells.

        The fifth and the fourth are both *whole hunts*. They take turns
        hunting down to the lead and back up to the back, and each time
        one of them lies at the lead a single change is made among the
        other three, which run the six changes between them [1]_:

            "the Fifth and Fourth are both whole Hunts, each of which
            does hunt down before the Bells by turns, and lies there
            twice together and then hunts up again: The 1, 2 and 3 goes
            the six changes, one of which is made every time, either of
            the whole Hunts lies before the Bells."

        Six changes among three bells, one for each visit to the lead,
        with seven hunting changes between one visit and the next: forty
        eight changes, and back into rounds.

        Parameters
        ----------
        peal_name : string
            The key to store the peal under in ``peals``.

        Returns
        -------
        list of Permutation
            One permutation per row, from rounds onwards, forty eight of
            them. The closing change back into rounds is implied.

        Raises
        ------
        ValueError
            If this object was not built for five bells. Unlike
            :meth:`twenty_all_over`, this is a composition for a
            particular number of bells rather than a rule that holds for
            any: the two whole hunts and the three bells ringing the six
            changes between them are five, and the forty eight is what
            that arrangement comes to.

        Notes
        -----
        The three bells that are not hunts ring the plain changes on
        three, which is why the six changes are the same six
        :class:`music.PlainChanges` produces for that many elements.

        The implementation is the rule above rather than the table, and
        the test checks that it reproduces Tintinnalogia's forty eight
        rows exactly, in order.

        References
        ----------
        .. [1] Duckworth, Richard, and Fabian Stedman. *Tintinnalogia,
               or, The Art of Ringing*, 1668.
               https://www.gutenberg.org/files/18567/18567-h/18567-h.htm

        Examples
        --------
        >>> peal = Peals(nelements=5).an_eight_and_forty()
        >>> len(peal)
        48

        """
        count = self.nelements
        if count != 5:
            raise ValueError(
                "an_eight_and_forty is a peal on five bells, and this "
                f"Peals was built for {count}; the two whole hunts and "
                "the three bells that ring the six changes are five"
            )
        state = list(range(count))
        rows = [tuple(state)]
        hunts = (count - 1, count - 2)  # the fifth and the fourth
        turn = 0
        trio_pair = 0

        def change(index):
            state[index], state[index + 1] = state[index + 1], state[index]
            rows.append(tuple(state))

        # Ring until it comes round, which is what ends a peal, rather
        # than until a count is reached. The forty eight falls out: the
        # sixth six-change is followed by the ascent that restores
        # rounds partway through, so the last cycle is cut short by the
        # peal being over rather than by arithmetic.
        rounds = tuple(state)
        while rounds not in rows[1:]:
            hunt = hunts[turn % 2]
            while state.index(hunt) > 0:            # down before the bells
                change(state.index(hunt) - 1)
            change(1 + trio_pair)                   # one of the six changes
            trio_pair = 1 - trio_pair
            while state.index(hunt) < count - 1:
                change(state.index(hunt))           # and up again
            turn += 1
        return self._rows_to_peal(rows[:rows.index(rounds, 1)], peal_name)
