"""
Present plain changes as swaps and act in domains to make peals.

Reference:
- http://www.gutenberg.org/files/18567/18567-h/18567-h.htm
"""

import sympy


class PlainChanges:
    """
    Presents plain changes as swaps and acts in domains to make peals.
    """

    def __init__(self, nelements=4, nhunts=None, hunts=None):
        """
        Initializes a PlainChanges object.

        Parameters:
            nelements (int, optional): The number of elements. Defaults to 4.
            nhunts (int, optional): The number of hunts. Defaults to None.
            hunts (dict, optional): The hunts dictionary. Defaults to None.

        Raises:
            ValueError: If the number of hunts is invalid.
        """
        self.peal_direct = None
        self.peal_sequence = None
        self.domain = None
        self.acted_peals = None
        self.peals = None
        hunts = self.initialize_hunts(nelements, nhunts)
        self.neutral_perm = sympy.combinatorics.Permutation([0],
                                                            size=nelements)
        self.neighbor_swaps = [
            sympy.combinatorics.Permutation(i, i + 1, size=nelements)
            for i in range(nelements - 1)]
        self.domains = []
        self.hunts = hunts
        self.nelements = nelements

    def initialize_hunts(self, nelements=4, nhunts=None):
        """
        Initializes the hunts dictionary.

        Parameters:
            nelements (int, optional): The number of elements. Defaults to 4.
            nhunts (int, optional): The number of hunts. Defaults to None.

        Returns:
            dict: The hunts dictionary.

        Raises:
            ValueError: If the number of hunts is invalid.
        """
        if not nhunts:
            if nelements > 4:
                nhunts = 2
            else:
                nhunts = 1
        assert nelements > 0
        if nhunts > nelements:
            raise ValueError("There cannot be more hunts than elements")
        elif nhunts > nelements - 3:
            print("peals are the same if there are", nhunts - (nelements - 3),
                  "hunts less")
        hunts_dict = {}
        for hunt in range(nhunts):
            if hunt == nhunts - 1:
                next_ = None
            else:
                next_ = "hunt" + str(hunt + 1)
            hunts_dict["hunt" + str(hunt)] = dict(level=hunt, position=hunt,
                                                  status="started",
                                                  direction="up", next_=next_)
        return hunts_dict

    def perform_peal(self, nelements, hunts=None):
        """
        Performs a peal.

        Parameters:
            nelements (int): The number of elements.
            hunts (dict, optional): The hunts dictionary. Defaults to None.

        Returns:
            dict: The updated hunts dictionary.
        """
        if hunts is None:
            hunts = self.initialize_hunts(nelements)
        permutation, hunts = self.perform_change(nelements, hunts)
        total_perm = permutation
        peal_direct = [self.neutral_perm]
        peal_sequence = [permutation]
        while total_perm != self.neutral_perm:
            peal_direct += [total_perm]
            permutation, hunts = self.perform_change(nelements, hunts)
            total_perm = permutation * total_perm
            peal_sequence += [permutation]
        self.peal_direct = peal_direct
        self.peal_sequence = peal_sequence
        return hunts

    def perform_change(self, nelements, hunts, hunt=None):
        """
        Performs a change procedure.

        Parameters:
            nelements (int): The number of elements.
            hunts (dict): The hunts dictionary.
            hunt (str, optional): The current hunt. Defaults to None.

        Returns:
            Permutation: The permutation of the change.
            dict: The updated hunts dictionary.
        """
        if hunt is None:
            hunt = "hunt0"
        hunt_ = hunts[hunt]
        direction = hunt_["direction"]
        assert direction in {"up", "down"}
        position = hunt_["position"]
        swap_with = (position - 1, position + 1)[direction == "up"]
        cut_bellow = sum([hunts["hunt" + str(i)]["direction"] == "up"
                          for i in range(hunt_["level"])])
        cut_above = nelements - (hunt_["level"] - cut_bellow)
        domain = list(range(nelements))[cut_bellow:cut_above]
        self.domains += [(domain, cut_bellow, cut_above, hunt_["level"], hunt,
                          position, swap_with)]
        if swap_with in domain:
            swap = self.neighbor_swaps[(position - 1, position)
                                       [direction == "up"]]
            for ahunt in hunts:
                if hunts[ahunt]["position"] == swap_with:
                    hunts[ahunt]["position"] = position
            hunts[hunt]["position"] = swap_with
        else:
            new_direction = ("up", "down")[direction == "up"]
            hunts[hunt]["direction"] = new_direction
            self.domains += ["invert", new_direction, hunt]
            if hunt_["next_"] is None:
                swap = self.neighbor_swaps[(domain[0], domain[-2])
                                           [new_direction == "up"]]
            else:
                subsequent_hunt = hunt_["next_"]
                swap, hunts = self.perform_change(nelements, hunts,
                                                  subsequent_hunt)
        self.domains += [swap]
        return swap, hunts

    def act(self, domain=None, peal=None):
        """
        Acts in a domain using a peal.

        Parameters:
            domain (list, optional): The domain. Defaults to None.
            peal (list, optional): The peal. Defaults to None.

        Returns:
            list: The acted domain.
        """
        if domain is None:
            domain = list(range(self.nelements))
        if peal is None:
            peal = self.peal_direct
        return [i(domain) for i in peal]

    def act_all(self, domain=None):
        """
        Acts in all peals using a domain.

        Parameters:
            domain (list, optional): The domain. Defaults to None.
        """
        if domain is None:
            domain = list(range(self.nelements))
        acted_peals = {}
        for peal in self.peals:
            acted_peals[peal + "_acted"] = [i(domain)
                                            for i in self.peals[peal]]
        self.domain = domain
        self.acted_peals = acted_peals
