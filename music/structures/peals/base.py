class GenericPeal:
    """A collection of named peals that can be acted on a domain.

    The base for the *named-peal* model: `peals` maps a name to a list of
    permutations, and `act` applies one of them by name. :class:`Peals`
    works that way and inherits this.

    :class:`music.PlainChanges` deliberately does not. It is built from one
    peal rather than holding several, so its `act` acts that peal and takes
    the domain first -- ``act(domain)`` rather than ``act(name, domain)``.
    The two are different operations that happen to share a verb.

    Attributes
    ----------
    nelements : int
        The number of elements in the domain.
    peals : dict
        A dictionary containing the peals and their
        corresponding actions.
    acted_peals : dict
        A dictionary containing the acted peals and their
        results.
    domain : list
        The domain on which the peals are acted.

    Methods
    -------
    act
        Acts a specific peal on the specified domain.
    act_all
        Acts all peals on the specified domain.

    """

    #: name -> list of permutations. Empty rather than None: a collection
    #: with nothing in it is what an unpopulated one is, and act() refuses
    #: to run on either.
    peals: dict
    acted_peals: dict

    def __init__(self):
        """Initializes a GenericPeal object."""
        self.nelements = None
        self.peals = {}
        self.acted_peals = {}
        self.domain = None

    def act(self, peal, domain=None):
        """Acts a specific peal on the specified domain.

        Parameters
        ----------
        peal : str
            The name of the peal to act.
        domain : list, optional
            The domain on which to act the peal.
            Defaults to None.

        Returns
        -------
        list
            The result of acting the peal on the specified domain.

        """
        if not self.peals:
            raise ValueError(
                "no peals have been defined on this object yet"
            )
        if peal not in self.peals:
            raise KeyError(
                f"no peal named {peal!r}; defined: {sorted(self.peals)}"
            )
        if domain is None:
            if self.nelements is None:
                raise ValueError(
                    "nelements has not been set, so no default domain can "
                    "be built; pass domain explicitly"
                )
            domain = list(range(self.nelements))
        return [i(domain) for i in self.peals[peal]]

    def act_all(self, domain=None):
        """Acts all peals on the specified domain.

        Parameters
        ----------
        domain : list, optional
            The domain on which to act the peals.
            Defaults to None.

        """
        if not self.peals:
            raise ValueError(
                "no peals have been defined on this object yet"
            )
        if domain is None:
            if self.nelements is None:
                raise ValueError(
                    "nelements has not been set, so no default domain can "
                    "be built; pass domain explicitly"
                )
            domain = list(range(self.nelements))
        acted_peals = {}
        for peal in self.peals:
            acted_peals[peal+"_acted"] = [i(domain) for i in self.peals[peal]]
        self.domain = domain
        self.acted_peals = acted_peals
