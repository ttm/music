class GenericPeal:
    """Represents a generic peal.

    Attributes:
        nelements (int): The number of elements in the domain.
        peals (dict): A dictionary containing the peals and their
                      corresponding actions.
        acted_peals (dict): A dictionary containing the acted peals and their
                            results.
        domain (list): The domain on which the peals are acted.

    Methods:
        - act: Acts a specific peal on the specified domain.
        - act_all: Acts all peals on the specified domain.
    """

    def __init__(self):
        """Initializes a GenericPeal object."""
        self.nelements = None
        self.peals = None
        self.acted_peals = None
        self.domain = None

    def act(self, peal, domain=None):
        """Acts a specific peal on the specified domain.

        Parameters:
            peal (str): The name of the peal to act.
            domain (list, optional): The domain on which to act the peal.
                                     Defaults to None.

        Returns:
            list: The result of acting the peal on the specified domain.
        """
        if domain is None:
            domain = list(range(self.nelements))
        return [i(domain) for i in self.peals[peal]]

    def act_all(self, domain=None):
        """Acts all peals on the specified domain.

        Parameters:
            domain (list, optional): The domain on which to act the peals.
                                     Defaults to None.
        """
        if domain is None:
            domain = list(range(self.nelements))
        acted_peals = {}
        for peal in self.peals:
            acted_peals[peal+"_acted"] = [i(domain) for i in self.peals[peal]]
        self.domain = domain
        self.acted_peals = acted_peals
