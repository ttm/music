class GenericPeal:
    """_summary_
    """
    def __init__(self):
        self.nelements = None
        self.peals = None
        self.acted_peals = None
        self.domain = None

    def act(self, peal, domain=None):
        """_summary_

        Parameters
        ----------
        peal : _type_
            _description_
        domain : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """
        if domain is None:
            domain = list(range(self.nelements))
        return [i(domain) for i in self.peals[peal]]

    def act_all(self, domain=None):
        """_summary_

        Parameters
        ----------
        domain : _type_, optional
            _description_, by default None
        """
        if domain is None:
            domain = list(range(self.nelements))
        acted_peals = {}
        for peal in self.peals:
            acted_peals[peal+"_acted"] = [i(domain) for i in self.peals[peal]]
        self.domain = domain
        self.acted_peals = acted_peals
