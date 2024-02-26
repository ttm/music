class GenericPeal:
    def __init__(self):
        self.nelements = None
        self.peals = None
        self.acted_peals = None
        self.domain = None

    def act(self, peal, domain=None):
        if domain is None:
            domain = list(range(self.nelements))
        return [i(domain) for i in self.peals[peal]]

    def act_all(self, domain=None):
        if domain is None:
            domain = list(range(self.nelements))
        acted_peals = {}
        for peal in self.peals:
            acted_peals[peal+"_acted"] = [i(domain) for i in self.peals[peal]]
        self.domain = domain
        self.acted_peals = acted_peals
