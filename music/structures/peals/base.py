class GenericPeal:
    def act(self,peal,domain=None):
        if domain==None:
            domain=list(range(self.nelements))
        return [i(domain) for i in self.peals[peal]]
    def actAll(self,domain=None):
        if domain==None:
            domain=list(range(self.nelements))
        acted_peals={}
        for peal in self.peals:
            acted_peals[peal+"_acted"]=[i(domain) for i in self.peals[peal]]
        self.domain=domain
        self.acted_peals=acted_peals

