import numpy as n, pylab as p
class Basic:
    """Provide primary tables for lookup

    create sine, triangle, square and saw wave periods
    with size samples.
    """
    def __init__(self,size=2048):
        self.size=size
        self.makeTables(size)
    def makeTables(self,size):
        self.sine=n.sin(n.linspace(0,2*n.pi,size,endpoint=False))
        self.saw=n.linspace(-1,1,size)
        self.square=n.hstack( ( n.ones(size//2)*-1 , n.ones(size//2) )  )
        foo=n.linspace(-1,1,size/2,endpoint=False)
        self.triangle=n.hstack(  ( foo , foo*-1 )   )
    def drawTables(self):
        p.plot(self.sine,    "-o")
        p.plot(self.saw,     "-o")
        p.plot(self.square,  "-o")
        p.plot(self.triangle,"-o")
        p.xlim(-self.size*0.1,self.size*1.1)
        p.ylim(-1.1,1.1)
        p.show()


