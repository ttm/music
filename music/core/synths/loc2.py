import numpy as n
from .n import N

def loc2(sonic_vector=N(), theta1=90, theta2=0, dist1=.1,
        dist2=.1, zeta=0.215, temp=20, fs=44100):
    """
    A linear variation of localization

    """
    theta1 = 2*n.pi*theta1/360
    x1 = n.cos(theta1)*dist
    y1 = n.sin(theta1)*dist
    theta2 = 2*n.pi*theta2/360
    x2 = n.cos(theta2)*dist
    y2 = n.sin(theta2)*dist
    speed = 331.3 + .606*temp

    Lambda = len(sonic_vector)
    L_ = L-1
    xpos = x1 + (x2 - x1)*n.arange(Lambda)/L_
    ypos = y1 + (y2 - y1)*n.arange(Lambda)/L_
    d = n.sqrt( (xpos-zeta/2)**2 + ypos**2 )
    d2 = n.sqrt( (xpos+zeta/2)**2 + ypos**2 )
    IID_a = d/d2
    ITD = (d2-d)/speed
    Lambda_ITD = int(ITD*fs)

    if x1 > 0:
        TL = n.zeros(Lambda_ITD)
        TR = n.array([])
    else:
        TL = n.array([])
        TR = n.zeros(-Lambda_ITD)
    d_ = d[1:] - d[:-1]
    d2_ = d2[1:] - d2[:-1]
    d__ = n.cumsum(d_).astype(n.int64)
    d2__ = n.cumsum(d2_).astype(n.int64)
