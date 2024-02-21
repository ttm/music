import numpy as n
from .n import N

def loc(sonic_vector=N(), theta=0, dist=0, x=.1, y=.01, zeta=0.215, temp=20, fs=44100):
    """
    Make a mono sound stereo and localize it by a very naive method.

    See bellow for implementation notes.

    Parameters
    ----------
    sonic_vector : array_like
        An one dimensional with the PCM samples of the sound.
    x : scalar
        The lateral component of the position in meters.
    y : scalar
        The frontal component of the position in meters.
    theta : scalar
        The azimuthal angle of the position in degrees.
        If theta is supplied, x and y are ignored
        and dist must also be supplied
        for the sound localization to have effect.
    dist : scalar
        The distance of the source from the listener
        in meters.
    zeta : scalar
        The distance between the ears in meters.
    temp : scalar
        The temperature in Celsius used for calculating
        the speed of sound.
    fs : integer
        The sample rate.

    Returns
    -------
    s : ndarray
        A (2, nsamples) shaped array with the PCM
        samples of the stereo sound.

    See Also
    --------
    R : A reverberator.
    loc_ : a less naive implementation of localization
    by ITD and IID.
    hrtf : performs localization by means of a
    Head Related Transfer Function.

    Examples
    --------
    >>> WS(loc())  # write a soundfile that is localized
    >>> WS(H([loc(V(d=1), x=i, y=j) for i, j in
    ...   zip([.1,.7,n.pi-.1,n.pi-.7], [.1,.1,.1,.1])]))

    Notes
    -----
    Uses the most naive ITD and IID calculations as described in [1].
    A less naive method is implemented in loc_().
    Nonetheless, if dist is small enough (e.g. <.3),
    the perception of theta occurs and might be used.
    The advantages of this method are:
      - It is fast.
      - It is simple.
      - It is true to sound propagation phenomenon
      (although it does not consider the human body
      beyond the localization of the ears).
      - It can be used easily for tweaks
      (such as for a moving source resulting
      in a Doppler Effect).

    When az = tan^{-1}(y/x) lies in the 'cone of confusion',
    many values of x and y have the same ITD and IID [1].
    Furthermore, lateral sources have the low frequencies
    diffracted and reach the opposite ear with a delay
    of ~0.7s [1].
    The height of a source and if it is in front or
    behind a listener are cues given by te HRTF [1].
    These issues are not taken into account in this
    function.

    The value of zeta is ~0.215 for adult humans [1].

    This implementation assumes that the speed
    of sound (in air) is s = 331.3+0.606*temp.

    Cite the following article whenever you use this function.

    References
    ----------
    .. [1] Fabbri, Renato, et al. "Musical elements in the 
    discrete-time representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)

    """
    if theta:
        theta = 2*n.pi*theta/360
        x = n.cos(theta)*dist
        y = n.sin(theta)*dist
    speed = 331.3 + .606*temp

    dr = n.sqrt((x-zeta/2)**2+y**2)  # distance from right ear
    dl = n.sqrt((x+zeta/2)**2+y**2)  # distance from left ear

    IID_a = dr/dl  # proportion of amplitudes from left to right ear
    ITD = (dl-dr)/speed  # seconds
    Lambda_ITD = int(ITD*fs)

    if x > 0:
        TL = n.hstack((n.zeros(Lambda_ITD), IID_a*sonic_vector))
        TR = n.hstack((sonic_vector, n.zeros(Lambda_ITD)))
    else:
        TL = n.hstack((sonic_vector, n.zeros(-Lambda_ITD)))
        TR = n.hstack((n.zeros(-Lambda_ITD), sonic_vector*(1/IID_a)))
    s = n.vstack((TL, TR))
    return s