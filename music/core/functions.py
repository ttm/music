# This file is a copy of mass/src/aux/functions.py
# imported
# in music/utils.py
# as
# from .functions import *
# but it should be integrated into music package more
# properly.
# 
import numpy as n
from scipy.signal import fftconvolve as convolve_
# from HRTF import *
from numbers import Number


__doc__ = """
This file holds minimal implementations
to avoid repetitions in the
musical pieces of the MASS framework:
    https://github.com/ttm/mass

Sounds are represented as arrays of
PCM samples.
Stereo files are represented
by arrays of shape (2, nsamples).

See the music Python Package:
    https://github.com/ttm/music
for a usage of these implementations
within a package and derived routines.

See the file HRTF.py, in this same directory,
for the functions that use impulse responses of
Head Related Transfer Functions (HRTFs).

This file is a copy of mass/src/aux/functions.py
imported
in music/utils.py
as
from .functions import *
but it should be integrated into music package more
properly.

"""

### In this file are functions (only) for:
# IO
# Vibrato and Tremolo
# ADSR

H = n.hstack
def H_(*args):
    stereo = 0
    args = [n.array(a) for a in args]
    for a in args:
        if len(a.shape) == 2:
            stereo = 1
    if stereo:
        for i,a in enumerate(args):
            if len(a.shape) == 1:
                args[i] = n.array(( a, a ))
    return n.hstack(args)
#####################
# IO
def __n(sonic_vector, remove_bias=True):
    """
    Normalize mono sonic_vector.
    
    The final array will have values only between -1 and 1.
    
    Parameters
    ----------
    sonic_vector : array_like
        A (nsamples,) shaped array.
    remove_bias : boolean
        Whether to remove or not the bias (or offset)
    
    Returns
    -------
    s : ndarray
        A numpy array with values between -1 and 1.
    remove_bias : boolean
        Whether to remove or not the bias (or offset)

    """
    t = n.array(sonic_vector)
    if n.all(t==0):
        return t
    else:
        if remove_bias:
            s = t - t.mean()
            fact = max(s.max(), -s.min())
            s = s/fact
        else:
            s = ( (t-t.min()) / (t.max() -t.min()) )*2. -1.
        return s


def __ns(sonic_vector, remove_bias=True, normalize_sep=False):
    """
    Normalize a stereo sonic_vector.
    
    The final array will have values only between -1 and 1.
    
    Parameters
    ----------
    sonic_vector : array_like
        A (2, nsamples) shaped array.
    remove_bias : boolean
        Whether to remove or not the bias (or offset)
    normalize_sep : boolean
        Set to True if each channel should be normalized
        separately. If False (default), the arrays will be
        rescaled in the same proportion
        (preserves loudness proportion).
    
    Returns
    -------
    s : ndarray
        A numpy array with values between -1 and 1.

    """
    t = n.array(sonic_vector)
    if n.all(t==0):
        return t
    else:
        if remove_bias:
            s = t
            s[0] = s[0] - s[0].mean()
            s[1] = s[1] - s[1].mean()
            if normalize_sep:
                fact = max(s[0].max(), -s[0].min())
                s[0] = s[0]/fact
                fact = max(s[1].max(), -s[1].min())
                s[1] = s[1]/fact
            else:
                fact = max(s.max(), -s.min())
                s = s/fact
        else:
            if normalize_sep:
                amb1 = t[0].max() - t[0].min()
                amb2 = t[1].max() - t[1].min()
                t[0] = (t[0] - t[0].min())/amb1
                t[1] = (t[1] - t[1].min())/amb2
                s = t*2 - 1
            else:
                amb1 = t.max() - t.min()
                amb = max(amb1, amb2)
                t = (t - t.min())/amb
                t = (t - t.min())/amb
                s = t*2 - 1
        return s
