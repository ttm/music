""" This file holds minimal implementations to avoid repetitions in the
    musical pieces of the MASS framework: https://github.com/ttm/mass

    Sounds are represented as arrays of PCM samples. Stereo files are
    represented by arrays of shape (2, nsamples).

    See the file HRTF.py, in this same directory, for the functions that use
    impulse responses of Head Related Transfer Functions (HRTFs).

    This file is a copy of mass/src/aux/functions.py imported in
    music/utils.py as `from .functions import *` but it should be integrated
    into music package more properly.
"""
import numpy as np


def normalize_mono(sonic_vector, remove_bias=True):
    """
    Normalize a mono sonic vector.

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
    t = np.array(sonic_vector)
    if np.all(t == 0):
        return t
    else:
        if remove_bias:
            s = t - t.mean()
            fact = max(s.max(), -s.min())
            s = s / fact
        else:
            s = ((t - t.min()) / (t.max() - t.min())) * 2. - 1.
        return s


def normalize_stereo(sonic_vector, remove_bias=True, normalize_sep=False):
    """
    Normalize a stereo sonic vector.

    The final array will have values only between -1 and 1.

    Parameters
    ----------
    sonic_vector : array_like
        A (2, nsamples) shaped array.
    remove_bias : boolean
        Whether to remove or not the bias (or offset)
    normalize_sep : boolean
        Set to True if each channel should be normalized separately.
        If False (default), the arrays will be rescaled in the same proportion
        (preserves loudness proportion).

    Returns
    -------
    sv_normalized : ndarray
        A numpy array with values between -1 and 1.

    """
    sv_copy = np.array(sonic_vector)
    if np.all(sv_copy == 0):
        return sv_copy

    if remove_bias:
        sv_normalized = sv_copy
        sv_normalized[0] = sv_normalized[0] - sv_normalized[0].mean()
        sv_normalized[1] = sv_normalized[1] - sv_normalized[1].mean()
        if normalize_sep:
            fact = max(sv_normalized[0].max(), -sv_normalized[0].min())
            sv_normalized[0] = sv_normalized[0] / fact
            fact = max(sv_normalized[1].max(), -sv_normalized[1].min())
            sv_normalized[1] = sv_normalized[1] / fact
        else:
            fact = max(sv_normalized.max(), -sv_normalized.min())
            sv_normalized = sv_normalized / fact
    else:
        amplitude_ch_1 = sv_copy[0].max() - sv_copy[0].min()
        amplitude_ch_2 = sv_copy[1].max() - sv_copy[1].min()
        if normalize_sep:
            sv_copy[0] = (sv_copy[0] - sv_copy[0].min()) / amplitude_ch_1
            sv_copy[1] = (sv_copy[1] - sv_copy[1].min()) / amplitude_ch_2
            sv_normalized = sv_copy * 2 - 1
        else:
            amplitude = max(amplitude_ch_1, amplitude_ch_2)
            sv_copy = (sv_copy - sv_copy.min()) / amplitude
            sv_copy = (sv_copy - sv_copy.min()) / amplitude
            sv_normalized = sv_copy * 2 - 1
    return sv_normalized
