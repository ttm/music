"""Core audio processing utilities reused across the package."""
import numpy as np


def _scaled(values, factor):
    """Divide by `factor`, or return silence when there is nothing to scale.

    A constant signal has no dynamic range: once its offset is removed it
    is silence, and the scale factor is zero. Dividing anyway filled the
    result with NaN.
    """
    if factor == 0:
        return np.zeros_like(values, dtype=np.float64)
    return values / factor


def normalize_mono(sonic_vector, remove_bias=True):
    """
    Normalize a mono sonic vector.

    The final array will have values only between -1 and 1.

    Parameters
    ----------
    sonic_vector : array_like
        A (nsamples,) shaped array.
    remove_bias : boolean
        If True (default), subtract the mean and divide by the larger of the
        two peaks, which preserves the waveform's shape. If False, map
        [min, max] onto [-1, 1] affinely, which fills the range but stretches
        an asymmetric waveform. Either way the result is normalized; this
        chooses how.

    Returns
    -------
    s : ndarray
        A numpy array with values between -1 and 1.

    Examples
    --------
    >>> normalize_mono([-1., -.5, 0., .5, 1.])  # already normalized
    array([-1. , -0.5,  0. ,  0.5,  1. ])
    >>> normalize_mono([0., 1., 2.])  # centred, then scaled
    array([-1.,  0.,  1.])

    """
    t = np.array(sonic_vector, dtype=np.float64)
    if t.max() == t.min():
        # Constant, including all-zero: silence once the offset is gone.
        return np.zeros_like(t)
    if remove_bias:
        s = t - t.mean()
        return _scaled(s, max(s.max(), -s.min()))
    return ((t - t.min()) / (t.max() - t.min())) * 2. - 1.


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
    sv_copy = np.array(sonic_vector, dtype=np.float64)
    if sv_copy.max() == sv_copy.min():
        # Constant, including all-zero: silence once the offset is gone.
        return np.zeros_like(sv_copy)

    if remove_bias:
        sv_normalized = sv_copy
        sv_normalized[0] = sv_normalized[0] - sv_normalized[0].mean()
        sv_normalized[1] = sv_normalized[1] - sv_normalized[1].mean()
        if normalize_sep:
            for channel in (0, 1):
                values = sv_normalized[channel]
                sv_normalized[channel] = _scaled(
                    values, max(values.max(), -values.min())
                )
        else:
            sv_normalized = _scaled(
                sv_normalized,
                max(sv_normalized.max(), -sv_normalized.min())
            )
    else:
        amplitude_ch_1 = sv_copy[0].max() - sv_copy[0].min()
        amplitude_ch_2 = sv_copy[1].max() - sv_copy[1].min()
        if normalize_sep:
            sv_copy[0] = _scaled(sv_copy[0] - sv_copy[0].min(),
                                 amplitude_ch_1)
            sv_copy[1] = _scaled(sv_copy[1] - sv_copy[1].min(),
                                 amplitude_ch_2)
            sv_normalized = sv_copy * 2 - 1
        else:
            amplitude = max(amplitude_ch_1, amplitude_ch_2)
            sv_copy = _scaled(sv_copy - sv_copy.min(), amplitude)
            sv_normalized = sv_copy * 2 - 1
    return sv_normalized
