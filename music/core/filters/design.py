"""Coefficients for the IIR filters the MASS article specifies.

:func:`~music.core.filters.impulse_response.iir` applies coefficients; these
compute them. Each routine returns the ``(a, b)`` pair that function takes,
with ``b[0] = 1`` as the article's normalisation gives it, so a filter is
designed and applied in two steps::

    >>> a, b = low_pass(cutoff=0.05)
    >>> filtered = iir(note(), a, b)

Cutoff, centre and bandwidth are all fractions of the sample rate and so
lie in ``(0, 0.5)``: 0.5 is the Nyquist frequency, above which there is
nothing to filter. :func:`fraction_of` converts a frequency in Hertz.

References
----------
.. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time
       representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)
"""
import numpy as np
from numpy.typing import NDArray

__all__ = ['fraction_of', 'low_pass', 'high_pass', 'band_pass', 'band_reject']


def _check(name: str, value: float) -> float:
    """A frequency given as a fraction of the sample rate."""
    value = float(value)
    if not 0 < value < 0.5:
        raise ValueError(
            f'{name} is a fraction of the sample rate and must lie in '
            f'(0, 0.5); got {value}. Above 0.5 -- the Nyquist frequency -- '
            f'there is nothing to filter. Use fraction_of() to convert a '
            f'frequency in Hertz.')
    return value


def fraction_of(freq: float, sample_rate: int = 44100) -> float:
    """A frequency in Hertz, as the fraction of the sample rate a design wants.

    Parameters
    ----------
    freq : scalar
        The frequency in Hertz.
    sample_rate : integer
        The sample rate the filter will run at.

    Returns
    -------
    float
        ``freq / sample_rate``, which is in ``(0, 0.5)`` for any frequency
        below the Nyquist limit.

    Raises
    ------
    ValueError
        If `freq` is not positive.

    Examples
    --------
    >>> fraction_of(4410, sample_rate=44100)
    0.1
    >>> a, b = low_pass(fraction_of(1000))
    """
    if freq <= 0:
        raise ValueError(f'freq must be positive; got {freq}')
    return freq / sample_rate


def low_pass(cutoff: float = 0.1) -> tuple[NDArray[np.float64],
                                           NDArray[np.float64]]:
    """A one-pole low pass, from the article's equation ``eq:passa-baixas``.

    ``x = exp(-2 pi f_c)``, then ``a_0 = 1 - x`` and ``b_1 = x``.

    Parameters
    ----------
    cutoff : scalar
        Where the filter attenuates by 3 dB, as a fraction of the sample
        rate, in ``(0, 0.5)``.

    Returns
    -------
    a, b : ndarray
        The feedforward and feedback coefficients, for :func:`iir`.

    Raises
    ------
    ValueError
        If `cutoff` is outside ``(0, 0.5)``, which is where a fraction of
        the sample rate has a filter to describe.

    See Also
    --------
    high_pass : the complementary filter, from the same pole.
    fraction_of : converts a frequency in Hertz into a `cutoff`.

    Examples
    --------
    >>> a, b = low_pass(cutoff=0.05)
    >>> [round(float(v), 6) for v in a]
    [0.269597]
    >>> [round(float(v), 6) for v in b]
    [1.0, 0.730403]

    Notes
    -----
    One pole, so the roll-off above the cutoff is 6 dB per octave. The
    article gives it "for didactic purposes and as a reference", and names
    biquad recipes and Chebyshev coefficients as what to reach for when
    this is not sharp enough.
    """
    x = np.exp(-2 * np.pi * _check('cutoff', cutoff))
    return np.array([1 - x]), np.array([1.0, x])


def high_pass(cutoff: float = 0.1) -> tuple[NDArray[np.float64],
                                            NDArray[np.float64]]:
    """A one-pole high pass, from the article's equation ``eq:passa-altas``.

    ``x = exp(-2 pi f_c)``, then ``a_0 = (x + 1) / 2``, ``a_1 = -a_0`` and
    ``b_1 = x``.

    Parameters
    ----------
    cutoff : scalar
        Where the filter attenuates by 3 dB, as a fraction of the sample
        rate, in ``(0, 0.5)``.

    Returns
    -------
    a, b : ndarray
        The feedforward and feedback coefficients, for :func:`iir`.

    Raises
    ------
    ValueError
        If `cutoff` is outside ``(0, 0.5)``, which is where a fraction of
        the sample rate has a filter to describe.

    See Also
    --------
    low_pass : the complementary filter, from the same pole.

    Examples
    --------
    >>> a, b = high_pass(cutoff=0.05)
    >>> [round(float(v), 6) for v in a]
    [0.865201, -0.865201]
    """
    x = np.exp(-2 * np.pi * _check('cutoff', cutoff))
    half = (x + 1) / 2
    return np.array([half, -half]), np.array([1.0, x])


def _resonator(centre: float, bandwidth: float) -> tuple[float, float, float]:
    """The auxiliary variables ``eq:varAux`` shares between the two filters."""
    centre = _check('centre', centre)
    bandwidth = _check('bandwidth', bandwidth)
    r = 1 - 3 * bandwidth
    cosine = np.cos(2 * np.pi * centre)
    k = (1 - 2 * r * cosine + r ** 2) / (2 - 2 * cosine)
    return r, k, cosine


def band_pass(centre: float = 0.1, bandwidth: float = 0.05) \
        -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """A two-pole band pass, from the article's ``eq:passa-banda``.

    With ``R`` and ``K`` from ``eq:varAux``: ``a = [1 - K,
    2(K - R)cos(2 pi f_c), R**2 - K]`` and ``b = [1, 2R cos(2 pi f_c),
    -R**2]``.

    Parameters
    ----------
    centre : scalar
        The centre of the band, as a fraction of the sample rate.
    bandwidth : scalar
        The full width of the band between its 3 dB points, so they fall at
        ``centre +/- bandwidth / 2``. The article's prose says
        ``centre +/- bandwidth``; its coefficients give half that, measured
        across every bandwidth tried. See `DISCREPANCIES.md`.

    Returns
    -------
    a, b : ndarray
        The feedforward and feedback coefficients, for :func:`iir`.

    Raises
    ------
    ValueError
        If `centre` or `bandwidth` is outside ``(0, 0.5)``.

    See Also
    --------
    band_reject : the same geometry, keeping what this one discards.

    Examples
    --------
    >>> a, b = band_pass(centre=0.1, bandwidth=0.02)
    >>> len(a), len(b)
    (3, 3)

    Notes
    -----
    The article warns that these two filters amplify rather than attenuate
    when the centre is low and the band wide, and that at high centres they
    spread towards the bass rather than holding their shape. Neither is a
    defect in the coefficients; both are what a two-pole design does.
    """
    r, k, cosine = _resonator(centre, bandwidth)
    a = np.array([1 - k, 2 * (k - r) * cosine, r ** 2 - k])
    b = np.array([1.0, 2 * r * cosine, -r ** 2])
    return a, b


def band_reject(centre: float = 0.1, bandwidth: float = 0.05) \
        -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """A two-pole notch, from the article's ``eq:rejeita-banda``.

    With ``R`` and ``K`` from ``eq:varAux``: ``a = [K, -2K cos(2 pi f_c),
    K]`` and ``b = [1, 2R cos(2 pi f_c), -R**2]``.

    Parameters
    ----------
    centre : scalar
        The centre of the band to remove, as a fraction of the sample rate.
    bandwidth : scalar
        The full width of the band between its 3 dB points, as in
        :func:`band_pass`.

    Returns
    -------
    a, b : ndarray
        The feedforward and feedback coefficients, for :func:`iir`.

    Raises
    ------
    ValueError
        If `centre` or `bandwidth` is outside ``(0, 0.5)``.

    See Also
    --------
    band_pass : the same geometry, keeping what this one discards.

    Examples
    --------
    >>> a, b = band_reject(centre=0.1, bandwidth=0.02)
    >>> len(a), len(b)
    (3, 3)
    """
    r, k, cosine = _resonator(centre, bandwidth)
    a = np.array([k, -2 * k * cosine, k])
    b = np.array([1.0, 2 * r * cosine, -r ** 2])
    return a, b
