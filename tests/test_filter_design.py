"""The IIR filter designs, against the equations that specify them.

`music.iir` applies coefficients; these compute them, from the four designs
`body.tex` gives "for didactic purposes and as a reference" -- equations
``eq:passa-baixas``, ``eq:passa-altas``, ``eq:varAux``, ``eq:passa-banda``
and ``eq:rejeita-banda``.

Each is checked twice: that the coefficients are the ones the equation
writes, and that the filter they make behaves the way the article says it
will -- 3 dB down at the cutoff, and a band whose edges sit where the
bandwidth puts them.

References
----------
.. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time
       representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)
"""

import numpy as np
import pytest

import music

MINUS_THREE_DB = 1 / np.sqrt(2)


def _response(a, b, freqs):
    """|H(f)| of the filter `iir` would apply, as a fraction of f_s."""
    freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
    z = np.exp(-2j * np.pi * freqs)
    numerator = sum(a[j] * z ** j for j in range(len(a)))
    denominator = 1 - sum(b[k] * z ** k for k in range(1, len(b)))
    magnitude = np.abs(numerator / denominator)
    return magnitude if magnitude.size > 1 else float(magnitude[0])


def _applied_response(a, b, freq, sample_rate=44100):
    """The same, measured by running a tone through `music.iir`."""
    duration = 0.2
    tone = music.note(freq=freq * sample_rate, duration=duration,
                      waveform_table=music.WAVEFORM_SINE,
                      sample_rate=sample_rate)
    filtered = music.iir(sonic_vector=tone, a=a, b=b)
    settled = slice(len(tone) // 2, None)     # past the transient
    return (np.sqrt(np.mean(filtered[settled] ** 2))
            / np.sqrt(np.mean(tone[settled] ** 2)))


# --------------------------------------------------------------------------
# The coefficients the equations write
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cutoff", [0.01, 0.05, 0.1, 0.25])
def test_low_pass_coefficients_are_equation_passa_baixas(cutoff):
    """x = exp(-2 pi f_c); a_0 = 1 - x; b_1 = x."""
    a, b = music.low_pass(cutoff)
    x = np.exp(-2 * np.pi * cutoff)
    assert a == pytest.approx([1 - x])
    assert b == pytest.approx([1.0, x])


@pytest.mark.parametrize("cutoff", [0.01, 0.05, 0.1, 0.25])
def test_high_pass_coefficients_are_equation_passa_altas(cutoff):
    """x = exp(-2 pi f_c); a_0 = (x + 1) / 2; a_1 = -a_0; b_1 = x."""
    a, b = music.high_pass(cutoff)
    x = np.exp(-2 * np.pi * cutoff)
    assert a == pytest.approx([(x + 1) / 2, -(x + 1) / 2])
    assert b == pytest.approx([1.0, x])


@pytest.mark.parametrize("centre, bandwidth",
                         [(0.05, 0.01), (0.1, 0.02), (0.2, 0.05)])
def test_band_pass_coefficients_are_equations_varaux_and_passa_banda(
        centre, bandwidth):
    """R = 1 - 3bw, K from eq:varAux, then the five coefficients."""
    a, b = music.band_pass(centre, bandwidth)
    r = 1 - 3 * bandwidth
    cosine = np.cos(2 * np.pi * centre)
    k = (1 - 2 * r * cosine + r ** 2) / (2 - 2 * cosine)

    assert a == pytest.approx([1 - k, 2 * (k - r) * cosine, r ** 2 - k])
    assert b == pytest.approx([1.0, 2 * r * cosine, -r ** 2])


@pytest.mark.parametrize("centre, bandwidth",
                         [(0.05, 0.01), (0.1, 0.02), (0.2, 0.05)])
def test_band_reject_coefficients_are_equation_rejeita_banda(
        centre, bandwidth):
    """The same R and K, giving a_0 = K, a_1 = -2K cos, a_2 = K."""
    a, b = music.band_reject(centre, bandwidth)
    r = 1 - 3 * bandwidth
    cosine = np.cos(2 * np.pi * centre)
    k = (1 - 2 * r * cosine + r ** 2) / (2 - 2 * cosine)

    assert a == pytest.approx([k, -2 * k * cosine, k])
    assert b == pytest.approx([1.0, 2 * r * cosine, -r ** 2])


# --------------------------------------------------------------------------
# The behaviour the article says they will have
# --------------------------------------------------------------------------

@pytest.mark.parametrize("design", [music.low_pass, music.high_pass])
@pytest.mark.parametrize("cutoff", [0.005, 0.01, 0.02])
def test_the_one_pole_filters_are_three_decibels_down_at_their_cutoff(
        design, cutoff):
    """The article's definition of f_c, where the design is faithful to it.

    ``x = exp(-2 pi f_c)`` is the sampled form of an analogue one-pole, and
    it puts the 3 dB point where the article says only while f_c is small
    against the sample rate. Held to half a percent up to f_c = 0.02, which
    is 880 Hz at 44.1 kHz.
    """
    a, b = design(cutoff)
    assert _response(a, b, cutoff) == pytest.approx(
        MINUS_THREE_DB, rel=0.005)


@pytest.mark.parametrize("design", [music.low_pass, music.high_pass])
def test_the_cutoff_drifts_from_three_decibels_as_it_nears_nyquist(design):
    """And the drift is monotone, so a caller knows which way it goes.

    The article states f_c as "where the filter performs an attenuation of
    -3dB" without qualifying it. That holds near the bottom of the range
    and not near the top: at f_c = 0.25 the low pass is 0.78 rather than
    0.707, which is 2 dB rather than 3. See DISCREPANCIES.md.
    """
    errors = [abs(_response(*design(cutoff), cutoff) - MINUS_THREE_DB)
              for cutoff in (0.01, 0.05, 0.1, 0.2, 0.25)]
    assert errors == sorted(errors)
    assert errors[0] < 0.001
    assert errors[-1] > 0.05


def test_the_low_pass_passes_low_and_stops_high():
    a, b = music.low_pass(0.05)
    assert _response(a, b, 0.001) > 0.99
    assert _response(a, b, 0.4) < 0.3


def test_the_high_pass_passes_high_and_stops_low():
    a, b = music.high_pass(0.05)
    assert _response(a, b, 0.001) < 0.03
    assert _response(a, b, 0.4) > 0.9


@pytest.mark.parametrize("centre, bandwidth",
                         [(0.05, 0.01), (0.1, 0.02), (0.15, 0.04)])
def test_the_band_pass_peaks_at_its_centre_and_is_as_wide_as_asked(
        centre, bandwidth):
    """The 3 dB points sit at centre +/- bandwidth / 2.

    The article's prose says centre +/- bandwidth. Its coefficients give
    half that, at every bandwidth measured, so `bandwidth` here is the full
    width between the two 3 dB points. See DISCREPANCIES.md.
    """
    a, b = music.band_pass(centre, bandwidth)
    freqs = np.linspace(0.0005, 0.4995, 20000)
    magnitude = _response(a, b, freqs)
    magnitude = magnitude / magnitude.max()

    assert freqs[np.argmax(magnitude)] == pytest.approx(centre, abs=0.002)

    passband = freqs[magnitude >= MINUS_THREE_DB]
    measured = (passband[-1] - passband[0])
    assert measured == pytest.approx(bandwidth, rel=0.1)


@pytest.mark.parametrize("centre", [0.05, 0.1, 0.2])
def test_the_band_reject_takes_out_its_centre_and_leaves_the_rest(centre):
    a, b = music.band_reject(centre, 0.02)
    assert _response(a, b, centre) < 1e-9
    assert _response(a, b, centre / 4) > 0.7
    assert _response(a, b, min(centre * 4, 0.45)) > 0.7


# --------------------------------------------------------------------------
# The designs, applied
# --------------------------------------------------------------------------

@pytest.mark.parametrize("design, passes, stops", [
    (music.low_pass, 0.005, 0.3),
    (music.high_pass, 0.3, 0.005),
])
def test_a_designed_filter_does_that_to_a_real_sound(design, passes, stops):
    """The frequency response, measured by running tones through `iir`."""
    a, b = design(0.05)
    assert _applied_response(a, b, passes) > 0.8 * _applied_response(
        a, b, stops)
    assert _applied_response(a, b, passes) > 0.5


def test_fraction_of_converts_hertz_to_what_the_designs_take():
    assert music.fraction_of(4410, sample_rate=44100) == pytest.approx(0.1)
    a, b = music.low_pass(music.fraction_of(1000, sample_rate=44100))
    assert len(a) == 1 and len(b) == 2


@pytest.mark.parametrize("design, kwargs", [
    (music.low_pass, {"cutoff": 0.0}),
    (music.low_pass, {"cutoff": 0.5}),
    (music.high_pass, {"cutoff": -0.1}),
    (music.band_pass, {"centre": 0.6, "bandwidth": 0.05}),
    (music.band_reject, {"centre": 0.1, "bandwidth": 0.0}),
])
def test_a_frequency_outside_the_useful_range_is_refused(design, kwargs):
    """Above the Nyquist fraction there is nothing to filter."""
    with pytest.raises(ValueError, match="fraction of the sample rate"):
        design(**kwargs)


def test_fraction_of_refuses_a_frequency_that_is_not_positive():
    with pytest.raises(ValueError, match="must be positive"):
        music.fraction_of(0)
