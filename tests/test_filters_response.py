"""What the FIR and IIR filters must do to a signal.

These were the least-covered modules in the package at 14%, and both had
defects that only a test of the filtering itself would catch: `fir` applied
a magnitude response by convolving with the magnitudes rather than with
their inverse transform, so a *flat* response low-passed the signal instead
of leaving it alone.
"""

import time

import numpy as np
import pytest

import music

SAMPLE_RATE = 44100


def _band_energy(signal, low, high, sample_rate=SAMPLE_RATE):
    freqs = np.fft.rfftfreq(len(signal), 1 / sample_rate)
    spectrum = np.abs(np.fft.rfft(signal))
    return float((spectrum[(freqs >= low) & (freqs < high)] ** 2).sum())


def _noise(length=4096, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, length)


# --------------------------------------------------------------------------
# fir
# --------------------------------------------------------------------------

@pytest.mark.parametrize("max_freq", [True, False])
def test_a_flat_magnitude_response_leaves_the_signal_alone(max_freq):
    """Regression: this convolved with the magnitudes themselves, so a flat
    response was a boxcar moving average rather than the identity."""
    signal = _noise(512)
    flat = np.ones(33)

    filtered = music.fir(flat, signal, freq=True, max_freq=max_freq)

    kernel_length = len(np.hstack(
        (flat, flat[1:-1][::-1] if max_freq else flat[1:][::-1])
    ))
    delay = kernel_length // 2
    assert np.allclose(filtered[delay:delay + len(signal)], signal)


def test_a_low_pass_magnitude_response_removes_the_high_band():
    """A response that passes only the lowest quarter must do so sharply."""
    signal = _noise(8192)
    magnitudes = np.zeros(65)
    magnitudes[:16] = 1.0

    filtered = music.fir(magnitudes, signal, freq=True)

    before = (_band_energy(signal, 0, 5000)
              / _band_energy(signal, 15000, 22050))
    after = (_band_energy(filtered, 0, 5000)
             / _band_energy(filtered, 15000, 22050))
    assert after > 100 * before


def test_an_impulse_response_is_convolved_as_given():
    """With freq=False the samples are the kernel, untransformed."""
    signal = _noise(64)
    delay_by_one = np.array([0.0, 1.0, 0.0])

    filtered = music.fir(delay_by_one, signal, freq=False)

    assert np.allclose(filtered[1:len(signal) + 1], signal)


def test_fir_is_linear():
    """Filtering is linear: it distributes over a weighted sum."""
    first, second = _noise(256, seed=1), _noise(256, seed=2)
    magnitudes = np.linspace(1, 0, 17)

    together = music.fir(magnitudes, 2 * first - 3 * second)
    apart = (2 * music.fir(magnitudes, first)
             - 3 * music.fir(magnitudes, second))
    assert np.allclose(together, apart)


def test_fir_accepts_lists():
    """The parameters are documented as array_like."""
    out = music.fir([1.0, 1.0, 1.0], [0.0, 1.0, 0.0, 0.0])
    assert np.isfinite(out).all()


# --------------------------------------------------------------------------
# iir
# --------------------------------------------------------------------------

def test_iir_identity_passes_the_signal_through():
    signal = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.allclose(music.iir(signal, [1.0], [1.0]), signal)


@pytest.mark.parametrize("pole", [0.5, 0.9, -0.5])
def test_iir_one_pole_matches_its_closed_form(pole):
    """y[n] = x[n] + pole * y[n-1], driven by an impulse, is pole ** n."""
    impulse = np.zeros(16)
    impulse[0] = 1.0

    out = music.iir(impulse, [1.0], [1.0, pole])

    assert np.allclose(out, pole ** np.arange(16))


def test_iir_feedforward_matches_a_convolution():
    """With no feedback the filter is an ordinary FIR convolution."""
    signal = _noise(64)
    taps = [0.5, 0.25, 0.125]

    out = music.iir(signal, taps, [1.0])

    assert np.allclose(out, np.convolve(signal, taps)[:len(signal)])


def test_iir_accepts_lists_as_documented():
    """Regression: the coefficients are documented as an iterable of
    scalars, but two Python lists multiplied together raise TypeError."""
    out = music.iir([1.0, 0.0, 0.0, 0.0], [1.0], [1.0, 0.5])
    assert np.allclose(out, [1.0, 0.5, 0.25, 0.125])


def test_iir_matches_the_recurrence_it_documents():
    """Pin the semantics independently of the implementation.

    The docstring states b0*y[n] = sum a_k x[n-k] + sum_{j>=1} b_j y[n-j].
    Note the plus on the feedback term, which is not what
    scipy.signal.lfilter does -- so the convention is worth a test that
    does not go through any numpy vectorisation.
    """
    rng = np.random.RandomState(0)
    x = rng.randn(60)
    a = [0.7, -0.2, 0.1]
    b = [2.0, 0.3, -0.15]

    expected = []
    for n in range(len(x)):
        total = sum(a[k] * x[n - k] for k in range(len(a)) if n - k >= 0)
        total += sum(b[j] * expected[n - j]
                     for j in range(1, len(b)) if n - j >= 0)
        expected.append(total / b[0])

    assert np.allclose(music.iir(x, a, b), expected)


def test_iir_cost_is_linear_in_the_signal_length():
    """Regression: it rebuilt a reversed copy of the whole signal so far
    on every sample, so cost grew with the square of the length. One
    second of audio took about three seconds, and ten seconds took five
    minutes. Only the last len(a) inputs and len(b) - 1 outputs are ever
    read, so the slices are bounded by the filter order.

    Quadratic growth would show as roughly 16x here; linear shows as 4x.
    The threshold is loose because this is wall-clock on shared CI.
    """
    a, b = [1.0], [1.0, -0.9]
    music.iir(np.zeros(256), a, b)  # warm up

    def best_of_three(n):
        signal = np.zeros(n)
        times = []
        for _ in range(3):
            start = time.perf_counter()
            music.iir(signal, a, b)
            times.append(time.perf_counter() - start)
        return min(times)

    small = best_of_three(2000)
    large = best_of_three(8000)
    assert large / small < 8, (
        f"quadrupling the input cost {large / small:.1f}x, which looks "
        f"quadratic rather than linear"
    )


def test_iir_is_linear():
    first, second = _noise(128, seed=3), _noise(128, seed=4)
    a, b = [1.0, 0.3], [1.0, -0.4]

    together = music.iir(2 * first - 3 * second, a, b)
    apart = 2 * music.iir(first, a, b) - 3 * music.iir(second, a, b)
    assert np.allclose(together, apart)
