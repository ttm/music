"""What the mutation audit of the band-limited oscillators found unasserted.

The tests measured these oscillators as spectra: how much energy fell off
the harmonics, how much a sweep jumped. A spectrum is not a sample. Every
Fourier coefficient of the frequency path could change, a square could be
summed with a triangle's coefficients, PolyBLEP's sawtooth could run
backwards, and each test still passed. These read the samples, against
references built apart from the code: the waveforms' closed forms,
PolyBLEP as its authors define it, SciPy called directly, and sines
read through the oversampler's filter.
"""

import numpy as np
import pytest

import music
from music.core.synths import bandlimited, polyblep
from music.core.synths.frequency_path import bandlimited_frequency_path
from music.core.synths.polyblep import (
    PolyBLEPOscillator, polyblep_frequency_path,
)


@pytest.fixture(autouse=True)
def fresh_tables():
    """Filter each test's tables anew, rather than read another test's.

    bandlimited_note keeps the table it filters for a pitch, so every
    test after the first would read tables filtered before it ran -- in
    a mutation run, before the mutant it judges was switched on.
    """
    bandlimited._filtered_table.cache_clear()


def phase_of(frequencies, rate):
    """The trapezoidally integrated phase, from zero, in radians."""
    f = np.asarray(frequencies, dtype=float)
    return 2 * np.pi / rate * np.r_[0., np.cumsum((f[:-1] + f[1:]) / 2)]


#: Each waveform as its Fourier series, from the standard tables: the k-th
#: sine's coefficient, or None where the series has no such term.
SERIES = {
    "sawtooth": lambda k: 2 / np.pi * (-1) ** (k + 1) / k,
    "square": lambda k: 4 / (np.pi * k) if k % 2 else None,
    "triangle": lambda k: (8 / np.pi ** 2 * (-1) ** ((k - 1) // 2) / k ** 2
                           if k % 2 else None),
}


def series(waveform, frequencies, rate, transition, harmonics):
    """The waveform's series, each term faded as the docstring says."""
    f = np.asarray(frequencies, dtype=float)
    theta = phase_of(f, rate)
    nyquist = rate / 2
    out = np.zeros_like(f)
    for k in range(1, harmonics + 1):
        c = SERIES[waveform](k)
        if c is None:
            continue
        if transition:
            weight = np.clip((nyquist - k * f) / (nyquist * transition),
                             0, 1)
        else:
            weight = (k * f < nyquist).astype(float)
        out += c * weight * np.sin(k * theta)
    return out


# --------------------------------------------------------------------------
# the frequency path
# --------------------------------------------------------------------------

#: Each waveform in closed form, as a function of the phase: what its
#: series sums to.
CLOSED = {
    "sawtooth": lambda t: (t / np.pi + 1) % 2 - 1,
    "square": lambda t: np.sign(np.sin(t)),
    "triangle": lambda t: 2 / np.pi * np.arcsin(np.sin(t)),
}


@pytest.mark.parametrize("waveform,tolerance", [
    ("sawtooth", .06), ("square", .08), ("triangle", .002)])
def test_a_low_note_is_its_waveform_with_the_partials_it_can_have(
        waveform, tolerance):
    """At 100 Hz, 239 partials sit below the Nyquist frequency of 48 kHz,
    and the sum is its waveform but for the ringing at each edge."""
    f = np.full(4800, 100.)
    out = bandlimited_frequency_path(f, sample_rate=48000, waveform=waveform,
                                     transition=0, max_harmonics=2048)
    ideal = CLOSED[waveform](phase_of(f, 48000))
    assert np.sqrt(np.mean((out - ideal) ** 2)) < tolerance


@pytest.mark.parametrize("waveform", ["sawtooth", "square", "triangle"])
@pytest.mark.parametrize("transition", [0, .15, 1])
def test_each_partial_has_its_coefficient_and_its_fade(waveform,
                                                       transition):
    """A sweep that carries partials up past the Nyquist frequency, where
    the fade must leave them at nothing, never below it."""
    f = np.linspace(2000, 11000, 2400)
    out = bandlimited_frequency_path(f, sample_rate=48000, waveform=waveform,
                                     transition=transition)
    np.testing.assert_allclose(
        out, series(waveform, f, 48000, transition, 12), atol=1e-12)


def test_a_sine_fades_by_the_fraction_of_nyquist_it_is_given():
    f = np.linspace(15000, 23900, 960)
    out = bandlimited_frequency_path(f, sample_rate=48000, waveform="sine",
                                     transition=.5)
    weight = np.clip((24000 - f) / 12000, 0, 1)
    np.testing.assert_allclose(out, weight * np.sin(phase_of(f, 48000)),
                               atol=1e-12)


def test_the_default_limit_is_256_partials():
    f = np.full(480, 50.)
    default = bandlimited_frequency_path(f, sample_rate=48000)
    assert np.array_equal(default, bandlimited_frequency_path(
        f, sample_rate=48000, waveform="sawtooth", transition=.15,
        max_harmonics=256))
    assert not np.array_equal(default, bandlimited_frequency_path(
        f, sample_rate=48000, max_harmonics=257))


@pytest.mark.parametrize("kwargs", [
    {"frequencies": [.5], "sample_rate": 48000},
    {"frequencies": [20000.], "sample_rate": 48000},
    {"frequencies": [.25], "sample_rate": 1},
    {"frequencies": [1000.], "transition": 1},
    {"frequencies": [1000.], "max_harmonics": 1},
    {"frequencies": [1000.], "max_harmonics": 2048},
])
def test_the_edges_of_what_the_path_takes_are_taken(kwargs):
    assert np.isfinite(bandlimited_frequency_path(**kwargs)).all()


def test_whole_frequencies_are_taken_as_numbers():
    whole = bandlimited_frequency_path(np.full(480, 1000), sample_rate=48000)
    assert np.array_equal(whole, bandlimited_frequency_path(
        np.full(480, 1000.), sample_rate=48000))


def test_the_work_bound_takes_exactly_fifty_million():
    f = np.full(200_000, 20.)
    assert bandlimited_frequency_path(
        f, sample_rate=48000, waveform="sine",
        max_harmonics=250).shape == (200_000,)
    with pytest.raises(ValueError, match="50 million"):
        bandlimited_frequency_path(np.full(200_001, 20.), sample_rate=48000,
                                   waveform="sine", max_harmonics=250)


# --------------------------------------------------------------------------
# PolyBLEP
# --------------------------------------------------------------------------

def blep(t, dt):
    """The polynomial band-limited step, as its authors write it."""
    if t < dt:
        t /= dt
        return t + t - t * t - 1
    if t > 1 - dt:
        t = (t - 1) / dt
        return t * t + t + t + 1
    return 0.


def textbook(frequencies, rate, waveform, phase):
    """PolyBLEP one sample at a time: the phase moves by the mean of each
    pair of frequencies, from where it starts, and each wrap is smoothed."""
    out, previous = [], None
    for f in frequencies:
        if previous is not None:
            phase = (phase + (previous + f) / (2 * rate)) % 1
        previous = f
        dt = f / rate
        if waveform == "sawtooth":
            out.append(2 * phase - 1 - blep(phase, dt))
        else:
            value = 1. if phase < .5 else -1.
            out.append(value + blep(phase, dt) - blep((phase + .5) % 1, dt))
    return np.array(out)


@pytest.mark.parametrize("waveform", ["sawtooth", "square"])
@pytest.mark.parametrize("phase", [0., .37, .5])
def test_polyblep_renders_what_its_definition_does(waveform, phase):
    f = 1000 + 7000 * (1 + np.sin(np.linspace(0, 6, 3000))) / 2
    out = polyblep_frequency_path(f, sample_rate=48000, waveform=waveform,
                                  phase_cycles=phase)
    np.testing.assert_allclose(out, textbook(f, 48000, waveform, phase),
                               atol=1e-12)


def test_the_path_s_defaults_are_a_sawtooth_at_44100_from_phase_zero():
    f = np.full(100, 3000.)
    assert np.array_equal(polyblep_frequency_path(f), polyblep_frequency_path(
        f, sample_rate=44100, waveform="sawtooth", phase_cycles=0.))
    np.testing.assert_allclose(polyblep_frequency_path(f), textbook(
        f, 44100, "sawtooth", 0.), atol=1e-12)
    oscillator = PolyBLEPOscillator()
    assert (oscillator.sample_rate, oscillator.waveform) == (44100,
                                                             "sawtooth")
    assert PolyBLEPOscillator(waveform="square").waveform == "square"


@pytest.mark.parametrize("rate", [8000, 192000])
def test_the_lowest_and_highest_rates_are_taken(rate):
    assert PolyBLEPOscillator(sample_rate=rate).sample_rate == rate


@pytest.mark.parametrize("kwargs,match", [
    ({"sample_rate": 7999}, "sample_rate must be 8000 to 192000 Hz"),
    ({"sample_rate": 192001}, "sample_rate must be 8000 to 192000 Hz"),
    ({"sample_rate": 48000.0}, "sample_rate must be 8000 to 192000 Hz"),
    ({"waveform": "sine"}, "waveform must be sawtooth or square"),
    ({"phase_cycles": 1.0}, r"phase_cycles must be finite in \[0, 1\)"),
])
def test_an_oscillator_refuses_what_it_cannot_render_and_says_so(kwargs,
                                                                 match):
    with pytest.raises(ValueError, match=match):
        PolyBLEPOscillator(**kwargs)


@pytest.mark.parametrize("path", [[.5], [16000.]])
def test_the_edges_of_what_a_block_takes_are_taken(path):
    assert np.isfinite(PolyBLEPOscillator(sample_rate=48000).render(
        path)).all()


@pytest.mark.parametrize("path,match", [
    (np.zeros((2, 3)), "frequencies must be one-dimensional"),
    ([0.], "frequencies must be finite, positive, < Nyquist"),
    ([24000.], "frequencies must be finite, positive, < Nyquist"),
])
def test_a_block_refuses_what_it_cannot_render_and_says_so(path, match):
    with pytest.raises(ValueError, match=match):
        PolyBLEPOscillator(sample_rate=48000).render(path)


def test_a_block_may_be_as_long_as_the_limit(monkeypatch):
    monkeypatch.setattr(polyblep, "_MAX_BLOCK_SAMPLES", 10)
    oscillator = PolyBLEPOscillator(sample_rate=48000)
    assert oscillator.render(np.full(10, 440.)).shape == (10,)
    with pytest.raises(ValueError, match="block exceeds 5 million frames"):
        oscillator.render(np.full(11, 440.))


# --------------------------------------------------------------------------
# oversampling
# --------------------------------------------------------------------------

def test_oversampling_filters_the_whole_generator_called_once():
    signal = pytest.importorskip("scipy.signal")
    from music.core.synths.oversampling import _decimation_filter

    calls = []

    def gated(**kwargs):
        calls.append(dict(kwargs))
        return music.isochronic_tones(**kwargs)

    out = music.render_oversampled(
        gated, sample_rate=48000, number_of_samples=960,
        factor=4, carrier_freq=9000, pulse_rate=40)
    assert calls == [{"number_of_samples": 3840, "sample_rate": 192000,
                      "carrier_freq": 9000, "pulse_rate": 40}]
    raw = music.isochronic_tones(number_of_samples=3840, sample_rate=192000,
                                 carrier_freq=9000, pulse_rate=40)
    expected = signal.resample_poly(
        raw, 1, 4, window=_decimation_filter(4))[:960]
    np.testing.assert_array_equal(out, expected)


def test_the_filter_reaches_sixty_four_samples_either_side():
    """Its length is its cost, and how far into the silence it reaches."""
    pytest.importorskip("scipy")
    from music.core.synths.oversampling import _decimation_filter

    lengths = [len(_decimation_filter(factor)) for factor in (2, 4, 16)]
    assert lengths == [259, 515, 2053]


def sine(hz):
    def render(*, number_of_samples, sample_rate):
        return np.sin(2 * np.pi * hz * np.arange(number_of_samples)
                      / sample_rate)
    return render


@pytest.mark.parametrize("factor", (2, 3, 4, 16))
def test_the_filter_keeps_nine_tenths_of_the_band_and_shuts_the_rest(factor):
    """Within 1.1e-5 of a sine below 0.9 Nyquist; 99 dB down from it up.

    Read in the middle of a render, beyond the 65 samples at each end
    that the filter reaches into the silence past it.
    """
    pytest.importorskip("scipy")
    rate, count = 8000, 1200
    n = np.arange(count)[100:-100]
    for fraction in (.05, .5, .8, .9):
        hz = fraction * rate / 2
        out = music.render_oversampled(sine(hz), sample_rate=rate,
                                       number_of_samples=count,
                                       factor=factor)[100:-100]
        exact = np.sin(2 * np.pi * hz * n / rate)
        assert np.max(np.abs(out - exact)) < 1.1e-5
    for fraction in (1, 1.02, 1.5, factor - .02):
        out = music.render_oversampled(sine(fraction * rate / 2),
                                       sample_rate=rate,
                                       number_of_samples=count,
                                       factor=factor)[100:-100]
        assert np.max(np.abs(out)) < 10 ** (-99 / 20)


def zeros(*, number_of_samples, sample_rate):
    return np.zeros(number_of_samples)


def test_oversampling_defaults_to_two_seconds_at_44100():
    pytest.importorskip("scipy")
    assert music.render_oversampled(zeros).shape == (88200,)
    assert music.render_oversampled(zeros, duration=.01,
                                    sample_rate=48000).shape == (480,)


@pytest.mark.parametrize("kwargs", [
    {"sample_rate": 1, "number_of_samples": 4},
    {"factor": 16, "number_of_samples": 10},
])
def test_the_edges_of_what_oversampling_takes_are_taken(kwargs):
    pytest.importorskip("scipy")
    assert music.render_oversampled(zeros, **kwargs).shape == (
        kwargs["number_of_samples"],)


class Reached(Exception):
    """The renderer was called: nothing before it refused the render."""


def test_the_frame_bound_takes_exactly_fifty_million():
    pytest.importorskip("scipy")

    def reached(**kwargs):
        raise Reached

    with pytest.raises(Reached):
        music.render_oversampled(reached, number_of_samples=12_500_000,
                                 factor=4)
    with pytest.raises(ValueError, match="50 million"):
        music.render_oversampled(reached, number_of_samples=12_500_001,
                                 factor=4)


# --------------------------------------------------------------------------
# a fixed pitch
# --------------------------------------------------------------------------

def test_a_band_limited_note_defaults_to_two_seconds_of_220_hz_sine():
    assert np.array_equal(music.bandlimited_note(), music.bandlimited_note(
        220., duration=2., waveform="sine", number_of_samples=0,
        sample_rate=44100, table_size=16384, rolloff_hz=0.))
    assert music.bandlimited_note().shape == (88200,)
    assert not np.array_equal(music.bandlimited_note(),
                              music.bandlimited_note(table_size=16385))


@pytest.mark.parametrize("kwargs", [
    {"freq": .5},
    {"freq": 20000.},
    {"freq": .25, "sample_rate": 1, "number_of_samples": 8},
    {"table_size": 4},
])
def test_the_edges_of_what_a_band_limited_note_takes_are_taken(kwargs):
    kwargs.setdefault("number_of_samples", 64)
    assert np.isfinite(music.bandlimited_note(**kwargs)).all()


@pytest.mark.parametrize("size", (1000, 1001))
def test_a_table_is_built_at_the_size_asked_for(size):
    """A sine's table holds the sine, whatever its size, even or odd.

    At fifty entries a sample every read lands on an entry, so the note
    is the sine itself, sampled.
    """
    rate = 44100
    out = music.bandlimited_note(50 * rate / size, number_of_samples=40,
                                 sample_rate=rate, table_size=size)
    np.testing.assert_allclose(
        out, np.sin(2 * np.pi * 50 * np.arange(40) / size), atol=1e-9)


def test_a_cached_table_is_read_only():
    """The cache hands every caller the same array, which none may change."""
    table = bandlimited._filtered_table("sawtooth", 1000., 48000, 4096, 0.)
    assert not table.flags.writeable


def test_a_rolloff_given_fades_the_partials_it_reaches():
    """A 5 kHz sawtooth with a 6 kHz rolloff keeps its 4th partial, at
    20 kHz, at (22,050 - 20,000) / 6,000 of what it has without one."""
    plain = np.abs(np.fft.rfft(music.bandlimited_note(
        5000, duration=1, waveform="sawtooth")))
    faded = np.abs(np.fft.rfft(music.bandlimited_note(
        5000, duration=1, waveform="sawtooth", rolloff_hz=6000)))
    assert faded[20000] / plain[20000] == pytest.approx(2050 / 6000,
                                                        rel=1e-3)
    assert faded[15000] / plain[15000] == pytest.approx(1, rel=1e-3)


# --------------------------------------------------------------------------
# what the second run found unasserted
# --------------------------------------------------------------------------

def test_no_rolloff_is_the_default_even_a_hair_below_nyquist():
    """A sawtooth's second partial half a hertz below the Nyquist frequency
    is kept whole: by default nothing fades."""
    note = music.bandlimited_note(11024.75, waveform="sawtooth",
                                  number_of_samples=64)
    assert np.array_equal(note, music.bandlimited_note(
        11024.75, waveform="sawtooth", number_of_samples=64, rolloff_hz=0.))
    assert not np.array_equal(note, music.bandlimited_note(
        11024.75, waveform="sawtooth", number_of_samples=64, rolloff_hz=1.))


def test_more_than_2048_partials_is_refused():
    with pytest.raises(ValueError, match="max_harmonics"):
        bandlimited_frequency_path([1000.], max_harmonics=2049)


def test_the_work_bound_counts_only_the_partials_below_nyquist():
    """A 1 kHz path at 48 kHz has 24 partials below Nyquist, so 600,000
    samples of it are 14.4 million harmonic-samples, whatever the cap."""
    assert bandlimited_frequency_path(
        np.full(600_000, 1000.), sample_rate=48000, waveform="sine",
        max_harmonics=2048).shape == (600_000,)


def test_one_over_fifty_million_is_refused():
    """19 partials of 1250 Hz at 48 kHz, for 2,631,579 samples, and a
    render of 16,666,667 samples at three times the rate: each
    50,000,001."""
    with pytest.raises(ValueError, match="50 million"):
        bandlimited_frequency_path(np.full(2_631_579, 1250.),
                                   sample_rate=48000, waveform="sine")
    pytest.importorskip("scipy")

    def reached(**kwargs):
        raise Reached

    with pytest.raises(ValueError, match="50 million"):
        music.render_oversampled(reached, number_of_samples=16_666_667,
                                 factor=3)


def test_a_single_precision_render_is_filtered_in_double():
    signal = pytest.importorskip("scipy.signal")

    def single(*, number_of_samples, sample_rate):
        return np.sin(np.arange(number_of_samples) * .37).astype(np.float32)

    out = music.render_oversampled(single, number_of_samples=500,
                                   sample_rate=48000)
    raw = np.asarray(single(number_of_samples=2000, sample_rate=192000),
                     dtype=np.float64)
    from music.core.synths.oversampling import _decimation_filter

    np.testing.assert_array_equal(out, signal.resample_poly(
        raw, 1, 4, window=_decimation_filter(4))[:500])


def test_single_precision_frequencies_are_integrated_in_double():
    f = np.linspace(500, 9000, 2000)
    assert np.array_equal(
        polyblep_frequency_path(f.astype(np.float32), sample_rate=48000),
        polyblep_frequency_path(f.astype(np.float32).astype(np.float64),
                                sample_rate=48000))
