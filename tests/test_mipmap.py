"""Mipmapped tables, read against the Fourier series they are built from."""

import re

import numpy as np
import pytest

from music.core.synths import mipmap
from music.core.synths.frequency_path import bandlimited_frequency_path
from music.core.synths.mipmap import (
    PARTIALS, SIZES, MipmapOscillator, mipmap_frequency_path, tables,
)


@pytest.fixture(autouse=True)
def fresh_tables(monkeypatch):
    """Build the tables in each test, rather than read another test's.

    The module keeps them once built, so every test after the first
    would read tables built before it ran -- in a mutation run, before
    the mutant it judges was switched on.
    """
    monkeypatch.setattr(mipmap, "_TABLES", {})


def refused(message):
    """The exact refusal, not merely some refusal containing it."""
    return pytest.raises(ValueError, match=f"^{re.escape(message)}$")


def coefficient(waveform, n):
    """The n-th sine's coefficient, from the standard tables."""
    if waveform == "sawtooth":
        return 2 * (-1) ** (n + 1) / (np.pi * n)
    if n % 2 == 0:
        return 0.
    if waveform == "square":
        return 4 / (np.pi * n)
    return 8 * (-1) ** ((n - 1) // 2) / (np.pi * n) ** 2


def partial_sum(waveform, gains, cycles):
    """sum of gain * coefficient * sin(2 pi n cycles), partial by partial."""
    out = np.zeros_like(cycles, dtype=float)
    for n, gain in enumerate(gains, start=1):
        if gain:
            out += gain * coefficient(waveform, n) * np.sin(
                2 * np.pi * n * cycles)
    return out


def gains(fits):
    """Each partial's weight where ``fits`` partials would, per the docs."""
    counts = list(PARTIALS)
    j = max(i for i, c in enumerate(counts) if c <= fits)
    following = (counts[j + 1] if j + 1 < len(counts)
                 else 2 * counts[-1] - counts[-2])
    weight = min((fits - counts[j]) / (following - counts[j]), 1.)
    lower = counts[max(j - 1, 0)]
    out = np.zeros(counts[-1])
    out[:counts[j]] = weight
    out[:lower] = 1.
    if j == 0:
        out[0] = 1.
    return out


def test_the_counts_are_every_one_to_sixteen_then_four_an_octave():
    assert PARTIALS[:16] == tuple(range(1, 17))
    assert PARTIALS[16:] == (19, 23, 27, 32, 38, 45, 54, 64, 76, 91, 108,
                             128, 152, 181, 215, 256, 304, 362, 431, 512,
                             609, 724, 861, 1024)
    # Sixteen entries or more to a cycle of a table's top partial.
    assert SIZES == (4096,) * 32 + (8192,) * 4 + (16384,) * 4
    assert all(size & (size - 1) == 0 for size in SIZES)


@pytest.mark.parametrize("waveform", ("sawtooth", "square", "triangle"))
def test_each_table_is_its_partial_sum(waveform):
    rows = tables(waveform)
    assert [len(row) for row in rows] == [size + 1 for size in SIZES]
    assert all(row.dtype == np.float64 and not row.flags.writeable
               for row in rows)
    assert tables(waveform)[0].base is rows[0].base
    for level in (0, 1, 2, 15, 16, 31, 32, 35, 36, len(PARTIALS) - 1):
        size = SIZES[level]
        cycles = np.arange(size + 1) / size
        expected = partial_sum(waveform, [1.] * PARTIALS[level], cycles)
        np.testing.assert_allclose(rows[level], expected, atol=1e-12)


def test_an_unknown_waveform_is_refused():
    with refused("waveform must be sawtooth, square or triangle"):
        tables("sine")
    with refused("waveform must be sawtooth, square or triangle"):
        MipmapOscillator(waveform="sine")


@pytest.mark.parametrize("waveform", ("sawtooth", "square", "triangle"))
@pytest.mark.parametrize("step", (1, 2, 3, 20, 120, 600, 1000, 1365))
def test_a_fixed_pitch_reads_the_bracketing_tables_at_their_weights(
        waveform, step):
    """At a whole number of entries a sample, every read is an entry."""
    rate = 48000
    hz = step * rate / 4096
    fits = rate / 2 / hz
    out = mipmap_frequency_path(np.full(64, hz), sample_rate=rate,
                                waveform=waveform)
    cycles = (np.arange(64) * step % 4096) / 4096
    expected = partial_sum(waveform, gains(fits), cycles)
    np.testing.assert_allclose(out, expected, atol=1e-11)
    assert np.all(np.flatnonzero(gains(fits)) + 1 < fits)


@pytest.mark.parametrize("fits, bound", ((1.5, 3e-7), (3.4, 1e-6),
                                         (17.2, 3e-5), (100., 6e-4),
                                         (1300., 5e-3)))
def test_reading_between_entries_is_off_by_what_linear_reading_costs(
        fits, bound):
    """Next to an edge, a wave of many partials bends between entries.

    Most where a table has the fewest entries to its top partial's cycle:
    sixteen, for 256 partials and for 1024.
    """
    rate = 48000
    hz = rate / 2 / fits
    out = mipmap_frequency_path(np.full(4096, hz), sample_rate=rate,
                                waveform="square")
    cycles = (np.arange(4096) * hz / rate) % 1
    error = np.abs(out - partial_sum("square", gains(fits), cycles))
    assert error.max() < bound
    assert error.max() > bound / 10


@pytest.mark.parametrize("count", (2, 3, 16, 19, 1024))
def test_nothing_steps_as_the_pitch_crosses_a_table_s_count(count):
    rate = 48000
    below = MipmapOscillator(sample_rate=rate, phase_cycles=.3).render(
        [rate / 2 / (count - 1e-9)])
    at = MipmapOscillator(sample_rate=rate, phase_cycles=.3).render(
        [rate / 2 / count])
    assert below == pytest.approx(at, abs=1e-8)


def test_past_the_last_table_its_weight_grows_then_stays_whole():
    rate = 48000
    rows = tables("sawtooth")
    last, before = PARTIALS[-1], PARTIALS[-2]
    for fits, weight in ((last + (last - before) / 2, .5),
                         (2 * last - before, 1.), (5000., 1.)):
        out = MipmapOscillator(sample_rate=rate, phase_cycles=.25).render(
            [rate / 2 / fits])[0]
        expected = ((1 - weight) * rows[-2][SIZES[-2] // 4]
                    + weight * rows[-1][SIZES[-1] // 4])
        assert out == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("rate", (44100, 48000, 96000))
@pytest.mark.parametrize("waveform", ("sawtooth", "square"))
def test_nothing_folds_at_any_pitch(rate, waveform):
    """Less than 1e-8 of the energy off the harmonics, as documented.

    The most is 9.4e-9, a 101 Hz sawtooth at 44.1 kHz, where the reads
    between entries of a table of hundreds of partials stray most.
    """
    for fraction in (.0011, .00229, .0092, .0367, .1467, .20835, .3):
        hz = int(round(fraction * rate))
        power = np.abs(np.fft.rfft(mipmap_frequency_path(
            np.full(rate, float(hz)), sample_rate=rate,
            waveform=waveform))) ** 2
        on = np.zeros(len(power), dtype=bool)
        on[np.arange(hz, rate // 2, hz)] = True
        assert power[~on].sum() / power.sum() < 1e-8


def test_alone_below_nyquist_a_fundamental_is_the_series_sine():
    rate = 48000
    path = np.linspace(12500, 23000, 4000)
    out = mipmap_frequency_path(path, sample_rate=rate)
    summed = bandlimited_frequency_path(path, sample_rate=rate,
                                        waveform="sawtooth", transition=0)
    np.testing.assert_allclose(out, summed, atol=2e-6)


def test_a_moving_pitch_is_integrated_trapezoidally():
    rate = 48000
    path = np.linspace(13000, 20000, 3000)
    out = mipmap_frequency_path(path, sample_rate=rate, phase_cycles=.1)
    cycles = .1 + np.r_[0., np.cumsum((path[:-1] + path[1:]) / 2)] / rate
    np.testing.assert_allclose(out, 2 / np.pi * np.sin(2 * np.pi * cycles),
                               atol=2e-6)


def test_blocks_join_as_one_render_and_an_empty_one_changes_nothing():
    rate = 48000
    path = np.geomspace(40, 20000, 20000)
    whole = mipmap_frequency_path(path, sample_rate=rate, waveform="square")
    oscillator = MipmapOscillator(sample_rate=rate, waveform="square")
    pieces = []
    for start, stop in ((0, 1), (1, 1), (1, 700), (700, 701),
                        (701, 13000), (13000, 20000)):
        pieces.append(oscillator.render(path[start:stop]))
    assert pieces[1].shape == (0,) and pieces[1].dtype == np.float64
    np.testing.assert_allclose(np.concatenate(pieces), whole, atol=1e-8)


def test_a_refused_block_leaves_the_state_as_it_was():
    rate = 48000
    oscillator = MipmapOscillator(sample_rate=rate)
    first = oscillator.render([1000., 1000.])
    with refused("frequencies must be one-dimensional"):
        oscillator.render([[1000.]])
    for bad in ([0.], [-1.], [np.nan], [np.inf], [rate / 2]):
        with refused("frequencies must be finite, positive, < Nyquist"):
            oscillator.render(bad)
    resumed = oscillator.render([1000.])
    whole = mipmap_frequency_path([1000.] * 3, sample_rate=rate)
    np.testing.assert_allclose(np.r_[first, resumed], whole, atol=1e-12)


@pytest.mark.parametrize("hz", (440., 10.))
def test_a_phase_a_hair_below_one_reads_the_table_s_start(hz):
    """At 10 Hz the richest table is read, whose end is the array's.

    The phase times a table's size is exact, a power of two, so even the
    largest phase below one reads inside the table.
    """
    out = MipmapOscillator(phase_cycles=float(np.nextafter(1., 0.))).render(
        [hz])
    assert out[0] == pytest.approx(0., abs=1e-9)


def test_a_pitch_below_one_hertz_is_taken():
    out = MipmapOscillator(sample_rate=8000, phase_cycles=.25).render(
        [.5, .5])
    rows = tables("sawtooth")
    assert out[0] == pytest.approx(rows[-1][len(rows[-1]) // 4], abs=1e-12)


def test_single_precision_frequencies_are_integrated_in_double():
    path = np.linspace(500, 9000, 2000).astype(np.float32)
    np.testing.assert_array_equal(
        mipmap_frequency_path(path, sample_rate=48000),
        mipmap_frequency_path(path.astype(np.float64), sample_rate=48000))


def test_the_block_bound_is_five_million(monkeypatch):
    assert mipmap._MAX_BLOCK_SAMPLES == 5_000_000
    monkeypatch.setattr(mipmap, "_MAX_BLOCK_SAMPLES", 10)
    oscillator = MipmapOscillator()
    assert oscillator.render(np.full(10, 440.)).shape == (10,)
    with refused("frequency block exceeds 5 million frames"):
        oscillator.render(np.full(11, 440.))


@pytest.mark.parametrize("rate", (8000, 192000))
def test_the_edges_of_the_rates_are_taken(rate):
    out = MipmapOscillator(sample_rate=rate).render([rate / 2 - 1e-6])
    assert out.shape == (1,)


@pytest.mark.parametrize("kwargs", (
    {"sample_rate": 7999}, {"sample_rate": 192001}, {"sample_rate": 48000.},
    {"sample_rate": True},
))
def test_what_the_oscillator_refuses_for_a_rate(kwargs):
    with refused("sample_rate must be 8000 to 192000 Hz"):
        MipmapOscillator(**kwargs)


@pytest.mark.parametrize("phase", (-.1, 1., np.nan, "0"))
def test_what_the_oscillator_refuses_for_a_phase(phase):
    with refused("phase_cycles must be finite in [0, 1)"):
        MipmapOscillator(phase_cycles=phase)


def test_defaults_are_a_sawtooth_at_44100_from_phase_zero():
    oscillator = MipmapOscillator()
    assert oscillator.sample_rate == 44100
    assert oscillator.waveform == "sawtooth"
    out = mipmap_frequency_path([11025.])
    assert out[0] == pytest.approx(0., abs=1e-12)
    np.testing.assert_array_equal(oscillator.render([11025.] * 3),
                                  mipmap_frequency_path([11025.] * 3))
