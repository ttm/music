"""Tests of the package's central claim: that synthesis matches the model.

The rest of the suite checks that functions run and return arrays of the
expected shape.  These check that the *samples* are right — that each routine
reproduces the closed-form expression from the MASS article it cites, rather
than merely something of the correct length.

References
----------
.. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time
       representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)
"""

import numpy as np
import pytest

import music
from music.core.functions import normalize_mono
from music.legacy.tables import Basic
from music.tables import PrimaryTables
from music.utils import (WAVEFORMS, WAVEFORM_SAWTOOTH, WAVEFORM_SINE,
                         WAVEFORM_SQUARE, WAVEFORM_TRIANGULAR,
                         waveform_table)

SAMPLE_RATE = 44100


def _spectral_slope(samples, sample_rate=SAMPLE_RATE, low=100, high=8000):
    """Fit the magnitude spectrum in dB against log2(f): dB per octave."""
    spectrum = np.abs(np.fft.rfft(samples))
    freqs = np.fft.rfftfreq(len(samples), 1 / sample_rate)
    band = (freqs >= low) & (freqs <= high)
    slope, _ = np.polyfit(
        np.log2(freqs[band]), 20 * np.log10(spectrum[band] + 1e-30), 1
    )
    return slope


# --------------------------------------------------------------------------
# Sample-exact agreement with the synthesis equations
# --------------------------------------------------------------------------

@pytest.mark.parametrize("freq", [55.0, 220.0, 443.7, 1000.0])
@pytest.mark.parametrize("table", [WAVEFORM_SINE, WAVEFORM_TRIANGULAR])
def test_note_matches_the_lookup_equation_sample_for_sample(freq, table):
    """note() is table[floor(n * f * L / fs) mod L]."""
    duration = 0.05
    produced = music.note(freq=freq, duration=duration, waveform_table=table)

    count = int(duration * SAMPLE_RATE)
    n = np.arange(count)
    length = len(table)
    gamma = (n * freq * length / SAMPLE_RATE).astype(np.int64)
    expected = table[gamma % length]

    assert produced.shape == expected.shape
    assert np.array_equal(produced, expected)


def test_note_with_vibrato_matches_the_accumulated_phase_equation():
    """The vibrato pattern is folded into the lookup, not applied after it:
    the table index is the cumulative sum of a per-sample frequency."""
    freq, vibrato_freq, max_pitch_dev, duration = 220.0, 6.0, 2.0, 0.05
    produced = music.note_with_vibrato(
        freq=freq, duration=duration, vibrato_freq=vibrato_freq,
        max_pitch_dev=max_pitch_dev,
    )

    count = int(duration * SAMPLE_RATE)
    n = np.arange(count)
    vibrato_length = len(WAVEFORM_SINE)
    gamma_v = (n * vibrato_freq * vibrato_length / SAMPLE_RATE).astype(
        np.int64
    )
    pattern = WAVEFORM_SINE[gamma_v % vibrato_length]
    instantaneous = freq * 2.0 ** (pattern * max_pitch_dev / 12)

    length = len(WAVEFORM_TRIANGULAR)
    gamma = np.cumsum(instantaneous * (length / SAMPLE_RATE)).astype(np.int64)
    expected = WAVEFORM_TRIANGULAR[gamma % length]

    assert np.array_equal(produced, expected)


def test_vibrato_sweeps_exactly_the_requested_number_of_semitones():
    """max_pitch_dev is in semitones, so the instantaneous frequency spans
    freq * 2 ** (+/- max_pitch_dev / 12)."""
    freq, max_pitch_dev = 440.0, 3.0
    lowest = freq * 2 ** (-max_pitch_dev / 12)
    highest = freq * 2 ** (max_pitch_dev / 12)

    # One vibrato cycle, sampled densely enough to see both extremes.
    samples = music.note_with_vibrato(
        freq=freq, duration=2.0, vibrato_freq=0.5,
        max_pitch_dev=max_pitch_dev, waveform_table=WAVEFORM_SINE,
    )
    analytic = np.abs(np.fft.rfft(samples))
    freqs = np.fft.rfftfreq(len(samples), 1 / SAMPLE_RATE)
    energetic = freqs[analytic > analytic.max() * 0.02]

    assert energetic.min() == pytest.approx(lowest, rel=0.03)
    assert energetic.max() == pytest.approx(highest, rel=0.03)


# --------------------------------------------------------------------------
# Amplitude envelopes, in the decibel terms their parameters are stated in
# --------------------------------------------------------------------------

@pytest.mark.parametrize("max_db_dev", [3.0, 6.0, 12.0])
def test_tremolo_envelope_spans_the_requested_decibel_range(max_db_dev):
    """tremolo() is 10 ** (pattern * max_db_dev / 20), so the envelope
    swings between +/- max_db_dev around unity gain."""
    envelope = music.tremolo(
        duration=1.0, tremolo_freq=5.0, max_db_dev=max_db_dev,
    )
    assert 20 * np.log10(envelope.max()) == pytest.approx(max_db_dev, abs=0.1)
    assert 20 * np.log10(envelope.min()) == pytest.approx(-max_db_dev, abs=0.1)


@pytest.mark.parametrize("sustain_level", [-3.0, -5.0, -20.0])
def test_adsr_sustain_plateau_sits_at_the_requested_decibels(sustain_level):
    """The sustain stage is a plateau at 10 ** (sustain_level / 20)."""
    envelope = music.adsr(
        envelope_duration=1.0, attack_duration=20, decay_duration=20,
        sustain_level=sustain_level, release_duration=50,
    )
    attack = int(20 * SAMPLE_RATE * 0.001)
    decay = int(20 * SAMPLE_RATE * 0.001)
    release = int(50 * SAMPLE_RATE * 0.001)
    plateau = envelope[attack + decay:len(envelope) - release]

    expected = 10 ** (sustain_level / 20)
    assert np.allclose(plateau, expected)
    assert envelope.max() == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------
# Noise colour
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "noise_type, db_per_octave",
    [("brown", -6), ("pink", -3), ("white", 0),
     ("blue", 3), ("violet", 6), ("black", -12)],
)
def test_noise_colour_has_its_documented_spectral_slope(noise_type,
                                                        db_per_octave):
    """Each named colour is a stated gain per octave; measure it back."""
    np.random.seed(0)
    samples = music.noise(noise_type, duration=4)
    assert _spectral_slope(samples) == pytest.approx(db_per_octave, abs=0.2)


def test_numeric_noise_type_is_taken_as_decibels_per_octave():
    """The docstring says ntype=3.5 means 3.5 dB gain per octave."""
    np.random.seed(0)
    assert _spectral_slope(music.noise(3.5, duration=4)) == pytest.approx(
        3.5, abs=0.2
    )


# --------------------------------------------------------------------------
# Spatialisation
# --------------------------------------------------------------------------

def test_localize_delays_and_attenuates_the_far_ear_by_the_geometry():
    """A source to the right reaches the right ear first and louder. Both
    the delay and the amplitude ratio follow from the ear positions."""
    x, y, zeta, air_temp = 0.5, 0.1, 0.215, 20
    click = np.zeros(2000)
    click[0] = 1.0

    stereo = music.localize(sonic_vector=click, x=x, y=y, zeta=zeta,
                            air_temp=air_temp, sample_rate=SAMPLE_RATE)
    left, right = stereo

    speed = 331.3 + 0.606 * air_temp
    dist_right = np.sqrt((x - zeta / 2) ** 2 + y ** 2)
    dist_left = np.sqrt((x + zeta / 2) ** 2 + y ** 2)
    delay = int((dist_left - dist_right) / speed * SAMPLE_RATE)

    assert delay > 0, "the far (left) ear must lag for a source on the right"
    assert np.argmax(right) == 0
    assert np.argmax(left) == delay
    assert left.max() == pytest.approx(dist_right / dist_left, rel=1e-9)


# --------------------------------------------------------------------------
# Quantisation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bit_depth", [8, 16, 32])
def test_wav_round_trip_stays_within_one_quantisation_step(bit_depth,
                                                           tmp_path):
    """Writing and reading back may only cost the quantiser's own step.

    The writers always normalize, so the reference is the signal as it is
    stored, not the raw input.
    """
    signal = music.note(440, 0.05, waveform_table=WAVEFORM_SINE)
    stored = normalize_mono(signal, True)
    path = tmp_path / f"rt{bit_depth}.wav"

    music.write_wav_mono(signal, filename=str(path), bit_depth=bit_depth)
    restored = music.read_wav(str(path))

    step = 1.0 / 2 ** (bit_depth - 1)
    assert restored.shape == stored.shape
    assert np.abs(restored - stored).max() <= step


@pytest.mark.parametrize("bit_depth", [8, 16, 32])
def test_wav_round_trip_is_unity_gain(bit_depth, tmp_path):
    """Exactly representable levels must come back unchanged: the writer's
    scale and read_wav's divisor have to agree.

    These levels are symmetric with zero mean and unit peak, so the writer's
    normalization is the identity on them and any difference is the
    quantiser's. +1.0 is the one exception, two's complement having one
    fewer positive value than negative.
    """
    levels = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    assert np.array_equal(normalize_mono(levels, True), levels)

    path = tmp_path / f"levels{bit_depth}.wav"
    music.write_wav_mono(levels, filename=str(path), bit_depth=bit_depth)
    restored = music.read_wav(str(path))

    assert np.array_equal(restored[:4], levels[:4])
    assert restored[4] == pytest.approx(1.0, abs=1.0 / 2 ** (bit_depth - 1))


def test_stereo_wav_round_trip_preserves_channel_balance(tmp_path):
    """The two channels must keep their relative level through a write."""
    left = music.note(440, 0.05)
    stereo = np.vstack((left, left * 0.5))
    path = tmp_path / "balance.wav"

    music.write_wav_stereo(stereo, filename=str(path), bit_depth=16)
    restored = music.read_wav(str(path))

    ratio = np.abs(restored[0]).max() / np.abs(restored[1]).max()
    assert ratio == pytest.approx(2.0, rel=0.01)


# --------------------------------------------------------------------------
# Waveform lookup tables
# --------------------------------------------------------------------------

def _ideal(kind, size):
    """The continuous waveform sampled at phase k / size."""
    phase = np.arange(size) / size
    return {
        "sine": np.sin(2 * np.pi * phase),
        "sawtooth": 2 * phase - 1,
        "square": np.where(phase < .5, -1.0, 1.0),
        "triangle": np.where(phase <= .5, -1 + 4 * phase, 3 - 4 * phase),
    }[kind]


@pytest.mark.parametrize("kind", WAVEFORMS)
@pytest.mark.parametrize("size", [4, 16, 15, 2048, 2049, 16384])
def test_waveform_table_is_exact_at_any_size(kind, size):
    """Each table must equal the continuous waveform sampled at its phase.

    Regression: the tables were built by halves. That made the sawtooth
    step 2/(size-1) instead of 2/size, gave the triangle a flat two-sample
    top that never reached full amplitude, and returned one sample fewer
    than asked for at odd sizes.
    """
    table = waveform_table(kind, size)

    assert len(table) == size
    assert np.allclose(table, _ideal(kind, size), atol=1e-15)
    assert table.min() >= -1.0 and table.max() <= 1.0


@pytest.mark.parametrize("kind", WAVEFORMS)
def test_waveform_table_starts_at_the_bottom_of_its_period(kind):
    """Every table opens at phase zero, which is -1 for all but the sine."""
    table = waveform_table(kind, 1024)
    expected = 0.0 if kind == "sine" else -1.0
    assert table[0] == pytest.approx(expected, abs=1e-15)


def test_triangle_reaches_full_amplitude_at_its_midpoint():
    """Regression: the old triangle peaked at 1 - 2/size, never at 1."""
    size = 1024
    table = waveform_table("triangle", size)
    assert table[size // 2] == pytest.approx(1.0, abs=1e-15)
    assert table.max() == pytest.approx(1.0, abs=1e-15)


def test_sawtooth_ramps_by_exactly_one_period_per_table():
    """Consecutive samples rise by 2/size, so one period spans the table."""
    size = 1024
    table = waveform_table("sawtooth", size)
    assert np.allclose(np.diff(table), 2 / size)


def test_every_table_source_agrees():
    """Regression: three separate implementations of the same four tables
    had drifted apart at the triangle's peak sample."""
    size = len(WAVEFORM_TRIANGULAR)
    tables = PrimaryTables(size=size)
    legacy = Basic(size=size)

    assert np.array_equal(tables.sine, WAVEFORM_SINE)
    assert np.array_equal(tables.saw, WAVEFORM_SAWTOOTH)
    assert np.array_equal(tables.square, WAVEFORM_SQUARE)
    assert np.array_equal(tables.triangle, WAVEFORM_TRIANGULAR)

    for name in ("sine", "saw", "square", "triangle"):
        assert np.array_equal(getattr(legacy, name), getattr(tables, name))


def test_waveform_table_rejects_nonsense():
    with pytest.raises(ValueError, match="unknown waveform"):
        waveform_table("ocarina")
    with pytest.raises(ValueError, match="size must be positive"):
        waveform_table("sine", 0)
