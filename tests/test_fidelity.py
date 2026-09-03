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

import warnings

import numpy as np
import pytest

import music
from music.core.functions import normalize_mono
from music.legacy.tables import Basic
from music.tables import PrimaryTables
from music.utils import _integrate_phase
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

@pytest.mark.parametrize("bit_depth", [8, 16, 24, 32])
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


@pytest.mark.parametrize("bit_depth", [8, 16, 24, 32])
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


# --------------------------------------------------------------------------
# Phase integration (issue #102)
# --------------------------------------------------------------------------

SAMPLE_RATE = 44100
TABLE_LENGTH = 16384


def _phase_error(phase, increment, count, length=TABLE_LENGTH):
    """How far a computed phase is from the exact one, in table entries.

    Both are folded into one period, so a difference of nearly a whole
    period is really a small one across the fold.
    """
    exact = (np.arange(1, count + 1, dtype=np.float64) * increment) % length
    difference = np.abs(phase - exact)
    return np.minimum(difference, length - difference).max()


def test_phase_integration_does_not_drift_with_the_length_of_the_render():
    """The property, not a magnitude.

    ``np.cumsum`` accumulates into a total that keeps growing, so each
    addition loses low bits against a larger running value and the error
    grows with the render -- in one direction, so it is drift rather
    than noise. What replaced it folds the total into one table period
    as it goes, and its error stays where it starts.
    """
    increment = 200.0 * TABLE_LENGTH / SAMPLE_RATE
    drifting, steady = {}, {}
    for seconds in (5, 30):
        count = seconds * SAMPLE_RATE
        increments = np.full(count, increment)
        drifting[seconds] = _phase_error(
            np.cumsum(increments) % TABLE_LENGTH, increment, count)
        steady[seconds] = _phase_error(
            _integrate_phase(increments, TABLE_LENGTH), increment, count)

    # Six times the render, and the old way is orders of magnitude worse
    assert drifting[30] > 100 * drifting[5]

    # while this one has barely moved, and is small in absolute terms
    assert steady[30] < 2 * steady[5]
    assert steady[30] < 1e-5


def test_integrating_phase_agrees_with_the_index_the_caller_would_take():
    """The routines truncate the phase and index a table with it. The
    fold has to give the same entry the caller's own ``% length`` did,
    or every rendered sample moves."""
    increment = 200.0 * TABLE_LENGTH / SAMPLE_RATE
    increments = np.full(1000, increment)

    folded = _integrate_phase(increments, TABLE_LENGTH).astype(np.int64)
    unfolded = np.cumsum(increments).astype(np.int64) % TABLE_LENGTH
    assert np.array_equal(folded, unfolded)


def test_both_paths_of_the_integrator_agree():
    """Short inputs skip the blocking entirely; the two paths must not
    disagree where they overlap.

    Compared across the fold, because a phase just under a whole period
    and one just over zero are neighbours: a handful of samples land
    there, and a plain subtraction reports the difference between them
    as nearly a whole period rather than as the 1e-8 it is.
    """
    increment = 200.0 * TABLE_LENGTH / SAMPLE_RATE
    increments = np.full(5000, increment)

    short = _integrate_phase(increments, TABLE_LENGTH, block=10_000)
    blocked = _integrate_phase(increments, TABLE_LENGTH, block=512)

    difference = np.abs(short - blocked)
    across_the_fold = np.minimum(difference, TABLE_LENGTH - difference)
    assert across_the_fold.max() < 1e-6


# --------------------------------------------------------------------------
# Synthesis routines that had only their shape checked
#
# Each of these was covered by a test that asserted a length and a range,
# which silence and white noise both satisfy. What follows checks the
# samples against the equation the docstring describes, in the style of
# test_note_matches_the_lookup_equation_sample_for_sample above.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("phase", [0.0, np.pi / 2, np.pi, 2 * np.pi])
def test_note_with_phase_offsets_the_lookup_by_the_phase(phase):
    """gamma = floor(phase * L / 2pi + n * f * L / fs), read mod L."""
    freq, duration = 220.0, 0.05
    produced = music.note_with_phase(freq=freq, duration=duration,
                                     phase=phase,
                                     waveform_table=WAVEFORM_SINE)

    count = int(duration * SAMPLE_RATE)
    length = len(WAVEFORM_SINE)
    i0 = phase * length / (2 * np.pi)
    gamma = (i0 + np.arange(count) * freq * length
             / SAMPLE_RATE).astype(np.int64)
    expected = WAVEFORM_SINE[gamma % length]

    assert np.array_equal(produced, expected)


def test_a_phase_of_zero_is_the_plain_note():
    """The two routines describe the same sound when the phase is zero,
    so they must render the same samples, not merely similar ones."""
    plain = music.note(freq=220, duration=0.05,
                       waveform_table=WAVEFORM_SINE)
    phased = music.note_with_phase(freq=220, duration=0.05, phase=0,
                                   waveform_table=WAVEFORM_SINE)
    assert np.array_equal(plain, phased)


def test_a_phase_of_a_quarter_turn_advances_a_quarter_of_the_table():
    """The offset is in the table, not in time: a quarter turn starts the
    lookup a quarter of the way through the period."""
    length = len(WAVEFORM_SINE)
    phased = music.note_with_phase(freq=SAMPLE_RATE / length, duration=0.05,
                                   phase=np.pi / 2,
                                   waveform_table=WAVEFORM_SINE)
    assert phased[0] == WAVEFORM_SINE[length // 4]


def test_note_with_fm_matches_the_modulated_lookup_equation():
    """The modulator indexes its own table, sets the instantaneous
    frequency, and that is integrated into the carrier's lookup."""
    freq, fm, deviation, duration = 220.0, 30.0, 40.0, 0.05
    produced = music.note_with_fm(freq=freq, duration=duration, fm=fm,
                                  max_fm_deviation=deviation,
                                  waveform_table=WAVEFORM_SINE,
                                  fm_waveform_table=WAVEFORM_SINE)

    count = int(duration * SAMPLE_RATE)
    samples = np.arange(count)
    modulator_length = len(WAVEFORM_SINE)
    gamma_m = (samples * fm * modulator_length
               / SAMPLE_RATE).astype(np.int64)
    modulator = WAVEFORM_SINE[gamma_m % modulator_length]

    instantaneous = freq + modulator * deviation
    length = len(WAVEFORM_SINE)
    gamma = _integrate_phase(instantaneous * length / SAMPLE_RATE,
                             length).astype(np.int64)
    expected = WAVEFORM_SINE[gamma % length]

    assert np.array_equal(produced, expected)


def test_fm_with_no_deviation_is_a_steady_tone():
    """Zero deviation leaves the instantaneous frequency at `freq`, so
    the modulator cannot appear in the output at all. The old test for
    this routine asserted only a length and a range, which silence
    satisfies."""
    steady = music.note_with_fm(freq=440, duration=0.2, fm=7,
                                max_fm_deviation=0,
                                waveform_table=WAVEFORM_SINE)
    spectrum = np.abs(np.fft.rfft(steady))
    freqs = np.fft.rfftfreq(len(steady), 1 / SAMPLE_RATE)

    assert freqs[np.argmax(spectrum)] == pytest.approx(440, abs=6)
    # Nothing at the modulation rate, which a real deviation would put there.
    at_modulator = spectrum[np.argmin(np.abs(freqs - 7))]
    assert at_modulator < spectrum.max() / 1000


def test_fm_sweeps_the_carrier_by_the_deviation_it_was_given():
    """A slow, deep modulation puts energy across freq +/- deviation and
    essentially none outside it."""
    freq, deviation = 2000.0, 400.0
    swept = music.note_with_fm(freq=freq, duration=1.0, fm=0.5,
                               max_fm_deviation=deviation,
                               waveform_table=WAVEFORM_SINE)
    spectrum = np.abs(np.fft.rfft(swept))
    freqs = np.fft.rfftfreq(len(swept), 1 / SAMPLE_RATE)

    inside = ((freqs >= freq - deviation - 20)
              & (freqs <= freq + deviation + 20))
    band = (freqs > 100) & (freqs < 6000)
    energy_inside = (spectrum[inside] ** 2).sum()
    energy_in_band = (spectrum[band] ** 2).sum()
    assert energy_inside / energy_in_band > 0.95


@pytest.mark.parametrize("method, alpha", [("exp", 1), ("exp", 2),
                                           ("lin", 1)])
def test_glissando_follows_its_documented_frequency_law(method, alpha):
    """Exponential glissando is f0 * (f1/f0) ** (n/N) ** alpha; linear is
    the straight line between the two."""
    start, end, duration = 220.0, 440.0, 0.05
    produced = music.note_with_glissando(start_freq=start, end_freq=end,
                                         duration=duration, alpha=alpha,
                                         method=method,
                                         waveform_table=WAVEFORM_SINE)

    count = int(duration * SAMPLE_RATE)
    samples = np.arange(count)
    if method == "exp":
        instantaneous = start * (end / start) ** (
            (samples / (count - 1)) ** alpha)
    else:
        # Grouped as the implementation groups it. Association matters
        # here: (end - start) * samples / (count - 1) and
        # (end - start) * (samples / (count - 1)) differ in the last
        # bit, and the lookup index is an integer floor, so one bit is
        # enough to change a sample. That sensitivity is a property of
        # table synthesis worth stating rather than tolerating away.
        instantaneous = start + (end - start) * samples / (count - 1)
    length = len(WAVEFORM_SINE)
    gamma = _integrate_phase(instantaneous * length / SAMPLE_RATE,
                             length).astype(np.int64)
    expected = WAVEFORM_SINE[gamma % length]

    assert np.array_equal(produced, expected)


def test_a_glissando_between_equal_frequencies_is_a_plain_tone():
    """Nothing to slide between, so the pitch must not move."""
    flat = music.note_with_glissando(start_freq=440, end_freq=440,
                                     duration=0.2,
                                     waveform_table=WAVEFORM_SINE)
    spectrum = np.abs(np.fft.rfft(flat))
    freqs = np.fft.rfftfreq(len(flat), 1 / SAMPLE_RATE)
    assert freqs[np.argmax(spectrum)] == pytest.approx(440, abs=6)


@pytest.mark.parametrize("sample_rate", [22050, 44100, 48000])
def test_trill_lasts_the_time_it_was_asked_for_at_any_rate(sample_rate):
    """Regression: the note length and the loop bound were hardcoded to
    44100 while `sample_rate` was declared, documented and passed to
    note(). A trill asked for at 22050 rendered two seconds for every
    one, at half the note rate. The old test asserted only that some
    audio came back.
    """
    duration = 0.5
    out = music.trill(freqs=[400, 500], notes_per_second=8,
                      duration=duration, sample_rate=sample_rate)
    assert len(out) == pytest.approx(duration * sample_rate,
                                     abs=sample_rate / 8)


def test_trill_alternates_between_the_frequencies_it_was_given():
    """Each note in turn, which is what makes it a trill rather than a
    sequence of identical notes."""
    notes_per_second = 4
    out = music.trill(freqs=[400, 1200], notes_per_second=notes_per_second,
                      duration=1.0)
    # Only whole notes are rendered, so the trill is three notes rather
    # than four: the loop stops when the next one would not fit.
    note_length = SAMPLE_RATE // notes_per_second
    assert len(out) == 3 * note_length

    def peak(signal):
        spectrum = np.abs(np.fft.rfft(signal))
        return np.fft.rfftfreq(len(signal), 1 / SAMPLE_RATE)[
            np.argmax(spectrum)]

    # A window inside each note, clear of the ADSR release at its end.
    window = note_length // 2
    peaks = [peak(out[i * note_length:i * note_length + window])
             for i in range(3)]
    assert peaks[0] == pytest.approx(400, abs=40)
    assert peaks[1] == pytest.approx(1200, abs=60)
    assert peaks[2] == pytest.approx(400, abs=40)


# --------------------------------------------------------------------------
# Filters and envelopes, checked against the curves they describe
# --------------------------------------------------------------------------

@pytest.mark.parametrize("trans_dev", [-20.0, -6.0, 6.0, 20.0])
@pytest.mark.parametrize("alpha", [1.0, 2.0])
def test_loud_is_the_decibel_curve_it_documents(trans_dev, alpha):
    """e = 10 ** ((n/N) ** alpha * trans_dev / 20), sample for sample.

    The existing test checked the two endpoints, which any monotonic
    curve between them would satisfy. This checks the curve.
    """
    from music.core.filters.loud import loud

    count = 512
    produced = loud(trans_dev=trans_dev, alpha=alpha, method="exp", to=1,
                    number_of_samples=count)

    # The divisor is the last index, not the count: the curve reaches
    # its full deviation at the final sample rather than one step short.
    samples = np.arange(count)
    expected = 10 ** ((samples / (count - 1)) ** alpha * trans_dev / 20)

    assert np.allclose(produced, expected, rtol=0, atol=1e-12)


def test_loud_with_no_deviation_is_unity_gain():
    """Zero decibels of transition must leave a signal alone, exactly."""
    from music.core.filters.loud import loud

    tone = music.note(freq=220, duration=0.05)
    unchanged = loud(trans_dev=0, method="exp", sonic_vector=tone)
    assert np.array_equal(unchanged, tone)


def test_a_fade_out_ends_where_its_decibels_say():
    """fade(db=-80) must arrive at -80 dB, not merely get quieter."""
    from music.core.filters.fade import fade

    envelope = fade(db=-80, number_of_samples=1024, fade_out=True, perc=0)
    assert envelope[0] == pytest.approx(1.0)
    assert envelope[-1] == pytest.approx(10 ** (-80 / 20), rel=1e-3)


def test_the_last_percent_of_a_fade_runs_to_true_silence():
    """`perc` is why a fade ends at zero rather than at -80 dB.

    A decibel curve never reaches zero, and a signal cut off at -80 dB
    still steps to silence at the boundary, which is a click. The last
    `perc` of the fade is linear to exactly zero for that reason, and
    the two-sided behaviour is easy to lose in a refactor: at 8 samples
    one percent rounds to nothing and the fade ends at -80 dB instead.
    """
    from music.core.filters.fade import fade

    with_tail = fade(db=-80, number_of_samples=1024, fade_out=True)
    assert with_tail[-1] == 0.0
    # The decibel curve still passes through -80 dB, where the linear
    # run to zero takes over.
    assert np.isclose(with_tail, 10 ** (-80 / 20), rtol=1e-9).any()
    # And it is a run rather than a step: the last ten samples descend
    # to zero in even increments, which is what makes it linear.
    tail = with_tail[-10:]
    steps = np.diff(tail)
    assert np.all(steps < 0)
    assert np.allclose(steps, steps[0])


def test_a_fade_in_is_the_fade_out_reversed():
    """The two are the same curve read in opposite directions, so one
    must be the other's mirror rather than merely also monotonic."""
    from music.core.filters.fade import fade

    out = fade(db=-40, number_of_samples=512, fade_out=True, perc=0)
    into = fade(db=-40, number_of_samples=512, fade_out=False, perc=0)
    assert np.allclose(out, into[::-1], atol=1e-12)


def test_reverb_decays_by_the_decibels_it_was_given():
    """The impulse response's envelope is 10 ** ((decay/20) * n/(N-1)),
    so the last incidence sits `decay` dB below the first.

    The existing test rendered one and checked its length.
    """
    np.random.seed(0)
    decay, duration, sample_rate = -60.0, 0.5, 8000
    impulse = music.reverb(duration=duration, first_phase_duration=0.05,
                           decay=decay, sample_rate=sample_rate)

    count = len(impulse)
    window = count // 10
    # The response is sparse incidences under a decaying envelope, so the
    # peak of a window follows the envelope where individual samples do
    # not.
    early = np.abs(impulse[1:window]).max()
    late = np.abs(impulse[-window:]).max()
    measured = 20 * np.log10(late / early)

    assert measured == pytest.approx(decay, abs=12)
    assert late < early


@pytest.mark.parametrize("duration", [0.25, 0.5, 1.0, 2.0])
def test_stretches_gives_each_repeat_the_duration_it_asked_for(duration):
    """Each repeat is resampled to last `duration` seconds. The existing
    test checked the total against the sum, which a single wrong segment
    compensated by another would satisfy.
    """
    sample_rate = 8000
    fragment = music.note(freq=220, duration=1.0, sample_rate=sample_rate)
    out = music.stretches(fragment, durations=(duration,),
                          sample_rate=sample_rate)

    assert len(out) == pytest.approx(duration * sample_rate, abs=2)


def test_stretches_squeezes_rather_than_truncates():
    """A half-length repeat must be the whole fragment at twice the
    speed, not the first half of it: the last sample of the fragment has
    to survive into the squeezed copy."""
    sample_rate = 8000
    fragment = music.note(freq=110, duration=1.0, sample_rate=sample_rate)
    squeezed = music.stretches(fragment, durations=(0.5,),
                               sample_rate=sample_rate)

    # Reading every other sample is what halving the duration means.
    assert np.array_equal(squeezed, fragment[::2][:len(squeezed)])


# --------------------------------------------------------------------------
# The localization family, against the geometry it does model
#
# These check the interaural cues the code computes. They are not tests
# that localization is "correct": there is no head-related transfer
# function anywhere in this package, so elevation and front-versus-back
# are not modelled at all, and `localize2` says in its own docstring that
# its calculations are "not standard and are only to illustrate the
# method".
# --------------------------------------------------------------------------

def _interaural_lag(left, right, sample_rate=SAMPLE_RATE):
    """How far `left` lags `right`, in seconds, by cross-correlation.

    Positive means the left channel arrives later. Calibrated against a
    known shift in test_the_lag_measurement_reads_a_known_shift below,
    because a sign convention assumed rather than checked is how a
    measurement ends up confirming whatever it was pointed at.
    """
    count = min(len(left), len(right))
    a = left[:count] - left[:count].mean()
    b = right[:count] - right[:count].mean()
    correlation = np.correlate(a, b, "full")
    return (np.argmax(correlation) - (count - 1)) / sample_rate


def test_the_lag_measurement_reads_a_known_shift():
    """The measurement the tests below depend on, checked first."""
    tone = music.note(freq=400, duration=0.1, waveform_table=WAVEFORM_SINE)
    delayed_right = _interaural_lag(tone[10:], tone[:-10])
    assert delayed_right * SAMPLE_RATE == pytest.approx(-10)


def test_localize2_leaves_a_source_on_the_median_plane_alone():
    """No angle, no cues: the two channels must be the same samples.

    `theta=0` does not do this -- it is falsy, so the routine falls back
    to the x/y position -- which is why this passes x=0 instead.
    """
    tone = music.note(freq=500, duration=0.1, waveform_table=WAVEFORM_SINE)
    both = music.localize2(tone, theta=0, x=0, y=1)
    assert np.array_equal(both[0], both[1])


def test_localize2_mirrors_across_the_median_plane():
    """A source at -theta is the source at +theta with the ears swapped.

    Nothing in the geometry distinguishes the two sides, so any
    asymmetry would be an implementation artefact.
    """
    tone = music.note(freq=500, duration=0.1, waveform_table=WAVEFORM_SINE)
    left_side = music.localize2(tone, theta=40)
    right_side = music.localize2(tone, theta=-40)

    assert np.allclose(left_side[0], right_side[1])
    assert np.allclose(left_side[1], right_side[0])


@pytest.mark.parametrize("low, high", [(200, 1000), (1000, 3000)])
def test_localize2_attenuates_the_far_ear_more_at_higher_frequencies(low,
                                                                    high):
    """The IID the code applies is 1 + (f/1000) ** .8 * sin|theta|, so it
    grows with frequency: a head shadows treble more than bass."""
    def ratio(freq):
        tone = music.note(freq=freq, duration=0.1,
                          waveform_table=WAVEFORM_SINE)
        out = music.localize2(tone, theta=40)
        return np.abs(out[0]).max() / np.abs(out[1]).max()

    assert ratio(high) > ratio(low)


def test_localize2_widens_the_level_difference_with_the_angle():
    """sin|theta| again: further off centre, more shadow."""
    tone = music.note(freq=1000, duration=0.1, waveform_table=WAVEFORM_SINE)

    def ratio(theta):
        out = music.localize2(tone, theta=theta)
        return np.abs(out[0]).max() / np.abs(out[1]).max()

    assert ratio(70) > ratio(40) > ratio(10)


def _shift_at(rendered, source, freq, sample_rate=SAMPLE_RATE):
    """The time shift of `rendered` against `source` at `freq`, in seconds.

    Read from the phase of one FFT bin, which resolves a fraction of a
    sample where cross-correlation resolves whole ones. Positive is a
    delay.
    """
    count = min(len(rendered), len(source))
    bin_index = int(round(freq * count / sample_rate))
    ratio = (np.fft.fft(rendered[:count])[bin_index]
             / np.fft.fft(source[:count])[bin_index])
    return -np.angle(ratio) / (2 * np.pi * freq)


@pytest.mark.parametrize("freq", [200.0, 400.0, 1000.0])
def test_localize2_realizes_the_interaural_delay_it_computes(freq):
    """The phase change applied must be worth the ITD in the line above it.

    It was worth exactly twice: ``df`` was ``2 * sample_rate / lambda_l``
    where the spacing between FFT bins is ``sample_rate / lambda_l``, so
    every frequency the routine worked with was an octave high. The
    factor came out as 2.000 at every frequency tested, which is what a
    wrong frequency axis looks like and what a modelling choice does not.
    """
    theta = 40.0
    tone = music.note(freq=freq, duration=0.5, waveform_table=WAVEFORM_SINE)
    rendered = music.localize2(tone, theta=theta, method="ifft")

    speed = 331.3 + .606 * 20
    itd = .3 * 0.215 * np.sin(np.arcsin(np.sin(np.radians(theta)))) / speed
    measured = _shift_at(rendered[1], tone, freq)

    assert abs(measured) == pytest.approx(itd, rel=1e-6)


def test_localize2_leaves_the_near_ear_in_step_with_its_input():
    """Only the difference between the ears is applied, as in localize:
    one channel carries the whole delay and the other does not move."""
    tone = music.note(freq=400, duration=0.5, waveform_table=WAVEFORM_SINE)
    rendered = music.localize2(tone, theta=40, method="ifft")
    assert _shift_at(rendered[0], tone, 400) == pytest.approx(0, abs=1e-9)


@pytest.mark.parametrize("theta, near, far", [(40, 0, 1), (-40, 1, 0)])
def test_localize2_delays_the_far_ear_and_not_the_near_one(theta, near, far):
    """The ear that hears it louder must hear it first.

    Both branches used to advance the far ear rather than delay it, so
    the louder ear arrived after the quieter one -- on both sides, and
    for every frequency. Which ear is near is not an outside convention
    here: the routine's own x/y fallback computes theta with
    ``arctan2(-x, y)``, so positive theta is the left side, and its IID
    agrees by amplifying the left ear there.
    """
    freq = 400.0
    tone = music.note(freq=freq, duration=0.5, waveform_table=WAVEFORM_SINE)
    rendered = music.localize2(tone, theta=theta, method="ifft")

    assert np.abs(rendered[near]).max() > np.abs(rendered[far]).max()

    speed = 331.3 + .606 * 20
    itd = .3 * 0.215 * np.sin(abs(np.arcsin(
        np.sin(np.radians(theta))))) / speed
    # The near ear stays in step with the input and the far ear carries
    # the whole difference, as in localize().
    assert _shift_at(rendered[near], tone, freq) == pytest.approx(0, abs=1e-9)
    assert _shift_at(rendered[far], tone, freq) == pytest.approx(itd,
                                                                rel=1e-6)


def test_localize2_puts_the_same_source_on_the_same_side_as_localize():
    """The family must agree about which side is which.

    `localize` and `localize_linear` put the right ear at +zeta/2, so a
    source at positive x is on the right. `localize2` reaches the same
    place by a different route -- arctan2(-x, y) -- and this checks the
    two arrive together rather than trusting that they do.
    """
    tone = music.note(freq=400, duration=0.2, waveform_table=WAVEFORM_SINE)

    def louder_ear(rendered):
        return 0 if np.abs(rendered[0]).max() > np.abs(rendered[1]).max() \
            else 1

    # A source well to the right, given to each routine in its own terms.
    naive = music.localize(tone, x=1.0, y=0.01)
    experimental = music.localize2(tone, x=1.0, y=0.01)

    assert louder_ear(naive) == 1, "localize should favour the right ear"
    assert louder_ear(experimental) == 1


def _rendered(method, **kwargs):
    """localize2 with the brute method's warning suppressed."""
    tone = music.note(freq=400, duration=0.2, waveform_table=WAVEFORM_SINE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tone, music.localize2(tone, method=method, **kwargs)


def _peak_freq(signal, sample_rate=SAMPLE_RATE):
    spectrum = np.abs(np.fft.rfft(signal))
    return np.fft.rfftfreq(len(signal), 1 / sample_rate)[np.argmax(spectrum)]


@pytest.mark.parametrize("method", ["ifft", "brute"])
@pytest.mark.parametrize("freq", [220.0, 400.0, 1000.0])
def test_localize2_returns_the_frequency_it_was_given(method, freq):
    """Regression: `brute` resynthesizes each spectral component, and the
    bin it stopped at was the one carrying the energy over the cutoff --
    the loudest one. It rebuilt a 400 Hz tone from everything below
    400 Hz and returned it peaking at 298 Hz. Placing a sound must not
    change its pitch.
    """
    tone = music.note(freq=freq, duration=0.1, waveform_table=WAVEFORM_SINE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rendered = music.localize2(tone, theta=40, method=method)

    assert _peak_freq(rendered[0]) == pytest.approx(freq, rel=0.02)


#: Each case names the side the source is on, and so the near ear, in
#: both of the ways localize2 accepts a position. `theta` defaults to
#: -70 rather than 0, so x and y are only consulted when theta is passed
#: as 0 explicitly.
SIDES = [
    ("angle, left", {"theta": 40}, 0, 1),
    ("angle, right", {"theta": -40}, 1, 0),
    ("position, left", {"theta": 0, "x": -1.0, "y": 0.01}, 0, 1),
    ("position, right", {"theta": 0, "x": 1.0, "y": 0.01}, 1, 0),
]


@pytest.mark.parametrize("method", ["ifft", "brute"])
@pytest.mark.parametrize("label, kwargs, near, far", SIDES)
def test_localize2_puts_the_near_ear_first_and_loudest(method, label,
                                                       kwargs, near, far):
    """The ear nearer the source hears it louder and sooner. Both.

    Three separate faults each broke half of this. Both `ifft` branches
    advanced the far ear rather than delaying it, so the louder ear
    arrived last. `brute` chose the delayed ear from `theta` while
    choosing the amplified ear from `theta_`, and those disagree
    whenever a caller gives a position, because `theta` is 0 there.
    """
    tone, rendered = _rendered(method, **kwargs)

    assert np.abs(rendered[near]).max() > np.abs(rendered[far]).max(), (
        f"{label}: the near ear should be the louder one")

    lag = _interaural_lag(rendered[0], rendered[1])
    later = 0 if lag > 0 else 1
    assert later == far, f"{label}: the far ear should be the later one"


@pytest.mark.parametrize("method", ["ifft", "brute"])
def test_the_two_localize2_methods_agree_about_the_side(method):
    """They are offered as alternatives, so they must not disagree about
    where the sound is. They did: `brute` returned the wrong pitch and
    `ifft` the wrong ear order, so the two placed a source differently.
    """
    _, rendered = _rendered(method, theta=40)
    assert np.abs(rendered[0]).max() > np.abs(rendered[1]).max()


def test_localize2_ignores_x_and_y_unless_theta_is_zero():
    """`theta` defaults to -70, not to 0, so a caller who passes only a
    position gets the default angle and no error. Documented in the
    signature and easy to miss; pinned so it is a choice rather than a
    surprise.
    """
    tone = music.note(freq=400, duration=0.1, waveform_table=WAVEFORM_SINE)
    ignored = music.localize2(tone, x=1.0, y=0.01)
    default = music.localize2(tone)
    assert np.array_equal(ignored, default)

    honoured = music.localize2(tone, theta=0, x=1.0, y=0.01)
    assert not np.array_equal(honoured, default)
