"""Sensory-stimulation stimuli, and the distinction that defines them.

SSTIM records, per technique, whether a rendering puts a modulation
physically into the world or whether the listener constructs it. That is
not a labelling detail: it decides what a recording of the output
contains, and it is the difference between two techniques that share a
signal path. The tests below hold the package to it.
"""

import numpy as np
import pytest

import music
from music.stimulation import _oscillator, _sample_count
from music.utils import WAVEFORM_SINE, WAVEFORM_SQUARE

SR = 44100


def dominant_freq(signal, sample_rate=SR):
    """The strongest frequency present in `signal`."""
    spectrum = np.abs(np.fft.rfft(signal))
    return np.fft.rfftfreq(len(signal), 1 / sample_rate)[np.argmax(spectrum)]


def gate_of(signal, window=64):
    """Which blocks of `signal` are sounding.

    Read directly, `signal != 0` finds the carrier's own zero crossings
    rather than the gate, and counts five times too many pulses.
    """
    usable = len(signal) // window * window
    blocks = np.abs(signal[:usable]).reshape(-1, window)
    return blocks.max(axis=1) > 1e-6


def envelope_peak(signal, lo=2, hi=50, sample_rate=SR):
    """Frequency and strength of the amplitude envelope in a band."""
    envelope = np.abs(signal)
    spectrum = np.abs(np.fft.rfft(envelope - envelope.mean()))
    freqs = np.fft.rfftfreq(len(signal), 1 / sample_rate)
    band = (freqs >= lo) & (freqs <= hi)
    return freqs[band][np.argmax(spectrum[band])], spectrum[band].max()


# --------------------------------------------------------------------------
# binaural beats -- the beat is perceptual, not in the signal
# --------------------------------------------------------------------------

def test_binaural_beats_is_stereo_with_carriers_split_about_the_centre():
    out = music.binaural_beats(carrier_freq=200, beat_freq=10, duration=1)
    assert out.shape == (2, SR)
    assert dominant_freq(out[0]) == pytest.approx(195, abs=1)
    assert dominant_freq(out[1]) == pytest.approx(205, abs=1)


def test_binaural_beats_puts_no_beat_in_either_channel():
    """The defining property: each ear receives a steady tone. A
    spectrum of one channel does not contain the difference frequency,
    because the beat is constructed by the listener.

    Measured against a real beat rather than against a fixed number, so
    the test says "orders of magnitude below a beat that is there"
    rather than depending on the amplitude and duration in use.
    """
    binaural = music.binaural_beats(carrier_freq=200, beat_freq=10,
                                    duration=2)
    monaural = music.monaural_beats(carrier_freq=200, beat_freq=10,
                                    duration=2)
    _, present = envelope_peak(monaural)
    for channel in binaural:
        _, in_channel = envelope_peak(channel)
        assert in_channel < present / 1000


def test_summing_a_binaural_beat_turns_it_into_a_monaural_one():
    """Channel separation is the mechanism, so a downmix does not
    preserve the stimulus -- it silently substitutes a different
    technique, whose beat is physically present."""
    binaural = music.binaural_beats(carrier_freq=200, beat_freq=10,
                                    duration=2)
    monaural = music.monaural_beats(carrier_freq=200, beat_freq=10,
                                    duration=2)
    assert np.allclose(binaural.mean(axis=0), monaural)

    freq, summed = envelope_peak(binaural.mean(axis=0))
    _, in_channel = envelope_peak(binaural[0])
    assert freq == pytest.approx(10, abs=0.5)
    assert summed > 1000 * in_channel


# --------------------------------------------------------------------------
# monaural beats -- the beat is physically present
# --------------------------------------------------------------------------

def test_monaural_beats_is_mono_and_carries_a_real_envelope():
    out = music.monaural_beats(carrier_freq=200, beat_freq=10, duration=2)
    assert out.ndim == 1
    freq, strength = envelope_peak(out)
    assert freq == pytest.approx(10, abs=0.5)

    # A steady tone of the same amplitude, as the floor to beat.
    steady = music.note(freq=200, duration=2, waveform_table=WAVEFORM_SINE)
    _, floor = envelope_peak(steady)
    assert strength > 1000 * floor


# --------------------------------------------------------------------------
# isochronic tones
# --------------------------------------------------------------------------

@pytest.mark.parametrize("duty_cycle", [0.25, 0.5, 0.75])
def test_isochronic_tones_gates_for_the_requested_fraction(duty_cycle):
    out = music.isochronic_tones(carrier_freq=1000, pulse_rate=5,
                                 duty_cycle=duty_cycle, duration=4)
    assert gate_of(out).mean() == pytest.approx(duty_cycle, abs=0.03)


def test_isochronic_tones_pulses_at_the_requested_rate():
    out = music.isochronic_tones(pulse_rate=10, duty_cycle=0.5, duration=2)
    onsets = np.count_nonzero(np.diff(gate_of(out).astype(int)) > 0)
    assert onsets == pytest.approx(20, abs=1)


def test_isochronic_tones_rejects_a_duty_cycle_outside_the_unit_interval():
    """Zero is silence and above one is not a gate; both are far more
    likely to be a mistake than an intention."""
    for bad in (0, -0.1, 1.5):
        with pytest.raises(ValueError, match="duty_cycle"):
            music.isochronic_tones(duty_cycle=bad)


def test_isochronic_ramp_softens_the_step_that_makes_a_click():
    """An abrupt gate is a step discontinuity twice per pulse, which
    spreads energy across the spectrum.

    `carrier_freq / pulse_rate` must not be a whole number for this to
    be visible at all: when it is, every gate edge lands on the same
    carrier phase -- zero, for a sine -- and the hard gate happens to
    produce no discontinuity. 200/7 is deliberately not 200/10.
    """
    hard = music.isochronic_tones(carrier_freq=200, pulse_rate=7,
                                  duration=1)
    ramped = music.isochronic_tones(carrier_freq=200, pulse_rate=7,
                                    duration=1, ramp_duration=0.005)

    # The step is very nearly full scale without a ramp, and no larger
    # than the carrier's own slope with one.
    assert np.abs(np.diff(hard)).max() > 0.5
    assert np.abs(np.diff(ramped)).max() < 0.05

    def energy_above(signal, cutoff=2000):
        spectrum = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(len(signal), 1 / SR)
        return spectrum[freqs > cutoff].sum()

    assert energy_above(ramped) < energy_above(hard) / 10


def test_isochronic_tones_rejects_a_negative_ramp():
    """It would otherwise scale every pulse by a clipped negative and
    hand back silence, with nothing said."""
    with pytest.raises(ValueError, match="ramp_duration"):
        music.isochronic_tones(ramp_duration=-0.005)


def test_a_ramp_longer_than_the_pulse_gives_a_triangle_not_an_error():
    """The two ramps overlap and the pulse never reaches full
    amplitude, which is a sensible reading of the request rather than a
    mistake to reject."""
    out = music.isochronic_tones(pulse_rate=10, duration=0.5,
                                 ramp_duration=0.05)
    assert 0 < np.abs(out).max() < 0.6


def test_a_ramp_shorter_than_a_sample_leaves_the_gate_hard():
    """`int()` of a sub-sample ramp is zero, and dividing by it would
    raise rather than doing nothing."""
    out = music.isochronic_tones(pulse_rate=10, duration=0.5,
                                 ramp_duration=1e-9)
    hard = music.isochronic_tones(pulse_rate=10, duration=0.5)
    assert np.allclose(out, hard)


# --------------------------------------------------------------------------
# amplitude modulation
# --------------------------------------------------------------------------

def test_amplitude_modulation_puts_the_envelope_at_the_modulation_rate():
    """The modulation rate, not the carrier, sets the response
    frequency -- which is the point of the stimulus."""
    out = music.amplitude_modulation(carrier_freq=200, modulation_freq=40,
                                     duration=1)
    freq, _ = envelope_peak(out, lo=2, hi=100)
    assert freq == pytest.approx(40, abs=1)


def test_zero_depth_leaves_the_carrier_untouched():
    out = music.amplitude_modulation(carrier_freq=200, modulation_depth=0,
                                     duration=0.5)
    plain = music.note(freq=200, duration=0.5,
                       waveform_table=WAVEFORM_SINE)
    assert np.allclose(out, plain)


def test_full_depth_takes_the_envelope_to_silence():
    out = music.amplitude_modulation(carrier_freq=200, modulation_freq=10,
                                     modulation_depth=1, duration=1)
    assert np.abs(out).min() == pytest.approx(0, abs=1e-3)


def test_amplitude_modulation_rejects_a_depth_outside_the_unit_interval():
    """Past 1 the envelope goes negative, which inverts the carrier's
    phase rather than deepening the modulation."""
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match="modulation_depth"):
            music.amplitude_modulation(modulation_depth=bad)


# --------------------------------------------------------------------------
# frequency modulation
# --------------------------------------------------------------------------

def test_zero_deviation_is_a_plain_tone():
    """With no deviation the routine reduces to a steady carrier -- but
    not to a bit-identical copy of `note`. Integrating the
    instantaneous frequency with `cumsum` accumulates rounding that a
    single multiplication does not, and it leads by one sample. The
    difference is under half a thousandth of full scale.
    """
    out = music.frequency_modulation(carrier_freq=200,
                                     frequency_deviation=0, duration=1)
    assert dominant_freq(out) == pytest.approx(200, abs=1)

    plain = music.note(freq=200, duration=1, waveform_table=WAVEFORM_SINE)
    assert np.abs(out[:-1] - plain[1:]).max() < 0.001


def test_frequency_modulation_sweeps_the_carrier_by_the_deviation():
    """The instantaneous frequency should reach `carrier +/- deviation`,
    measured where the modulator is at its extremes rather than from
    the spectrum as a whole, which the carrier dominates."""
    out = music.frequency_modulation(carrier_freq=1000, modulation_freq=1,
                                     frequency_deviation=200, duration=2)
    window = 4096
    heard = [dominant_freq(out[i:i + window])
             for i in range(0, len(out) - window, window)]
    assert min(heard) == pytest.approx(800, abs=60)
    assert max(heard) == pytest.approx(1200, abs=60)


# --------------------------------------------------------------------------
# shared conventions
# --------------------------------------------------------------------------

def test_number_of_samples_is_taken_instead_of_duration():
    assert _sample_count(2, 0, 100) == 200
    assert _sample_count(2, 55, 100) == 55


@pytest.mark.parametrize("routine", [
    music.binaural_beats, music.monaural_beats, music.isochronic_tones,
    music.amplitude_modulation, music.frequency_modulation,
])
def test_every_routine_honours_number_of_samples(routine):
    assert routine(number_of_samples=1000).shape[-1] == 1000


def test_the_modulator_can_be_driven_from_any_table():
    """A modulation need not be sinusoidal; the table is a parameter,
    as it already is for the package's own FM."""
    square = WAVEFORM_SQUARE
    sine = music.amplitude_modulation(modulation_freq=10, duration=0.5)
    other = music.amplitude_modulation(
        modulation_freq=10, duration=0.5,
        modulation_waveform_table=square)
    assert not np.allclose(sine, other)
    assert len(_oscillator(square, 10, 100, SR)) == 100
