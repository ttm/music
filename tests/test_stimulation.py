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
from music.stimulation.stimuli import _oscillator, _sample_count
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


def envelope_strength_at(signal, target, sample_rate=SR):
    """How much of the amplitude envelope sits at `target` Hertz."""
    envelope = np.abs(signal)
    spectrum = np.abs(np.fft.rfft(envelope - envelope.mean()))
    freqs = np.fft.rfftfreq(len(signal), 1 / sample_rate)
    return spectrum[np.argmin(np.abs(freqs - target))]


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


# --------------------------------------------------------------------------
# modulated noise -- a spectral stimulus until a rate is given
# --------------------------------------------------------------------------

def test_modulated_noise_puts_the_envelope_at_the_modulation_rate():
    out = music.modulated_noise(modulation_freq=7, duration=2)
    freq, _ = envelope_peak(out)
    assert freq == pytest.approx(7, abs=0.6)


def test_unmodulated_noise_is_the_bare_noise_bed():
    """Zero rate is techBroadbandNoise, not amplitude modulation.

    SSTIM types the two differently, and the difference has to be in
    the signal: with no rate there is nothing at 7 Hz but the envelope
    fluctuation any noise has. Seeded, because `noise` draws its phases
    from `np.random` and an unseeded threshold is a flaky test.
    """
    np.random.seed(1)
    flat = envelope_strength_at(
        music.modulated_noise(modulation_freq=0, duration=2), 7)
    np.random.seed(1)
    modulated = envelope_strength_at(
        music.modulated_noise(modulation_freq=7, duration=2), 7)
    assert modulated > 5 * flat


def test_noise_colour_changes_the_spectral_tilt():
    """Brown falls off faster than white, which is what colour means."""
    def tilt(signal):
        spectrum = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(len(signal), 1 / SR)
        band = (freqs >= 100) & (freqs <= 10000)
        low = spectrum[band][:len(spectrum[band]) // 10].mean()
        high = spectrum[band][-len(spectrum[band]) // 10:].mean()
        return low / high

    brown = music.modulated_noise(noise_type='brown', modulation_freq=0,
                                  duration=1)
    white = music.modulated_noise(noise_type='white', modulation_freq=0,
                                  duration=1)
    assert tilt(brown) > tilt(white)


def test_full_depth_takes_the_noise_envelope_to_silence():
    out = music.modulated_noise(modulation_freq=10, modulation_depth=1,
                                duration=1)
    assert np.abs(out).min() < 1e-9


def test_zero_depth_leaves_the_noise_bed_alone():
    """Depth zero and rate zero must render the same samples.

    The two reach the answer by different branches -- one skips the
    modulation, the other applies an envelope of ones -- and a bed that
    differed between them would mean the envelope was not neutral.
    """
    kwargs = dict(noise_type='white', duration=0.5, number_of_samples=0)
    np.random.seed(0)
    flat = music.modulated_noise(modulation_freq=0, **kwargs)
    np.random.seed(0)
    zero_depth = music.modulated_noise(modulation_freq=10,
                                       modulation_depth=0, **kwargs)
    assert np.allclose(flat, zero_depth)


def test_modulated_noise_rejects_a_depth_outside_the_unit_interval():
    with pytest.raises(ValueError, match='modulation_depth'):
        music.modulated_noise(modulation_depth=1.5)


def test_modulated_noise_rejects_a_negative_rate():
    with pytest.raises(ValueError, match='modulation_freq'):
        music.modulated_noise(modulation_freq=-1)


# --------------------------------------------------------------------------
# spatial motion -- real interaural cues, and only the ones we have
# --------------------------------------------------------------------------

def test_spatial_motion_is_stereo_and_moves_between_the_channels():
    """The energy balance must swing, and swing both ways."""
    out = music.spatial_motion(motion_rate=1, duration=2, theta1=180,
                               theta2=0)
    assert out.shape[0] == 2
    blocks = out.reshape(2, -1, 441)
    balance = (np.abs(blocks[0]).max(axis=1)
               - np.abs(blocks[1]).max(axis=1))
    assert balance.max() > 0.05
    assert balance.min() < -0.05


def test_spatial_motion_sweeps_at_the_requested_rate():
    """One round trip per cycle: the balance completes `rate` cycles."""
    rate = 2
    duration = 4
    out = music.spatial_motion(motion_rate=rate, duration=duration,
                               theta1=180, theta2=0)
    window = 64
    blocks = out[:, :out.shape[1] // window * window]
    envelope = np.abs(blocks).reshape(2, -1, window).max(axis=2)
    balance = envelope[0] - envelope[1]
    spectrum = np.abs(np.fft.rfft(balance - balance.mean()))
    freqs = np.fft.rfftfreq(len(balance), window / SR)
    assert freqs[np.argmax(spectrum)] == pytest.approx(rate, abs=0.3)


def test_a_still_source_is_the_same_as_localizing_it_once():
    """Zero rate parks the source at theta1, where localize agrees.

    The trajectory reduces to a fixed position, and a fixed position is
    what `localize_linear` renders when its endpoints coincide. Sharing
    `_localize_positions` is what makes them agree exactly rather than
    approximately.
    """
    tone = music.note(freq=200, duration=0.5)
    moving = music.spatial_motion(motion_rate=0, sonic_vector=tone,
                                  theta1=45, theta2=-45)
    still = music.localize_linear(tone, theta1=45, theta2=45, dist=0.1)
    assert np.allclose(moving, still)


def test_spatial_motion_can_move_a_sound_it_did_not_synthesize():
    bed = music.modulated_noise(modulation_freq=0, duration=0.5)
    out = music.spatial_motion(sonic_vector=bed, motion_rate=1)
    assert out.shape == (2, len(bed))


def test_spatial_motion_of_nothing_is_nothing():
    out = music.spatial_motion(sonic_vector=np.zeros(0))
    assert out.shape == (2, 0)


def test_spatial_motion_rejects_a_negative_rate():
    with pytest.raises(ValueError, match='motion_rate'):
        music.spatial_motion(motion_rate=-1)


def test_the_nearer_ear_is_never_delayed_past_the_farther_one():
    """The cue must have the sign the geometry has.

    A source to the left reaches the left ear first and louder. If the
    trajectory were rendered with the ears swapped this test is the one
    that catches it, and no test of shape or of rate would.
    """
    tone = music.note(freq=200, duration=0.3)
    left = music.spatial_motion(motion_rate=0, sonic_vector=tone,
                                theta1=180, theta2=180)
    right = music.spatial_motion(motion_rate=0, sonic_vector=tone,
                                 theta1=0, theta2=0)
    assert np.abs(left[0]).max() > np.abs(left[1]).max()
    assert np.abs(right[1]).max() > np.abs(right[0]).max()


def test_ahead_and_behind_are_the_same_sound_because_there_is_no_hrtf():
    """The gap the package documents, pinned as behaviour.

    90 and -90 degrees are ahead and behind: same distance to both
    ears, so the same interaural cues, so the same two channels. Front
    from back is an HRTF cue and this package has no HRTF. If one is
    ever added, this test fails, and that failure is the notice.
    """
    tone = music.note(freq=200, duration=0.2)
    ahead = music.spatial_motion(motion_rate=0, sonic_vector=tone,
                                 theta1=90, theta2=90)
    behind = music.spatial_motion(motion_rate=0, sonic_vector=tone,
                                  theta1=-90, theta2=-90)
    assert np.array_equal(ahead, behind)
    assert np.array_equal(ahead[0], ahead[1])
