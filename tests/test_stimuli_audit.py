"""Exact samples from the stimulus generators.

The first mutation audit of :mod:`music.stimulation.stimuli` found the
envelopes, gates and trajectories reached by tests that only measured
their spectra or their shapes: every arithmetic edit to the amplitude
envelope survived, and so did most edits to the orbit. These tests read
the samples themselves. A ramp table renders a carrier's phase, and a
constant one renders an envelope or a gate on its own.
"""

import numpy as np
import pytest

import music
from music.core.filters.localization import _localize_positions

#: A lookup ramp reads the phase of a carrier, to within one cell.
RAMP = np.arange(4096, dtype=float) / 4096
#: A constant table: the carrier is one, so the output is the envelope.
ONE = np.ones(4)
#: A four-entry modulator that steps one entry per sample at 250 Hz and
#: 1 kHz, so the modulating values are exactly 1, 0, -1 and 0.
STEPS = np.array([1.0, 0.0, -1.0, 0.0])


def _phase(freq, count, sample_rate):
    """What the ramp table renders for a steady frequency."""
    return (np.arange(count) * freq / sample_rate) % 1


# --------------------------------------------------------------------------
# carriers, at a rate other than 44.1 kHz
# --------------------------------------------------------------------------

def test_binaural_carriers_sit_either_side_of_the_centre_at_any_rate():
    """No test ran a stimulus at another rate, so every call that passes
    the rate on to `note` could have dropped it."""
    left, right = music.binaural_beats(carrier_freq=300, beat_freq=40,
                                       waveform_table=RAMP,
                                       number_of_samples=64,
                                       sample_rate=8000)
    np.testing.assert_allclose(left, _phase(280, 64, 8000), atol=1 / 4096)
    np.testing.assert_allclose(right, _phase(320, 64, 8000), atol=1 / 4096)


def test_a_monaural_beat_is_the_mean_of_its_two_carriers_at_any_rate():
    mixed = music.monaural_beats(carrier_freq=300, beat_freq=40,
                                 waveform_table=RAMP, number_of_samples=64,
                                 sample_rate=8000)
    expected = (_phase(280, 64, 8000) + _phase(320, 64, 8000)) / 2
    np.testing.assert_allclose(mixed, expected, atol=1 / 4096)


@pytest.mark.parametrize("routine, settings", [
    ("isochronic_tones", dict(pulse_rate=10, duty_cycle=1)),
    ("amplitude_modulation", dict(modulation_depth=0)),
])
def test_a_gated_or_modulated_carrier_keeps_its_frequency_at_any_rate(
        routine, settings):
    rendered = getattr(music, routine)(carrier_freq=300, waveform_table=RAMP,
                                       number_of_samples=64,
                                       sample_rate=8000, **settings)
    np.testing.assert_allclose(rendered, _phase(300, 64, 8000),
                               atol=1 / 4096)


def test_an_orbiting_tone_is_the_carrier_it_was_asked_for():
    """The synthesized source is `note` at the frequency and rate given,
    so it moves exactly as that note would."""
    settings = dict(motion_rate=1, theta1=150, theta2=30, dist=0.5,
                    sample_rate=8000)
    synthesized = music.spatial_motion(carrier_freq=300,
                                       number_of_samples=800, **settings)
    given = music.spatial_motion(
        sonic_vector=music.note(300, number_of_samples=800,
                                waveform_table=music.WAVEFORM_SINE,
                                sample_rate=8000), **settings)
    np.testing.assert_array_equal(synthesized, given)


# --------------------------------------------------------------------------
# lengths at the edges
# --------------------------------------------------------------------------

@pytest.mark.parametrize("routine, shape", [
    ("binaural_beats", (2, 1)), ("monaural_beats", (1,)),
    ("isochronic_tones", (1,)), ("amplitude_modulation", (1,)),
    ("frequency_modulation", (1,)), ("modulated_noise", (1,)),
    ("spatial_motion", (2, 1)),
])
def test_a_one_sample_stimulus_is_one_sample(routine, shape):
    """Zero samples is refused as nothing; one is not zero."""
    assert getattr(music, routine)(number_of_samples=1).shape == shape


@pytest.mark.parametrize("routine, shape", [
    ("binaural_beats", (2, 0)), ("spatial_motion", (2, 0)),
])
def test_nothing_in_stereo_is_two_empty_channels(routine, shape):
    assert getattr(music, routine)(duration=0).shape == shape


@pytest.mark.parametrize("routine", ["modulated_noise", "spatial_motion"])
def test_the_noise_and_the_orbit_honour_number_of_samples(routine):
    """The other five were already checked; these two were not."""
    assert getattr(music, routine)(number_of_samples=321).shape[-1] == 321


# --------------------------------------------------------------------------
# envelopes and gates, sample by sample
# --------------------------------------------------------------------------

@pytest.mark.parametrize("depth", [0, .5, 1])
def test_the_amplitude_envelope_is_one_minus_depth_times_its_trough(depth):
    """``1 - depth * (1 - m) / 2``: one at the modulator's crest, and
    ``1 - depth`` at its trough. Every arithmetic edit to that line
    survived, because the tests measured only where the envelope's
    energy sat in the spectrum."""
    rendered = music.amplitude_modulation(
        modulation_freq=250, modulation_depth=depth, waveform_table=ONE,
        modulation_waveform_table=STEPS, number_of_samples=8,
        sample_rate=1000)
    crest_to_trough = np.array([1, 1 - depth / 2, 1 - depth, 1 - depth / 2])
    np.testing.assert_allclose(rendered, np.tile(crest_to_trough, 2))


@pytest.mark.parametrize("depth", [.5, 1])
def test_the_noise_envelope_is_the_same_envelope_on_the_same_bed(depth):
    """`noise` draws from `np.random`, so seeding it twice gives the bed
    that `modulated_noise` shaped, and their ratio is the envelope."""
    band = dict(noise_type="white", min_freq=100, max_freq=300,
                number_of_samples=8, sample_rate=1000)
    np.random.seed(7)
    bed = music.noise(**band)
    np.random.seed(7)
    rendered = music.modulated_noise(
        modulation_freq=250, modulation_depth=depth,
        modulation_waveform_table=STEPS, **band)
    crest_to_trough = np.array([1, 1 - depth / 2, 1 - depth, 1 - depth / 2])
    np.testing.assert_allclose(rendered, bed * np.tile(crest_to_trough, 2))


def test_unmodulated_noise_is_the_band_it_was_asked_for_at_its_rate():
    """Its colour, band and rate reach `noise`. Dropping `min_freq`,
    `max_freq` or `sample_rate` on the way survived."""
    band = dict(noise_type="pink", min_freq=100, max_freq=300,
                number_of_samples=4000, sample_rate=8000)
    np.random.seed(11)
    bed = music.noise(**band)
    np.random.seed(11)
    rendered = music.modulated_noise(modulation_freq=0, **band)
    np.testing.assert_array_equal(rendered, bed)


def test_frequency_modulation_raises_the_pitch_while_the_modulator_is_up():
    """A square modulator holds each sign for half a period, so the pitch
    is 120 Hz and then 80, never the other way round: a sign error in the
    deviation sweeps the same range and passed every spectral test."""
    rendered = music.frequency_modulation(
        carrier_freq=100, modulation_freq=10, frequency_deviation=20,
        waveform_table=RAMP, modulation_waveform_table=np.array([1., -1.]),
        number_of_samples=100, sample_rate=1000)
    pitch = np.repeat([120., 80.], 50)
    expected = np.cumsum(pitch) / 1000 % 1
    np.testing.assert_allclose(rendered, expected, atol=1 / 4096)


def _gate(**settings):
    """An isochronic train on a carrier of one: the gate itself."""
    return music.isochronic_tones(pulse_rate=10, waveform_table=ONE,
                                  number_of_samples=200, sample_rate=1000,
                                  **settings)


def test_the_gate_opens_for_exactly_the_duty_cycle():
    """At 10 Hz and 1 kHz a period is 100 samples and the edge falls on a
    sample, which a comparison that included it would have kept on."""
    period = np.r_[np.ones(50), np.zeros(50)]
    np.testing.assert_array_equal(_gate(duty_cycle=.5), np.tile(period, 2))


def test_a_duty_cycle_of_one_never_closes_the_gate():
    np.testing.assert_array_equal(_gate(duty_cycle=1), np.ones(200))


def test_a_ramp_rises_and_falls_linearly_over_its_own_length():
    """Ten samples each way, measured from the nearer edge of the pulse."""
    k = np.arange(50)
    period = np.r_[np.clip(np.minimum(k, 50 - k) / 10, 0, 1), np.zeros(50)]
    np.testing.assert_allclose(_gate(duty_cycle=.5, ramp_duration=.01),
                               np.tile(period, 2), atol=1e-12)


@pytest.mark.parametrize("pulse_rate", [0, -10])
def test_an_isochronic_train_needs_a_positive_rate(pulse_rate):
    """At zero a ramp divided by the period and raised ZeroDivisionError;
    a negative rate ran the gate backwards, silent for the first half of
    each period instead of the second."""
    with pytest.raises(ValueError, match="pulse_rate"):
        music.isochronic_tones(pulse_rate=pulse_rate, ramp_duration=.01)


@pytest.mark.parametrize("table", [STEPS, -STEPS, np.array([.5, -1.])])
def test_amplitude_modulation_at_no_rate_leaves_the_carrier_alone(table):
    """As `modulated_noise` documents for its own zero rate. The held
    modulator scaled the carrier by whatever the table began with: by
    half for a sine at full depth, not at all for a table starting at
    one."""
    rendered = music.amplitude_modulation(
        modulation_freq=0, modulation_depth=1, waveform_table=ONE,
        modulation_waveform_table=table, number_of_samples=8)
    np.testing.assert_array_equal(rendered, np.ones(8))


@pytest.mark.parametrize("table", [STEPS, -STEPS, np.array([.5, -1.])])
def test_frequency_modulation_at_no_rate_holds_the_carrier(table):
    """The held modulator shifted the pitch by the table's first entry."""
    rendered = music.frequency_modulation(
        carrier_freq=103, modulation_freq=0, frequency_deviation=20,
        waveform_table=RAMP, modulation_waveform_table=table,
        number_of_samples=100, sample_rate=1000)
    # 103 Hz, so that no sample lands exactly on the wrap of the ramp.
    expected = np.cumsum(np.full(100, 103.)) / 1000 % 1
    np.testing.assert_allclose(rendered, expected, atol=1 / 4096)


@pytest.mark.parametrize("routine, rate", [
    ("amplitude_modulation", "modulation_freq"),
    ("frequency_modulation", "modulation_freq"),
    ("isochronic_tones", "pulse_rate"),
])
def test_a_rate_below_one_hertz_is_a_rate(routine, rate):
    """Breathing-rate modulation is the point of a slow FM, and the
    example script uses a quarter of a hertz."""
    rendered = getattr(music, routine)(number_of_samples=100,
                                       **{rate: .25})
    assert rendered.shape == (100,) and np.abs(rendered).max() > 0


@pytest.mark.parametrize("routine", ["amplitude_modulation",
                                     "frequency_modulation"])
def test_a_modulation_refuses_a_negative_rate(routine):
    """As `modulated_noise` already did, and for the reason it gives."""
    with pytest.raises(ValueError, match="modulation_freq"):
        getattr(music, routine)(modulation_freq=-10)


# --------------------------------------------------------------------------
# the orbit
# --------------------------------------------------------------------------

@pytest.mark.parametrize("theta1, theta2", [(150, 30), (0, 180)])
def test_an_orbit_is_triangular_in_azimuth_through_every_cycle(theta1,
                                                               theta2):
    """Two and a half round trips, so the fold at each turn and the
    wrap into the next cycle are both in the render. Nothing measured
    the trajectory: edits to the phase, the fold and the azimuth all
    survived."""
    rate, sample_rate, count, dist = 2.5, 1000, 1000, 0.5
    source = music.note(200, number_of_samples=count, sample_rate=1000)
    rendered = music.spatial_motion(
        sonic_vector=source, motion_rate=rate, theta1=theta1, theta2=theta2,
        dist=dist, sample_rate=sample_rate)
    phase = (np.arange(count) * rate / sample_rate) % 1
    progress = np.where(phase < .5, 2 * phase, 2 - 2 * phase)
    azimuth = np.radians(theta1 + (theta2 - theta1) * progress)
    expected = _localize_positions(source, dist * np.cos(azimuth),
                                   dist * np.sin(azimuth), 0.215, 20,
                                   sample_rate)
    np.testing.assert_allclose(rendered, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("sound", [
    np.random.default_rng(3).standard_normal(500).astype(np.float32),
    np.arange(500) % 7 > 3,
])
def test_an_orbit_moves_any_sound_as_its_float64_values(sound):
    """The fractional delay interpolates between neighbouring samples,
    which in float32 rounds differently and on booleans cannot subtract.
    Nothing moved a sound of either type, so the conversion could go."""
    settings = dict(motion_rate=3, theta1=160, theta2=20)
    np.testing.assert_array_equal(
        music.spatial_motion(sonic_vector=sound, **settings),
        music.spatial_motion(sonic_vector=sound.astype(np.float64),
                             **settings))


def test_an_orbit_moves_a_single_sample():
    rendered = music.spatial_motion(sonic_vector=[.5], theta1=0, theta2=0)
    assert rendered.shape == (2, 1) and np.abs(rendered).max() > 0


def test_an_orbit_refuses_a_sound_that_is_already_stereo():
    with pytest.raises(ValueError, match="mono"):
        music.spatial_motion(sonic_vector=np.zeros((2, 100)))


# --------------------------------------------------------------------------
# defaults
# --------------------------------------------------------------------------

@pytest.mark.parametrize("routine, declared", [
    ("binaural_beats", dict(carrier_freq=200, beat_freq=10, duration=2)),
    ("monaural_beats", dict(carrier_freq=200, beat_freq=10, duration=2)),
    ("isochronic_tones", dict(carrier_freq=200, pulse_rate=10,
                              duty_cycle=.5, duration=2, ramp_duration=0)),
    ("amplitude_modulation", dict(carrier_freq=200, modulation_freq=10,
                                  modulation_depth=1, duration=2)),
    ("frequency_modulation", dict(carrier_freq=200, modulation_freq=10,
                                  frequency_deviation=20, duration=2)),
    ("modulated_noise", dict(noise_type="pink", modulation_freq=10,
                             modulation_depth=1, duration=2, min_freq=15,
                             max_freq=15000)),
    ("spatial_motion", dict(carrier_freq=200, motion_rate=.2, duration=2,
                            theta1=180, theta2=0, dist=.1, zeta=.215,
                            air_temp=20)),
])
def test_a_bare_call_renders_the_defaults_it_declares(routine, declared):
    """Every default could change without a test noticing. Seeded, for
    the noise."""
    np.random.seed(0)
    bare = getattr(music, routine)()
    np.random.seed(0)
    explicit = getattr(music, routine)(sample_rate=44100, **declared)
    np.testing.assert_array_equal(bare, explicit)
    assert bare.shape[-1] == 2 * 44100
