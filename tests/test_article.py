"""Routines checked against the article's numbered equations.

`RECONCILIATION.md` establishes that the package agrees with the MASS
*reference implementation*. It says, in its own last section, that this is
not the same as agreeing with the *article*: where the two implementations
differ, the register argues which is right, and an argument in a comment is
not a proof.

This is the other leg. Each test below names the equation it checks by the
label the article's LaTeX source gives it, so that a reader can find it:
`eq:branco` is `\\label{eq:branco}` in `doc/body.tex` of ttm/mass, and
`article.pdf` beside it is the same document typeset.

Only what the article actually specifies can be checked here.
`tests/test_article_coverage.py` measures how much of it that is.

References
----------
.. [1] Fabbri, Renato, et al. "Musical elements in the discrete-time
       representation of sound." arXiv preprint arXiv:abs/1412.6853 (2017)
"""

import numpy as np
import pytest

import music
from music.core.filters.localization import localize2

SAMPLE_RATE = 44100


# --------------------------------------------------------------------------
# eq:branco -- the frequency a DFT coefficient stands for
# --------------------------------------------------------------------------

def test_a_coefficient_stands_for_the_frequency_equation_branco_gives_it():
    """f_i = i * f_s / Lambda, and noise's band edges land where it says.

    The article states this while defining white noise: the band is built
    by zeroing the coefficients below f_min, so where that edge falls is
    a direct reading of the spacing the routine used.
    """
    duration, min_freq = 0.5, 400.0
    length = int(duration * SAMPLE_RATE)
    spacing = SAMPLE_RATE / length            # eq:branco

    np.random.seed(0)
    samples = music.noise(noise_type="white", duration=duration,
                          min_freq=min_freq, max_freq=8000.0,
                          sample_rate=SAMPLE_RATE)
    spectrum = np.abs(np.fft.fft(samples))
    edge = int(np.floor(min_freq / spacing))

    # Silent below the edge the equation puts it at, and not silent above.
    assert spectrum[1:edge].max() < 1e-9 * spectrum.max()
    assert spectrum[edge:edge + 20].max() > 1e-6 * spectrum.max()
    # And the edge really is where min_freq falls, not at half or twice it.
    assert edge * spacing <= min_freq < (edge + 1) * spacing


def test_localize2_reads_its_coefficients_at_the_spacing_branco_states():
    """The crossover between its two delays falls at the frequency it names.

    `localize2` switches delay coefficient at 4000 Hz. It read the spacing
    as 2*f_s/Lambda, so every coefficient stood for twice its true
    frequency and the switch fired at a real 2000 Hz. This pins the fix to
    the equation rather than to the comment that argues for it: a tone
    either side of 2000 Hz must now be treated alike, and one either side
    of 4000 Hz must not.
    """
    def delay_between_ears(freq):
        tone = music.note(freq=freq, duration=0.1,
                          waveform_table=music.WAVEFORM_SINE,
                          sample_rate=SAMPLE_RATE)
        left, right = localize2(sonic_vector=tone, theta=-70, method="ifft",
                                sample_rate=SAMPLE_RATE)
        correlation = np.correlate(left, right, mode="full")
        return int(np.argmax(correlation)) - (len(left) - 1)

    below_old, above_old = delay_between_ears(1900), delay_between_ears(2100)
    below_new, above_new = delay_between_ears(3900), delay_between_ears(4100)

    assert below_old == above_old, (
        "the delay changed across 2000 Hz, which is where the crossover "
        "lands when the coefficient spacing is read as twice eq:branco's")
    assert below_new != above_new, (
        "the delay did not change across 4000 Hz, where this routine says "
        "its crossover is")


# --------------------------------------------------------------------------
# eq:rosa, eq:marrom, eq:azul, eq:violeta, eq:preto -- the noise colours
# --------------------------------------------------------------------------

@pytest.mark.parametrize("colour, decibels_per_octave, label", [
    ("pink", -3, "eq:rosa"),
    ("brown", -6, "eq:marrom"),
    ("blue", 3, "eq:azul"),
    ("violet", 6, "eq:violeta"),
    ("black", -12, "eq:preto"),
])
def test_noise_scales_each_coefficient_by_the_articles_alpha(
        colour, decibels_per_octave, label):
    """alpha_i = (10 ** (beta/20)) ** log2(f_i / f_min), coefficient by
    coefficient.

    `tests/test_fidelity.py` fits a slope across the band, which a
    different law with the same average tilt would also pass. This checks
    the factor the article writes, at every coefficient in the band.
    """
    duration, min_freq, max_freq = 0.5, 15.0, 15000.0
    length = int(duration * SAMPLE_RATE)
    spacing = SAMPLE_RATE / length

    seed = 12345
    np.random.seed(seed)
    coloured = music.noise(noise_type=colour, duration=duration,
                           min_freq=min_freq, max_freq=max_freq,
                           sample_rate=SAMPLE_RATE)
    np.random.seed(seed)
    white = music.noise(noise_type="white", duration=duration,
                        min_freq=min_freq, max_freq=max_freq,
                        sample_rate=SAMPLE_RATE)

    coloured_spectrum = np.abs(np.fft.fft(coloured))
    white_spectrum = np.abs(np.fft.fft(white))

    first = max(1, int(np.floor(min_freq / spacing)))
    last = int(np.floor(max_freq / spacing))
    index = np.arange(first, last)
    freqs = np.clip(index * spacing, spacing, None)
    denom = max(min_freq, spacing)

    expected = (10. ** (decibels_per_octave / 20.)) ** np.log2(freqs / denom)
    produced = coloured_spectrum[first:last] / white_spectrum[first:last]

    # `noise` normalizes what it returns, and the two renders are each
    # scaled by their own peak, so the ratio carries one constant factor
    # that the equation says nothing about. Divide it out at one bin and
    # every other bin must then agree exactly.
    assert np.allclose(produced / produced[0], expected / expected[0],
                       rtol=1e-9, atol=1e-12), (
        f"{colour} noise does not scale its coefficients by the alpha "
        f"{label} states")


def test_black_noise_is_steeper_than_the_bound_equation_preto_puts_on_it():
    """eq:preto says only that beta > 6; the package chose 12."""
    duration = 0.3
    np.random.seed(0)
    black = music.noise(noise_type="black", duration=duration,
                        sample_rate=SAMPLE_RATE)
    np.random.seed(0)
    brown = music.noise(noise_type="brown", duration=duration,
                        sample_rate=SAMPLE_RATE)

    def slope(samples):
        spectrum = np.abs(np.fft.rfft(samples))
        freqs = np.fft.rfftfreq(len(samples), 1 / SAMPLE_RATE)
        band = (freqs >= 100) & (freqs <= 8000)
        fit, _ = np.polyfit(np.log2(freqs[band]),
                            20 * np.log10(spectrum[band] + 1e-30), 1)
        return fit

    assert slope(black) < slope(brown) < -6 + 1e-6
    assert slope(black) == pytest.approx(-12, abs=0.5)


# --------------------------------------------------------------------------
# eq:potencia and eq:dobraVol -- power, and the amplitude a decibel buys
# --------------------------------------------------------------------------

def test_power_is_the_mean_square_equation_potencia_defines():
    """pow(T) = sum(t_i ** 2) / Lambda."""
    tone = music.note(freq=440, duration=0.1, sample_rate=SAMPLE_RATE)
    power = float(np.sum(tone ** 2) / len(tone))
    assert music.amp_to_db(np.sqrt(power)) == pytest.approx(
        10 * np.log10(power))


def test_ten_decibels_is_exactly_the_square_root_of_ten_in_amplitude():
    """eq:dobraVol derives this one exactly: t\' = sqrt(10) . t."""
    assert music.db_to_amp(10) == pytest.approx(np.sqrt(10))
    assert music.amp_to_db(np.sqrt(10)) == pytest.approx(10)


@pytest.mark.parametrize("gain, approximate_decibels", [
    (np.sqrt(2), 3),        # eq:potVol: doubling the power is *about* 3 dB
    (2.0, 6),
    (0.5, -6),
])
def test_the_articles_round_decibel_figures_are_the_approximations_it_says(
        gain, approximate_decibels):
    """eq:ampVol and eq:potVol give 3 dB with an approximately sign.

    10*log10(2) is 3.0103, not 3, so a gain of exactly 3 dB is not exactly
    sqrt(2). The article writes these as approximations and the conversions
    are exact, which is the right way round: this pins the gap rather than
    letting either side drift into the other\'s rounding.
    """
    exact = music.amp_to_db(gain)
    assert exact == pytest.approx(approximate_decibels, abs=0.021)
    assert exact != approximate_decibels
    assert music.db_to_amp(exact) == pytest.approx(gain)


def test_ten_decibels_multiplies_the_power_by_ten():
    """The statement eq:dobraVol makes, on a rendered sound."""
    tone = music.note(freq=440, duration=0.1, sample_rate=SAMPLE_RATE)
    louder = tone * music.db_to_amp(10)
    assert np.mean(louder ** 2) == pytest.approx(10 * np.mean(tone ** 2))


# --------------------------------------------------------------------------
# eq:fDoppler -- a moving source
# --------------------------------------------------------------------------

def test_the_doppler_shift_is_the_ratio_equation_fdoppler_gives():
    """f = ((s_sound + s_r) / (s_sound + s_s)) * f_0, with the receiver still.

    The listener does not move, so s_r is zero and the shift is
    s_sound / (s_sound + s_s): a source approaching at s_s < 0 raises the
    pitch, one receding lowers it, and the two are not symmetric.
    """
    air_temp = 20
    speed_of_sound = 331.3 + .606 * air_temp        # the article's v_sound

    duration, freq = 0.4, 1000.0
    stereo = music.note_with_doppler(
        freq=freq, duration=duration, x=(-10, 10), y=(0.01, 0.01),
        stereo=True, air_temp=air_temp, sample_rate=SAMPLE_RATE)

    # The source crosses from far left to far right at a constant rate, so
    # its speed along the line to the listener flips sign at the crossing.
    left = np.asarray(stereo)[0]
    first, last = left[:len(left) // 4], left[-len(left) // 4:]

    def dominant(samples):
        spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
        return float(np.fft.rfftfreq(
            len(samples), 1 / SAMPLE_RATE)[np.argmax(spectrum)])

    approaching, receding = dominant(first), dominant(last)
    assert approaching > freq > receding

    # The magnitudes follow the equation, given the source's own speed.
    source_speed = 20 / duration                      # metres per second
    predicted_high = speed_of_sound / (speed_of_sound - source_speed) * freq
    predicted_low = speed_of_sound / (speed_of_sound + source_speed) * freq
    assert approaching == pytest.approx(predicted_high, rel=0.05)
    assert receding == pytest.approx(predicted_low, rel=0.05)


# --------------------------------------------------------------------------
# eq:micro -- tuning, from the companion paper on notes in music
# --------------------------------------------------------------------------

@pytest.mark.parametrize("semitones", [0, 1, 4, 7, 12, -12, 19])
def test_a_pitch_is_the_power_of_two_equation_micro_gives(semitones):
    """f_i = f * 2 ** (s_i / eta), with eta = 12 steps to the octave.

    `notesInMusic.tex` states this while introducing microtonal tunings,
    where eta is the number of steps the octave is divided into. The
    package divides it into twelve everywhere, which is that equation with
    eta fixed.
    """
    eta, reference = 12, 220.0
    expected = reference * 2 ** (semitones / eta)

    assert music.pitch_to_freq(start_freq=reference,
                               semitones=(semitones,)) == \
        pytest.approx([expected])
    assert music.midi_to_hz_interval(semitones) == pytest.approx(
        2 ** (semitones / eta))


def test_midi_and_hertz_are_each_other_s_inverse_under_that_equation():
    """69 is a-440 in MIDI, and eq:micro fixes everything either side."""
    assert music.midi_to_hz(69) == pytest.approx(440.0)
    assert music.hz_to_midi(440.0) == pytest.approx(69.0)
    for midi in (21, 60, 69, 108):
        assert music.hz_to_midi(music.midi_to_hz(midi)) == pytest.approx(midi)
        # And the step to the next semitone is the twelfth root of two.
        assert music.midi_to_hz(midi + 1) / music.midi_to_hz(midi) == \
            pytest.approx(2 ** (1 / 12))


def test_a_change_of_tuning_rescales_the_step_count_as_equation_micro_says():
    """s' = s * eta' / eta: the same pitch, counted in a finer division.

    The package has no eta but 12, so this checks the equation holds of
    the frequencies rather than of an API it does not have -- 7 steps of
    12 to the octave is 14 steps of 24, and both must land on one pitch.
    """
    reference = 220.0
    for steps, eta, finer_eta in ((7, 12, 24), (4, 12, 36), (1, 12, 48)):
        coarse = reference * 2 ** (steps / eta)
        finer_steps = steps * finer_eta / eta
        assert reference * 2 ** (finer_steps / finer_eta) == \
            pytest.approx(coarse)
        assert music.pitch_to_freq(start_freq=reference,
                                   semitones=(steps,)) == \
            pytest.approx([coarse])
