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

import math

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


# --------------------------------------------------------------------------
# eq:dur, eq:notaBasica, eq:lut -- the note, and how long it is
# --------------------------------------------------------------------------

@pytest.mark.parametrize("duration", [0.1, 0.25, 1.0, 0.333])
def test_a_sound_lasts_the_samples_equation_dur_counts(duration):
    """Lambda = floor(Delta * f_s), the floor and not a rounding."""
    expected = int(np.floor(duration * SAMPLE_RATE))
    assert len(music.note(duration=duration,
                          sample_rate=SAMPLE_RATE)) == expected
    assert len(music.silence(duration=duration,
                             sample_rate=SAMPLE_RATE)) == expected


@pytest.mark.parametrize("freq", [110.0, 220.0, 443.7])
def test_the_note_is_the_lookup_equation_lut_writes(freq):
    """gamma_i = floor(i * f * Lambda_table / f_s), read modulo the table.

    eq:notaBasica states the note as a period repeated; eq:lut is how that
    is done with a table of a fixed size, which is what the package does.
    """
    duration = 0.05
    table = music.WAVEFORM_TRIANGULAR
    produced = music.note(freq=freq, duration=duration, waveform_table=table,
                          sample_rate=SAMPLE_RATE)

    count = int(np.floor(duration * SAMPLE_RATE))
    index = np.arange(count)
    table_length = len(table)
    gamma = np.floor(
        index * freq * table_length / SAMPLE_RATE).astype(np.int64)
    assert np.array_equal(produced, np.asarray(table)[gamma % table_length])


# --------------------------------------------------------------------------
# eq:distOuvidos, eq:dti, eq:dii, eq:angulo -- the geometric localization
# --------------------------------------------------------------------------

def test_localize_delays_and_attenuates_by_the_geometry_the_article_gives():
    """d, d', ITD and IID, straight from the figure the article draws.

    d = sqrt((x - zeta/2) ** 2 + y ** 2) is the near ear and d' the far one
    (eq:distOuvidos); ITD is their difference over the speed of sound
    (eq:dti); IID as a multiplier is d / d' (eq:dii, as eq:locImpl applies
    it).
    """
    x, y, zeta, air_temp = 2.0, 1.0, 0.215, 20
    speed = 331.3 + .606 * air_temp

    near = np.sqrt((x - zeta / 2) ** 2 + y ** 2)          # eq:distOuvidos
    far = np.sqrt((x + zeta / 2) ** 2 + y ** 2)
    itd = (far - near) / speed                            # eq:dti
    iid = near / far                                      # eq:dii
    expected_delay = int(itd * SAMPLE_RATE)               # eq:locImpl

    tone = music.note(freq=440, duration=0.1, sample_rate=SAMPLE_RATE)
    left, right = music.localize(sonic_vector=tone, x=x, y=y, zeta=zeta,
                                 air_temp=air_temp, sample_rate=SAMPLE_RATE)

    # The near ear carries the sound unattenuated and unmoved.
    assert np.array_equal(right[:len(tone)], tone)
    # The far one is quieter by the ratio, and later by the delay.
    delayed = left[expected_delay:expected_delay + len(tone)]
    assert np.allclose(delayed, tone * iid, atol=1e-12)
    assert np.allclose(left[:expected_delay], 0)


def test_the_azimuth_is_the_arctangent_equation_angulo_gives():
    """theta = arctan(y, x), measured from the axis through the ears.

    A source on the median plane is equidistant from both ears, so the two
    channels come out identical; one on the ear axis is as far to a side as
    the model goes. This is the convention that makes theta = 90 and
    theta = -90 the same sound, which is the cone of confusion the article
    names and the HRTF gap this package documents.
    """
    tone = music.note(freq=440, duration=0.05, sample_rate=SAMPLE_RATE)

    # x = 0 puts the source straight ahead: arctan(y, 0) is a right angle.
    ahead = music.localize(sonic_vector=tone, x=0.0, y=1.0,
                           sample_rate=SAMPLE_RATE)
    assert np.array_equal(ahead[0], ahead[1])

    # And behind, arctan(-y, 0), renders the same two channels.
    behind = music.localize(sonic_vector=tone, x=0.0, y=-1.0,
                            sample_rate=SAMPLE_RATE)
    assert np.array_equal(behind[0], behind[1])


# --------------------------------------------------------------------------
# eq:conv and eq:diferencas -- the two filters
# --------------------------------------------------------------------------

def test_fir_is_the_convolution_equation_conv_writes():
    """t'_i = sum_j h_j * t_(i-j), of length Lambda_t + Lambda_h - 1."""
    signal = music.note(freq=330, duration=0.02, sample_rate=SAMPLE_RATE)
    kernel = np.array([1.0, 0.5, -0.25, 0.125])

    produced = music.fir(samples=kernel, sonic_vector=signal, freq=False,
                         max_freq=False)
    expected = np.convolve(signal, kernel)

    assert len(produced) == len(signal) + len(kernel) - 1
    assert np.allclose(produced, expected, atol=1e-12)


def test_iir_is_the_difference_equation_diferencas_writes():
    """t'_i = (sum_j a_j t_(i-j) + sum_k b_k t'_(i-k)) / b_0."""
    signal = music.note(freq=330, duration=0.01, sample_rate=SAMPLE_RATE)
    a = np.array([0.6, 0.3])
    b = np.array([1.0, -0.4])

    produced = music.iir(sonic_vector=signal, a=a, b=b)

    expected = np.zeros(len(signal))
    for i in range(len(signal)):
        feedforward = sum(a[j] * signal[i - j]
                          for j in range(min(len(a), i + 1)))
        feedback = sum(b[k] * expected[i - k]
                       for k in range(1, min(len(b), i + 1)))
        expected[i] = (feedforward + feedback) / b[0]

    assert np.allclose(produced, expected, atol=1e-12)


# --------------------------------------------------------------------------
# eq:mixagem and eq:concatenacao -- putting sounds together
# --------------------------------------------------------------------------

def test_mixing_is_the_sample_by_sample_sum_equation_mixagem_writes():
    """t_i = sum_k t_(k,i), and nothing else."""
    first = music.note(freq=220, duration=0.05, sample_rate=SAMPLE_RATE)
    second = music.note(freq=330, duration=0.05, sample_rate=SAMPLE_RATE)
    third = music.note(freq=440, duration=0.05, sample_rate=SAMPLE_RATE)

    assert np.allclose(music.mix(first, second), first + second)
    assert np.allclose(music.mix_many([first, second, third]),
                       first + second + third)


def test_concatenation_lays_the_sounds_end_to_end_as_equation_concatenacao():
    """Sound l starts where the sum of the lengths before it ends."""
    parts = [music.note(freq=f, duration=d, sample_rate=SAMPLE_RATE)
             for f, d in ((220, 0.03), (330, 0.05), (440, 0.02))]
    joined = music.horizontal_stack(*parts)

    assert len(joined) == sum(len(part) for part in parts)
    start = 0
    for part in parts:
        assert np.array_equal(joined[start:start + len(part)], part)
        start += len(part)


# --------------------------------------------------------------------------
# eq:reconsCompleta -- rebuilding a real signal from half its spectrum
# --------------------------------------------------------------------------

def test_a_real_signal_is_the_cosine_sum_equation_reconscompleta_writes():
    """t_i from half a spectrum, as `spectra.tex` derives it.

    A real signal needs only the coefficients up to the Nyquist bin,
    because the rest are their conjugates. `noise` builds exactly such a
    half spectrum and inverts it, and this is the check that its conjugate
    handling is right -- the MASS reference built the same array with a
    real dtype, silently discarding the imaginary part of every coefficient
    it set.

    Two departures from the equation as it is typeset, both established
    below by reconstructing a known spectrum:

    * The phase is ``+arctan(b_k, a_k)``, not ``-``. The line above it in
      `spectra.tex` gives the sum as ``a_k cos(w_k i) - b_k sin(w_k i)``,
      and ``a cos(x) - b sin(x)`` is ``R cos(x + arctan2(b, a))``: matching
      ``R cos(phi) = a`` and ``R sin(phi) = b`` fixes the sign.
    * The Nyquist term is ``a_(L/2) cos(pi i) / L``, not the constant
      ``a_(L/2) / L``. That coefficient stands for the alternating sequence
      at half the sample rate, so it changes sign every sample.

    With both, the reconstruction is exact to 6e-15; with the equation as
    typeset it is out by 4e-3 on the spectrum used here.
    """
    length = 512
    rng = np.random.default_rng(0)
    coefficients = np.zeros(length, dtype=complex)
    coefficients[:length // 2] = rng.uniform(0, 1, length // 2) * np.exp(
        1j * rng.uniform(0, 2 * np.pi, length // 2))
    # a_0 and the Nyquist coefficient are real, which is what writing them
    # as a_0 and a_(Lambda/2) rather than as moduli means.
    coefficients[0] = coefficients[0].real
    coefficients[length // 2] = 1.0
    coefficients[length // 2 + 1:] = np.conj(
        coefficients[1:length // 2][::-1])

    inverted = np.fft.ifft(coefficients)
    assert np.allclose(inverted.imag, 0, atol=1e-12)   # it is real

    a, b = coefficients.real, coefficients.imag
    index = np.arange(length)
    omega = 2 * np.pi * np.arange(length) / length
    partials = (2 / length) * sum(
        np.sqrt(a[k] ** 2 + b[k] ** 2)
        * np.cos(omega[k] * index + np.arctan2(b[k], a[k]))
        for k in range(1, length // 2))

    corrected = (a[0] / length
                 + a[length // 2] * np.cos(np.pi * index) / length
                 + partials)
    assert np.allclose(inverted.real, corrected, atol=1e-12)

    as_typeset = (a[0] / length
                  + a[length // 2] / length * (1 - length % 2)
                  + partials)
    assert not np.allclose(inverted.real, as_typeset, atol=1e-6), (
        "the equation as typeset now reconstructs the signal, so this "
        "test's note about the Nyquist term is out of date")


def test_noise_comes_back_real_because_its_spectrum_is_conjugate_symmetric():
    """The consequence of eq:reconsCompleta, on the routine that uses it."""
    np.random.seed(0)
    samples = music.noise(noise_type="pink", duration=0.1,
                          sample_rate=SAMPLE_RATE)
    assert np.isrealobj(samples)
    assert np.isfinite(samples).all()

    spectrum = np.fft.fft(samples)
    upper = spectrum[1:len(spectrum) // 2]
    lower = spectrum[len(spectrum) // 2 + 1:][::-1]
    assert np.allclose(upper, np.conj(lower), atol=1e-9)


# --------------------------------------------------------------------------
# eq:groups -- the axioms the permutation structures are built on
# --------------------------------------------------------------------------

@pytest.mark.parametrize("family", ["rotations", "mirrors", "dihedral"])
def test_the_permutation_structures_satisfy_the_axioms_of_equation_groups(
        family):
    """Closure, associativity, an identity, and an inverse for each element.

    `notesInMusic.tex` states these while introducing the cyclic and
    dihedral structures the package generates. `rotations` is the cyclic
    group and `dihedral` the one it sits inside; `mirrors` is the coset of
    reflections, which is closed only under composition with itself into
    the rotations, so this checks it as a subset of the dihedral group
    rather than as a group of its own.
    """
    from sympy.combinatorics import Permutation

    structures = music.InterestingPermutations(nelements=4)
    elements = list(getattr(structures, family))
    assert elements

    identity = Permutation(list(range(4)))
    whole = set(structures.dihedral) | {identity}

    for first in elements:
        for second in elements:
            assert first * second in whole              # closure
        # an inverse, inside the same closed set
        assert first ** -1 in whole
        assert first * (first ** -1) == identity

    for a in elements[:3]:
        for b in elements[:3]:
            for c in elements[:3]:
                assert (a * b) * c == a * (b * c)       # associativity


def test_the_rotations_are_a_group_in_their_own_right():
    """The cyclic group closes on itself, which the reflections do not."""
    structures = music.InterestingPermutations(nelements=4)
    rotations = set(structures.rotations)
    for first in rotations:
        for second in rotations:
            assert first * second in rotations


# --------------------------------------------------------------------------
# eq:notaBasicaTimbre -- a note of any timbre is one period repeated
# --------------------------------------------------------------------------

@pytest.mark.parametrize("table", ["sine", "square", "triangle", "sawtooth"])
def test_a_note_of_any_timbre_is_one_period_repeated(table):
    """t_i = l_(i % (f_s / f)): the period is the timbre, and it repeats.

    eq:notaBasica writes the note for the primary waveforms; this one says
    the same of an arbitrary period `l`. The package reaches that through
    the lookup of eq:lut, so the repetition is what to check rather than
    the indexing.
    """
    freq, duration = 441.0, 0.1        # 441 Hz divides 44100 exactly
    produced = music.note(freq=freq, duration=duration,
                          waveform_table=music.waveform_table(table),
                          sample_rate=SAMPLE_RATE)

    period = int(SAMPLE_RATE / freq)
    whole_periods = len(produced) // period
    assert whole_periods > 1
    first = produced[:period]
    for k in range(1, whole_periods):
        assert np.array_equal(produced[k * period:(k + 1) * period], first)


# --------------------------------------------------------------------------
# eq:adsr and eq:adsrApl -- the envelope, and applying it
# --------------------------------------------------------------------------

def test_the_adsr_envelope_is_the_four_pieces_equation_adsr_writes():
    """Attack, decay, sustain and release, each its own expression.

    The article gives an exponential and a linear form for each sloped
    piece; the package's `transition="exp"` is the first of them, rising
    from xi to 1, falling to a_S, holding, and falling from a_S to xi.
    """
    duration, attack, decay, release = 1.0, 100.0, 100.0, 200.0
    sustain_db, db_dev = -6.0, -80.0
    envelope = music.adsr(envelope_duration=duration,
                          attack_duration=attack, decay_duration=decay,
                          sustain_level=sustain_db,
                          release_duration=release, transition="exp",
                          db_dev=db_dev, sample_rate=SAMPLE_RATE)

    lambda_a = int(attack * SAMPLE_RATE * 0.001)
    lambda_d = int(decay * SAMPLE_RATE * 0.001)
    lambda_r = int(release * SAMPLE_RATE * 0.001)
    xi = 10 ** (db_dev / 20)
    a_s = 10 ** (sustain_db / 20)

    i = np.arange(lambda_a)
    attack_expected = xi * (1 / xi) ** (i / (lambda_a - 1))
    assert np.allclose(envelope[:lambda_a], attack_expected)
    assert envelope[0] == pytest.approx(xi)
    assert envelope[lambda_a - 1] == pytest.approx(1.0)

    j = np.arange(lambda_d)
    decay_expected = a_s ** (j / (lambda_d - 1))
    assert np.allclose(envelope[lambda_a:lambda_a + lambda_d],
                       decay_expected)

    sustain = envelope[lambda_a + lambda_d:len(envelope) - lambda_r]
    assert np.allclose(sustain, a_s)

    # The release is where the three sources part company. eq:adsr writes
    # a_S (xi / a_S) ** t, which lands on xi; the package multiplies a fade
    # by a_S and so lands on xi * a_S, and the MASS reference does exactly
    # the same. Package and reference agree to the sample; both differ from
    # the article. See DISCREPANCIES.md -- this is the author's to settle,
    # not something to change under a sample-exact register row.
    k = np.arange(lambda_r)
    as_the_article_writes_it = a_s * (xi / a_s) ** (k / (lambda_r - 1))
    as_implemented = a_s * xi ** (k / (lambda_r - 1))
    release = envelope[len(envelope) - lambda_r:]

    assert np.allclose(release, as_implemented)
    assert not np.allclose(release, as_the_article_writes_it)
    assert release[0] == pytest.approx(a_s)
    assert release[-1] == pytest.approx(xi * a_s)
    # They part by exactly the sustain level, which is what multiplying by
    # a_S rather than interpolating to xi costs: 6 dB here, and growing
    # with any deeper sustain.
    largest_gap = np.max(np.abs(
        20 * np.log10(release / as_the_article_writes_it)))
    assert largest_gap == pytest.approx(-sustain_db, abs=1e-6)


def test_applying_the_envelope_is_the_product_equation_adsrapl_writes():
    """t_i^ADSR = t_i . a_i, sample by sample and nothing else."""
    tone = music.note(freq=440, duration=0.5, sample_rate=SAMPLE_RATE)
    envelope = music.adsr(envelope_duration=0.5, sample_rate=SAMPLE_RATE)
    applied = music.adsr(sonic_vector=tone, sample_rate=SAMPLE_RATE)
    assert np.allclose(applied, tone * envelope)


# --------------------------------------------------------------------------
# eq:p1rev, eq:p2rev, eq:rev -- the two periods of the reverberation
# --------------------------------------------------------------------------

def test_the_reverberation_is_the_two_periods_equations_p1rev_and_p2rev():
    """Sparse reincidences, then noise, both under one decay curve.

    eq:p1rev makes the first period a train of impulses whose probability
    grows as (i/Lambda_1)**2; eq:p2rev fills the second with noise; both
    are scaled by 10 ** (V_dB/20 . i/(Lambda_R - 1)), and eq:rev joins them
    at Lambda_1.
    """
    duration, first_phase, decay = 0.4, 0.1, -50.0
    lambda_r = int(duration * SAMPLE_RATE)
    lambda_1 = int(first_phase * SAMPLE_RATE)

    np.random.seed(0)
    impulse = music.reverb(duration=duration,
                           first_phase_duration=first_phase, decay=decay,
                           sample_rate=SAMPLE_RATE)
    assert len(impulse) == lambda_r

    i = np.arange(lambda_r)
    envelope = 10. ** ((decay / 20) * (i / (lambda_r - 1)))

    # The first period is sparse: most samples are exactly zero, and the
    # ones that are not sit on the decay curve.
    first = impulse[1:lambda_1]                # index 0 is the direct sound
    assert np.count_nonzero(first) < 0.75 * len(first)
    struck = first != 0
    assert np.allclose(first[struck], envelope[1:lambda_1][struck])

    # And it fills up towards its end, as p_i = (i / Lambda_1) ** 2 says.
    early = np.count_nonzero(first[:len(first) // 4])
    late = np.count_nonzero(first[-len(first) // 4:])
    assert late > 3 * early

    # The second period is not sparse, and stays under the same curve.
    second = impulse[lambda_1:]
    assert np.count_nonzero(second) > 0.99 * len(second)
    assert np.all(np.abs(second) <= envelope[lambda_1:] + 1e-12)


def test_the_reverberation_decays_by_exactly_the_curve_it_documents():
    """Dividing the curve out leaves the same sound at any decay.

    The impulse response is noise under 10 ** (V_dB/20 . i/(Lambda_R - 1)),
    and that noise is not stationary -- brown noise wanders, so its local
    RMS is no measure of the envelope over it. Divide the curve out
    instead: if the decay enters only through it, two renders from the same
    seed that differ only in `decay` come back identical, and they do, to
    the sample.
    """
    duration, first_phase = 0.6, 0.1
    renders = {}
    for decay in (-20.0, -50.0, -80.0):
        np.random.seed(0)
        impulse = music.reverb(duration=duration,
                               first_phase_duration=first_phase, decay=decay,
                               sample_rate=SAMPLE_RATE)
        i = np.arange(len(impulse))
        curve = 10. ** ((decay / 20) * (i / (len(impulse) - 1)))
        renders[decay] = impulse / curve

    reference = renders[-20.0]
    for decay, divided in renders.items():
        assert np.allclose(divided, reference, atol=1e-9), (
            f'dividing out the curve for {decay} dB left a different sound, '
            f'so the decay is not entering only through that curve')

    # And the curve really does span the decibels asked for.
    for decay in renders:
        i = np.arange(int(duration * SAMPLE_RATE))
        curve = 10. ** ((decay / 20) * (i / (len(i) - 1)))
        assert music.amp_to_db(curve[-1] / curve[0]) == pytest.approx(decay)


# --------------------------------------------------------------------------
# eq:fmEsp, eq:Bessel, eq:specAM -- what modulation does in the spectrum
# --------------------------------------------------------------------------

def _spectrum(samples, sample_rate=SAMPLE_RATE):
    """Magnitudes and their frequencies, windowed."""
    windowed = samples * np.hanning(len(samples))
    magnitude = np.abs(np.fft.rfft(windowed)) * 2 / np.sum(np.hanning(
        len(samples)))
    return np.fft.rfftfreq(len(samples), 1 / sample_rate), magnitude


def _bessel(k, mu, points=20001):
    """J_k(mu), by the integral `eq:Bessel` writes it as.

    Evaluated here rather than imported so that the sideband test below
    rests on the article's own definition, and so that checking a spectrum
    costs the package no dependency it does not already have.
    """
    k_bar = k % 2
    w = np.linspace(0, np.pi / 2, points)
    integrand = (np.cos(k_bar * np.pi / 2 + mu * np.sin(w))
                 * np.cos(k_bar * np.pi / 2 + k * w))
    return (2 / np.pi) * np.trapezoid(integrand, w)


def _at(freqs, magnitude, target, width=12.0):
    """The peak magnitude within `width` Hz of `target`."""
    band = np.abs(freqs - target) <= width
    return float(magnitude[band].max()) if band.any() else 0.0


@pytest.mark.parametrize("index", [0.5, 1.0, 2.0, 3.0])
def test_fm_puts_bessel_sidebands_where_equation_fmesp_says(index):
    """A carrier at f gains partials at f + k f', each J_k(mu) tall.

    The article expands the FM signal into that sum (eq:fmEsp), with the
    coefficients the Bessel integral of eq:Bessel gives. This is the
    strongest statement it makes about a spectrum, and checking it is the
    difference between rendering something that varies in pitch and
    rendering frequency modulation.
    """
    carrier, modulator, duration = 4000.0, 400.0, 0.5
    deviation = index * modulator          # mu = deviation / f'

    samples = music.note_with_fm(
        freq=carrier, duration=duration, fm=modulator,
        max_fm_deviation=deviation,
        waveform_table=music.WAVEFORM_SINE,
        fm_waveform_table=music.WAVEFORM_SINE, sample_rate=SAMPLE_RATE)

    freqs, magnitude = _spectrum(samples)
    carrier_height = _at(freqs, magnitude, carrier)
    assert carrier_height > 0

    for k in (1, 2, 3):
        expected = abs(_bessel(k, index) / _bessel(0, index))
        for side in (carrier + k * modulator, carrier - k * modulator):
            measured = _at(freqs, magnitude, side) / carrier_height
            assert measured == pytest.approx(expected, abs=0.06), (
                f'the sideband at {side} Hz is {measured:.3f} of the '
                f'carrier; J_{k}({index}) / J_0({index}) is {expected:.3f}')


@pytest.mark.parametrize("k", range(5))
@pytest.mark.parametrize("mu", [0.5, 1.0, 2.0, 3.5])
def test_the_bessel_integral_of_equation_bessel_is_the_function_it_names(
        k, mu):
    """J_k(mu) = (2/pi) int_0^(pi/2) cos(k_bar pi/2 + mu sin w)
    cos(k_bar pi/2 + k w) dw.

    The article writes the coefficients of eq:fmEsp as that integral rather
    than by name. It is the Bessel function of the first kind, and the
    series that defines that function is independent of the integral, so
    agreeing with it establishes both.
    """
    series = sum((-1) ** m / (math.factorial(m) * math.factorial(m + k))
                 * (mu / 2) ** (2 * m + k)
                 for m in range(40))
    assert _bessel(k, mu) == pytest.approx(series, abs=1e-9)


def test_am_puts_one_sideband_each_side_as_equation_specam_says():
    """(1 + M sin(f' t)) P sin(f t) is the carrier plus two partials.

    eq:specAM expands it: the carrier at P, and sidebands at f -/+ f' each
    of height P M / 2. Two partials, not the infinite family FM makes.
    """
    carrier, modulator, depth, duration = 4000.0, 300.0, 0.4, 0.5

    tone = music.note(freq=carrier, duration=duration,
                      waveform_table=music.WAVEFORM_SINE,
                      sample_rate=SAMPLE_RATE)
    modulated = music.am(duration=duration, fm=modulator,
                         max_amplitude=depth,
                         waveform_table=music.WAVEFORM_SINE,
                         sonic_vector=tone, sample_rate=SAMPLE_RATE)

    freqs, magnitude = _spectrum(modulated)
    carrier_height = _at(freqs, magnitude, carrier)

    for side in (carrier - modulator, carrier + modulator):
        ratio = _at(freqs, magnitude, side) / carrier_height
        assert ratio == pytest.approx(depth / 2, abs=0.02)

    # And nothing at twice the modulator away, where FM would have partials.
    for far in (carrier - 2 * modulator, carrier + 2 * modulator):
        assert _at(freqs, magnitude, far) / carrier_height < 0.02
