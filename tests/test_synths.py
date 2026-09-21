import math
import warnings

import numpy as np
import pytest

import music


def test_note_and_phase_consistency():
    dur = 0.01
    n = music.note(freq=440, duration=dur)
    n_phase = music.note_with_phase(freq=440, duration=dur, phase=0)
    assert len(n) == int(dur * 44100)
    assert np.allclose(n, n_phase)


def test_note_with_fm_output_shape():
    dur = 0.01
    n_fm = music.note_with_fm(freq=440, duration=dur, fm=0, max_fm_deviation=0)
    assert len(n_fm) == int(dur * 44100)
    assert n_fm.max() <= 1 and n_fm.min() >= -1


def test_glissando_and_vibrato_lengths():
    dur = 0.01
    g = music.note_with_glissando(start_freq=330, end_freq=330, duration=dur)
    assert len(g) == int(dur * 44100)

    g2 = music.note_with_glissando_vibrato(
        start_freq=220, end_freq=220, duration=dur, max_pitch_dev=0
    )
    assert len(g2) == int(dur * 44100)


def test_noise_and_silence_generation():
    sil = music.silence(duration=0.005)
    assert np.allclose(sil, np.zeros_like(sil))

    white = music.noise('white', duration=0.005)
    assert len(white) == int(0.005 * 44100)
    assert white.max() <= 1 and white.min() >= -1

    gauss = music.gaussian_noise(duration=1)
    assert len(gauss) == 44100
    assert gauss.max() <= 1 and gauss.min() >= -1


def test_noise_no_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        music.noise('white', duration=0.005)


def test_note_with_doppler_stereo_shape():
    data = music.note_with_doppler(number_of_samples=100, stereo=True)
    assert data.shape[0] == 2
    assert data.shape[1] >= 100


def test_gaussian_noise_takes_a_fractional_duration():
    """Regression: `length = duration * sample_rate` stayed a float, so
    np.random.uniform was handed 22050.0 as a size and raised TypeError.
    Every duration that was not a whole number of seconds failed, which
    is most of the durations anyone would ask for.
    """
    samples = music.gaussian_noise(duration=0.5)
    assert len(samples) == int(0.5 * 44100)
    assert np.isfinite(samples).all()


def _one_vibrato_line(freqs, durations, vibratos_freqs, devs, alpha, tables):
    """Render with every vibrato in turn silenced, and with none silenced."""
    full = music.note_with_vibratos_glissandos(
        freqs=freqs, durations=durations, vibratos_freqs=vibratos_freqs,
        vibratos_max_pitch_devs=devs, alpha=alpha, waveform_tables=tables)
    flattened = []
    for silenced in range(len(vibratos_freqs)):
        muted = tuple(tuple(0 for _ in group) if i == silenced else group
                      for i, group in enumerate(devs))
        flattened.append(music.note_with_vibratos_glissandos(
            freqs=freqs, durations=durations, vibratos_freqs=vibratos_freqs,
            vibratos_max_pitch_devs=muted, alpha=alpha,
            waveform_tables=tables))
    return full, flattened


def test_every_vibrato_reaches_the_rendered_note():
    """Each vibrato must change the sound, not just the last one.

    `note_with_vibratos_glissandos` and
    `note_with_vibrato_seq_localization` reused one name for both the list
    of vibrato lines and the list of segments within a line, so each pass
    of the outer loop threw away the vibrato before it, and each appended
    its own concatenation back into the list it was concatenating. The
    result had the right length, so nothing caught it.
    """
    freqs = (220, 440, 330)
    durations = ((0.01, 0.012), (0.008, 0.015, 0.01),
                 (0.006, 0.01, 0.012, 0.004, 0.004))
    vibratos_freqs = ((2, 6, 1), (0.5, 15, 2, 6, 3))
    devs = ((2, 1, 5), (4, 3, 7, 10, 3))
    alpha = ((1, 1), (1, 1, 1), (1, 1, 1, 1, 1))
    tables = ((music.WAVEFORM_TRIANGULAR,) * 2,
              (music.WAVEFORM_SINE,) * 3, (music.WAVEFORM_SINE,) * 5)

    full, flattened = _one_vibrato_line(freqs, durations, vibratos_freqs,
                                        devs, alpha, tables)
    assert len(flattened) == 2
    for i, muted in enumerate(flattened):
        assert not np.array_equal(full, muted), (
            f'silencing vibrato {i} left the rendered note unchanged, so it '
            f'never contributed to it')


def test_the_vibrato_lines_multiply_rather_than_stack_up():
    """A vibrato of no depth is a factor of one, whatever its frequency.

    Not a guard on the accumulator defect above, which survives this
    invariant: with every deviation zero the lines are all ones however
    they are assembled. It pins the surrounding claim instead -- that the
    vibrato frequencies reach the sound only through those lines, so
    nothing else may carry them into the product.
    """
    common = dict(
        freqs=(220, 440, 330),
        durations=((0.01, 0.012), (0.008, 0.015, 0.01),
                   (0.006, 0.01, 0.012, 0.004, 0.004)),
        vibratos_max_pitch_devs=((0, 0, 0), (0, 0, 0, 0, 0)),
        alpha=((1, 1), (1, 1, 1), (1, 1, 1, 1, 1)),
        waveform_tables=((music.WAVEFORM_TRIANGULAR,) * 2,
                         (music.WAVEFORM_SINE,) * 3,
                         (music.WAVEFORM_SINE,) * 5))

    slow = music.note_with_vibratos_glissandos(
        vibratos_freqs=((2, 6, 1), (0.5, 15, 2, 6, 3)), **common)
    fast = music.note_with_vibratos_glissandos(
        vibratos_freqs=((40, 90, 17), (33, 71, 12, 55, 8)), **common)

    assert np.array_equal(slow, fast)
    assert np.isfinite(slow).all()
    assert np.abs(slow).max() <= 1.0


# The distortion index on a vibrato, which nothing asserted until the
# mutation audit: every arithmetic mutation of the `alpha != 1` line
# survived, and the defect underneath them did too. See MUTATION_AUDIT.md.

def _measured_freq(samples, sample_rate=44100):
    """The frequency of a steady tone, from its upward zero crossings."""
    negative = np.signbit(samples)
    crossings = np.where(~negative[1:] & negative[:-1])[0]
    assert len(crossings) > 2, 'too few crossings to measure a frequency'
    return sample_rate / np.mean(np.diff(crossings))


def _vibrato_halves(alpha, freq=440, max_pitch_dev=6):
    """A one-second note whose square vibrato holds each extreme for half.

    A square table makes the oscillatory pattern exactly -1 then +1, so
    the note is two steady tones and the pitch of each can be measured
    rather than inferred.
    """
    note = music.note_with_vibrato(
        freq=freq, duration=1.0, vibrato_freq=1,
        max_pitch_dev=max_pitch_dev, alpha=alpha,
        vibrato_waveform_table=music.WAVEFORM_SQUARE)
    first = _measured_freq(note[int(0.05 * 44100):int(0.45 * 44100)])
    second = _measured_freq(note[int(0.55 * 44100):int(0.95 * 44100)])
    return first, second


def test_a_vibrato_bends_the_pitch_by_the_semitones_it_was_given():
    # Six semitones either way about 440 Hz is 311.13 and 622.25.
    low, high = _vibrato_halves(alpha=1)
    assert low == pytest.approx(440 * 2 ** -0.5, rel=1e-3)
    assert high == pytest.approx(440 * 2 ** 0.5, rel=1e-3)


@pytest.mark.parametrize("alpha", [0.5, 1.5, 2, 3])
def test_a_distorted_vibrato_bends_by_its_index_without_folding(alpha):
    """The index bends the deviation; it does not rectify it.

    Raising the signed pattern directly, as the reference does, made a
    fractional index NaN over the half-cycle where the pattern is
    negative and made an even one push both halves the same way. Neither
    showed up in a test: every arithmetic mutation of this line survived
    the first audit, because nothing measured the pitch it produces.
    """
    bend = (6 / 12) ** alpha
    low, high = _vibrato_halves(alpha=alpha)
    assert low == pytest.approx(440 * 2 ** -bend, rel=1e-3)
    assert high == pytest.approx(440 * 2 ** bend, rel=1e-3)
    # ...and the two halves sit either side of the carrier, which is what
    # an even index destroyed by making both of them the same.
    assert low < 440 < high
    assert low * high == pytest.approx(440 ** 2, rel=1e-2)


#: The five routines that carry a distortion index on a vibrato, with a
#: fractional one for each. `note_with_vibrato_seq_localization` takes its
#: indices as one row per line, the first of them the glissando's.
_FRACTIONAL = [
    ("note_with_vibrato", dict(duration=0.5, alpha=0.5)),
    ("note_with_two_vibratos", dict(duration=0.5, alphav1=0.5, alphav2=1.5)),
    ("note_with_glissando_vibrato", dict(duration=0.5, alpha_vibrato=0.5)),
    ("note_with_two_vibratos_glissando",
     dict(duration=0.5, alphav1=0.5, alphav2=1.5)),
    ("note_with_vibrato_seq_localization",
     dict(alpha=((1, 1), (0.5, 0.5, 0.5), (1, 1, 1, 1, 1), (1, 1, 1)))),
]


@pytest.mark.parametrize("routine, settings", _FRACTIONAL)
def test_a_fractional_index_does_not_latch_the_render(routine, settings):
    """The regression for the defect the audit found.

    A fractional index made the per-sample frequency NaN wherever the
    vibrato pattern was negative. That NaN reached the accumulated phase
    and then an `int64` cast, which turns it into INT64_MIN, so the table
    index became constant: the note played normally until the first
    negative half-cycle and then latched to full-scale DC for the rest of
    its length. Every sample was finite, so nothing that checked for NaN
    or for a finite render could see it.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rendered = np.asarray(getattr(music, routine)(**settings))
    assert np.isfinite(rendered).all()
    tail = rendered[..., rendered.shape[-1] // 2:]
    assert len(np.unique(tail)) > 1, 'the render latched to a single value'
    assert np.abs(tail).max() <= 1.0


def test_the_note_with_vibrato_defaults_are_the_ones_its_docstring_promises():
    # 220 Hz for two seconds, wobbling four times a second by two
    # semitones. Nothing compared a bare call with those values, so each
    # of the four could be changed without a test noticing.
    assert np.array_equal(
        music.note_with_vibrato(),
        music.note_with_vibrato(freq=220, duration=2, vibrato_freq=4,
                                max_pitch_dev=2))
    assert len(music.note_with_vibrato()) == 2 * 44100


@pytest.mark.parametrize("start_freq, end_freq", [
    (0.5, 2), (2, 0.5), (1, 4), (4, 1),
])
def test_a_glissando_sweeps_frequencies_below_one_hertz(start_freq, end_freq):
    """The ratio guard is "positive", not "greater than one".

    `_require_a_ratio` refuses a sweep that has no ratio, which means a
    zero or negative endpoint. Nothing swept from or to a frequency in
    (0, 1], so the guard could have been `> 1` at either end and stayed
    green -- refusing sub-hertz sweeps that are perfectly well defined.
    """
    rendered = music.note_with_glissando(
        start_freq=start_freq, end_freq=end_freq, duration=0.05)
    assert len(rendered) == int(0.05 * 44100)
    assert np.isfinite(rendered).all()


# The multi-vibrato routines, whose distorted paths the first oscillator
# pass left unmeasured in the same way `note_with_vibrato`'s was.

def _bend(deviation, index):
    """The documented bend, in plain Python so the test does not borrow
    the implementation's own helper to check the implementation."""
    return math.copysign(abs(deviation) ** index, deviation)


def _steady_pitch(samples, start, end, sample_rate=44100):
    """The frequency of the render between two times, in seconds."""
    window = samples[int(start * sample_rate):int(end * sample_rate)]
    return _measured_freq(window, sample_rate)


@pytest.mark.parametrize("alphav1, alphav2", [(1, 1), (2, 1), (1, 2)])
def test_two_vibratos_each_bend_the_pitch_by_their_own_depth(alphav1, alphav2):
    """Two square vibratos at different rates make four steady pitches.

    One at 1 Hz and one at 2 Hz divide a second into quadrants holding
    (-1,-1), (-1,+1), (+1,-1) and (+1,+1), so each quadrant is a steady
    tone whose frequency identifies `nu1`, `nu2`, both signs and the
    product that combines them. Nothing measured any of it: every
    arithmetic edit to the line survived, and so did the mutants that
    changed which branch a lopsided pair of indices takes.
    """
    freq, nu1, nu2 = 440, 2, 7
    note = music.note_with_two_vibratos(
        freq=freq, duration=1.0, vibrato_freq=1, secondary_vibrato_freq=2,
        nu1=nu1, nu2=nu2, alphav1=alphav1, alphav2=alphav2,
        vibrato_waveform_table=music.WAVEFORM_SQUARE,
        sec_vibrato_waveform_table=music.WAVEFORM_SQUARE)
    # Both tables named: until the secondary read was corrected, naming
    # only the first gave the second one the same square wave by accident,
    # and this test passed because of the defect it now guards.
    quadrants = ((0.03, 0.22, -1, -1), (0.28, 0.47, -1, 1),
                 (0.53, 0.72, 1, -1), (0.78, 0.97, 1, 1))
    for start, end, first, second in quadrants:
        expected = (freq * 2 ** _bend(first * nu1 / 12, alphav1)
                    * 2 ** _bend(second * nu2 / 12, alphav2))
        assert _steady_pitch(note, start, end) == pytest.approx(expected,
                                                                rel=2e-3)


def test_the_note_with_two_vibratos_defaults_are_the_ones_it_declares():
    assert np.array_equal(
        music.note_with_two_vibratos(),
        music.note_with_two_vibratos(freq=220, duration=2, vibrato_freq=2,
                                     secondary_vibrato_freq=6, nu1=2, nu2=4,
                                     sample_rate=44100))
    assert len(music.note_with_two_vibratos()) == 2 * 44100


@pytest.mark.parametrize("alpha, alpha_vibrato", [(1, 1), (2, 1), (1, 2)])
def test_a_glissando_vibrato_bends_about_the_sweep_it_rides(alpha,
                                                            alpha_vibrato):
    """With the sweep flat, the vibrato alone sets the pitch."""
    note = music.note_with_glissando_vibrato(
        start_freq=440, end_freq=440, duration=1.0, vibrato_freq=1,
        max_pitch_dev=6, alpha=alpha, alpha_vibrato=alpha_vibrato,
        vibrato_waveform_table=music.WAVEFORM_SQUARE)
    bend = _bend(6 / 12, alpha_vibrato)
    assert _steady_pitch(note, 0.05, 0.45) == pytest.approx(440 * 2 ** -bend,
                                                            rel=2e-3)
    assert _steady_pitch(note, 0.55, 0.95) == pytest.approx(440 * 2 ** bend,
                                                            rel=2e-3)


def test_a_glissando_vibrato_sweeps_by_a_constant_ratio():
    """With no deviation, the vibrato is a factor of one and the sweep shows.

    Halfway through, an exponential sweep from 220 to 880 is at their
    geometric mean, 440, where a linear one would be at 550. Nothing
    distinguished the two, nor the endpoints it sweeps between.
    """
    note = music.note_with_glissando_vibrato(
        start_freq=220, end_freq=880, duration=1.0, max_pitch_dev=0)
    assert _steady_pitch(note, 0.47, 0.53) == pytest.approx(440, rel=5e-3)
    assert _steady_pitch(note, 0.0, 0.06) == pytest.approx(229.3, rel=1e-2)
    assert _steady_pitch(note, 0.94, 1.0) == pytest.approx(844.2, rel=1e-2)


def test_the_glissando_vibrato_defaults_are_the_ones_it_declares():
    assert np.array_equal(
        music.note_with_glissando_vibrato(),
        music.note_with_glissando_vibrato(start_freq=220, end_freq=440,
                                          duration=2, vibrato_freq=4,
                                          max_pitch_dev=2, sample_rate=44100))
    assert len(music.note_with_glissando_vibrato()) == 2 * 44100


@pytest.mark.parametrize("routine, primary, secondary", [
    ("note_with_two_vibratos", "vibrato_waveform_table",
     "sec_vibrato_waveform_table"),
    ("note_with_two_vibratos_glissando", "tabv1", "tabv2"),
])
def test_the_second_vibrato_reads_its_own_waveform_table(routine, primary,
                                                         secondary):
    """It read the first one's, as the MASS reference still does.

    The second table was accepted, documented, and used only for its
    length, so asking for a square second vibrato under a sine first one
    gave back a sound identical to two sines. No mutant found this: the
    name was wrong, not the arithmetic, and swapping one valid table for
    another is not an edit mutmut makes.
    """
    render = getattr(music, routine)
    sine, square = music.WAVEFORM_SINE, music.WAVEFORM_SQUARE
    both_sine = render(duration=0.2, **{primary: sine, secondary: sine})
    square_second = render(duration=0.2, **{primary: square, secondary: sine})
    sine_second = render(duration=0.2, **{primary: sine, secondary: square})
    assert not np.array_equal(both_sine, sine_second), (
        'the second vibrato ignored the table it was given')
    assert not np.array_equal(both_sine, square_second)
    assert not np.array_equal(square_second, sine_second)


@pytest.mark.parametrize("routine, primary, secondary", [
    ("note_with_two_vibratos", "vibrato_waveform_table",
     "sec_vibrato_waveform_table"),
    ("note_with_two_vibratos_glissando", "tabv1", "tabv2"),
])
def test_two_vibratos_may_have_tables_of_different_lengths(routine, primary,
                                                           secondary):
    # The second lookup took its modulus from the second table's length
    # and applied it to the first table, so a shorter first table was
    # indexed past its end.
    render = getattr(music, routine)
    short, long = music.WAVEFORM_SINE[::4], music.WAVEFORM_SINE
    for tables in ((short, long), (long, short)):
        out = render(duration=0.05,
                     **{primary: tables[0], secondary: tables[1]})
        assert np.isfinite(out).all()
        assert np.abs(out).max() <= 1.0


@pytest.mark.parametrize("alpha, alphav1, alphav2",
                         [(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2)])
def test_two_vibratos_over_a_glissando_each_bend_by_their_own_depth(
        alpha, alphav1, alphav2):
    """The same four quadrants, over a sweep held flat."""
    freq, dev1, dev2 = 440, 2, 7
    note = music.note_with_two_vibratos_glissando(
        start_freq=freq, end_freq=freq, duration=1.0, vibrato_freq=1,
        secondary_vibrato_freq=2, max_pitch_dev=dev1,
        secondary_max_pitch_dev=dev2, alpha=alpha, alphav1=alphav1,
        alphav2=alphav2, tabv1=music.WAVEFORM_SQUARE,
        tabv2=music.WAVEFORM_SQUARE)
    quadrants = ((0.03, 0.22, -1, -1), (0.28, 0.47, -1, 1),
                 (0.53, 0.72, 1, -1), (0.78, 0.97, 1, 1))
    for start, end, first, second in quadrants:
        expected = (freq * 2 ** _bend(first * dev1 / 12, alphav1)
                    * 2 ** _bend(second * dev2 / 12, alphav2))
        assert _steady_pitch(note, start, end) == pytest.approx(expected,
                                                                rel=2e-3)


def test_two_vibratos_over_a_glissando_sweep_by_a_constant_ratio():
    note = music.note_with_two_vibratos_glissando(
        start_freq=220, end_freq=880, duration=1.0, max_pitch_dev=0,
        secondary_max_pitch_dev=0)
    assert _steady_pitch(note, 0.47, 0.53) == pytest.approx(440, rel=5e-3)
    assert _steady_pitch(note, 0.0, 0.06) == pytest.approx(229.3, rel=1e-2)
    assert _steady_pitch(note, 0.94, 1.0) == pytest.approx(844.2, rel=1e-2)


def test_the_two_vibratos_glissando_defaults_are_the_ones_it_declares():
    assert np.array_equal(
        music.note_with_two_vibratos_glissando(),
        music.note_with_two_vibratos_glissando(
            start_freq=220, end_freq=440, duration=2, vibrato_freq=2,
            secondary_vibrato_freq=6, max_pitch_dev=2,
            secondary_max_pitch_dev=.5, sample_rate=44100))
    assert len(music.note_with_two_vibratos_glissando()) == 2 * 44100


@pytest.mark.parametrize("routine, settings", [
    ("note_with_glissando_vibrato", dict(max_pitch_dev=0)),
    ("note_with_two_vibratos_glissando",
     dict(max_pitch_dev=0, secondary_max_pitch_dev=0)),
])
@pytest.mark.parametrize("alpha", [1, 2, 0.5])
def test_a_glissando_curves_its_sweep_by_the_index(routine, settings, alpha):
    """`alpha` bends *when* the sweep gets where it is going.

    The sweep is `start * (end / start) ** ((i / (L - 1)) ** alpha)`, so
    an index of 2 is a quarter of the way up in pitch at halfway through,
    and an index of a half is past three quarters. Measuring a flat sweep,
    or a curved one at index 1, leaves every term of it unpinned: the
    ratio, the exponent and the `L - 1` that normalises the ramp.
    """
    note = getattr(music, routine)(start_freq=220, end_freq=880,
                                   duration=1.0, alpha=alpha, **settings)
    for moment in (0.25, 0.5, 0.75):
        expected = 220 * 4 ** (moment ** alpha)
        assert _steady_pitch(note, moment - 0.03,
                             moment + 0.03) == pytest.approx(expected,
                                                             rel=5e-3)


#: One glissando line and one vibrato line, both a single segment long.
#: `durations` drives both: row 0 is the sweep's segments, each later row
#: a vibrato line's.
def _one_line(alpha_vibrato=1, dev=6, table=None):
    return music.note_with_vibratos_glissandos(
        freqs=(440, 440), durations=((1.0,), (1.0,)),
        vibratos_freqs=((1,),), vibratos_max_pitch_devs=((dev,),),
        alpha=((1,), (alpha_vibrato,)),
        waveform_tables=((music.WAVEFORM_SINE,),
                         (table if table is not None
                          else music.WAVEFORM_SQUARE,)))


@pytest.mark.parametrize("alpha_vibrato", [1, 2, 0.5])
def test_a_vibrato_line_bends_by_its_own_index(alpha_vibrato):
    """The sequence form carried the same signed power, on a line the
    earlier scan missed because its `**` ended a line and its `alpha`
    began the next one. A fractional index was NaN here too."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        note = _one_line(alpha_vibrato=alpha_vibrato)
    bend = _bend(6 / 12, alpha_vibrato)
    assert _steady_pitch(note, 0.05, 0.45) == pytest.approx(440 * 2 ** -bend,
                                                            rel=2e-3)
    assert _steady_pitch(note, 0.55, 0.95) == pytest.approx(440 * 2 ** bend,
                                                            rel=2e-3)


def test_each_glissando_segment_sweeps_between_its_own_frequencies():
    """Segments of different lengths, each reaching its own endpoints.

    Halfway through a segment an exponential sweep is at the geometric
    mean of its ends. Equal durations could not tell the segment lengths
    apart, and a flat sweep could not tell the endpoints apart.
    """
    note = music.note_with_vibratos_glissandos(
        freqs=(220, 880, 220), durations=((0.25, 0.75),),
        vibratos_freqs=((1, 1),), vibratos_max_pitch_devs=((0, 0),),
        alpha=((1, 1), (1, 1)),
        waveform_tables=((music.WAVEFORM_SINE,) * 2,
                         (music.WAVEFORM_SINE,) * 2))
    assert len(note) == 44100
    assert _steady_pitch(note, 0.105, 0.145) == pytest.approx(440, rel=5e-3)
    assert _steady_pitch(note, 0.605, 0.645) == pytest.approx(440, rel=5e-3)


def test_a_vibrato_line_reads_the_table_it_was_given():
    square = _one_line(table=music.WAVEFORM_SQUARE)
    sine = _one_line(table=music.WAVEFORM_SINE)
    assert not np.array_equal(square, sine)


def test_the_vibratos_glissandos_defaults_are_the_ones_it_declares():
    assert np.array_equal(
        music.note_with_vibratos_glissandos(),
        music.note_with_vibratos_glissandos(sample_rate=44100))
