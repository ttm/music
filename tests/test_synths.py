import numpy as np
import warnings

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
