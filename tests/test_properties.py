"""Laws that should hold whatever the parameters, read several ways.

The rest of the suite checks routines one at a time and mostly on their
own defaults: `test_fidelity.py` against the closed form each implements,
`test_artifacts.py` for defects in what comes out, `test_degenerate.py`
for what happens at the edge of a parameter. Between those sits the range
nobody was looking at -- ordinary parameters that are simply not the
default ones, and routines used together rather than alone.

Three axes, and the file is in three parts.

**Parameters.** The same law at several sample rates, durations and
pitches. This is where the two noise routines were found to be broken for
any odd number of samples -- half a second at 22,050 Hz is 11,025, and
`gaussian_noise` raised where `noise` quietly discarded an imaginary part
worth 8.7% of the signal.

**Analyses.** The same render read more than one way. A spectrum says
whether the partials are where they belong; an autocorrelation says
whether the period is; Parseval says whether the energy in one agrees
with the energy in the other. A defect that hides from one measure
usually does not hide from all three, which is the argument for having
more than one.

**Combinations.** Routines composed, which is how anyone actually uses
them: two shapers over a note, a sum of sums, a piece stacked and written
and read back. Several of these are algebraic laws -- multiplication
commutes, mixing associates, a linear filter is linear -- and an algebraic
law is a good test because it is cheap, exact, and fails loudly when
something has quietly stopped being what it was.
"""

import inspect

import numpy as np
import pytest

import music
from test_public_api import ZERO_ARG_EXPORTS, _callable_with_defaults

SAMPLE_RATES = [8000, 22050, 44100, 48000]


def _takes(name, *parameters):
    """Whether the export named takes all of `parameters`."""
    signature = inspect.signature(_callable_with_defaults(name)).parameters
    return all(parameter in signature for parameter in parameters)


TIMED = sorted(name for name in ZERO_ARG_EXPORTS
               if _takes(name, "duration", "sample_rate"))

#: Renders whose length is deliberately not the duration times the rate.
#: Everything else is exact to the sample, at every rate and duration
#: tried, which is a tighter law than it looks: a routine that rounds
#: where another truncates would show up here.
LENGTH_IS_NOT_THE_DURATION = {
    "note_with_doppler": (
        "the far ear is delayed and the render is as long as the delayed "
        "channel, so it runs up to 29 samples over"),
    "trill": (
        "it renders whole notes, so a duration that is not a whole number "
        "of them comes up short -- by 1,412 samples of a second at 17 "
        "notes per second. A duration that *is* a whole number of notes "
        "now renders all of them: it used to drop the last one whenever "
        "the note length divided the duration exactly, which a tolerance "
        "of one whole note in test_fidelity.py could not tell from a "
        "right answer"),
}


# ---------------------------------------------------------------------
# Parameters: the same law somewhere other than the defaults
# ---------------------------------------------------------------------

@pytest.mark.parametrize("name", TIMED)
@pytest.mark.parametrize("sample_rate", SAMPLE_RATES)
def test_a_render_is_as_long_as_the_duration_and_rate_it_was_given(
        name, sample_rate):
    """Exact to the sample, except where the register says otherwise."""
    if name in LENGTH_IS_NOT_THE_DURATION:
        return
    function = _callable_with_defaults(name)
    for duration in (0.25, 0.5, 1.0):
        rendered = np.asarray(function(duration=duration,
                                       sample_rate=sample_rate), dtype=float)
        wanted = int(duration * sample_rate)
        assert rendered.shape[-1] == wanted, (
            f"{name} at {sample_rate} Hz for {duration}s rendered "
            f"{rendered.shape[-1]} samples rather than {wanted}. If that "
            "is deliberate, register it with the reason.")


@pytest.mark.parametrize("name", sorted(LENGTH_IS_NOT_THE_DURATION))
def test_a_registered_length_is_still_wrong(name):
    """So the register cannot outlive the thing it excuses."""
    function = _callable_with_defaults(name)
    deviations = {abs(np.asarray(function(duration=d, sample_rate=r),
                                 dtype=float).shape[-1] - int(d * r))
                  for r in SAMPLE_RATES for d in (0.25, 0.5, 1.0)}
    assert max(deviations) > 0, (
        f"{name} now renders exactly the duration it is given; delete the "
        "entry.")


@pytest.mark.parametrize("sample_rate", SAMPLE_RATES)
@pytest.mark.parametrize("freq", [110, 220, 440, 1000, 2000])
def test_a_note_reads_as_the_frequency_it_was_asked_for(freq, sample_rate):
    """Across two decades of pitch and from 8 kHz to 48 kHz.

    The strongest partial of a note should be the note. It is the
    cheapest possible check and it covers a lot: a table read at the
    wrong rate, a phase accumulated per sample instead of per second, a
    frequency scaled by the wrong constant.
    """
    rendered = np.asarray(music.note(freq, 1.0, sample_rate=sample_rate),
                          dtype=float)
    spectrum = np.abs(np.fft.rfft(rendered))
    spectrum[0] = 0.0
    strongest = np.fft.rfftfreq(len(rendered), 1 / sample_rate)[
        spectrum.argmax()]
    assert strongest == pytest.approx(freq, rel=0.01)


@pytest.mark.parametrize("sample_rate", SAMPLE_RATES)
def test_noise_renders_at_any_rate_including_an_odd_number_of_samples(
        sample_rate):
    """Half a second at 22,050 Hz is 11,025 samples, and both noises
    assumed an even count: `gaussian_noise` raised, and `noise` built a
    spectrum that was not Hermitian and threw away the imaginary part the
    inverse transform came back with."""
    for name in ("noise", "gaussian_noise"):
        rendered = np.asarray(getattr(music, name)(
            duration=0.5, sample_rate=sample_rate), dtype=float)
        assert rendered.size == int(0.5 * sample_rate)
        assert np.isfinite(rendered).all()


# ---------------------------------------------------------------------
# Analyses: the same render, read more than one way
# ---------------------------------------------------------------------

@pytest.mark.parametrize("samples", [11025, 11026])
def test_a_noise_spectrum_describes_a_real_signal(samples):
    """A real signal's spectrum satisfies X[N - k] = conj(X[k]).

    Both noises build a spectrum and invert it, taking `.real` of what
    comes back -- so a spectrum that is not Hermitian does not fail, it
    silently becomes a different signal than the one specified. The odd
    case is the one that was broken, and the imaginary part being
    discarded there was worth 8.7% of the signal.

    Reading the render's own spectrum, rather than the one the routine
    built, is what makes this a check on the result rather than on the
    arithmetic.
    """
    rate = 44100
    rendered = np.asarray(music.noise(duration=samples / rate,
                                      sample_rate=rate), dtype=float)
    if rendered.size != samples:      # pragma: no cover - rounding
        pytest.skip(f"{samples} samples did not come out of that duration")

    spectrum = np.fft.fft(rendered)
    mirrored = np.conj(spectrum[1:][::-1])
    assert np.abs(spectrum[1:] - mirrored).max() < 1e-9 * np.abs(
        spectrum).max()


@pytest.mark.parametrize("freq", [220, 440, 1000])
def test_a_note_repeats_at_the_period_its_frequency_implies(freq):
    """An autocorrelation, which says what a spectrum does in another
    language: the lag of the first strong peak is the period.

    Two measures agreeing on the pitch is worth more than one measure
    agreeing with itself, since a defect in the table read would move
    both and a defect in one analysis would move neither.
    """
    rendered = np.asarray(music.note(freq, 0.5), dtype=float)
    centred = rendered - rendered.mean()
    correlation = np.correlate(centred, centred, "full")[len(centred) - 1:]

    period = 44100 / freq
    window = correlation[int(period * 0.5):int(period * 2)]
    lag = int(period * 0.5) + int(np.argmax(window))
    assert 44100 / lag == pytest.approx(freq, rel=0.01)


@pytest.mark.parametrize("name", ["note", "note_with_vibrato", "silence"])
def test_the_energy_in_the_samples_is_the_energy_in_the_spectrum(name):
    """Parseval's theorem, which is a check on the render and on the
    reading of it at once: if either the samples or the transform of them
    were wrong, the two sums would part company."""
    rendered = np.asarray(getattr(music, name)(duration=0.2), dtype=float)
    in_time = float(np.sum(rendered ** 2))
    in_frequency = float(np.sum(np.abs(np.fft.fft(rendered)) ** 2)
                         / rendered.size)
    assert in_frequency == pytest.approx(in_time, rel=1e-9, abs=1e-12)


def test_a_fade_out_never_rises():
    """Monotonicity, which no spectral measure would notice.

    An envelope that dips and recovers has the same length, the same
    endpoints and nearly the same spectrum as one that does not.
    """
    for method in ("exp", "linear"):
        for perc in (1, 50, 100):
            envelope = np.asarray(music.fade(duration=0.5, method=method,
                                             perc=perc), dtype=float)
            assert np.all(np.diff(envelope) <= 1e-12), (
                f"a {method} fade out at perc={perc} rises somewhere")


@pytest.mark.parametrize("shaper", ["adsr", "fade", "tremolo", "loud", "am"])
def test_an_envelope_applied_is_the_envelope_multiplied_in(shaper):
    """Every shaper documents both uses -- pass a sound to shape it, omit
    one to get the envelope -- and the two should be the same operation.

    They are, exactly, for all five. Which is worth knowing rather than
    assuming: a routine that shaped by some other route would still look
    right on its own and stop agreeing with the envelope it hands out.
    """
    sound = music.note(440, 0.3)
    function = getattr(music, shaper)
    applied = np.asarray(function(sonic_vector=sound), dtype=float)
    envelope = np.asarray(function(number_of_samples=len(sound)),
                          dtype=float)
    assert np.allclose(applied, envelope * sound, atol=1e-12)


# ---------------------------------------------------------------------
# Combinations: routines used the way anyone uses them
# ---------------------------------------------------------------------

def test_two_multiplicative_shapers_commute():
    """An ADSR over a tremolo is a tremolo over an ADSR, to the last bit.

    Both multiply, and multiplication commutes -- so this is really a test
    that both of them still only multiply. A shaper that started
    normalizing, or clipping, or reading the sound it was given for
    anything but its length, would break this and little else.
    """
    sound = music.note(440, 0.3)
    one_way = music.adsr(sonic_vector=music.tremolo(sonic_vector=sound))
    other = music.tremolo(sonic_vector=music.adsr(sonic_vector=sound))
    assert np.allclose(one_way, other, atol=1e-12)


def test_mixing_associates_and_commutes():
    """Sums of sums, in any order and grouping."""
    a, b, c = (music.note(f, 0.3) for f in (440, 550, 660))
    assert np.allclose(music.mix(music.mix(a, b), c),
                       music.mix(a, music.mix(b, c)), atol=1e-12)
    assert np.allclose(music.mix(a, b), music.mix(b, a), atol=1e-12)


def test_stacking_associates():
    """And the same for laying sounds end to end."""
    a, b, c = (music.note(f, 0.3) for f in (440, 550, 660))
    assert np.array_equal(
        music.horizontal_stack(music.horizontal_stack(a, b), c),
        music.horizontal_stack(a, b, c))


@pytest.mark.parametrize("design,args", [
    ("low_pass", (0.1,)), ("high_pass", (0.2,)),
    ("band_pass", (0.15, 0.05)), ("band_reject", (0.3, 0.02)),
])
def test_a_designed_filter_is_linear_and_homogeneous(design, args):
    """Filtering a sum is summing the filtered, and scaling passes
    through. `test_filters_response.py` establishes this for `iir` with
    coefficients written by hand; this says the designs produce
    coefficients that are still what `iir` was proved linear over.
    """
    feedforward, feedback = getattr(music, design)(*args)
    a, b = music.note(440, 0.2), music.note(660, 0.2)

    summed = np.asarray(music.iir(a + b, feedforward, feedback), dtype=float)
    separately = (np.asarray(music.iir(a, feedforward, feedback), dtype=float)
                  + np.asarray(music.iir(b, feedforward, feedback),
                               dtype=float))
    assert np.allclose(summed, separately, atol=1e-9)

    scaled = np.asarray(music.iir(3.0 * a, feedforward, feedback),
                        dtype=float)
    assert np.allclose(scaled, 3.0 * np.asarray(
        music.iir(a, feedforward, feedback), dtype=float), atol=1e-9)


def test_a_composed_piece_survives_being_written_and_read(tmp_path):
    """The whole chain a reader of the tutorial would write: notes,
    shaped, stacked, written, read back.

    Nothing else in the suite runs the parts together and then off the
    disk. The round trip is checked by correlation rather than by
    equality, since writing normalizes and quantises -- and correlation is
    the measure that would catch a channel swapped, a fade applied twice
    or a stack in the wrong order, none of which changes a peak level.
    """
    piece = music.horizontal_stack(
        *[music.adsr(sonic_vector=music.note(freq, 0.2))
          for freq in (440, 550, 660)])
    path = tmp_path / "piece.wav"
    music.write_wav_mono(piece, str(path))
    back = music.read_wav(str(path))

    assert len(back) == len(piece)
    assert np.corrcoef(piece, back)[0, 1] > 0.9999


def test_a_stereo_chain_survives_being_written_and_read(tmp_path):
    """The same for a sound placed in space, where a channel could be
    lost or swapped without changing anything a mono check would see."""
    placed = np.asarray(music.localize(
        sonic_vector=music.adsr(sonic_vector=music.note(440, 0.3))),
        dtype=float)
    path = tmp_path / "placed.wav"
    music.write_wav_stereo(placed, str(path))
    back = np.asarray(music.read_wav(str(path)), dtype=float)

    assert back.shape == placed.shape
    for channel in range(2):
        assert np.corrcoef(placed[channel], back[channel])[0, 1] > 0.999
    # And the channels stayed where they were put, which a per-channel
    # correlation would not notice if both were the same sound.
    assert (np.corrcoef(placed[0], back[0])[0, 1]
            > np.corrcoef(placed[0], back[1])[0, 1])


@pytest.mark.parametrize("sample_rate", [22050, 44100])
def test_a_whole_short_piece_renders_at_either_rate(sample_rate):
    """Several routines at once, away from every default, at two rates.

    A piece rather than a note: a melody with vibrato, shaped, against a
    filtered noise bed, mixed. If any of them disagreed about what a
    sample rate means, the two would come out different lengths and the
    mix would pad one of them.
    """
    melody = music.horizontal_stack(*[
        music.adsr(sonic_vector=music.note_with_vibrato(
            freq=freq, duration=0.2, vibrato_freq=6, max_pitch_dev=0.5,
            sample_rate=sample_rate))
        for freq in (330, 392, 440)])

    feedforward, feedback = music.low_pass(0.05)
    bed = 0.2 * np.asarray(music.iir(
        music.noise(duration=0.6, sample_rate=sample_rate),
        feedforward, feedback), dtype=float)

    assert len(melody) == int(0.6 * sample_rate)
    assert len(bed) == int(0.6 * sample_rate)

    together = music.mix(melody, bed)
    assert len(together) == len(melody)
    assert np.isfinite(together).all()
    assert np.abs(together).max() > 0.1
