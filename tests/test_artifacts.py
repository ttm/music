"""Artifacts: defects in the samples that the shape of a render hides.

Issue #76 asked for tests that catch a sound rendered with artifacts even
where an ear would not.  The rest of the suite checks that a routine
computes what the article says; these check that what comes out is
playable -- that nothing in it is a defect of the rendering rather than a
property of the sound.

Two measures, both defined below and both applied to every export that
renders on its own defaults:

**A click** is a sample-to-sample step that the signal's own neighbourhood
does not explain.  Largeness alone will not do: a steep attack and a click
are the same size, and what separates them is that the attack's neighbours
are steep too.  So a step counts only when it is more than
`CLICK_RATIO` times the median step of the `BLOCK` of samples around it
*and* more than `CLICK_FLOOR` of the render's peak.  Both conditions earn
their place. Without the ratio, every envelope corner is a click; without
the floor, `louds` fails on a step of 0.005 because its neighbourhood at
that point is flat, and a tiny step in a flat region is not audible as
anything.

**A DC offset** is a mean far from zero relative to the signal's own RMS.
It costs headroom, it moves a speaker cone off centre, and it is exactly
the sort of thing no one hears until two sounds are mixed and the sum
clips early.

Both measures are relative, and the last section of this file is what that
costs. A click has to stand out from what the signal is already doing, so
in silence any step is found, in a 110 Hz note it takes 9% of full scale,
in a 440 Hz note 36%, and at 4 kHz nothing is detectable at all -- eight
times the local median is past the ±1 a sample can hold. The sweep
passing therefore means no render carries a seam-like or gate-like step,
which is what it was built to find, and not that no render clicks.

What the sweep found, and what came of it, is in ASSESSMENT.md.  The short
version: one defect in `pan_transitions`, which interpolated every leg of a
pan from the first point rather than from the previous one and so rendered
a full-scale step at a join; and a family of steps that are not defects,
each registered below with the reason it is there.
"""

import functools
import pathlib
import re

import numpy as np
import pytest

import music
from music.core.functions import normalize_mono
from music.core.io import _quantize
from music.utils import (WAVEFORM_SAWTOOTH, WAVEFORM_SINE, WAVEFORM_SQUARE,
                         WAVEFORM_TRIANGULAR)
from test_public_api import ZERO_ARG_EXPORTS, _callable_with_defaults

#: The neighbourhood a step is judged against, in samples.  129 at 44.1 kHz
#: is 3 ms: long enough to hold a couple of cycles of anything above 700 Hz,
#: short enough that an envelope's corner is judged against the envelope
#: rather than against the whole render.
#:
#: The signal is cut into blocks of this size and each step is judged
#: against the median step of the block it falls in, rather than against a
#: window centred on it. It is the same idea an order of magnitude faster:
#: a window per sample means a median over 129 values 88,200 times per
#: render, which took a minute and a half across the sweep, and the blocks
#: take a second. A click is one sample in its block either way, and one
#: sample does not move a median.
BLOCK = 129

#: How far a step must stand out from its neighbourhood to be a click.
#:
#: Where the margin actually is, since a threshold is only as good as the
#: gap it sits in: most renders that pass sit at 1 to 2, but `trill`
#: reaches 6.1 and `localize2` 5.8 -- both concatenate or delay, and the
#: joins show. The registered ones start at 47.7. So the gap is 6.1 to
#: 47.7 and this sits nearer the bottom of it than the middle, which
#: buys sensitivity at the cost of a routine drifting into a false
#: positive. A failure here is a prompt to look, not proof of a click.
CLICK_RATIO = 8.0

#: And how large it must be against the render's peak, so that a step in a
#: region where nothing is happening is not mistaken for one.
CLICK_FLOOR = 0.01

#: How far the mean may sit from zero, as a fraction of RMS.  White noise
#: over 88,200 samples lands near 0.003 by chance alone.
DC_LIMIT = 0.05


def clicks(samples, ratio=CLICK_RATIO, floor=CLICK_FLOOR):
    """Every step the neighbourhood does not explain, as (index, step)."""
    samples = np.asarray(samples, dtype=float)
    if samples.ndim > 1:
        return [found for channel in samples for found in clicks(
            channel, ratio, floor)]

    peak = float(np.max(np.abs(samples)))
    steps = np.abs(np.diff(samples))
    if not peak or steps.size < BLOCK:
        return []

    whole = steps.size // BLOCK
    scale = np.median(steps[:whole * BLOCK].reshape(whole, BLOCK), axis=1)
    local = np.repeat(scale, BLOCK)
    if local.size < steps.size:      # the tail that does not fill a block
        local = np.concatenate(
            [local, np.full(steps.size - local.size, scale[-1])])

    # A step standing in a neighbourhood of exact zeros is a step out of
    # silence: the ratio cannot score it, and the floor has already found
    # it.
    silent = local == 0
    stands_out = np.divide(steps, local, out=np.full_like(steps, np.inf),
                           where=~silent) > ratio
    hits = np.flatnonzero((steps > floor * peak) & (silent | stands_out))
    return [(int(index), float(steps[index])) for index in hits]


def dc_offset(samples):
    """The mean, as a fraction of RMS.  Zero for anything that oscillates."""
    samples = np.asarray(samples, dtype=float).ravel()
    rms = float(np.sqrt(np.mean(samples ** 2)))
    return abs(float(np.mean(samples))) / rms if rms else 0.0


#: Exports whose default result is a control signal rather than a sound:
#: an envelope to multiply into audio, or an impulse response to convolve
#: with it.  A mean far from zero is what an envelope is *for* -- an ADSR
#: that averaged zero would be silent -- so the DC sweep does not apply.
CONTROL_SIGNALS = {
    "adsr": "an ADSR envelope",
    "adsr_stereo": "an ADSR envelope, per channel",
    "am": "an amplitude-modulation envelope",
    "fade": "a fade envelope",
    "loud": "a loudness transition",
    "louds": "a sequence of loudness transitions",
    "pan_transitions": "a pair of panning envelopes",
    "reverb": "an impulse response",
    "tremolo": "a tremolo envelope",
    "tremolos": "a sequence of tremolo envelopes",
}

#: Renders the click measure cannot speak about at all.  It asks whether a
#: step is explained by its neighbourhood, and a random signal's steps are
#: explained by nothing: over 88,200 samples the largest lands around seven
#: times the median by chance, close enough to CLICK_RATIO that a threshold
#: would decide it differently from run to run.  These are exempt because
#: the question does not apply, not because the answer is known.
UNMEASURABLE = {
    "noise": "a random signal has no neighbourhood that explains anything",
    "gaussian_noise": "the same, from a Gaussian draw",
    "modulated_noise": "the same, under an envelope",
}

#: Renders that really do step, each for a reason worth knowing.  A step
#: here is not a defect, but it is also not nothing, and
#: `test_a_registered_step_is_still_there` fails when one stops happening,
#: so this register cannot rot into a list of stale excuses.
DELIBERATE_STEPS = {
    "reverb": (
        "an impulse response is mostly zeros with an impulse in it, and "
        "the impulse is the point"),
    "isochronic_tones": (
        "a gate is a step twice per pulse. The ramp softens it -- "
        "tests/test_stimulation.py checks that it does -- and what is "
        "left is 2.8% of peak rather than 100%"),
    "localize": (
        "delaying one ear lengthens the render, and the other channel is "
        "zero-padded to match. So one channel steps *into* the sound at "
        "the start -- zeros, then the sound's opening sample, which is "
        "the bottom of the wavetable a note opens at -- and the other "
        "steps *out* of it at the end, from wherever its last sample "
        "landed down to the padding. The second is the larger: 0.98 "
        "against 0.06, since the delayed channel is also attenuated"),
    "note_with_doppler": "the same padding, from the Doppler geometry",
    "note_with_vibrato_seq_localization": (
        "the same padding again. An earlier version of this entry said "
        "\"per segment\", which was a guess: there is one step per "
        "channel, at the two ends, exactly as above"),
    "tremolos": (
        "its two envelope groups are 12 and 16 seconds long, so the "
        "shorter one ends inside the render and the envelope steps back "
        "to unity where it does. RECONCILIATION.md records this routine "
        "as sample-exact with the MASS reference, which does the same"),
}


@functools.lru_cache(maxsize=None)
def _render(name):
    """The export's default result, as an array, or None if it is not one.

    Cached: five tests sweep the same forty-odd renders, two of which run
    to over a million samples, and rendering each once rather than five
    times is the difference between seven seconds and two minutes.
    """
    result = _callable_with_defaults(name)()
    try:
        array = np.asarray(result, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.dtype.kind != "f" or array.size < 512:
        return None
    return array


RENDERS = sorted(name for name in ZERO_ARG_EXPORTS
                 if _render(name) is not None)

#: Everything the sweep leaves out, and what it is instead.  A count on
#: its own would let a real render drop quietly out of the sweep -- six
#: could go before `>= 30` noticed -- so this names them, and the test
#: below asserts the two sets between them account for every export.
NOT_A_RENDER = {
    "band_pass": "filter coefficients",
    "band_reject": "filter coefficients",
    "high_pass": "filter coefficients",
    "low_pass": "filter coefficients, as a pair of arrays",
    "chord": "semitones",
    "harmonic_series": "ratios",
    "mode_by_rotation": "semitones",
    "pitch_to_freq": "frequencies",
    "scale": "semitones",
    "rhythm_to_durations": "durations in seconds",
    "proportional": "a bond, which is a function",
    "inversely_proportional": "a bond, which is a function",
}


def test_the_sweep_finds_the_renders():
    """Guard the sweep itself, the way test_public_api.py guards its own."""
    assert len(RENDERS) >= 30
    assert "note" in RENDERS and "silence" in RENDERS


def test_every_export_is_either_swept_or_named_as_not_a_render():
    """So nothing can leave the sweep without someone saying why.

    The sweep takes what comes back as a float array of at least 512
    samples, which is a rule about shape rather than about meaning: a
    routine that started returning something shorter, or an array of
    something else, would drop out of it in silence.
    """
    unaccounted = set(ZERO_ARG_EXPORTS) - set(RENDERS) - set(NOT_A_RENDER)
    assert not unaccounted, (
        f"{sorted(unaccounted)} are neither swept for artifacts nor "
        "registered as something other than a render. If one of them is a "
        "render, find out why the sweep stopped seeing it.")

    stale = set(NOT_A_RENDER) & set(RENDERS)
    assert not stale, (
        f"{sorted(stale)} are registered as not being renders and are "
        "being swept as renders; delete the entries.")


@pytest.mark.parametrize("name", sorted(set(CONTROL_SIGNALS)
                                        | set(DELIBERATE_STEPS)
                                        | set(UNMEASURABLE)))
def test_the_registers_name_things_that_are_still_rendered(name):
    """A register that names a routine nobody renders is a dead comment."""
    assert name in RENDERS, (
        f"{name} is registered in tests/test_artifacts.py but no longer "
        "renders on its own defaults. Delete the entry, or fix the sweep.")


@pytest.mark.parametrize("name", RENDERS)
def test_no_render_clicks_except_the_ones_registered(name):
    """The sweep. A step no one asked for is a defect in the rendering."""
    if name in DELIBERATE_STEPS or name in UNMEASURABLE:
        return
    found = clicks(_render(name))
    assert not found, (
        f"{name} steps {len(found)} time(s) where its own neighbourhood "
        f"does not explain it, the largest by {max(s for _i, s in found):.4f} "
        f"at sample {max(found, key=lambda f: f[1])[0]}. Either the "
        "rendering has a defect, or the step is deliberate and belongs in "
        "DELIBERATE_STEPS with the reason written down.")


@pytest.mark.parametrize("name", RENDERS)
def test_a_registered_step_is_still_there(name):
    """So the register cannot rot the other way, silently over-excusing."""
    if name not in DELIBERATE_STEPS:
        return
    assert clicks(_render(name)), (
        f"{name} is registered as stepping for a stated reason and no "
        "longer steps. If the reason stopped being true, delete the entry.")


@pytest.mark.parametrize("name", RENDERS)
def test_no_render_carries_a_dc_offset(name):
    """Except the envelopes, whose whole business is a nonzero mean."""
    if name in CONTROL_SIGNALS:
        return
    offset = dc_offset(_render(name))
    assert offset < DC_LIMIT, (
        f"{name} has a mean {offset:.3f} of its own RMS. That is headroom "
        "spent on nothing, and it is inaudible until something is mixed "
        "with it. If the routine returns a control signal rather than a "
        "sound, register it in CONTROL_SIGNALS.")


# ---------------------------------------------------------------------
# The seam: where a render is joined to the next one
# ---------------------------------------------------------------------

def _seam(first, second):
    """The step where two renders are stacked, and the largest step inside."""
    joined = music.horizontal_stack(first, second)
    at = len(first)
    steps = np.abs(np.diff(joined))
    inside = np.max(np.concatenate([steps[:at - 2], steps[at + 1:]]))
    return abs(float(joined[at] - joined[at - 1])), float(inside)


def test_a_note_opens_at_the_bottom_of_its_wavetable():
    """Which is where the joins come from, so it is worth pinning."""
    for freq in (220, 440, 443):
        opening = music.note(freq=freq, duration=0.25)[0]
        assert opening == pytest.approx(-1.0), (
            f"a {freq} Hz note opens at {opening}, not at the bottom of "
            "the table. tests/test_fidelity.py fixes the table's phase; "
            "this is what that means for a note placed after another one.")


@pytest.mark.parametrize("freq,whole_cycles", [
    (440.0, True),      # 440 * 0.25 = 110 cycles exactly
    (392.0, True),      # 98
    (443.0, False),     # 110.75
    (261.63, False),    # 65.4
    (555.5, False),     # 138.875
])
def test_raw_notes_step_where_they_join_unless_the_cycles_come_out_whole(
        freq, whole_cycles):
    """The artifact this file was written around.

    A note ends wherever its phase lands and the next one opens at the
    bottom of the table, so concatenating them steps -- by up to the full
    scale, against a largest step of 0.04 inside a note. Whether it happens
    at all depends on the frequency and the duration multiplying out to a
    whole number of cycles, which is why it is so easy to miss: 440 Hz for
    a quarter second is 110 cycles and joins cleanly, and 443 Hz for the
    same quarter second steps by 1.04.

    Nothing here is a defect. It is what a bare oscillator does, and the
    envelopes are how it is fixed -- see the test below. It is pinned so
    that it cannot change without someone deciding to change it.
    """
    step, inside = _seam(music.note(freq=freq, duration=0.25),
                         music.note(freq=freq * 1.5, duration=0.25))
    if whole_cycles:
        assert step <= inside, (
            f"{freq} Hz for a quarter second is a whole number of cycles, "
            "so the join should be no worse than the note's own steps")
    else:
        assert step > 5 * inside, (
            f"{freq} Hz for a quarter second lands mid-cycle, so the join "
            f"should step: {step:.4f} against {inside:.4f} inside")


def test_an_envelope_closes_the_seam():
    """`adsr` runs each note to silence at both ends, so the join is one."""
    for freq in (443.0, 261.63, 555.5):
        step, inside = _seam(
            music.adsr(sonic_vector=music.note(freq=freq, duration=0.25)),
            music.adsr(sonic_vector=music.note(freq=freq * 1.5,
                                               duration=0.25)))
        assert step < inside / 100, (
            f"an ADSR-shaped {freq} Hz note still joins with a step of "
            f"{step:.6f}, against {inside:.4f} inside the notes")


#: The first fenced python block in the README, which is the first code
#: anyone here runs.
_README_BLOCK = re.compile(r"```python\n(.*?)```", re.S)


def test_the_scale_the_readme_renders_does_not_click(tmp_path, monkeypatch):
    """The example a reader copies first, and it used to click nine times.

    Twelve joins, of which nine stepped by up to 1.88 out of a full scale
    of 2.0, because a chromatic scale from 440 Hz is whole cycles at the
    root and mid-cycle nearly everywhere else.

    This runs the README's own block rather than a copy of it, so
    dropping the `adsr` from the page fails here. An earlier version did
    keep a copy, and said in this docstring that reverting the README
    would fail it -- which it would not have, since the copy would have
    gone on passing on its own.
    """
    readme = (pathlib.Path(__file__).parent.parent / "README.md").read_text()
    block = _README_BLOCK.search(readme)
    assert block, "the README no longer opens with a python block"

    monkeypatch.chdir(tmp_path)          # the block writes scale.wav
    namespace: dict = {}
    exec(compile(block.group(1), "README.md[first block]", "exec"), namespace)

    assert "scale" in namespace, (
        "the README's block no longer binds `scale`; this test reads that "
        "name to check what the page renders")
    assert not clicks(music.horizontal_stack(*namespace["scale"]))


def test_the_melody_the_tutorial_renders_does_not_click(tmp_path,
                                                        monkeypatch):
    """The same, for the tutorial's first sequence of notes.

    `tests/test_tutorial.py` already runs every block on the page in one
    namespace; this borrows its extraction and checks what one of them
    renders rather than only that it ran.
    """
    from test_tutorial import BLOCKS

    monkeypatch.chdir(tmp_path)
    namespace: dict = {}
    for block in BLOCKS:
        if block.lstrip().startswith(">>>"):
            continue
        exec(compile(block, "tutorial.rst", "exec"), namespace)
        if "melody" in namespace:
            break

    assert "melody" in namespace, (
        "no block in the tutorial binds `melody` any more")
    assert not clicks(namespace["melody"])


# ---------------------------------------------------------------------
# What the sweep found
# ---------------------------------------------------------------------

def test_pan_transitions_moves_between_the_points_it_was_given():
    """The defect this file was written to find.

    `pp_` was assigned once before the loop and never updated, so every
    leg interpolated from `p[0]` rather than from the point the leg before
    it reached. On the defaults the third leg ran (1,1) -> (1,1) instead of
    (0,1) -> (1,1): a channel that should have risen from silence sat at
    full amplitude for two seconds, and the join before it stepped by the
    full scale. The docstring has always said what it should do -- "each
    pan transition i starts and ends amplitude envelope of channel c in
    p[i][c] and p[i+1][c]" -- so this asserts the docstring.
    """
    points = ((1, 1), (1, 0), (0, 1), (1, 1))
    seconds = 2
    rendered = np.asarray(music.pan_transitions(p=points,
                                                d=(seconds,) * 3), dtype=float)
    leg = seconds * 44100

    for channel in (0, 1):
        for index in range(3):
            segment = rendered[channel][index * leg:(index + 1) * leg]
            assert segment[0] == pytest.approx(points[index][channel]), (
                f"channel {channel}, leg {index + 1} starts at "
                f"{segment[0]:.3f} rather than at {points[index][channel]}")
            # The last sample is one step short of the destination, since
            # the ramp is over [0, 1) -- the next leg supplies the arrival.
            assert segment[-1] == pytest.approx(points[index + 1][channel],
                                                abs=2 / leg)

    assert not clicks(rendered)


def test_pan_transitions_ignores_the_method_it_is_given():
    """A known limitation, pinned so that fixing it is a decision.

    The signature offers 'lin', 'circ' and 'exp', the docstring explains
    what each would do, and the body interpolates linearly whatever it is
    given: `method` is never read. Implementing the three laws is a
    feature rather than a fix, so this records the state of things and
    fails when someone implements them, which is the point at which the
    docstring and ASSESSMENT.md need to change too.
    """
    rendered = [np.asarray(music.pan_transitions(p=((0, 1), (1, 0)), d=(1,),
                                                 method=(method,)),
                           dtype=float)
                for method in ("lin", "circ", "exp")]
    assert np.array_equal(rendered[0], rendered[1])
    assert np.array_equal(rendered[0], rendered[2])


# ---------------------------------------------------------------------
# Aliasing: partials that fold back down
# ---------------------------------------------------------------------

#: One second at 44.1 kHz.  At this length a partial at k Hz lands exactly
#: on bin k, so nothing leaks into its neighbours -- and the partials that
#: fold back land on bins too, which is what makes the two separable at
#: all.
SECOND = 44100


def stray_energy(samples, freq):
    """The share of the energy that is not at a multiple of `freq`.

    A wavetable is read at whatever rate the frequency asks for and
    nothing band-limits it, so a partial above the Nyquist frequency folds
    back down to `sample_rate - k * freq` and lands somewhere it does not
    belong.  This measures the energy that is not at a multiple of the
    fundamental, which is that plus anything else off the harmonics --
    the table read's own error, mostly. The sine row is what separates
    them: it carries no partial that can fold, and reads 1.2e-08, so
    everything above that in the rich tables is folding.

    The measure is blind whenever the sample rate is a whole multiple of
    the frequency, because then the folded partials land on multiples of
    `freq` as well and hide behind the harmonics they are being separated
    from.  100 Hz at 44.1 kHz is such a frequency and reads as 1e-31,
    which is not the same as clean; the test below keeps that written
    down.  `samples` must be `SECOND` long for the bins to line up.
    """
    assert len(samples) == SECOND, "the bins only line up at one second"
    energy = np.abs(np.fft.rfft(samples)) ** 2
    energy[0] = 0.0                     # a bias is not a partial
    bins = np.arange(energy.size)
    harmonic = (bins > 0) & (bins % freq == 0)
    return float(energy[~harmonic].sum() / energy.sum())


#: The tables by name, so a tone can be cached on one.
TABLES = {
    "sine": WAVEFORM_SINE,
    "triangular": WAVEFORM_TRIANGULAR,
    "sawtooth": WAVEFORM_SAWTOOTH,
    "square": WAVEFORM_SQUARE,
}


@functools.lru_cache(maxsize=None)
def _tone(freq, table):
    """A second of one table at one frequency, rendered once."""
    return music.note(freq=freq, number_of_samples=SECOND,
                      waveform_table=TABLES[table])


#: What each table strays by at 1, 5 and 10 kHz, measured.  A sine has one
#: partial and nothing to fold, so it is not here; the others carry
#: partials all the way up and every one above Nyquist comes back down.
#: This is what the synthesis method costs, not a defect in it: MASS
#: specifies a table read sample by sample, and a band-limited table is a
#: different instrument.
ALIASING = {
    "triangular": (1.55e-05, 2.30e-03, 1.45e-02),
    "sawtooth": (2.68e-02, 1.35e-01, 2.40e-01),
    "square": (1.83e-02, 9.93e-02, 1.89e-01),
}


#: What a sine strays, which is the table read and nothing else. Pinned
#: rather than merely bounded, because ASSESSMENT.md quotes it and a
#: figure a document quotes should have somewhere it comes from.
SINE_STRAY = 1.2e-08


@pytest.mark.parametrize("freq", [1000, 5000, 10000])
def test_a_sine_table_has_nothing_to_fold(freq):
    """One partial, so the only stray energy is the table read itself."""
    stray = stray_energy(_tone(freq, "sine"), freq)
    assert stray == pytest.approx(SINE_STRAY, rel=0.1), (
        f"a {freq} Hz sine strays {stray:.2e} of its energy off the "
        f"fundamental, where it strayed {SINE_STRAY:.1e}")


@pytest.mark.parametrize("table,strays", sorted(ALIASING.items()))
def test_a_rich_table_aliases_by_the_amount_it_always_has(table, strays):
    """Measured, so that a change to the synthesis has to move a number."""
    for freq, expected in zip((1000, 5000, 10000), strays):
        assert stray_energy(_tone(freq, table), freq) == pytest.approx(
            expected, rel=0.05)


@pytest.mark.parametrize("table", sorted(ALIASING))
def test_aliasing_grows_with_the_frequency(table):
    """The higher the note, the more of it comes back down in the wrong
    place: at 10 kHz a quarter of a sawtooth's energy is not at a harmonic
    of the note being played."""
    measured = [stray_energy(_tone(freq, table), freq)
                for freq in (1000, 5000, 10000)]
    assert measured[0] < measured[1] < measured[2]


def test_the_alias_measure_is_blind_at_some_frequencies():
    """Guard the measure, since a blind spot that nobody wrote down is
    worse than no measure: 44100 / 100 is a whole number, so every folded
    partial of a 100 Hz note lands on a multiple of 100 and cannot be told
    from a harmonic. A sawtooth there reads as clean as arithmetic allows,
    and it is not clean."""
    assert stray_energy(_tone(100, "sawtooth"), 100) < 1e-20
    assert stray_energy(_tone(1000, "sawtooth"), 1000) > 1e-2


# ---------------------------------------------------------------------
# What a write does to the level, and what it does at the rails
# ---------------------------------------------------------------------

@pytest.mark.parametrize("peak", [0.01, 1.0, 26.6])
def test_writing_normalizes_whatever_it_is_handed(tmp_path, peak):
    """A file's level is not the render's level.

    `write_wav_mono` runs `normalize_mono` over everything it is given, so
    a passage at a hundredth of full scale and one at twenty-six times it
    both arrive at exactly full scale. It is in the docstring; it is here
    because it is the sort of thing read once and not believed until a
    quiet passage comes back loud.
    """
    path = tmp_path / "level.wav"
    music.write_wav_mono(peak * music.note(440, 0.2), str(path))
    assert np.abs(music.read_wav(str(path))).max() == pytest.approx(
        1.0, abs=1e-4)


def test_two_passages_written_separately_lose_their_relative_level(tmp_path):
    """Which is what the normalization above costs, stated as a defect.

    A piece written a phrase at a time is a piece whose dynamics are gone.
    The way round it is to stack the phrases and write once, or to pass
    `remove_bias=False` and scale by hand -- neither of which the caller
    can know to do from a routine that just works.
    """
    loud, quiet = music.note(440, 0.2), 0.05 * music.note(440, 0.2)
    peaks = []
    for index, passage in enumerate((loud, quiet)):
        path = tmp_path / f"passage{index}.wav"
        music.write_wav_mono(passage, str(path))
        peaks.append(float(np.abs(music.read_wav(str(path))).max()))

    assert peaks[0] == pytest.approx(peaks[1], abs=1e-4), (
        "a passage at a twentieth of the other's level came back at a "
        "different level, which would mean the normalization documented "
        "above had stopped happening")

    together = music.horizontal_stack(loud, quiet)
    path = tmp_path / "together.wav"
    music.write_wav_mono(together, str(path))
    both = music.read_wav(str(path))
    half = len(loud)
    assert np.abs(both[half:]).max() < 0.1 * np.abs(both[:half]).max(), (
        "written in one pass, the quiet phrase should still be quiet")


@pytest.mark.parametrize("bit_depth", [8, 16, 24])
def test_the_quantizer_clips_rather_than_wraps(bit_depth):
    """The failure this would otherwise be is not a distortion.

    A sample past full scale that wraps comes back with its sign reversed
    -- the loudest possible sample becomes the quietest -- and what that
    sounds like is not a loud note but a detonation. `_quantize` clips,
    and since `write_wav_mono` normalizes first, nothing should ever reach
    it out of range anyway. This is the second of those two, tested
    because the first would hide it.
    """
    rail = 2 ** (bit_depth - 1)
    shift = 256 if bit_depth in (8, 24) else 1
    quantized = _quantize(np.array([1.5, -1.5, 1.0, -1.0, 0.0]), bit_depth)

    assert quantized[0] == (rail - 1) * shift
    assert quantized[1] == -rail * shift
    assert quantized[2] == (rail - 1) * shift
    assert quantized[3] == -rail * shift
    assert quantized[4] == 0
    assert np.sign(quantized[:4]).tolist() == [1, -1, 1, -1], (
        "a sample past full scale came back with its sign reversed, which "
        "is what wrapping does and what clipping exists to prevent")


# ---------------------------------------------------------------------
# Quantisation: what a bit depth is worth here
# ---------------------------------------------------------------------

#: Round-trip signal-to-noise, in dB, measured against the samples that
#: were actually written -- which is the normalized signal, not the one
#: handed in.
#:
#: The figure to compare these against is not 6.02b + 1.76. That assumes
#: a full-scale *sine*, and the default wavetable is triangular, whose
#: RMS is 1/sqrt(3) rather than 1/sqrt(2) -- 1.77 dB less signal for the
#: same peak. Against the triangular figure, 8-bit and 24-bit land within
#: a tenth of a decibel (48.2 and 144.5 predicted), and only 16-bit is an
#: outlier: it beats its own prediction by 25 dB because the source is
#: coarser than the format, which the test below is about.
ROUND_TRIP_SNR = {8: 48.1, 16: 121.1, 24: 144.6}


@pytest.mark.parametrize("bit_depth", sorted(ROUND_TRIP_SNR))
def test_the_round_trip_is_as_quiet_as_this_format_gets(tmp_path,
                                                        bit_depth):
    """Measured against what was written, so normalization is not counted
    as noise."""
    rendered = music.note(440, 1.0)
    written = normalize_mono(rendered, True)
    path = tmp_path / f"depth{bit_depth}.wav"
    music.write_wav_mono(rendered, str(path), bit_depth=bit_depth)
    back = music.read_wav(str(path))

    length = min(len(written), len(back))
    error = back[:length] - written[:length]
    snr = 10 * np.log10(np.sum(written[:length] ** 2)
                        / max(float(np.sum(error ** 2)), 1e-300))
    assert snr == pytest.approx(ROUND_TRIP_SNR[bit_depth], abs=1.0)


def test_a_rendered_note_is_coarser_than_the_file_it_is_written_to():
    """Thirteen bits, whatever the file says.

    The default wavetable holds 16,384 entries but only 8,193 distinct
    values, every one a multiple of 1/4096, and a note is a read out of it.
    So a note carries thirteen bits of amplitude resolution, a 16-bit file
    cannot lose anything it has -- which is why the round trip above
    measures 121 dB where this waveform at this depth predicts 96 -- and a
    24-bit file buys nothing at all. Anything that shapes a note afterwards, an
    envelope or a mix, leaves this behind; a bare note does not.
    """
    rendered = music.note(440, 0.2)
    assert np.allclose(rendered * 4096, np.round(rendered * 4096), atol=1e-12)
    assert not np.allclose(rendered * 2048, np.round(rendered * 2048),
                           atol=1e-12), "thirteen bits, and not twelve"

    shaped = music.adsr(sonic_vector=rendered)
    assert not np.allclose(shaped * 4096, np.round(shaped * 4096), atol=1e-12)


# ---------------------------------------------------------------------
# Mixing: sounds summed rather than joined end to end
# ---------------------------------------------------------------------

def test_mixing_a_render_with_itself_is_exactly_twice_it():
    """Sample-for-sample, with no phase error to cancel anything.

    It reads as a triviality and it is the thing that would break first if
    a render ever stopped being deterministic, or if `mix` padded at the
    wrong end.
    """
    rendered = music.note(440, 0.5)
    assert np.array_equal(music.mix(rendered, rendered), 2 * rendered)


def test_mixing_a_render_with_its_inverse_is_exact_silence():
    """The same statement from the other side, and a sharper one: any
    phase error at all would leave a residue here."""
    rendered = music.note(440, 0.5)
    assert not np.any(music.mix(rendered, -rendered))


def test_two_renders_of_the_same_note_are_the_same_samples():
    """Nothing in a note is drawn at random, so mixing cannot beat."""
    assert np.array_equal(music.note(440, 0.3), music.note(440, 0.3))


@pytest.mark.parametrize("offset,samples", [
    (0.25, 11025),              # a whole number of samples
    (1 / 440, 100),             # 100.227 samples, and the fraction is lost
    (0.5 / 44100, 0),           # half a sample, and all of it is lost
])
def test_an_offset_is_truncated_to_whole_samples(offset, samples):
    """A sub-sample offset is discarded rather than rounded or resampled.

    `mix_with_offset` takes seconds and delays by `int(seconds * rate)`, so
    an offset of half a sample is an offset of none. That is the honest
    thing for a routine that only indexes, but it means the comb filtering
    and the fractional delays that live below one sample cannot be asked
    for this way -- and a caller sweeping an offset finely gets a staircase
    rather than a sweep, which is audible as a stepped rather than a smooth
    effect. Pinned rather than fixed: rounding would be a different
    routine, and resampling a much larger one.
    """
    rendered = music.note(440, 0.5)
    mixed = music.mix_with_offset(rendered, rendered, duration=offset)
    assert len(mixed) == len(rendered) + samples


# ---------------------------------------------------------------------
# Filters: the artifact that is silent until it is not
# ---------------------------------------------------------------------

def poles(feedback):
    """The poles of a filter as `iir` means its coefficients.

    `iir` implements ``b0 y[n] = sum_k a_k x[n-k] + sum_{j>=1} b_j y[n-j]``
    -- note the plus on the feedback sum, which is not the usual
    convention -- so the characteristic polynomial is
    ``[b0, -b1, -b2, ...]`` and not `b` itself. Getting that backwards
    reads a stable filter as unstable, which is how this measure was
    written the first time.
    """
    feedback = np.atleast_1d(np.asarray(feedback, dtype=float))
    if feedback.size < 2:
        return np.zeros(1)          # no feedback, so nothing to diverge
    return np.roots(np.concatenate([[feedback[0]], -feedback[1:]]))


#: Cutoffs across the useful band, including both ends of it.  The designs
#: refuse 0 and 0.5 themselves, which `test_filter_design.py` covers.
CUTOFFS = [1e-6, 1e-4, 0.001, 0.01, 0.1, 0.25, 0.4, 0.49, 0.4999]


@pytest.mark.parametrize("design", ["low_pass", "high_pass"])
@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_a_one_pole_design_is_stable_at_any_cutoff(design, cutoff):
    """A pole on or outside the unit circle is a filter that does not
    settle: it rings for ever, or it grows without bound and takes the
    render with it. Nothing about the coefficients says so on inspection,
    and nothing in the sound says so until it does. The narrowest cutoff
    here puts the pole at 0.999994, which is as close as this design comes
    and still inside."""
    _feedforward, feedback = getattr(music, design)(cutoff)
    assert np.abs(poles(feedback)).max() < 1.0


@pytest.mark.parametrize("design", ["band_pass", "band_reject"])
@pytest.mark.parametrize("centre", [0.001, 0.05, 0.25, 0.45, 0.499])
@pytest.mark.parametrize("bandwidth", [0.001, 0.01, 0.1, 0.4])
def test_a_two_pole_design_is_stable_anywhere_in_its_grid(design, centre,
                                                          bandwidth):
    """The two-pole designs put their poles at 1 - bandwidth, so the
    narrowest band is the closest call: 0.997 at a bandwidth of 0.001."""
    _feedforward, feedback = getattr(music, design)(centre, bandwidth)
    assert np.abs(poles(feedback)).max() < 1.0


@pytest.mark.parametrize("design,args", [
    ("low_pass", (0.1,)), ("low_pass", (0.001,)),
    ("high_pass", (0.1,)), ("high_pass", (0.49,)),
    ("band_pass", (0.1, 0.05)), ("band_reject", (0.4, 0.01)),
])
def test_a_designed_filter_settles_rather_than_ringing_on(design, args):
    """Stability says the ringing dies; this says how fast.

    Driven by an impulse, every one of these is below a millionth of its
    own peak within four thousand samples -- a tenth of a second, and less
    than the shortest note anyone writes.
    """
    feedforward, feedback = getattr(music, design)(*args)
    impulse = np.zeros(4096)
    impulse[0] = 1.0
    response = np.asarray(music.iir(impulse, feedforward, feedback),
                          dtype=float)

    assert np.isfinite(response).all()
    peak = np.abs(response).max()
    assert np.abs(response[-256:]).max() < 1e-6 * peak


# ---------------------------------------------------------------------
# Intermodulation: the partials a modulation puts where it was not asked
# ---------------------------------------------------------------------

def partials(samples, floor=0.02):
    """The frequencies carrying more than `floor` of the strongest one.

    In Hz, since `SECOND` samples put one bin at one hertz.
    """
    assert len(samples) == SECOND, "the bins only line up at one second"
    magnitude = np.abs(np.fft.rfft(samples))
    magnitude[0] = 0.0
    strongest = magnitude.max()
    return sorted(int(bin_) for bin_
                  in np.flatnonzero(magnitude > floor * strongest))


def test_amplitude_modulation_puts_its_sidebands_where_it_should():
    """A carrier at 5 kHz modulated at 300 Hz is 4700, 5000 and 5300, and
    nothing else. Anything else would be intermodulation the model does
    not predict."""
    modulated = music.amplitude_modulation(
        carrier_freq=5000, modulation_freq=300, duration=1.0,
        waveform_table=TABLES["sine"])
    assert partials(modulated) == [4700, 5000, 5300]


def test_frequency_modulation_puts_its_sidebands_where_it_should():
    """FM spreads into a comb at the carrier plus and minus multiples of
    the modulator, with Bessel amplitudes. Every partial should be one of
    those and none should be anywhere else."""
    modulated = music.note_with_fm(
        freq=5000, number_of_samples=SECOND, fm=300, max_fm_deviation=300,
        waveform_table=TABLES["sine"], fm_waveform_table=TABLES["sine"])
    found = set(partials(modulated))

    # Both halves, because a subset alone would pass on an empty set --
    # which is what this assertion said before, and a render that came
    # back silent would have satisfied it.
    assert {4700, 5000, 5300} <= found, (
        "the carrier and its first sidebands should all be here")
    assert found <= {5000 + k * 300 for k in range(-4, 5)}, (
        "and nothing should be outside the comb")


def test_frequency_modulation_folds_when_its_sidebands_pass_nyquist():
    """The measured half of the same thing.

    A carrier at 18 kHz swung by 8 kHz puts sidebands well past 22,050 Hz,
    and they come back down: partials appear below 10 kHz, which is the
    lowest frequency the modulation could legitimately reach. It is the
    aliasing measured further up, arriving by a different route, and it is
    what a caller gets for asking for a bright sound near the top of the
    band.
    """
    modulated = music.note_with_fm(
        freq=18000, number_of_samples=SECOND, fm=2000, max_fm_deviation=8000,
        waveform_table=TABLES["sine"], fm_waveform_table=TABLES["sine"])
    lowest_legitimate = 18000 - 8000
    assert [f for f in partials(modulated) if f < lowest_legitimate]


def test_a_glissando_through_nyquist_folds_rather_than_breaking():
    """Sweeping to 40 kHz at a 44.1 kHz rate asks for what cannot exist.

    What comes out stays finite and inside the band -- the sweep folds at
    the top and comes back down -- rather than producing infinities or a
    silent render. Nobody should write this; it is here because the
    routine accepts it and the result should at least be a sound.
    """
    swept = music.note_with_glissando(start_freq=10000, end_freq=40000,
                                      number_of_samples=SECOND,
                                      waveform_table=TABLES["sine"])
    assert np.isfinite(swept).all()
    assert np.abs(swept).max() <= 1.0

    second_half = swept[SECOND // 2:]
    spectrum = np.abs(np.fft.rfft(second_half))
    spectrum[0] = 0.0
    freqs = np.fft.rfftfreq(len(second_half), 1 / SECOND)
    strongest = freqs[spectrum.argmax()]
    assert strongest < SECOND / 2


# ---------------------------------------------------------------------
# The rail: where two's complement is one code short
# ---------------------------------------------------------------------

@pytest.mark.parametrize("short_by,cost", [(0.4, 0.6), (0.01, 0.99)])
def test_the_positive_rail_is_one_code_short(short_by, cost):
    """A sample near full scale rounds to a code that does not exist.

    A PCM integer runs from -2**(b-1) to 2**(b-1) - 1, one code shorter on
    the positive side, so a sample above (rail - 0.5) / rail rounds up to
    a code there is no room for and is clipped to the one below. What that
    costs runs up to a whole least significant bit, at a sample just under
    full scale. At eight bits the affected band is the top 0.4% of the
    range and a note lands in it; at sixteen it is the top 0.0015% and
    nothing does.

    What it does *not* explain is the 8-bit round-trip figure, which an
    earlier version of this file claimed. Zeroing every rail-clipped error
    in a one-second note moves that measurement by 0.06 dB. The 1.8 dB
    that separates it from 6.02b + 1.76 is the waveform: see
    `ROUND_TRIP_SNR` above.
    """
    rail = 128
    near_full_scale = (rail - short_by) / rail
    quantized = _quantize(np.array([near_full_scale]), 8)
    assert quantized[0] == (rail - 1) * 256   # clipped to 127, shifted

    lost = abs(near_full_scale - (rail - 1) / rail) * rail
    assert lost == pytest.approx(cost, abs=0.01)

    # The same sample at sixteen bits is nowhere near the rail.
    assert _quantize(np.array([near_full_scale]), 16)[0] == round(
        near_full_scale * 32768)


# ---------------------------------------------------------------------
# What the measures can and cannot see
# ---------------------------------------------------------------------

#: The smallest planted step `clicks` finds, as a fraction of the render's
#: peak, in a few contexts.  Measured by bisection, and the point of the
#: table is the last row.
CLICK_SENSITIVITY = {
    "silence": 0.0,
    "note at 110 Hz": 0.090,
    "note at 440 Hz": 0.358,
    "note at 4000 Hz": None,        # nothing inside the representable range
}


def _smallest_step_found(signal, at=10000):
    """Bisect for the smallest step at `at` that `clicks` reports."""
    peak = float(np.max(np.abs(signal))) or 1.0
    low, high = 0.0, 4.0
    for _ in range(40):
        middle = (low + high) / 2
        hurt = np.array(signal, dtype=float)
        hurt[at:] += middle
        low, high = (low, middle) if clicks(hurt) else (middle, high)
    return high / peak


@pytest.mark.parametrize("label,expected", sorted(CLICK_SENSITIVITY.items()))
def test_what_the_click_measure_can_and_cannot_see(label, expected):
    """The measure is relative, and this is what that costs.

    A step counts when it stands `CLICK_RATIO` times over the median step
    of its block, so how large a click has to be depends entirely on how
    fast the signal around it is already moving. In silence any step is
    found. In a 110 Hz note it takes 9% of full scale. In a 440 Hz note,
    36%. At 4 kHz a sine advances about 0.57 per sample, so eight times
    that is 4.5 -- past the ±1 a sample can hold -- and **no click is
    detectable at all**, whatever the threshold, because no threshold
    makes a relative measure work where the signal's own steps are the
    largest steps available.

    So the sweep passing means no render carries a seam-like or gate-like
    step, not that no render clicks. It found what it was built to find --
    a full-scale step at a join, a delay opening out of zeros -- and the
    README's scale, whose joins stood 47 times over their neighbours. A
    click buried in a bright, fast passage it would not see.

    Nothing above tested the measure against a click anyone planted, which
    is how this went unwritten: every check was that clicks are *absent*,
    and absence is what a blind measure reports too.
    """
    signals = {
        "silence": music.silence(0.5),
        "note at 110 Hz": music.note(110, 0.5),
        "note at 440 Hz": music.note(440, 0.5),
        "note at 4000 Hz": music.note(4000, 0.5),
    }
    found = _smallest_step_found(signals[label])
    if expected is None:
        assert found > 1.0, (
            f"{label}: a step of {found:.2f} of peak was detected, so the "
            "measure is no longer blind here and this row should say so")
    else:
        assert found == pytest.approx(expected, abs=0.02)


def test_the_dc_measure_flags_an_offset_worth_flagging():
    """And what it lets past: below about 3% of peak, nothing is said.

    A tone biased by 2% of its peak passes the sweep. That is a real
    offset and it is not nothing -- it is a fortieth of the headroom --
    but it is well inside what an unlucky mean over a short render can
    produce on its own, which is what `DC_LIMIT` is set against.
    """
    tone = music.note(440, 0.5)
    assert dc_offset(tone + 0.02) < DC_LIMIT
    assert dc_offset(tone + 0.05) > DC_LIMIT


def test_the_alias_measure_reads_a_spectrum_built_to_order():
    """A positive control, which is what the click measure lacked.

    Every use of `stray_energy` above asks whether a render is clean, and
    a measure that answered "clean" to everything would satisfy all of
    them. So: a pure tone at a bin frequency, then the same tone with a
    partial of known energy planted somewhere that is not a multiple of
    the fundamental. What comes back should be the fraction planted.
    """
    samples = np.arange(SECOND)
    fundamental = np.sin(2 * np.pi * 1000 * samples / SECOND)
    assert stray_energy(fundamental, 1000) < 1e-20

    for fraction in (0.5, 0.1, 0.01):
        # An amplitude a carries energy a**2, so a**2 / (1 + a**2) of the
        # total is the planted partial's share.
        amplitude = np.sqrt(fraction / (1 - fraction))
        planted = fundamental + amplitude * np.sin(
            2 * np.pi * 1300 * samples / SECOND)
        assert stray_energy(planted, 1000) == pytest.approx(fraction,
                                                            abs=1e-6)


def test_the_partial_finder_reads_a_signal_built_to_order():
    """The same for `partials`, which the modulation tests lean on."""
    samples = np.arange(SECOND)
    built = (np.sin(2 * np.pi * 700 * samples / SECOND)
             + 0.5 * np.sin(2 * np.pi * 2100 * samples / SECOND)
             + 0.1 * np.sin(2 * np.pi * 5000 * samples / SECOND))

    assert partials(built) == [700, 2100, 5000]
    # And the floor is a floor: raised above the quietest, it drops it.
    assert partials(built, floor=0.2) == [700, 2100]


def test_the_pole_measure_reads_a_filter_built_to_order():
    """The positive control `poles` never had.

    Every use of it above asserts a design is stable, which a measure
    that reported everything as stable would satisfy. `iir` puts a plus
    on its feedback sum, so `b = [1, r]` is a single pole at exactly `r`
    -- a fact `test_filters_response.py` already leans on, since it
    checks that such a filter's impulse response is `r ** n`.
    """
    for radius in (0.0, 0.5, 0.99, 1.0, 1.05):
        assert float(np.abs(poles([1.0, radius])).max()) == pytest.approx(
            radius, abs=1e-12)

    # And the one that matters: a pole outside the circle diverges, which
    # is the thing being ruled out for every design in this file.
    impulse = np.zeros(600)
    impulse[0] = 1.0
    runaway = np.asarray(music.iir(impulse, [1.0], [1.0, 1.05]), dtype=float)
    assert np.abs(runaway[-1]) > 1e6
    settled = np.asarray(music.iir(impulse, [1.0], [1.0, 0.99]), dtype=float)
    assert np.abs(settled[-1]) < 1.0


#: Where zeroing one parameter renders a mean far from zero, and why.
#:
#: All seven are the same thing: an oscillator asked for no frequency
#: never advances through its table, so it holds whatever value the table
#: starts at. For a bare note that is the bottom of it -- a constant -1,
#: which is a DC offset at full scale, the worst there is.
#:
#: It is arithmetic rather than a defect, and it is here because the DC
#: sweep looks only at what a routine renders on its defaults, so this
#: whole class of input went unexamined until someone asked.
DC_WHEN_ZEROED = {
    ("frequency_modulation", "carrier_freq"),
    ("note", "freq"),
    ("note_with_doppler", "freq"),
    ("note_with_fm", "freq"),
    ("note_with_phase", "freq"),
    ("note_with_two_vibratos", "freq"),
    ("note_with_vibrato", "freq"),
}
