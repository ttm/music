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

What the sweep found, and what came of it, is in ASSESSMENT.md.  The short
version: one defect in `pan_transitions`, which interpolated every leg of a
pan from the first point rather than from the previous one and so rendered
a full-scale step at a join; and a family of steps that are not defects,
each registered below with the reason it is there.
"""

import functools

import numpy as np
import pytest

import music
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
#: The renders that pass sit at 1 to 2; the registered ones start at 48.
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
        "the far ear is the near ear delayed, so that channel begins with "
        "zeros and then steps to the sound's opening sample. Every note "
        "opens at the bottom of its wavetable, which is where that "
        "opening sample is"),
    "note_with_doppler": "the same delay, from the Doppler geometry",
    "note_with_vibrato_seq_localization": "the same delay, per segment",
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


def test_the_sweep_finds_the_renders():
    """Guard the sweep itself, the way test_public_api.py guards its own."""
    assert len(RENDERS) >= 30
    assert "note" in RENDERS and "silence" in RENDERS


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


def test_the_scale_the_readme_renders_does_not_click():
    """The example a reader copies first, and it used to click nine times.

    Twelve joins, of which nine stepped by up to 1.88 out of a full scale
    of 2.0, because a chromatic scale from 440 Hz is whole cycles at the
    root and mid-cycle nearly everywhere else. The README now shapes each
    note before stacking them; this fails if that is reverted.
    """
    scale = [music.adsr(sonic_vector=music.note(440 * 2 ** (i / 12),
                                                duration=0.25))
             for i in range(13)]
    assert not clicks(music.horizontal_stack(*scale))


def test_the_melody_the_tutorial_renders_does_not_click():
    """The same, for the tutorial's first sequence of notes."""
    freqs = [261.63, 293.66, 329.63, 349.23, 392.0]
    melody = music.horizontal_stack(
        *[music.adsr(sonic_vector=music.note(f, 0.35)) for f in freqs])
    assert not clicks(melody)


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
