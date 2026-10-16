Band-limited synthesis
======================

:func:`music.note` reads its wavetable at every sample, as MASS specifies,
and nothing band-limits the read: every partial of the table above the
Nyquist frequency folds back into the band and lands where it does not
belong. A sine has nothing to fold. A rich table does, and at 10 kHz a
sawtooth puts 24 % of its energy between the harmonics of the note being
played.

That is what the method costs, and ``note`` keeps it, sample for sample.
The routines on this page are a different instrument, and have to be
asked for. Which one depends on what makes the folding: a table read at a
fixed pitch, a pitch that moves, or something the rendering itself does,
such as frequency modulation, a hard gate or a nonlinearity.

A fixed pitch
-------------

:func:`music.bandlimited_note` drops a table's partials at or above the
Nyquist frequency for the note's own pitch, then reads the table with
linear interpolation rather than truncating the position:

.. code-block:: python

   import music

   folded = music.note(10000, duration=1,
                       waveform_table=music.waveform_table("sawtooth"))
   clean = music.bandlimited_note(10000, duration=1, waveform="sawtooth")

The 10 kHz sawtooth that kept 24 % of its energy off its harmonics keeps
about 1e-16. The table is filtered for each pitch and cached, and the
waveform is one of ``sine``, ``sawtooth``, ``square`` and ``triangle``:
an arbitrary array has no pitch to filter it for. Near the Nyquist
frequency little survives, so a square or a sawtooth there approaches a
sine. ``rolloff_hz`` fades the last partials out over that many hertz
instead of cutting them, so that neighbouring pitches do not differ by a
whole partial.

A truncated harmonic series overshoots near a waveform's edges, so a
band-limited square or sawtooth can exceed full scale: leave headroom when
writing it. And the pitch has to hold still. A table filtered for one
pitch knows nothing of a glissando, FM, a gate or a nonlinearity applied
afterwards, which make partials of their own.

``python examples/aliasing_demo.py`` writes both sawtooths above at one
gain, to be heard side by side.

A pitch that moves
------------------

Two oscillators in ``music.core.synths`` take one instantaneous frequency
per output sample. Neither is exported at the top level.

``frequency_path.bandlimited_frequency_path`` sums a sine, sawtooth,
square or triangle partial by partial, recomputing at every sample how
much of each to keep, and fades each one to nothing as it nears the
Nyquist frequency:

.. code-block:: python

   import numpy as np
   from music.core.synths.frequency_path import bandlimited_frequency_path

   rate = 48000
   sweep = bandlimited_frequency_path(
       np.linspace(300, 12000, rate), sample_rate=rate,
       waveform="sawtooth", transition=.15, max_harmonics=256)

``transition`` is the fraction of the Nyquist frequency over which a
partial fades, so that one does not vanish with a click in the middle of
a glissando. The phase is integrated trapezoidally, as if the frequency
moved linearly between samples, and a linear sweep ends where its closed
form does. ``max_harmonics``, from 1 to 2048, caps the work, so a low
note loses its partials above that number.

``polyblep`` corrects only a sawtooth's or a square's edges, with a short
polynomial step around each one, at a cost per sample that does not grow
with the number of partials:

.. code-block:: python

   import numpy as np
   from music.core.synths.polyblep import (
       PolyBLEPOscillator, polyblep_frequency_path,
   )

   rate = 48000
   frequency = np.full(rate, 10001.)
   audio = polyblep_frequency_path(
       frequency, sample_rate=rate, waveform="sawtooth")

   oscillator = PolyBLEPOscillator(sample_rate=rate, waveform="sawtooth")
   first = oscillator.render(frequency[:500])
   rest = oscillator.render(frequency[500:])
   np.testing.assert_allclose(np.concatenate((first, rest)), audio,
                              atol=1e-8)

At 48 kHz and 10,001 Hz, PolyBLEP leaves 0.68 % of a sawtooth's energy
off its harmonics where a naive one leaves 24 %, and 0.73 % of a square's
against 19 %. Summing partials leaves nothing measurable, and at a pitch
that high, with at most two partials below Nyquist, it is the cheaper of
the two; PolyBLEP's cost is the same at every pitch, where summing
partials costs more the lower the note.

``PolyBLEPOscillator`` keeps its phase and its last frequency between
calls, so a long render can be made in blocks that join, to within 1e-8,
as one render would; an empty block does not move it. A block holds fewer
than five million samples. This is rendering in pieces, offline, not a
real-time callback: each block allocates its own arrays.

Neither oscillator bounds the sidebands that fast modulation makes, which
no count of instantaneous harmonics can, and neither corrects a
nonlinearity applied afterwards. PolyBLEP corrects steps, not modulation,
and a square very near the Nyquist frequency is under-resolved even
corrected.

Whatever the rendering does
---------------------------

:func:`music.render_oversampled` renders a whole generator at a multiple
of the rate, low-passes it and decimates it, for the folding no table can
prevent: frequency modulation, glissandi, hard gates and waveshaping,
whose partials are made as the sound is rendered. It needs SciPy:

.. code-block:: console

   pip install 'music[antialias]'

.. code-block:: python

   import music

   sound = music.render_oversampled(
       music.isochronic_tones, sample_rate=48000, duration=5, factor=4,
       carrier_freq=12000, pulse_rate=100, duty_cycle=0.5)

The generator is called once, here at 192 kHz, and SciPy's
``resample_poly`` filters with a Kaiser window of beta 8.6 and decimates to
48 kHz.
Any generator that takes ``number_of_samples`` and ``sample_rate`` will
do, and its other arguments are passed through; it must return time on
the last axis, mono or channels first. The result has exactly the samples
asked for, at the level they were rendered: nothing is normalized. The
factor runs from 2 to 16, and the oversampled render is capped at fifty
million frames.

Filtering is the price. A hard gate's edges are rounded along with the
folding, since an instantaneous edge has partials without end; what
folds past the higher rate's own Nyquist frequency still folds; memory
and time grow with the factor; and the filter rings briefly at the ends.
It renders an array, offline. Rendering an SSTIM description can be
oversampled the same way: see :doc:`sstim`.

Measuring it
------------

Four tools measure signals, without data or accounts from anywhere:

.. code-block:: console

   python tools/benchmark_spectral_aliasing.py --json static.json
   python tools/benchmark_polyblep.py --json polyblep.json
   python tools/benchmark_dynamic_aliasing.py --json dynamic.json
   python tools/compare_stimulation_renderers.py --json stimulation.json

The first two render static pitches for exactly one second at whole
frequencies, so that every harmonic falls on a bin and the energy
anywhere else can be counted; they report that fraction and the render
time, and the second compares a naive oscillator, PolyBLEP and summed
partials at 44.1, 48 and 96 kHz, with the peak of Python's own
allocations, which is not the process's memory. The count is blind
wherever a folded partial lands on a harmonic, and it says nothing about
a pitch that moves.

For one that moves, ``benchmark_dynamic_aliasing.py`` builds each signal
at sixteen times the rate from its closed form, filters it to the target
rate, and compares that with the package's render, direct and at 4x and
8x. Four signals -- FM of wide deviation, FM swinging past the Nyquist
frequency, a tanh waveshaper and an abruptly gated carrier -- at 44.1, 48
and 96 kHz. By default their frequencies are fractions of the target
rate, so every rate is pressed as hard; ``--fixed-reference-rate 48000``
holds them at the frequencies defined for 48 kHz instead, which measures
how much more room the higher rates leave. CI publishes both matrices.

Each record gives the RMS difference over the whole render and over its
worst eighth, so that a short burst is not averaged away; a difference of
windowed spectral magnitudes, which ignores phase; the median render
time; and the size of the intermediate buffer, which is not a measure of
peak memory. At 48 kHz, the package's renders compare like this:

.. list-table::
   :header-rows: 1

   * - Signal
     - Spectral, direct
     - 4x
     - 8x
     - RMS, direct
     - 4x
   * - Wide FM
     - 0.119
     - 0.0026
     - 0.0013
     - 1.15
     - 0.34
   * - FM past Nyquist
     - 0.635
     - 0.0136
     - 0.0072
     - 1.14
     - 0.33
   * - tanh waveshaper
     - 0.335
     - 0.0157
     - 0.0092
     - 1.29
     - 1.26
   * - Abrupt gate
     - 0.162
     - 0.0230
     - 0.0095
     - 0.075
     - 0.012

Read them with care. The spectral difference is not the folded power
alone, and the waveshaper's shows it: its spectrum comes much closer while
its waveform does not, because the package's oscillator and the closed
form disagree in phase, which the RMS counts and the spectrum ignores.
The FM's large direct RMS is the same disagreement, not folding. The
gate's high frequencies fall because its edges are rounded, not because
the gate arrives intact. And the reference is finite: sixteen times the
rate is an approximation, and a demanding signal wants a check at 32x or
64x on a short render. The record also states the closed form's own
improvement separately, since it is no measure of the package's.

``compare_stimulation_renderers.py`` checks four of the stimuli against
sinusoids written out in NumPy, which catches a wrong frequency, depth,
gate or channel layout but not an assumption both share. Its JSON names
each technique's SSTIM term, which is not a claim of conformance.

Listening
---------

None of the figures above says how anything sounds. Two tools prepare a
listening test and run none:

.. code-block:: console

   python tools/prepare_polyblep_listening.py --out blep-listening
   python tools/compare_singing.py --out singing-comparison
   python tools/prepare_singing_listening.py singing-comparison listening

Each writes pairs matched in RMS level, in random order, and a blank
form. Give a listener only the ``reviewer`` folder; the ``organizer``
folder holds the key. Start at a low volume, since a rich wave near the
Nyquist frequency can be harsh. No test has been run with them, so no
claim about what a listener hears rests on them.

Reference
---------

:func:`music.bandlimited_note` and :func:`music.render_oversampled` are in
the :doc:`api`. The oscillators for a pitch that moves are not exported
at the top level, and are imported from their modules:

.. autofunction:: music.core.synths.frequency_path.bandlimited_frequency_path

.. autofunction:: music.core.synths.polyblep.polyblep_frequency_path

.. autoclass:: music.core.synths.polyblep.PolyBLEPOscillator
   :members: render
