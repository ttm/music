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

Three oscillators in ``music.core.synths`` take one instantaneous
frequency per output sample. None is exported at the top level.

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
than five million samples.

``mipmap`` reads tables of a sawtooth's, square's or triangle's first
partials, one table for every count from 1 to 16 and then four to an
octave up to 1024. At each sample it reads the two tables whose counts
bracket what fits below the Nyquist frequency at that pitch, and fades
from one to the other, so that no partial folds and none appears or
vanishes in a step:

.. code-block:: python

   import numpy as np
   from music.core.synths.mipmap import (
       MipmapOscillator, mipmap_frequency_path,
   )

   rate = 48000
   path = np.geomspace(55, 20000, rate)
   sweep = mipmap_frequency_path(path, sample_rate=rate, waveform="square")

   oscillator = MipmapOscillator(sample_rate=rate, waveform="square")
   blocks = [oscillator.render(path[start:start + 512])
             for start in range(0, rate, 512)]
   np.testing.assert_allclose(np.concatenate(blocks), sweep, atol=1e-8)

Its waveforms, and their phase, are ``bandlimited_frequency_path``'s, and
``MipmapOscillator`` keeps its phase between blocks as
``PolyBLEPOscillator`` does. It gives up part of the top of the band:
where more than 16 partials would fit, those above the larger table's
count, at most a sixth of them. Its reads are linear between entries, of
which a table has at least sixteen to a cycle of its top partial: next to
the edge of a wave of hundreds of partials that is off by up to 0.004,
and what it adds between the harmonics lies 80 dB below the wave. The
tables take 1.75 MiB a waveform, built the first time it is asked for.

At a fixed pitch, at 48 kHz, the three and a naive sawtooth compare like
this. *Folded* is the fraction of the energy off the harmonics; *off the
series* is how far the harmonics' magnitudes are from the Fourier
series', relative to it, which is what an oscillator did to the wave it
kept; the times are per second of sound, on one machine:

.. list-table::
   :header-rows: 1

   * - Sawtooth
     - Folded, 110 Hz
     - 1,762 Hz
     - 10,001 Hz
     - Off the series, 1,762 Hz
     - ms, 110 Hz
     - ms, 10,001 Hz
   * - Naive
     - 0.0028
     - 0.045
     - 0.24
     - 7e-8
     - 1.4
     - 2.0
   * - PolyBLEP
     - 6.8e-5
     - 0.0013
     - 0.0068
     - 0.084
     - 1.2
     - 2.2
   * - Mipmapped
     - 9.3e-9
     - 3.5e-12
     - 1.3e-14
     - 0.023
     - 2.6
     - 2.9
   * - Summed partials
     - 2e-25
     - 3e-24
     - 1e-23
     - 3e-15
     - 105
     - 1.2

The naive sawtooth keeps its harmonics nearly exact and folds the most;
PolyBLEP folds a little and softens the top harmonics; the mipmapped
tables fold less than 1e-8 of their energy at any pitch tried, and lose
part of the top; summing partials is exact and costs the more partials
fit. Along a sweep from 48
Hz to 21.6 kHz, the energy away from each moment's harmonics is 0.058
naive, 0.001 with PolyBLEP and 5e-8 with the tables.

None of the three bounds the sidebands that fast modulation makes, which
no count of instantaneous harmonics can, and none corrects a nonlinearity
applied afterwards. PolyBLEP corrects steps, not modulation, and a square
very near the Nyquist frequency is under-resolved even corrected.

In a stream
-----------

``PolyBLEPOscillator`` and ``MipmapOscillator`` render in blocks. Rendering
five seconds of a vibrato in blocks of 64 samples, which at 48 kHz last
1,333 µs each, the median block took 44 µs with PolyBLEP and 50 µs with
the tables, and the slowest of 3,750 took 111 and 171 µs; none overran.
Larger blocks cost less per sample: a block of 4,096, lasting 85 ms, took
0.14 and 0.25 ms. The blocks join to within 1e-8 of one render.

That is one machine, measured, not a promise. Each block allocates its own
arrays, and Python can pause for its collector at any time, so neither is
a real-time callback. Summed partials and ``render_oversampled`` render a
whole array at once and do not stream: the first keeps no state between
calls, and the second filters against the silence at its ends.

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
``resample_poly`` decimates it to 48 kHz through a Kaiser-window low-pass
that is flat to within 0.0001 dB up to 90 % of the Nyquist frequency and
about 100 dB down from that frequency up. ``resample_poly``'s own default
filter centres its transition on the Nyquist frequency instead, and passes
a line 5 % above it at -10.5 dB, to fold, whatever the factor.
Any generator that takes ``number_of_samples`` and ``sample_rate`` will
do, and its other arguments are passed through; it must return time on
the last axis, mono or channels first. The result has exactly the samples
asked for, at the level they were rendered: nothing is normalized. The
factor runs from 2 to 16, and the oversampled render is capped at fifty
million frames.

Filtering is the price. The top tenth of the band goes with the folding.
A hard gate's edges are rounded, since an instantaneous edge has partials
without end; what folds past the higher rate's own Nyquist frequency still
folds; and the first and last 64 or so samples are filtered against the
silence beyond the render. Memory and time grow with the factor, and most
of the time is the filter's: a second of mono sound at 48 kHz, which
takes 1 to 2 ms to render directly, took about 15 ms at 2x, 30 ms at 4x
and 120 ms at 16x on the machine these figures come from. FM needs only
2x, as the measurements below show; a hard gate or a waveshaper wants
more.
It renders an array, offline. Rendering an SSTIM description can be
oversampled the same way: see :doc:`sstim`.

Measuring it
------------

Five tools measure signals, without data or accounts from anywhere:

.. code-block:: console

   python tools/benchmark_spectral_aliasing.py --json static.json
   python tools/benchmark_oscillators.py --json oscillators.json
   python tools/benchmark_dynamic_aliasing.py --json dynamic.json
   python tools/attribute_aliasing.py --json attribution.json
   python tools/compare_stimulation_renderers.py --json stimulation.json

The first two render static pitches for exactly one second at whole
frequencies, so that every harmonic falls on a bin and the energy
anywhere else can be counted. The count is blind wherever a folded
partial lands on a harmonic. ``benchmark_oscillators.py`` gives the
figures above, at 44.1, 48 and 96 kHz: for a naive oscillator, PolyBLEP,
the mipmapped tables and summed partials, at five pitches, along a sweep
and in blocks, with the peak of Python's own allocations, which is not
the process's memory.

For one that moves, ``benchmark_dynamic_aliasing.py`` builds each signal
at sixteen times the rate from its closed form, filters it to the target
rate through ``render_oversampled``'s own filter, so that both mean the
same band, and compares that with the package's render, direct and at 4x
and 8x. Four signals -- FM of wide deviation, FM swinging past the Nyquist
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
     - 0.201
     - 0.00038
     - 0.00010
     - 1.15
     - 0.34
   * - FM past Nyquist
     - 0.768
     - 0.00039
     - 0.00010
     - 1.12
     - 0.33
   * - tanh waveshaper
     - 0.387
     - 0.037
     - 0.0024
     - 0.35
     - 0.033
   * - Abrupt gate
     - 0.181
     - 0.023
     - 0.0095
     - 0.083
     - 0.012

Read them with care. The spectral difference is not the folded power
alone; the next section measures that. The FM's large RMS is not folding
either. The package's ``frequency_modulation`` adds each sample's phase
step before reading it, so its carrier starts one sample into its cycle,
at whatever rate it renders: the RMS counts that head start in full, and
the spectrum ignores it. Rendered at 4x, the step is a quarter as long,
which is most of the 4x figure. The gate's high frequencies fall because
its edges are rounded, not because the gate arrives intact. And the
reference is finite: sixteen times the rate is an approximation, and a
demanding signal wants a check at 32x or 64x on a short render. The
record also states the closed form's own improvement separately, since it
is no measure of the package's.

Where the folding comes from
----------------------------

A spectral difference says how far a render is from a reference, not how
much of it folded. ``attribute_aliasing.py`` says that, line by line. Each
of the four signals, and the gate again with three-sample ramps, has a
line spectrum known in closed form: FM's sidebands are Bessel functions of
its index, a tanh waveshaper's harmonics a Fourier series, a gate's
sidebands a pulse train's. With every frequency a whole number of hertz,
one second holds a whole number of periods and its DFT puts each line on a
bin, so where a line above the Nyquist frequency lands is arithmetic. The
tool predicts a render's folded power, direct or through
``render_oversampled``'s filter, and reads the render at exactly those
bins. For the closed forms, measurement and prediction agree to 0.01 dB;
for the package's renders, to 0.2 dB direct.

At 48 kHz, the folded power in the package's renders, in dB relative to
the band-limited signal's, is:

.. list-table::
   :header-rows: 1

   * - Signal
     - Direct
     - 2x
     - 4x
     - 8x
     - 16x
     - Changed by the filter
   * - Wide FM
     - -19.2
     - -119
     - -123
     - -129
     - -127
     - -15.7
   * - FM past Nyquist
     - -2.2
     - -113
     - -112
     - -113
     - -114
     - -42.2
   * - tanh waveshaper
     - -8.2
     - -18.9
     - -28.6
     - -52.5
     - -84.6
     - -117
   * - Abrupt gate
     - -14.7
     - -23.8
     - -30.2
     - -36.3
     - -42.3
     - -22.6
   * - Gate, ramped
     - -20.8
     - -50.4
     - -64.3
     - -77.1
     - -89.1
     - -36.2

Two kinds of folding behave differently. FM's sidebands die away soon
beyond its deviation, so the ones that fold lie a little above the Nyquist
frequency, where the filter removes them. At a factor k, a line between
the Nyquist frequency and 2k - 1 times it is removed rather than folded,
and FM's lie well below three times it: 2x is as good as 16x. The FM sideband that
folds most, the sixth above the carrier at 24,582 Hz, lands at 23,418 Hz
at -19.9 dB direct and at -145 dB at 4x. What is left is not folding: the
closed form's render leaves -142 dB at 2x and -139 dB at 4x, and the rest
is the package's own table reads landing on the same bins.

A gate's lines and a waveshaper's go on far beyond any rate, and what
folds there folds at the higher rate itself, before the filter sees it.
Only a higher factor helps. An abrupt gate's lines fall as 1/k, so each
doubling of the factor takes 6 dB; ramps make them fall faster, and with
ramps of three samples each doubling takes 12 to 14 dB. A waveshaper's
fall away exponentially, so each doubling takes more than the last, from
-19 dB at 2x to -85 dB at 16x.

The last column is what the filter costs: the power by which it changes
the band-limited signal, mostly by cutting its lines above 90 % of the
Nyquist frequency. Wide FM loses a sideband at 23 kHz.

The package's FM leaves something else besides, at -50 dB at 48 kHz and a
deviation of 7.2 kHz. That is not folding, and oversampling does not take
it away. Its modulator is read from a table without interpolation, and
the table's steps modulate the carrier too. It falls by 20 dB for each
tenfold smaller deviation, down to the -79 dB that the carrier's own table
leaves.

``compare_stimulation_renderers.py`` checks four of the stimuli against
sinusoids written out in NumPy, which catches a wrong frequency, depth,
gate or channel layout but not an assumption both share. Its JSON names
each technique's SSTIM term, which is not a claim of conformance.

Which to use, and what stays the default
----------------------------------------

For a rich wave whose pitch moves, the mipmapped tables fold less than
1e-8 of their energy and keep about as much of the wave as PolyBLEP, for
up to twice its time and 1.75 MiB of tables. PolyBLEP needs no tables but
folds 0.1 % to 0.7 % of its energy from 1.8 to 10 kHz. Summing partials is exact, and
costs the more partials fit: for low notes, it is for rendering offline.
For what the rendering makes -- FM, gates, waveshaping --
``render_oversampled`` removes the folding of whatever ends below 2k - 1
times the Nyquist frequency at a factor k, which for FM is already 2x. A
hard gate or a waveshaper needs a higher factor, or ramps.

None of this changes a default. ``note`` reads its table as MASS does,
every generator renders as it did, and the routines on this page are asked
for by name. The figures above describe signals. Whether a listener hears
the differences between them is what the listening test in
`issue 125 <https://github.com/ttm/music/issues/125>`_ is for, and
changing a default for a difference nobody has heard would trade MASS's
samples for nothing shown.

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
claim about what a listener hears rests on them; `issue 125
<https://github.com/ttm/music/issues/125>`_ is for running one.

Reference
---------

:func:`music.bandlimited_note` and :func:`music.render_oversampled` are in
the :doc:`api`. The oscillators for a pitch that moves are not exported
at the top level, and are imported from their modules:

.. autofunction:: music.core.synths.frequency_path.bandlimited_frequency_path

.. autofunction:: music.core.synths.polyblep.polyblep_frequency_path

.. autoclass:: music.core.synths.polyblep.PolyBLEPOscillator
   :members: render

.. autofunction:: music.core.synths.mipmap.mipmap_frequency_path

.. autoclass:: music.core.synths.mipmap.MipmapOscillator
   :members: render
