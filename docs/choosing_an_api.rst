Choosing the smallest API
=========================

Every public name is importable from ``music`` itself, which is a long
list, and most of it is variations. Start with the few routines below,
and reach for the specialized ones when a piece or an experiment needs
them.

.. list-table::
   :header-rows: 1
   :widths: 26 32 42

   * - Task
     - Start here
     - What to know
   * - Render a note
     - :func:`music.note`
     - The wavetable read as MASS reads it, sample for sample. Anything
       richer than a sine folds its partials above the Nyquist frequency
       back into the band, more the higher the note.
   * - Render a rich waveform at a high pitch
     - :func:`music.bandlimited_note`
     - The table with the partials the pitch cannot hold taken out, read
       with interpolation; a sine by default. For a pitch that holds
       still: a glissando or FM makes partials of its own.
   * - Join notes without clicks
     - :func:`music.adsr`
     - Shape each note before joining them: a note ends wherever its
       phase lands, and a bare join steps. A hard gate makes broadband
       energy of its own, whatever the carrier.
   * - Turn data into sound
     - :func:`music.pitch_to_freq`
     - Map a value to semitones rather than to hertz, so that equal
       differences are equal intervals wherever they fall.
       ``examples/sonification.py`` plays a temperature record three
       ways, and :doc:`listen` plays the first.
   * - Place notes in time
     - :class:`music.Sequencer`
     - Notes at start times, overlapping as they will, rather than
       stacked end to end by hand.
   * - Render a named sensory-stimulation technique
     - :mod:`music.stimulation`
     - Binaural, monaural and isochronic beats, amplitude and frequency
       modulation, modulated noise and spatial motion, each tested for
       the property that defines it. None is a claim about what it does
       to a listener.
   * - Describe a stimulus for another tool, or render one it described
     - :func:`music.stimulation.sstim.to_sstim_graph`,
       :func:`~music.stimulation.sstim.render_sstim`
     - SSTIM 0.20.0 descriptions, with ``pip install 'music[sstim]'``;
       what SSTIM has no term for is given, never guessed.
   * - Run a phase-structured stimulation protocol
     - :class:`music.StimulationSession`
     - Phases in order, crossfaded rather than cut, lasting exactly the
       sum of their durations, each with the parameters it was given.
   * - Sing a lyric
     - :func:`music.sing`
     - Needs espeak-ng and ``pip install 'music[singing]'``. The words
       do not always come through clearly.
   * - Read or write files
     - :func:`music.read_audio`, :func:`music.write_audio`
     - Writers normalize levels unless given ``normalize=False``; preserve
       mix relationships by writing once, or by not normalizing. Ask
       ``read_audio`` for a file's rate with ``return_sample_rate=True``.

Changing pitch, nonlinear effects and hard gates
------------------------------------------------

Use :func:`music.render_oversampled` with existing generators when
sample-rate conversion is part of the synthesis itself. It is opt-in and
requires ``pip install 'music[antialias]'``. This protects more general
modulated output than a static harmonic table, with increased memory and
rendering cost: 2x already removes FM's folding, while a hard gate or a
waveshaper needs a higher factor or ramps. For a rich wave whose pitch
moves, the mipmapped tables in ``music.core.synths.mipmap`` fold less
than 1e-8 of its energy, at a fixed cost per sample. :doc:`bandlimited_synthesis`
explains measurement and limits.

Machine-readable stimulus exchange
----------------------------------

The optional ``music.stimulation.sstim`` module writes the seven stimuli,
and sessions of them, as SSTIM 0.20.0 descriptions and renders them back;
it also renders a description another tool wrote, from SSTIM's terms and
what they cannot say. It is a separate module, rather than new names in
the flat API. See :doc:`sstim` for examples and validation.

Legacy classes
--------------

``music.legacy`` remains available for old code. Prefer functions in
``music.core`` and the focused ``music.stimulation`` package in
new projects. No previously exported name is removed or renamed here.

Spectral accuracy
-----------------

``music.note`` reads its table exactly as MASS does, and is the one to
use when comparing results with the article or its reference code.
``music.bandlimited_note`` is a separate decision: it filters a periodic
waveform's spectrum for the requested **fixed pitch**, then reads it
with linear interpolation. Near the Nyquist frequency a square or a
sawtooth approaches a sine, because only its lowest harmonics survive;
at a low pitch many more fit, so the table is filtered again for each
pitch, and cached.

This filters the *source waveform*. It does not prevent aliasing after
nonlinear effects, sampling-rate conversion, frequency modulation, or
isochronic gate discontinuities. Gibbs overshoot near sharp edges can
exceed full scale; apply gain/headroom before exporting. An optional
``rolloff_hz`` tapers harmonics below Nyquist instead of turning them
off abruptly as the pitch changes.

Measurements
------------

Run these without proprietary datasets or external service accounts:

.. code-block:: console

   python tools/benchmark_spectral_aliasing.py --json aliasing.json
   python tools/compare_stimulation_renderers.py --json stimulation.json

The first measures off-harmonic FFT energy and warmed render latency
against the MASS LUT. It uses coherent integer-frequency test tones,
not an inference from an arbitrary FFT peak. The second checks four
SSTIM-named stimulus techniques against independently written analytic
NumPy equations. The resulting JSON contains technique IRIs for
traceability, but is **not** an SSTIM RDF conformance claim.

For singing comparisons:

.. code-block:: console

   python tools/compare_singing.py --out singing-comparison
   python tools/prepare_singing_listening.py singing-comparison listening

Give the reviewer only ``listening/reviewer``. The backend key remains
in ``listening/organizer``. This prepares listening trials; it does
not supply listener observations, preference results, or intelligibility
measurements by itself.
