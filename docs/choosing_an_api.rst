Choosing the smallest API
=========================

The package retains its flat public namespace for compatibility. You do
not need to learn all of it. Start with the smallest set below and use
specialized routines when an experiment requires them.

.. list-table::
   :header-rows: 1
   :widths: 26 32 42

   * - Task
     - Start here
     - What to know
   * - Render a calibrated oscillator
     - :func:`music.note`
     - MASS-compatible periodic LUT; even a rich waveform may alias.
   * - Render a static-pitch, high-frequency tone
     - :func:`music.bandlimited_note`
     - Nyquist-filtered harmonic LUT with interpolation; not for FM or
       glissandi. Defaults to sine, preserving a smooth ordinary note.
   * - Control joins and envelope clicks
     - :func:`music.adsr`
     - Apply per note before concatenation. A hard gate creates its own
       broadband energy even with a band-limited carrier.
   * - Compose notes into a timeline
     - :class:`music.Sequencer`
     - Prefer scheduled notes to assembling timing by hand.
   * - Render a named sensory-stimulation technique
     - :mod:`music.stimulation`
     - Explicit binaural, monaural, isochronic, amplitude/FM, noise and
       spatial-motion routines; test stimulus properties independently of
       clinical claims.
   * - Describe a stimulus for another tool, or render one it described
     - :func:`music.stimulation.sstim.to_sstim_graph`,
       :func:`~music.stimulation.sstim.render_sstim`
     - SSTIM 0.19.0 descriptions, with ``pip install 'music[sstim]'``;
       what SSTIM has no term for is given, never guessed.
   * - Run a phase-structured stimulation protocol
     - :class:`music.StimulationSession`
     - Ordered phases, duration-preserving crossfades, reproducible params.
   * - Sing a lyric
     - :func:`music.sing`
     - Optional espeak-ng/PSOLA dependencies; intelligibility is variable.
   * - Read or write files
     - :func:`music.read_audio`, :func:`music.write_audio`
     - Writers normalize levels; preserve mix relationships by writing
       once rather than normalizing every phrase separately.

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
and sessions of them, as SSTIM 0.19.0 descriptions and renders them back;
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

``music.note`` intentionally follows MASS LUT lookup and should be used
when comparing exact numerical results to the reference paper/code.
``music.bandlimited_note`` is a separate DSP decision: it FFT-filters
a periodic waveform according to the requested **static fundamental**
and linearly interpolates sample positions. Near Nyquist a square or
sawtooth approaches a sinusoid because only its low harmonics survive.
At low pitches harmonics can still exceed Nyquist when their harmonic
number becomes large, so the table is recalculated by pitch and cached.

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
