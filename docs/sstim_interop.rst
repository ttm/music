SSTIM stimulus interoperability and anti-aliasing
==================================================

The optional ``music.stimulation.sstim_io`` module exports **SSTIM 0.19.0**
RDF descriptions that can be consumed back into MUSIC. Its portable
semantic layer uses ``StimulusSpecification``, ``StimulationSignal``,
``StimulusChannel`` and ``SignalRendering``, with declared mechanism,
carrier, duration, frequency extent, and physical/perceptual presence.

The original five deterministic generators are supported here: ``binaural_beats``,
``monaural_beats``, ``isochronic_tones``,
``amplitude_modulation`` and ``frequency_modulation``. The
library's exact numeric arguments, including duty cycle and depth, are
stored in a **separate MUSIC extension namespace** so that portable
SSTIM terms are not falsely presented as an executable program.
A third-party renderer understands the SSTIM triples, but must also
understand these engine hints to reproduce MUSIC output sample-for-sample.

Install
-------

.. code-block:: console

   pip install 'music[sstim,antialias]'

Create, validate and render an SSTIM description:

.. code-block:: python

   from music.stimulation.sstim_io import (
       to_sstim_graph, render_sstim, validate_sstim,
   )

   graph = to_sstim_graph(
       "binaural_beats",
       parameters={"carrier_freq": 200, "beat_freq": 10},
       duration=30,
       sample_rate=48000,
       base="https://your-institution.example/stimuli/run-001/",
   )
   report = validate_sstim(graph, version="0.19.0")
   if not report.ok:
       raise ValueError(str(report))
   graph.serialize("stimulus.ttl", format="turtle")
   audio = render_sstim(graph)

This creates a **stimulus specification**, not a session execution
record, a claim of effectiveness, or a statement that the listener
actually received the audio. Use SSTIM's own ``sstim.Session`` for
executions with a clock and playback events.

If you import a Turtle description from another author, the engine
will refuse an unknown generator, unexpected types, conflicting
carrier/method assertions, or a non-portable rate. The input's
``music-impl:generator`` is a *closed enumeration*, not an import
path; parsing does not load remote URLs. It only renders a bounded
subset, not arbitrary SSTIM graphs.

Anti-alias changing modulation and gates
----------------------------------------

``music.bandlimited_note`` already removes out-of-band harmonics in
**static-pitch** periodic waveforms. For glissandi, FM, abrupt pulses or
nonlinear post-processing, filtering a low-rate WAV is too late: the
aliasing already folded into the audible band. Render at a higher rate
*before* those operations, then low-pass and decimate:

.. code-block:: python

   import music

   sound = music.render_oversampled(
       music.isochronic_tones, sample_rate=48000,
       duration=5, factor=4, carrier_freq=12000,
       pulse_rate=100, duty_cycle=0.5,
   )

The routine calls the original generator at 192 kHz and returns audio
at 48 kHz, using SciPy's polyphase resampling and Kaiser filtering.
It accepts mono or channels-first audio, guarantees the output
sample count, and does **not** normalize gain. The original generator
still behaves exactly as before, with no changed defaults.

Important limits:

* Oversampling is not an anti-alias guarantee for an arbitrarily fast FM
  or a waveform whose source was already sampled with folded energy.
* Sharp pulse edges are deliberately rounded by low-pass filtering.
  A truly instantaneous edge necessarily contains unbounded harmonics.
* Larger factors cost RAM and CPU, and convolution introduces short
  transient behavior at the boundaries. Benchmark against a higher-rate
  reference signal in the application's frequency range.
* For real-time applications, do not interpret this offline,
  array-based implementation as a low-latency streaming renderer.

Reproduce the DSP comparison:

.. code-block:: console

   python -m pytest tests/test_oversampling.py -q

The test compares direct 48 kHz rendering and 4x oversampled rendering
against a 16x high-rate reference on a gated 15 kHz carrier. The
criterion is a reduction in sample-domain reference error, not merely
a better-looking spectrogram or an unqualified perceptual claim.

These capabilities are independent. ``render_sstim(...,
oversampling_factor=4)`` combines them, but the SSTIM graph itself
does **not** claim that its carrier was rendered alias-free.

For stochastic noise, geometric motion and multi-phase programs, see
:doc:`advanced_sstim_dsp`.
