Stimuli in SSTIM
================

`SSTIM <https://w3id.org/sstim>`_, the Sensory Stimulation Vocabulary,
describes a stimulus in RDF: its signals, the channels it is presented
on, and how each signal is rendered onto each channel. Each of the
package's seven auditory stimuli, and a session of them, can be written
as an SSTIM 0.19.0 description and rendered back from one, which needs:

.. code-block:: console

   pip install 'music[sstim]'

A description says what a stimulus is. It is not a record that anyone
heard it, nor a claim that it does anything, and SSTIM fixes no
oscillator, phase, gain or random generator, so another engine can
render a valid description differently and still be right. What the
package can promise, it promises for its own renders.

Writing a stimulus, and rendering it back
-----------------------------------------

``music.stimulation.sstim_io`` writes the five deterministic stimuli --
binaural, monaural and isochronic beats, amplitude and frequency
modulation:

.. code-block:: python

   from music.stimulation.sstim_io import (
       render_sstim, to_sstim_graph, validate_sstim,
   )

   graph = to_sstim_graph(
       "binaural_beats",
       parameters={"carrier_freq": 200, "beat_freq": 10},
       duration=30,
       sample_rate=48000,
       base="https://your-lab.example/stimuli/run-001/",
   )
   report = validate_sstim(graph, version="0.19.0")
   if not report.ok:
       raise ValueError(str(report))
   graph.serialize("stimulus.ttl", format="turtle")
   audio = render_sstim("stimulus.ttl")

The description is a ``StimulusSpecification`` holding the signals,
channels and renderings, in SSTIM's ``StimulationSignal``,
``StimulusChannel`` and ``SignalRendering``: the mechanism, the carrier,
the duration, the frequencies, and whether the beat is physically in the
signal or made by the listener. The arguments SSTIM has no terms for, a
duty cycle or a modulation depth, go in the package's own namespace,
``https://github.com/ttm/music/terms#``, rather than being minted as
SSTIM's. Those, with the package's algorithms, are what make the render
exact: ``render_sstim`` renders a
description as the call it records would, sample for sample, and with
``oversampling_factor=4`` renders it through
:func:`music.render_oversampled` (see :doc:`bandlimited_synthesis`); the
description itself says nothing about how it was rendered.

``validate_sstim`` checks a description with SSTIM's own validator, in
its Full profile, which the ``sstim`` package fetches the first time and
caches. It checks the description, not the samples.

Noise and spatial motion
------------------------

``sstim_advanced`` writes the other two. A noise is described as a
stochastic band of a named colour, not a carrier, and its seed is
recorded, so the same description renders the same noise; without one,
each render draws anew.

.. code-block:: python

   from music.stimulation.sstim_advanced import (
       render_sstim_advanced, to_sstim_advanced_graph,
   )

   noise = to_sstim_advanced_graph(
       "modulated_noise",
       parameters={"noise_type": "pink", "min_freq": 100,
                   "max_freq": 10000, "modulation_freq": 10, "seed": 42},
       duration=15,
       sample_rate=48000,
       base="https://your-lab.example/stimuli/noise-1/",
   )
   audio = render_sstim_advanced(noise)

Spatial motion is described as a tone and, separately, a signal of
position, both presented on the two ears:

.. code-block:: python

   motion = to_sstim_advanced_graph(
       "spatial_motion",
       parameters={"carrier_freq": 440, "motion_rate": 0.5,
                   "theta1": 180, "theta2": 0, "dist": 0.1},
       duration=12,
       sample_rate=48000,
       base="https://your-lab.example/stimuli/motion-1/",
   )

The motion is the package's geometric one, from interaural time and
intensity differences, not a measured head: it carries no elevation and
cannot tell front from back. Only the stimuli's own sine modulator and
tone are written; a custom waveform table is not.

Sessions
--------

``sstim_program`` writes a :class:`music.StimulationSession`, its phases
in order with their ramps and gains, each phase an SSTIM description of
its own, and renders it back as the session would:

.. code-block:: python

   import music
   from music.stimulation.sstim_program import (
       render_sstim_program, to_sstim_program_graph,
   )

   plan = music.StimulationSession(sample_rate=48000, end_ramp=.05)
   plan.add(music.binaural_beats, duration=10, beat_freq=10)
   plan.add(music.modulated_noise, duration=10, ramp=.05,
            noise_type="pink", seed=42)
   graph = to_sstim_program_graph(
       plan, base="https://your-lab.example/plans/001/")
   graph.serialize("program.ttl", format="turtle")
   replay = render_sstim_program("program.ttl")

A session has no clock, no listener and no record of playback, so it is
written as a plan, in the package's terms: ``StimulationProgram`` and its
phases are the package's, and only the stimuli in them are SSTIM's. It is
never an ``sstim:SessionInstance``, which would claim a delivery; the
``sstim`` package's own ``Session`` writes one, from what happened.
Validation checks the stimuli, not the plan.

Any of the seven stimuli can be a phase, with portable arguments: a
session holding a rendered array, another callable or a custom table is
refused, and so is one of more than fifty phases or five million samples.
A noise phase without a seed stays random.

A plan of a whole number of seconds, from a minute to two hours, can also
be written as SSTIM's own ``SessionSpecification``, which SSTIM's Full
profile accepts only with a preset, a creation time and a master volume.
``sstim_planned`` asks for all three rather than inventing them:

.. code-block:: python

   from datetime import datetime, timezone

   import music
   from music.stimulation.sstim_planned import (
       from_sstim_planned_session_graph, to_sstim_planned_session_graph,
   )

   plan = music.StimulationSession(sample_rate=8000)
   plan.add(music.binaural_beats, duration=60, beat_freq=10)
   planned = to_sstim_planned_session_graph(
       plan,
       preset_iri="https://your-lab.example/presets/alpha-v1",
       preset_label="Alpha, version 1",
       created_at=datetime.now(timezone.utc),
       master_volume=0.2,
       base="https://your-lab.example/plans/002/")
   checked = from_sstim_planned_session_graph(planned)

The specification wraps the program and declares only
``reproEquivalentPresentation``. Its ``configurationDigest`` is a
SHA-256, named ``sha256-music-plan-json-v1``, of a JSON object holding
the program's N-Triples, sorted, with the preset, the master volume and
the duration, so that it changes with any of them. It identifies the
plan as the package writes it, not the samples another engine would
render, and it is still a plan: no ``SessionInstance``, event or exposure
is written. A session shorter than a minute, or not a whole number of
seconds, stays a program in the package's terms.

Reading SSTIM's terms alone
---------------------------

Everything above reads back what the package wrote. ``sstim_semantic``
reads what another tool could have written: a binaural or monaural beat
described in SSTIM's terms alone. Here the package's three triples are
taken out of a description, and the rest is enough:

.. code-block:: python

   from music.stimulation.sstim_io import MUSIC, to_sstim_graph
   from music.stimulation.sstim_semantic import (
       inspect_sstim_beat, render_semantic_beat,
   )

   graph = to_sstim_graph(
       "binaural_beats",
       parameters={"carrier_freq": 200, "beat_freq": 10},
       duration=30, sample_rate=48000,
       base="https://your-lab.example/stimuli/beat-1/")
   for triple in list(graph):
       if any(str(term).startswith(str(MUSIC)) for term in triple):
           graph.remove(triple)

   beat = inspect_sstim_beat(graph, sample_rate=48000)
   print(beat.technique, beat.carrier_freq, beat.beat_freq)
   samples = render_semantic_beat(
       beat, profile="zero-phase-equal-gain-beats-v1")

It accepts one determinate sine signal at a fixed beat rate, and either
two channels on the left and right ears with a binaural mechanism the
listener makes, or one channel with a monaural mechanism physically in
the signal, its two tones taken symmetrically about the carrier. It
checks the classes, the auditory, air-conducted delivery, the
placements, the carriers and the durations, and refuses anything
ambiguous. SSTIM need not choose a sample rate, so the caller gives the
one to render at.

The render names its convention, ``zero-phase-equal-gain-beats-v1``:
both carriers start at zero phase and a monaural beat's tones have equal
gains. SSTIM fixes neither, so a render that agrees in beat, laterality
and mechanism can still differ in every sample from another engine's,
and the convention is the package's, not SSTIM's.

The other stimuli are not read from SSTIM's terms alone, because they
cannot be rendered from them: a duty cycle and its ramps, a depth, a
deviation, a noise's spectral model and generator, a trajectory's
geometry. They would have to be invented, and are not.

Rendering another tool's description
------------------------------------

Any of the seven can be rendered from a description another tool wrote,
if you say in full how: what SSTIM's terms do not give has to come from
somewhere, and the package will not take it from its own defaults.
``sstim_contract`` takes the graph, with none of the package's terms in
it, and a sidecar naming the generator and every one of its controls,
and renders only if the two agree:

.. code-block:: python

   from music.stimulation.sstim_contract import (
       render_sstim_contract, resolve_sstim_contract,
   )

   resolution = resolve_sstim_contract(
       "tests/fixtures/sstim_third_party_am.ttl",
       generator="amplitude_modulation",
       parameters={"carrier_freq": 220.0, "modulation_freq": 9.0,
                   "modulation_depth": 0.7},
       duration=0.04, sample_rate=48000)
   samples = render_sstim_contract(resolution)

That Turtle file was written by hand, without the package, as were the
noise and the spatial motion beside it in the repository's
``tests/fixtures``. What is compared is what the two say, not how: the
IRIs, the order of the statements, the lexical form of a number and the
labels may differ. Every SSTIM statement about the stimulus, its
signals, channels and renderings must agree with the sidecar's
stimulus, and an SSTIM statement the reader does not know is refused,
since it might change the stimulus; statements in other vocabularies are
ignored. A graph carrying the package's own terms is refused here, and
read with ``from_sstim_graph`` instead, so that the two sets of terms
cannot disagree unseen. What a noise's description leaves to its
renderer -- the colour, the seed, the depth -- and a motion's -- the
trajectory and the geometry -- come from the sidecar. The render is the
package's, and ``exact_pcm_portable`` stays false: another engine given
the same sidecar owes nothing to these samples.

The same check runs from the command line, on a Turtle file and a JSON
sidecar of at most 8 KiB:

.. code-block:: console

   python tools/sstim_interop_check.py \
       tests/fixtures/sstim_third_party_am.ttl \
       tests/fixtures/sstim_third_party_am.contract.json \
       --json interop-report.json

Its report gives the package's version, the generator, the shape, the
peak and RMS, and SHA-256 digests of the samples, as little-endian
float64, and of the sidecar, which identify a render on this
installation and nothing more. ``--official`` also validates the file
against SSTIM's Full profile, and ``--wav`` writes the samples as 24-bit
PCM.

None of this is an agreement between engines. A shared reference for
phase and amplitude, a random stream, a localization model, calibration
of the hardware, timing and records of delivery would each need a
rendering profile versioned apart from any one engine, with shared
fixtures and tolerances, and a perceptual comparison would need
listeners.

What can be done with a graph
-----------------------------

``inspect_sstim_capabilities`` says, before anything renders, which of
the readers above would take a graph:

.. code-block:: python

   from music.stimulation.sstim_capabilities import (
       inspect_sstim_capabilities,
   )

   report = inspect_sstim_capabilities("stimulus.ttl")
   print(report.mode, report.technique, report.can_render)
   print(report.missing_contract)

``portable-beat-reference`` is a beat SSTIM's terms suffice for, rendered
under the convention above. ``music-extension`` and ``music-program`` are
a stimulus and a session in the package's terms, which the package
renders as it wrote them, and ``sstim-planned-music-program`` a planned
session as ``sstim_planned`` writes it. ``descriptive-only`` is an SSTIM
description the package recognizes but cannot render without what it
lacks, which ``missing_contract`` lists and a sidecar can supply.
``unsupported`` is anything unknown,
ambiguous or contradictory. ``exact_pcm_portable`` is false for all of
them, since conforming to SSTIM fixes no samples. This is a quick local
inspection, not the validator.

A graph carrying any of the package's terms is judged by them first: the
description written above reports ``music-extension``, and one whose own
terms disagree with its SSTIM terms -- 40 Hz in one and 10 Hz in the
other -- is ``unsupported``, rather than read by whichever set suits.
Only a graph in SSTIM's terms alone can be a ``portable-beat-reference``.

Reading safely
--------------

All three readers take a graph, a local file or Turtle text, never a
URL, and follow no link in a graph: up to 2 MiB of Turtle and 15,000
triples. No name in a graph is imported or run; the stimuli it can name
are a fixed list, and a render is capped at five million samples.

The bounds stop a large file being read by accident. They do not make a
hostile one safe, since the parser has run before the triples can be
counted: do not read untrusted Turtle without isolating the process and
limiting what it may use.
