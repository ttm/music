Stimuli in SSTIM
================

`SSTIM <https://w3id.org/sstim>`_, the Sensory Stimulation Vocabulary,
describes a stimulus in RDF: its signals, the channels it is presented
on, and how each signal is rendered onto each channel. One module,
``music.stimulation.sstim``, writes the package's seven auditory stimuli
and sessions of them as SSTIM 0.19.0 descriptions, renders them back,
and reads what other tools write. It needs:

.. code-block:: console

   pip install 'music[sstim]'

A description says what a stimulus is. It is not a record that anyone
heard it, nor a claim that it does anything, and SSTIM fixes no
oscillator, phase, gain or random generator, so another engine can
render a valid description differently and still be right. What the
package can promise, it promises for its own renders.

Writing a stimulus, and rendering it back
-----------------------------------------

:func:`~music.stimulation.sstim.to_sstim_graph` describes a stimulus as
the call that renders it:

.. code-block:: python

   import music
   from music.stimulation.sstim import (
       render_sstim, to_sstim_graph, validate_sstim,
   )

   graph = to_sstim_graph(
       music.binaural_beats, carrier_freq=200, beat_freq=10,
       duration=30, sample_rate=48000,
       base="https://your-lab.example/stimuli/run-001/")
   report = validate_sstim(graph)
   if not report.ok:
       raise ValueError(str(report))
   graph.serialize("stimulus.ttl", format="turtle")
   audio = render_sstim("stimulus.ttl")

The stimulus is named by its function or its name, ``"binaural_beats"``,
and given the arguments it would be called with. Every argument left to
its default is written out, so that a later default cannot change what
the description renders. ``base`` is an IRI of your own that the
description's resources are named under; the default, under
``example.org``, is for trying things, not for publishing.

The description is a ``StimulusSpecification`` holding SSTIM's
``StimulationSignal``, ``StimulusChannel`` and ``SignalRendering``: the
signals and their frequencies, the channels and where they are
presented, and how each channel renders each signal -- the mechanism,
the parameter it drives, the carrier, and whether the result is
physically in the sound or made by the listener. Each channel also
states its frequencies in SSTIM's own channel terms, so that a consumer
reading only those finds them:

======================  =====================================================
Stimulus                Channel terms, in ``sstim-ex:``
======================  =====================================================
binaural beats          ``hasCarrierFrequencyHz``, each ear's own tone;
                        ``hasBeatFrequencyHz``
monaural beats          ``hasCarrierFrequencyHz``, ``hasBeatFrequencyHz``
isochronic tones        ``hasCarrierFrequencyHz``,
                        ``hasModulationFrequencyHz``, ``hasDutyCycle``
amplitude modulation    ``hasCarrierFrequencyHz``,
                        ``hasModulationFrequencyHz``
frequency modulation    ``hasCarrierFrequencyHz``,
                        ``hasModulationFrequencyHz``
modulated noise         ``hasStimulusPattern``, its colour;
                        ``hasModulationFrequencyHz``
spatial motion          ``hasToneFrequencyHz``; ``hasModulationFrequencyHz``,
                        the motion's
======================  =====================================================

``sstim-ex:`` standing for ``https://w3id.org/sstim/exposure#``. What
SSTIM has no term for -- a depth, a deviation, a ramp, a seed, a
geometry, the sample rate -- goes in the package's own namespace,
``https://github.com/ttm/music/terms#``, never minted as SSTIM's,
together with the name of the stimulus that renders it. Those make the
render exact: :func:`~music.stimulation.sstim.render_sstim` renders a
description as the call it records would, sample for sample, and with
``oversampling_factor=4`` renders it through
:func:`music.render_oversampled` (see :doc:`bandlimited_synthesis`).

:func:`~music.stimulation.sstim.validate_sstim` checks a description with
SSTIM's own validator, in its Full profile, which the ``sstim`` package
fetches the first time and caches. It checks the description, not the
samples.

Noise and spatial motion
------------------------

A noise is described as a stochastic band, not a carrier. Its colour is
SSTIM's noise pattern where SSTIM has one: white, pink, brown, blue and
violet do. The package's black noise, which SSTIM's black -- silence --
is not, and a slope in decibels an octave have none, and stay in the
package's terms. A seed is recorded, so the description renders the same
noise every time; without one, each render draws anew.

.. code-block:: python

   noise = to_sstim_graph(
       "modulated_noise", noise_type="pink", min_freq=100,
       max_freq=10000, modulation_freq=10, seed=42,
       duration=15, sample_rate=48000,
       base="https://your-lab.example/stimuli/noise-1/")

Spatial motion is described as a tone and, separately, a signal of
position, both presented on the two ears:

.. code-block:: python

   motion = to_sstim_graph(
       "spatial_motion", carrier_freq=440, motion_rate=0.5,
       theta1=180, theta2=0, dist=0.1,
       duration=12, sample_rate=48000,
       base="https://your-lab.example/stimuli/motion-1/")

The motion is the package's geometric one, from interaural time and
intensity differences, not a measured head: it carries no elevation and
cannot tell front from back. Only the stimuli's own sine modulator and
tone are written; a custom waveform table is not.

Sessions
--------

A :class:`music.StimulationSession` is written the same way. SSTIM 0.19
has no terms for phases in order, so a session is a program in the
package's terms -- a ``StimulationProgram`` and its phases, with their
order, ramps and gains -- each phase an SSTIM description of its own:

.. code-block:: python

   session = music.StimulationSession(sample_rate=48000, end_ramp=.05)
   session.add(music.binaural_beats, duration=600, beat_freq=10)
   session.add(music.modulated_noise, duration=300, ramp=2,
               noise_type="pink", seed=42)
   graph = to_sstim_graph(
       session, base="https://your-lab.example/plans/001/")
   graph.serialize("program.ttl", format="turtle")

Any of the seven stimuli can be a phase, with arguments a description
holds: a session holding a rendered array, another callable or a custom
table is refused, and so is one of more than fifty phases. A noise phase
without a seed stays random.

Given a :class:`~music.stimulation.sstim.SessionPlan`, a session of a
whole number of seconds, from a minute to two hours, is also SSTIM's own
``SessionSpecification``, which SSTIM's Full profile accepts only with a
preset, a creation time and a master volume. The plan gives all three,
rather than the package inventing them:

.. code-block:: python

   from datetime import datetime, timezone

   from music.stimulation.sstim import SessionPlan

   plan = SessionPlan(
       preset_iri="https://your-lab.example/presets/alpha-v1",
       preset_label="Alpha, version 1",
       created_at=datetime.now(timezone.utc),
       master_volume=0.2)
   planned = to_sstim_graph(
       session, plan=plan, base="https://your-lab.example/plans/002/")

The specification wraps the program and declares only
``reproEquivalentPresentation``. Its ``configurationDigest``, named
``sha256-music-plan-json-v1``, is the SHA-256 of a JSON object, its keys
sorted, written without spaces and in UTF-8 rather than escaped to
ASCII: ``program_ntriples``, the program's N-Triples lines, sorted;
``preset_iri``; ``master_volume``, as a plain decimal such as
``"0.25"``; and ``duration_seconds``, a whole number. It changes with
any of them, and identifies the plan as the package writes it, not the
samples another engine would render. Neither a program nor a plan is an ``sstim:SessionInstance``,
which would claim a delivery: no clock, listener or record of playback
is written. The ``sstim`` package's own ``Session`` writes one, from what
happened.

Reading a description back
--------------------------

:func:`~music.stimulation.sstim.from_sstim_graph` gives back what renders
a description: an :class:`~music.stimulation.sstim.SstimStimulus` -- the
stimulus's name, its arguments, its length and its rate -- a
:class:`music.StimulationSession`, or a
:class:`~music.stimulation.sstim.PlannedSession`, each with its
``render()``:

.. code-block:: python

   from music.stimulation.sstim import from_sstim_graph

   stimulus = from_sstim_graph("stimulus.ttl")
   print(stimulus.generator, stimulus.parameters)
   audio = stimulus.render()

A description the package wrote is read strictly. Its statements in
SSTIM's terms and the package's, about the resources it names, must be
exactly those the package writes for what it says: one whose package
terms say 40 Hz and whose SSTIM terms say 10, or that adds a gain or a
phase in either, is refused rather than half believed. Statements in
other vocabularies -- a label, a comment, provenance -- are left alone,
so a description can be annotated freely. What is read back describes
itself again: given the same ``base``,
``to_sstim_graph(from_sstim_graph(graph), base=base)`` writes the same
statements.

Reading another tool's description
----------------------------------

A description another tool wrote, in SSTIM's terms alone, is read as one
of the seven stimuli, which its mechanisms and signals say:

==========================================  ======================
It states                                   It is read as
==========================================  ======================
a noise signal                              modulated noise
a signal rendered onto spatial position     spatial motion
a binaural beat of a sine signal            binaural beats
a monaural beat of a sine signal            monaural beats
amplitude modulation by a square signal     isochronic tones
amplitude modulation by a sine signal       amplitude modulation
frequency modulation by a sine signal       frequency modulation
==========================================  ======================

Whatever it states is read from it: carriers, rates, a duty cycle, a
noise's band and colour, its length. What SSTIM has no term for is
given, with the sample rate, which SSTIM never fixes, and nothing is
taken from the package's defaults.
:func:`~music.stimulation.sstim.inspect_sstim` says what is needed:

.. code-block:: python

   from music.stimulation.sstim import inspect_sstim, render_sstim

   path = "tests/fixtures/sstim_third_party_am.ttl"
   print(inspect_sstim(path).needs)
   # ('sample_rate', 'modulation_depth')
   audio = render_sstim(path, sample_rate=48000,
                        controls={"modulation_depth": 0.7})

That file was written by hand, without the package, as were the noise
and the spatial motion beside it in the repository's ``tests/fixtures``.
A beat needs only the sample rate. The others need, in ``controls``:

======================  ====================================================
Stimulus                Controls
======================  ====================================================
isochronic tones        ``ramp_duration``; ``duty_cycle``, unless
                        ``hasDutyCycle`` states it
amplitude modulation    ``modulation_depth``
frequency modulation    ``frequency_deviation``
modulated noise         ``seed``; ``modulation_depth``, if it is modulated;
                        ``noise_type``, unless a pattern names it
spatial motion          ``theta1``, ``theta2``, ``dist``, ``zeta`` and
                        ``air_temp``
======================  ====================================================

and ``duration``, given apart, when no channel states one. A control the
description states may be given too, and must then agree with it.

The package's own description of the stimulus read must then state
everything the given one does. What is compared is what the two say, not
how: node names, which may be blank, the order of statements, how a
number is written and the labels may all differ, and numbers agree to a
part in a billion. A description may leave out what SSTIM makes
optional, such as a carrier or a channel's frequencies, and may state a
more general term where the package writes a specific one: a modulation
frequency for a beat, or a channel's plain ``hasFrequencyHz``. It may
state what says nothing about the sound -- what the stimulation targets,
a band a signal lies within, a channel's role, comfort boundary or
device -- and that is left alone. It may not state an SSTIM term the
package does not render, such as a gain, a phase offset or a sound
pressure level, since the render would not be what it describes; nor
contradict itself, as two carriers not one beat apart do. Each refusal
says what was found.

A session in SSTIM's terms alone names a preset only its own engine
renders, and is refused.

Beats by a named convention
---------------------------

A binaural or monaural beat can also be rendered without the package's
generator, by a convention another engine can follow to within the
rounding of its sines, :data:`~music.stimulation.sstim.BEAT_PROFILE`:
both tones are sines starting at phase zero, and a monaural beat is
their mean.

.. code-block:: python

   from music.stimulation.sstim import BEAT_PROFILE, from_sstim_graph

   beat = from_sstim_graph("stimulus.ttl", profile=BEAT_PROFILE)
   print(beat.generator, beat.carrier_freq, beat.beat_freq)
   samples = beat.render()

That is the binaural beat written at the top of this page. A description
the package wrote fixes its own sample rate; one in SSTIM's terms alone
fixes none, and is read with a ``sample_rate`` given too. SSTIM fixes
neither the phase nor the gain, so the convention is the package's,
named, and not SSTIM's: a render that agrees in beat, laterality and
mechanism can differ in every sample from another engine's that does not
follow it.

From the command line
---------------------

``tools/sstim_interop_check.py`` renders a Turtle file with
``render_sstim``'s arguments, read from a JSON file of at most 8 KiB:

.. code-block:: console

   python tools/sstim_interop_check.py \
       tests/fixtures/sstim_third_party_am.ttl \
       tests/fixtures/sstim_third_party_am.render.json \
       --json interop-report.json

where the JSON is ``{"sample_rate": 48000, "controls":
{"modulation_depth": 0.7}}``. Its report gives the package's version,
the stimulus and what rendered it, the shape, the peak and RMS, and
SHA-256 digests of the samples, as little-endian float64, and of the
arguments, which identify a render on this installation and nothing
more. ``--official`` also validates the file against SSTIM's Full
profile, and ``--wav`` writes the samples as 24-bit PCM.

None of this is an agreement between engines. A shared reference for
phase and amplitude, a random stream, a localization model, calibration
of the hardware, timing and records of delivery would each need a
rendering profile versioned apart from any one engine, with shared
fixtures and tolerances, and a perceptual comparison would need
listeners.

What can be done with a description
-----------------------------------

:func:`~music.stimulation.sstim.inspect_sstim` says, before anything
renders, what the package can do with a description:

.. code-block:: python

   capability = inspect_sstim("stimulus.ttl")
   print(capability.kind, capability.generator, capability.needs)

``kind`` is ``stimulus``, ``session``, ``planned-session``, or
``unsupported`` with the ``reason``. ``written_by_package`` says whether
it uses the package's terms, by which it is then judged whole: one whose
package terms disagree with its SSTIM terms is ``unsupported``, rather
than read by whichever set suits. ``needs`` lists what rendering still
needs, ``profiles`` the conventions a beat can also be rendered by, and
``can_render`` is true when nothing more is needed. It is a quick local
reading, not the validator.

Reading safely
--------------

Every reader takes a graph, a local file or Turtle text, never a URL,
and follows no link in a graph: up to 2 MiB of Turtle and 15,000
triples. No name in a graph is imported or run; the stimuli it can name
are a fixed list.

A description is only numbers, so writing or reading one has no length
cap: a two-hour plan is written and read like a minute's.
``render_sstim`` renders at most :data:`~music.stimulation.sstim.MAX_SAMPLES`
samples a channel, about 113 s at 44.1 kHz, unless given a larger
``max_samples``, so that a description from elsewhere cannot exhaust
memory by accident. What ``from_sstim_graph`` returns renders with no
cap, once its length has been seen.

The bounds stop a large file being read by accident. They do not make a
hostile one safe, since the parser has run before the triples can be
counted: do not read untrusted Turtle without isolating the process and
limiting what it may use.

Reference
---------

.. currentmodule:: music.stimulation.sstim

None of these is exported at the top level: import them from
``music.stimulation.sstim``, with ``pip install 'music[sstim]'``.

.. autofunction:: to_sstim_graph

.. autofunction:: from_sstim_graph

.. autofunction:: render_sstim

.. autofunction:: inspect_sstim

.. autofunction:: validate_sstim

What they take and give
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SstimStimulus
   :members: render

.. autoclass:: SstimBeat
   :members: render

.. autoclass:: SessionPlan

.. autoclass:: PlannedSession
   :members: render

.. autoclass:: SstimCapability
   :members: can_render

Constants
~~~~~~~~~

.. data:: STIMULI

   The seven stimuli a description can be written for, by name:
   ``binaural_beats``, ``monaural_beats``, ``isochronic_tones``,
   ``amplitude_modulation``, ``frequency_modulation``, ``modulated_noise``
   and ``spatial_motion``.

.. data:: BEAT_PROFILE
   :value: "zero-phase-equal-gain-beats-v1"

   The convention a beat in SSTIM's terms alone can be rendered by.

.. data:: SSTIM_VERSION
   :value: "0.19.0"

   The SSTIM release descriptions are written in and validated against.

.. data:: MAX_SAMPLES
   :value: 5000000

   How many samples a channel :func:`render_sstim` renders at most, by
   default.

.. data:: SSTIM
          SSTIM_V
          SSTIM_EX
          MUSIC

   The namespaces: SSTIM's core, ``https://w3id.org/sstim#``; its
   vocabulary, ``https://w3id.org/sstim/vocab#``; its exposure module,
   ``https://w3id.org/sstim/exposure#``; and the package's own,
   ``https://github.com/ttm/music/terms#``, as
   :class:`rdflib.Namespace`.
