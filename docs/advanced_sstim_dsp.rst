Advanced SSTIM exchange and spectral synthesis
==============================================

The full set of MUSIC auditory generators now has an **opt-in adapter**
for exchanging structured stimulus descriptions. SSTIM 0.19.0 supplies
the portable vocabulary for signals, channels and renderings. MUSIC owns
the executable generator names, random seeds, spatial geometry, ordering,
gain and crossfade policy. An SSTIM description does not by itself
authorize a clinical claim or say that anything was actually delivered.

Noise and spatial movement
--------------------------

A noise source is **stochastic**, specified by its band and spectral
color, not by a fictitious sine carrier. A seed allows a reproducible
*realization* in the MUSIC renderer. Changing the seed changes the actual
waveform without changing its underlying stochastic specification.

.. code-block:: python

   from music.stimulation.sstim_advanced import (
       to_sstim_advanced_graph, render_sstim_advanced,
   )

   g = to_sstim_advanced_graph(
       "modulated_noise",
       parameters={
           "noise_type": "pink", "min_freq": 100, "max_freq": 10000,
           "modulation_freq": 10, "seed": 42,
       },
       duration=15,
       sample_rate=48000,
       base="https://your-lab.example/stimuli/noise-1/",
   )
   audio = render_sstim_advanced(g)  # same samples on repeated render

For spatial movement, the graph carries a source tone and a separate
time-varying *spatial-position signal*. Two ear channels reflect the
stereo presentation. MUSIC uses geometric ITD/ILD cues, not a measured
head-related transfer function. It cannot claim full three-dimensional
localization or resolve front/back ambiguity.

.. code-block:: python

   trajectory = to_sstim_advanced_graph(
       "spatial_motion",
       parameters={
           "carrier_freq": 440, "motion_rate": 0.5,
           "theta1": 180, "theta2": 0, "dist": 0.1,
       },
       duration=12,
       sample_rate=48000,
       base="https://your-lab.example/stimuli/motion-1/",
   )

Sequential programs
-------------------

A :class:`music.StimulationSession` has no execution clock, participant
or playback-event record. Accordingly, we represent its **planned**
ordered phases with MUSIC terms and embed actual SSTIM
``StimulusSpecification`` descriptions of each constituent phase.
We do not create an ``sstim:SessionInstance`` for audio that was
merely synthesized.

.. code-block:: python

   import music
   from music.stimulation.sstim_program import (
       to_sstim_program_graph, from_sstim_program_graph,
       render_sstim_program,
   )

   plan = music.StimulationSession(
       sample_rate=48000, end_ramp=.05)
   plan.add(music.binaural_beats, duration=10, beat_freq=10)
   plan.add(music.modulated_noise, duration=10, ramp=.05,
            noise_type="pink", seed=42)
   graph = to_sstim_program_graph(
       plan, base="https://your-lab.example/plans/001/")
   graph.serialize("program.ttl", format="turtle")
   replay = render_sstim_program("program.ttl")

All seven generators can be used as phases if they have portable,
allowed numeric arguments. The adapter rejects pre-rendered arrays,
arbitrary callables, nondefault waveform tables and mismatched triples.
It preserves phase ordering, fades, gain, sample rate and exact timeline.
An unseeded noise phase correctly remains nondeterministic.

The Full-profile SSTIM validator checks the constituent descriptions.
The higher-level MUSIC ``StimulationProgram`` record is a **MUSIC
extension**, not a claim of standardised SSTIM protocol semantics. To
describe a real execution and its timing and delivery, use the dedicated
SSTIM session builder and validate that record separately.

Frequency-dependent band limiting
---------------------------------

``music.core.synths.frequency_path.bandlimited_frequency_path``
synthesizes a sine, sawtooth, square or triangle from a frequency value
for **each output sample**. The fundamental phase is integrated through
the trajectory and harmonic gains fade towards zero near Nyquist.

.. code-block:: python

   import numpy as np
   from music.core.synths.frequency_path import bandlimited_frequency_path

   fs = 48000
   hz = np.linspace(300, 12000, fs)
   sweep = bandlimited_frequency_path(
       hz, sample_rate=fs, waveform="sawtooth",
       transition=.15, max_harmonics=256)

This is more appropriate for changing-pitch *rich periodic waveforms*
than filtering a table once according to the initial pitch. It is not a
proof of alias-free fast FM: the time variation itself produces spectral
sidebands. Rapid modulation and nonlinear processing can still benefit
from ``music.render_oversampled``.

Computational limits and evidence
---------------------------------

The additive oscillator caps frequency-harmonic operations and the
SSTIM adapters bound each rendering to five million samples. Run the
spectral benchmarks with 44.1, 48 and 96 kHz sample rates before
adopting a parameter range. Anti-aliasing changes waveform timbre,
especially near Nyquist, and no independent listener preference is
established by purely numerical tests.

The portable RDF may be read by other SSTIM-aware tools, but
**sample-exact playback from the RDF still depends on MUSIC-owned
generator parameters**. SSTIM deliberately does not define the
particular DSP implementation, random generator algorithm, resampling
window, or MUSIC fade interpolation convention.
