Portable SSTIM descriptions with explicit rendering contracts
==============================================================

**SSTIM 0.19.0** distinguishes a portable sensory stimulus description
from the engine-specific configuration that realizes it. MUSIC's
interoperability bridge respects that difference. The on-graph
``MUSIC.generator``/``MUSIC.parametersJson`` format is suitable
for MUSIC-to-MUSIC replay, but is *not* an interoperable guarantee that
two unrelated synth engines produce identical PCM.

Standard-only RDF with a complete sidecar
-----------------------------------------

``music.stimulation.sstim_contract`` can now accept a stimulus RDF
graph independently authored by another SSTIM producer, with no
``MUSIC`` triples and arbitrary resource IRIs, **provided the caller
also supplies an explicit, complete MUSIC engine contract**.

.. code-block:: python

   from music.stimulation.sstim_contract import (
       resolve_sstim_contract, render_sstim_contract,
   )

   # This local file can be produced by any SSTIM 0.19.0 authoring tool.
   resolution = resolve_sstim_contract(
       "auditory-description.ttl",
       generator="amplitude_modulation",
       parameters={
           "carrier_freq": 220.0,
           "modulation_freq": 9.0,
           "modulation_depth": 0.7,
       },
       duration=0.04, sample_rate=48000)
   assert resolution.exact_pcm_portable is False
   samples = render_sstim_contract(resolution)

The sidecar is a rendering choice made by the consumer, **not**
information encoded by SSTIM. The package insists that the sidecar
explicitly pin all the selected generator's supported controls rather
than fill omitted controls with MUSIC's local defaults.

For a **determinate** signal, the expected carrier, beat/modulation
rate, channel placements, physical-versus-perceptual presence, signal
shape, duration and channel mechanism are checked against the graph.
For stochastic noise, its RDF describes the spectral band and
modulation mechanism; the **noise color, seed/PRNG, and modulation depth**
remain engine controls. Spatial descriptions state a position
modulation and two ear channels, but a concrete trajectory, geometry,
ITD/ILD/HRTF assumptions and waveform still require a separate
implementation contract.

The checker performs a semantic projection rather than literal
triple-set equality. It is insensitive to RDF subject IRIs,
statement order, equivalent numeric lexical representations, and
arbitrary descriptive labels. It rejects contradictory or unknown
**SSTIM** assertions attached to participating signal/channel/rendering
nodes instead of silently discarding fields that might change the
physical stimulus.

It does **not** infer engine controls from labels, annotations,
similarities, unstated defaults or a scientific-effect claim.
Unknown generators never load source code from the RDF.

Interoperability levels and safety
----------------------------------

* **SSTIM-only beat reference:** ``sstim_semantic`` understands
  binaural/monaural sine beats, but exact samples still require
  the explicit zero-phase/equal-gain convention.
* **SSTIM with external MUSIC sidecar:** ``sstim_contract`` can
  resolve all seven built-in generators after checking each relevant
  SSTIM signal/channel/rendering assertion. It reproduces the selected
  MUSIC engine's behavior, *not* an arbitrary vendor's PCM.
* **MUSIC-owned execution hints already in RDF:** use the strict
  ``from_sstim_graph`` or ``from_sstim_advanced_graph`` readers.
  Sidecar import deliberately refuses graphs with MUSIC predicates
  instead of overriding potentially contradictory instructions.
* **MUSIC phase programs:** ``sstim_program`` describes ordered
  phases as a MUSIC-owned extension containing SSTIM stimulus graphs.
* **Eligible planned sessions:** ``sstim_planned`` adds an *actual*
  SSTIM ``SessionSpecification`` around a MUSIC program **only when**
  the caller supplies a versioned preset, a timezone-aware immutable
  creation timestamp, a master volume, and an integral duration between
  60 and 7,200 seconds. It declares only the conservative
  ``reproEquivalentPresentation`` level. The underlying phase ordering
  remains MUSIC-owned, not a general SSTIM execution protocol.
  No ``SessionInstance`` or actual exposure is asserted.

.. code-block:: python

   from datetime import datetime, timezone
   from music import StimulationSession, binaural_beats
   from music.stimulation.sstim_planned import (
       to_sstim_planned_session_graph,
       from_sstim_planned_session_graph,
   )

   program = StimulationSession(sample_rate=8000)
   program.add(binaural_beats, duration=60, beat_freq=10)
   planned = to_sstim_planned_session_graph(
       program,
       preset_iri="https://example.org/presets/declared-version-1",
       preset_label="Named versioned auditory preset",
       created_at=datetime.now(timezone.utc),
       master_volume=0.2,
       base="https://example.org/studies/plan-1/")
   checked = from_sstim_planned_session_graph(planned)

This example constructs an **intended plan**, not a report of a
participant session. A 20-second waveform demo cannot be renamed
``SessionSpecification`` under SSTIM's frozen Full profile because
that shape requires an integer duration from 60 to 7,200 seconds.
Preset/version identity is supplied explicitly, never invented by the
MUSIC renderer.

To check ontology/profile conformance, use the official pinned
``sstim.validate(..., profile="full", version="0.19.0")``
API separately. Successful local MUSIC rendering is **not** a
replacement for the official SHACL checks.

Input is restricted to local Turtle/in-memory graphs with the existing
2 MiB serialized input and 15,000 RDF triple budgets. These are useful
limits, **not** a security sandbox: untrusted Turtle parser execution
should be externally isolated if offered as a public service.

Scope and unresolved agreements
-------------------------------

This bridge makes it feasible for an unrelated producer to publish
standard SSTIM stimulus RDF while a MUSIC consumer chooses an explicit
renderer. It does not yet specify a cross-vendor phase and amplitude
reference, precise stochastic random stream, standardized localization
model, hardware calibration, real-time timing or generic session
delivery records.

Such choices require independently versioned rendering profiles,
conformance fixtures, cross-engine numerical tolerances and (for
perceptual comparisons) actual controlled evaluations. Sample-exact
replay within a pinned MUSIC installation does not establish
cross-implementation equivalence, clinical relevance or outcome
efficacy.
