SSTIM import limits and renderer capability negotiation
=========================================================

SSTIM 0.19.0 provides a portable vocabulary for descriptions of
sensory stimuli and separately distinguishes planned sessions from
observed executions. MUSIC's ordinary RDF bridge and its seven
allowlisted generator implementations are **not** equivalent to a
universal engine-independent audio renderer.

Local-only ingestion
--------------------

All three MUSIC SSTIM import paths share the same bounded Turtle loader.
It accepts only an in-memory RDFLib graph, local ``Path`` or a local
Turtle text string / file path string. It refuses URI-based imports such
as HTTP, FTP and file URLs and does not dereference network RDF links.

Limits are **2 MiB** of original Turtle bytes and **15,000 triples**
after parsing. These checks prevent accidental unrestricted large-file
imports and silent URL fetching. A small hostile Turtle input can still
cause parser CPU/memory use *before* the post-parse triple count is
checked, so the limits are **not a security sandbox**. Do not offer
untrusted public Turtle processing without external process isolation,
resource constraints and appropriate transport validation.

.. code-block:: python

   from music.stimulation.sstim_capabilities import (
       inspect_sstim_capabilities,
   )

   report = inspect_sstim_capabilities("local-stimulus.ttl")
   print(report.mode, report.technique, report.can_render)
   print(report.missing_contract)

Capability contracts
--------------------

``inspect_sstim_capabilities`` is an honest decision interface,
not a clinical validator or certification of playback equipment:

* ``portable-beat-reference``: standard SSTIM terms are sufficient
  to interpret a restricted determinate binaural/monaural sine beat.
  Reproduction still needs an *explicit* zero-phase/equal-gain reference
  convention and cannot claim arbitrary exact output calibration.
* ``music-extension``: a MUSIC-owned parameter dictionary passes the
  strict local replayer, so the selected MUSIC engine can render it.
  Exact samples across independently developed renderers are NOT
  guaranteed.
* ``music-program``: an ordered MUSIC-owned program passes its
  planned-phase integrity checks. This does not claim a real delivered
  ``sstim:SessionInstance`` or automatically become a standard
  ``sstim:SessionSpecification``.
* ``descriptive-only``: known SSTIM audio descriptions whose
  execution needs missing implementation controls (AM depth and ramp,
  FM deviation and phase, colored-noise spectral model/PRNG, or spatial
  trajectory/ITD/ILD geometry).
* ``unsupported``: unknown, ambiguous or contradictory input.
  No graph-provided function names, executable code, downloaded models,
  remote waveforms or arbitrary generators are invoked.

For **all statuses**, ``exact_pcm_portable`` is false: SSTIM
conceptual conformance alone cannot stipulate a digital audio algorithm,
bit-exact seed realization, phase conventions and output calibration.
A separately pinned MUSIC build and implementation-specific data can
provide repeatable MUSIC renders without implying cross-vendor equality.

Official SSTIM Full-profile validation remains separate. Use
``validate_sstim(graph, version="0.19.0")`` when ontology conformance
is required; the capability function is a fast bounded local
dispatchability inspection, not a replacement for SHACL.
