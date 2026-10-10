Independent DSP references and portable SSTIM beat profiles
============================================================

Dynamic DSP comparison
----------------------

The existing static-wavetable benchmark is not valid for time-varying
FM or nonlinear synthesis. Use the **numerical high-rate reference**
instead:

.. code-block:: console

   python tools/benchmark_dynamic_aliasing.py --json dynamic-alias.json
   python tools/benchmark_dynamic_aliasing.py --rates 48000 --repeats 3

The full default matrix evaluates 44.1, 48 and 96 kHz, four sources
(wide-deviation FM, extreme FM periodically exceeding output Nyquist,
tanh-shaped nonlinear tone and an abrupt carrier gate), and direct/4x/8x
synthesis against a 16x analytic reference low-pass filtered and
decimated using the same specified FIR family.

Each record states both **engine = analytic** and **engine = MUSIC**
separately. The former establishes what oversampling can do to the
analytic signal; the latter measures the actual current library, which
also has finite lookup-table and phase-integration approximations.
It is not sound science to paste the analytic engine's improvements onto
the MUSIC renderer without measuring it.

The report includes global RMS difference, worst eighth-of-record RMS
difference, windowed FFT-magnitude difference (less sensitive to phase),
rendering median milliseconds, peak magnitude, and the nominal
intermediate float64 audio-buffer size. The latter excludes filter and
renderer scratch allocations and is not peak resident-memory usage. The short-window metrics help prevent
short bursts of folded energy from disappearing inside a long-record
global average. All comparisons use the same target sample count and
rate; requested physical carrier and modulation frequencies remain
unchanged as the internal render rate increases.

A 16x finite-rate signal is a **convergence approximation**, not an
infinite-bandwidth ground truth. For more demanding modulation, compare
against 32x/64x using short records; consider time-domain delays,
filter-end transients, and sideband power, not RMS alone. The magnitude metric is not a\npure folding-power estimate. The matrix is
a reproducible developer diagnostic, **not** a blanket promise of
alias-free rendering, superior human preference or neural efficacy.

Independent SSTIM interpretation
--------------------------------

The separate ``music.stimulation.sstim_semantic`` module consumes a
restricted SSTIM 0.19.0 **auditory beat profile** using only standard
SSTIM properties, without relying on MUSIC-specific generator names,
sample-rate hints or JSON engine parameters.

It currently accepts one fixed positive sine signal and either:

* Two auditory channels with left/right ear placements, matching
  carrier difference and ``mechanismBinauralBeat``, marked
  **perceptual**.
* One auditory channel with ``mechanismMonauralBeat``, marked
  **physical**, with the sine pair understood as symmetric around the
  declared carrier.

It checks signal/renderer classes, physical delivery, channel modality,
rendering target, shape, fixed rate, durations and allowed output
sampling rate, and refuses ambiguous or unsupported cases.

.. code-block:: python

   from music.stimulation.sstim_semantic import (
       inspect_sstim_beat, render_semantic_beat,
   )

   # 'independently-authored.ttl' needs no MUSIC engine-hints namespace.
   contract = inspect_sstim_beat(
       "independently-authored.ttl", sample_rate=48000)
   print(contract.technique, contract.carrier_freq, contract.beat_freq)

   # Explicit, non-normative analytic rendering assumptions:
   samples = render_semantic_beat(
       contract, profile="zero-phase-equal-gain-beats-v1")

That profile fixes carrier starting phases to zero and sets monaural
amplitudes to an equal-gain mean. **Those assumptions are not guaranteed
by SSTIM.** Two conforming audio engines may render equivalent
beat frequency, laterality and mechanism but different waveform samples
because output gain, phase, carrier waveform implementation, calibration
and hardware were not fixed by the source graph. Compare semantic and
spectral observables before testing exact PCM identity.

The tests construct RDF Turtle independently of MUSIC's exporter,
then independently synthesize zero-phase carriers using NumPy.
Removing all MUSIC implementation hints from an existing
``to_sstim_graph`` graph also leaves its standard supported signal
semantics interpretable. This is a meaningful interoperability step,
not complete cross-vendor conformance or universal session execution.

Isochronic, generic amplitude/frequency modulation, stochastic noise,
spatial trajectories and MUSIC-owned phase programs remain outside the
semantic-only **executable** subset. Their exact output requires
additional constraints, such as envelope duty cycle/ramp, depth,
waveform, frequency deviation, noise seed/PRNG, localization model,
phase conventions and timing. Never silently invent these from
``hasRenderingMechanism`` alone.

Resource and trust boundaries
-----------------------------

The decoder limits output to five million samples and refuses unknown
methods and conflicting signal/channel assertions. Parsing Turtle has
its own resource/canonicalization risks; the output bound does not make
arbitrarily large, untrusted RDF safe to parse. Use the official pinned
SSTIM Full-profile validator for ontology checks when importing a
scientific data asset.

This reference does not add new efficacy assertions. It also does not
replace the separate MUSIC-specific replay functions, whose purpose is
to preserve implementation parameters, not to establish portability.
