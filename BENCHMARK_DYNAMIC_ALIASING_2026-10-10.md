# Dynamic anti-aliasing benchmark: first measured matrix

**Date:** 2026-10-10. **Source:** [GitHub Actions run #38067785797](https://github.com/ttm/music/actions/runs/38067785797), downloadable artifact **dynamic-alias-report** with 72 JSON records and Markdown summary. The tested commit is `553ce9f1` in [PR #120](https://github.com/ttm/music/pull/120). Source generator: `tools/benchmark_dynamic_aliasing.py`.

## Reproduction

```sh
python tools/benchmark_dynamic_aliasing.py --json dynamic-alias-report.json
```

The report records **both** MUSIC output and a separate continuously specified analytic source. Target rates are 44,100, 48,000 and 96,000 samples/s; physical carrier and modulation frequencies are **fractions of the target sampling rate** for comparable normalized stress, so this is *not* the same fixed-Hz waveform resampled at each rate. Each trial contains 4,096 target frames, a 16×-sampled analytic reference FIR-filtered to target Nyquist, and direct, 4× and 8× renderings. Timings below are GitHub runner median milliseconds, two warm repetitions, and are not universal CPU benchmarks.

**Actual MUSIC, 48 kHz:** RMS errors are waveform differences; spectral values are windowed FFT **magnitude** distances, phase-insensitive but not a clean isolation of aliased energy. Lower is better in both columns.

| Source | Direct RMS | 4× RMS | 8× RMS | Direct spectral | 4× spectral | 8× spectral | Direct ms | 4× ms | 8× ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Wide FM | 1.147 | 0.336 | 0.168 | 0.1191 | 0.00256 | 0.00128 | 0.111 | 0.750 | 1.916 |
| Extreme FM | 1.144 | 0.328 | 0.166 | 0.6346 | 0.01357 | 0.00724 | 0.112 | 0.760 | 1.920 |
| Nonlinear tanh | 1.294 | 1.259 | 1.259 | 0.3348 | 0.0157 | 0.00922 | 0.041 | 0.470 | 1.295 |
| Abrupt gate | 0.07493 | 0.01238 | 0.00536 | 0.1623 | 0.02301 | 0.00946 | 0.080 | 0.633 | 1.232 |

### Interpretation

- **Wide and extreme FM:** internal polyphase decimation reduces spectral-magnitude mismatch strongly. The extremely large direct **waveform** RMS is not a pure alias fraction. MUSIC's wavetable sampling and numerical phase are not the same calculation as the independent analytic reference, and these differences must not be conflated with foldover.
- **Nonlinear tanh:** the spectral improvement does not translate to comparable waveform RMS improvement. This is a useful warning that sample-phase disagreement dominates this particular time-domain comparison. A metric that shows better matching of spectral magnitudes is **not** by itself a result about listening preference.
- **Hard gate:** both spectral and waveform differences reduce, but the low-pass reconstruction smooths the intentionally abrupt edges. Its reduction in high-frequency energy is a numerical band limitation, not evidence that the user receives an unaltered hard gate.
- **Timing/memory:** in this CI run MUSIC 4× took approximately 7–12× the direct rendering time for these short buffers. The JSON records a nominal float64 intermediate-frame footprint, excluding all filter and generator scratch allocations; this is not a measured peak resident-memory profile.

Because the carrier and modulation frequencies scale with each sampling rate, the normalized alias-pressure results are similar across 44.1/48/96 kHz. Future tests should add **fixed physical frequency** cases across rates, longer durations, audio playback examples and independent listening judgments.

The reference uses a **finite 16× rate**, not an analytic infinite-bandwidth brick-wall filter; nonlinear spectra and hard edges theoretically extend without bound. Convergence checks at larger factors and short-time alias-specific sideband measurements remain necessary before making broad alias-free claims. In particular, no clinical or entrainment efficacy is established by these plots or errors.

## SSTIM companion

The same PR adds `music.stimulation.sstim_semantic`: an independent RDF reader that interprets supported standard SSTIM 0.19.0 binaural/monaural beat descriptions with **no MUSIC implementation hints**. The sample synthesizer demands an explicit `zero-phase-equal-gain-beats-v1` assumption; this is a reproducible test convention rather than a normative SSTIM rendering rule.

The suite includes independently authored Turtle, negative tests of signal/channel/mechanism conflicts, and official pinned SSTIM Full-profile validation. These do not yet supply a universal independent execution contract for all seven generators or for full sessions.

**No human singing or listening experiment was performed.**
