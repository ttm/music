# The MASS reconciliation

*Measured **2026-09-04**, `music` 1.4.0 against
[ttm/mass](https://github.com/ttm/mass) at `e516b08`.*

This package's central claim is fidelity to the MASS framework. Until this
file existed, the claim rested on the docstrings: every routine named the
article section it came from, and nothing checked that the samples agreed.
`ASSESSMENT.md` said so, and issue #67 tracked it.

This is the comparison. For each of the 35 routines and constants in the
reference's `src/aux/functions.py`, the package either reproduces it sample
for sample, or diverges for a reason stated below.

```console
python tools/mass_reconcile.py                  # print the register
python tools/mass_reconcile.py --write-fixture  # refresh the test fixture
python tools/mass_reconcile.py --register RECONCILIATION.md   # this table
```

**26 sample-exact, 5 divergent, 4 where the reference does not run.**

## How to read it

The reference and the package are handed matched arguments, and both are
given the *reference's* waveform tables, so a difference in a synthesis
routine is never confused with a difference in a constant. The tables are
compared as entries in their own right, at the end. Sequence arguments
mirror the shapes the reference's own defaults use, with the durations cut
down so the fixture stays small.

`tools/mass_reconcile.py` runs both implementations and fails when this
table disagrees with what it measures, so a divergence cannot appear or
widen without someone writing down why.
`tests/test_mass_reconciliation.py` checks the same register on every push,
reading the recorded outputs rather than the reference itself.

## The reference is GPL-3; this package is MIT

So the reference is never vendored here. `tools/mass_reference.py` reads
whatever checkout you point it at — `--mass PATH`, `$MASS_SRC`, or a
conventional location — and the only thing that enters this repository is
the numbers that came out of running it, in
`tests/fixtures/mass_reference.npz`: a digest for every routine that must
agree exactly, and the samples themselves where the two differ.

It also does not run. The file was written for Python 2 and NumPy 1.x, and
loading it at all needs four source patches, each listed with its reason in
`PATCHES`: an unimportable `from HRTF import *`, a table length that true
division makes a float, and `np.int` and `np.float`, both removed in NumPy
1.24.

## What the comparison found

Two defects in exported package routines, both now fixed.
`note_with_vibratos_glissandos` and `note_with_vibrato_seq_localization`
had collapsed the reference's two accumulators — one per vibrato, one per
segment within a vibrato — into a single name. Each outer pass discarded
the vibrato before it, and each appended its own concatenation back into
the list it was concatenating, so the frequency contour was multiplied by
every segment *and* by their joins. Both routines returned an array of
plausible length, which is why nothing caught it: 99.9 % of the samples
were wrong. With the accumulators separated, both agree with the reference.

Nine reference defects, which the package had already fixed or worked
around, and which are now on the record rather than implicit in the code.
Four of them mean a reference routine has never run at all: `loc2` and `R`
read variables their own signatures do not declare, `noises` indexes with a
float, and `FIR` calls a `convolve` that recurses into itself in both
branches. Where the register says the reference does not run, the package
does, and the test asserts it.

<!-- register:begin -->
| MASS | `music` | Outcome | Why |
|---|---|---|---|
| `__n` | `normalize_mono` | exact | sample-exact agreement |
| `__ns` | `normalize_stereo` | exact | sample-exact agreement |
| `N` | `note` | exact | sample-exact agreement |
| `N_` | `note_with_phase` | exact | sample-exact agreement |
| `V` | `note_with_vibrato` | exact | sample-exact agreement |
| `FM` | `note_with_fm` | exact | sample-exact agreement |
| `P` | `note_with_glissando` | exact | sample-exact agreement |
| `PV` | `note_with_glissando_vibrato` | exact | sample-exact agreement |
| `VV` | `note_with_two_vibratos` | exact | sample-exact agreement |
| `PVV` | `note_with_two_vibratos_glissando` | exact | sample-exact agreement |
| `PV_` | `note_with_vibratos_glissandos` | exact | sample-exact agreement |
| `trill` | `trill` | divergent | trill takes no waveform table, so it synthesizes through the package's corrected triangular table while the reference uses its own; max&nbsp;\|Δ\|&nbsp;=&nbsp;0.000241 |
| `noises` | `noise` | reference does not run | the reference indexes coefs[Lambda/2] with a true- division float, which has raised IndexError since Python 3; under Python 2 it ran, but into a real-valued coefficient array that discarded the imaginary part of every randomized phase, leaving a spectrum with no phase randomization at all — `IndexError` |
| `T` | `tremolo` | exact | sample-exact agreement |
| `T_` | `tremolos` | exact | sample-exact agreement (the reference assigns into its own arguments, so it requires lists where the package accepts any sequence) |
| `AM` | `am` | exact | sample-exact agreement |
| `AD` | `adsr` | exact | sample-exact agreement |
| `ADS` | `adsr_stereo` | exact | sample-exact agreement |
| `L` | `loud` | exact | sample-exact agreement |
| `L_` | `louds` | exact | sample-exact agreement |
| `F` | `fade` | exact | sample-exact agreement |
| `loc` | `localize` | exact | sample-exact agreement |
| `loc2` | `localize_linear` | reference does not run | the reference declares dist1 and dist2 and its body reads an undefined dist, so loc2 raises NameError on every call and has never run — `NameError` |
| `loc_` | `localize2` | divergent | four corrections the package carries and documents in place: the FFT bin spacing read 2*fs/Lambda rather than fs/Lambda, so every frequency was an octave high; the interaural delay was applied with the sign that advances the far ear rather than delaying it; the delay was wrapped into one period of f before becoming a phase, which is redundant and wrapped the two branches inconsistently; and the reference prints rather than raises on an unknown method. The reference's brute branch additionally builds n.zeros((2, maxsize)) from a float and raises TypeError; max&nbsp;\|Δ\|&nbsp;=&nbsp;1.04 |
| `D` | `note_with_doppler` | exact | sample-exact agreement |
| `D_` | `note_with_vibrato_seq_localization` | divergent | the package folds the running phase into one table period as it goes rather than accumulating it with cumsum, so the two round to different table indexes at a boundary; the difference is bounded by one step of the table and does not grow with the length of the render (issue #102); max&nbsp;\|Δ\|&nbsp;=&nbsp;0.000244 |
| `FIR` | `fir` | reference does not run | the reference's convolve recurses into itself in both branches and never terminates, so FIR raises RecursionError for every input; its frequency-domain branch also builds a symmetric kernel and then discards it, convolving with the raw samples either way — `RecursionError` |
| `IIR` | `iir` | exact | sample-exact agreement (the reference multiplies its coefficients elementwise, so it requires arrays where the package accepts lists) |
| `R` | `reverb` | reference does not run | the reference reads an undefined decay1 where its own signature declares decay, so R raises NameError on every call and has never run — `NameError` |
| `mix2` | `mix2` | exact | sample-exact agreement (compared without an offset: the reference's offset branch zero-pads by the whole offset sequence rather than by each vector's own offset, and raises for any offset given) |
| `rhythymToDurations` | `rhythm_to_durations` | exact | sample-exact agreement |
| `Tr` | `WAVEFORM_TRIANGULAR` | divergent | the reference builds the triangle as hstack((ramp, ramp[::-1])), which duplicates the sample at the peak and tops out at 1 - 2/8192 instead of 1; the package reaches full amplitude at the midpoint of its period; max&nbsp;\|Δ\|&nbsp;=&nbsp;0.000244 |
| `S` | `WAVEFORM_SINE` | exact | sample-exact agreement |
| `Q` | `WAVEFORM_SQUARE` | exact | sample-exact agreement |
| `Sa` | `WAVEFORM_SAWTOOTH` | divergent | the reference ramps with linspace(-1, 1, Lt) including the endpoint, so its step is 2/16383 and the table does not tile: the wrap is a jump of 2.0 rather than one step. The package excludes the endpoint; max&nbsp;\|Δ\|&nbsp;=&nbsp;0.000122 |
<!-- register:end -->

## What this does not establish

That the package agrees with the reference implementation, not that either
agrees with the article. `tests/test_article.py` is the other leg: it
checks routines against the article's numbered equations, citing each by
the label its LaTeX source gives it, and `tools/article_coverage.py`
measures how far that has got — **12 of the article's 47 labelled
equations** across `body.tex`, `spectra.tex` and `notesInMusic.tex`.

One row of the register above is settled by it. `loc_`/`localize2` diverges
for four corrections that were argued in comments; the first of them is now
proved. The article fixes the frequency a DFT coefficient stands for,
`f_i = i * f_s / Lambda` in `eq:branco`, and the routine had read it as
twice that. The other three remain arguments, and the frequency-dependent
ITD/IID model the routine implements is in none of the article's sources —
its docstring used to say the calculations were "as described in [1]",
which they are not, and now says what the article does and does not
support.

Nothing here touches the parts of the package that MASS has no counterpart
for: `music.stimulation`, `music.singing`, `music.structures`, the
sequencer, or the twenty-odd exported routines with no reference to
reconcile against. See `ASSESSMENT.md` for what those do and do not claim.
