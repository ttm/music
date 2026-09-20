# Discrepancies

*Run on **2026-09-05** against [ttm/mass](https://github.com/ttm/mass) at
`e516b08`. Every entry below has a test that re-checks it against
`music` 1.8.1 on every push.*

Three things claim to describe the same synthesis: the **article**
(`doc/body.tex`, `doc/spectra.tex` and `doc/notesInMusic.tex` in ttm/mass,
typeset as `article.pdf`), the **reference implementation**
(`src/aux/functions.py` in the same repository), and this **package**. They
do not all agree. This file is where the disagreements are written down.

Each entry says what the three say, which this package follows, and where
the check lives. Nothing here is a suspicion: every one was found by
running something, and every one has a test that fails if it stops being
true.

- `RECONCILIATION.md` compares the package with the reference, routine by
  routine. Where those two differ for a stated reason, the reason is there
  and the entry here says only what the *article* adds.
- `tests/test_article.py` compares the package with the article's numbered
  equations; `tools/article_coverage.py` measures how far that has got.
- `tools/assessment_figures.py` checks the version stamped above, so this
  file cannot go on naming a release the package has left behind.

## The article and the code disagree

> **A patch for the three typographic ones** is in `errata-proposal/` in
> the `ttm/mass` checkout: `mass-errata.patch`, against `e516b08`, with a
> note explaining each. Nothing is applied — whether the paper wants an
> erratum, a quiet fix to the LaTeX, or neither is a judgment about the
> paper rather than about this package.

### `eq:reconsCompleta` — the phase sign

`spectra.tex` writes the reconstruction of a real signal as

> t_i = a_0/Λ + a_(Λ/2)/Λ (1 − Λ%2) + (2/Λ) Σ √(a_k² + b_k²) cos[ω_k i − arctan(b_k, a_k)]

The phase should be **plus** `arctan(b_k, a_k)`. The line immediately above
it in the same source gives the sum as `a_k cos(ω_k i) − b_k sin(ω_k i)`,
and `a cos x − b sin x = R cos(x + arctan2(b, a))`: matching
`R cos φ = a` against `R sin φ = b` fixes the sign. With the minus, a known
spectrum does not come back.

All four readings were checked on a known spectrum, since the article's own
gloss names the arguments `arctan(x, y)` and so admits more than one:

| phase term | reconstruction error |
|---|---|
| `− arctan(b_k, a_k)`, as typeset | 1.2e-01 |
| **`+ arctan(b_k, a_k)`** | **5.8e-15** |
| `− arctan(a_k, b_k)` | 1.4e-01 |
| `+ arctan(a_k, b_k)` | 1.4e-01 |

Only the plus reconstructs. Separately, and not a defect in the equation:
that gloss names its arguments in the opposite order to the way the
equation uses them, since `arctan(b_k, a_k)` must mean the angle of the
point `(a_k, b_k)`.

**This package follows the derivation, not the typesetting.**
`tests/test_article.py::test_a_real_signal_is_the_cosine_sum_equation_reconscompleta_writes`
reconstructs a known spectrum both ways and asserts the corrected form is
exact while the typeset one is not.

### `eq:reconsCompleta` — the Nyquist term

The same equation gives the Nyquist contribution as the constant
`a_(Λ/2)/Λ (1 − Λ%2)`. That coefficient stands for the sequence alternating
at half the sample rate, so its contribution is
**`a_(Λ/2) cos(π i) / Λ`** — it changes sign every sample. The
`(1 − Λ%2)` factor is right and does its job: for odd Λ there is no Nyquist
bin and the term vanishes.

With both corrections the reconstruction is exact to 6e-15. With the
equation as typeset it is out by 4e-3 on the spectrum the test uses.

### `eq:passa-banda` and `eq:rejeita-banda` — what `bw` measures

The prose says "In both frequencies `f_c ± bw` there is an attenuation of
−3dB". The coefficients the same section gives put those points at
**`f_c ± bw/2`**: measured across every bandwidth tried, the half-width of
the pass band is `bw/2` to within 2 %.

**This package treats `bandwidth` as the full width between the two 3 dB
points**, which is what the coefficients do, and says so in
`music.band_pass`. `tests/test_filter_design.py` measures it.

### `eq:passa-baixas` and `eq:passa-altas` — where the cutoff is 3 dB down

The article defines `f_c` as "where the filter performs an attenuation of
−3dB ≈ 0.707", without qualification. `x = exp(−2π f_c)` is the sampled
form of an analogue one-pole and holds that only while `f_c` is small
against the sample rate:

| `f_c` | low pass | high pass |
|---|---|---|
| 0.01 | 0.7072 | 0.7073 |
| 0.05 | 0.7100 | 0.7129 |
| 0.10 | 0.7186 | 0.7300 |
| 0.25 | 0.7755 | 0.8362 |

At `f_c = 0.25` the low pass is 2 dB down rather than 3. This is what a
one-pole design does rather than an error in the coefficients, but the
article does not say so. `tests/test_filter_design.py` pins both the
accuracy at the bottom of the range and the monotone drift towards Nyquist.

### `eq:adsr` — where the release lands

The article's release is `a_S (ξ/a_S)^t`, which reaches **ξ** at the end of
the envelope. The package multiplies a fade by `a_S`, so it reaches
**ξ·a_S**; the MASS reference does exactly the same, to the sample. Package
and reference agree; both differ from the paper.

The two curves part by exactly the sustain level — 6 dB at `sustain_level =
-6`, more at a deeper sustain — over the course of the release. Both end
inaudible, so nothing here is broken; the question is which of the author's
two artifacts is the specification.

**This package follows the reference**, because `AD` is a sample-exact row
of `RECONCILIATION.md` and changing it would break that on a reading of the
paper rather than on a decision by its author. Attack, decay and sustain
match `eq:adsr` exactly.
`tests/test_article.py::test_the_adsr_envelope_is_the_four_pieces_equation_adsr_writes`
asserts the implemented form and asserts the article's form is *not* what
comes out, so this entry cannot go stale in either direction.

Separately, `eq:adsr` has no linear approach to zero: it begins at `ξ` and
ends at `ξ`, and the straight millisecond the package puts at each end to
reach silence — its `to_zero` — is the package's own. That test therefore
passes `to_zero=0`, which is the article's envelope exactly. The reference
has the parameter too but passes it on a hundred times too small, so it
never took effect; `RECONCILIATION.md` records that under `AD` and `ADS`.

### `alpha` distorts a tremolo the article never gives one for

`eq:trT` writes the tremolo amplitude as `a_i = 10^(V_dB/20 · t'_i)`, with
no distortion index anywhere in it. The article's `alpha` belongs to the
*transitions*, where it is applied to `(i/(Λ-1))^α` — a ramp that runs from
0 to 1 and is never negative. Both this package and the MASS reference
carry an `alpha` on the tremolo as well, applied to the signed oscillation
`t'_i · V_dB/20`.

Raised directly, as the reference does, that has no real value wherever the
waveform is negative, which is half of every cycle:

| `alpha` | as the reference raises it | sign-preserving |
|---|---|---|
| 1 | the tremolo | identical |
| 0.5 | **NaN on 499 of 1000 samples** | 0.196 … 5.09 |
| 1.5 | **NaN on 499 of 1000 samples** | 0.443 … 2.26 |
| 2 | 1.000 … 1.778 — it only ever boosts | 0.562 … 1.778 |
| 3, 9 | the distorted tremolo | identical |

An even `alpha` rectifies the pattern, so a "tremolo" that cannot fall
below unity; a fractional one is not a real number at all and arrived as
NaN behind a NumPy warning, which
[`tests/test_degenerate.py`](tests/test_degenerate.py) states as the one
thing no routine here may return.

**This package applies `alpha` to the magnitude and keeps the sign**:
`sign(x)·|x|^α`. `alpha=1` and every odd whole `alpha` are bit-identical to
the reference, since `sign(x)|x|³` is `x³`, so the `T` and `T_` rows of
`RECONCILIATION.md` are still sample-exact — every case there uses
`alpha=1`. Only the readings that were NaN or rectified have changed.
`tests/test_envelopes.py` checks that a distorted tremolo is finite, that
it still cuts as well as boosts, and that an odd index is unchanged.

### `localize2` implements a model the article does not give

The frequency-dependent ITD and IID in `music.localize2` — a crossover at
4 kHz between two delay coefficients, and a head shadow growing as
`1 + (f/1000)**0.8` scaled by `sin|θ|` — appear in **none** of the article's
sources. The article gives the geometric ITD and IID that `music.localize`
implements (`eq:dti`, `eq:dii`) and one sentence saying low frequencies
diffract and reach the far ear later.

The routine's docstring used to say its calculations were "as described in
[1]". It now says what the article does and does not support. Treat the
refinement as a rule of thumb; a full treatment needs an HRTF, which
nothing here has.

### `eq:serieHarmonica` — the sixth partial

The article tabulates the first twenty partials in semitones. Nineteen of
them are `12 log2(n)` to within the two decimals they are printed at. The
sixth is printed as **`31 + 0.2`** where the exact value is **31.02**,
which `31 + 0.02` would give — and the same `+0.02` appears at the third
partial, an octave below it. A typo rather than a different claim.

**This package computes `12 log2(n)`.**
`music.theory.scales.HARMONIC_SERIES_AS_PRINTED` keeps the table as the
paper prints it so the two can be compared, and
`tests/test_theory.py::test_the_printed_table_is_the_computed_series_but_for_one_digit`
asserts that the sixth is the only one that differs — so if this is ever
corrected upstream, the test says so.

## The reference implementation and the article disagree

These are defects in `src/aux/functions.py` that the article's own
equations settle. `RECONCILIATION.md` carries the full register; this is
the subset where the article is the authority rather than general
correctness.

| Reference | What the article says | What the reference does |
|---|---|---|
| `loc_` | `f_i = i·f_s/Λ` (`eq:branco`) | reads the bin spacing as `2·f_s/Λ`, so every coefficient stands for twice its frequency |
| `noises` | `c_i = e^{jx} α_i`, a complex coefficient (`eq:rosa`) | builds the coefficient array with a real dtype, discarding the imaginary part of every randomised phase |
| `Tr` | a triangle reaching full amplitude | `hstack((ramp, ramp[::-1]))`, which duplicates the peak and tops out at `1 − 2/8192` |
| `Sa` | a table that tiles | `linspace(-1, 1, Lt)` including the endpoint, so the wrap is a jump of 2.0 |
| `readHRTF` | the KEMAR responses, read from disk | reads them with a `data_type` that is defined in neither `HRTF.py` nor `functions.py`, so it raises `NameError` on every call and has never run |

## What the article states that this package does not implement

Not disagreements: scope. `tools/article_coverage.py` lists these as `[-]`
and reports them separately from what is merely unchecked, so that "100 %
of the equations" is never mistaken for the target.

No labelled, testable equation remains unimplemented. `eq:intervalos`, the
interval nomenclature, is implemented by `music.theory.intervals` and
checked in `tests/test_theory.py`. The four IIR designs of `body.tex` were
on this list until `music.core.filters.design` implemented them, and the
scales, minor scales and harmonic series of `notesInMusic.tex` were until
`music.theory` did. `eq:vinculos` remains outside the set of testable
equations for the reason below.

## What no test could settle

- **`eq:vinculos`** is a schema rather than a formula: it says a vibrato
  rate may be a function of the note frequency, without fixing which
  function. It describes a way of composing, not a routine.
