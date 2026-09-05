# Discrepancies

*Last measured **2026-09-05**, `music` 1.4.0 against
[ttm/mass](https://github.com/ttm/mass) at `e516b08`.*

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

## The article and the code disagree

### `eq:reconsCompleta` — the phase sign

`spectra.tex` writes the reconstruction of a real signal as

> t_i = a_0/Λ + a_(Λ/2)/Λ (1 − Λ%2) + (2/Λ) Σ √(a_k² + b_k²) cos[ω_k i − arctan(b_k, a_k)]

The phase should be **plus** `arctan(b_k, a_k)`. The line immediately above
it in the same source gives the sum as `a_k cos(ω_k i) − b_k sin(ω_k i)`,
and `a cos x − b sin x = R cos(x + arctan2(b, a))`: matching
`R cos φ = a` against `R sin φ = b` fixes the sign. With the minus, a known
spectrum does not come back.

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

## What the article states that this package does not implement

Not disagreements: scope. `tools/article_coverage.py` lists these as `[-]`
and reports them separately from what is merely unchecked, so that "100 %
of the equations" is never mistaken for the target.

What remains outside the package is the scale, interval and harmonic
material of `notesInMusic.tex`: `eq:intervalos`, `eq:escalas`,
`eq:relacaoDia`, `eq:escalasMenores` and `eq:serieHarmonica`. The four IIR
designs of `body.tex` were on this list until `music.core.filters.design`
implemented them. Run the tool for the current list rather than trusting
this paragraph.

## What no test could settle

- **`eq:vinculos`** is a schema rather than a formula: it says a vibrato
  rate may be a function of the note frequency, without fixing which
  function. It describes a way of composing, not a routine.
