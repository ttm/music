# Contributing

Thank you for looking. This is a small project with a specific idea of what
"working" means, and this file is about that idea more than about process.

The [README's Contributing section](README.md#contributing) has the commands:
which checks exist, what runs in CI, and what runs at release time. Read that
first. What follows is the part that is harder to guess.

## What a change has to survive

Six checks run in CI on Python 3.10 through 3.14, for every push and every
pull request. Four of them look at the package, one looks at the
documentation, and one looks at a caller:

```console
pytest                    # tests, doctests, and 100% line coverage
mypy music                # types, with function bodies inspected
ruff check ...            # lint, at PEP 8's 79 columns
sphinx-build -W ...       # docs, with warnings as errors
python tools/run_examples.py         # every example in examples/
python tools/assessment_figures.py   # the figures in the docs vs the package
```

Coverage is at 100% and the gate fails below it. That is not a claim that
every routine is correct — it is a claim that no line is unreached, which is
a much smaller thing, and this project has found most of its real defects by
asking the larger question instead.

## The larger question

**A test that checks the shape of the output is worth little here.** The
package's claim is fidelity to a published framework, so the tests that
matter assert the *samples* against the equation the routine documents.

That is not an aspiration; it is the existing practice, and there is a lot of
it to read before writing more:

- [`RECONCILIATION.md`](RECONCILIATION.md) compares every routine with the
  MASS reference implementation, sample for sample.
- [`tests/test_article.py`](tests/test_article.py) checks routines against the
  article's numbered equations, citing each by its LaTeX label.
- [`DISCREPANCIES.md`](DISCREPANCIES.md) records where the article, the
  reference and this package disagree, and which one the package follows.

If you add a routine that implements something the article states, cite the
equation and check the samples against it. `tools/article_coverage.py` will
tell you what is already covered.

## Examples are tests

`--doctest-modules` is on, so every `>>>` in a docstring is executed. An
example that does not run fails the build.

This matters more than it sounds. Six of the thirty examples added most
recently were wrong when first written, because they were guesses about what a
routine returned — and every guess was caught the moment it ran. Write the
example, then run it, then believe it.

`tests/test_docstring_references.py` additionally fails on an example that
does not parse, that names something the package does not export, or that
passes a parameter the signature does not have.

## Reporting something that sounds wrong

This is the most useful kind of issue and the hardest to write. What helps:

- The **code that produced it**, short enough to run.
- What you **expected**, and where that expectation comes from — an equation,
  a section of the article, another implementation, or your ears.
- What you **got**: a number, a shape, a spectrum, a described sound.

"It sounds wrong" is a legitimate report and has led to real fixes here. It is
just much easier to act on with a frequency, a duration and a routine name
attached.

## Documents that measure themselves

`ASSESSMENT.md` records what the package does *not* do. It is meant to be
uncomfortable and it is kept current by `tools/assessment_figures.py`, which
fails when a number in it — or in the README — disagrees with the package.

If your change moves one of those numbers, run:

```console
python tools/assessment_figures.py --write
```

If it closes a known limitation, move the entry to **No longer true** rather
than deleting it. The record reading worse than the code is a smaller problem
than the record being wrong.

## Style

- Docstrings are [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html),
  and `sphinx-build -W` fails on ones it cannot parse.
- Code follows [PEP 8](https://peps.python.org/pep-0008/), 79 columns.
- Comments explain *why*, not *what*. A comment that restates the line above it
  is noise; a comment recording why an obvious approach was rejected is the
  most valuable thing in the file.
- New work does not go in `legacy/`.

## Releasing

Maintainers only, and [`RELEASING.md`](RELEASING.md) is the procedure. The
gate refuses to publish unless the version agrees across three files, the
working tree is clean and in sync, every check above passes, `ASSESSMENT.md`
still describes the package it ships with, and the source distribution passes
its own tests from inside itself.

## Code of conduct

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).
