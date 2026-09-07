<!-- Thank you. CONTRIBUTING.md has the detail; this is the short version. -->

## What this changes

<!-- And why. If it fixes something, what was wrong. -->

## How you know it works

<!--
A test that asserts the samples rather than their shape is worth more here
than any amount of description. If the routine implements something the MASS
article states, cite the equation.
-->

## Checks

- [ ] `pytest` passes, including the doctests, at 100% coverage
- [ ] `mypy music` and `ruff check music tests examples tools conftest.py` are clean
- [ ] `sphinx-build -b html -W docs docs/_build/html` builds
- [ ] `python tools/run_examples.py` runs every example
- [ ] `python tools/assessment_figures.py` agrees, or `--write` was run
- [ ] `CHANGELOG.md` has an entry under `[Unreleased]`
- [ ] If this closes a known limitation, `ASSESSMENT.md` moved it to **No longer true**
