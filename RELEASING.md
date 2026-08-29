# Releasing

## Before

1. `ruff check music tests examples conftest.py`
2. `mypy music`
3. `pytest` — must report 100% coverage; it is configured to fail below it
4. `sphinx-build -b html -W docs docs/_build/html`
5. Bump `version` in `pyproject.toml` **and** in `CITATION.cff`, and set
   `date-released` in `CITATION.cff` to the release date
6. Move the changelog's unreleased entries under the new version heading

## Publish

```console
rm -rf dist build
python -m build
twine check dist/*
twine upload dist/*
```

Then tag and release. **Tag the commit the artifacts were built from**, which
is not necessarily the tip of `master`:

```console
git tag -a vX.Y.Z <commit> -m "music X.Y.Z"
git push origin vX.Y.Z
gh release create vX.Y.Z --verify-tag --notes-file <notes>
```

Publishing the GitHub release is what triggers Zenodo. It archives the
tarball and mints a version DOI under the existing concept DOI
[10.5281/zenodo.22151793](https://doi.org/10.5281/zenodo.22151793).

## After: top up the Zenodo record

Zenodo builds each new deposit from `.zenodo.json` in the repository, so the
title, the author, Jacopo Donati as a contributor, the description and the
free-text keywords all come across automatically. **One thing does not.**

Zenodo's *controlled-vocabulary* subjects — the MeSH, GEMET and EuroSciVoc
terms, which appear in its interface prefixed with the vocabulary name — have
no representation in `.zenodo.json`, whose `keywords` field is a plain list of
strings. Writing `"(MeSH) Music"` there produces a free-text keyword that
merely reads like a MeSH term; it is not linked to the vocabulary, and
aggregators will not treat it as one.

So after each release, open the new record, click **Edit**, and re-add them
under *Keywords and subjects* by typing the term and picking the suggestion
with the vocabulary prefix:

| Vocabulary | Terms |
|---|---|
| MeSH | Music · Psychoacoustics · Signal Processing, Computer-Assisted |
| GEMET | Music |
| EuroSciVoc | Signal processing |

Then **Publish**. The DOI does not change.

Two things to check while you are in there, both of which the interface makes
easy to get wrong:

- **Names are entered family-name-first.** A contributor typed as
  `Jacopo, Donati` has "Jacopo" recorded as the family name.
- **Edits are not live until you press Publish.** A record can sit with a
  draft full of changes while the public page still shows the old metadata.

## Not part of the release

Music therapy terms. The package is not a music-therapy tool, and tagging it
as one surfaces it in searches it cannot serve. If the sensory-stimulation
work in the issue tracker becomes real, that is when the terms are earned.
