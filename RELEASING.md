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

## After: sync the Zenodo record

Zenodo builds each new deposit from `.zenodo.json` when the release is
published, but do not rely on that alone: whether its ingestion honours the
`subjects` block -- the controlled-vocabulary terms, as opposed to the
free-text keywords -- is not something we control, and a record corrected by
hand drifts from the file the moment anyone edits it.

Push the file onto the record instead:

```console
python tools/zenodo_sync.py            # show what it would send
python tools/zenodo_sync.py --write    # apply it
```

It resolves each controlled term to the identifier Zenodo knows it by,
insisting on an exact match rather than accepting the nearest suggestion, then
edits, updates and republishes the record. **The DOI does not change.** By
default it targets the newest version under the concept DOI; `--record ID`
overrides that.

`--write` needs a token with the `deposit:write` and `deposit:actions` scopes,
from https://zenodo.org/account/settings/applications/tokens/new/, in
`ZENODO_TOKEN`. Reading needs none.

Verify afterwards with the DataCite export rather than the record endpoint,
which omits subjects entirely and will make a fully keyworded record look bare:

```console
curl -sL https://zenodo.org/records/<id>/export/datacite-json \
  | python3 -c 'import json,sys; [print(s) for s in json.load(sys.stdin)["subjects"]]'
```

Entries carrying a `subjectScheme` are linked to their vocabulary; entries
without one are free text.

Two traps in Zenodo's web interface, if you edit there instead. Both have
caught us, and both are why the script exists:

- **Names are entered family-name-first.** A contributor typed as
  `Jacopo, Donati` has "Jacopo" recorded as the family name.
- **Edits are not live until Publish is pressed.** A record can sit with a
  draft full of changes while the public page still shows the old metadata.

## Not part of the release

Music therapy terms. The package is not a music-therapy tool, and tagging it
as one surfaces it in searches it cannot serve. If the sensory-stimulation
work in the issue tracker becomes real, that is when the terms are earned.
