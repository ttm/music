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

## After: check the Zenodo record

Zenodo builds each new deposit from `.zenodo.json`, so the title, the author,
Jacopo Donati as a contributor, the description, the keywords and the
controlled-vocabulary subjects should all come across without anyone touching
the interface.

The `subjects` block is the part to verify, because it is the part that could
silently do nothing: Zenodo's legacy metadata schema accepts
`{term, scheme, identifier}` entries, but whether its GitHub ingestion honours
them was established at 1.1.1 and not before. Check with:

```console
curl -sL https://zenodo.org/records/<id>/export/datacite-json \
  | python3 -c 'import json,sys; [print(s) for s in json.load(sys.stdin)["subjects"]]'
```

Entries carrying a `subjectScheme` are linked to their vocabulary; entries
without one are free text. If the MeSH, GEMET and EuroSciVoc terms come back
without a scheme, or not at all, the ingestion ignored the block, and they
have to be re-added by hand: open the record, **Edit**, type the term under
*Keywords and subjects*, and pick the suggestion carrying the vocabulary
prefix. Then **Publish**; the DOI does not change.

Two traps in that interface, both of which have caught us:

- **Names are entered family-name-first.** A contributor typed as
  `Jacopo, Donati` has "Jacopo" recorded as the family name.
- **Edits are not live until Publish is pressed.** A record can sit with a
  draft full of changes while the public page still shows the old metadata.

One quirk worth knowing when checking: Zenodo's own
`/api/records/<id>` endpoint omits `subjects` entirely, so a record can look
unkeyworded there while the DataCite export and the web page both show the
full set. Trust the export.

## Not part of the release

Music therapy terms. The package is not a music-therapy tool, and tagging it
as one surfaces it in searches it cannot serve. If the sensory-stimulation
work in the issue tracker becomes real, that is when the terms are earned.
