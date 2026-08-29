# Releasing

## Cut it

```console
python tools/release.py            # check and build; changes nothing
python tools/release.py publish    # upload, tag, release
```

`check` refuses to go on unless the version in `pyproject.toml`,
`CITATION.cff` and `CHANGELOG.md` agree, master is clean and in sync with
origin, the tag does not exist, PyPI does not already have that version, and
lint, types, tests and docs all pass. Then it builds from scratch and runs
`twine check`.

`publish` re-runs all of that and then does the three things that cannot be
taken back: uploads to PyPI, which never releases a version number back; tags
**the commit the artifacts were built from**; and creates the GitHub release,
which is what triggers Zenodo.

So before either, bump `version` in `pyproject.toml` and in `CITATION.cff`,
set `date-released` in `CITATION.cff`, and move the changelog's entries under
the new heading. The script will tell you if you missed one.

Uploading needs a PyPI token in `~/.pypirc`; the GitHub release needs `gh`
logged in.

## After: sync the Zenodo record

Zenodo builds each new deposit from `.zenodo.json` when the release is
published, and gets most of it right: at 1.1.1 the title, the author, Jacopo
Donati as a contributor, the description and all nineteen free-text keywords
came across untouched.

It also honours `related_identifiers`, which arrived complete with their
relation types.

**It ignores the `subjects` block entirely.** The controlled-vocabulary terms
-- the MeSH, GEMET and EuroSciVoc entries -- arrived as zero linked subjects,
measured on the 1.1.1 deposit. This is not a maybe; the release step below is
required, not a precaution.

So push the file onto the record after every release:

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
