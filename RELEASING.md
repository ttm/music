# Releasing

## Cut it

```console
python tools/release.py            # check and build; changes nothing
python tools/release.py publish    # upload, tag, release
```

`check` refuses to go on unless the version in `pyproject.toml`,
`CITATION.cff` and `CHANGELOG.md` agree, master is clean and in sync with
origin, the tag does not exist, PyPI does not already have that version,
lint, types, tests and docs all pass, and `ASSESSMENT.md` still describes
the package it ships with. Then it builds from scratch and runs
`twine check`.

That last one is there because the file went stale four times in two days
when it depended on someone remembering, once with the wrong test count
already committed during a release. `python tools/assessment_figures.py
--write` corrects it.

`publish` re-runs all of that and then does the three things that cannot be
taken back: uploads to PyPI, which never releases a version number back; tags
**the commit the artifacts were built from**; and creates the GitHub release,
which is what triggers Zenodo.

So before either, bump `version` in `pyproject.toml` and in `CITATION.cff`,
set `date-released` in `CITATION.cff`, and move the changelog's entries under
the new heading. The script will tell you if you missed one.

Uploading needs a PyPI token in `~/.pypirc`; the GitHub release needs `gh`
logged in.

### If the gate stops at `twine check`

On Apple Silicon the gate can pass lint, types, tests and docs and then fail
with

```
ImportError: dlopen(.../nh3/nh3.abi3.so, 0x0002):
  (mach-o file, but is an incompatible architecture
   (have 'x86_64', need 'arm64'))
```

`nh3` is a compiled extension that `twine check` reaches through
`readme_renderer`, and the copy on the path was built for the other
architecture -- usually because it was installed from a shell running under
Rosetta, which is easy to do without noticing. The interpreter is fine and
so is the package; only that one wheel is wrong.

Confirm it before reinstalling anything. The error already names the file,
so ask it and the interpreter what each one is:

```console
file /the/path/from/the/error/nh3.abi3.so
python3 -c 'import platform; print(platform.machine())'
```

If those two disagree, force the right wheel in, from a shell of the
architecture the interpreter reports:

```console
arch -arm64 python3 -m pip install --force-reinstall --no-cache-dir nh3
```

This costs nothing but time. `twine check` runs inside the gate, which is
the part of `release.py` that changes nothing: it happens before the upload,
the tag and the GitHub release, so a failure here has published nothing and
can simply be fixed and rerun.

The release body is generated, not written: the `pip install` line, links to
the tutorial, API reference and changelog, the changelog's own section, and a
sponsorship footer under it. None of that is typed at release time, and the
footer lives in `release_notes()` in `tools/release.py`, so it cannot go out
on one release and be forgotten on the next. Change it there, not in the
GitHub release editor, or the next release will not carry the change.

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
insisting on an exact match rather than accepting the nearest suggestion,
attaches the changelog entry for that version as an additional description,
then edits, updates and republishes the record. **The DOI does not change.**
By default it targets the newest version under the concept DOI; `--record ID`
overrides that, `--no-release-notes` leaves the changelog off, and
`--description` also replaces the record's own description with the abstract
from `.zenodo.json`.

`--write` needs a token with the `deposit:write` and `deposit:actions` scopes,
from https://zenodo.org/account/settings/applications/tokens/new/, in
`ZENODO_TOKEN`. Reading needs none.

Then update the version DOI in `CITATION.cff` to the one Zenodo has just
minted, and commit it. The concept DOI never changes, but the version DOI
names a specific archive, and it can only be known after the release — which
is why it is a step here rather than something `tools/release.py` can check.

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
