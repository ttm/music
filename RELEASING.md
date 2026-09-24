# Releasing

## Cut it

```console
python tools/release.py            # check and build; publish nothing
python tools/release.py publish    # upload, tag, release
python tools/release.py verify     # verify/build during development too
```

`check` refuses to go on unless the version in `pyproject.toml`,
`CITATION.cff` and `CHANGELOG.md` agree, every entry in the version's
changelog section opens with a bold headline, master is clean and in sync with
origin, the tag does not exist, PyPI does not already have that version,
lint, types, tests, docs and examples all pass, `ASSESSMENT.md` still
describes the package it ships with, and the source distribution passes
its own tests from inside itself. The article, live MASS reference and
archival subject checks run too. Then it builds from scratch, runs
`twine check`, and smoke-tests the installed wheel outside the checkout.

Install the release tools and optional dependencies used by these checks:

```console
python -m pip install -e '.[dev,docs,plot]' build twine
python tools/release.py verify --mass ../mass
```

`verify` runs the same gate and build checks on a working tree, including
an already released version. It does not require a clean/pushed branch,
an unused tag or an available PyPI version, and it cannot publish. Use it
while changing release tooling; `check` and `publish` retain all release
readiness requirements.

### External checks

`--mass PATH` accepts a MASS checkout or its `src/aux/functions.py`.
Otherwise the tools use `MASS_SRC`, then the conventional `../mass`,
`~/repos/mass` and `~/rep/mass` locations. An explicitly selected missing
path fails; it does not silently select another checkout. The gate
resolves the reference once and passes the same file to both checks:

- `article_coverage.py --strict` requires all three article sources and
  a test citation for every implemented, testable equation. The documented
  schema exception remains visible in its report.
- `mass_reconcile.py` compares the running reference with the package and
  rejects any outcome that disagrees with its register. Registered
  divergences and broken reference routines remain visible. Release
  verification does not rewrite fixtures or the register.
- `verify_subjects.py --strict` needs the network. Wrong labels and
  unavailable lookups fail. Only the two explicitly registered EuroSciVoc
  subjects may remain unconfirmed; they are printed as exceptions, not
  counted as verified. An additional unresolved subject fails the gate.

A missing checkout, or a vocabulary service still unavailable after
three tries, stops verification.
The successful reports are printed so these limits remain visible.

`--skip-gate` is an explicit bypass for reuse of a previously completed
gate. It skips **all** source and external checks, including examples,
article/MASS comparisons and subject lookups, and prints that verification
is incomplete. It still checks release readiness for `check`/`publish`,
rebuilds the artifacts, runs `twine check`, and tests the installed wheel.
It is not an automatic fallback for failed checks.

### Installed-wheel check

```console
python tools/check_wheel.py --wheel dist/music-1.8.1-py3-none-any.whl
```

The checker installs the selected wheel without dependencies or network
access into a temporary directory. A separate Python process outside the
checkout verifies that imports and distribution metadata came from that
installation, checks its version and typing marker, then exercises
normalization, mono/stereo WAV/FLAC fades at a nondefault rate, and short
session transitions. Dependencies come from the current environment.
The same check runs in CI on Linux, macOS and Windows with Python 3.12.
Each job builds and installs the wheel with its runtime dependencies
before checking a separate temporary installation outside the checkout.

### Source distribution and assessment

The source-distribution check is `tools/check_sdist.py`. The sdist ships
`tests/`, but testing the working tree does not establish that those tests
can run from the tarball. Up to and including 1.5.0 it shipped
thirty-eight test files without `conftest.py`, `pytest.ini`, `tools/`,
`docs/` or the fixture, so none of them collected and nothing said so.
`MANIFEST.in` is what it checks.

It clears `music.egg-info` before building, and so does `build`. setuptools
reuses the `SOURCES.txt` it left there rather than re-reading
`MANIFEST.in`, so a file dropped from the manifest goes on being shipped
and a build "from scratch" that keeps the egg-info is not from scratch.

The assessment check catches stale figures. They went stale four times in
two days when updating them depended on someone remembering, once with the
wrong test count already committed during a release.
`python tools/assessment_figures.py --write` corrects them.

### Prepare the release

`publish` re-runs all of that and then does the three things that cannot be
taken back: uploads to PyPI, which never releases a version number back; tags
**the commit the artifacts were built from**; and creates the GitHub release,
which is what triggers Zenodo.

Before `check` or `publish`, bump `version` in `pyproject.toml` and in
`CITATION.cff`,
set `date-released` in `CITATION.cff`, and move the changelog's entries under
the new heading. Each entry opens with a bold headline, `- **Like this.**`:
the Zenodo record summarises a release by those headlines alone, so an
entry without one would be missing from it. A test checks the newest
section on every push, and the gate checks the release's. Then run `python tools/assessment_figures.py --write`,
which carries the bump into the version stamped at the top of
`ASSESSMENT.md`, `RECONCILIATION.md` and `DISCREPANCIES.md` and re-measures
the figures while it is there. The script will tell you if you missed one:
the gate runs that check too, so a release cannot ship documents naming the
version before it. All three named 1.4.0 while the package was 1.7.0, which
is what added the check.

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

## Metadata scope

The sensory-stimulation toolkit now exists, and `.zenodo.json` includes
Music Therapy subjects. These describe the intended audience and subject
area, not evidence of clinical efficacy. The tests establish properties
of rendered signals; `tools/verify_subjects.py` checks vocabulary labels
and identifiers. Neither establishes a therapeutic effect. Keep metadata
claims aligned with the implemented capabilities and `ASSESSMENT.md`.
