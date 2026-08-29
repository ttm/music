#!/usr/bin/env python3
"""Check, build and publish a release.

The steps are not hard, but they are easy to do in the wrong order or to
half-do: the version lives in three files that must agree, the tag has to
point at the commit the artifacts were built from rather than wherever
master has since moved to, and PyPI will not let a version number be
reused if any of it goes wrong.

So ``check`` refuses to proceed on any disagreement, and ``publish``
re-checks before it touches anything outside this machine.

Usage
-----
::

    python tools/release.py            # check and build; changes nothing
    python tools/release.py publish    # upload, tag, release

``publish`` is not reversible. It uploads to PyPI, which never releases a
version number back, pushes a tag, and creates a GitHub release, which is
what triggers Zenodo.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys
import urllib.error
import urllib.request

# Running this as a script puts tools/ first on sys.path, so its sibling
# is importable by name. The CA handling belongs in one place.
from zenodo_sync import ssl_context

ROOT = pathlib.Path(__file__).parent.parent


class ReleaseError(RuntimeError):
    """A problem the caller should read, not a traceback."""


def run(*command, capture=True, check=True):
    """Run a command in the repository root."""
    result = subprocess.run(command, cwd=ROOT, check=False,
                            capture_output=capture, text=True)
    if check and result.returncode:
        output = (result.stderr or result.stdout or "").strip()
        raise ReleaseError(f"{' '.join(command)} failed:\n{output}")
    return (result.stdout or "").strip()


def step(message):
    print(f"  {message}")


# ---------------------------------------------------------------------
# What the version is, and whether everything agrees about it
# ---------------------------------------------------------------------

def declared_version():
    """The version in pyproject.toml, which is the one that counts."""
    text = (ROOT / "pyproject.toml").read_text()
    match = re.search(r"^version = '([^']+)'", text, re.MULTILINE)
    if not match:
        raise ReleaseError("no version found in pyproject.toml")
    return match.group(1)


def check_versions_agree(version):
    """The same version must appear in every file that states one."""
    citation = (ROOT / "CITATION.cff").read_text()
    if f"version: {version}\n" not in citation:
        raise ReleaseError(
            f"CITATION.cff does not say version: {version}")

    changelog = (ROOT / "CHANGELOG.md").read_text()
    if not re.search(rf"^## \[{re.escape(version)}\]", changelog,
                     re.MULTILINE):
        raise ReleaseError(
            f"CHANGELOG.md has no section for {version}")
    step(f"pyproject, CITATION.cff and CHANGELOG all say {version}")


def check_repository_state(version):
    """A release comes off a clean, pushed master."""
    branch = run("git", "rev-parse", "--abbrev-ref", "HEAD")
    if branch != "master":
        raise ReleaseError(f"on branch {branch}, not master")
    if run("git", "status", "--porcelain"):
        raise ReleaseError("the working tree has uncommitted changes")
    run("git", "fetch", "--quiet", "origin")
    if run("git", "rev-parse", "HEAD") != run("git", "rev-parse",
                                              "origin/master"):
        raise ReleaseError("master and origin/master have diverged")
    if run("git", "tag", "-l", f"v{version}"):
        raise ReleaseError(f"tag v{version} already exists")
    step(f"on master, clean, in sync with origin, v{version} untagged")


def check_not_on_pypi(version):
    """PyPI never releases a version number back, so ask first."""
    url = f"https://pypi.org/pypi/music/{version}/json"
    try:
        urllib.request.urlopen(url, timeout=30,
                               context=ssl_context()).read()
    except urllib.error.HTTPError as error:
        if error.code == 404:
            step(f"PyPI does not have {version} yet")
            return
        raise ReleaseError(f"asking PyPI about {version}: {error}") from None
    except urllib.error.URLError as error:
        raise ReleaseError(f"cannot reach PyPI: {error.reason}") from None
    raise ReleaseError(f"PyPI already has music {version}")


# ---------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------

def run_gate():
    """Everything CI runs, before anything leaves the machine."""
    checks = [
        ("lint", ("ruff", "check", "music", "tests", "examples", "tools",
                  "conftest.py")),
        ("types", (sys.executable, "-m", "mypy", "music")),
        ("tests", (sys.executable, "-m", "pytest", "-q", "--cov=music",
                   "--cov-fail-under=100")),
        ("docs", (sys.executable, "-m", "sphinx", "-b", "html", "-W",
                  "docs", "docs/_build/html")),
    ]
    for name, command in checks:
        run(*command)
        step(f"{name} passed")


def build():
    """Build the artifacts, from nothing, and validate them."""
    run("rm", "-rf", "dist", "build")
    run(sys.executable, "-m", "build")
    run(sys.executable, "-m", "twine", "check",
        *[str(path) for path in sorted((ROOT / "dist").iterdir())])
    names = sorted(path.name for path in (ROOT / "dist").iterdir())
    step(f"built and validated {', '.join(names)}")
    return names


# ---------------------------------------------------------------------
# Publishing
# ---------------------------------------------------------------------

def release_notes(version):
    """The changelog's section for this version, as the release body."""
    changelog = (ROOT / "CHANGELOG.md").read_text()
    match = re.search(
        rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[)",
        changelog, re.MULTILINE | re.DOTALL)
    if not match:
        raise ReleaseError(f"no changelog section for {version}")
    body = match.group(1).strip()
    return (
        f"```console\npip install -U music\n```\n\n"
        f"[Tutorial](https://ttm.github.io/music/tutorial.html) · "
        f"[API reference](https://ttm.github.io/music/) · "
        f"[Changelog]"
        f"(https://github.com/ttm/music/blob/master/CHANGELOG.md)\n\n"
        f"---\n\n{body}\n"
    )


def publish(version):
    """Upload, tag, release. None of this can be taken back."""
    commit = run("git", "rev-parse", "HEAD")

    step(f"uploading dist/* to PyPI as {version}")
    run(sys.executable, "-m", "twine", "upload",
        *[str(path) for path in sorted((ROOT / "dist").iterdir())])

    step(f"tagging {commit[:7]} as v{version}")
    run("git", "tag", "-a", f"v{version}", commit,
        "-m", f"music {version}\n\n"
              f"Built from {commit[:7]}. See CHANGELOG.md.")
    run("git", "push", "--quiet", "origin", f"v{version}")

    notes = ROOT / "dist" / "release-notes.md"
    notes.write_text(release_notes(version))
    step("creating the GitHub release, which is what triggers Zenodo")
    run("gh", "release", "create", f"v{version}", "--verify-tag",
        "--title", f"music {version}", "--notes-file", str(notes))

    print()
    step(f"https://pypi.org/project/music/{version}/")
    step(f"https://github.com/ttm/music/releases/tag/v{version}")
    print("\nZenodo mints the DOI within a minute or two. Then:")
    print("    python tools/zenodo_sync.py            # see what it sends")
    print("    python tools/zenodo_sync.py --write    # apply it")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", nargs="?", default="check",
                        choices=["check", "publish"])
    parser.add_argument("--skip-gate", action="store_true",
                        help="do not re-run lint, types, tests and docs")
    args = parser.parse_args(argv)

    version = declared_version()
    print(f"music {version}\n")

    check_versions_agree(version)
    check_repository_state(version)
    check_not_on_pypi(version)
    if not args.skip_gate:
        run_gate()
    build()

    if args.action == "check":
        print(f"\nready. `python {pathlib.Path(__file__).name} publish` "
              f"uploads it, and that cannot be undone.")
        return 0

    print()
    publish(version)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ReleaseError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
