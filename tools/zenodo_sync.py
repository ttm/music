#!/usr/bin/env python3
"""Push ``.zenodo.json`` onto a published Zenodo record.

Zenodo builds each deposit from the repository when a release is
published, but two things make that insufficient on its own.  Its
ingestion may or may not honour the ``subjects`` block -- the
controlled-vocabulary terms, as opposed to the free-text keywords -- and
a record whose metadata was corrected by hand has no way to be brought
back in line with the file short of retyping it.

This script closes both gaps.  It reads ``.zenodo.json``, translates it
into the record metadata Zenodo's REST API expects, and writes it to a
published record: edit, update, publish.  The DOI does not change.

Run it after a release, or any time ``.zenodo.json`` changes and the
record should follow.

Usage
-----
::

    python tools/zenodo_sync.py                  # show what would change
    python tools/zenodo_sync.py --write          # apply it

``--write`` needs a personal access token with the ``deposit:write`` and
``deposit:actions`` scopes, created at
https://zenodo.org/account/settings/applications/tokens/new/ and passed
in the ``ZENODO_TOKEN`` environment variable.  Reading needs no token.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request

#: The concept DOI, which always resolves to the newest version.
CONCEPT_RECORD = "22151793"

BASE = "https://zenodo.org/api"

#: Zenodo names contributor roles in lowercase; ``.zenodo.json`` uses the
#: capitalised form its own documentation shows.
ROLE_IDS = {
    "contactperson", "datacollector", "datacurator", "datamanager",
    "distributor", "editor", "hostinginstitution", "producer",
    "projectleader", "projectmanager", "projectmember", "registrationagency",
    "registrationauthority", "relatedperson", "researcher", "researchgroup",
    "rightsholder", "supervisor", "sponsor", "workpackageleader", "other",
}


class SyncError(RuntimeError):
    """Something went wrong that the caller should see, not a traceback."""


def ssl_context():
    """A context that can verify zenodo.org.

    A Python installed from python.org on macOS ships without a CA
    bundle wired in, and every HTTPS call fails with
    CERTIFICATE_VERIFY_FAILED until its Install Certificates command has
    been run.  ``certifi`` is installed alongside it either way, so
    prefer that and leave the platform default as the fallback.
    """
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def request(path, *, method="GET", payload=None, token=None):
    """Call the Zenodo API and return the decoded body, or None."""
    url = path if path.startswith("http") else f"{BASE}{path}"
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers,
                                 method=method)
    try:
        with urllib.request.urlopen(req, timeout=60,
                                    context=ssl_context()) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:800]
        raise SyncError(
            f"{method} {url} -> HTTP {error.code}\n{detail}") from None
    except urllib.error.URLError as error:
        raise SyncError(f"{method} {url} -> {error.reason}") from None
    return json.loads(body) if body else None


# ---------------------------------------------------------------------
# Translating .zenodo.json into what the API wants
# ---------------------------------------------------------------------

def split_name(name):
    """Split ``"Family, Given"`` the way Zenodo's own interface reads it.

    The interface takes the family name first, which is not obvious from
    looking at it, and a name typed the other way round is silently
    recorded with the given name as the surname.  Doing the split here
    means the file is the only place it can go wrong.
    """
    family, _, given = (part.strip() for part in name.partition(","))
    return family, given


def as_person(entry, *, default_role=None):
    """One creator or contributor, in the API's shape."""
    family, given = split_name(entry["name"])
    person = {"type": "personal", "family_name": family}
    if given:
        person["given_name"] = given
    if entry.get("orcid"):
        person["identifiers"] = [{"scheme": "orcid",
                                  "identifier": entry["orcid"]}]

    out = {"person_or_org": person}
    if entry.get("affiliation"):
        out["affiliations"] = [{"name": entry["affiliation"]}]

    role = (entry.get("type") or default_role or "").lower()
    if role:
        if role not in ROLE_IDS:
            raise SyncError(f"unknown contributor role {entry['type']!r}")
        out["role"] = {"id": role}
    return out


def resolve_subject(term, scheme):
    """Find the vocabulary entry Zenodo knows this term by.

    ``.zenodo.json`` records a term and its vocabulary URI, which is what
    the release-time ingestion reads.  The API instead wants Zenodo's own
    identifier for the same entry, so look it up and insist on an exact
    match rather than accepting the closest suggestion.
    """
    query = urllib.parse.urlencode({"q": f'"{term}"', "size": 50})
    hits = request(f"/subjects?{query}")["hits"]["hits"]
    for hit in hits:
        if hit.get("subject") == term and hit.get("scheme") == scheme:
            return hit["id"]
    raise SyncError(
        f"Zenodo has no {scheme} subject called {term!r}. Check it in "
        f"{BASE}/subjects?q={urllib.parse.quote(term)}")


def build_metadata(config):
    """The metadata payload, from the contents of ``.zenodo.json``."""
    subjects = [{"subject": keyword} for keyword in config["keywords"]]
    for entry in config.get("subjects", []):
        subjects.append({"id": resolve_subject(entry["term"],
                                               entry["scheme"])})

    metadata = {
        "title": config["title"],
        "description": f"<p>{config['description']}</p>",
        "creators": [as_person(person, default_role="projectleader")
                     for person in config["creators"]],
        "contributors": [as_person(person)
                         for person in config.get("contributors", [])],
        "subjects": subjects,
    }
    return {key: value for key, value in metadata.items() if value}


# ---------------------------------------------------------------------
# Reading and writing the record
# ---------------------------------------------------------------------

def latest_record_id(concept=CONCEPT_RECORD):
    """The newest version under the concept DOI."""
    record = request(f"/records/{concept}")
    versions = record.get("links", {}).get("latest")
    if versions:
        return str(request(versions)["id"])
    return str(record["id"])


def describe(metadata):
    """A readable rendering of what would be sent."""
    def named(person):
        who = person["person_or_org"]
        return f"{who['family_name']}, {who.get('given_name', '')}"

    lines = [f"  title         {metadata['title']}"]
    for person in metadata.get("creators", []):
        lines.append(f"  creator       {named(person)}")
    for person in metadata.get("contributors", []):
        role = person.get("role", {}).get("id", "-")
        lines.append(f"  contributor   {named(person)}  [{role}]")
    free = [s["subject"] for s in metadata["subjects"] if "subject" in s]
    linked = [s["id"] for s in metadata["subjects"] if "id" in s]
    lines.append(f"  keywords      {len(free)}: {', '.join(free)}")
    lines.append(f"  subjects      {len(linked)}: {', '.join(linked)}")
    return "\n".join(lines)


def sync(record_id, metadata, token):
    """Edit, update and republish. The DOI is unchanged by this."""
    print(f"  creating a draft of record {record_id}")
    draft = request(f"/records/{record_id}/draft", method="POST", token=token)

    merged = dict(draft["metadata"])
    merged.update(metadata)
    print("  writing the metadata")
    request(f"/records/{record_id}/draft", method="PUT",
            payload={"metadata": merged}, token=token)

    print("  publishing")
    published = request(f"/records/{record_id}/draft/actions/publish",
                        method="POST", token=token)
    return published


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true",
                        help="apply the changes (needs ZENODO_TOKEN)")
    parser.add_argument("--record", metavar="ID",
                        help="record to update (default: newest version)")
    parser.add_argument("--config", type=pathlib.Path,
                        default=pathlib.Path(__file__).parent.parent
                        / ".zenodo.json")
    args = parser.parse_args(argv)

    config = json.loads(args.config.read_text())
    metadata = build_metadata(config)
    record_id = args.record or latest_record_id()

    print(f"record https://zenodo.org/records/{record_id}")
    print(describe(metadata))

    if not args.write:
        print("\ndry run; pass --write to apply")
        return 0

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        raise SyncError(
            "ZENODO_TOKEN is not set. Create a token with the "
            "deposit:write and deposit:actions scopes at "
            "https://zenodo.org/account/settings/applications/tokens/new/")

    published = sync(record_id, metadata, token)
    print(f"\ndone: {published['links']['self_html']}")
    print(f"DOI {published['doi']} (unchanged)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SyncError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
