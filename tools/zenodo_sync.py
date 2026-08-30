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
import html
import json
import os
import pathlib
import re
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


#: Zenodo serves records in two shapes. The default is its legacy one,
#: where a resource type is ``{"title": ..., "type": ...}`` and a relation
#: is a bare string; the write API speaks the other, where both are
#: ``{"id": ...}``. Reading in the wrong one and writing it back strips
#: exactly those fields, and the publish then fails validation on them.
RDM = "application/vnd.inveniordm.v1+json"


def request(path, *, method="GET", payload=None, token=None,
            accept="application/json"):
    """Call the Zenodo API and return the decoded body, or None."""
    url = path if path.startswith("http") else f"{BASE}{path}"
    data = None
    headers = {"Accept": accept}
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
    # A list lets an affiliation carry its ROR identifier, which resolves
    # to a real organisation, alongside ones that have no ROR entry and
    # can only be named. The single string is the fallback, and is what
    # the release-time ingestion reads.
    if entry.get("affiliations"):
        out["affiliations"] = entry["affiliations"]
    elif entry.get("affiliation"):
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


# ---------------------------------------------------------------------
# The changelog entry, as an additional description
# ---------------------------------------------------------------------

def changelog_section(version, path=None):
    """The changelog's entry for ``version``, as markdown.

    Returns None when there is no section for it, which is the normal
    case for a record whose version predates the file.
    """
    path = path or pathlib.Path(__file__).parent.parent / "CHANGELOG.md"
    version = version.lstrip("v")
    # The lookahead has to accept end-of-file as well as the next
    # heading, or the oldest release in the file never matches and its
    # notes silently fail to attach.
    match = re.search(
        rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[|\Z)",
        path.read_text(), re.MULTILINE | re.DOTALL)
    return match.group(1).strip() if match else None


def _inline(text):
    """Escape the text, then put back the markup the changelog uses.

    Code spans are held out of the rest of the conversion. This package's
    changelog quotes expressions such as ``2 ** (bit_depth - 1)``, and a
    bold rule that ran over them would pair those asterisks with the next
    ones outside the span and emit tags that cross.
    """
    pieces = []
    for index, part in enumerate(re.split(r"`([^`]+)`", text)):
        escaped = html.escape(part)
        if index % 2:  # the captured inside of a code span
            pieces.append(f"<code>{escaped}</code>")
            continue
        escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>",
                         escaped)
        pieces.append(
            re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>',
                   escaped))
    return "".join(pieces)


def markdown_to_html(markdown):
    """Convert the subset of markdown the changelog actually uses.

    Headings, bullet lists one level deep, paragraphs, and inline code,
    links and bold. Deliberately not a general converter: anything else
    in the changelog would come through as plain text rather than
    silently mangled, and Zenodo sanitises what it is given anyway.
    """
    out, paragraph, depth = [], [], 0

    def flush():
        if paragraph:
            out.append(f"<p>{_inline(' '.join(paragraph))}</p>")
            paragraph.clear()

    def close_lists(to):
        """Close nested levels from the inside out.

        A nested list belongs inside the item it hangs off, so closing a
        level closes that item too -- a bare ``<ul>`` inside a ``<ul>``
        is not valid, and a sanitiser is free to drop or re-parent it.
        """
        nonlocal depth
        while depth > to:
            out.append("</ul></li>" if depth > 1 else "</ul>")
            depth -= 1

    for line in markdown.splitlines():
        heading = re.match(r"^(#{1,6}) +(.*)$", line)
        bullet = re.match(r"^( *)- +(.*)$", line)
        if heading:
            flush(), close_lists(0)
            level = min(len(heading.group(1)) + 1, 6)
            out.append(f"<h{level}>{_inline(heading.group(2))}"
                       f"</h{level}>")
        elif bullet:
            flush()
            # A nested level has to hang off an item, so a list that
            # opens already indented starts at depth one rather than
            # emitting a <ul> straight inside a <ul>.
            want = min(1 if len(bullet.group(1)) < 2 else 2, depth + 1)
            close_lists(want)
            while depth < want:
                if depth and out[-1].endswith("</li>"):
                    # Reopen the item this nested list hangs off.
                    out[-1] = out[-1][:-len("</li>")]
                out.append("<ul>")
                depth += 1
            out.append(f"<li>{_inline(bullet.group(2))}</li>")
        elif not line.strip():
            flush(), close_lists(0)
        elif depth:
            # A wrapped continuation of the bullet above it.
            out[-1] = out[-1][:-len("</li>")] + " " + _inline(line.strip()) \
                + "</li>"
        else:
            paragraph.append(line.strip())

    flush(), close_lists(0)
    return "\n".join(out)


def build_metadata(config, *, with_description=False, release_notes=None):
    """The metadata payload, from the contents of ``.zenodo.json``.

    The description is left alone unless asked for. Zenodo fills it with
    the release notes, which say what changed in that version and are
    worth more on a per-version record than the package's standing
    abstract; overwriting them by default would quietly discard them.
    """
    subjects = [{"subject": keyword} for keyword in config["keywords"]]
    for entry in config.get("subjects", []):
        subjects.append({"id": resolve_subject(entry["term"],
                                               entry["scheme"])})

    metadata = {
        "title": config["title"],
        "creators": [as_person(person, default_role="projectleader")
                     for person in config["creators"]],
        "contributors": [as_person(person)
                         for person in config.get("contributors", [])],
        "subjects": subjects,
    }
    if with_description:
        metadata["description"] = f"<p>{config['description']}</p>"
    if release_notes:
        metadata["additional_descriptions"] = [{
            "description": markdown_to_html(release_notes),
            "type": {"id": "technical-info"},
        }]

    if config.get("language"):
        metadata["languages"] = [{"id": config["language"]}]
    if config.get("references"):
        metadata["references"] = [{"reference": reference}
                                  for reference in config["references"]]
    if config.get("dates"):
        # .zenodo.json holds these in the shape Zenodo's release-time
        # ingestion validates -- "start" and a capitalised type. The API
        # wants "date" and a lowercase vocabulary id.
        metadata["dates"] = [
            {"date": entry.get("start") or entry["date"],
             "type": {"id": entry["type"].lower()},
             **({"description": entry["description"]}
                if entry.get("description") else {})}
            for entry in config["dates"]]

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


def record_version(record_id):
    """The version string the record carries, such as ``v1.1.1``."""
    record = request(f"/records/{record_id}", accept=RDM)
    return record.get("metadata", {}).get("version") or ""


def describe(metadata, custom_fields=None):
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
    lines.append("  description   "
                 + ("replaced with the abstract from .zenodo.json"
                    if "description" in metadata
                    else "left as it is on the record"))
    extra = metadata.get("additional_descriptions")
    lines.append("  release notes "
                 + (f"attached, {len(extra[0]['description'])} characters "
                    f"of HTML from the changelog"
                    if extra else "none found in the changelog"))
    for key in ("languages", "dates", "references"):
        if metadata.get(key):
            lines.append(f"  {key:<13} {len(metadata[key])}")
    if custom_fields:
        lines.append(f"  custom fields {', '.join(sorted(custom_fields))}")
    return "\n".join(lines)


def sync(record_id, metadata, token, custom_fields=None):
    """Edit, update and republish. The DOI is unchanged by this.

    A draft carries everything the published record has, and a PUT
    replaces the metadata wholesale, so what is not being changed has to
    be read back and sent again -- in the same shape the write API
    expects, which is why the draft is re-read as ``RDM``.
    """
    print(f"  creating a draft of record {record_id}")
    request(f"/records/{record_id}/draft", method="POST", token=token)
    draft = request(f"/records/{record_id}/draft", token=token, accept=RDM)

    merged = dict(draft["metadata"])
    merged.update(metadata)

    # custom_fields are a sibling of metadata in the payload, not a key
    # inside it, and the software block lives there.
    fields = dict(draft.get("custom_fields") or {})
    fields.update(custom_fields or {})

    print("  writing the metadata")
    request(f"/records/{record_id}/draft", method="PUT",
            payload={"metadata": merged, "custom_fields": fields},
            token=token, accept=RDM)

    print("  publishing")
    published = request(f"/records/{record_id}/draft/actions/publish",
                        method="POST", token=token)
    return published


def discard(record_id, token):
    """Throw away a draft, leaving the published record as it was."""
    request(f"/records/{record_id}/draft", method="DELETE", token=token)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true",
                        help="apply the changes (needs ZENODO_TOKEN)")
    parser.add_argument("--record", metavar="ID",
                        help="record to update (default: newest version)")
    parser.add_argument("--no-release-notes", action="store_true",
                        help="do not attach the changelog entry for this "
                             "version as an additional description")
    parser.add_argument("--description", action="store_true",
                        help="also replace the record's description, which "
                             "Zenodo fills with the release notes")
    parser.add_argument("--config", type=pathlib.Path,
                        default=pathlib.Path(__file__).parent.parent
                        / ".zenodo.json")
    args = parser.parse_args(argv)

    config = json.loads(args.config.read_text())
    record_id = args.record or latest_record_id()

    notes = None
    if not args.no_release_notes:
        version = record_version(record_id)
        notes = changelog_section(version) if version else None

    metadata = build_metadata(config, with_description=args.description,
                              release_notes=notes)

    print(f"record https://zenodo.org/records/{record_id}")
    print(describe(metadata, config.get("custom_fields")))

    if not args.write:
        print("\ndry run; pass --write to apply")
        return 0

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        raise SyncError(
            "ZENODO_TOKEN is not set. Create a token with the "
            "deposit:write and deposit:actions scopes at "
            "https://zenodo.org/account/settings/applications/tokens/new/")

    try:
        published = sync(record_id, metadata, token,
                         config.get("custom_fields"))
    except SyncError:
        print("  discarding the draft; the published record is untouched")
        discard(record_id, token)
        raise
    print(f"\ndone: {published['links']['self_html']}")
    print(f"DOI {published['doi']} (unchanged)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SyncError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
