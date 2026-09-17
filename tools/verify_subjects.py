#!/usr/bin/env python3
"""Check that every subject in ``.zenodo.json`` is the term it says it is.

The archival subjects are controlled-vocabulary terms rather than free
keywords: each carries a scheme, a label and an identifier, and the point of
them is that a machine can follow the identifier. Nothing checked that they
led anywhere. ``tests/test_zenodo_metadata.py`` checks their shape offline,
which catches a missing field and not a wrong identifier, and
``ASSESSMENT.md`` claimed the lookup had been done without anything to run.

This runs it. For each subject it resolves the identifier against its own
vocabulary and compares the label that comes back with the one the file
declares.

    python tools/verify_subjects.py           # check them all
    python tools/verify_subjects.py --strict  # fail incomplete verification
    python tools/verify_subjects.py --term "Auditory Perception"
                                              # search MeSH for a candidate

Needs the network. It is not part of the test suite for that reason; the
release gate is the place to run it, and it is cheap enough to run by hand
whenever a subject is added.

Strict mode allows only the two explicitly registered EuroSciVoc entries
to remain unverified. Those exceptions are not queried; all other missing
labels, failed lookups and unsupported identifiers prevent release.
"""
from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

METADATA = Path(__file__).resolve().parent.parent / '.zenodo.json'
TIMEOUT = 20

# These exact entries are documented as unconfirmable in ASSESSMENT.md.
# They are exceptions to verification, not successful network lookups.
UNVERIFIED_EXCEPTIONS = frozenset({
    ('http://data.europa.eu/8mn/euroscivoc/1215',
     'EuroSciVoc', 'Signal processing'),
    ('http://data.europa.eu/8mn/euroscivoc/273', 'EuroSciVoc', 'Acoustics'),
})

MESH_DETAILS = 'https://id.nlm.nih.gov/mesh/lookup/details?descriptor={}'
#: Descriptors, not terms: a subject is a descriptor, and the term endpoint
#: answers with the entry terms that lead to one, whose ids are not usable.
MESH_SEARCH = ('https://id.nlm.nih.gov/mesh/lookup/descriptor?label={}'
               '&match=contains&limit=12')
GEMET_CONCEPT = ('https://www.eionet.europa.eu/gemet/getConcept'
                 '?concept_uri=http://www.eionet.europa.eu/gemet/concept/{}'
                 '&language=en')


def _get_json(url: str):
    request = urllib.request.Request(
        url, headers={'Accept': 'application/json',
                      'User-Agent': 'music/verify_subjects'})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return json.load(response)


def resolve_mesh(identifier: str) -> str | None:
    """The label MeSH gives a descriptor or a qualified descriptor.

    A plain descriptor is ``D009146``; a topical qualifier appends its own
    code, as ``D009146Q000523`` does, and the pair is looked up as the
    descriptor with the qualifier's label appended -- which is how the
    subject is written in the file.
    """
    code = identifier.rsplit('/', 1)[-1]
    descriptor, _, qualifier = code.partition('Q')
    try:
        details = _get_json(MESH_DETAILS.format(descriptor))
    except (OSError, ValueError):
        return None
    if not isinstance(details, dict):
        return None
    # The descriptor's own label is the preferred one among its terms; the
    # rest are entry terms that lead to it, and are not what a subject means.
    label = next((term.get('label') for term in details.get('terms', [])
                  if isinstance(term, dict) and term.get('preferred')), None)
    if label is None:
        return None
    if qualifier:
        for entry in details.get('qualifiers', []):
            if entry.get('resource', '').endswith(f'Q{qualifier}'):
                return f"{label}/{entry['label']}"
        return None
    return label


def resolve_gemet(identifier: str) -> str | None:
    """The English preferred label of a GEMET concept.

    The concept URL itself serves a web page; the label comes from GEMET's
    own ``getConcept`` service, which wants the concept URI rather than the
    language-prefixed one a reader is given.
    """
    code = identifier.rstrip('/').rsplit('/', 1)[-1]
    try:
        payload = _get_json(GEMET_CONCEPT.format(code))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    label = (payload or {}).get('preferredLabel') or {}
    return label.get('string') if isinstance(label, dict) else None


def resolve(subject: dict) -> str | None:
    """The label a vocabulary gives the identifier, or None if it gives none.

    Only MeSH and GEMET are queried, when the declared scheme matches
    the identifier's provider. Other entries, including EuroSciVoc,
    return None without a lookup.
    """
    identifier = subject['identifier']
    parsed = urllib.parse.urlparse(identifier)
    if parsed.scheme not in ('http', 'https'):
        return None
    if (subject['scheme'] == 'MeSH' and parsed.hostname == 'id.nlm.nih.gov'
            and parsed.path.startswith('/mesh/')):
        return resolve_mesh(identifier)
    if (subject['scheme'] == 'GEMET'
            and parsed.hostname in ('eionet.europa.eu', 'www.eionet.europa.eu')
            and parsed.path.startswith('/gemet/')):
        return resolve_gemet(identifier)
    return None


def search(term: str) -> int:
    """Print MeSH descriptors whose label contains `term`."""
    try:
        hits = _get_json(MESH_SEARCH.format(urllib.parse.quote(term)))
    except (OSError, ValueError) as exc:
        print(f'lookup failed: {exc}')
        return 1
    if not hits:
        print(f'no MeSH descriptor contains {term!r}')
        return 1
    for hit in hits:
        print(f"  {hit['resource'].rsplit('/', 1)[-1]:<16} {hit['label']}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--term', help='search MeSH for a candidate subject')
    parser.add_argument('--strict', action='store_true',
                        help='fail unresolved subjects except known entries')
    args = parser.parse_args(argv)
    if args.term:
        return search(args.term)

    subjects = json.loads(METADATA.read_text()).get('subjects', [])
    if not subjects:
        print('no subjects in .zenodo.json')
        return 1

    wrong, unreachable, exceptions = [], [], []
    for subject in subjects:
        declared = subject['term']
        entry = (subject['identifier'], subject['scheme'], declared)
        if entry in UNVERIFIED_EXCEPTIONS:
            exceptions.append(subject)
            print(f'  ?  {declared:<40} {subject["identifier"]} '
                  '(known exception; not queried, unverified)')
            continue
        found = resolve(subject)
        if not isinstance(found, str) or not found:
            unreachable.append(subject)
            print(f'  ?  {declared:<40} {subject["identifier"]}')
        elif found.lower() != declared.lower():
            wrong.append((subject, found))
            print(f'  !  {declared:<40} resolves to {found!r}')
        else:
            print(f'  ok {declared:<40} {subject["scheme"]}')

    verified = len(subjects) - len(wrong) - len(unreachable) - len(exceptions)
    print(f'\n{verified} of '
          f'{len(subjects)} subjects resolve to the term they declare')
    if exceptions:
        print(f'{len(exceptions)} known exceptions were not queried and '
              'remain unverified.')
    if unreachable:
        print(f'{len(unreachable)} could not be confirmed: the lookup was '
              'unavailable, returned no label, or is unsupported.')
    if wrong:
        for subject, found in wrong:
            print(f'{subject["identifier"]} is {found!r}, '
                  f'not {subject["term"]!r}')
        return 1
    if args.strict and unreachable:
        print('strict verification failed: unresolved subjects remain')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
