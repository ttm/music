""".zenodo.json has to satisfy Zenodo's release-time validator.

A malformed field there does not degrade the deposit, it *fails the
archive*: 1.2.0's first two attempts died with "Extra metadata load
failed" because ``dates`` used the REST API's shape -- ``date`` and a
lowercase type -- in a file that Zenodo validates against its legacy
schema, which wants ``start`` and a capitalised one.

Nothing catches that until a release has already gone out, so these
check statically what the validator would have said.
"""

import json
import pathlib

import pytest

ZENODO = pathlib.Path(__file__).parent.parent / ".zenodo.json"

#: Keys Zenodo's legacy deposit metadata accepts, plus custom_fields,
#: which the REST API reads and the ingestion tolerates.
KNOWN_KEYS = {
    "upload_type", "publication_type", "image_type", "publication_date",
    "title", "creators", "description", "access_right", "license",
    "embargo_date", "access_conditions", "doi", "prereserve_doi",
    "keywords", "notes", "related_identifiers", "contributors",
    "references", "communities", "grants", "subjects", "version",
    "language", "locations", "dates", "method", "custom_fields",
}

#: The four the legacy schema documents for a date entry.
DATE_TYPES = {"Collected", "Valid", "Withdrawn", "Created"}


@pytest.fixture(scope="module")
def metadata():
    return json.loads(ZENODO.read_text())


def test_every_key_is_one_zenodo_knows(metadata):
    unknown = set(metadata) - KNOWN_KEYS
    assert not unknown, f"Zenodo will not recognise {sorted(unknown)}"


def test_dates_use_the_shape_the_ingestion_validates(metadata):
    """Regression: `date` and a lowercase type failed the whole archive."""
    for entry in metadata.get("dates", []):
        assert "start" in entry, (
            f"{entry} needs 'start'; 'date' is the REST API's spelling"
        )
        assert "date" not in entry, f"{entry} must not use 'date'"
        assert entry.get("type") in DATE_TYPES, (
            f"{entry.get('type')!r} is not one of {sorted(DATE_TYPES)}"
        )


def test_creators_and_contributors_are_named_family_first(metadata):
    """Zenodo reads these family-name-first, which is how a contributor
    once ended up recorded with his given name as his surname."""
    for group in ("creators", "contributors"):
        for person in metadata.get(group, []):
            assert "," in person["name"], (
                f"{person['name']!r} should read 'Family, Given'"
            )


def test_subjects_carry_a_scheme_and_a_resolvable_identifier(metadata):
    for entry in metadata.get("subjects", []):
        assert entry["scheme"] and entry["term"]
        assert entry["identifier"].startswith("http"), entry
