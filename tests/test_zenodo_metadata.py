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


# --------------------------------------------------------------------------
# The release notes that go onto the archival record
# --------------------------------------------------------------------------

def test_the_release_notes_are_summarised_rather_than_reproduced():
    """The record gets the headlines; the changelog keeps the reasoning.

    1.5.0 put fourteen thousand characters of changelog on its Zenodo
    landing page. This repository writes an entry as a bolded headline and
    then several paragraphs of why, which is right for a file a maintainer
    reads and wrong for a record someone lands on.
    """
    from tools.zenodo_sync import changelog_section, summarise

    notes = changelog_section("1.5.0")
    assert notes is not None
    summary = summarise(notes)

    assert len(summary) < len(notes) / 3
    assert "CHANGELOG.md" in summary          # the rest is a link away


def test_a_wrapped_headline_survives_the_summary_whole():
    """A bullet runs over several source lines, and must not be cut at one.

    Reading the changelog a line at a time truncates a headline wherever
    the paragraph happened to wrap, which is what the first version of
    this did: it produced "the scales, chords and harmonic series of the
    MASS" and stopped.
    """
    from tools.zenodo_sync import summarise

    notes = (
        "### Added\n"
        "- **A thing**, which is described across\n"
        "  more than one line of the source file.\n"
        "\n"
        "  And then a paragraph of reasoning that should not appear.\n"
    )
    summary = summarise(notes)

    assert "described across more than one line" in summary
    assert "reasoning that should not appear" not in summary


def test_the_upgrade_note_is_kept_whole():
    """It is the part a reader of the record needs in front of them."""
    from tools.zenodo_sync import summarise

    notes = (
        "### Note for anyone upgrading\n"
        "**Something changed.** Here is exactly what, at length, because a\n"
        "caller needs it before they upgrade rather than after.\n"
        "\n"
        "### Added\n"
        "- **A routine**, with paragraphs of reasoning below it.\n"
        "\n"
        "  The reasoning, which is dropped.\n"
    )
    summary = summarise(notes)

    assert "at length" in summary
    assert "before they upgrade" in summary
    assert "The reasoning, which is dropped" not in summary


def test_summarising_a_section_with_no_bullets_says_nothing_extra():
    from tools.zenodo_sync import summarise

    summary = summarise("### Added\n\nProse with no entries in it.\n")
    assert "Prose with no entries" not in summary
    assert "CHANGELOG.md" in summary
