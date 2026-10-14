"""Actual pinned SSTIM SessionSpecification plan semantics.

Never infer observed SessionInstance. SHACL requirements are frozen at
SSTIM 0.19.0: a preset, creation time, master volume and integer
duration of 60 to 7200 seconds.
"""
from datetime import datetime, timezone
import os

import pytest

pytest.importorskip("rdflib")
from rdflib import Literal, URIRef  # noqa: E402
from rdflib.namespace import DCTERMS, RDF, XSD  # noqa: E402

import music  # noqa: E402
from music.stimulation.sstim_io import (  # noqa: E402
    MUSIC, SSTIM, SSTIM_V, validate_sstim,
)
from music.stimulation.sstim_planned import (  # noqa: E402
    from_sstim_planned_session_graph, to_sstim_planned_session_graph,
)
from music.stimulation.sstim_capabilities import (  # noqa: E402
    inspect_sstim_capabilities,
)


CREATED = datetime(2026, 10, 11, 0, 0, tzinfo=timezone.utc)
PRESET = "https://example.org/real-preset/version-1"


def plan(*, duration=60., sample_rate=8000):
    """Build a short, eligible MUSIC program without rendering PCM."""
    session = music.StimulationSession(sample_rate=sample_rate)
    session.add(music.binaural_beats, duration=duration,
                carrier_freq=200., beat_freq=10.,
                label="planned auditory stimulation")
    return session


def graph(**kwargs):
    arguments = dict(session=plan(), preset_iri=PRESET,
                     preset_label="Versioned test preset",
                     created_at=CREATED, master_volume=.2,
                     base="https://example.org/study/plan/")
    arguments.update(kwargs)
    return to_sstim_planned_session_graph(**arguments)


def test_real_session_specification_roundtrips_with_pinned_preset():
    g = graph()
    assert (None, RDF.type, SSTIM.SessionSpecification) in g
    assert (None, RDF.type, SSTIM.Preset) in g
    assert (None, RDF.type, SSTIM.SessionInstance) not in g
    spec = from_sstim_planned_session_graph(g)
    assert spec.preset_iri == PRESET
    assert spec.created_at == CREATED
    assert spec.master_volume == .2
    assert spec.session.duration == 60
    assert len(spec.session.phases) == 1
    assert (None, SSTIM.hasReproducibilityLevel,
            SSTIM_V.reproEquivalentPresentation) in g
    assert not any(str(subject).startswith("https://w3id.org/sstim")
                   for subject, _, _ in g)
    assert (None, SSTIM.durationSeconds,
            Literal(60, datatype=XSD.integer)) in g
    root = next(g.subjects(RDF.type, SSTIM.SessionSpecification))
    digest = str(next(g.objects(root, SSTIM.configurationDigest)))
    assert len(digest) == 64
    assert (root, SSTIM.digestAlgorithm,
            Literal("sha256-music-plan-json-v1",
                    datatype=XSD.string)) in g
    assert str(next(g.objects(root, SSTIM.configurationDigest))) == (
        str(next(graph().objects(root, SSTIM.configurationDigest))))
    text = g.serialize(format="turtle")
    assert from_sstim_planned_session_graph(text).preset_iri == PRESET


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("SSTIM_LIVE") != "1",
                    reason="SSTIM 0.19.0 Full-profile live validator")
def test_planned_session_conforms_to_official_0_19_profile():
    result = validate_sstim(graph(), version="0.19.0")
    assert result.ok, str(result)


@pytest.mark.parametrize("override,match", [
    ({"preset_iri": ""}, "preset_iri"),
    ({"preset_iri": "https://w3id.org/sstim#Fake"}, "preset_iri"),
    ({"preset_label": ""}, "preset_label"),
    ({"created_at": datetime(2026, 10, 11)}, "timezone-aware"),
    ({"created_at": "2026-10-11"}, "timezone-aware"),
    ({"master_volume": -1}, "master_volume"),
    ({"master_volume": float("nan")}, "master_volume"),
    ({"master_volume": True}, "master_volume"),
    ({"session": plan(duration=59)}, "integer 60"),
    ({"session": plan(duration=60.5)}, "integer 60"),
    ({"session": plan(duration=7201, sample_rate=8000)}, "integer 60"),
])
def test_nonconformant_planned_sessions_are_refused(override, match):
    with pytest.raises(ValueError, match=match):
        graph(**override)


@pytest.mark.parametrize("mutation,match", [
    ("missing-root", "one planned"),
    ("wrong-root-iri", "invalid planned-session IRI"),
    ("wrong-program", "reference its MUSIC program"),
    ("missing-preset-type", "typed sstim:Preset"),
    ("missing-label", "expected exactly one"),
    ("missing-created", "expected exactly one"),
    ("bad-created", "timestamp must be"),
    ("malformed-created", "invalid planned-session metadata"),
    ("bad-volume", "master_volume"),
    ("bad-duration", "invalid planned-session metadata"),
    ("wrong-reproducibility", "unsupported reproducibility"),
    ("changed-duration", "planned duration contradicts"),
    ("extra-assertion", "contradicts MUSIC plan"),
    ("digest-tampered", "contradicts MUSIC plan"),
    ("digest-algorithm-missing", "contradicts MUSIC plan"),
])
def test_planned_session_rejects_conflicts(mutation, match):
    g = graph()
    root = next(g.subjects(RDF.type, SSTIM.SessionSpecification))
    preset = next(g.objects(root, SSTIM.referencesPreset))
    if mutation == "missing-root":
        g.remove((root, RDF.type, SSTIM.SessionSpecification))
    elif mutation == "wrong-root-iri":
        g.remove((root, RDF.type, SSTIM.SessionSpecification))
        g.add((URIRef("urn:invalid"), RDF.type,
               SSTIM.SessionSpecification))
    elif mutation == "wrong-program":
        g.remove((root, MUSIC.hasProgram, None))
        g.add((root, MUSIC.hasProgram, URIRef("urn:invalid")))
    elif mutation == "missing-preset-type":
        g.remove((preset, RDF.type, SSTIM.Preset))
    elif mutation == "missing-label":
        g.remove((preset, URIRef("http://www.w3.org/2000/01/rdf-schema#label"),
                  None))
    elif mutation == "missing-created":
        g.remove((root, DCTERMS.created, None))
    elif mutation == "bad-created":
        g.remove((root, DCTERMS.created, None))
        g.add((root, DCTERMS.created, Literal("not-a-date")))
    elif mutation == "malformed-created":
        g.remove((root, DCTERMS.created, None))
        g.add((root, DCTERMS.created,
               Literal("no-such-date", datatype=XSD.dateTime)))
    elif mutation == "bad-volume":
        g.remove((root, SSTIM.masterVolume, None))
        g.add((root, SSTIM.masterVolume, Literal("nan")))
    elif mutation == "bad-duration":
        g.remove((root, SSTIM.durationSeconds, None))
        g.add((root, SSTIM.durationSeconds, Literal("not-int")))
    elif mutation == "wrong-reproducibility":
        g.remove((root, SSTIM.hasReproducibilityLevel, None))
        g.add((root, SSTIM.hasReproducibilityLevel,
               SSTIM_V.reproIdenticalRendering))
    elif mutation == "changed-duration":
        g.remove((root, SSTIM.durationSeconds, None))
        g.add((root, SSTIM.durationSeconds,
               Literal(61, datatype=XSD.integer)))
    elif mutation == "digest-tampered":
        g.remove((root, SSTIM.configurationDigest, None))
        g.add((root, SSTIM.configurationDigest,
               Literal("0" * 64, datatype=XSD.string)))
    elif mutation == "digest-algorithm-missing":
        g.remove((root, SSTIM.digestAlgorithm, None))
    else:
        g.add((root, MUSIC.unrecognizedControl, Literal(17)))
    with pytest.raises(ValueError, match=match):
        from_sstim_planned_session_graph(g)


def test_missing_music_plan_rejected_before_claiming_native_session():
    g = graph()
    root = next(g.subjects(RDF.type, SSTIM.SessionSpecification))
    prefix = str(root)[:-len("session-specification")]
    for sub, pred, obj in list(g):
        if str(sub) == prefix + "program":
            g.remove((sub, pred, obj))
    with pytest.raises(ValueError, match="StimulationProgram"):
        from_sstim_planned_session_graph(g)


def test_refuse_multiple_planned_roots():
    g = graph()
    g.add((URIRef("urn:other"), RDF.type, SSTIM.SessionSpecification))
    with pytest.raises(ValueError, match="one planned"):
        from_sstim_planned_session_graph(g)


def test_planned_sstim_capability_is_distinct_from_actual_execution():
    good = graph()
    capability = inspect_sstim_capabilities(good)
    assert capability.mode == "sstim-planned-music-program"
    assert capability.can_render
    assert not capability.exact_pcm_portable
    root = next(good.subjects(RDF.type, SSTIM.SessionSpecification))
    good.remove((root, SSTIM.durationSeconds, None))
    assert inspect_sstim_capabilities(good).mode == "unsupported"


def test_configuration_digest_changes_with_preset_and_master_volume():
    def digest(g):
        root = next(g.subjects(RDF.type, SSTIM.SessionSpecification))
        return str(next(g.objects(root, SSTIM.configurationDigest)))
    original = digest(graph(master_volume=.2))
    assert digest(graph(master_volume=.3)) != original
    assert digest(graph(preset_iri=PRESET + "-different")) != original
    assert digest(graph(master_volume=.20)) == original
