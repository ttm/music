"""External release checks must distinguish verified from unavailable.

The transport and the reference checkout are local fixtures; no test here
depends on the network or a maintainer's own MASS installation.
"""

import json
import urllib.error

import pytest

from tools import article_coverage as article
from tools import mass_reference as reference
from tools import verify_subjects as subjects


def make_reference(root):
    path = root / 'src' / 'aux' / 'functions.py'
    path.parent.mkdir(parents=True)
    path.write_text('# reference fixture\n')
    return path


@pytest.fixture
def mass_checkout(tmp_path, monkeypatch):
    root = tmp_path / 'mass'
    make_reference(root)
    monkeypatch.delenv('MASS_SRC', raising=False)
    monkeypatch.setattr(reference, 'DEFAULT_LOCATIONS', (str(root),))
    return root


def test_mass_default_search_still_finds_a_checkout(mass_checkout):
    assert reference.locate() == mass_checkout / 'src' / 'aux' / 'functions.py'


@pytest.mark.parametrize('file_path', [False, True])
def test_mass_accepts_an_explicit_checkout_or_reference(mass_checkout,
                                                      file_path):
    source = mass_checkout / 'src' / 'aux' / 'functions.py'
    selected = source if file_path else mass_checkout
    assert reference.locate(str(selected)) == source


@pytest.mark.parametrize('selector', ['argument', 'environment'])
def test_invalid_mass_selection_cannot_fall_back_to_another_checkout(
        mass_checkout, monkeypatch, selector):
    missing = str(mass_checkout / 'missing')
    if selector == 'environment':
        monkeypatch.setenv('MASS_SRC', missing)
    with pytest.raises(reference.ReferenceNotFound, match='no MASS reference'):
        reference.locate(missing if selector == 'argument' else None)


def test_mass_argument_takes_precedence_over_environment(mass_checkout,
                                                       monkeypatch):
    monkeypatch.setenv('MASS_SRC', str(mass_checkout / 'missing'))
    assert reference.locate(str(mass_checkout)).is_file()


def test_mass_environment_selects_its_checkout(mass_checkout, tmp_path,
                                             monkeypatch):
    selected = make_reference(tmp_path / 'selected')
    monkeypatch.setenv('MASS_SRC', str(selected))
    assert reference.locate() == selected


def test_missing_mass_explains_how_to_supply_it(tmp_path, monkeypatch):
    monkeypatch.delenv('MASS_SRC', raising=False)
    monkeypatch.setattr(reference, 'DEFAULT_LOCATIONS', (str(tmp_path),))
    with pytest.raises(reference.ReferenceNotFound, match='MASS_SRC'):
        reference.locate()


@pytest.fixture
def article_sources(mass_checkout, monkeypatch):
    doc = mass_checkout / 'doc'
    doc.mkdir()
    labels = ['checked', 'missing', 'vinculos']
    for name, label in zip(article.SOURCES, labels):
        (doc / name).write_text('\\label{eq:' + label + '}\n')
    monkeypatch.setattr(article, 'cited', lambda: {'checked'})
    return mass_checkout


def test_article_report_stays_informative_without_strict(article_sources):
    assert article.main(['--mass', str(article_sources)]) == 0


def test_article_strict_refuses_an_uncited_reachable_equation(article_sources,
                                                            capsys):
    assert article.main(['--mass', str(article_sources), '--strict']) == 1
    output = capsys.readouterr().out
    assert 'implemented equations without a test citation' in output


def test_article_strict_accepts_coverage_with_documented_exceptions(
        article_sources, monkeypatch):
    monkeypatch.setattr(article, 'cited', lambda: {'checked', 'missing'})
    assert article.main(['--strict', '--mass', str(article_sources)]) == 0


@pytest.mark.parametrize('missing', article.SOURCES)
def test_article_strict_requires_each_source(article_sources, missing, capsys):
    (article_sources / 'doc' / missing).unlink()
    assert article.main(['--strict', '--mass', str(article_sources)]) == 1
    output = capsys.readouterr().out
    assert 'missing article sources' in output
    assert missing in output


def test_article_missing_checkout_is_a_readable_failure(mass_checkout, capsys):
    missing = str(mass_checkout / 'gone')
    assert article.main(['--strict', '--mass', missing]) == 1
    assert 'no MASS reference' in capsys.readouterr().out


def test_article_with_only_a_schema_has_no_zero_division(article_sources,
                                                        monkeypatch, capsys):
    for name in article.SOURCES:
        (article_sources / 'doc' / name).write_text('')
    (article_sources / 'doc' / article.SOURCES[0]).write_text(
        '\\label{eq:' + 'vinculos' + '}\n')
    monkeypatch.setattr(article, 'cited', lambda: set())
    assert article.main(['--strict', '--mass', str(article_sources)]) == 0
    assert '(n/a)' in capsys.readouterr().out


MESH = {'term': 'Music', 'scheme': 'MeSH',
        'identifier': 'https://id.nlm.nih.gov/mesh/D009146'}
GEMET = {'term': 'sound', 'scheme': 'GEMET',
         'identifier': 'https://www.eionet.europa.eu/gemet/en/concept/7913'}


@pytest.fixture
def metadata(tmp_path, monkeypatch):
    path = tmp_path / 'metadata.json'
    monkeypatch.setattr(subjects, 'METADATA', path)

    def write(entries):
        path.write_text(json.dumps({'subjects': entries}))

    return write


def test_subjects_strict_accepts_verified_labels(metadata, monkeypatch,
                                               capsys):
    metadata([MESH, GEMET])

    def lookup(url):
        if 'descriptor=' in url:
            return {'terms': [{'preferred': True, 'label': 'Music'}]}
        return {'preferredLabel': {'string': 'sound'}}

    monkeypatch.setattr(subjects, '_get_json', lookup)
    assert subjects.main(['--strict']) == 0
    assert '2 of 2 subjects resolve' in capsys.readouterr().out


@pytest.mark.parametrize('error', [urllib.error.URLError('offline'),
                                 TimeoutError('timed out'), OSError('TLS'),
                                 ValueError('invalid JSON')])
def test_subjects_strict_fails_when_lookups_are_unavailable(
        metadata, monkeypatch, capsys, error):
    metadata([MESH, GEMET])

    def unavailable(url):
        raise error

    monkeypatch.setattr(subjects, '_get_json', unavailable)
    assert subjects.main(['--strict']) == 1
    output = capsys.readouterr().out
    assert '0 of 2 subjects resolve' in output
    assert 'strict verification failed' in output


def test_subjects_report_keeps_unavailable_lookups_visible(
        metadata, monkeypatch, capsys):
    metadata([MESH])
    monkeypatch.setattr(subjects, '_get_json', lambda url: {})
    assert subjects.main([]) == 0
    assert '1 could not be confirmed' in capsys.readouterr().out


@pytest.mark.parametrize('strict', [False, True])
def test_subject_label_mismatch_always_fails(metadata, monkeypatch, strict):
    metadata([MESH])
    monkeypatch.setattr(subjects, '_get_json', lambda url: {
        'terms': [{'preferred': True, 'label': 'Different term'}]})
    assert subjects.main(['--strict'] if strict else []) == 1


def test_exact_euroscivoc_exceptions_are_explicitly_unverified(metadata,
                                                           capsys):
    metadata([{'identifier': identifier, 'scheme': scheme, 'term': term}
              for identifier, scheme, term in subjects.UNVERIFIED_EXCEPTIONS])
    assert subjects.main(['--strict']) == 0
    output = capsys.readouterr().out
    assert '0 of 2 subjects resolve' in output
    assert ('2 known exceptions were not queried and remain unverified'
            in output)


@pytest.mark.parametrize('field, replacement', [
    ('identifier', 'http://data.europa.eu/8mn/euroscivoc/999'),
    ('scheme', 'Other scheme'),
    ('term', 'Different term'),
])
def test_changed_euroscivoc_entries_are_not_exempt(metadata, field,
                                                replacement):
    entry = {'identifier': 'http://data.europa.eu/8mn/euroscivoc/273',
             'scheme': 'EuroSciVoc', 'term': 'Acoustics'}
    entry[field] = replacement
    metadata([entry])
    assert subjects.main(['--strict']) == 1


@pytest.mark.parametrize('identifier', [
    'https://example.org/term',
    'https://example.org/id.nlm.nih.gov/mesh/D009146',
    'https://example.org/eionet.europa.eu/gemet/concept/7913',
    'ftp://id.nlm.nih.gov/mesh/D009146',
])
def test_unknown_identifiers_cannot_be_verified(metadata, identifier):
    metadata([{**MESH, 'identifier': identifier}])
    assert subjects.main(['--strict']) == 1


@pytest.mark.parametrize('entry, wrong_scheme', [(MESH, 'GEMET'),
                                               (GEMET, 'MeSH')])
def test_supported_identifier_with_wrong_scheme_cannot_verify(
        metadata, monkeypatch, entry, wrong_scheme):
    metadata([{**entry, 'scheme': wrong_scheme}])
    calls = []

    def matching_label(url):
        calls.append(url)
        return {'terms': [{'preferred': True, 'label': entry['term']}],
                'preferredLabel': {'string': entry['term']}}

    monkeypatch.setattr(subjects, '_get_json', matching_label)
    assert subjects.main(['--strict']) == 1
    assert calls == []


@pytest.mark.parametrize('payload', [None, [], {}, {'preferredLabel': []}])
def test_empty_or_malformed_gemet_response_is_unavailable(
        metadata, monkeypatch, payload):
    metadata([GEMET])
    monkeypatch.setattr(subjects, '_get_json', lambda url: payload)
    assert subjects.main(['--strict']) == 1


def test_subject_search_handles_raw_timeout(monkeypatch, capsys):
    def unavailable(url):
        raise TimeoutError('timed out')

    monkeypatch.setattr(subjects, '_get_json', unavailable)
    assert subjects.main(['--term', 'Music']) == 1
    assert 'lookup failed' in capsys.readouterr().out


def _answers(*outcomes):
    """A stand-in for urlopen that fails or answers in the order given."""
    import io
    import json as json_
    calls = []

    def urlopen(request, timeout):
        outcome = outcomes[len(calls)]
        calls.append(request.full_url)
        if isinstance(outcome, Exception):
            raise outcome
        return io.BytesIO(json_.dumps(outcome).encode())
    return urlopen, calls


def test_a_lookup_is_retried_after_a_network_failure(monkeypatch):
    """GEMET failed now and then and answered a moment later; twice that
    stopped a release gate that had passed everything else."""
    import urllib.error
    urlopen, calls = _answers(urllib.error.URLError("down"),
                              urllib.error.HTTPError("u", 503, "", {}, None),
                              {"ok": True})
    monkeypatch.setattr(subjects.urllib.request, "urlopen", urlopen)
    assert subjects._get_json("https://example.org", pause=0) == {"ok": True}
    assert len(calls) == 3


def test_a_lookup_is_not_retried_after_an_answer(monkeypatch):
    """A 404 is about the identifier, and asking again changes nothing."""
    import urllib.error
    urlopen, calls = _answers(urllib.error.HTTPError("u", 404, "", {}, None))
    monkeypatch.setattr(subjects.urllib.request, "urlopen", urlopen)
    with pytest.raises(urllib.error.HTTPError):
        subjects._get_json("https://example.org", pause=0)
    assert len(calls) == 1


def test_a_lookup_gives_up_after_its_attempts(monkeypatch):
    import urllib.error
    urlopen, calls = _answers(*[urllib.error.URLError("down")] * 3)
    monkeypatch.setattr(subjects.urllib.request, "urlopen", urlopen)
    with pytest.raises(urllib.error.URLError):
        subjects._get_json("https://example.org", pause=0)
    assert len(calls) == 3
