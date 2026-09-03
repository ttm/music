"""Every documented parameter must exist, and every parameter be documented.

Two exported routines documented parameters under names the signature did
not have -- `note_with_vibrato` said `max_pitch_deviation` where the
argument is `max_pitch_dev`, and `note_with_two_vibratos` said
`secondary_vibrato_waveform_table` for `sec_vibrato_waveform_table`. Code
written from the published reference raised `TypeError`. Nothing failed,
because nothing was looking.

This is the thing that looks. It is deliberately mechanical: it does not
judge whether a description is any good, only whether the names in the
docstring and the names in the signature are the same names, which is the
part a reader relies on and the part that silently rots.
"""

import inspect

import pytest

import music


def _section(doc, title):
    """The lines of one numpydoc section, or None if it is absent."""
    lines = doc.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == title and i + 1 < len(lines) \
                and set(lines[i + 1].strip()) == {'-'}:
            break
    else:
        return None
    body = []
    for line in lines[i + 2:]:
        if set(line.strip()) == {'-'} and body:
            body.pop()          # that line was the next section's title
            break
        body.append(line)
    return body


def documented_names(doc, title='Parameters'):
    """The names numpydoc declares in `title`, in the order given.

    A declaration sits at column zero -- `inspect.getdoc` has already
    dedented -- and its description is indented under it. That
    distinction is the whole parser, and it is also why a continuation
    line accidentally left at column zero shows up here as a parameter
    that does not exist, which is a real rendering defect rather than a
    false positive: Sphinx reads it the same way.
    """
    body = _section(doc, title)
    if body is None:
        return None
    names = []
    for line in body:
        if line.strip() and line[:1].strip():
            head = line.split(':', 1)[0].strip()
            names.extend(n.strip().lstrip('*') for n in head.split(',')
                         if n.strip())
    return names


def _public(predicate):
    """Exported names satisfying `predicate`, for parametrization."""
    found = []
    for name in sorted(music.__all__):
        obj = getattr(music, name, None)
        if predicate(obj):
            found.append(name)
    return found


FUNCTIONS = _public(lambda o: callable(o) and not inspect.isclass(o))
CLASSES = _public(inspect.isclass)


def _signature_params(obj):
    """The parameter names of `obj`, or None if it has no inspectable one.

    A method's first parameter is the instance, and dropping it by
    position rather than by the name ``self`` matters here: the legacy
    synths spell it ``s``, and documenting that as an argument would be
    documenting a fiction.
    """
    is_class = inspect.isclass(obj)
    target = obj.__init__ if is_class else obj
    if is_class and target is object.__init__:
        return []
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return None
    names = [p.lstrip('*') for p in signature.parameters]
    if is_class and names:
        names = names[1:]
    return names


@pytest.mark.parametrize('name', FUNCTIONS)
def test_documented_parameters_are_the_real_parameters(name):
    obj = getattr(music, name)
    actual = _signature_params(obj)
    if actual is None:  # pragma: no cover - builtins
        pytest.skip(f'{name} has no inspectable signature')
    doc = inspect.getdoc(obj) or ''
    documented = documented_names(doc)

    if not actual:
        return
    assert documented is not None, (
        f'{name} takes {actual} and documents no parameters at all')
    assert documented == actual, (
        f'{name}: documented {documented}, signature has {actual}')


@pytest.mark.parametrize('name', CLASSES)
def test_classes_document_what_their_constructors_take(name):
    """A dataclass documents Attributes rather than Parameters, and its
    generated ``__init__`` mirrors them, so either section counts."""
    obj = getattr(music, name)
    actual = _signature_params(obj)
    if not actual:
        return
    doc = inspect.getdoc(obj.__init__) or ''
    class_doc = inspect.getdoc(obj) or ''
    documented = (documented_names(doc)
                  or documented_names(class_doc)
                  or documented_names(class_doc, 'Attributes'))

    assert documented is not None, (
        f'{name}.__init__ takes {actual} and documents none of it')
    undocumented = [p for p in actual if p not in documented]
    assert not undocumented, f'{name} does not document {undocumented}'


@pytest.mark.parametrize('name', FUNCTIONS)
def test_a_routine_that_raises_says_so(name):
    """`ValueError` is part of the contract when it is deliberate.

    Six exports raised one with no Raises section, so the only way to
    discover the guard was to trip it.
    """
    obj = getattr(music, name)
    try:
        source = inspect.getsource(obj)
    except (OSError, TypeError):  # pragma: no cover - builtins
        pytest.skip(f'{name} has no source')
    if 'raise ValueError' not in source:
        return
    doc = inspect.getdoc(obj) or ''
    assert _section(doc, 'Raises') is not None, (
        f'{name} raises ValueError and documents no Raises section')
