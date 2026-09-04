"""Every name a docstring points at must be a name that exists.

`note` said `See Also: V, T` and its example called `H`. Those are the MASS
reference implementation's names for `note_with_vibrato`, `tremolo` and
`horizontal_stack`; this package exports none of them. Ninety-two such
references survived the port, so a reader who followed a cross-reference
found nothing and a reader who copied an example got a `NameError`. Eight
more examples were not even parseable, having lost the `...` prompt on their
continuation lines.

This is the companion to test_docstring_signature.py, which checks the names
*inside* a signature. This one checks the names a docstring points *outward*
at: the entries under See Also, and every name an example uses without
binding it first. Like that one it is mechanical, and judges nothing about
whether the reference is a helpful one.
"""

import ast
import builtins
import inspect
import pathlib
import re

import pytest

import music

PACKAGE = pathlib.Path(music.__file__).parent
PUBLIC = {name for name in dir(music) if not name.startswith('_')}
BUILTINS = set(dir(builtins))
#: Names an example may use without this package exporting them.
AMBIENT = {'np', 'numpy', 'music', 'plt', 'sympy'}

_SECTION = re.compile(r'^(See Also|Examples)$')


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


def _see_also_names(doc):
    """The names an entry under See Also points at."""
    body = _section(doc, 'See Also')
    if body is None:
        return []
    names = []
    for line in body:
        if not line.strip() or line.startswith('  '):
            continue            # a continued description, not a new entry
        match = re.match(r'([A-Za-z_][A-Za-z0-9_.]*)\s*(?::|$)', line.strip())
        if match:
            names.append(match.group(1))
    return names


def _example_source(doc):
    """The code of an Examples block, with its doctest prompts removed."""
    body = _section(doc, 'Examples')
    if body is None:
        return None
    code, continuing = [], False
    for line in body:
        stripped = line.strip()
        if stripped.startswith('>>> '):
            code.append(stripped[4:])
            continuing = True
        elif stripped.startswith('... '):
            code.append('    ' + stripped[4:])
        elif stripped in ('>>>', '...'):
            code.append('')
        else:
            continuing = False  # expected output, or prose
        if not continuing and not stripped:
            continue
    return '\n'.join(code)


def _free_names(code):
    """Names the code reads without ever binding them itself."""
    tree = ast.parse(code)
    bound, used = set(), []

    class Walk(ast.NodeVisitor):
        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Store):
                bound.add(node.id)
            else:
                used.append(node.id)

        def visit_arg(self, node):
            bound.add(node.arg)

        def visit_alias(self, node):
            bound.add((node.asname or node.name).split('.')[0])

    Walk().visit(tree)
    return [name for name in dict.fromkeys(used) if name not in bound]


def _documented():
    """Every documented definition in the package, with its module's names."""
    for path in sorted(PACKAGE.rglob('*.py')):
        tree = ast.parse(path.read_text())
        # Only what the module binds at its own top level.  Walking the
        # whole tree would count every local variable in every function
        # body, which let an example call `t()` -- a loop variable in an
        # unrelated routine -- and be taken for a valid reference.
        local = {node.name for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                      ast.ClassDef))}
        local |= {target.id for node in tree.body
                  if isinstance(node, ast.Assign) for target in node.targets
                  if isinstance(target, ast.Name)}
        local |= {(alias.asname or alias.name).split('.')[0]
                  for node in tree.body
                  if isinstance(node, (ast.Import, ast.ImportFrom))
                  for alias in node.names}
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                     ast.ClassDef)):
                continue
            doc = ast.get_docstring(node)
            if doc:
                yield path, node, doc, PUBLIC | BUILTINS | AMBIENT | local


DOCUMENTED = list(_documented())
IDS = [f'{path.relative_to(PACKAGE.parent)}::{node.name}'
       for path, node, _doc, _known in DOCUMENTED]


def test_the_package_is_actually_being_scanned():
    """A scan that silently found nothing would pass every test below."""
    assert len(DOCUMENTED) > 100


@pytest.mark.parametrize('path, node, doc, known', DOCUMENTED, ids=IDS)
def test_see_also_points_at_names_that_exist(path, node, doc, known):
    """A cross-reference to a name the package does not have helps nobody."""
    for name in _see_also_names(doc):
        root = name.split('.')[0]
        assert root in known, (
            f'{node.name} in {path.name} refers under See Also to {name!r}, '
            f'which this package does not export')


@pytest.mark.parametrize('path, node, doc, known', DOCUMENTED, ids=IDS)
def test_examples_parse_as_python(path, node, doc, known):
    """An example that does not parse was never a runnable doctest."""
    source = _example_source(doc)
    if not source:
        return
    try:
        ast.parse(source)
    except SyntaxError as exc:
        pytest.fail(f"{node.name} in {path.name} has an example that does not "
                    f'parse: {exc.msg} (line {exc.lineno}). A continuation '
                    f'line needs its own `...` prompt.')


@pytest.mark.parametrize('path, node, doc, known', DOCUMENTED, ids=IDS)
def test_examples_only_call_names_that_exist(path, node, doc, known):
    """Anyone who copies an example should not meet a NameError."""
    source = _example_source(doc)
    if not source:
        return
    try:
        free = _free_names(source)
    except SyntaxError:
        return              # reported by test_examples_parse_as_python
    for name in free:
        assert name in known, (
            f'{node.name} in {path.name} has an example using {name!r}, '
            f'which this package does not export')


@pytest.mark.parametrize('path, node, doc, known', DOCUMENTED, ids=IDS)
def test_examples_pass_arguments_the_signature_has(path, node, doc, known):
    """An example calling a routine by MASS's parameter names raises.

    `tremolo` was shown as `t(fa=i, dB=j)`, `louds` as
    `note_with_vibrato(d=8)`, and `loud` as `note_with_vibrato(duraton=...)`
    -- two reference names and a typo. All three are `TypeError` for anyone
    who runs them.
    """
    source = _example_source(doc)
    if not source:
        return
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return              # reported by test_examples_parse_as_python
    for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
        if not isinstance(call.func, ast.Name):
            continue
        target = getattr(music, call.func.id, None)
        if not callable(target):
            continue
        try:
            signature = inspect.signature(target)
        except (TypeError, ValueError):
            continue
        parameters = signature.parameters
        if any(p.kind == p.VAR_KEYWORD for p in parameters.values()):
            continue
        for keyword in call.keywords:
            assert keyword.arg is None or keyword.arg in parameters, (
                f'{node.name} in {path.name} has an example calling '
                f'{call.func.id}({keyword.arg}=...), which is not a parameter '
                f'of {call.func.id}{signature}')

        if any(p.kind == p.VAR_POSITIONAL for p in parameters.values()):
            continue
        given = sum(1 for arg in call.args if not isinstance(arg, ast.Starred))
        positional = (inspect.Parameter.POSITIONAL_ONLY,
                      inspect.Parameter.POSITIONAL_OR_KEYWORD)
        accepts = sum(1 for p in parameters.values() if p.kind in positional)
        assert given <= accepts, (
            f'{node.name} in {path.name} has an example passing {given} '
            f'positional arguments to {call.func.id}, which takes {accepts}')
