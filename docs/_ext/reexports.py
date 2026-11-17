"""The documentation of the constants ``music`` re-exports.

A constant is documented by the ``#:`` comment above its assignment, in
the module that defines it. The API reference lists it where a reader
uses it, as ``music.SCALES``, and there autodoc looked for that comment in
``music/__init__.py``, found none, and documented the constant with the
docstring of its type: ``music.ELEVATIONS`` was "Built-in immutable
sequence". This hands autodoc the comment from the defining module.
"""

import sys

from sphinx.pycode import ModuleAnalyzer


def defining_comment(name, value):
    """The ``#:`` comment of the module-level `name` that holds `value`.

    Every module of the package that binds `name` to this very object is
    asked, in a fixed order, and the first that documents it answers;
    None if none does.
    """
    for modname in sorted(sys.modules):
        module = sys.modules[modname]
        if not modname.startswith("music.") or \
                getattr(module, name, None) is not value:
            continue
        comment = ModuleAnalyzer.for_module(modname).find_attr_docs().get(
            ("", name))
        if comment:
            return list(comment)
    return None


def _document(app, what, name, obj, options, lines):
    if what != "data" or not name.startswith("music.") \
            or name.count(".") != 1:
        return
    comment = defining_comment(name.split(".")[1], obj)
    if comment is not None:
        lines[:] = comment


def setup(app):
    app.connect("autodoc-process-docstring", _document)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
