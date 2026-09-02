"""Sphinx configuration for the music package documentation."""

from importlib.metadata import version as _version

project = "music"
author = "Renato Fabbri, Jacopo Donati"
copyright = "2024-2026, Renato Fabbri"
release = _version("music")
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "numpydoc",
]

# The docstrings in this package are numpydoc-style, with Parameters,
# Returns, See Also, Examples, Notes and References sections.
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
numpydoc_xref_param_type = True

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_member_order = "bysource"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = f"music {release}"
html_theme_options = {
    "source_repository": "https://github.com/ttm/music/",
    "source_branch": "master",
    "source_directory": "docs/",
    "light_css_variables": {
        "color-brand-primary": "#0e6b76",
        "color-brand-content": "#0e6b76",
    },
    "dark_css_variables": {
        "color-brand-primary": "#4fbecb",
        "color-brand-content": "#4fbecb",
    },
}

# Some docstring Examples reference names that are illustrative rather than
# importable; do not let those become nitpick failures.
nitpicky = False
