"""Sphinx configuration for the music package documentation."""

from importlib.metadata import version as _version
from pathlib import Path
import sys

# The repository's root, for tools/listen.py, and _ext, for the sound
# directive the Listen page is written in.
_DOCS = Path(__file__).resolve().parent
sys.path[:0] = [str(_DOCS.parent), str(_DOCS / "_ext")]

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
    "sphinx_sitemap",
    "sphinxext.opengraph",
    "sounds",
    "reexports",
]

# The docstrings in this package are numpydoc-style, with Parameters,
# Returns, See Also, Examples, Notes and References sections.
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
numpydoc_xref_param_type = True

autosummary_generate = True
# The table and the function have one name in two cases, and a file system
# that ignores case -- macOS's, Windows' -- holds one file for both: the
# table's page overwrote the function's, or was never written.
autosummary_filename_map = {"music.CONSONANCE": "music.CONSONANCE-table"}
autodoc_default_options = {
    "members": True,
    # Peals and IteratorSynth list what they inherit in their pages' tables
    # of methods and attributes; without this, those entries linked to
    # nothing, since only a class's own members were documented on its page.
    "inherited-members": True,
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
# What a browser tab, a search result and a link preview call these pages:
# what the package is, in the words someone looking for it would use. The
# sidebar names the version instead, in _templates/sidebar/brand.html.
html_title = "music: extreme-fidelity sound synthesis in Python"
templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = ["sounds.css"]
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

# Where the pages are served. The sitemap lists them against it, each page's
# canonical link points into it, and link previews are built on it.
html_baseurl = "https://ttm.github.io/music/"
# One version in one language, so a page's address is its link alone,
# not the extension's default of "{lang}{version}{link}".
sitemap_url_scheme = "{link}"
# The index, the search page and the highlighted sources serve a reader
# already here; the sitemap lists the pages someone might be looking for.
sitemap_excludes = ["genindex.html", "search.html", "_modules/*"]

ogp_site_url = html_baseurl
# Drawn from the package's own output by tools/social_preview.py. Upload
# the same file as the repository's social preview in GitHub's settings.
ogp_image = "_static/social-preview.png"
ogp_image_alt = ("The spectrogram of a phrase the package renders, gliding "
                 "from note to note with a vibrato, beside its name and "
                 "tagline.")
ogp_custom_meta_tags = [
    '<meta property="og:image:width" content="1280" />',
    '<meta property="og:image:height" content="640" />',
    '<meta name="twitter:card" content="summary_large_image" />',
]

# Every reference must resolve, so that a name that moved, or was never
# documented, fails the build rather than rendering as plain text where a
# link was meant: fourteen did, among them sing's two links to a function
# the reference has never listed, and every member Peals inherits. What
# is let through below is not the package's to resolve -- the plain words
# of numpydoc's parameter types, the annotations numpy's typing spells by
# private paths, and libraries with no inventory here -- or is a module or
# class the reference deliberately leaves to its routines.
nitpicky = True
numpydoc_xref_aliases = {
    "integer": "int",
    "Path": "pathlib.Path",
    "NDArray": "numpy.typing.NDArray",
    "StimulationSession": "music.StimulationSession",
}
numpydoc_xref_ignore = {
    "a", "are", "array_likes", "by", "context", "durations", "each",
    "fewer", "floats", "followed", "function", "in", "integers", "lists",
    "manager", "mapping", "No", "no", "notes", "numeric", "of", "optional",
    "optionally", "or", "pair", "path", "Rendered", "scalars", "sonic",
    "sonic_arrays", "strings", "tuples", "vectors",
}
nitpick_ignore = [
    ("py:class", "numpy._typing._array_like.ArrayLike"),
    ("py:class", "numpy._typing._array_like.NDArray"),
    ("py:class", "numpy.float64"),
    ("py:class", "music._rate._RateSetting"),
    ("py:class", "music.sequencer.NoteEvent"),
    ("py:obj", "NoteEvent"),
    ("py:class", "rdflib.graph.Graph"),
    ("py:obj", "rdflib.Graph"),
    ("py:class", "rdflib.Namespace"),
    ("py:obj", "sstim.Report"),
    ("py:obj", "Permutation"),
    ("py:obj", "sympy.combinatorics.Permutation"),
    ("py:mod", "sounddevice"),
    ("py:mod", "music.stimulation"),
    ("py:mod", "music.theory"),
    ("py:mod", "music.structures"),
    ("py:mod", "music.singing.psola"),
]
