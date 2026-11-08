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
    "sphinx_sitemap",
    "sphinxext.opengraph",
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
# What a browser tab, a search result and a link preview call these pages:
# what the package is, in the words someone looking for it would use. The
# sidebar names the version instead, in _templates/sidebar/brand.html.
html_title = "music: extreme-fidelity sound synthesis in Python"
templates_path = ["_templates"]
html_static_path = ["_static"]
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

# Some docstring Examples reference names that are illustrative rather than
# importable; do not let those become nitpick failures.
nitpicky = False
