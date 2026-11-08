"""The ``sound`` directive: a block of Python, and a player for its sound.

    .. sound:: scale
       :needs: structures

       import music

       sound = music.note(440, duration=1)

The page shows the code, and under it a player for the array the code
leaves in ``sound``: run when the page is read, brought to the page's
common level and written as FLAC, which loses nothing. Running, levelling
and writing are ``tools/listen.py``'s, which the tests and the notebook
share. The clips are kept with the doctrees and copied into the built
site when the build ends.

A block whose extra is missing is shown without its sound, and the build
log says so. Where ``MUSIC_DOCS_SOUNDS=strict``, as in the documentation's
workflow, it fails the build instead, so the published page never lacks
one.
"""
import os
from pathlib import Path
import posixpath
import shutil

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.util import logging
from sphinx.util.docutils import SphinxDirective

from tools import listen

FOLDER = "_sounds"
logger = logging.getLogger(__name__)


class Sound(SphinxDirective):
    """Show a block's code, run it, and play the sound it leaves."""

    required_arguments = 1
    has_content = True
    option_spec = {"needs": directives.unchanged}

    def run(self):
        name = self.arguments[0]
        code = "\n".join(self.content)
        shown = nodes.literal_block(code, code, language="python")
        self.set_source_info(shown)
        named = self.env.temp_data.setdefault("sound_names", set())
        if name in named:
            raise self.error(f"two sounds on this page are named {name!r}")
        named.add(name)

        lacking = listen.missing(self.options.get("needs", "").split())
        if lacking:
            message = f"the sound {name!r} needs {', '.join(lacking)}"
            if os.environ.get("MUSIC_DOCS_SOUNDS") == "strict":
                raise self.error(message)
            logger.info(f"{message}; shown without it", location=shown)
            return [shown]
        try:
            sound, rate = listen.run(code, f"{self.env.docname}:{name}")
        except Exception as error:
            raise self.error(f"the sound {name!r} failed: "
                             f"{type(error).__name__}: {error}")

        cache = Path(self.env.doctreedir) / FOLDER
        cache.mkdir(parents=True, exist_ok=True)
        listen.write(cache / f"{name}.flac", sound, rate)
        here = posixpath.dirname(self.env.docname) or "."
        source = posixpath.relpath(f"{FOLDER}/{name}.flac", here)
        label = name.replace("-", " ")
        player = nodes.raw(
            "", f'<audio class="sound" controls preload="metadata" '
                f'src="{source}" aria-label="{label}"></audio>',
            format="html")
        return [shown, player]


def publish(app, exception):
    """Copy the clips into the built site, beside the pages that play them."""
    cache = Path(app.doctreedir) / FOLDER
    if exception is None and app.builder.format == "html" and cache.is_dir():
        shutil.copytree(cache, Path(app.outdir) / FOLDER, dirs_exist_ok=True)


def setup(app):
    app.add_directive("sound", Sound)
    app.connect("build-finished", publish)
    return {"parallel_read_safe": False, "parallel_write_safe": True}
