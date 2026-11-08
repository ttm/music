#!/usr/bin/env python3
"""The Listen page's sounds: its blocks, run, levelled, and as a notebook.

``docs/listen.rst`` shows each sound under the code that computes it, in
a ``sound`` directive. Three things read those blocks, and all three go
through this module, so that they cannot disagree about what a block is:
the directive (``docs/_ext/sounds.py``), which runs each block when the
documentation is built and writes its clip; ``tests/test_listen.py``,
which runs them in the suite; and the notebook, which plays them in a
browser and is written from them here:

    python tools/listen.py        # rewrites examples/listen.ipynb
"""
import argparse
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import re
import textwrap

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "docs" / "listen.rst"
NOTEBOOK = ROOT / "examples" / "listen.ipynb"
SITE = "https://ttm.github.io/music/"
COLAB = ("https://colab.research.google.com/github/ttm/music/blob/master/"
         "examples/listen.ipynb")

#: Every clip's level, in decibels below full scale: its RMS, unless that
#: would take a sample past the peak, whichever is quieter.
RMS_DB, PEAK_DB = -20.0, -1.0

_SOUND = re.compile(
    r"^\.\. sound:: (?P<name>[a-z0-9-]+)\n"
    r"(?P<options>(?: {3}:[a-z]+:[^\n]*\n)*)"
    r"\n(?P<body>(?:(?: {3}[^\n]*)?\n)+)", re.M)
_HEADING = re.compile(r"^(?P<title>\S[^\n]*)\n-{3,}\n", re.M)

#: What the notebook runs before a block that sings: Colab has neither
#: espeak-ng nor Praat.
_SINGING_SETUP = (
    "# Singing needs espeak-ng, a system program, and Praat. On Colab:\n"
    "!apt-get -qq install --yes espeak-ng > /dev/null\n"
    "%pip install --quiet 'music[singing]'")


@dataclass(frozen=True)
class Block:
    """A sound on the page: its name, its section, its extras, its code."""

    name: str
    title: str
    needs: tuple[str, ...]
    code: str


def blocks(text=None):
    """The page's sound blocks, in the order it shows them."""
    text = PAGE.read_text() if text is None else text
    found = []
    for match in _SOUND.finditer(text):
        headings = list(_HEADING.finditer(text, 0, match.start()))
        options = dict(re.findall(r":([a-z]+): *([^\n]*)",
                                  match.group("options")))
        found.append(Block(
            name=match.group("name"),
            title=headings[-1].group("title") if headings else "",
            needs=tuple(options.get("needs", "").split()),
            code=textwrap.dedent(match.group("body")).strip("\n")))
    return found


def missing(needs):
    """What the extras a block needs lack here, by name; empty if none."""
    lacking = []
    for extra in needs:
        if extra == "singing":
            from music.singing import psola
            lacking += psola.missing_requirements()
        elif extra == "structures":
            if importlib.util.find_spec("sympy") is None:
                lacking.append("sympy")
        else:
            raise ValueError(f"no such extra a block can need: {extra!r}")
    return lacking


def run(code, name="<sound>"):
    """Run a block in a namespace of its own; its sound and sample rate.

    The rate is the default the block leaves, which is put back
    afterwards, so one block cannot change the next one's.
    """
    import music

    before = music.get_sample_rate()
    namespace = {"__name__": "__listen__"}
    try:
        exec(compile(code, name, "exec"), namespace)
        rate = music.get_sample_rate()
    finally:
        music.set_sample_rate(before)
    if "sound" not in namespace:
        raise ValueError(f"{name} leaves nothing in `sound`")
    sound = np.asarray(namespace["sound"], dtype=np.float64)
    if not (sound.ndim == 1 or sound.ndim == 2 and len(sound) == 2):
        raise ValueError(f"{name} leaves a sound of shape {sound.shape}; "
                         f"a page plays one channel, or two as (2, n)")
    return sound, rate


def level(sound, rms_db=RMS_DB, peak_db=PEAK_DB):
    """Scale a sound to the common RMS, or to the peak if that is lower.

    A page of clips at one RMS keeps a noise from being louder than a
    sine only because it is a noise; the peak keeps a clip whose energy
    sits in a few loud moments from clipping.
    """
    sound = np.asarray(sound, dtype=np.float64)
    if not np.isfinite(sound).all():
        raise ValueError("a sound to level must be finite")
    rms = float(np.sqrt(np.mean(sound ** 2)))
    if rms == 0:
        raise ValueError("a silent sound has no level to bring anywhere")
    peak = float(np.max(np.abs(sound)))
    return sound * min(10 ** (rms_db / 20) / rms, 10 ** (peak_db / 20) / peak)


def write(path, sound, rate):
    """Write a sound levelled, as 16-bit FLAC: what the page plays."""
    import soundfile

    soundfile.write(str(path), level(sound).T, rate, format="FLAC",
                    subtype="PCM_16")


def _cell(kind, ident, text):
    cell = {"cell_type": kind, "id": ident, "metadata": {},
            "source": text.splitlines(keepends=True)}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


def notebook(found=None):
    """The notebook that plays the page's blocks, as the JSON written."""
    found = blocks() if found is None else found
    cells = [
        _cell("markdown", "intro",
              "# Listen to music\n\n"
              f"Each cell computes a sound with [music]({SITE}), a Python "
              "package for sound synthesis, and plays it here. "
              f"[The Listen page]({SITE}listen.html) says what each one "
              "is, and why it sounds as it does."),
        _cell("code", "install", "%pip install --quiet 'music[structures]'"),
        # Each block imports music itself, as it does on the page.
        _cell("code", "imports", "from IPython.display import Audio"),
    ]
    for block in found:
        cells.append(_cell("markdown", f"{block.name}-title",
                           f"## {block.title}"))
        if "singing" in block.needs:
            cells.append(_cell("code", f"{block.name}-setup",
                               _SINGING_SETUP))
        cells.append(_cell("code", block.name, block.code + "\n\n"
                           "Audio(sound, rate=music.get_sample_rate())"))
    document = {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(document, indent=1, sort_keys=True,
                      ensure_ascii=False) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=NOTEBOOK,
                        help="where to write the notebook")
    out = parser.parse_args(argv).out
    out.write_text(notebook())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
