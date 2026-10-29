"""Run the code on the documentation's other pages.

`test_tutorial.py` runs the tutorial and `test_readme.py` the README; the
band-limited synthesis guide, the SSTIM guide and the front page held
fifteen blocks that nothing ran. Each page's blocks run here in order, in
one namespace for the page, as a reader working down it would run them.

A block that needs an extra the environment lacks is left out: SciPy for
`render_oversampled`, and RDFLib for the whole SSTIM page. The SSTIM page
reads a fixture by its path in the repository, so the scratch directory
holds a copy at that path. And it validates a description, which fetches
SSTIM's release from the network the first time: the suite does not use
the network, so a validator that passes stands in, and the page's own
steps around it run. Conformance itself is checked where ``SSTIM_LIVE=1``
lets the network be used, in `test_sstim_stimuli.py`.
"""

import importlib.util
import pathlib
import re
import shutil
import sys
import textwrap
from types import SimpleNamespace

import pytest

DOCS = pathlib.Path(__file__).parent.parent / "docs"
_BLOCK = re.compile(
    r"^\.\. code-block:: python\n\n((?:(?:[ \t]+[^\n]*)?\n)+)", re.M)

#: Each page, and how many blocks it is known to hold, so a change of the
#: directive's formatting cannot leave this testing nothing.
PAGES = {"bandlimited_synthesis.rst": 5, "sstim.rst": 9, "index.rst": 1}


def _blocks(page):
    return [textwrap.dedent(match.group(1)).strip("\n")
            for match in _BLOCK.finditer((DOCS / page).read_text())]


def _has(module):
    return importlib.util.find_spec(module) is not None


@pytest.mark.parametrize("page", sorted(PAGES))
def test_the_page_has_the_blocks_this_test_thinks_it_has(page):
    assert len(_blocks(page)) == PAGES[page]


@pytest.mark.parametrize("page", sorted(PAGES))
def test_every_block_on_the_page_runs(page, tmp_path, monkeypatch):
    if page == "sstim.rst":
        if not _has("rdflib"):
            pytest.skip("the SSTIM page needs the sstim extra")
        monkeypatch.setitem(sys.modules, "sstim", SimpleNamespace(
            validate=lambda graph, **options: SimpleNamespace(ok=True)))
    fixture = pathlib.Path("tests") / "fixtures"
    shutil.copytree(pathlib.Path(__file__).parent / "fixtures",
                    tmp_path / fixture)
    monkeypatch.chdir(tmp_path)
    namespace: dict = {}
    for index, block in enumerate(_blocks(page)):
        if "render_oversampled" in block and not _has("scipy"):
            continue
        try:
            exec(compile(block, f"{page}[block {index}]", "exec"), namespace)
        except Exception as error:  # pragma: no cover - the failure report
            pytest.fail(f"block {index} of {page} failed with "
                        f"{type(error).__name__}: {error}\n\n{block}")
