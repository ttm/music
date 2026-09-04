"""How much of the article the test suite actually checks.

`RECONCILIATION.md` compares the package with the MASS reference
implementation. `tests/test_article.py` compares it with the article the
reference accompanies. This measures the second: it reads the labelled
equations out of the article's LaTeX source, finds which of them a test
names, and reports the rest.

    python tools/article_coverage.py                 # the report
    python tools/article_coverage.py --mass PATH     # against a checkout

An equation counts as checked when a test cites its label, so the number
below is only as honest as the citations are. It is a measure of what has
been looked at, not a proof that the looking was thorough.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.mass_reference import locate  # noqa: E402

TESTS = Path(__file__).resolve().parent.parent / 'tests'
#: The article's own sources, in the order a reader meets them.
SOURCES = ('body.tex', 'spectra.tex', 'notesInMusic.tex')


def equations(doc: Path) -> OrderedDict:
    """Every labelled equation in one source, with the section it sits in."""
    found: OrderedDict = OrderedDict()
    for name in SOURCES:
        path = doc / name
        if not path.is_file():
            continue
        text = path.read_text()
        section = ''
        pattern = r'\\(?:sub)*section\*?\{([^}]*)\}|\\label\{eq:([^}]*)\}'
        for match in re.finditer(pattern, text):
            if match.group(1) is not None:
                section = match.group(1)
            else:
                found[match.group(2)] = (name, section)
    return found


def cited() -> set:
    """The equation labels the test suite names."""
    names = set()
    for path in TESTS.glob('test_*.py'):
        names |= set(re.findall(r'eq:([A-Za-z_-]+)', path.read_text()))
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mass', help='path to a MASS checkout')
    args = parser.parse_args()

    doc = locate(args.mass).parent.parent.parent / 'doc'
    found = equations(doc)
    if not found:
        print(f'no article sources under {doc}')
        return 1
    checked = cited()

    by_source: OrderedDict = OrderedDict()
    for label, (source, section) in found.items():
        by_source.setdefault(source, []).append((label, section))

    for source, entries in by_source.items():
        hits = sum(1 for label, _ in entries if label in checked)
        print(f'\n{source}: {hits} of {len(entries)} equations checked')
        for label, section in entries:
            mark = 'x' if label in checked else ' '
            print(f'  [{mark}] eq:{label:<16} {section}')

    total_hits = sum(1 for label in found if label in checked)
    print(f'\n{total_hits} of {len(found)} labelled equations are cited by a '
          f'test ({100 * total_hits / len(found):.0f} %)')

    stray = checked - set(found)
    if stray:
        print(f'\ncited by a test but not in the article: '
              f'{sorted("eq:" + s for s in stray)}')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
