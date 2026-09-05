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

#: Why an equation is not checked, where the reason is not "nobody has yet".
#: `100 %` is not the target: some of what the article states is about
#: things this package does not implement, and a test for those would be a
#: test of nothing. Anything absent from this map and uncited is simply
#: outstanding work.
UNIMPLEMENTED = {
    'intervalos': 'interval nomenclature; the package counts semitones and '
                  'does not name intervals',
    'escalas': 'the scale degrees; the package has no scales',
    'relacaoDia': 'the diatonic step pattern, as above',
    'escalasMenores': 'the minor scales, as above',
    'serieHarmonica': 'the harmonic series as a scale, as above',
}

#: Statements the article makes that no test could settle.
NOT_A_CHECK = {
    'vinculos': 'a schema rather than a formula: it says a vibrato rate may '
                'be a function of the note frequency, without fixing which',
}
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
            if label in checked:
                mark, note = 'x', ''
            elif label in UNIMPLEMENTED:
                mark, note = '-', f'  -- {UNIMPLEMENTED[label]}'
            elif label in NOT_A_CHECK:
                mark, note = '.', f'  -- {NOT_A_CHECK[label]}'
            else:
                mark, note = ' ', ''
            print(f'  [{mark}] eq:{label:<16} {section}{note}')

    total = len(found)
    total_hits = sum(1 for label in found if label in checked)
    absent = sum(1 for label in found if label in UNIMPLEMENTED)
    unsettleable = sum(1 for label in found if label in NOT_A_CHECK)
    reachable = total - absent - unsettleable

    print(f'\n  [x] checked                     {total_hits}')
    print(f'  [ ] implemented, not yet checked '
          f'{reachable - total_hits}')
    print(f'  [-] not implemented here         {absent}')
    print(f'  [.] not settleable by a test     {unsettleable}')
    print(f'\n{total_hits} of {total} labelled equations are cited by a test '
          f'({100 * total_hits / total:.0f} %); {total_hits} of {reachable} '
          f'of the ones a test could settle '
          f'({100 * total_hits / reachable:.0f} %)')

    for label in sorted(set(UNIMPLEMENTED) | set(NOT_A_CHECK)):
        if label in checked:
            print(f'\neq:{label} is listed as unreachable but a test cites '
                  f'it; move it out of the map')
            return 1

    stray = checked - set(found)
    if stray:
        print(f'\ncited by a test but not in the article: '
              f'{sorted("eq:" + s for s in stray)}')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
