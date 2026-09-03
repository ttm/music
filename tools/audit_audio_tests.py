#!/usr/bin/env python3
"""Classify every test that renders audio by what it asserts about it.

100% line coverage says every line ran. It does not say anything was
checked about what came out, and for a synthesis package that is the
gap that matters: a test asserting a length and an amplitude range is
satisfied by silence, and by white noise, and by a sine at the wrong
frequency.

This counts the difference. It is the tool that chooses the work for
issue #67 -- verifying each routine against the mathematics it
documents -- by naming the tests that assert nothing about their audio,
so the next batch is picked from a list rather than from memory.

Usage
-----
::

    python tools/audit_audio_tests.py            # summary and the list
    python tools/audit_audio_tests.py --summary  # just the counts

The classification is a heuristic over the test source, not a proof. It
reads what a test mentions, so a test doing something clever the regexes
do not recognise is filed lower than it deserves -- it under-reports
rather than flatters, which is the right direction for a tool that picks
work. Treat the counts as a direction and the list as a worklist, not as
a score.
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import re
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).parent.parent
TESTS = ROOT / "tests"

#: Exports that return rendered audio. A test calling none of these is
#: not about audio and is not counted at all.
RENDERERS = {
    "note", "note_with_vibrato", "note_with_two_vibratos",
    "note_with_glissando", "note_with_glissando_vibrato",
    "note_with_two_vibratos_glissando", "note_with_vibratos_glissandos",
    "note_with_vibrato_seq_localization", "note_with_fm", "note_with_phase",
    "note_with_doppler", "trill", "noise", "gaussian_noise", "silence",
    "tremolo", "tremolos", "am", "adsr", "adsr_stereo", "adsr_vibrato",
    "fade", "cross_fade", "loud", "louds", "localize", "localize2",
    "localize_linear", "reverb", "fir", "iir", "stretches",
    "binaural_beats", "monaural_beats", "isochronic_tones",
    "amplitude_modulation", "frequency_modulation", "modulated_noise",
    "spatial_motion",
}

SHAPE = re.compile(r"\b(len|\.shape|\.size|\.ndim)\b")
FINITE = re.compile(r"\b(isfinite|isnan)\b")
SPECTRAL = re.compile(r"\b(fft|rfft|dominant_freq|envelope_peak|spectrum)\b")
DERIVED = re.compile(r"(np\.sin|np\.cos|2 \*\* |10 \*\* |np\.exp|"
                     r"expected|reference|table\[)")
COMPARE = re.compile(r"\b(allclose|isclose|array_equal|approx|"
                     r"assert_allclose)\b")

#: Best to worst. A test is filed under the strongest thing it does.
ORDER = ["value", "spectral", "compared", "shape/finite only", "other"]

DESCRIPTIONS = {
    "value": "checked against a derived expectation",
    "spectral": "checked as a spectral or perceptual property",
    "compared": "compared to something, but not to a derived expectation",
    "shape/finite only": "only length, shape, range or finiteness",
    "other": "renders audio incidentally; asserts about something else",
}


def classify(source: str) -> str:
    """The strongest claim `source` makes about the audio it renders."""
    if COMPARE.search(source) and DERIVED.search(source):
        return "value"
    if SPECTRAL.search(source):
        return "spectral"
    if COMPARE.search(source):
        return "compared"
    if SHAPE.search(source) or FINITE.search(source):
        return "shape/finite only"
    return "other"


def audit():
    """Every audio-rendering test, as (file, name, classification)."""
    found = []
    for path in sorted(TESTS.glob("test_*.py")):
        source = path.read_text()
        lines = source.splitlines()
        for node in ast.parse(source).body:
            if not (isinstance(node, ast.FunctionDef)
                    and node.name.startswith("test")):
                continue
            called = {
                call.func.attr if isinstance(call.func, ast.Attribute)
                else getattr(call.func, "id", None)
                for call in ast.walk(node) if isinstance(call, ast.Call)
            }
            if not called & RENDERERS:
                continue
            segment = "\n".join(lines[node.lineno - 1:node.end_lineno])
            found.append((path.name, node.name, classify(segment)))
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="store_true",
                        help="print the counts and stop")
    args = parser.parse_args()

    found = audit()
    if not found:
        print("no audio-rendering tests found", file=sys.stderr)
        return 1

    counts = Counter(kind for _, _, kind in found)
    print(f"tests that render audio: {len(found)}\n")
    for kind in ORDER:
        n = counts.get(kind, 0)
        print(f"  {kind:20s} {n:4d}  ({100 * n / len(found):3.0f}%)  "
              f"{DESCRIPTIONS[kind]}")

    checked = counts.get("value", 0) + counts.get("spectral", 0)
    about_audio = len(found) - counts.get("other", 0)
    print(f"\n{checked} of {about_audio} tests that are about the audio "
          f"check what the audio is.")

    if args.summary:
        return 0

    print("\n--- asserting nothing about the audio itself ---")
    for name, test, kind in found:
        if kind == "shape/finite only":
            print(f"  {name}::{test}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
