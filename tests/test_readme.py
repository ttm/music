"""Run the README.

The README is the first code most readers copy, and only its first block
was ever run, by `test_artifacts.py`, for the clicks between its notes.
Every ```python block in it is run here, in order and in one shared
namespace, as a reader working down the page would run them, as
`test_tutorial.py` runs the tutorial's.

Three things a reader's machine has are stood in for. The file the
sample-rate block reads is written first. The KEMAR measurements are
replaced by two short responses, since the suite does not reach the
network: what the block does with them is what is checked. And the
singing block, which needs espeak-ng, Praat and the eCantorix engine, is
left to CI's singing job, as the tutorial's is.
"""

import contextvars
import importlib.util
import pathlib
import re

import numpy as np
import pytest

import music

README = pathlib.Path(__file__).parent.parent / "README.md"
_BLOCK = re.compile(r"```python\n(.*?)```", re.S)
BLOCKS = _BLOCK.findall(README.read_text())

#: What a block needs beyond the dev extra, by a name it uses.
NEEDS = {"render_oversampled": "scipy", "sstim": "rdflib"}


def test_the_readme_has_the_blocks_this_test_thinks_it_has():
    """Guard the extraction, so a change of fence cannot leave this file
    testing nothing."""
    assert len(BLOCKS) >= 15
    assert any("music.sing(" in block for block in BLOCKS)
    assert any("localize_hrtf" in block for block in BLOCKS)


def _runnable(block):
    if "music.sing(" in block:
        return False
    return all(importlib.util.find_spec(module) is not None
               for name, module in NEEDS.items() if name in block)


def test_every_readme_block_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    music.write_wav_mono(music.note(duration=0.1), "in.wav",
                         sample_rate=48000)
    left, right = np.array([1.0, 0.3]), np.array([0.0, 0.6, 0.2])
    monkeypatch.setattr(music, "setup_hrtf", lambda: tmp_path)
    monkeypatch.setattr(music, "hrir",
                        lambda elevation=0, azimuth=90: (
                            elevation, azimuth, left, right))
    namespace: dict = {}

    def run():
        # The sample-rate block sets the default for the rest of the
        # page, as it would for a reader; a copied context keeps it from
        # outliving this test.
        for index, block in enumerate(BLOCKS):
            if not _runnable(block):
                continue
            try:
                exec(compile(block, f"README.md[block {index}]", "exec"),
                     namespace)
            except Exception as error:  # pragma: no cover - the report
                pytest.fail(f"block {index} of the README failed with "
                            f"{type(error).__name__}: {error}\n\n{block}")

    contextvars.copy_context().run(run)
    for written in ("scale.wav", "chord.wav", "bound.wav", "behind.wav",
                    "campanology.wav", "passing.wav", "colours.wav",
                    "session.wav"):
        assert (tmp_path / written).exists(), written
    assert music.get_sample_rate() == 44100
