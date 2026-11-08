"""Run the Listen page's sounds, and hold its notebook to them.

``docs/listen.rst`` shows each sound under the code that computes it, and
the documentation's build runs that code to write the clips it plays.
Here each block runs on its own and must leave what a page can play:
finite, not silent, one channel or two, and short. The singing block
needs espeak-ng and Praat, which only CI's singing job installs; it is
left out elsewhere. The notebook is written from the same blocks by
``tools/listen.py``, and must be what that writes now, so the page and
the notebook cannot drift apart.
"""
import json
import re

import numpy as np
import pytest
import soundfile

from tools import listen

BLOCKS = listen.blocks()


def test_the_page_has_the_blocks_this_test_thinks_it_has():
    """Guard the extraction: a change to the directive's formatting must
    not quietly leave this file testing nothing."""
    assert len(BLOCKS) == 12
    assert len({block.name for block in BLOCKS}) == len(BLOCKS)
    assert all(block.title and block.code.startswith("import music\n")
               for block in BLOCKS)
    assert [block.name for block in BLOCKS if block.needs] == [
        "plain-changes", "singing"]


@pytest.mark.parametrize("block", BLOCKS, ids=lambda block: block.name)
def test_every_block_leaves_a_sound_a_page_can_play(block, tmp_path,
                                                    monkeypatch):
    lacking = listen.missing(block.needs)
    if lacking:
        pytest.skip(f"needs {', '.join(lacking)}")
    monkeypatch.chdir(tmp_path)
    sound, rate = listen.run(block.code, block.name)
    assert rate == 44100
    assert np.isfinite(sound).all() and np.abs(sound).max() > 0
    assert 1 <= sound.shape[-1] / rate <= 20
    assert sound.ndim == 1 or sound.shape[0] == 2
    assert list(tmp_path.iterdir()) == [], "a block wrote a file"


def test_a_block_cannot_change_the_next_ones_rate():
    import music

    before = music.get_sample_rate()
    sound, rate = listen.run(
        "import music\nmusic.set_sample_rate(8000)\n"
        "sound = music.note(440, duration=0.5)")
    assert rate == 8000 and len(sound) == 4000
    assert music.get_sample_rate() == before


@pytest.mark.parametrize("code, refusal", [
    ("x = 1", "leaves nothing in `sound`"),
    ("import numpy\nsound = numpy.zeros((3, 10))", "shape (3, 10)"),
])
def test_run_refuses_what_a_page_cannot_play(code, refusal):
    with pytest.raises(ValueError, match=re.escape(refusal)):
        listen.run(code)


def test_level_brings_every_clip_to_one_rms_under_the_peak():
    rng = np.random.default_rng(0)
    quiet = 0.01 * np.sin(np.linspace(0, 200 * np.pi, 10000))
    noise = rng.uniform(-1, 1, (2, 10000))
    for sound in (quiet, noise):
        levelled = listen.level(sound)
        assert np.sqrt(np.mean(levelled ** 2)) == pytest.approx(0.1)
        assert np.abs(levelled).max() < 10 ** (listen.PEAK_DB / 20)
    # One loud moment in silence: the RMS would clip it, so the peak wins.
    click = np.zeros(10000)
    click[5000] = 0.5
    assert np.abs(listen.level(click)).max() == pytest.approx(
        10 ** (listen.PEAK_DB / 20))


@pytest.mark.parametrize("sound", [np.zeros(10), np.array([np.nan, 1.0])])
def test_level_refuses_silence_and_what_is_not_finite(sound):
    with pytest.raises(ValueError):
        listen.level(sound)


def test_write_writes_the_levelled_sound_as_flac(tmp_path):
    stereo = np.vstack([np.sin(np.linspace(0, 100 * np.pi, 4410)),
                        np.cos(np.linspace(0, 100 * np.pi, 4410))])
    path = tmp_path / "clip.flac"
    listen.write(path, stereo, 44100)
    read, rate = soundfile.read(path)
    assert soundfile.info(str(path)).format == "FLAC"
    assert rate == 44100 and read.shape == (4410, 2)
    np.testing.assert_allclose(read.T, listen.level(stereo), atol=2 ** -14)


def test_missing_knows_only_the_extras_the_page_uses():
    assert listen.missing(()) == []
    with pytest.raises(ValueError, match="no such extra"):
        listen.missing(("plot",))


def test_the_notebook_is_what_the_page_writes():
    assert listen.NOTEBOOK.read_text() == listen.notebook(), (
        "examples/listen.ipynb is stale; run python tools/listen.py")


def test_the_notebook_installs_then_plays_every_block_in_order():
    cells = json.loads(listen.NOTEBOOK.read_text())["cells"]
    code = ["".join(cell["source"]) for cell in cells
            if cell["cell_type"] == "code"]
    assert code[0].startswith("%pip install")
    played = [source for source in code
              if source.endswith("Audio(sound, rate=music.get_sample_rate())")]
    assert [source.split("\n\nAudio(")[0] for source in played] == [
        block.code for block in BLOCKS]
    assert all(cell["outputs"] == [] for cell in cells
               if cell["cell_type"] == "code")
