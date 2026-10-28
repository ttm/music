"""The default sample rate, and every routine's use of it.

Every routine that takes a ``sample_rate`` used to default it to 44,100
on its own, so a caller at any other rate had to pass it to every call,
and the one call that did not ran at 44.1 kHz without a word. The package
shipped that defect three times -- the writers' fades, `cross_fade` and
`reverb` -- each found by measuring, long after.

The default now lives in one place, and these tests hold every routine to
it: each one, on its own defaults, renders under `set_sample_rate` exactly
what it renders when handed the rate, and a scan of the source refuses a
routine that declares a rate of its own again.
"""

import ast
import inspect
import pathlib
import threading

import numpy as np
import pytest

import music
from music.core.synths.mipmap import MipmapOscillator
from music.core.synths.polyblep import PolyBLEPOscillator
from music.singing import perform
from test_public_api import ZERO_ARG_EXPORTS, _callable_with_defaults

RATE = 8000


def test_the_default_is_the_rate_every_routine_used_before():
    assert music.get_sample_rate() == 44100
    assert len(music.note(duration=1)) == 44100


def test_set_on_its_own_line_it_lasts():
    music.set_sample_rate(RATE)
    assert music.get_sample_rate() == RATE
    assert len(music.note(duration=1)) == RATE


def test_as_a_context_manager_it_lasts_for_the_block():
    with music.set_sample_rate(RATE) as rate:
        assert rate == RATE
        assert len(music.note(duration=1)) == RATE
    assert music.get_sample_rate() == 44100


def test_the_rate_before_comes_back_however_the_block_ends():
    with pytest.raises(KeyError):
        with music.set_sample_rate(RATE):
            raise KeyError
    assert music.get_sample_rate() == 44100


def test_blocks_nest():
    with music.set_sample_rate(48000):
        with music.set_sample_rate(RATE):
            assert music.get_sample_rate() == RATE
        assert music.get_sample_rate() == 48000
    assert music.get_sample_rate() == 44100


def test_a_rate_given_to_a_routine_wins():
    with music.set_sample_rate(RATE):
        assert len(music.note(duration=1, sample_rate=22050)) == 22050


@pytest.mark.parametrize("rate", [0, -8000, 44100.5, 44100.0, True, "44100",
                                  None])
def test_a_rate_that_is_not_a_positive_whole_number_is_refused(rate):
    with pytest.raises(ValueError, match="positive whole number"):
        music.set_sample_rate(rate)
    assert music.get_sample_rate() == 44100


def test_one_sample_a_second_is_a_rate():
    with music.set_sample_rate(1) as rate:
        assert rate == music.get_sample_rate() == 1


def test_a_numpy_integer_is_a_whole_number():
    with music.set_sample_rate(np.int64(RATE)) as rate:
        assert type(rate) is int
        assert music.get_sample_rate() == RATE


def test_a_new_thread_starts_at_the_default():
    """A setting belongs to the thread that made it, so two threads
    rendering at two rates cannot change each other's sound."""
    seen = []
    music.set_sample_rate(RATE)
    worker = threading.Thread(target=lambda: seen.append(
        music.get_sample_rate()))
    worker.start()
    worker.join()
    assert seen == [44100]
    assert music.get_sample_rate() == RATE


def _takes_a_rate(name):
    function = _callable_with_defaults(name)
    return "sample_rate" in inspect.signature(function).parameters


RATED = sorted(name for name in ZERO_ARG_EXPORTS if _takes_a_rate(name))


def test_the_sweep_finds_the_routines_that_take_a_rate():
    assert len(RATED) >= 35
    assert {"note", "adsr", "reverb", "binaural_beats"} <= set(RATED)


@pytest.mark.parametrize("name", RATED)
def test_every_routine_renders_at_the_default_rate(name):
    """What `name()` renders under the default equals what it renders
    when handed that rate, for every routine that takes one."""
    function = _callable_with_defaults(name)
    np.random.seed(0)
    given = function(sample_rate=RATE)
    with music.set_sample_rate(RATE):
        np.random.seed(0)
        defaulted = function()
    if isinstance(given, tuple):
        for left, right in zip(given, defaulted):
            np.testing.assert_array_equal(left, right)
    else:
        np.testing.assert_array_equal(given, defaulted)


def test_no_routine_declares_a_rate_of_its_own():
    """A default of 44100 on one routine is the defect this replaced:
    that routine alone would ignore the rate the caller set."""
    root = pathlib.Path(music.__file__).parent
    declared = []
    for path in sorted(root.rglob("*.py")):
        if "ecantorix" in path.parts:   # pragma: no cover - a clone
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            arguments = node.args
            positional = arguments.posonlyargs + arguments.args
            defaults = ([None] * (len(positional) - len(arguments.defaults))
                        + list(arguments.defaults))
            pairs = (list(zip(positional, defaults))
                     + list(zip(arguments.kwonlyargs, arguments.kw_defaults)))
            for argument, default in pairs:
                if (argument.arg == "sample_rate" and default is not None
                        and not (isinstance(default, ast.Constant)
                                 and default.value is None)):
                    declared.append(f"{path.relative_to(root)}:"
                                    f"{node.lineno} {node.name}")
    assert not declared, (
        "these declare a sample_rate default of their own, so they ignore "
        f"set_sample_rate: {declared}. Default it to None and resolve it "
        "with music._rate.resolve_rate")


@pytest.mark.parametrize("make", [
    music.Sequencer, music.StimulationSession, MipmapOscillator,
    PolyBLEPOscillator,
], ids=lambda make: make.__name__)
def test_an_object_keeps_the_rate_it_was_made_at(make):
    with music.set_sample_rate(RATE):
        made = make()
    assert made.sample_rate == RATE
    assert make().sample_rate == 44100


def test_a_sequencer_made_at_a_rate_renders_at_it():
    with music.set_sample_rate(RATE):
        sequencer = music.Sequencer()
    sequencer.add_note(440, start=0.5, duration=0.5)
    assert sequencer.render().shape == (RATE,)


def test_a_legacy_synth_takes_the_rate_too():
    with music.set_sample_rate(RATE):
        synth = music.CanonicalSynth()
    assert synth.samplerate == RATE


def test_a_description_is_written_at_the_rate_the_stimulus_would_render():
    pytest.importorskip("rdflib")
    from music.stimulation.sstim import from_sstim_graph, to_sstim_graph

    with music.set_sample_rate(48000):
        graph = to_sstim_graph("binaural_beats", duration=1)
    assert from_sstim_graph(graph).sample_rate == 48000


def test_a_file_read_at_its_rate_and_written_at_it_is_unchanged(tmp_path):
    """The round trip that came back 9 % long and a semitone and a half
    flat: a 48 kHz file, read without its rate and written back."""
    path, again = tmp_path / "in.wav", tmp_path / "out.wav"
    with music.set_sample_rate(48000):
        music.write_wav_mono(music.note(440, 1), str(path))
    samples, rate = music.read_audio(str(path), return_sample_rate=True)
    with music.set_sample_rate(rate):
        music.write_wav_mono(samples, str(again))
    back, back_rate = music.read_audio(str(again), return_sample_rate=True)
    assert back_rate == 48000
    np.testing.assert_allclose(back, samples, atol=2 ** -15)


# ---------------------------------------------------------------------
# Singing, which both backends render at 44,100 Hz
# ---------------------------------------------------------------------

def _sine(freq, seconds, rate, extra=0):
    return np.sin(2 * np.pi * freq * np.arange(int(seconds * rate) + extra)
                  / rate)


def test_a_line_at_the_singers_rate_is_left_as_it_is():
    line = _sine(440, 0.1, 44100)
    assert perform._at_the_default_rate(line) is line


@pytest.mark.parametrize("rate,extra", [
    (48000, 0),      # up, the padded length even: the Nyquist bin splits
    (48000, 147),    # up, the padded length odd
    (22050, 0),      # down
])
def test_a_sung_line_is_resampled_to_the_default_rate(rate, extra):
    line = _sine(440, 1, 44100, extra)
    with music.set_sample_rate(rate):
        resampled = perform._at_the_default_rate(line)
    assert len(resampled) == round(len(line) * rate / 44100)
    # The same sine at the new rate, away from the two ends, where the
    # line starts and stops abruptly rather than fading as a sung one does.
    expected = _sine(440, 1, rate, len(resampled) - int(rate))
    middle = slice(len(resampled) // 4, 3 * len(resampled) // 4)
    np.testing.assert_allclose(resampled[middle], expected[middle],
                               atol=1e-3)
    assert np.abs(resampled).max() == pytest.approx(1)


def test_a_stereo_line_is_resampled_channel_by_channel():
    line = np.vstack((_sine(440, 0.2, 44100), _sine(660, 0.2, 44100)))
    with music.set_sample_rate(48000):
        resampled = perform._at_the_default_rate(line)
    assert resampled.shape == (2, 9600)


def test_an_empty_line_stays_empty():
    with music.set_sample_rate(48000):
        assert perform._at_the_default_rate(np.zeros((2, 0))).shape == (2, 0)


def test_sing_hands_its_line_back_at_the_default_rate(monkeypatch):
    from music.singing import psola

    monkeypatch.setattr(psola, "sing",
                        lambda *args, **kwargs: _sine(440, 1, 44100))
    with music.set_sample_rate(48000):
        sung = music.sing()
    assert len(sung) == 48000
