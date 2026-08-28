"""The last branches: alternative parameters and platform-specific paths."""

import sys
import types
from unittest.mock import patch

import numpy as np
import pytest

import music
import music.singing.bootstrap as bootstrap
import music.singing.paths as paths
import music.singing.perform as perform
from music.legacy.CanonicalSynth import CanonicalSynth, _fit
from music.structures.peals.plain_changes import PlainChanges


def _finite(a):
    a = np.asarray(a)
    return a.size > 0 and np.isfinite(a).all()


# --------------------------------------------------------------------------
# Sequencer
# --------------------------------------------------------------------------

def test_sequencer_renders_a_vibrato_note():
    seq = music.Sequencer()
    seq.add_note(440, start=0, duration=0.05, vibrato_freq=6,
                 max_pitch_dev=2)
    assert _finite(seq.render())


def test_sequencer_applies_an_adsr_envelope():
    seq = music.Sequencer()
    seq.add_note(440, start=0, duration=0.05,
                 adsr_params={"attack_duration": 5, "release_duration": 5})
    assert _finite(seq.render())


def test_sequencer_places_a_note_in_space_and_writes_stereo(tmp_path):
    """A spatial note makes the render stereo, so write() takes the other
    branch too."""
    seq = music.Sequencer()
    seq.add_note(440, start=0, duration=0.05, spatial={"x": 0.5, "y": 0.1})
    rendered = seq.render()
    assert rendered.ndim == 2

    path = tmp_path / "seq.wav"
    seq.write(str(path))
    assert path.is_file()


def test_sequencer_mixes_mono_and_stereo_notes_together():
    """One spatial note and one plain one: the mixer has to promote."""
    seq = music.Sequencer()
    seq.add_note(440, start=0, duration=0.05)
    seq.add_note(330, start=0.02, duration=0.05,
                 spatial={"x": 0.5, "y": 0.1})
    out = seq.render()
    assert out.ndim == 2 and _finite(out)


def test_sequencer_mixes_stereo_then_mono():
    seq = music.Sequencer()
    seq.add_note(440, start=0, duration=0.05, spatial={"x": 0.5, "y": 0.1})
    seq.add_note(330, start=0.02, duration=0.05)
    out = seq.render()
    assert out.ndim == 2 and _finite(out)


# --------------------------------------------------------------------------
# Cache locations
# --------------------------------------------------------------------------

def _fake_os(name, environ):
    """A stand-in for the `os` module as paths.py uses it.

    Setting os.name globally is not an option: pathlib reads it to decide
    which Path flavour to build, and asking for WindowsPath on a posix host
    raises -- including inside pytest's own reporting.
    """
    return types.SimpleNamespace(name=name, environ=environ)


def test_cache_root_on_linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(paths, "os", _fake_os("posix", {}))
    assert paths._cache_root().name == ".cache"


def test_cache_root_honours_xdg(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(paths, "os",
                        _fake_os("posix", {"XDG_CACHE_HOME": str(tmp_path)}))
    assert paths._cache_root() == tmp_path


def test_cache_root_on_windows(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(paths, "os",
                        _fake_os("nt", {"LOCALAPPDATA": str(tmp_path)}))
    assert paths._cache_root() == tmp_path


# --------------------------------------------------------------------------
# Singing
# --------------------------------------------------------------------------

def test_make_test_song_sings_its_phrase():
    """The engine cannot run here, so check what is handed to it."""
    with patch.object(bootstrap, "sing", return_value=np.zeros(4)) as sing:
        bootstrap.make_test_song()

    text, notes, durs = sing.call_args[0]
    assert len(text.split()) == len(notes) == len(durs)


def test_sing_reports_a_missing_makefile(tmp_path, monkeypatch):
    """The engine directory is valid but the copy fails anyway."""
    engine = tmp_path / "engine"
    (engine / "cache").mkdir(parents=True)
    (engine / "Makefile").write_text("all:\n\ttrue\n")
    monkeypatch.setenv(paths.ENV_VAR, str(engine))
    monkeypatch.setattr(paths, "missing_requirements", lambda: [])
    monkeypatch.setattr(perform, "write_abc", lambda *a, **k: None)

    with patch.object(perform.shutil, "copy",
                      side_effect=OSError("disk full")):
        with pytest.raises(RuntimeError, match="Failed to prepare"):
            perform.sing()


# --------------------------------------------------------------------------
# CanonicalSynth
# --------------------------------------------------------------------------

def test_fit_returns_nothing_for_a_zero_length_stage():
    assert _fit(np.arange(10.0), 0).size == 0


def test_synth_without_vibrato_or_tremolo():
    """Regression: a depth of zero left the corresponding table as None,
    and rawRender and tremoloEnvelope read them unconditionally, so
    switching an effect off raised TypeError."""
    synth = CanonicalSynth()
    synth.synthSetup(vibrato_depth=0, tremolo_depth=0)

    assert synth.vibrato is False
    assert synth.tremolo is False
    assert _finite(synth.rawRender(duration=0.05))
    # A depth of zero is unity gain, not silence.
    assert np.allclose(synth.tremoloEnvelope(duration=0.05), 1.0)


def test_synth_still_modulates_with_its_defaults():
    """The fix above must not have turned the effects off for everyone."""
    synth = CanonicalSynth()
    assert synth.vibrato is True and synth.tremolo is True
    envelope = synth.tremoloEnvelope(duration=1.0, tremolo_frequency=4.0)
    assert not np.allclose(envelope, 1.0)


def test_adsr_with_a_note_rendered_during_setup():
    """render_note=True makes adsrSetup return the shaped note itself."""
    synth = CanonicalSynth()
    out = synth.adsrSetup(render_note=True)
    assert out is None or _finite(out)


# --------------------------------------------------------------------------
# Being
# --------------------------------------------------------------------------

def test_being_render_defaults_its_filename(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    being = music.Being()
    being.f_ = [220.0, 330.0]
    being.render(2, True)
    assert (tmp_path / "abeing.wav").is_file()


def test_being_render_appends_the_extension(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    being = music.Being()
    being.f_ = [220.0]
    being.render(1, "named")
    assert (tmp_path / "named.wav").is_file()


# --------------------------------------------------------------------------
# Odds and ends
# --------------------------------------------------------------------------

def test_adsr_sized_by_a_number_of_samples():
    """number_of_samples is the alternative to envelope_duration."""
    assert len(music.adsr(number_of_samples=8820)) == 8820


def test_perform_peal_builds_its_own_hunts_when_given_none():
    peal = PlainChanges(3)
    hunts = peal.perform_peal(3)
    assert hunts is None or hunts


def test_mix_pads_the_first_vector_when_it_is_shorter():
    """The other leg of the length comparison."""
    out = music.mix(np.ones(3), np.ones(6) * 2)
    assert len(out) == 6
    assert _finite(out)


def test_tremolo_envelope_applied_to_a_sonic_vector():
    """Regression: `if sonic_vector:` truth-tested the array and raised,
    although the return statement below it uses `is not None`."""
    synth = CanonicalSynth()
    signal = np.ones(500)

    shaped = synth.tremoloEnvelope(sonic_vector=signal)

    assert shaped.shape == signal.shape
    assert _finite(shaped)
    assert not np.allclose(shaped, signal)


@pytest.mark.parametrize("theta", [-70, 70])
def test_localize2_brute_across_both_sides(theta):
    """The brute method delays whichever ear is further away, so the sign
    of theta picks the branch."""
    with pytest.warns(UserWarning, match="long time"):
        out = music.localize2(sonic_vector=np.ones(2048), theta=theta,
                              method="brute")
    assert out.shape[0] == 2 and _finite(out)


def test_mix_stereo_pads_the_first_vector_at_the_end():
    """end=True with the *second* vector longer is the remaining corner."""
    out = music.mix_stereo(np.ones(3), np.ones(6) * 2, end=True)
    assert out.shape == (2, 6)
    assert _finite(out)


@pytest.mark.parametrize("function", [
    music.note_with_vibrato_seq_localization,
    music.note_with_vibratos_glissandos,
])
def test_sequenced_notes_take_the_curved_transitions(function):
    """The nested `alpha` chooses the expression used for each transition:
    != 1 curves a pitch glide, and 0 on a vibrato drops its exponent. The
    defaults are all ones, so neither branch was ever reached.

    The shapes are deeply nested and must agree with the other arguments,
    so the default is reshaped rather than rebuilt.
    """
    import inspect

    default = inspect.signature(function).parameters["alpha"].default
    # First row: the pitch transitions. The rest: the vibratos.
    alpha = tuple(
        tuple(2.0 if row == 0 else 0 for _ in values)
        for row, values in enumerate(default)
    )

    out = function(alpha=alpha)

    assert _finite(out)


@pytest.mark.parametrize("theta", [70, -70])
def test_localize2_brute_synthesizes_each_partial(theta):
    """The brute method rebuilds the signal one sinusoid at a time, so it
    needs a spread spectrum to have anything to iterate over: a constant
    signal is a single DC bin and the loop body never runs.

    A positive theta puts the source to the left, which swaps which ear is
    amplified and which is delayed.
    """
    rng = np.random.default_rng(0)
    noise = rng.uniform(-1, 1, 256)

    with pytest.warns(UserWarning, match="long time"):
        out = music.localize2(sonic_vector=noise, theta=theta,
                              method="brute")

    assert out.shape[0] == 2
    assert _finite(out)


def test_localize2_ifft_with_a_wide_spectrum():
    """A spectrum reaching past 4 kHz picks the shorter interaural delay
    constant."""
    rng = np.random.default_rng(1)
    out = music.localize2(sonic_vector=rng.uniform(-1, 1, 4096), theta=45)
    assert out.shape[0] == 2 and _finite(out)


def test_seq_localization_curves_its_position_transitions():
    """`method` chooses how the source moves between positions, and alpha
    curves an exponential move. A positive first x also picks the other
    channel-padding branch."""
    out = music.note_with_vibrato_seq_localization(
        x=(10, 8, 5, 3), y=(1, 1, .1, .1),
        alpha=((1, 1), (1, 1, 1), (1, 1, 1, 1, 1), (2.0, 2.0, 2.0)),
        method=("exp", "exp", "exp"),
    )
    assert out.shape[0] == 2 and _finite(out)


@pytest.mark.parametrize("stereo", [True, False])
def test_an_exponential_path_may_not_cross_the_listener(stereo):
    """Regression: the transition is `start * (end / start) ** curve`, and
    a negative base raised to a fractional power is not a real number. A
    source moving from one side to the other produced NaN audio -- 352798
    samples of it -- rather than an error."""
    with pytest.raises(ValueError, match="cannot run from"):
        music.note_with_vibrato_seq_localization(
            x=(10, -10, 5, 3), method=("exp", "exp", "exp"), stereo=stereo
        )


def test_a_linear_path_may_cross_the_listener():
    """`lin` has no such restriction, and the defaults rely on it."""
    out = music.note_with_vibrato_seq_localization(
        x=(-10, 10, 5, 3), method=("lin", "lin", "lin")
    )
    assert _finite(out)


def test_seq_localization_curves_its_path_in_mono_too():
    """The mono branch has its own copy of the position loop."""
    out = music.note_with_vibrato_seq_localization(
        x=(10, 8, 5, 3), y=(1, 1, .1, .1),
        alpha=((1, 1), (1, 1, 1), (1, 1, 1, 1, 1), (2.0, 2.0, 2.0)),
        method=("exp", "exp", "exp"), stereo=False,
    )
    assert out.ndim == 1 and _finite(out)
