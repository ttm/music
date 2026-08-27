"""Contract tests for the public API surface.

Every name re-exported from ``music`` is documented as callable with its
default arguments.  Several were not: ``stretches`` and ``trill`` raised
unconditionally, and ``localize_linear`` crashed on its own defaults.  The
sweep below fails if any exported function stops honouring its own
signature again.
"""

import inspect
import warnings

import numpy as np
import pytest

import music

#: Exports whose defaults touch the filesystem, an audio device, the network
#: or an external binary, and so cannot run unattended.
SIDE_EFFECTING = {
    "write_wav_mono",
    "write_wav_stereo",
    "read_wav",
    "play_audio",
    "get_engine",
    "setup_engine",
    "make_test_song",
    "print_peal",
}


def _callable_with_defaults(name):
    """Return the export ``name`` if it can be called with no arguments."""
    obj = getattr(music, name)
    if not callable(obj) or inspect.isclass(obj) or name in SIDE_EFFECTING:
        return None
    try:
        signature = inspect.signature(obj)
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return None
    defaulted = 0
    for parameter in signature.parameters.values():
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if parameter.default is inspect.Parameter.empty:
            return None
        defaulted += 1
    # Variadic collectors such as horizontal_stack(*arrays) document no
    # defaults at all, so calling them with nothing is not a documented use.
    return obj if defaulted else None


ZERO_ARG_EXPORTS = sorted(
    name for name in music.__all__ if _callable_with_defaults(name)
)


def test_zero_arg_exports_are_discovered():
    """Guard the sweep itself: it must actually be exercising the API."""
    assert len(ZERO_ARG_EXPORTS) >= 25


@pytest.mark.parametrize("name", ZERO_ARG_EXPORTS)
def test_export_runs_with_its_documented_defaults(name):
    """Every exported function works when called the way its docs show."""
    result = _callable_with_defaults(name)()
    assert result is not None

    array = np.asarray(result)
    if array.dtype.kind == "f":
        assert array.size > 0, f"{name} returned an empty result"
        assert np.isfinite(array).all(), f"{name} returned NaN or infinity"


def test_every_export_resolves_to_a_function_not_a_module():
    """`music.core.filters` has submodules named after the functions it
    re-exports (adsr, fade, loud, reverb, stretches).  When the import cycle
    between filters and synths bound the module instead of the function,
    ``trill`` raised "'module' object is not callable" at runtime.
    """
    import types

    for name in music.__all__:
        obj = getattr(music, name)
        assert not isinstance(obj, types.ModuleType), (
            f"music.{name} is a module, not the {name} function"
        )

    from music.core.synths import notes

    assert not isinstance(notes.adsr, types.ModuleType)


def test_trill_renders_audio():
    """Regression: trill() raised TypeError via the filters import cycle."""
    trill = music.trill(duration=0.2)
    assert trill.shape[0] > 0
    assert np.isfinite(trill).all()


def test_stretches_resamples_to_the_requested_durations():
    """Regression: stretches() raised AttributeError on abandoned scratch
    code, and rounding could index one sample past the fragment."""
    fragment = music.note(220, 1)
    durations = (1, 2, 0.5)
    out = music.stretches(fragment, durations=durations)
    assert abs(out.shape[0] - sum(durations) * 44100) <= len(durations)

    stereo = music.stretches(
        music.note_with_doppler(number_of_samples=44100), durations=durations
    )
    assert stereo.shape[0] == 2
    assert abs(stereo.shape[1] - sum(durations) * 44100) <= len(durations)


def test_louds_pads_the_envelope_to_the_signal():
    """Regression: louds() assigned the padded envelope to the wrong name
    and then raised a broadcasting ValueError."""
    signal = np.ones(44100 * 3)
    out = music.louds(
        durations=(1, 1), trans_devs=(-10, -20), alpha=(1, 1),
        sonic_vector=signal,
    )
    assert out.shape == signal.shape

    short = np.ones(4410)
    assert music.louds(
        durations=(1, 1), trans_devs=(-10, -20), alpha=(1, 1),
        sonic_vector=short,
    ).shape[0] >= short.shape[0]


def test_localize_linear_reports_that_it_is_unimplemented():
    """It used to crash with an opaque TypeError from its own defaults."""
    from music.core.filters.localization import localize_linear

    assert "localize_linear" not in music.__all__
    with pytest.raises(NotImplementedError):
        localize_linear()


@pytest.mark.parametrize("length", [1, 2, 3, 5, 10, 441, 2205])
def test_adsr_envelope_matches_the_signal_length(length):
    """Regression: the sustain stage got a negative length when attack,
    decay and release together outlasted the sound, and single-sample
    stages produced NaN from a 0/0 in the transition builder."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = music.adsr(sonic_vector=np.ones(length))
    assert out.shape[0] == length
    assert np.isfinite(out).all()


@pytest.mark.parametrize(
    "stage",
    ["attack_duration", "decay_duration", "release_duration"],
)
def test_adsr_accepts_a_zero_length_stage(stage):
    """A zero stage used to divide by zero, and once guarded it made the
    envelope longer than requested because fade() reads 0 as 'unset'."""
    envelope = music.adsr(**{stage: 0})
    assert envelope.shape[0] == music.adsr().shape[0]
    assert np.isfinite(envelope).all()
