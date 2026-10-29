"""Every method that takes arguments, given good ones and then bad ones.

`test_public_api.py` calls every export on its defaults, and every method
that runs without arguments; `test_degenerate.py` sets each numeric
parameter of the exports to zero and below. Neither reached a method's
arguments, which are most of what the sequencer, the session, the bonds
and the peals do. The first sweep of them found a peal performed on more
bells than its object was built for failing as an ``IndexError``, a
session taking any ramp shape as the default one, a phase made by hand
refused nowhere, and a stereo sound failing in numpy in the legacy
synthesizers' envelope.

Each method here is called with arguments that work, and then with each
argument changed by its type: a number to zero and below, a sequence to
an empty one, a sound to stereo and to no samples, a mapping to an empty
one. Each call works, with nothing infinite or NaN in what it returns, or
is refused with a ``ValueError`` the package raises itself, naming the
argument it refuses.
"""

import inspect
import numbers
import os
import traceback
import warnings

import numpy as np
import pytest

import music
from music.structures import GenericPeal, InterestingPermutations, PlainChanges

NOTE = music.note(220, 0.05)


def _sequenced(**note):
    sequencer = music.Sequencer()
    sequencer.add_note(**note)
    return sequencer.render()


def _written_sequence(tmp_path, **arguments):
    sequencer = music.Sequencer()
    sequencer.add_note(220, 0, 0.05)
    sequencer.write(str(tmp_path / "sequence.wav"), **arguments)
    return music.read_audio(str(tmp_path / "sequence.wav"))


def _session(**phase):
    session = music.StimulationSession()
    session.add(**phase)
    return session.render()


def _written_session(tmp_path, **arguments):
    session = music.StimulationSession()
    session.add(music.note, duration=0.05)
    session.write(str(tmp_path / "session.wav"), **arguments)
    return music.read_audio(str(tmp_path / "session.wav"))


def _peals():
    peals = music.Peals(4)
    peals.transpositions_peal(peals.permutations[1])
    return peals


def _generic():
    peal = GenericPeal()
    peal.peals = {"rows": PlainChanges(4).peal_direct[:4]}
    return peal


def _performed_change(**arguments):
    changes = PlainChanges(4)
    return changes.perform_change(
        hunts=changes.initialize_hunts(arguments.pop("nelements")),
        nelements=4, **arguments)


def _being(method, **arguments):
    """A Being as its examples prepare one, `method` called on it, and
    four of its notes rendered."""
    being = music.Being()
    being.grid = [220 * 2 ** (step / 12) for step in range(12)]
    being.pointer = 0
    being.seqsize = 3
    being.curseq = "f_"
    being.f_ = [220]
    being.domain = None
    being.perms = PlainChanges(3).peal_direct
    getattr(being, method)(**arguments)
    return being.render(4)


def _acted_all(peal, **arguments):
    peal.act_all(**arguments)
    return peal.acted_peals


def _synthesized(synth, method, **arguments):
    """What `method` returns, or the note the synthesizer renders after
    it, for a method that only sets the synthesizer up."""
    instrument = synth()
    returned = getattr(instrument, method)(**arguments)
    return instrument.render() if returned is None else returned


PACKAGE = os.path.dirname(music.__file__) + os.sep
BONDS = music.Bonds(vibrato_freq=music.proportional(1 / 50))
DOMAIN = [220, 275, 330, 440]

#: Each method that needs arguments: a call of it, and arguments it works
#: with. A call that needs a scratch directory takes ``tmp_path`` first.
CALLS = {
    "Sequencer.add_note": (_sequenced, dict(
        freq=220, start=0.01, duration=0.05, vibrato_freq=5.0,
        max_pitch_dev=2.0, adsr_params={"attack_ms": 10},
        spatial={"theta": 30})),
    "Sequencer.write": (_written_sequence, dict(bit_depth=16)),
    "StimulationSession.add": (_session, dict(
        stimulus=music.note, duration=0.05, ramp=0.01, gain=0.5, freq=330)),
    "StimulationSession.write": (_written_session, dict(bit_depth=16)),
    "StimulusPhase": (music.StimulusPhase, dict(
        stimulus=NOTE, duration=0.05, ramp=0.01, gain=0.5)),
    "Bonds.characteristics": (BONDS.characteristics, dict(freq=440)),
    "Bonds.render": (BONDS.render, dict(freqs=[220, 440], duration=0.05)),
    "Peals.act": (lambda **kw: _peals().act(**kw), dict(
        peal="transposition_peal", domain=DOMAIN)),
    "Peals.even_odd": (lambda **kw: _peals().even_odd(**kw), dict(
        sequence=[1, 0, 3, 2])),
    "Peals.transpositions_peal": (
        lambda **kw: music.Peals(4).transpositions_peal(**kw),
        dict(permutation=music.Peals(4).permutations[1])),
    "GenericPeal.act": (lambda **kw: _generic().act(**kw), dict(
        peal="rows", domain=DOMAIN)),
    "InterestingPermutations.even_odd": (
        lambda **kw: InterestingPermutations(4).even_odd(**kw),
        dict(sequence=[1, 0, 3, 2])),
    "PlainChanges.perform_change": (_performed_change, dict(nelements=4)),
    "PlainChanges.perform_peal": (
        lambda **kw: PlainChanges(4).perform_peal(**kw), dict(nelements=4)),
    "PlainChanges.saturating_hunts": (PlainChanges.saturating_hunts,
                                      dict(nelements=4)),
    "PlainChanges.act": (lambda **kw: PlainChanges(4).act(**kw), dict(
        domain=DOMAIN)),
    "PrimaryTables.make_tables": (
        lambda **kw: music.PrimaryTables().make_tables(**kw),
        dict(size=1024)),
    "CanonicalSynth.adsrApply": (
        lambda **kw: music.CanonicalSynth().adsrApply(**kw),
        dict(audio_vec=NOTE)),
    "IteratorSynth.adsrApply": (
        lambda **kw: music.IteratorSynth().adsrApply(**kw),
        dict(audio_vec=NOTE)),
    "Being.walk": (lambda **kw: _being("walk", **kw), dict(n=3)),
    "Being.stay": (lambda **kw: _being("stay", **kw), dict(n=3)),
    "Being.setSize": (lambda **kw: _being("setSize", **kw), dict(ss=3)),
    "Being.setPerms": (lambda **kw: _being("setPerms", **kw), dict(
        perms=PlainChanges(3).peal_direct)),
    "Being.addSeq": (lambda **kw: _being("addSeq", **kw), dict(
        sequence=[330, 440])),
    "Being.render": (lambda **kw: music.Being().render(**kw), dict(nn=3)),
    "Bonds.note": (BONDS.note, dict(freq=440, duration=0.05)),
    "GenericPeal.act_all": (lambda **kw: _acted_all(_generic(), **kw),
                            dict(domain=DOMAIN)),
    "Peals.act_all": (lambda **kw: _acted_all(_peals(), **kw),
                      dict(domain=DOMAIN)),
    "PlainChanges.act_all": (lambda **kw: _acted_all(PlainChanges(4), **kw),
                             dict(domain=DOMAIN)),
    "PlainChanges.initialize_hunts": (
        lambda **kw: PlainChanges(4).initialize_hunts(**kw),
        dict(nelements=5, nhunts=2)),
}
for _synth in (music.CanonicalSynth, music.IteratorSynth):
    for _method, _arguments in (
            ("synthSetup", dict(vibrato_depth=0.1, vibrato_frequency=2.0,
                                tremolo_depth=3.0, tremolo_frequency=0.2,
                                duration=0.05, fundamental_frequency=220)),
            ("adsrSetup", dict(A=10.0, D=10, S=-5.0, R=10)),
            ("absorbState", dict(duration=0.05, fundamental_frequency=330)),
            ("render", dict(duration=0.05, fundamental_frequency=330)),
            ("render2", dict(duration=0.05, fundamental_frequency=330)),
            ("rawRender", dict(duration=0.05, fundamental_frequency=330)),
            ("tremoloEnvelope", dict(sonic_vector=NOTE, tremolo_depth=3.0)),
    ):
        CALLS[f"{_synth.__name__}.{_method}"] = (
            lambda synth=_synth, method=_method, **kw: _synthesized(
                synth, method, **kw), _arguments)
CALLS["IteratorSynth.renderIterate"] = (
    lambda **kw: music.IteratorSynth().renderIterate(**kw),
    dict(duration=0.05, fundamental_frequency=330))

#: The methods taking arguments that are not called here, and why.
NOT_CALLED = {
    "Being.setPar": "it reads `fgrid` and `fpointer`, which its docstring "
                    "says the caller assigns first, and takes a name",
    "Peals.an_eight_and_forty": "it takes only the name to store the peal "
                                "under",
    "Peals.twenty_all_over": "it takes only the name to store the peal "
                             "under",
}


def _arguments(function, required):
    """The parameters of `function` besides its instance, all of them or
    only those without a default."""
    return [parameter for parameter
            in inspect.signature(function).parameters.values()
            if parameter.name not in ("self", "s")
            and not (required and (
                parameter.default is not inspect.Parameter.empty
                or parameter.kind in (parameter.VAR_POSITIONAL,
                                      parameter.VAR_KEYWORD)))]


def _methods_taking_arguments():
    """Each exported class that cannot be made without arguments -- the
    rest are swept as exports -- and each public method taking any."""
    for name in music.__all__:
        cls = getattr(music, name)
        if not inspect.isclass(cls):
            continue
        if _arguments(cls.__init__, required=True):
            yield name
        for base in cls.__mro__:
            if not base.__module__.startswith("music"):
                continue
            for attribute, member in vars(base).items():
                function = getattr(member, "__func__", member)
                if (not attribute.startswith("_") and callable(function)
                        and not inspect.isclass(function)
                        and _arguments(function, required=False)):
                    yield f"{name}.{attribute}"


def test_every_method_taking_arguments_is_called_here():
    """A method added without a call here would be swept by nothing."""
    assert sorted(set(_methods_taking_arguments())) == sorted(
        [*CALLS, *NOT_CALLED])


def _call(name, tmp_path, **changes):
    call, arguments = CALLS[name]
    arguments = {**arguments, **changes}
    if "tmp_path" in inspect.signature(call).parameters:
        return call(tmp_path, **arguments)
    return call(**arguments)


def _finite(result):
    if isinstance(result, dict):
        return all(_finite(value) for value in result.values())
    if isinstance(result, (list, tuple)):
        return all(_finite(value) for value in result)
    if result is None or callable(result) or isinstance(result, str):
        return True
    try:
        samples = np.asarray(result, dtype=float)
    except (TypeError, ValueError):
        return True
    return bool(np.isfinite(samples).all())


@pytest.mark.parametrize("name", sorted(CALLS))
def test_each_method_works_on_the_arguments_given_for_it(name, tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert _finite(_call(name, tmp_path))


def _changed(value):
    """Each value of `value`'s kind at the edge of what it can be."""
    if isinstance(value, bool):
        return
    if isinstance(value, numbers.Real):
        yield 0
        yield -abs(value)
    elif isinstance(value, np.ndarray):
        yield np.vstack((value, value))
        yield value[:0]
    elif isinstance(value, (list, tuple)):
        yield type(value)()
    elif isinstance(value, dict):
        yield {}


CHANGES = [(name, argument, value)
           for name, (_, arguments) in sorted(CALLS.items())
           for argument, given in arguments.items()
           for value in _changed(given)]


def test_the_sweep_changes_every_kind_of_argument():
    kinds = {type(value).__name__ for _, _, value in CHANGES}
    assert {"int", "float", "ndarray", "list", "dict"} <= kinds
    assert len(CHANGES) >= 60


@pytest.mark.parametrize("name,argument,value", CHANGES,
                         ids=lambda value: (
                             f"{type(value).__name__}{np.shape(value)}"
                             if isinstance(value, (np.ndarray, list))
                             else str(value)))
def test_an_argument_at_its_edge_works_or_is_refused_by_name(
        name, argument, value, tmp_path):
    np.random.seed(0)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = _call(name, tmp_path, **{argument: value})
    except ValueError as refusal:
        frame = traceback.extract_tb(refusal.__traceback__)[-1]
        assert ("raise ValueError" in (frame.line or "")
                and frame.filename.startswith(PACKAGE)), (
            f"{name}({argument}={value!r}) failed where the arithmetic gave "
            f"up, at {frame.filename}:{frame.lineno}: {refusal}")
        assert argument in str(refusal), (
            f"{name}({argument}={value!r}) was refused without naming "
            f"{argument}: {refusal}")
        return
    assert _finite(result), f"{name}({argument}={value!r}) is not finite"
