"""A stand-in for the programs the eCantorix engine runs, for the tests
that check what is handed to it without running it."""

import types


def fake_run(cache, calls=None, make_fails=None):
    """`subprocess.run` as the engine's programs would answer it.

    ``make`` writes an empty ``achant.wav`` in `cache`, which the tests
    read through a stubbed ``soundfile.read``, or raises `make_fails`;
    anything else, such as ``espeak --version``, answers with nothing.
    Every command is appended to `calls`, with its keyword arguments.
    """
    def run(command, *args, **kwargs):
        if calls is not None:
            calls.append((command, kwargs))
        if command[0] == "make":
            if make_fails is not None:
                raise make_fails
            (cache / "achant.wav").write_bytes(b"")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    return run
