"""Isolated child process for check_wheel.py; no checkout imports allowed."""
from __future__ import annotations

import argparse
from importlib import metadata
import importlib.util
from pathlib import Path
import sys


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _check_origins(target: Path) -> None:
    for name, module in list(sys.modules.items()):
        if name == 'music' or name.startswith('music.'):
            location = getattr(module, '__file__', None)
            inside = (bool(location)
                      and Path(location).resolve().is_relative_to(target))
            _require(inside,
                     f'{name} came from outside the wheel: {location}')


def smoke(target: Path, version: str) -> None:
    """Check artifact provenance before exercising representative behavior."""
    target = target.resolve()
    package = target / 'music'
    _require((package / '__init__.py').is_file(),
             f'the installed wheel has no music package in {target}')
    checkout = Path(__file__).resolve().parent.parent
    sys.path[:] = [str(target)] + [
        entry for entry in sys.path
        if entry and Path(entry).resolve() not in
        (target, checkout, checkout / 'tools')]
    spec = importlib.util.find_spec('music')
    _require(spec is not None and spec.origin is not None
             and Path(spec.origin).resolve().is_relative_to(package),
             f'music does not resolve to the installed wheel: {spec}')
    distribution = metadata.distribution('music')
    metadata_files = [item for item in distribution.files or ()
                      if str(item).endswith('.dist-info/METADATA')]
    _require(len(metadata_files) == 1,
             'installed music distribution has no unique METADATA file')
    metadata_path = Path(distribution.locate_file(metadata_files[0])).resolve()
    _require(metadata_path.is_relative_to(target) and metadata_path.is_file(),
             f'music metadata came from outside the wheel: {metadata_path}')
    _require(distribution.version == version,
             f'installed version {distribution.version} != wheel {version}')
    _require((package / 'py.typed').is_file(),
             'the installed wheel is missing music/py.typed')

    import music
    import numpy as np
    import soundfile as sf

    _check_origins(target)
    _require(music.__version__ == version,
             f'imported version {music.__version__} != wheel {version}')
    np.testing.assert_allclose(
        music.normalize_stereo([[0, 1], [1, 2]], remove_bias=False),
        [[-1, 0], [0, 1]], rtol=0, atol=1e-15)

    for sample_rate in (8000, 48000):
        tone = np.tile([-1.0, 1.0], sample_rate * 3 // 4)
        for stereo in (False, True):
            signal = np.vstack((tone, tone * 0.25)) if stereo else tone
            for extension in ('wav', 'flac'):
                path = Path(f'fade-{sample_rate}-{stereo}.{extension}')
                music.write_audio(signal, str(path), sample_rate=sample_rate,
                                  fades=np.array([100, 150]))
                restored = music.read_audio(str(path))
                _require(restored.shape == signal.shape,
                         f'{path}: writing changed the signal shape')
                _require(sf.info(str(path)).samplerate == sample_rate,
                         f'{path}: wrong output sample rate')
                channel = restored[0] if stereo else restored
                full = np.flatnonzero(np.abs(channel) > 0.9999)
                _require(full.size > 0, f'{path}: no full-level plateau')
                _require(full[0] == sample_rate // 10 - 1,
                         f'{path}: attack duration is not 100 ms')
                _require(full[-1] == len(tone) - sample_rate * 150 // 1000,
                         f'{path}: release duration is not 150 ms')
                plateau = slice(sample_rate // 10,
                                len(tone) - sample_rate * 150 // 1000)
                np.testing.assert_allclose(channel[plateau], tone[plateau],
                                           rtol=0, atol=1 / 32768)
                if stereo:
                    np.testing.assert_allclose(
                        restored[1], restored[0] * 0.25,
                        rtol=0, atol=1 / 32768)

    def constant(number_of_samples, sample_rate):
        return np.ones(number_of_samples)

    for durations in ((0.1, 0.1), (1.0, 0.1, 1.0)):
        session = music.StimulationSession(sample_rate=1000,
                                           ramp_shape='linear')
        for index, duration in enumerate(durations):
            session.add(constant, duration=duration, ramp=1 if index else 0)
        expected = np.ones(round(sum(durations) * 1000))
        np.testing.assert_allclose(session.render(), expected,
                                   rtol=0, atol=1e-15)
    session = music.StimulationSession(sample_rate=1000, end_ramp=1)
    session.add(constant, duration=0.1, ramp=1)
    rendered = session.render()
    _require(rendered.shape == (100,) and rendered[0] == 0
             and rendered[-1] < 0.04 and rendered.max() > 0.98,
             'competing session ramps lost a fade or changed duration')
    _check_origins(target)
    print(f'installed music {version} from {package}')
    print(f'distribution metadata: {metadata_path}')
    print('wheel smoke passed: py.typed, normalization, mono/stereo '
          'WAV/FLAC fades at 8/48 kHz, and short session ramps')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--version', required=True)
    args = parser.parse_args()
    smoke(args.target, args.version)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
