#!/usr/bin/env python3
"""Render another tool's SSTIM description, given what SSTIM cannot say.

Example:
  python tools/sstim_interop_check.py \
      tests/fixtures/sstim_third_party_am.ttl \
      tests/fixtures/sstim_third_party_am.render.json

The JSON holds render_sstim's keyword arguments: the sample rate and the
controls SSTIM has no term for; a duration, if the description's channels
state none; or a profile, to render a beat by a named convention. All
input is local. Validation against SSTIM's Full profile is opt-in, and
fetches the release's modules on a first run. The SHA-256 identifies this
installation's float64 render, not the samples another engine makes.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

import music
from music.stimulation.sstim import (
    SSTIM_VERSION, SstimBeat, SstimStimulus, from_sstim_graph, render_sstim,
    validate_sstim,
)

_KEYS = {"sample_rate", "duration", "controls", "profile"}


def check(turtle, arguments, *, official=False, wav=None):
    """Report on rendering a local description with the given arguments."""
    path = Path(arguments)
    if not path.is_file() or path.stat().st_size > 8192:
        raise ValueError(
            "the arguments must be an existing local JSON file under 8 KiB")
    try:
        given = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise ValueError("the arguments are not valid JSON") from exc
    if (not isinstance(given, dict) or not set(given) <= _KEYS
            or not isinstance(given.get("controls", {}), dict)):
        raise ValueError(
            "the arguments must be an object of sample_rate, duration, "
            "controls and profile, its controls an object")
    conforms = None
    if official:
        conforms = bool(validate_sstim(Path(turtle),
                                       version=SSTIM_VERSION).ok)
        if not conforms:
            raise ValueError(f"the description does not conform to SSTIM "
                             f"{SSTIM_VERSION}'s Full profile")
    described = from_sstim_graph(Path(turtle), **given)
    if not isinstance(described, (SstimStimulus, SstimBeat)):
        raise ValueError("the tool renders a single stimulus, not a session")
    samples = np.asarray(render_sstim(Path(turtle), **given), dtype=float)
    if samples.ndim not in (1, 2) or not np.isfinite(samples).all():
        raise ValueError("the render returned invalid samples")
    canonical = np.ascontiguousarray(samples, dtype="<f8")
    result = {
        "sstim_version": SSTIM_VERSION,
        "sstim_full_profile_conforms": conforms,
        "engine": "music",
        "music_version": music.__version__,
        "generator": described.generator,
        "rendered_by": ("profile " + given["profile"]
                        if isinstance(described, SstimBeat)
                        else "music." + described.generator),
        "sample_rate_hz": described.sample_rate,
        "shape": list(samples.shape),
        "sample_dtype": "little-endian float64",
        "render_sha256": sha256(canonical.tobytes()).hexdigest(),
        "arguments_sha256": sha256(
            json.dumps(given, sort_keys=True, separators=(",", ":"))
            .encode("utf-8")).hexdigest(),
        "rms": float(np.sqrt(np.mean(samples**2))),
        "peak": float(np.max(np.abs(samples))),
        "exact_pcm_cross_vendor": False,
    }
    if wav:
        import soundfile as sf
        output = samples.T if samples.ndim == 2 else samples
        sf.write(str(wav), output, described.sample_rate, subtype="PCM_24")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("turtle", help="A local SSTIM description, in Turtle")
    parser.add_argument("arguments",
                        help="render_sstim's keyword arguments, in JSON")
    parser.add_argument("--official", action="store_true",
                        help="Validate against SSTIM's Full profile first")
    parser.add_argument("--wav", help="Also write the render, as PCM24 WAV")
    parser.add_argument("--json", help="Also write the report to this path")
    args = parser.parse_args(argv)
    result = check(args.turtle, args.arguments,
                   official=args.official, wav=args.wav)
    encoded = json.dumps(result, indent=2) + "\n"
    print(encoded, end="")
    if args.json:
        Path(args.json).write_text(encoded, encoding="utf-8")
    return result


if __name__ == "__main__":
    main()
