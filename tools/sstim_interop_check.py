#!/usr/bin/env python3
"""Check independently authored SSTIM against a complete MUSIC sidecar.

Example:
  python tools/sstim_interop_check.py \
      tests/fixtures/sstim_third_party_am.ttl \
      tests/fixtures/sstim_third_party_am.contract.json

All input is local. Official validation is opt-in and can fetch the
pinned SSTIM 0.19.0 closure on a first run. SHA256 identifies an
installation-specific float64 rendering, not cross-engine PCM identity.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

import music
from music.stimulation.sstim_contract import (
    render_sstim_contract, resolve_sstim_contract,
)
from music.stimulation.sstim_io import validate_sstim


def check(turtle, sidecar, *, official=False, wav=None):
    """Return report for an independently authored local stimulus."""
    path = Path(sidecar)
    if not path.is_file() or path.stat().st_size > 8192:
        raise ValueError("sidecar must be an existing local JSON under 8 KiB")
    try:
        params = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise ValueError("invalid sidecar JSON") from exc
    if (not isinstance(params, dict) or
            set(params) != {"generator", "parameters", "duration",
                            "sample_rate"} or
            not isinstance(params["parameters"], dict)):
        raise ValueError("unexpected or incomplete sidecar shape")

    resolution = resolve_sstim_contract(
        Path(turtle), **params)
    conforms = None
    if official:
        status = validate_sstim(Path(turtle), version="0.19.0")
        conforms = bool(status.ok)
        if not conforms:
            raise ValueError("SSTIM 0.19.0 Full-profile validation failed")
    samples = np.asarray(render_sstim_contract(resolution), dtype=float)
    if samples.ndim not in (1, 2) or not np.isfinite(samples).all():
        raise ValueError("renderer returned invalid samples")
    canonical = np.ascontiguousarray(samples, dtype="<f8")
    result = {
        "contract": "MUSIC explicit sidecar, not normative SSTIM DSP",
        "sstim_version": "0.19.0",
        "sstim_full_profile_conforms": conforms,
        "engine": "music",
        "music_version": music.__version__,
        "generator": resolution.generator,
        "sample_rate_hz": resolution.sample_rate,
        "shape": list(samples.shape),
        "sample_dtype": "little-endian float64",
        "render_sha256": sha256(canonical.tobytes()).hexdigest(),
        "sidecar_sha256": sha256(
            json.dumps(params, sort_keys=True, separators=(",", ":"))
            .encode("utf-8")).hexdigest(),
        "rms": float(np.sqrt(np.mean(samples**2))),
        "peak": float(np.max(np.abs(samples))),
        "exact_pcm_cross_vendor": False,
    }
    if wav:
        import soundfile as sf
        output = samples.T if samples.ndim == 2 else samples
        sf.write(str(wav), output, resolution.sample_rate, subtype="PCM_24")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("turtle", help="Local standard SSTIM 0.19.0 Turtle")
    parser.add_argument("sidecar", help="Complete MUSIC engine parameters JSON")
    parser.add_argument("--official", action="store_true",
                        help="Run the pinned official SHACL validation")
    parser.add_argument("--wav", help="Optional output PCM24 WAV")
    parser.add_argument("--json", help="Write report to this JSON path")
    args = parser.parse_args(argv)
    result = check(args.turtle, args.sidecar,
                   official=args.official, wav=args.wav)
    encoded = json.dumps(result, indent=2) + "\n"
    print(encoded, end="")
    if args.json:
        Path(args.json).write_text(encoded, encoding="utf-8")
    return result


if __name__ == "__main__":
    main()
