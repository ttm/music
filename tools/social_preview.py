#!/usr/bin/env python3
"""Draw the image a link to the package is previewed with.

The documentation names it as every page's og:image, and GitHub shows it
wherever the repository is linked once it is uploaded as the repository's
social preview (Settings, Social preview; there is no API for it). It is
a phrase the package renders, drawn as its spectrogram: one band-limited
sawtooth line that glides from note to note, each note gaining a vibrato
a moment after it starts, as a singer's does. Only the lettering is not
the package's own output.

Run: python tools/social_preview.py --out docs/_static/social-preview.png
It needs matplotlib, which ``pip install 'music[plot]'`` installs.
"""
import argparse
from pathlib import Path

import numpy as np

from music.core.synths.mipmap import mipmap_frequency_path

RATE = 44100
WIDTH, HEIGHT = 1280, 640
#: Each note: its frequency in hertz, the seconds it is held, and the
#: seconds it then glides for, to the next note's frequency.
PHRASE = (
    (220.00, 0.50, 0.16),
    (329.63, 0.42, 0.10),
    (293.66, 0.28, 0.20),
    (440.00, 0.90, 0.14),
    (392.00, 0.36, 0.12),
    (329.63, 0.70, 0.00),
)
#: Seconds of silence before the phrase, so that the lettering has the
#: left of the picture to itself.
LEAD = 3.4
#: The ramp, from the background to the loudest partial: one hue, the
#: documentation's teal, rising in lightness with the level.
RAMP = ("#0b1416", "#0d3a41", "#0e6b76", "#4fbecb", "#dff5f7")
INK, SECONDARY = "#f0efec", "#c3c2b7"


def _taper(n, rise, fall):
    """Rise from 0 over ``rise`` samples, fall to 0 over ``fall``."""
    shape = np.ones(n)
    shape[:rise] = np.linspace(0, 1, rise, endpoint=False)
    shape[n - fall:] = np.linspace(1, 0, fall)
    return shape


def frequency_path(phrase=PHRASE, rate=RATE, vibrato_freq=5.5,
                   max_pitch_dev=0.35):
    """The phrase's frequency at each sample, vibrato and glides included."""
    pieces = []
    for index, (freq, held, glide) in enumerate(phrase):
        n = round(held * rate)
        t = np.arange(n) / rate
        # The vibrato sets in 0.15 s into the note, takes 0.15 s to reach
        # its depth, and leaves before the note does, so that the glide
        # starts from the note's own frequency.
        depth = max_pitch_dev * np.clip((t - 0.15) / 0.15, 0, 1)
        depth *= np.clip((held - t) / 0.08, 0, 1)
        pieces.append(
            freq * 2 ** (depth * np.sin(2 * np.pi * vibrato_freq * t) / 12))
        if glide:
            pieces.append(np.geomspace(freq, phrase[index + 1][0],
                                       round(glide * rate), endpoint=False))
    return np.concatenate(pieces)


def render(rate=RATE):
    """The phrase as samples, after its lead of silence."""
    line = mipmap_frequency_path(frequency_path(rate=rate), sample_rate=rate,
                                 waveform="sawtooth")
    line *= _taper(len(line), round(0.7 * rate), round(0.45 * rate))
    return np.concatenate((np.zeros(round(LEAD * rate)), line))


def spectrogram(signal, rate=RATE, size=2048, hop=128, bottom=110.0,
                top=3000.0, floor=-66.0):
    """Level in decibels below the loudest bin, frequency by time.

    The band starts above the lowest bins, where the window's leakage
    from the fundamental draws stripes rather than the sound.
    """
    padded = np.pad(signal, size // 2)
    frames = np.lib.stride_tricks.sliding_window_view(padded, size)[::hop]
    spectrum = np.abs(np.fft.rfft(frames * np.hanning(size), axis=1))
    freqs = np.fft.rfftfreq(size, 1 / rate)
    keep = (freqs >= bottom) & (freqs <= top)
    level = 20 * np.log10(spectrum[:, keep] + 1e-12)
    return np.clip(level - level.max(), floor, 0).T


def draw(out):
    """Write the preview, WIDTH by HEIGHT pixels, as a PNG at ``out``."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    ramp = LinearSegmentedColormap.from_list("teal", RAMP)
    figure = plt.figure(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100,
                        facecolor=RAMP[0])
    axes = figure.add_axes((0, 0, 1, 1))
    axes.set_axis_off()
    level = spectrogram(render())
    # A gamma under one lifts the quiet upper partials into view.
    shade = ((level - level.min()) / -level.min()) ** 0.8
    axes.imshow(shade, origin="lower", aspect="auto", cmap=ramp,
                interpolation="antialiased", vmin=0, vmax=1)
    # The background, fading out from behind the lettering, so the line
    # emerges from under it rather than starting at an edge.
    veil = np.zeros((1, WIDTH, 4))
    veil[..., :3] = matplotlib.colors.to_rgb(RAMP[0])
    veil[..., 3] = np.clip((0.6 - np.linspace(0, 1, WIDTH)) / 0.16, 0, 1)
    axes.imshow(veil, aspect="auto", extent=axes.get_xlim() + axes.get_ylim())

    def write(y, text, size, colour=INK, **style):
        figure.text(72 / WIDTH, 1 - y / HEIGHT, text, color=colour,
                    fontsize=size, va="baseline", ha="left", **style)

    write(262, "music", 104, weight="bold")
    write(346, "Extreme-fidelity synthesis", 31)
    write(394, "of musical elements", 31)
    write(462, "Sound in Python, sample by sample", 20, SECONDARY)
    write(574, "pip install music", 20, family="DejaVu Sans Mono")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    figure.canvas.draw()
    pixels = np.asarray(figure.canvas.buffer_rgba())[..., :3]
    plt.close(figure)
    # Opaque RGB: an alpha channel would only add to a file GitHub caps
    # at a megabyte.
    from PIL import Image
    Image.fromarray(pixels).save(out, optimize=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="docs/_static/social-preview.png")
    draw(parser.parse_args(argv).out)


if __name__ == "__main__":
    main()
