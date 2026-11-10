"""Hold the sonification example to its data, and to what it says of them.

``examples/sonification.py`` plays a temperature record as pitch. Such a
file is worth hearing only if the sound is the data, which no run of the
script shows: it runs as well with a year mistyped, or a mapping other
than the one its comments describe. So the example is run here once and
then measured as a listener meets it, from the files:

- each year's note sounds at the pitch the mapping gives that year;
- the notes moved to a scale are on the scale, each the nearest, and the
  two files differ by what the script prints that the scale costs;
- the line passes through each year's pitch in the middle of the year
  and nowhere leaves the range of the data, and its ticks open the
  decades;
- nothing clicks but the ticks.

The Listen page plays the same record from a copy of the table, which
has to be the example's.
"""
import contextlib
import io
import pathlib
import re
import runpy
import types

import numpy as np
import pytest

import music
from test_artifacts import clicks
from tools import listen

EXAMPLE = (pathlib.Path(__file__).parent.parent / "examples"
           / "sonification.py")

#: The names of the example this file reads.
NAMES = ("FIRST_YEAR", "HUNDREDTHS", "TONIC", "SEMITONES_PER_DEGREE", "YEAR",
         "SCALE", "celsius", "years", "notes", "on_scale", "path", "clock",
         "tones")

#: A hundredth of a degree, the record's precision, is 12 cents under the
#: example's mapping. `pitch` is good to about half a cent, so a note is
#: on its year's pitch when it is within two.
CENTS = 2

#: The rate the example runs at: the package's default, which a script
#: that sets none gets. The fixture below sets it for itself, because a
#: fixture of the module's scope is made before conftest's, which sets
#: the rate for each test.
RATE = 44100


@pytest.fixture(scope="module")
def example(tmp_path_factory):
    """The example, run once: its names, what it printed and its files."""
    folder = tmp_path_factory.mktemp("sonification")
    printed = io.StringIO()
    with pytest.MonkeyPatch.context() as patch, music.set_sample_rate(RATE):
        patch.chdir(folder)
        with contextlib.redirect_stdout(printed):
            names = runpy.run_path(str(EXAMPLE))
        read = {path.name: music.read_audio(str(path), True)
                for path in folder.iterdir()}
    lacking = [name for name in NAMES if name not in names]
    assert not lacking, (
        f"the example no longer binds {lacking}; this file reads those "
        f"names to check what the example writes")
    return types.SimpleNamespace(
        printed=printed.getvalue(),
        files={name: sound for name, (sound, _rate) in read.items()},
        rates={rate for _sound, rate in read.values()},
        **{name: names[name] for name in NAMES})


def pitch(note, rate=RATE):
    """The frequency of a note's strongest partial, in hertz.

    The spectrum is padded until a bin is a sixth of a hertz, which is a
    cent of the lowest note here: the strongest bin is within half a
    cent of the note.
    """
    size = 1 << 18
    spectrum = np.abs(np.fft.rfft(note * np.hanning(len(note)), size))
    return spectrum.argmax() * rate / size


def pitches(sound, count):
    """The pitch of each of `count` equal notes of a sound, in hertz."""
    return np.array([pitch(note) for note in np.split(sound, count)])


def cents(freqs, reference):
    return 1200 * np.log2(np.asarray(freqs) / reference)


@pytest.fixture(scope="module")
def heard(example):
    """Each year's pitch in the two files of notes, in hertz."""
    return {name: pitches(example.files[f"sonification_{name}.wav"],
                          len(example.years))
            for name in ("notes", "scale")}


def test_the_example_writes_three_files_of_one_length(example):
    assert sorted(example.files) == [
        "sonification_line.wav", "sonification_notes.wav",
        "sonification_scale.wav"]
    assert example.rates == {RATE}
    lengths = {sound.shape for sound in example.files.values()}
    assert lengths == {(round(len(example.years) * example.YEAR * RATE),)}
    assert example.printed.count("\n") == 3
    assert all(name in example.printed for name in example.files)


def test_the_record_is_measured_from_the_baseline_it_names(example):
    """The table is differences from the mean of 1951-1980, so its mean
    over those years is zero, to the hundredth the table is written in."""
    assert example.years[0] == example.FIRST_YEAR
    assert len(example.celsius) == len(example.HUNDREDTHS.split())
    baseline = (example.years >= 1951) & (example.years <= 1980)
    assert baseline.sum() == 30
    assert abs(example.celsius[baseline].mean()) < 0.005


def test_each_year_sounds_at_the_pitch_of_its_temperature(example, heard):
    """The mapping, measured in the file rather than read from the
    script: zero is the tonic and a degree is the interval the constants
    say, so a year's pitch is the tonic times two to the power of its
    semitones over twelve."""
    semitones = example.SEMITONES_PER_DEGREE * np.array(
        example.HUNDREDTHS.split(), dtype=float) / 100
    off = cents(heard["notes"], example.TONIC * 2 ** (semitones / 12))
    assert np.abs(off).max() < CENTS

    # Up is warmer, and equal differences are equal intervals: the record
    # in cents from the tonic is the temperature, at 12 cents a hundredth.
    hundredths = (cents(heard["notes"], example.TONIC)
                  / example.SEMITONES_PER_DEGREE)
    np.testing.assert_array_equal(np.round(hundredths),
                                  np.round(100 * example.celsius))


def test_the_notes_on_the_scale_are_the_nearest_the_scale_has(example,
                                                               heard):
    degrees = np.array(music.scale(example.SCALE))
    assert np.isin(example.on_scale % 12, degrees).all()

    semitones = example.SEMITONES_PER_DEGREE * example.celsius
    # Every pitch of the scale within three octaves either way of the
    # tonic, which is further than the record goes.
    whole = np.concatenate([degrees + 12 * octave for octave in range(-3, 4)])
    nearest = np.abs(semitones[:, None] - whole).min(axis=1)
    np.testing.assert_allclose(np.abs(example.on_scale - semitones), nearest)

    off = cents(heard["scale"], example.TONIC * 2 ** (example.on_scale / 12))
    assert np.abs(off).max() < CENTS


def test_the_scale_costs_what_the_example_prints(example, heard):
    """The cost, heard: the distance between the two files, year by year,
    and the changes the first file has and the second does not."""
    plain = cents(heard["notes"], example.TONIC)
    scaled = cents(heard["scale"], example.TONIC)
    moved = np.abs(scaled - plain) / (100 * example.SEMITONES_PER_DEGREE)
    # A hundredth of a degree is 12 cents and a semitone 100: a step of
    # more than 6 cents is a change, in either file.
    changes = np.abs(np.diff(plain)) > 6
    merged = changes & (np.abs(np.diff(scaled)) < 6)

    line = next(line for line in example.printed.splitlines()
                if line.startswith("sonification_scale.wav"))
    mean, most, flattened, of = re.search(
        r"by ([\d.]+) of a degree on average and ([\d.]+) at most, and "
        r"plays (\d+) of the (\d+) changes", line).groups()
    assert f"the {example.SCALE} scale" in line
    # The print rounds to a hundredth of a degree.
    assert float(mean) == pytest.approx(moved.mean(), abs=0.006)
    assert float(most) == pytest.approx(moved.max(), abs=0.006)
    assert (int(flattened), int(of)) == (merged.sum(), changes.sum())
    # No year moves by more than half the scale's widest step, and the
    # scale does cost something.
    steps = np.diff(np.append(music.scale(example.SCALE), 12))
    assert 0 < moved.max() <= (steps.max() / 2 + 0.01) / (
        example.SEMITONES_PER_DEGREE)
    assert 0 < merged.sum() < changes.sum()


def test_the_line_meets_each_year_in_its_middle_and_stays_in_range(example):
    semitones = example.SEMITONES_PER_DEGREE * example.celsius
    per_year = len(example.notes) // len(example.years)
    middles = per_year * np.arange(len(example.years)) + per_year // 2
    assert len(example.path) == len(example.notes)
    np.testing.assert_allclose(example.path[middles], semitones, atol=1e-9)
    # Straight from one year to the next: nothing between two years is
    # outside them, which a smoother curve would not promise.
    for index in range(len(semitones) - 1):
        low, high = sorted(semitones[index:index + 2])
        between = example.path[middles[index]:middles[index + 1] + 1]
        assert (low - 1e-9 <= between).all() and (between <= high + 1e-9).all()
    # The ends hold the first and the last year.
    np.testing.assert_allclose(example.path[:middles[0]], semitones[0],
                               atol=1e-9)
    np.testing.assert_allclose(example.path[middles[-1]:], semitones[-1],
                               atol=1e-9)


def test_a_tick_opens_each_decade_and_is_louder_at_a_century(example):
    per_year = len(example.notes) // len(example.years)
    opened = [index for index, year in enumerate(example.years)
              if year % 10 == 0]
    assert opened
    ticking = np.zeros(len(example.notes), dtype=bool)
    peaks = {}
    for index in opened:
        start = index * per_year
        # A tick lasts a twentieth of a second.
        ticking[start:start + RATE // 20] = True
        peaks[int(example.years[index])] = np.abs(
            example.clock[start:start + RATE // 20]).max()
    clock = np.zeros(len(example.notes))
    clock[:len(example.clock)] = example.clock
    assert (clock[~ticking] == 0).all()
    assert min(peaks.values()) > 0
    ordinary = next(peak for year, peak in peaks.items() if year % 100)
    for year, peak in peaks.items():
        assert peak == pytest.approx(
            2 * ordinary if year % 100 == 0 else ordinary), year


def test_nothing_clicks_but_the_ticks(example):
    """Each note of the two melodies is shaped before the join, and the
    line's oscillator keeps its phase from one year to the next. The
    third file is not swept as written: a tick of noise is a step by
    design."""
    assert not clicks(example.files["sonification_notes.wav"])
    assert not clicks(example.files["sonification_scale.wav"])
    assert not clicks(example.tones)
    # A step out of silence, or into it, is at an end and not in the
    # samples, so the sweep cannot see one: the drone's table opens at
    # its trough, a quarter of full scale from the silence before it.
    assert example.tones[0] == 0 and example.tones[-1] == 0


def test_the_listen_page_plays_the_examples_record(example):
    """The page shows the table rather than importing it, so that its
    block stands alone; the copy must not drift from the example's."""
    (block,) = [block for block in listen.blocks()
                if block.name == "sonification"]
    table = re.search(r'"""(.*?)"""', block.code, re.S).group(1).split()
    assert table == example.HUNDREDTHS.split()

    # The page's clip is the example's first file. The two reach a
    # frequency by different arithmetic, so one can differ from the other
    # in its last place, and a table read by truncation then takes the
    # neighbouring entry now and then: 0.0002 away, in the triangle's.
    sound, _rate = listen.run(block.code, block.name)
    np.testing.assert_allclose(sound, example.notes, atol=1e-3)

    # And the page's prose names the years the table holds.
    prose = " ".join(listen.PAGE.read_text().split())
    assert (f"from {example.years[0]} to {example.years[-1]}" in prose), (
        "docs/listen.rst names other years than the record's")
