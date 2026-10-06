#!/usr/bin/env python3
"""How many of a sung lyric's words Whisper hears, variant by variant.

Whether a change makes the singer's words clearer is a listener's call.
This gives the listener a measurement to check against: a speech
recognizer hears each recording of a comparison that
``tools/compare_singing_timing.py`` wrote, and is scored on the lyric.

* **Words heard**: how many of the lyric's words the transcription has, in
  order (their longest common subsequence). Beam search, temperature 0,
  no prompt and no voice-activity filter, so a recording is always heard
  the same way.
* **Forced score**: the log-probability Whisper gives the lyric itself,
  given the recording, word by word. It moves when a word grows more or
  less recognizable, before a transcription changes.

Each variant is compared with the baseline over the same words: how many
score higher, and a two-sided sign test. A recognizer has a language
model, and hears "Mary had a little lamb" through almost anything; the
``--words`` lists of ``compare_singing_timing.py``, one-syllable words in
no order a sentence would put them, leave it less to guess. Scores whose
names differ only in a trailing number are summed together.

Usage
-----
::

    python tools/compare_singing_timing.py --words --out words
    python tools/score_singing_asr.py words

It needs ``pip install faster-whisper``, which the package does not depend
on, and downloads Whisper's ``small`` model, about 460 MB, on first use.
It writes ``asr.<model>.json`` and ``asr.<model>.md`` beside the
comparison's report, so that more than one model's hearing can be kept.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from pathlib import Path

#: What a listener would write for a lyric whose syllables, joined, are
#: not its words: "Mar-ry" is Mary, and "laa" is la.
LYRICS = {"Mar-ry had a litt-le lamb": "Mary had a little lamb",
          "laa laa laa laa laa": "La la la la la"}

#: Whisper reads 30 seconds at a time; the forced score reads only one.
WINDOW = 30.0


def lyric(text):
    """The words a listener would write for a sung `text`."""
    if text in LYRICS:
        return LYRICS[text]
    return " ".join(word.replace("-", "") for word in text.split())


def words(text):
    """`text` as lowercase words, without punctuation."""
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[^\w\s']", " ", text).split()


def heard(wanted, transcribed):
    """How many of the words `wanted` are in `transcribed`, in order."""
    previous = [0] * (len(transcribed) + 1)
    for word in wanted:
        current = [0]
        for j, other in enumerate(transcribed):
            current.append(previous[j] + 1 if word == other
                           else max(previous[j + 1], current[j]))
        previous = current
    return previous[-1]


def sign_test(differences):
    """The two-sided p-value of a sign test; ties are left out."""
    up = sum(difference > 0 for difference in differences)
    down = sum(difference < 0 for difference in differences)
    if not up + down:
        return 1.0
    tail = sum(math.comb(up + down, k) for k in range(min(up, down) + 1))
    return min(1.0, 2 * tail / 2 ** (up + down))


def group(name):
    """The scores `name` is summed with: its name without a final number."""
    return re.sub(r"\s+\d+$", "", name)


def forced(model, tokenizer, audio, text):
    """The log-probability Whisper gives each word of `text`, given `audio`.

    Teacher-forced, with no timestamps, over Whisper's first 30 seconds.
    """
    import numpy as np

    extractor = model.feature_extractor
    features = extractor(audio)
    frames = min(features.shape[-1] - 1, extractor.nb_max_frames)
    window = np.zeros((features.shape[0], extractor.nb_max_frames),
                      features.dtype)
    window[:, :frames] = features[:, :frames]
    tokens = tokenizer.encode(" " + text)
    start = [*tokenizer.sot_sequence, tokenizer.no_timestamps]
    result, = model.model.align(model.encode(window), start, [tokens],
                                frames)
    logs = np.log(np.maximum(result.text_token_probs[:len(tokens)], 1e-12))
    spelled, grouped = tokenizer.split_to_word_tokens(
        tokens + [tokenizer.eot])
    scores, at = [], 0
    for word, pieces in zip(spelled, grouped):
        if pieces == [tokenizer.eot]:
            break
        scores.append([word.strip(), float(logs[at:at + len(pieces)].sum())])
        at += len(pieces)
    return scores


def score(directory, model_name="small"):
    """Hear every recording of the comparison in `directory`."""
    from faster_whisper import WhisperModel, decode_audio
    from faster_whisper.tokenizer import Tokenizer

    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text())
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    tokenizers, rows = {}, []
    for result in report["results"]:
        name, variant = result["score"], result["backend"]
        sung = report["scores"][name]
        language = re.split(r"[-+]", sung.get("lang", "en"))[0]
        if language not in tokenizers:
            tokenizers[language] = Tokenizer(
                model.hf_tokenizer, model.model.is_multilingual,
                task="transcribe", language=language)
        audio = decode_audio(str(directory / result["raw_file"]),
                             sampling_rate=16000)
        segments, _ = model.transcribe(
            audio, language=language, beam_size=5, temperature=0.0,
            condition_on_previous_text=False, vad_filter=False,
            without_timestamps=True)
        transcription = " ".join(s.text.strip() for s in segments).strip()
        want = lyric(sung["text"])
        within = len(audio) / 16000 <= WINDOW
        rows.append({
            "score": name, "variant": variant, "lyric": want,
            "transcription": transcription,
            "words": len(words(want)),
            "heard": heard(words(want), words(transcription)),
            "forced": (forced(model, tokenizers[language], audio, want)
                       if within else None),
        })
        print(f"{name}: {variant}: {transcription!r}", flush=True)
    return {"model": model_name, "rows": rows}


def summary(scored, against=None):
    """Words heard and forced scores by group and variant, as Markdown.

    Each variant is compared with `against`: by default ``baseline``,
    or, in a comparison without one, the first variant sung.
    """
    rows = scored["rows"]
    groups = list(dict.fromkeys(group(row["score"]) for row in rows))
    variants = list(dict.fromkeys(row["variant"] for row in rows))
    if against is None:
        against = "baseline" if "baseline" in variants else variants[0]
    lines = [f"Heard by Whisper `{scored['model']}`.\n",
             "| Scores | Variant | Words heard | Forced score | "
             f"Words scored above `{against}` | Sign test |",
             "|---|---|---:|---:|---:|---:|"]
    for name in groups:
        mine = [row for row in rows if group(row["score"]) == name]
        base = {row["score"]: row for row in mine
                if row["variant"] == against}
        for variant in variants:
            taken = [row for row in mine if row["variant"] == variant]
            if not taken:
                continue
            words_heard = sum(row["heard"] for row in taken)
            total = sum(row["words"] for row in taken)
            usable = [row for row in taken if row["forced"] is not None]
            forced_sum = sum(lp for row in usable for _, lp in row["forced"])
            differences = [
                lp - base_lp for row in usable
                if row["score"] in base
                and base[row["score"]]["forced"] is not None
                for (_, lp), (_, base_lp) in zip(
                    row["forced"], base[row["score"]]["forced"])]
            if variant == against or not differences:
                above, p = "-", "-"
            else:
                up = sum(difference > 0 for difference in differences)
                above = f"{up}/{len(differences)}"
                p = f"{sign_test(differences):.2g}"
            lines.append(
                f"| {name} | {variant} | {words_heard}/{total} | "
                f"{forced_sum:.1f} | {above} | {p} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("directory",
                        help="a directory compare_singing_timing.py wrote")
    parser.add_argument("--model", default="small",
                        help="the Whisper model, as faster-whisper names it")
    parser.add_argument("--against", metavar="VARIANT",
                        help="the variant the others are compared with; "
                             "baseline, or the first sung, if not given")
    args = parser.parse_args(argv)
    directory = Path(args.directory)
    scored = score(directory, args.model)
    stem = f"asr.{Path(args.model).name}"
    (directory / f"{stem}.json").write_text(
        json.dumps(scored, indent=1) + "\n")
    table = summary(scored, args.against)
    (directory / f"{stem}.md").write_text(table)
    print(table, end="")


if __name__ == "__main__":
    main()
