"""The scoring of tools/score_singing_asr.py, without a recognizer.

Hearing needs faster-whisper, which the package does not depend on; what
is counted from what was heard, and how variants are compared, does not.
"""
import pytest

from tools import score_singing_asr as asr
from tools.compare_singing import expected_seconds
from tools.compare_singing_timing import SCORES, WORD_LINES, WORDS


@pytest.mark.parametrize("text, said", [
    ("Mar-ry had a litt-le lamb", "Mary had a little lamb"),
    ("laa laa laa laa laa", "La la la la la"),
    ("Al-le mei-ne Ent-chen", "Alle meine Entchen"),
    ("hey ma bro, why fly while dive?", "hey ma bro, why fly while dive?"),
])
def test_a_lyric_is_scored_as_a_listener_would_write_it(text, said):
    assert asr.lyric(text) == said


def test_every_score_s_lyric_is_words_whisper_could_write():
    # The joined syllables of the comparison's scores are words, or have
    # an entry in LYRICS; "Marry" and "laa" were not.
    for score in (*SCORES.values(), *WORDS.values()):
        assert "-" not in asr.lyric(score["text"])
    assert asr.words(asr.lyric(SCORES["mary"]["text"]))[0] == "mary"


def test_words_drop_case_and_punctuation_but_keep_apostrophes():
    assert asr.words("Let's go, Mary!  Bell-ring.") == [
        "let's", "go", "mary", "bell", "ring"]


@pytest.mark.parametrize("wanted, transcribed, count", [
    ("a b c", "a b c", 3),
    ("a b c", "b a c", 2),
    ("a b c", "x y z", 0),
    ("swim stop", "we'll stop kicking", 1),
    # A recognizer that loops, as Whisper does on a long held vowel,
    # hears a word once: a word error rate counts each repeat.
    ("m", " ".join(["m"] * 50), 1),
    ("", "a", 0),
    ("a", "", 0),
])
def test_words_heard_are_the_lyric_s_words_in_order(wanted, transcribed,
                                                   count):
    assert asr.heard(wanted.split(), transcribed.split()) == count


@pytest.mark.parametrize("differences, p", [
    ([1] * 8, 2 / 2 ** 8),
    ([-1] * 8, 2 / 2 ** 8),
    ([1, -1], 1.0),
    ([0, 0, 0], 1.0),
    ([], 1.0),
    ([1, 1, 1, 0, -1], 2 * 5 / 16),
])
def test_the_sign_test_is_exact_two_sided_and_drops_ties(differences, p):
    assert asr.sign_test(differences) == pytest.approx(p)


@pytest.mark.parametrize("name, together", [
    ("1 s words 3", "1 s words"), ("0.25 s words 8", "0.25 s words"),
    ("mary", "mary"), ("test song", "test song"), ("long coda", "long coda"),
])
def test_scores_differing_in_a_final_number_are_summed(name, together):
    assert asr.group(name) == together


def _row(score, variant, heard, forced):
    return {"score": score, "variant": variant, "words": 2,
            "heard": heard, "forced": forced}


def test_the_summary_compares_each_variant_word_by_word_with_baseline():
    scored = {"model": "small", "rows": [
        _row("1 s words 1", "baseline", 0, [["a", -5.0], ["b", -4.0]]),
        _row("1 s words 1", "slowed", 2, [["a", -1.0], ["b", -2.0]]),
        _row("1 s words 2", "baseline", 1, [["c", -3.0], ["d", -3.0]]),
        _row("1 s words 2", "slowed", 1, [["c", -2.0], ["d", -9.0]]),
        _row("mary", "baseline", 2, None),
        _row("mary", "slowed", 2, None),
    ]}
    table = asr.summary(scored)
    assert "| 1 s words | baseline | 1/4 | -15.0 | - | - |" in table
    assert "| 1 s words | slowed | 3/4 | -14.0 | 3/4 | 0.62 |" in table
    # Beyond Whisper's window there is no forced score to compare.
    assert "| mary | slowed | 2/2 | 0.0 | - | - |" in table


def test_the_word_lists_are_distinct_one_syllable_words_on_every_note():
    words = [word for line in WORD_LINES for word in line.split()]
    assert len(words) == len(set(words)) == 48
    lengths = set()
    for score in WORDS.values():
        assert len(score["text"].split()) == len(score["notes"]) \
            == len(score["durs"]) == 6
        lengths.update(expected_seconds(score))
    assert lengths == {0.25, 0.5, 1.0, 2.0}
