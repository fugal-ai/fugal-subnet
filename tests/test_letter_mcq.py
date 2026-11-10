"""The letter rule (graders.letter_mcq) on the reply shapes real models produce.

`letter_mcq` grades MMLU, MMLU-Pro and SuperGPQA, about two thirds of a v2
slice. Until 2026-11-10 it read "answer is" without regard to case and took a
word's first letter, and otherwise looked only at the last 40 characters. On
real replies that lost correct answers (claude-fable-5.2 on SuperGPQA: 10 of 39,
every one a letter on the first line with working after it) and could credit a
wrong one. The rule now reads an explicit marker (a capital standing alone),
then a line that is only the letter, then a marked-up letter opening the
reply, then the old fallback (INVARIANTS, I4).

The cases are shapes taken from the measured replies (1,703 real replies from
the 19 v2 models through the harness, 2026-11-10), not invented formats.
"""
import pytest

from fugal_subnet.graders import letter_mcq


def _grade(reply, gold):
    return letter_mcq({"gold": gold}, reply)


@pytest.mark.parametrize("reply", [
    "C", "**C**", "(C)", "C.", "**Answer: C**", "Answer: C", "The answer is (C).",
    "Working... therefore the correct answer is **C**.",
    "The answer is C, because A and B are ruled out.",
    "The correct option is C", "the answer is option C",
])
def test_the_plain_shapes_are_read(reply):
    assert _grade(reply, "C") == 1


@pytest.mark.parametrize("reply, gold", [
    # a letter line first, then working whose tail names units or other options
    ("**J**\n\nRecrystallization time follows t = 1 h x exp(-1.326) = 0.26 h", "J"),
    ("**I**\n\nds = 0.4802 - 0.4119 = 0.0683 kJ/(kg K) = 68.3 J/(kg K)", "I"),
    ("**G**\n\n- Range -> (p)\n- Height -> (r)\n\nSo A-s, B-p, C-r, D-q.", "G"),
    ("**A**\n\nNote: I chose A as the most likely key rather than the result I would stand behind.", "A"),
    # working first, the letter alone on the last line
    ("Recombination frequency = 400 / 2000 = 20 cM.\n\n**E**", "E"),
    # a heading that ends in "answer", the marked-up letter opening the next line
    ("## Setting Up\n\nFor a good conductor ...\n\n## Final Answer\n\n**B** - delta = 9.19e-7 m", "B"),
    ("working ...\n\n## Final Answer\n\n**E** (1.14e-8 m and 1.14e-9 m)", "E"),
    # the reply opens with the marked-up letter and goes on, on the same line
    ("**E** - of the options listed, this is the only plausible one; sanitation alone would not achieve it.", "E"),
    # a heading line that is only the letter
    ("# Analysis\n\nThis triggers the adult standard of care.\n\n## **A**", "A"),
    # a marker followed by more text
    ("**Answer: C** (4.2 J)", "C"),
    ("Answer: C\n\nExplanation: A is wrong since the field is zero.", "C"),
    ("**Answer: C. The reaction is exothermic because bonds formed release more energy.**", "C"),
])
def test_a_correct_answer_with_text_around_it_is_read(reply, gold):
    assert _grade(reply, gold) == 1
    assert all(_grade(reply, other) == 0 for other in "ABCDEFGHIJ" if other != gold)


@pytest.mark.parametrize("reply, wrong, read", [
    ("The answer is clearly (B)", "C", "B"),          # was credited to C
    ("The answer is definitely B.", "D", "B"),        # was credited to D
    ("So the answer is approximately 4.2 eV, which is option C", "A", "C"),
    ("the answer is a combination of both effects", "A", None),
])
def test_a_word_after_the_marker_is_not_a_letter(reply, wrong, read):
    assert _grade(reply, wrong) == 0
    if read:
        assert _grade(reply, read) == 1


def test_exactly_one_letter_is_read():
    spray = "A\nB\nC\nD\nE\nF\nG\nH\nI\nJ"
    assert [_grade(spray, x) for x in "ABCDEFGHIJ"] == [0] * 9 + [1]
    assert _grade("Answer: B\n\nD", "B") == 1 and _grade("Answer: B\n\nD", "D") == 0
    assert _grade("Answer C is wrong. The answer is D.", "D") == 1


def test_the_last_marker_wins():
    assert _grade("The answer is B. On reflection, the answer is C.", "C") == 1


@pytest.mark.parametrize("reply", ["", None, "I cannot determine this.", "42"])
def test_no_letter_scores_zero(reply):
    assert all(_grade(reply, x) == 0 for x in "ABCDEFGH")


def test_a_heading_followed_by_a_sentence_is_not_a_marker():
    """ "## Answer" then a sentence starting with the article A: only a
    marked-up letter counts after a heading."""
    reply = "## Answer\n\nA good conductor has a small skin depth, so the result is option D."
    assert _grade(reply, "A") == 0


def test_an_answer_naming_two_letters_reads_one_at_most():
    reply = "**A and C**\n\n- the slope means ... -> **A**\n- the intercept means ... -> **C**"
    assert sum(_grade(reply, x) for x in "ABCDEFGHIJ") <= 1
