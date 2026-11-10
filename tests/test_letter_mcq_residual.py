"""What the pinned letter rule reads today, right and wrong.

`graders.letter_mcq` grades MMLU, MMLU-Pro and SuperGPQA. It is part of the
hash-pinned grading contract, so it cannot be corrected in place: a change is
a new grader version with an activation block (INVARIANTS, I4, "the letter
rule misreads some reply formats"; MAINNET_LAUNCH, v2.1).

These cases pin the shipped behaviour, including the readings that are wrong,
so the record of what the rule does is executable and a new grader version is
written against it. A case moving here means graders.py changed, which the
hash pin reports first.
"""
import pytest

from fugal_subnet.graders import letter_mcq


def _grade(reply: str, gold: str) -> int:
    return letter_mcq({"gold": gold}, reply)


@pytest.mark.parametrize("reply", [
    "C",
    "**C**",
    "**Answer: C**",
    "The answer is (C).",
    "Working... therefore the correct answer is **C**.",
    "The answer is C, because A and B are ruled out.",
    "The correct option is C",
])
def test_a_reply_that_ends_with_its_letter_is_read_correctly(reply):
    assert _grade(reply, "C") == 1


@pytest.mark.parametrize("reply, read_as", [
    # No "answer is": the fallback takes the last standalone capital A-J in
    # the final 40 characters, so a unit or a later option letter wins.
    ("**Answer: C** (4.2 J)", "J"),
    ("Answer: C\n\nExplanation: A is wrong since the field is zero.", "A"),
    # "answer is" ignores case and takes a word's first letter.
    ("So the answer is approximately 4.2 eV, which is option C", "A"),
])
def test_known_wrong_a_correct_answer_is_graded_zero(reply, read_as):
    assert _grade(reply, "C") == 0
    assert _grade(reply, read_as) == 1


def test_known_wrong_the_letter_falls_outside_the_last_forty_characters():
    reply = "**Answer: C. The reaction is exothermic because bonds formed release more energy.**"
    assert _grade(reply, "C") == 0
    assert all(_grade(reply, letter) == 0 for letter in "ABCDEFGHIJ")


@pytest.mark.parametrize("reply, credited", [
    ("The answer is clearly (B)", "C"),        # the c of "clearly"
    ("The answer is definitely B.", "D"),      # the d of "definitely"
])
def test_known_wrong_a_wrong_answer_is_credited(reply, credited):
    assert _grade(reply, credited) == 1
    assert _grade(reply, "B") == 0
