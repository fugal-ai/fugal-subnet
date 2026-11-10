#!/usr/bin/env python3
# Adversarial attack suite (M5 self-bounty).
# Each ATTACK is (name, task, reply, allow_exec, expected, verdict, note).
# `expected` is the grade we ASSERT the checker returns today. BLOCKED attacks
# expect 0; KNOWN residuals pin today's grade (a false pass expects 1, a false
# fail expects 0) with a note on why it's acceptable and where it's mitigated.
# Run: python -m fugal_subnet.attacks.run_attacks
from __future__ import annotations

import time

from fugal_subnet import graders as g
from fugal_subnet.grading_task import build_grader_task

BIG = 10 ** 7


def _cs(constraints):
    return {"domain": "ifbench",
            "checker": {"id": "constraint_set", "params": {"constraints": constraints}}}


def _live(grader_id, benchmark, gold):
    """A pool question turned into a grader task exactly as the validator and
    the TEE harness do it (grading_task.build_grader_task)."""
    return build_grader_task({"gold": gold, "grader_id": grader_id,
                              "benchmark": benchmark, "metadata": {}})


def _io(func, inputs, gold):
    return {"domain": "codegen", "gold": gold,
            "checker": {"id": "exec_io", "params": {"func": func, "inputs": inputs}}}


# (name, task, reply, allow_exec, expected, verdict, note)
ATTACKS = [
    # ---- exec forgery: the headline result ----
    # Each forgery defines a WRONG f (so the NameError path, exit code 1, is not
    # what blocks it) and still exits 0: a grader that trusted the exit code or
    # stdout would pass it. tests/test_v1_rules_gaps.py pins the exit code.
    ("exec_io: sys.exit(0) early-exit forgery",
     _io("f", [[1, 2]], [3]),
     "```python\ndef f(a, b):\n    return a * b\n\nimport sys\nsys.exit(0)\n```", True, 0,
     "BLOCKED", "exits 0 at import, before the driver writes: the parent reads an "
                "empty outfile, and only parent-compared outputs pass"),
    ("exec_io: os._exit(0) forgery",
     _io("f", [[1, 2]], [3]),
     "```python\ndef f(a, b):\n    print(a + b, flush=True)\n    import os\n    os._exit(0)\n```",
     True, 0,
     "BLOCKED", "f prints the right answer then hard-exits 0 mid-driver, skipping "
                "finally/atexit: the outfile stays empty -> fail"),
    ("exec_io: print sabotage (write to stdout not the outfile)",
     _io("f", [[1, 2]], [3]),
     "```python\ndef f(a, b):\n    print(a + b)\n    return a * b\n```", True, 0,
     "BLOCKED", "stdout says 3, f returns 2: stdout is DEVNULL and only the "
                "parent-named outfile counts"),
    ("exec_io: wrong fn passes on 1 colliding input (WHY >=8 inputs)",
     _io("add", [[2, 2]], [4]), "```python\ndef add(a,b): return a*b\n```", True, 1,
     "KNOWN", "2*2==2+2: a single input lets a wrong fn collide; the F3 generator "
              "ALWAYS ships 8 inputs so collisions on all of them are ~impossible"),
    ("exec_io: wrong answer blocked with multiple inputs",
     _io("add", [[2, 2], [3, 4], [1, 5]], [4, 7, 6]),
     "```python\ndef add(a,b): return a*b\n```", True, 0,
     "BLOCKED", "3*4=12 != 7: divergent inputs catch the wrong function"),
    ("exec_io: correct solution still passes (control)",
     _io("add", [[2, 3], [4, 5]], [5, 9]),
     "```python\ndef add(a,b): return a+b\n```", True, 1,
     "CONTROL", "honest correct code must score 1"),

    # ---- legacy exec_unittest: the exploit we RETIRED it for ----
    ("exec_unittest: early-exit forgery (WHY it's legacy-only)",
     {"domain": "humaneval", "test": "def check(f): assert False", "entry": "None",
      "stub": ""}, "```python\nimport sys\nsys.exit(0)\n```", True, 1,
     "KNOWN", "exit-code contract is forgeable; exec_unittest is NEVER used for "
              "new epochs (legacy cache re-grading only) — exec_io replaces it"),

    # ---- resource / hang attacks on the sandbox ----
    ("exec_io: infinite loop (timeout -> fail, no hang)",
     _io("f", [[1]], [1]), "```python\ndef f(x):\n    while True: pass\n```", True, 0,
     "BLOCKED", "kill-tree timeout returns None -> grade 0; must not wedge"),
    ("exec_io: fork/child holding pipe (Windows reaping trap)",
     _io("f", [[1]], [1]),
     "```python\nimport subprocess,sys\nsubprocess.Popen([sys.executable,'-c',"
     "'import time;time.sleep(60)'])\ndef f(x):\n    while True: pass\n```", True, 0,
     "BLOCKED", "new process group + taskkill /T kills the whole tree"),
    ("exec_io: oversize program (cap)",
     _io("f", [[1]], [1]),
     "```python\n# " + "A" * (600_000) + "\ndef f(x): return 1\n```", True, 0,
     "BLOCKED", "program-size cap rejects before exec"),

    # ---- constraint_set string-logic hacks ----
    ("constraint_set: hedging spam ignoring constraints",
     _cs([{"kind": "word_count_range", "min": 3, "max": 6},
          {"kind": "must_include", "s": "ocean"}]),
     "I cannot be sure, but here is a very long rambling answer with no ocean "
     "keyword and far too many words to satisfy the count constraint", False, 0,
     "BLOCKED", "count + include both fail"),
    ("constraint_set: unknown constraint kind never passes",
     {"domain": "ifbench", "checker": {"id": "constraint_set",
      "params": {"constraints": [{"kind": "totally_made_up"}]}}},
     "anything", False, 0, "BLOCKED", "_check_constraint returns False on unknown kind"),
    ("constraint_set: empty constraints never auto-pass",
     _cs([]), "anything", False, 0, "BLOCKED", "empty set grades 0, not vacuous truth"),
    ("constraint_set: json bool smuggled as number",
     _cs([{"kind": "json_object", "required": {"n": "number"}}]),
     '{"n": true}', False, 0, "BLOCKED", "bool excluded from number type"),
    ("constraint_set: fence-strip does not let junk through",
     _cs([{"kind": "json_object", "required": {"n": "number"}}]),
     '```json\n{"n": 5}\n```extra junk after fence', False, 0,
     "BLOCKED", "trailing junk breaks the single-fence regex -> json parse fails"),

    # ---- numeric / letter extraction hacks ----
    ("numeric_final: spray many numbers hoping last matches",
     {"domain": "gsm8k", "gold": "42"},
     "maybe 1 or 2 or 3 or 4 ... the answer is 42", False, 1,
     "KNOWN", "last-number rule is intended; the gold format 'answer is <n>' plus "
              "temp-0 honest replies make this the correct behavior, not an exploit"),
    ("numeric_final: wrong last number fails",
     {"domain": "gsm8k", "gold": "42"}, "the answer is 43", False, 0,
     "BLOCKED", "last number 43 != 42"),
    ("letter_mcq: spray all letters",
     {"domain": "mmlupro", "gold": "C"},
     "could be A or B or C or D. The answer is B", False, 0,
     "BLOCKED", "'answer is' anchor wins over stray letters; picks B, != C"),
    ("letter_mcq: unicode/whitespace around letter",
     {"domain": "mmlupro", "gold": "C"}, "The answer is  C.", False, 1,
     "CONTROL", "honest answer with spacing still parses"),

    # ---- the graders the live pools use, built the way the validator builds them ----
    # integer_exact: aime / aime25 / beyondaime / omnimath_*. Replies come from
    # the routed model inside the TEE, not from the miner, so a grader quirk is
    # a measurement error shared by every miner, not something a miner can aim.
    ("integer_exact: honest integer answer (control)",
     _live("integer_exact", "aime", "42"), "... so the answer is 42", False, 1,
     "CONTROL", "the AIME/omnimath reply shape must score 1"),
    ("integer_exact: self-correction, last integer wrong",
     _live("integer_exact", "omnimath_hard", "42"),
     "the answer is 42, no wait, it is 43", False, 0,
     "BLOCKED", "last number 43 != 42; an earlier gold mention does not count"),
    ("integer_exact: spelled-out number",
     _live("integer_exact", "aime25", "42"), "The answer is forty-two.", False, 0,
     "BLOCKED", "no digits -> no answer -> 0"),
    ("integer_exact: thousands separator is not a sum",
     _live("integer_exact", "omnimath_mid", "42"), "The answer is 1,000,042", False, 0,
     "BLOCKED", "commas stripped: reads 1000042, not 42"),
    ("integer_exact: decimal truncates to gold (42.9 -> 42)",
     _live("integer_exact", "aime", "42"), "the answer is 42.9", False, 1,
     "KNOWN", "int(float()) truncates; the prompt asks for an integer and the TEE "
              "writes the reply, so no miner can aim it"),
    ("integer_exact: -0.5 truncates to gold 0",
     _live("integer_exact", "omnimath_mid", "0"), "-0.5", False, 1,
     "KNOWN", "truncation toward zero; same reason, a model-side quirk every "
              "miner shares"),
    ("integer_exact: exact integer gold above 2**53 fails",
     _live("integer_exact", "omnimath_hard", str(2 ** 53 + 1)),
     f"The answer is {2 ** 53 + 1}", False, 0,
     "KNOWN", "false FAIL via float round-trip; grades every model 0 alike, "
              "and AIME-style golds are nowhere near 2**53"),
    # letter_mcq with options A-J: mmlupro and supergpqa_hard.
    ("letter_mcq: A-J anchored answer J (control)",
     _live("letter_mcq", "supergpqa_hard", "J"), "The answer is (J)", False, 1,
     "CONTROL", "ten-option letters past D must parse"),
    ("letter_mcq: anchored wrong letter beats trailing gold",
     _live("letter_mcq", "mmlupro", "J"), "The answer is B, not J", False, 0,
     "BLOCKED", "the 'answer is' anchor wins over the tail fallback"),
    ("letter_mcq: listing every option A-J",
     _live("letter_mcq", "supergpqa_hard", "G"), "A B C D E F G H I J", False, 0,
     "BLOCKED", "fallback takes the LAST letter (J), not any letter"),
    ("letter_mcq: fallback reads '...I think' as answer I",
     _live("letter_mcq", "mmlupro", "I"), "Hard to choose between them, I think", False, 1,
     "KNOWN", "the pronoun I is a \\b-bounded A-J letter in the last 40 chars; "
              "the TEE writes the reply, and an honest reply names its letter"),

    # ---- empty / degenerate ----
    ("all: empty reply grades 0",
     {"domain": "gsm8k", "gold": "1"}, "", False, 0, "BLOCKED", "no crash, no pass"),
    ("all: None reply grades 0 (no crash)",
     {"domain": "mmlupro", "gold": "A"}, None, False, 0, "BLOCKED", "grade() catches"),
    ("boxed_math: empty boxed fails",
     {"domain": "math500", "gold": "5"}, "\\boxed{}", False, 0,
     "BLOCKED", "no match on empty box"),
]


def main():
    rows, blocked, known, controls, surprises = [], 0, 0, 0, []
    for name, task, reply, allow, expected, verdict, note in ATTACKS:
        t0 = time.time()
        got = g.grade(task, reply, allow)
        dt = time.time() - t0
        ok = got == expected
        rows.append((ok, verdict, name, got, expected, dt, note))
        if not ok:
            surprises.append(name)
        elif verdict == "BLOCKED":
            blocked += 1
        elif verdict == "KNOWN":
            known += 1
        else:
            controls += 1

    print(f"\n{'ATTACK SUITE':44s} got/exp  verdict   secs")
    print("-" * 78)
    for ok, verdict, name, got, exp, dt, note in rows:
        flag = "OK " if ok else "!! "
        print(f"{flag}{name:44.44s} {got}/{exp}     {verdict:7s}  {dt:4.1f}")
        if not ok:
            print(f"    SURPRISE: expected {exp}, got {got} — {note}")
    print("-" * 78)
    print(f"{blocked} blocked, {known} known-residual, {controls} controls, "
          f"{len(surprises)} surprises")
    if surprises:
        print("FAIL — reality diverged from documented behavior:", surprises)
        raise SystemExit(1)
    print("PASS — all attacks behave as documented.")


if __name__ == "__main__":
    main()
