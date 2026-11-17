"""Protocol v3 benchmark loaders: the sources added or re-mapped for the v3 question pool.

Loaders only. This module does not decide the pool mix, the category targets or the manifest;
`LOADERS_V3` is what a pool builder picks from, next to the v2 sets it carries over.

Rules every loader here follows:

- TRAIN SPLITS ONLY. Where a source has test or validation splits they are held back for
  evaluation and are never loaded here. A set whose only split is its published test set is
  listed in `HELD_BACK_REVIEW` and kept out of `LOADERS_V3`.
- Pinned revisions (`DATASET_REVISIONS_V3`), parquet only: the pinned `datasets` cannot run
  loading scripts.
- Existing checkers only (`graders.py` is immutable), and only questions a compliant reply can
  pass: anything the checker cannot judge the way the source means it is left out, with the
  reason counted in the optional `stats` counter each loader takes.
- Prompt suffixes are consensus text: `NUMERIC_SUFFIX` for every `numeric_final` set, and
  `v2_sets._mcq_prompt` ("Answer with the letter only.") for every `letter_mcq` set.
- Ids come from the source row's own id or from sha256 of its text, never from load order, and
  never share a form with a v1 or v2 id. Every loader returns its questions sorted by id.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter

from datasets import load_dataset

from fugal_subnet.benchmarks.loader import DATASET_REVISIONS
from fugal_subnet.benchmarks.v2_sets import (
    LETTERS,
    UNGRADEABLE_IFEVAL_KINDS,
    _by_hash,
    _gradeable_integer,
    _mcq_prompt,
)
from fugal_subnet.graders import CHECKERS, last_boxed

# Hugging Face revision pins for the v3 sources (commit SHAs, resolved 2026-11-13).
DATASET_REVISIONS_V3: dict[str, str] = {
    "openai/gsm8k": DATASET_REVISIONS["openai/gsm8k"],                            # the v1 pin
    "EleutherAI/hendrycks_math": DATASET_REVISIONS["EleutherAI/hendrycks_math"],  # the v1 pin
    "google/IFEval": DATASET_REVISIONS["google/IFEval"],                          # the v1 pin
    "allenai/RLVR-IFeval": "47c03c73621c4aab2b824b7818681117d662770e",
    "allenai/ai2_arc": "210d026faf9955653af8916fad021475a3f00453",
    # BIG-Bench Hard: `main` (BBH_SCRIPT_REVISION) holds only a loading script, which the pinned
    # `datasets` cannot run. This is the commit of the Hub's own parquet conversion of that
    # revision (refs/convert/parquet). Held back for review; see HELD_BACK_REVIEW.
    "maveriq/bigbenchhard": "a493490a030bfe06c7baa2db022263afbfa04cfb",
}
BBH_SCRIPT_REVISION = "d53c5b10a77edeb29da195f47e6086b29f2f7f74"

V3_LICENCES = {
    "gsm8k_train": "MIT (openai/gsm8k)",
    "math_l12": "MIT (EleutherAI/hendrycks_math)",
    "rlvr_ifeval": "ODC-BY (allenai/RLVR-IFeval)",
    "arc": "CC-BY-SA-4.0 (allenai/ai2_arc)",
    "ifeval": "Apache-2.0 (google/IFEval)",
}
HELD_BACK_LICENCES = {
    "bbh": "MIT (maveriq/bigbenchhard)",
    "rlvr_ifeval_repeat": "ODC-BY (allenai/RLVR-IFeval)",
}

# Loaders that work but are NOT in LOADERS_V3: each needs an owner decision first.
HELD_BACK_REVIEW = {
    "bbh": (
        "BIG-Bench Hard's single split is named `train` but is its published test set, and test "
        "sets are held back for evaluation. It is also read from the Hub's parquet conversion, "
        "because the source revision is script-only."
    ),
    "rlvr_ifeval_repeat": (
        "The source's gold for 'repeat the request' is the whole message, including the "
        "instruction sentence that says it is not part of the request, so a compliant reply "
        "fails it. This loader grades against the message with that sentence removed, which is "
        "a derived gold and not the source's."
    ),
}

NUMERIC_SUFFIX = "End your response with the final answer as a plain number."
ID_HASH_CHARS = 16
WORD_COUNT_NO_MAX = 999999       # how ifeval.py already writes "at least N words"
MIN_FORBIDDEN_WORD_CHARS = 4     # must_exclude is a substring test: "ate" would forbid "later"


def _note(stats: Counter | None, reason: str, n: int = 1) -> None:
    if stats is not None:
        stats[reason] += n


def _text_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:ID_HASH_CHARS]


def _numeric_prompt(question: str) -> str:
    return f"{question}\n\n{NUMERIC_SUFFIX}"


def _grading(q: dict) -> str:
    """What decides a grade besides the prompt: the gold, the checker and its parameters."""
    return json.dumps([q["gold"], q["grader_id"], q["metadata"].get("checker")], sort_keys=True)


def _unique(items: list[dict], stats: Counter | None = None) -> list[dict]:
    """One question per id and per prompt, sorted by id, whatever order the rows came in.

    A repeated prompt is kept once, under its smallest id. Rows that share a prompt but
    disagree on the grading are all dropped: the source contradicts itself and no reply can
    satisfy both. One id on two different prompts is a hash collision or a reused source id,
    and raises."""
    by_id: dict[str, str] = {}
    by_prompt: dict[str, list[dict]] = {}
    for q in sorted(items, key=lambda x: (x["question_id"], json.dumps(x, sort_keys=True))):
        if by_id.setdefault(q["question_id"], q["prompt"]) != q["prompt"]:
            raise ValueError(f"question id {q['question_id']!r} names two different prompts")
        by_prompt.setdefault(q["prompt"], []).append(q)
    out = []
    for group in by_prompt.values():
        if len({_grading(q) for q in group}) > 1:
            _note(stats, "conflicting_duplicate", len(group))
            continue
        _note(stats, "duplicate", len(group) - 1)
        out.append(group[0])
    return sorted(out, key=lambda q: q["question_id"])


def subset(items: list[dict], n: int) -> list[dict]:
    """The first n questions in sha256 order of their id: the subset a pool builder takes when
    it wants fewer than a loader returns, independent of load order (as v2's MMLU subset)."""
    return sorted(_by_hash(items, "question_id", n), key=lambda q: q["question_id"])


# --- gsm8k_train --------------------------------------------------------------

def _gsm8k_train_questions(rows, stats: Counter | None = None) -> list[dict]:
    from fugal_subnet.benchmarks.gsm8k import _extract_answer

    out = []
    for row in rows:
        _note(stats, "rows")
        gold = _extract_answer(row["answer"])
        if not _gradeable_integer(gold):
            _note(stats, "non_integer_gold")
            continue
        out.append({"prompt": _numeric_prompt(row["question"]), "gold": str(int(gold)),
                    "grader_id": "numeric_final", "benchmark": "gsm8k_train",
                    "question_id": f"gsm8k_train_{_text_id(row['question'])}",
                    "metadata": {"split": "train"}})
    return _unique(out, stats)


def load_gsm8k_train(stats: Counter | None = None) -> list[dict]:
    """GSM8K's train split (v1's `gsm8k` is the test split, which stays held back)."""
    ds = load_dataset("openai/gsm8k", "main", split="train", revision=DATASET_REVISIONS_V3["openai/gsm8k"])
    return _gsm8k_train_questions(ds, stats)


# --- math_l12 -----------------------------------------------------------------

MATH_L12_LEVELS = frozenset({"Level 1", "Level 2"})


def _math_l12_questions(config: str, rows, stats: Counter | None = None) -> list[dict]:
    out = []
    for row in rows:
        _note(stats, "rows")
        if row.get("level") not in MATH_L12_LEVELS:
            _note(stats, "other_level")
            continue
        gold = last_boxed(row["solution"])       # the last \boxed{} of the solution, as math_.py
        if not gold:
            _note(stats, "no_boxed_gold")
            continue
        out.append({"prompt": row["problem"], "gold": gold, "grader_id": "boxed_math",
                    "benchmark": "math_l12",
                    "question_id": f"math_train_{config}_{_text_id(row['problem'])}",
                    "metadata": {"level": row["level"], "type": row.get("type", config), "split": "train"}})
    return out


def load_math_l12(stats: Counter | None = None) -> list[dict]:
    """MATH levels 1-2 from the TRAIN split (v1 `math` and v2 `math_l35` are the test split)."""
    from fugal_subnet.benchmarks.math_ import _CONFIGS

    out = []
    for config in _CONFIGS:
        ds = load_dataset("EleutherAI/hendrycks_math", config, split="train",
                          revision=DATASET_REVISIONS_V3["EleutherAI/hendrycks_math"])
        out.extend(_math_l12_questions(config, ds, stats))
    return _unique(out, stats)


# --- constraint_set questions (rlvr_ifeval, ifeval) ------------------------------

def _constraint_question(prompt: str, constraints: list[dict], benchmark: str, question_id: str,
                         **metadata) -> dict:
    return {"prompt": prompt, "gold": "", "grader_id": "constraint_set", "benchmark": benchmark,
            "question_id": question_id,
            "metadata": {"checker": {"id": "constraint_set", "params": {"constraints": constraints}},
                         **metadata}}


def _distinct(words) -> list[str]:
    return list(dict.fromkeys(w for w in (words or []) if isinstance(w, str) and w))


def _word_list(raw) -> list[str] | None:
    """A source word list, repeats removed; None unless it is a non-empty list of non-empty
    strings with no surrounding whitespace (an empty must_include always passes and an empty
    must_exclude never does)."""
    if not isinstance(raw, (list, tuple)) or not raw:
        return None
    if any(not isinstance(w, str) or not w or w != w.strip() for w in raw):
        return None
    return list(dict.fromkeys(raw))


def _witness(constraints: list[dict]) -> str | None:
    """A reply the pinned checker accepts for these constraints, or None when this simple
    construction finds none. Loaders keep a constraint question only with a witness, so a
    contradictory set (all-lowercase plus an end phrase with a capital in it) is left out."""
    starts = list(dict.fromkeys(c["s"] for c in constraints if c["kind"] == "starts_with"))
    ends = list(dict.fromkeys(c["s"] for c in constraints if c["kind"] == "ends_with"))
    if len(starts) > 1 or len(ends) > 1:
        return None
    middle = [" ".join([c["s"]] * c.get("k", 1)) for c in constraints if c["kind"] == "must_include"]
    lo = max([c["min"] for c in constraints if c["kind"] == "word_count_range"], default=0)
    task = {"checker": {"id": "constraint_set", "params": {"constraints": constraints}}}
    fixed = len(" ".join(starts + middle + ends).split())
    for word in ("ok", "so", "x"):               # a filler the constraints do not forbid
        parts = starts + middle + [word] * max(lo - fixed, 0) + ends
        reply = " ".join(parts) or word
        if CHECKERS["constraint_set"](task, reply) == 1:
            return reply
    return None


# --- rlvr_ifeval --------------------------------------------------------------

RLVR_REPEAT_FUNC = "validate_repeat_prompt"


def _rlvr_constraints(gt: dict) -> tuple[list[dict] | None, str]:
    """(constraints, "") for a source check `constraint_set` judges exactly; (None, reason) for
    one it cannot (approximate word counts, keyword frequency, JSON, bullets, paragraphs, ...)."""
    fn = gt.get("func_name")
    if fn == "validate_lowercase":
        return [{"kind": "lowercase_only"}], ""
    if fn == "validate_no_commas":
        return [{"kind": "must_exclude", "s": ","}], ""
    if fn == "validate_quotation":
        return [{"kind": "starts_with", "s": '"'}, {"kind": "ends_with", "s": '"'}], ""
    if fn == "validate_end":
        phrase = gt.get("end_phrase")
        if not isinstance(phrase, str) or not phrase or phrase != phrase.strip():
            return None, "malformed_ground_truth"
        return [{"kind": "ends_with", "s": phrase}], ""
    if fn == "validate_forbidden_words":
        words = _word_list(gt.get("forbidden_words"))
        if words is None:
            return None, "malformed_ground_truth"
        if any(len(w) < MIN_FORBIDDEN_WORD_CHARS for w in words):
            return None, "forbidden_word_under_4_chars"
        return [{"kind": "must_exclude", "s": w} for w in words], ""
    if fn == "verify_keywords":
        words = _word_list(gt.get("keyword_list"))
        if words is None:
            return None, "malformed_ground_truth"
        return [{"kind": "must_include", "s": w, "k": 1} for w in words], ""
    if fn == "validate_word_constraint":
        n, quantifier = gt.get("N"), gt.get("quantifier")
        if type(n) is not int or n < 1:
            return None, "malformed_ground_truth"
        if quantifier == "at least":
            return [{"kind": "word_count_range", "min": n, "max": WORD_COUNT_NO_MAX}], ""
        if quantifier == "at most":
            return [{"kind": "word_count_range", "min": 0, "max": n}], ""
        return None, f"word_count_{str(quantifier).replace(' ', '_')}"
    return None, f"no_exact_kind:{fn}"


def _rlvr_repeat_request(content: str, instruction: str) -> str | None:
    """The request a compliant reply repeats: the message without the instruction sentence,
    which the sentence itself says is not part of the request. None when the sentence is not
    at one end of the message, where it cannot be cut out mechanically."""
    if not instruction:
        return None
    if content.startswith(instruction):
        request = content[len(instruction):].strip()
    elif content.endswith(instruction):
        request = content[:-len(instruction)].strip()
    else:
        return None
    return request or None


def _rlvr_questions(rows, stats: Counter | None = None, *, repeat: bool = False) -> list[dict]:
    """`repeat=False`: every exact kind except repeat-the-request. `repeat=True`: only those."""
    benchmark = "rlvr_ifeval_repeat" if repeat else "rlvr_ifeval"
    out = []
    for row in rows:
        _note(stats, "rows")
        messages = row["messages"]
        if len(messages) != 1 or messages[0]["role"] != "user" or not messages[0]["content"].strip():
            _note(stats, "not_one_user_message")
            continue
        content = messages[0]["content"]
        try:
            gt = json.loads(row["ground_truth"])
        except (TypeError, ValueError):
            gt = None
        if not isinstance(gt, dict):
            _note(stats, "malformed_ground_truth")
            continue
        fn = gt.get("func_name")
        if (fn == RLVR_REPEAT_FUNC) != repeat:
            _note(stats, "held_back_repeat_prompt" if fn == RLVR_REPEAT_FUNC else "not_repeat_prompt")
            continue
        if repeat:
            request = _rlvr_repeat_request(content, row.get("constraint") or "")
            constraints, reason = ([{"kind": "starts_with", "s": request}], "") if request else (None, "instruction_not_separable")
        else:
            constraints, reason = _rlvr_constraints(gt)
        if constraints is None:
            _note(stats, reason)
            continue
        if _witness(constraints) is None:
            _note(stats, "no_reply_can_pass")
            continue
        out.append(_constraint_question(content, constraints, benchmark, f"rlvr_ifeval_{_text_id(content)}",
                                        source_check=fn))
    return _unique(out, stats)


def _rlvr_rows():
    return load_dataset("allenai/RLVR-IFeval", split="train", revision=DATASET_REVISIONS_V3["allenai/RLVR-IFeval"])


def load_rlvr_ifeval(stats: Counter | None = None) -> list[dict]:
    """RLVR-IFeval rows whose single constraint maps EXACTLY onto a `constraint_set` kind."""
    return _rlvr_questions(_rlvr_rows(), stats)


def load_rlvr_ifeval_repeat(stats: Counter | None = None) -> list[dict]:
    """The repeat-the-request rows, graded on a derived gold. Held back: HELD_BACK_REVIEW."""
    return _rlvr_questions(_rlvr_rows(), stats, repeat=True)


# --- arc ----------------------------------------------------------------------

ARC_CONFIGS = (("ARC-Challenge", "challenge"), ("ARC-Easy", "easy"))


def _arc_questions(short: str, rows, stats: Counter | None = None) -> list[dict]:
    out = []
    for row in rows:
        _note(stats, "rows")
        labels = [str(x) for x in row["choices"]["label"]]
        texts = [str(t) for t in row["choices"]["text"]]
        if (len(labels) != len(texts) or not 2 <= len(texts) <= len(LETTERS)
                or len(set(labels)) != len(labels) or any(not t.strip() for t in texts)):
            _note(stats, "malformed_choices")
            continue
        if str(row["answerKey"]) not in labels:
            _note(stats, "answer_key_not_a_label")
            continue
        # Choices are re-lettered by position, so a source that labels them 1-4 and one that
        # labels them A-D read the same, and the gold is the POSITION of the answer key.
        gold = LETTERS[labels.index(str(row["answerKey"]))]
        source_id = str(row["id"])
        out.append({"prompt": _mcq_prompt(row["question"], texts), "gold": gold,
                    "grader_id": "letter_mcq", "benchmark": "arc",
                    "question_id": f"arc_{short}_{re.sub(r'[^A-Za-z0-9_.-]', '_', source_id)}",
                    "metadata": {"config": short, "source_id": source_id, "split": "train"}})
    return out


def load_arc(stats: Counter | None = None) -> list[dict]:
    """ARC Challenge and Easy, TRAIN split only (validation and test stay held back)."""
    out = []
    for config, short in ARC_CONFIGS:
        ds = load_dataset("allenai/ai2_arc", config, split="train", revision=DATASET_REVISIONS_V3["allenai/ai2_arc"])
        out.extend(_arc_questions(short, ds, stats))
    return _unique(out, stats)


# --- ifeval (the v3 mapping; v2's is ifeval.py and is not changed) ----------------

def _ifeval_constraints_v3(inst_id: str, kwargs: dict) -> list[dict]:
    """The v3 mapping of one IFEval instruction. It differs from `ifeval._ifeval_to_constraint`
    (v1 and v2, pinned) in two ways: "less than N" words is a limit of N-1 (v2 drops it), and
    every keyword and every forbidden word is a constraint (v2 keeps the first). Every other
    instruction is delegated to the v2 function unchanged."""
    from fugal_subnet.benchmarks.ifeval import _ifeval_to_constraint

    if "keywords:existence" in inst_id:
        return [{"kind": "must_include", "s": w, "k": 1} for w in _distinct(kwargs.get("keywords"))]
    if "keywords:forbidden_words" in inst_id:
        return [{"kind": "must_exclude", "s": w} for w in _distinct(kwargs.get("forbidden_words"))]
    if "length_constraints:number_words" in inst_id:
        rel, num = kwargs.get("relation"), kwargs.get("num_words")
        if type(num) is not int:
            return []
        if rel == "at least":
            return [{"kind": "word_count_range", "min": num, "max": WORD_COUNT_NO_MAX}]
        if rel == "at most":
            return [{"kind": "word_count_range", "min": 0, "max": num}]
        if rel == "less than":
            return [{"kind": "word_count_range", "min": 0, "max": num - 1}]
        return []
    c = _ifeval_to_constraint(inst_id, kwargs)
    return [c] if c else []


def _ifeval_v3_questions(rows, stats: Counter | None = None) -> list[dict]:
    out = []
    for i, row in enumerate(rows):
        _note(stats, "rows")
        ids = list(row.get("instruction_id_list") or [])
        kwargs_list = row.get("kwargs") or []
        if isinstance(kwargs_list, str):
            try:
                kwargs_list = json.loads(kwargs_list)
            except (json.JSONDecodeError, TypeError):
                kwargs_list = [{}] * len(ids)
        constraints, mapped = [], 0
        for inst_id, kwargs in zip(ids, kwargs_list):
            found = _ifeval_constraints_v3(inst_id, kwargs or {})
            constraints.extend(found)
            mapped += bool(found)
        if not constraints:
            _note(stats, "no_mapped_instruction")
            continue
        if {c["kind"] for c in constraints} & UNGRADEABLE_IFEVAL_KINDS:
            _note(stats, "ungradeable_kind")                    # as v2: bullet_lines, json_object
            continue
        if any(c["kind"] == "must_exclude" and len(c["s"]) < MIN_FORBIDDEN_WORD_CHARS and c["s"] != ","
               for c in constraints):
            _note(stats, "forbidden_word_under_4_chars")
            continue
        if _witness(constraints) is None:
            _note(stats, "no_reply_can_pass")
            continue
        out.append(_constraint_question(row["prompt"], constraints, "ifeval", f"ifeval_key_{int(row['key'])}",
                                        row_index=i, n_instructions=len(ids), n_mapped_instructions=mapped))
    return _unique(out, stats)


def load_ifeval_v3(stats: Counter | None = None) -> list[dict]:
    """IFEval under the v3 mapping. Ids are `ifeval_key_<source key>`, not v2's `ifeval_NNNN`:
    the same prompt is graded differently here, so it is a different question. As in v2, an
    instruction with no exact constraint is skipped and the question is graded on the rest
    (`n_mapped_instructions` of `n_instructions` in the metadata), and a question carrying a
    bullet-list or JSON check is left out. New here: a question is also left out when one of
    its forbidden words is under four characters, or when no reply can meet its constraints."""
    ds = load_dataset("google/IFEval", split="train", revision=DATASET_REVISIONS_V3["google/IFEval"])
    return _ifeval_v3_questions(ds, stats)


# --- bbh (held back) ------------------------------------------------------------

BBH_LETTER_TASKS = (
    "date_understanding", "disambiguation_qa", "hyperbaton", "logical_deduction_five_objects",
    "logical_deduction_seven_objects", "logical_deduction_three_objects", "movie_recommendation",
    "penguins_in_a_table", "reasoning_about_colored_objects", "ruin_names",
    "salient_translation_error_detection", "snarks", "temporal_sequences",
    "tracking_shuffled_objects_five_objects", "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
)
# Two-way tasks, recast as options A and B in this order. Targets match without regard to case.
BBH_BINARY_TASKS = {
    "boolean_expressions": ("True", "False"),
    "causal_judgement": ("Yes", "No"),
    "formal_fallacies": ("valid", "invalid"),
    "navigate": ("Yes", "No"),
    "sports_understanding": ("Yes", "No"),
    "web_of_lies": ("Yes", "No"),
}
BBH_INTEGER_TASKS = ("multistep_arithmetic_two", "object_counting")
BBH_DROPPED_TASKS = {
    "dyck_languages": "free-text target",
    "word_sorting": "free-text target",
    "geometric_shapes": "the gold reaches (K), past the A-J the checker reads",
}
BBH_TASKS = tuple(sorted((*BBH_LETTER_TASKS, *BBH_BINARY_TASKS, *BBH_INTEGER_TASKS)))
_BBH_OPTIONS = "\nOptions:\n"


def _bbh_letter(text: str, target: str) -> tuple[tuple[str, list[str], str, bool] | None, str]:
    stem, sep, block = text.rpartition(_BBH_OPTIONS)
    if not sep or not stem.strip():
        return None, "no_options_block"
    options = []
    for j, line in enumerate(block.split("\n")):
        m = re.fullmatch(r"\(([A-Z])\) (.+)", line)
        if not m or m.group(1) != chr(ord("A") + j):
            return None, "unparseable_options"
        options.append(m.group(2).strip())
    if len(options) < 2:
        return None, "unparseable_options"
    m = re.fullmatch(r"\(([A-Z])\)", target.strip())
    if not m:
        return None, "non_letter_target"
    index = ord(m.group(1)) - ord("A")
    if index >= len(options):
        return None, "target_not_an_option"
    if index >= len(LETTERS):
        return None, "gold_past_J"
    # More than ten options and a gold within the first ten: the first ten are shown (as v2
    # does for MMLU-Pro), so every letter in the prompt is one the checker can read.
    return (stem, options[:len(LETTERS)], LETTERS[index], len(options) > len(LETTERS)), ""


def _bbh_binary(text: str, target: str, options: tuple[str, str]) -> tuple[tuple[str, str] | None, str]:
    stem, sep, block = text.rpartition(_BBH_OPTIONS)
    if sep:
        shown = [line[2:].strip().lower() for line in block.split("\n") if line.startswith("- ")]
        if shown != [o.lower() for o in options] or len(shown) != len(block.split("\n")):
            return None, "unexpected_options"
    else:
        stem = text
    if not stem.strip():
        return None, "empty_question"
    lowered = [o.lower() for o in options]
    if target.strip().lower() not in lowered:
        return None, "unexpected_target"
    return (stem, LETTERS[lowered.index(target.strip().lower())]), ""


def _bbh_questions(task: str, rows, stats: Counter | None = None) -> list[dict]:
    if task not in BBH_TASKS:
        raise ValueError(f"BBH task {task!r} is not loaded: {BBH_DROPPED_TASKS.get(task, 'unknown task')}")
    out = []
    for row in rows:
        _note(stats, "rows")
        text, target = str(row["input"]), str(row["target"])
        metadata = {"task": task}
        if task in BBH_INTEGER_TASKS:
            if not _gradeable_integer(target.strip()):
                _note(stats, "non_integer_gold")
                continue
            prompt, gold, grader = _numeric_prompt(text), str(int(target.strip())), "numeric_final"
        elif task in BBH_BINARY_TASKS:
            parsed, reason = _bbh_binary(text, target, BBH_BINARY_TASKS[task])
            if parsed is None:
                _note(stats, reason)
                continue
            prompt, gold, grader = _mcq_prompt(parsed[0], list(BBH_BINARY_TASKS[task])), parsed[1], "letter_mcq"
        else:
            parsed, reason = _bbh_letter(text, target)
            if parsed is None:
                _note(stats, reason)
                continue
            stem, options, gold, truncated = parsed
            prompt, grader = _mcq_prompt(stem, options), "letter_mcq"
            if truncated:
                _note(stats, "kept_with_options_cut_to_ten")
                metadata["options_truncated"] = True
        out.append({"prompt": prompt, "gold": gold, "grader_id": grader, "benchmark": "bbh",
                    "question_id": f"bbh_{task}_{_text_id(text)}", "metadata": metadata})
    return out


def _bbh_rows(task: str):
    """One task's parquet file from the pinned conversion commit. Fetched by name: that branch
    carries no config metadata, and a `data_dir` load cannot be re-read from the cache offline."""
    from datasets import Dataset
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("maveriq/bigbenchhard", f"{task}/train/0000.parquet", repo_type="dataset",
                           revision=DATASET_REVISIONS_V3["maveriq/bigbenchhard"])
    return Dataset.from_parquet(path)


def load_bbh(stats: Counter | None = None) -> list[dict]:
    """BIG-Bench Hard, the gradeable tasks. Held back: HELD_BACK_REVIEW."""
    out = []
    for task in BBH_TASKS:
        out.extend(_bbh_questions(task, _bbh_rows(task), stats))
    return _unique(out, stats)


# --- registries -----------------------------------------------------------------

LOADERS_V3 = {
    "gsm8k_train": load_gsm8k_train,
    "math_l12": load_math_l12,
    "rlvr_ifeval": load_rlvr_ifeval,
    "arc": load_arc,
    "ifeval": load_ifeval_v3,       # replaces v2's `ifeval` in a v3 pool; never both
}
LOADERS_HELD_BACK = {
    "bbh": load_bbh,
    "rlvr_ifeval_repeat": load_rlvr_ifeval_repeat,
}
assert set(LOADERS_V3) == set(V3_LICENCES)
assert set(LOADERS_HELD_BACK) == set(HELD_BACK_REVIEW) == set(HELD_BACK_LICENCES)
assert not set(LOADERS_V3) & set(LOADERS_HELD_BACK)
