#!/usr/bin/env python3
#
# THE V5 GRADING CONTRACT (graders_arena): RouterArena's scoring, ported.
#
# Like graders.py and graders_code.py this file is a CONSENSUS RULE: score(task, reply) must
# give the same number on every miner TD. Its hash (grader_hash()) is pinned in
# scripts/check_safety_invariants.py; any byte change is a new grader version. graders.py
# stays immutable.
#
# Protocol v5 optimises routers for RouterArena (https://github.com/RouteWorks/RouterArena,
# Apache-2.0), so a v5 exam question is scored exactly the way RouterArena scores a question
# of the same source dataset: the same answer extraction (llm_evaluation/enhanced_extractor.py)
# and the same metric (llm_evaluation/metrics.py), dispatched the same way (the dataset key
# taken from the question's RouterArena-style id prefix, evaluate_models.
# determine_dataset_from_global_index, then config/eval_config/zero-shot/<dataset>.json).
# Ported from RouterArena at dataset revision a4a062ce (code checked out 2026-11-17).
#
# score_text gives a float in [0, 1]; grade() gives the label, an INTEGER 0..10000
# (round(score * 10000)). Scores are 0/1 for every metric except METEOR (NarrativeQA and the
# WMT19 translations), which RouterArena averages as a fraction, exactly as here.
#
# DELIBERATE DEVIATIONS from RouterArena (each measured by scripts/parity_graders_arena.py):
#   1. Symbolic math equality. RouterArena tries sympy's parse_latex, then parse_expr (which
#      calls eval() on the model's text: code execution on untrusted input), then
#      latex2sympy2. Here the symbolic step is math_verify's parser and verifier (already the
#      repo's math dependency, used by graders.boxed_math), which never evals the text.
#   2. Code (LiveCodeBench). RouterArena runs code with its own harness and LiveCodeBench's
#      pickled private tests. v5 code questions come from the repo's calibrated code sets and
#      are graded by the pinned sandbox in graders_code.py (exec_func / exec_stdio). Not
#      comparable item by item; graded 0/1.
# Everything else is a line-for-line port, quirks included (they decide RouterArena's grade):
#   - SuperGLUE-ClozeTest: with no \boxed{}, the extracted answer is the reply's first
#     character, upper-cased.
#   - ChessInstruct (both the move and the multiple-choice items) uses chess_accuracy.
#   - ArcMMLU uses the MMLUPro config, Ethics the Ethics_commonsense config (both mcq).
#   - An empty reply or an empty gold scores 0.
#
# SECURITY: no eval, no exec, no network, no file access. Code questions EXECUTE model-written
# Python only through graders_code (seccomp sandbox), and only with allow_exec=True.
from __future__ import annotations

import hashlib
import os
import re
from collections import Counter
from math import isclose

# ---------------------------------------------------------------------------------------------
# Dispatch: RouterArena dataset key -> metric.
# ---------------------------------------------------------------------------------------------
MATH, MCQ, CHESS, EXACT, METEOR, SG_EXACT, SG_CLOZE, CODE = (
    "math_metric", "mcq_accuracy", "chess_accuracy", "exact_match", "meteor_score",
    "superglue_exact_match", "superglue_clozetest", "code_accuracy")

METRIC_BY_DATASET: dict[str, str] = {
    "AIME": MATH, "AsDiv": MATH, "FinQA": MATH, "GSM8K": MATH, "MATH": MATH,
    "ArcMMLU": MCQ, "MMLUPro": MCQ, "MMLU": MCQ, "GeoBench": MCQ, "MathQA": MCQ,
    "MedMCQA": MCQ, "MusicTheoryBench": MCQ, "OpenTDB": MCQ, "PubMedQA": MCQ,
    "SocialiQA": MCQ, "SuperGLUE-CausalReasoning": MCQ, "Ethics": MCQ, "GPQA": MCQ,
    "ChessInstruct": CHESS,
    "GeoGraphyData": EXACT, "QANTA": EXACT,
    "NarrativeQA": METEOR,
    "WMT19-cs-en": METEOR, "WMT19-de-en": METEOR, "WMT19-fi-en": METEOR,
    "WMT19-gu-en": METEOR, "WMT19-kk-en": METEOR, "WMT19-lt-en": METEOR,
    "WMT19-ru-en": METEOR, "WMT19-zh-en": METEOR,
    "SuperGLUE-ClozeTest": SG_CLOZE,
    "SuperGLUE-Entailment": SG_EXACT, "SuperGLUE-QA": SG_EXACT, "SuperGLUE-RC": SG_EXACT,
    "SuperGLUE-Wic": SG_EXACT, "SuperGLUE-Wsc": SG_EXACT,
    "LiveCodeBench": CODE,
}


def dataset_of(global_index: str) -> str:
    """RouterArena's dataset key from a RouterArena-style id ("MMLUPro_law_17" -> "MMLUPro")."""
    return str(global_index).split("_")[0]


# ---------------------------------------------------------------------------------------------
# Answer extraction: RouterArena's EnhancedAnswerExtractor, verbatim.
# ---------------------------------------------------------------------------------------------
_P = {
    "mcq_letter": r"\b([A-Z])\b(?!\w)",
    "mcq_option": r"(?:option|choice|select)[:\s]*([A-Z])",
    "mcq_letter_alt": r"(?:the\s+)?(?:answer|choice)\s*[:\-]\s*([A-Z])",
    "mcq_answer_colon": r"answer\s*:\s*([A-Z])",
    "math_answer": r"(?:the\s+)?(?:final\s+)?answer\s*[:\-]\s*([^\n]+)",
    "math_result": r"(?:result|solution)\s*[:\-]\s*([^\n]+)",
    "math_number": r"(?:answer|result)\s*[:\-]\s*(\d+(?:\.\d+)?)",
    "math_is": r"(?:the\s+)?(?:answer|result)\s+is\s+([^\n]+)",
    "python_code": r"```python\s*\n(.*?)\n```",
    "code_block": r"```(?:python)?\s*\n(.*?)\n```",
    "code_function": r"def\s+\w+\([^)]*\):\s*\n(.*?)(?=\n\w|\n```|\Z)",
    "quoted": r'["\']([^"\']+)["\']',
    "parentheses": r"\(([^)]+)\)",
    "brackets": r"\[([^\]]+)\]",
    "colon_answer": r"(?:answer|result|solution)[:\s]+([^\n]+)",
    "chess_move": r"\b([a-h][1-8][a-h][1-8])\b",
    "chess_move_alt": r"(?:move|play)[:\s]*([a-h][1-8][a-h][1-8])",
}
_CLEANUP = (
    r"^(?:the\s+)?(?:answer|result|solution)\s*[:\-]\s*",
    r"^(?:option|choice)\s*[:\-]\s*",
    r"^(?:letter|choice)\s*[:\-]\s*",
    r"\s*[:\-]\s*(?:is\s+)?the\s+(?:answer|result|solution)\.?$",
    r"\s*[:\-]\s*(?:is\s+)?correct\.?$",
    r"^[:\-\s]+",
    r"[:\-\s]+$",
    r"\.$",
    r'^["\']|["\']$',
    r"^(?:the\s+)?(?:final\s+)?answer\s+is\s+",
    r"^(?:the\s+)?answer\s+is\s+",
    r"^(?:the\s+)?result\s+is\s+",
)


def _clean(answer: str) -> str:
    if not answer:
        return answer
    cleaned = answer.strip()
    for pattern in _CLEANUP:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    return " ".join(cleaned.split()).strip()


def _has_boxed(text: str) -> bool:
    return bool(text) and bool(re.search(r"\\boxed\{", text))


def _standard_boxed(text: str):
    text = text.replace("\\text{", "")
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    return matches[0].strip() if matches else None


def _standard_math_boxed(text: str):
    start_match = re.search(r"\\boxed\{", text)
    if start_match:
        start_pos = start_match.end() - 1
        brace_count = 0
        for i, char in enumerate(text[start_pos:], start_pos):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    return text[start_pos + 1:i].strip()
    return None


def _mcq_answer(text: str):
    for key, flags, last in (("mcq_answer_colon", re.IGNORECASE, False),
                             ("mcq_option", re.IGNORECASE, False),
                             ("mcq_letter_alt", re.IGNORECASE, False)):
        matches = re.findall(_P[key], text, flags)
        if matches:
            return matches[0].upper()
    matches = re.findall(_P["mcq_letter"], text)
    if matches:
        return matches[-1]
    return None


def _math_patterns_answer(text: str):
    for key in ("math_answer", "math_result", "math_number", "math_is"):
        matches = re.findall(_P[key], text, re.IGNORECASE)
        if matches:
            cleaned = _clean(matches[0].strip())
            if cleaned:
                return cleaned
    return None


def _code_answer(text: str):
    for key in ("python_code", "code_block", "code_function"):
        matches = re.findall(_P[key], text, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[-1].strip()
    return None


def _general_answer(text: str):
    for key in ("quoted", "parentheses", "brackets"):
        matches = re.findall(_P[key], text)
        if matches:
            return matches[-1].strip()
    matches = re.findall(_P["colon_answer"], text, re.IGNORECASE)
    if matches:
        cleaned = _clean(matches[-1].strip())
        if cleaned:
            return cleaned
    return None


def _enhanced_answer(text: str, dataset):
    if dataset == "LiveCodeBench":
        return _code_answer(text)
    if dataset == "SuperGLUE-ClozeTest":
        return text[0].upper()
    return _mcq_answer(text) or _math_patterns_answer(text) or _general_answer(text)


def _fallback_answer(text: str) -> str:
    matches = re.findall(r"\b([A-Z])\b", text)
    if matches:
        return matches[-1]
    return text.strip()


def extract_boxed_answer(text, dataset=None) -> str:
    if not text or not text.strip():
        return ""
    text = text.strip()
    if _has_boxed(text):
        standard = _standard_boxed(text)
        if standard:
            return standard
    enhanced = _enhanced_answer(text, dataset)
    if enhanced:
        return enhanced
    return _fallback_answer(text)


def extract_math_answer(text) -> str:
    if not text or not text.strip():
        return ""
    text = text.strip()
    if _has_boxed(text):
        standard = _standard_math_boxed(text)
        if standard:
            return standard
    for key in ("math_answer", "math_result", "math_number", "math_is"):
        matches = re.findall(_P[key], text, re.IGNORECASE)
        if matches:
            cleaned = _clean(matches[0].strip())
            if cleaned:
                return cleaned
    for pattern in (r"\b(\d+(?:\.\d+)?)\b", r"\b(\d+(?:\.\d+)?)\s*$"):
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1]
    return text.strip()


def _normalize_chess_extracted(move: str) -> str:
    if not move:
        return ""
    move = str(move).strip().lower().replace("-", "").replace(" ", "").replace("_", "")
    return re.sub(r"[^a-z0-9]", "", move)


def extract_chess_move(text) -> str:
    if not text or not text.strip():
        return ""
    text = text.strip()
    if _has_boxed(text):
        standard = _standard_boxed(text)
        if standard:
            return _normalize_chess_extracted(standard)
    matches = re.findall(_P["chess_move"], text, re.IGNORECASE)
    if matches:
        return _normalize_chess_extracted(matches[-1])
    matches = re.findall(_P["chess_move_alt"], text, re.IGNORECASE)
    if matches:
        return _normalize_chess_extracted(matches[-1])
    return _normalize_chess_extracted(_fallback_answer(text))


# ---------------------------------------------------------------------------------------------
# Metrics: RouterArena's metrics.py, one prediction at a time.
# ---------------------------------------------------------------------------------------------
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _gold_letter(gt):
    """mcq_accuracy's ground-truth parse; raises ValueError like RouterArena (graded 0)."""
    if gt in tuple(_LETTERS):
        return gt
    if gt in tuple(_LETTERS.lower()):
        return gt.upper()
    gt_int = int(float(gt))
    if gt_int in range(0, 26):
        return _LETTERS[gt_int]
    raise ValueError(f"Ground truth index out of range: {gt}")


def mcq_accuracy(reply, gold) -> float:
    try:
        gt_letter = _gold_letter(gold)
    except (ValueError, TypeError):
        return 0.0
    return float(extract_boxed_answer(reply, "MCQ") == gt_letter)


# -- math ------------------------------------------------------------------------------------
def _choice_answer_clean(pred: str) -> str:
    cleaned = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")
    matches = re.findall(r"\b(A|B|C|D|E)\b", cleaned.upper())
    result = matches[-1] if matches else cleaned.strip().strip(".")
    return result.rstrip(".").rstrip("/")


def _parse_digits(num):
    normalized = str(num).replace(",", "")
    try:
        return float(normalized)
    except Exception:
        if normalized.endswith("%"):
            normalized = normalized[:-1]
            if normalized.endswith("\\"):
                normalized = normalized[:-1]
            try:
                return float(normalized) / 100
            except Exception:
                pass
    return None


def _is_digit(num) -> bool:
    return _parse_digits(num) is not None


def _numeric_equal(prediction: float, reference: float) -> bool:
    return isclose(reference, prediction, rel_tol=1e-4)


def _str_to_pmatrix(input_str: str) -> str:
    stripped = input_str.strip()
    pmatrix_list = []
    for m in re.findall(r"\{.*,.*\}", stripped):
        content = m.strip("{}")
        pmatrix_list.append(r"\begin{pmatrix}" + content.replace(",", "\\") + r"\end{pmatrix}")
    return ", ".join(pmatrix_list)


SYMBOLIC_MAX_CHARS = 400   # longer expressions are never symbolically compared (bounded time)
_LATEX_COMMAND = re.compile(r"\\[A-Za-z]+")
_WORD = re.compile(r"[A-Za-z]{2,}")


def _symbolic_allowed(a: str, b: str) -> bool:
    """RouterArena's sympy parsers fail (so grade 0) on text answers, words and units, and on
    a value left in an extra pair of braces ("\\boxed{{62}}"); math_verify would accept them.
    Measured on 38,161 RouterArena-graded replies: every math_verify-only accept had one of
    these. So the symbolic step runs only on bare expressions."""
    for s in (a, b):
        if "\\text" in s or "\\mbox" in s or s.lstrip().startswith("{"):
            return False
        if _WORD.search(_LATEX_COMMAND.sub(" ", s)):
            return False
    return True


def _symbolic_equal(a, b) -> bool:
    """Deviation 1: math_verify instead of parse_latex / parse_expr(eval) / latex2sympy2,
    restricted to bare expressions (_symbolic_allowed) to stay as strict as RouterArena."""
    a, b = str(a), str(b)
    if a == b:
        return True
    if len(a) > SYMBOLIC_MAX_CHARS or len(b) > SYMBOLIC_MAX_CHARS or not _symbolic_allowed(a, b):
        return False
    try:
        from math_verify import parse, verify
        gold = parse("$" + b + "$", parsing_timeout=None)
        pred = parse("$" + a + "$", parsing_timeout=None)
        return bool(gold) and bool(pred) and bool(verify(gold, pred, timeout_seconds=None))
    except Exception:
        return False


def math_equal(prediction, reference, include_percentage: bool = True,
               is_close: bool = True) -> bool:
    if prediction is None or reference is None:
        return False
    if str(prediction.strip().lower()) == str(reference.strip().lower()):
        return True
    if reference in ["A", "B", "C", "D", "E"] and _choice_answer_clean(prediction) == reference:
        return True
    try:
        if _is_digit(prediction) and _is_digit(reference):
            pred_val = _parse_digits(prediction)
            ref_val = _parse_digits(reference)
            if pred_val is not None and ref_val is not None:
                gt_result = [ref_val / 100, ref_val, ref_val * 100] if include_percentage else [ref_val]
                for item in gt_result:
                    try:
                        if is_close:
                            if _numeric_equal(pred_val, item):
                                return True
                        elif item == pred_val:
                            return True
                    except Exception:
                        continue
                return False
    except Exception:
        pass
    if not prediction and prediction not in [0, False]:
        return False
    reference = str(reference).strip()
    prediction = str(prediction).strip()
    if "pmatrix" in prediction and "pmatrix" not in reference:
        reference = _str_to_pmatrix(reference)
    pred_str, ref_str = prediction, reference
    if ((prediction.startswith("[") and prediction.endswith("]") and not reference.startswith("("))
            or (prediction.startswith("(") and prediction.endswith(")")
                and not reference.startswith("["))):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for s in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    if pred_str.lower() == ref_str.lower():
        return True
    if (re.match(r"(\(|\[).+(\)|\])", prediction) is not None
            and re.match(r"(\(|\[).+(\)|\])", reference) is not None):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all(math_equal(pred_parts[i], ref_parts[i], include_percentage, is_close)
                   for i in range(len(pred_parts))):
                return True
    if ((prediction.startswith("\\begin{pmatrix}") or prediction.startswith("\\begin{bmatrix}"))
            and (prediction.endswith("\\end{pmatrix}") or prediction.endswith("\\end{bmatrix}"))
            and (reference.startswith("\\begin{pmatrix}") or reference.startswith("\\begin{bmatrix}"))
            and (reference.endswith("\\end{pmatrix}") or reference.endswith("\\end{bmatrix}"))):
        pred_lines = [ln.strip() for ln in prediction[len("\\begin{pmatrix}"):-len("\\end{pmatrix}")]
                      .split("\\\\") if ln.strip()]
        ref_lines = [ln.strip() for ln in reference[len("\\begin{pmatrix}"):-len("\\end{pmatrix}")]
                     .split("\\\\") if ln.strip()]
        matched = True
        if len(pred_lines) == len(ref_lines):
            for pred_line, ref_line in zip(pred_lines, ref_lines):
                pred_parts = pred_line.split("&")
                ref_parts = ref_line.split("&")
                if len(pred_parts) == len(ref_parts):
                    if not all(math_equal(pred_parts[i], ref_parts[i], include_percentage, is_close)
                               for i in range(len(pred_parts))):
                        matched = False
                        break
                else:
                    matched = False
                if not matched:
                    break
        else:
            matched = False
        if matched:
            return True
    if prediction.count("=") == 1 and reference.count("=") == 1:
        pred = prediction.split("=")
        pred = f"{pred[0].strip()} - ({pred[1].strip()})"
        ref = reference.split("=")
        ref = f"{ref[0].strip()} - ({ref[1].strip()})"
        if _symbolic_equal(pred, ref) or _symbolic_equal(f"-({pred})", ref):
            return True
    elif (prediction.count("=") == 1 and len(prediction.split("=")[0].strip()) <= 2
          and "=" not in reference):
        if math_equal(prediction.split("=")[1], reference, include_percentage, is_close):
            return True
    elif (reference.count("=") == 1 and len(reference.split("=")[0].strip()) <= 2
          and "=" not in prediction):
        if math_equal(prediction, reference.split("=")[1], include_percentage, is_close):
            return True
    return _symbolic_equal(prediction, reference)


def math_metric(reply, gold) -> float:
    try:
        extracted = extract_math_answer(reply)
        if len(extracted.split("=")) > 1:
            extracted = extracted.split("=")[-1].strip()
        return float(bool(math_equal(extracted, gold)))
    except Exception:
        return 0.0


# -- exact match (QANTA, GeoGraphyData) --------------------------------------------------------
def normalize_qanta_answer(answer) -> str:
    if not answer:
        return ""
    answer = str(answer).strip().replace("_", " ").lower()
    return " ".join(answer.split())


def exact_match(reply, gold) -> float:
    pred = normalize_qanta_answer(extract_boxed_answer(reply, "QANTA"))
    return float(pred == normalize_qanta_answer(gold))


# -- chess ------------------------------------------------------------------------------------
def normalize_chess_move(move) -> str:
    if not move:
        return ""
    move = str(move).strip().lower()
    if len(move) == 1 and move.isalpha():
        return move
    if move.isdigit():
        numeric_val = int(move)
        if 0 <= numeric_val <= 25:
            return chr(ord("a") + numeric_val)
        return move
    move = move.replace("-", "").replace(" ", "").replace("_", "")
    return re.sub(r"[^a-z0-9]", "", move)


def chess_accuracy(reply, gold) -> float:
    return float(normalize_chess_move(extract_chess_move(reply)) == normalize_chess_move(gold))


# -- METEOR (RouterArena's simplified METEOR) ---------------------------------------------------
def _tokenize(text):
    if not text:
        return []
    text = re.sub(r"[^\w\s]", " ", text.lower())
    if re.search(r"[\u4e00-\u9fff]", text):
        tokens = [char for char in text if char.strip() and char != " "]
    else:
        tokens = text.split()
    return [t for t in tokens if t.strip()]


def _count_chunks(tokens, matched_words) -> int:
    chunks, in_chunk = 0, False
    for token in tokens:
        if token in matched_words:
            if not in_chunk:
                chunks += 1
                in_chunk = True
        else:
            in_chunk = False
    return chunks


def single_meteor(prediction, ground_truth) -> float:
    pred_tokens = _tokenize(prediction)
    gt_tokens = _tokenize(ground_truth)
    if not pred_tokens or not gt_tokens:
        return 0.0
    pred_counter, gt_counter = Counter(pred_tokens), Counter(gt_tokens)
    matches = sum(min(pred_counter[w], gt_counter[w]) for w in pred_counter if w in gt_counter)
    precision = matches / len(pred_tokens)
    recall = matches / len(gt_tokens)
    if precision == 0 and recall == 0:
        return 0.0
    f_score = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    matched_words = set(pred_tokens) & set(gt_tokens)
    if not matched_words:
        penalty = 1.0
    else:
        min_chunks = min(_count_chunks(pred_tokens, matched_words),
                         _count_chunks(gt_tokens, matched_words))
        total = len(matched_words)
        penalty = 0.0 if min_chunks == total else min(1.0, 0.5 * (min_chunks / total) ** 2)
    return f_score * (1 - penalty)


def meteor_score(reply, gold) -> float:
    if not reply or not gold:
        return 0.0
    return float(single_meteor(extract_boxed_answer(reply), gold))


# -- SuperGLUE ---------------------------------------------------------------------------------
_TO_NUMBER = {"A": "0.0", "0": "0.0", "FALSE": "0.0", "False": "0.0", "false": "0.0",
              "B": "1.0", "1": "1.0", "TRUE": "1.0", "True": "1.0", "true": "1.0"}
_TO_YESNO = {"A": "Yes", "X": "Yes", "1": "Yes", "TRUE": "Yes", "True": "Yes", "true": "Yes",
             "B": "No", "Y": "No", "0": "No", "FALSE": "No", "False": "No", "false": "No"}


def _convert_superglue(extracted, ground_truth):
    if ground_truth in ["0.0", "1.0", "0", "1"]:
        return _TO_NUMBER.get(extracted.upper(), extracted)
    if ground_truth in ["Yes", "No", "yes", "no", "YES", "NO"]:
        return _TO_YESNO.get(extracted.upper(), extracted)
    return extracted


def superglue_exact_match(reply, gold) -> float:
    if not reply or not gold:
        return 0.0
    extracted = extract_boxed_answer(reply, "SuperGLUE")
    return float(_convert_superglue(extracted, gold) == gold)


def superglue_clozetest(reply, gold) -> float:
    try:
        if isinstance(gold, int):
            if gold not in range(0, 26):
                return 0.0
            gt = _LETTERS[gold]
        elif isinstance(gold, str):
            try:
                gt = _gold_letter(gold)
            except (ValueError, TypeError):
                gt = gold
        else:
            return 0.0
        return float(extract_boxed_answer(reply, "SuperGLUE-ClozeTest") == gt)
    except Exception:
        return 0.0


_METRICS = {
    MATH: math_metric, MCQ: mcq_accuracy, CHESS: chess_accuracy, EXACT: exact_match,
    METEOR: meteor_score, SG_EXACT: superglue_exact_match, SG_CLOZE: superglue_clozetest,
}


# ---------------------------------------------------------------------------------------------
# Entry points.
# ---------------------------------------------------------------------------------------------
def score_text(dataset: str, gold, reply) -> float:
    """RouterArena's score for one non-code reply of `dataset` (a key of METRIC_BY_DATASET).
    Never raises; anything unexpected scores 0."""
    try:
        metric = METRIC_BY_DATASET[dataset]
        if metric == CODE:
            return 0.0
        value = float(_METRICS[metric](reply if isinstance(reply, str) else "", gold))
        return value if 0.0 <= value <= 1.0 else 0.0
    except Exception:
        return 0.0


Y_SCALE = 10000   # a label is an integer 0..Y_SCALE (config.Y_SCALE_V5)


def to_units(score: float) -> int:
    """A [0, 1] score as an integer label 0..Y_SCALE: round(score * Y_SCALE), Python's
    round-half-even on the IEEE double. 0/1 scores map to exactly 0 and Y_SCALE."""
    units = int(round(score * Y_SCALE))
    return units if 0 <= units <= Y_SCALE else 0


def grade(task, reply, allow_exec=False) -> int:
    """The v5 label for one exam reply: an INTEGER in 0..Y_SCALE (Y_SCALE = 10000). `task` is
    a pool item: its metadata carries the RouterArena dataset key (`ra_dataset`); code items
    carry the graders_code checker and are graded by that sandbox (0 or Y_SCALE, only with
    allow_exec). Never raises; failure scores 0."""
    try:
        meta = task.get("metadata") or {}
        dataset = meta["ra_dataset"]
        if METRIC_BY_DATASET[dataset] == CODE:
            from fugal_subnet import graders_code
            code_task = {"gold": task["gold"], "checker": meta["checker"]}
            return Y_SCALE if graders_code.grade(code_task, reply, allow_exec) == 1 else 0
        return to_units(score_text(dataset, task["gold"], reply))
    except Exception:
        return 0


def grader_hash() -> str:
    """sha256 of this file's bytes, line endings normalised to LF (as graders.py)."""
    with open(os.path.abspath(__file__), "rb") as f:
        return "sha256:" + hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


if __name__ == "__main__":
    print(grader_hash())
