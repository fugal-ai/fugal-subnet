"""Protocol v5 exam pool: RouterArena-like questions from RouterArena's own source datasets.

Protocol v5 pays routers on RouterArena's formula, so the exam is drawn from the same public
source datasets RouterArena sampled its 8,400 questions from, in RouterArena's proportions,
EXCLUDING every question RouterArena evaluates on. Each question is put in RouterArena's row
form (Context, Question, Options, Answer), prompted with RouterArena's template for its
dataset (arena_prompts.json, copied verbatim from RouterArena's config/eval_config) exactly as
RouterArena's scripts/process_datasets/prep_datasets.py does, and graded by graders_arena.py
(RouterArena's metrics, ported). The pool is a pure function of pinned inputs:

1. SOURCES. Every source is a Hugging Face dataset at a pinned commit (SOURCES below); each
   converter is a pure function of its rows.
2. EXCLUSION. A candidate is dropped when its question text equals a RouterArena question, or
   (sources with long questions) shares any 12-word shingle with a RouterArena question of the
   same RouterArena dataset. RouterArena is read at its pinned revision; the dropped ids are
   written to exclusions_routerarena.json, which is pinned data.
3. CAPS. Large sources are cut to their cap (CAP_TOTAL x RouterArena share, at least
   CAP_FLOOR) in the order of sha256(POOL_SALT:question_id): deterministic, no RNG.
4. CODE. RouterArena's LiveCodeBench share is filled from the v3 pool's calibrated code sets
   (graded by the pinned sandbox in graders_code.py), never from LiveCodeBench itself.

Gaps (reported in the manifest): ArcMMLU (RouterArena uses an English translation that is not
published; the source is Chinese) is not in the pool and its share is spread over the rest.
OpenTDB is not on Hugging Face as such; OpenTriviaQA (the same kind of four-option trivia,
CC BY-SA) stands in for it under the OpenTDB key.

Pool item schema (as every loader in this package): prompt, gold, grader_id ("arena"),
route_text (what routers embed: the row's content without the template, route_text below),
benchmark ("arena_<RouterArena dataset key>"), question_id, metadata {ra_dataset, ra_name,
source, source_id, ...}. Code items keep their v3 checker in metadata.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from importlib.resources import files

# --- pinned inputs ----------------------------------------------------------------------------
ROUTERARENA = ("RouteWorks/RouterArena", "a4a062ce3313b56bb09c042e1bc37b61d34e3bd8")

SOURCES: dict[str, str] = {
    "TIGER-Lab/MMLU-Pro": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
    "cais/mmlu": "c30699e8356da336a370243923dbaf21066bb9fe",
    "openlifescienceai/medmcqa": "91c6572c454088bf71b679ad90aa8dffcd0d5868",
    "qiaojin/PubMedQA": "9001f2853fb87cab8d220904e0de81ac6973b318",
    "hendrycks/ethics": "b8b47c589f8bee77175b8648e5497278b68da48a",
    "aps/super_glue": "3de24cf8022e94f4ee4b9d55a6f539891524d646",
    "wmt/wmt19": "c8a87a86736e7bf10c74c22e9033eba9146774e9",
    "deepmind/narrativeqa": "2e643e7363944af1c33a652d1c87320d0871c4e4",
    "Thytu/ChessInstruct": "dcfd1a8075dffd498fa190916fe28b1e0f8300e9",
    "m-a-p/MusicTheoryBench": "81b65609885be1fba19ce167a2a9604ae676c5ae",
    "community-datasets/qanta": "e3c5602229e5c0c8572636a81cb366eb9ccfa890",
    "kth8/OpenTriviaQA": "5f856c93b8b9b14cd23ff3a8a9a683cca94b4f3f",
    "daven3/geobench": "fa6f583ad38f93a7e7f5a3ea1582fe6593437747",
    "openai/gsm8k": "740312add88f781978c0658806c59bc2815b9866",
    "EleutherAI/hendrycks_math": "21a5633873b6a120296cce3e2df9d5550074f4a3",
    "qq8933/AIME_1983_2024": "3e2cc86390666c5c756622afc0eeb9e6194496bc",
    # Script datasets: Hugging Face's own parquet conversion, pinned by its commit.
    "allenai/math_qa": "fafb9f7ee5b9ec4da9499f9c4177a4c91389f2d6",
    "allenai/social_i_qa": "537a2ec8ec565adc0b70b70752893e59e024df26",
    "dreamerdeo/finqa": "50a82979fa410ff5efc29e2c14d9aa43d3877666",
}

POOL_SALT = "fugal-arena-pool-v5"
CAP_TOTAL = 90_000          # caps are CAP_TOTAL x the dataset's RouterArena share ...
CAP_FLOOR = 300             # ... and at least this many when the source has them
SHINGLE_WORDS = 12
SHINGLE_DATASETS = frozenset({     # long questions: also excluded by any shared 12-word shingle
    "MMLUPro", "MMLU", "MedMCQA", "PubMedQA", "OpenTDB", "QANTA", "NarrativeQA", "GeoBench",
    "MusicTheoryBench", "MathQA", "SuperGLUE-ClozeTest", "SuperGLUE-RC", "SuperGLUE-QA",
    "SuperGLUE-Entailment", "Ethics", "GSM8K", "MATH", "AIME", "AsDiv", "FinQA", "SocialiQA",
    "SuperGLUE-CausalReasoning", "WMT19-cs-en", "WMT19-de-en", "WMT19-fi-en", "WMT19-gu-en",
    "WMT19-kk-en", "WMT19-lt-en", "WMT19-ru-en", "WMT19-zh-en"})
NOT_IN_POOL = {
    "ArcMMLU": "RouterArena's English ArcMMLU is not published (source is Chinese)",
    "GeoGraphyData": "RouterArena's GeoGraphyData_100k question set is not on Hugging Face",
}
CODE_BENCHMARKS = ("code_contests", "code_humanevalplus", "code_leetcode", "code_mbppplus")
_WMT_PAIRS = ("cs-en", "de-en", "fi-en", "gu-en", "kk-en", "lt-en", "ru-en", "zh-en")
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# --- fetching (the only I/O) ------------------------------------------------------------------
class Fetcher:
    """hf_hub_download at the pinned revision, into `cache_dir`. Parquet is read with pyarrow;
    nothing is unpickled or executed."""

    def __init__(self, cache_dir: str | None = None):
        self.cache_dir = cache_dir

    def path(self, repo: str, filename: str, revision: str | None = None, tries: int = 6) -> str:
        import time

        from huggingface_hub import hf_hub_download
        rev = revision or SOURCES[repo]
        for attempt in range(tries):      # transient network errors only; the bytes are pinned
            try:
                return hf_hub_download(repo, filename, repo_type="dataset", revision=rev,
                                       cache_dir=self.cache_dir)
            except Exception:
                if attempt == tries - 1:
                    raise
                time.sleep(5 * (attempt + 1))
        raise RuntimeError("unreachable")

    def parquet(self, repo, filename, columns=None, revision=None) -> list[dict]:
        import pyarrow.parquet as pq
        return pq.read_table(self.path(repo, filename, revision), columns=columns).to_pylist()

    def csv_rows(self, repo, filename) -> list[dict]:
        with open(self.path(repo, filename), encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    def json(self, repo, filename):
        with open(self.path(repo, filename), encoding="utf-8") as f:
            return json.load(f)


# --- RouterArena row form ---------------------------------------------------------------------
def _row(name, source, source_id, question, answer, *, context="", options=(), meta=None):
    return {"ra_name": name, "source": source, "source_id": str(source_id),
            "Context": context or "", "Question": question or "", "Options": list(options),
            "Answer": str(answer), "Metadata": meta or {}}


def ra_dataset_of(ra_name: str) -> str:
    """The grading key (RouterArena's global-index prefix): "MMLUPro_law" -> "MMLUPro"."""
    return ra_name.split("_")[0]


def _template_key(ra_name: str, has_options: bool) -> str:
    """prep_datasets.py's base_dataset_name."""
    if "Ethics" in ra_name:
        return ra_name
    if "ChessInstruct" in ra_name:
        return "ChessInstruct_mcq" if has_options else "ChessInstruct"
    return ra_name.split("_", 1)[0]


def _escape_braces(text):
    if not isinstance(text, str):
        return text
    out, i = "", 0
    while i < len(text):
        if text[i] == "{":
            out += "{{"
            i += 2 if i + 1 < len(text) and text[i + 1] == "{" else 1
        elif text[i] == "}":
            out += "}}"
            i += 2 if i + 1 < len(text) and text[i + 1] == "}" else 1
        else:
            out += text[i]
            i += 1
    return out


def _fmt(template: str, **kw) -> str:
    return template.format(**{k: _escape_braces(v) for k, v in kw.items()})


_PROMPTS: dict | None = None


def prompts() -> dict:
    global _PROMPTS
    if _PROMPTS is None:
        with open(str(files("fugal_subnet.benchmarks") / "arena_prompts.json"), encoding="utf-8") as f:
            _PROMPTS = json.load(f)["prompts"]
    return _PROMPTS


def format_prompt(row: dict) -> str:
    """prep_datasets.build_formatted_prompts_from_router_eval_benchmark, for one row."""
    name, options = row["ra_name"], row["Options"]
    tmpl = prompts()[_template_key(name, bool(options))]["prompt"]
    context = row["Context"]
    ctx = context if context != "" else "None"
    options_str = "".join(f"{_LETTERS[i] if i < 26 else '-'}. {o}\n" for i, o in enumerate(options))
    base = _template_key(name, bool(options))
    if base == "SuperGLUE-RC":
        p = _fmt(tmpl, Question=row["Question"], Answer="")
    elif base == "SuperGLUE-Wic":
        p = _fmt(tmpl, Question=row["Question"], Context=context)
    elif not options:
        p = _fmt(tmpl, Context=ctx, Question=row["Question"])
    else:
        p = _fmt(tmpl, Context=ctx, Question=row["Question"], Options=options_str)
    return _cut(p)


def _cut(text: str) -> str:
    """prep_datasets' length cut: the first and last 5,000 characters of a longer text."""
    return f"{text[:5000]}...{text[-5000:]}" if len(text) > 10000 else text


def route_text(row: dict) -> str:
    """What a router sees: the row's own content, never RouterArena's instruction template.

    RouterArena forbids routing on the instruction text its harness wraps around each question
    (it names the dataset and the answer format), and a real user sends no such template. So the
    text routers embed is the values the template inserts, in the template's own order, joined by
    a blank line: Context (only when non-empty; the template's "None" filler is not content),
    Question, and Options exactly as the template lists them ("A. ...", one per line). Every
    template sentence, label and format instruction is left out. The answering model still gets
    the full templated prompt (format_prompt), which grading needs. Deterministic: a pure
    function of the row and the pinned templates."""
    import string

    name, options = row["ra_name"], row["Options"]
    tmpl = prompts()[_template_key(name, bool(options))]["prompt"]
    values = {"Context": row["Context"], "Question": row["Question"],
              "Options": "".join(f"{_LETTERS[i] if i < 26 else '-'}. {o}\n"
                                 for i, o in enumerate(options)).rstrip("\n"),
              "Answer": ""}
    parts, seen = [], set()
    for _lit, field, _spec, _conv in string.Formatter().parse(tmpl):
        if field in values and field not in seen:
            seen.add(field)
            if values[field].strip():
                parts.append(values[field].strip("\n"))
    return _cut("\n\n".join(parts))


# --- converters (pure functions of pinned rows) -----------------------------------------------
def mmlupro(fx):
    for split in ("test", "validation"):
        fn = f"data/{split}-00000-of-00001.parquet"
        for r in fx.parquet("TIGER-Lab/MMLU-Pro", fn):
            yield _row(f"MMLUPro_{r['category']}", "TIGER-Lab/MMLU-Pro", r["question_id"],
                       r["question"], r["answer_index"], options=r["options"])


def mmlu(fx):
    # RouterArena takes MMLU from two subjects only.
    for subject in ("formal_logic", "management"):
        for split in ("test", "validation", "dev"):
            fn = f"{subject}/{split}-00000-of-00001.parquet"
            for i, r in enumerate(fx.parquet("cais/mmlu", fn)):
                yield _row(f"MMLU_{subject}", "cais/mmlu", f"{subject}/{split}/{i}",
                           r["question"], r["answer"], options=r["choices"])


def medmcqa(fx):
    for split in ("train", "validation"):
        for r in fx.parquet("openlifescienceai/medmcqa", f"data/{split}-00000-of-00001.parquet"):
            if r.get("choice_type") == "multi" or r.get("cop") is None:
                continue
            opts = [r["opa"], r["opb"], r["opc"], r["opd"]]
            if any(o is None for o in opts):
                continue
            yield _row("MedMCQA", "openlifescienceai/medmcqa", r["id"], r["question"], r["cop"],
                       options=opts, meta={"subject_name": r.get("subject_name")})


_PQA = ["yes", "no", "maybe"]


def pubmedqa(fx):
    for cfg in ("pqa_labeled", "pqa_artificial"):
        for r in fx.parquet("qiaojin/PubMedQA", f"{cfg}/train-00000-of-00001.parquet"):
            if r.get("final_decision") not in _PQA:
                continue
            ctx = r["context"]["contexts"] if isinstance(r.get("context"), dict) else []
            yield _row("PubMedQA", "qiaojin/PubMedQA", f"{cfg}/{r['pubid']}", r["question"],
                       _PQA.index(r["final_decision"]), context=str(list(ctx)), options=_PQA)


def opentdb(fx):
    # OpenTriviaQA (CC BY-SA) under RouterArena's OpenTDB key: four-option trivia.
    for i, r in enumerate(fx.parquet("kth8/OpenTriviaQA", "data/train-00000-of-00001.parquet")):
        try:
            choices = ast.literal_eval(r["choices"]) if isinstance(r["choices"], str) else list(r["choices"])
        except (ValueError, SyntaxError):
            continue
        if not isinstance(choices, list) or len(choices) < 2 or choices.count(r["answer"]) != 1:
            continue
        yield _row(f"OpenTDB_{r['category']}", "kth8/OpenTriviaQA", i, r["question"],
                   choices.index(r["answer"]), options=[str(c) for c in choices],
                   meta={"type": "multiple", "original_category": r["category"]})


_QANTA_CATS = ("Literature", "History", "Science", "Fine Arts", "Social Science", "Geography",
               "Philosophy")
_QANTA_FOLDS = ("guesstrain", "guessdev", "guesstest", "buzztrain", "buzzdev", "buzztest",
                "adversarial")


def qanta(fx):
    # RouterArena uses the first sentence of a tossup and the Wikipedia page as the answer.
    seen = set()
    for fold in _QANTA_FOLDS:
        fn = f"mode=full,char_skip=25/{fold}-00000-of-00001.parquet"
        for r in fx.parquet("community-datasets/qanta", fn,
                            columns=["qanta_id", "first_sentence", "page", "category"]):
            if r["category"] not in _QANTA_CATS or not r["page"] or r["qanta_id"] in seen:
                continue
            seen.add(r["qanta_id"])
            yield _row(f"QANTA_{r['category']}", "community-datasets/qanta", r["qanta_id"],
                       r["first_sentence"], r["page"].replace("_", " "))


def narrativeqa(fx):
    z = zipfile.ZipFile(fx.path("deepmind/narrativeqa", "data/narrativeqa-master.zip"))
    with z.open("narrativeqa-master/third_party/wikipedia/summaries.csv") as f:
        summaries = {r["document_id"]: r["summary"]
                     for r in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))}
    with z.open("narrativeqa-master/qaps.csv") as f:
        for i, r in enumerate(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))):
            s = summaries.get(r["document_id"])
            if s:
                yield _row("NarrativeQA", "deepmind/narrativeqa", f"{r['document_id']}/{i}",
                           r["question"], r["answer1"], context=s)


def geobench(fx):
    # Every apstudy item has the same "id"; the row index is the identity.
    for i, r in enumerate(fx.json("daven3/geobench", "geobenchmark_apstudy.json")):
        ch = r["question"]["choices"]
        labels = [c["label"] for c in ch]
        if r.get("answerKey") not in labels:
            continue
        yield _row("GeoBench", "daven3/geobench", f"apstudy/{i}", r["question"]["stem"],
                   labels.index(r["answerKey"]), options=[c["text"] for c in ch])


def musictheory(fx):
    for fn in ("data/dev-00000-of-00001-33c852d919b7c9a2.parquet",
               "data/test-00000-of-00001-0b1d2a25639002c2.parquet"):
        for r in fx.parquet("m-a-p/MusicTheoryBench", fn):
            try:
                opts = ast.literal_eval(r["options"]) if isinstance(r["options"], str) else r["options"]
            except (ValueError, SyntaxError):
                continue
            keys = sorted(opts)
            if r["answer"] not in keys:
                continue
            yield _row("MusicTheoryBench", "m-a-p/MusicTheoryBench", f"{r['split']}/{r['id']}",
                       r["stem"], keys.index(r["answer"]), options=[opts[k] for k in keys])


_MQA_OPT = re.compile(r"(?:^|,)\s*([a-e])\s*\)\s*")


def mathqa(fx):
    for split in ("train", "validation", "test"):
        for i, r in enumerate(fx.parquet("allenai/math_qa", f"default/{split}/0000.parquet",
                                         revision=SOURCES["allenai/math_qa"])):
            parts = _MQA_OPT.split(r["options"])
            opts = [p.strip().rstrip(",").strip() for p in parts[2::2]]
            if len(opts) != 5 or r["correct"] not in "abcde":
                continue
            yield _row("MathQA", "allenai/math_qa", f"{split}/{i}", r["Problem"], r["correct"],
                       options=opts)


_CHESS = {
    "FIND_LAST_MOVE": ("find_last_move", "missing move", None),
    "FIND_NEXT_BEST_MOVE": ("find_next_best_move", "next best move", None),
    "FIND_FINAL_SCORE": ("find_final_score", "score", (["White Wins", "Black Wins", "Draw"],
                                                       {"1-0": 0, "0-1": 1, "1/2-1/2": 2})),
    "FIND_ADVANTAGED_PLAYER": ("find_advantaged_player", "Most advantaged",
                               (["White", "Black"], {"White": 0, "Black": 1})),
}


def chess(fx):
    for fn in ("test.csv", "train.csv"):
        for i, r in enumerate(fx.csv_rows("Thytu/ChessInstruct", fn)):
            spec = _CHESS.get(r.get("KIND"))
            if not spec:
                continue
            kind, key, mcq = spec
            try:
                out = json.loads(r["expected_output"])[key]
            except (ValueError, KeyError, TypeError):
                continue
            if mcq is None:
                yield _row("ChessInstruct", "Thytu/ChessInstruct", f"{fn}/{i}", r["input"], out,
                           context=r["task"], meta={"puzzle_kind": kind})
            elif out in mcq[1]:
                yield _row("ChessInstruct_mcq", "Thytu/ChessInstruct", f"{fn}/{i}", r["input"],
                           mcq[1][out], context=r["task"].split("\nOutput Format")[0],
                           options=mcq[0], meta={"puzzle_kind": kind})


def superglue(fx):
    sg = "aps/super_glue"
    for split in ("train", "validation"):
        for r in fx.parquet(sg, f"copa/{split}-00000-of-00001.parquet"):
            q = "what's the effect of this?" if r["question"] == "effect" else "what's the cause for this?"
            yield _row("SuperGLUE-CausalReasoning", sg, f"copa/{split}/{r['idx']}", q,
                       f"{float(r['label'])}", context=r["premise"],
                       options=[r["choice1"], r["choice2"]])
        for r in fx.parquet(sg, f"boolq/{split}-00000-of-00001.parquet"):
            yield _row("SuperGLUE-QA", sg, f"boolq/{split}/{r['idx']}", r["question"],
                       f"{float(r['label'])}", context=f"{r['passage']}")
        for r in fx.parquet(sg, f"rte/{split}-00000-of-00001.parquet"):
            yield _row("SuperGLUE-Entailment", sg, f"rte/{split}/{r['idx']}",
                       f"Premise: {r['premise']}, Hypothesis: {r['hypothesis']}",
                       f"{float(r['label'])}")
        for r in fx.parquet(sg, f"multirc/{split}-00000-of-00001.parquet"):
            idx = r["idx"]
            sid = f"multirc/{split}/{idx['paragraph']}/{idx['question']}/{idx['answer']}"
            yield _row("SuperGLUE-RC", sg, sid,
                       f"Paragraph: {r['paragraph']}\nQuestion: {r['question']}",
                       f"{float(r['label'])}", options=[r["answer"]])
        for r in fx.parquet(sg, f"wic/{split}-00000-of-00001.parquet"):
            yield _row("SuperGLUE-Wic", sg, f"wic/{split}/{r['idx']}",
                       f"Sentence 1: {r['sentence1']}\nSentence 2: {r['sentence2']}\n",
                       "Yes" if int(r["label"]) == 1 else "No", context=r["word"],
                       options=["Yes", "No"])
        for r in fx.parquet(sg, f"wsc.fixed/{split}-00000-of-00001.parquet"):
            yield _row("SuperGLUE-Wsc", sg, f"wsc/{split}/{r['idx']}",
                       f"Text: {r['text']}\n span2_text: {r['span2_text']}\n span1_text: {r['span1_text']}\n",
                       "Yes" if int(r["label"]) == 1 else "No", options=["Yes", "No"])
        for r in fx.parquet(sg, f"record/{split}-00000-of-00001.parquet"):
            ents = list(dict.fromkeys(r["entities"]))
            if not r["answers"] or not ents:
                continue
            idx = r["idx"]
            yield _row("SuperGLUE-ClozeTest", sg, f"record/{split}/{idx['passage']}/{idx['query']}",
                       f"Passage: {r['passage']}\nQuery: {r['query']}", ", ".join(r["answers"]),
                       options=ents)


def wmt19(fx):
    for pair in _WMT_PAIRS:
        foreign = pair.split("-")[0]
        for i, r in enumerate(fx.parquet("wmt/wmt19", f"{pair}/validation-00000-of-00001.parquet")):
            t = r["translation"]
            if t.get("en") and t.get(foreign):
                yield _row(f"WMT19-{pair}", "wmt/wmt19", f"{pair}/validation/{i}", t["en"], t[foreign])


def _boxed(s: str):
    from fugal_subnet.graders import last_boxed
    return last_boxed(s)


def math_sets(fx):
    for split in ("test", "train"):
        for i, r in enumerate(fx.parquet("openai/gsm8k", f"main/{split}-00000-of-00001.parquet")):
            ans = r["answer"].split("####")[-1].strip().replace(",", "")
            yield _row("GSM8K", "openai/gsm8k", f"{split}/{i}", r["question"], ans)
    subjects = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                "number_theory", "prealgebra", "precalculus")
    for subject in subjects:
        for split in ("test", "train"):
            fn = f"{subject}/{split}-00000-of-00001.parquet"
            for i, r in enumerate(fx.parquet("EleutherAI/hendrycks_math", fn)):
                ans = _boxed(r["solution"])
                if ans:
                    yield _row("MATH", "EleutherAI/hendrycks_math", f"{subject}/{split}/{i}",
                               r["problem"], ans)
    for r in fx.csv_rows("qq8933/AIME_1983_2024", "AIME_Dataset_1983_2024.csv"):
        yield _row("AIME", "qq8933/AIME_1983_2024", r["ID"], r["Question"], str(r["Answer"]).strip())
    for split in ("train", "validation", "test"):
        for r in fx.parquet("dreamerdeo/finqa", f"default/{split}/0000.parquet",
                            revision=SOURCES["dreamerdeo/finqa"]):
            pre = r["pre_text"] if isinstance(r["pre_text"], list) else ast.literal_eval(r["pre_text"])
            post = r["post_text"] if isinstance(r["post_text"], list) else ast.literal_eval(r["post_text"])
            table = r["table"] if isinstance(r["table"], list) else ast.literal_eval(r["table"])
            ctx = " ".join(pre) + "\n" + "\n".join(" | ".join(map(str, row)) for row in table) \
                + "\n" + " ".join(post)
            yield _row("FinQA", "dreamerdeo/finqa", r["id"], r["question"], r["answer"], context=ctx)


def socialiqa(fx):
    for split in ("train", "validation"):
        for i, r in enumerate(fx.parquet("allenai/social_i_qa", f"default/{split}/0000.parquet",
                                         revision=SOURCES["allenai/social_i_qa"])):
            label = int(r["label"]) - 1
            if label not in (0, 1, 2):
                continue
            yield _row("SocialiQA", "allenai/social_i_qa", f"{split}/{i}", r["question"], label,
                       context=r["context"], options=[r["answerA"], r["answerB"], r["answerC"]])


def ethics(fx):
    # RouterArena: commonsense and justice are True/False with Answer = 1 - label; deontology
    # and virtue are one multiple-choice question per scenario (its consecutive rows), the
    # answer the one option labelled 1. Checked against RouterArena's 283 Ethics items.
    rev = "hendrycks/ethics"
    for split in ("train", "test", "test_hard"):
        for i, r in enumerate(fx.csv_rows(rev, f"data/commonsense/{split}.csv")):
            if r.get("is_short") == "True":
                yield _row("Ethics_commonsense", rev, f"commonsense/{split}/{i}", r["input"],
                           1 - int(r["label"]), options=["False", "True"])
        for i, r in enumerate(fx.csv_rows(rev, f"data/justice/{split}.csv")):
            yield _row("Ethics_justice", rev, f"justice/{split}/{i}", r["scenario"],
                       1 - int(r["label"]), options=["False", "True"])
        groups = []
        for r in fx.csv_rows(rev, f"data/deontology/{split}.csv"):
            if groups and groups[-1][0] == r["scenario"]:
                groups[-1][1].append(r)
            else:
                groups.append((r["scenario"], [r]))
        for g, (scen, rows) in enumerate(groups):
            if len(rows) == 4 and sum(int(x["label"]) for x in rows) == 1:
                yield _row("Ethics_deontology", rev, f"deontology/{split}/{g}", scen,
                           [int(x["label"]) for x in rows].index(1),
                           options=[x["excuse"] for x in rows])
        groups = []
        for r in fx.csv_rows(rev, f"data/virtue/{split}.csv"):
            scen, _, trait = r["scenario"].partition(" [SEP] ")
            if groups and groups[-1][0] == scen:
                groups[-1][1].append((trait, int(r["label"])))
            else:
                groups.append((scen, [(trait, int(r["label"]))]))
        for g, (scen, rows) in enumerate(groups):
            if len(rows) == 5 and sum(lab for _, lab in rows) == 1:
                yield _row("Ethics_virtue", rev, f"virtue/{split}/{g}", scen,
                           [lab for _, lab in rows].index(1), options=[t for t, _ in rows])


CONVERTERS = (mmlupro, mmlu, medmcqa, pubmedqa, opentdb, qanta, narrativeqa, geobench,
              musictheory, mathqa, chess, superglue, wmt19, math_sets, socialiqa, ethics)


# --- RouterArena reference: shares and exclusion ------------------------------------------------
def routerarena_rows(fx) -> list[dict]:
    repo, rev = ROUTERARENA
    return fx.parquet(repo, "data/full-00000-of-00001.parquet", revision=rev,
                      columns=["Dataset name", "Global Index", "Context", "Question"])


def shares(ra_rows) -> dict[str, int]:
    """RouterArena's dataset shares by grading key, in parts per million of the 8,400, over
    the keys this pool carries (NOT_IN_POOL removed, the rest renormalised)."""
    c = Counter(r["Global Index"].split("_")[0] for r in ra_rows)
    for k in NOT_IN_POOL:
        c.pop(k, None)
    total = sum(c.values())
    out = {k: (v * 1_000_000) // total for k, v in sorted(c.items())}
    first = sorted(out)[0]
    out[first] += 1_000_000 - sum(out.values())      # exact sum
    return out


_WORDS = re.compile(r"\w+", re.UNICODE)


def _norm(s: str) -> str:
    return " ".join(_WORDS.findall((s or "").lower()))


def _shingles(s: str) -> set[str]:
    w = _norm(s).split()
    return {" ".join(w[i:i + SHINGLE_WORDS]) for i in range(len(w) - SHINGLE_WORDS + 1)}


def _identity(row) -> str:
    """A question's text identity: its context and question together. Some datasets ask the
    same generic question of every item ("what's the effect of this?", "How would Others
    feel as a result?"); the context is what makes the item."""
    return _norm((row.get("Context") or "") + "\n" + (row.get("Question") or ""))


class Exclusion:
    def __init__(self, ra_rows):
        self.text = {_identity(r) for r in ra_rows}
        self.sh = defaultdict(set)
        for r in ra_rows:
            key = r["Global Index"].split("_")[0]
            if key in SHINGLE_DATASETS:
                self.sh[key] |= _shingles(r["Question"])

    def hit(self, row) -> str | None:
        if _identity(row) in self.text:
            return "text"
        key = ra_dataset_of(row["ra_name"])
        if key in SHINGLE_DATASETS and self.sh[key] & _shingles(row["Question"]):
            return "shingle"
        return None


# --- pool -------------------------------------------------------------------------------------
def question_id(row) -> str:
    h = hashlib.sha256(f"{row['source']}:{row['source_id']}".encode()).hexdigest()[:16]
    return f"arena_{ra_dataset_of(row['ra_name'])}_{h}"


def to_item(row) -> dict:
    key = ra_dataset_of(row["ra_name"])
    meta = {"ra_dataset": key, "ra_name": row["ra_name"], "source": row["source"],
            "source_id": row["source_id"]}
    if row["Metadata"]:
        meta["ra_metadata"] = row["Metadata"]
    return {"question_id": question_id(row), "benchmark": f"arena_{key}",
            "prompt": format_prompt(row), "route_text": route_text(row), "gold": row["Answer"],
            "grader_id": "arena", "metadata": meta}


def code_items(v3_pool: list[dict]) -> list[dict]:
    out = []
    for q in v3_pool:
        if q["benchmark"] in CODE_BENCHMARKS:
            meta = dict(q.get("metadata") or {})
            meta.update({"ra_dataset": "LiveCodeBench", "ra_name": "LiveCodeBench",
                         "source": q["benchmark"], "source_id": q["question_id"]})
            out.append({"question_id": f"arena_LiveCodeBench_{q['question_id']}",
                        "benchmark": "arena_LiveCodeBench", "prompt": q["prompt"],
                        # A code task is its own content: the v3 prompt carries no harness
                        # template, so the router sees the same text the model answers.
                        "route_text": q["prompt"],
                        "gold": q["gold"], "grader_id": "arena", "metadata": meta})
    return out


def _order(qid: str) -> str:
    return hashlib.sha256(f"{POOL_SALT}:{qid}".encode()).hexdigest()


def build_pool(fx: Fetcher, v3_pool: list[dict], report: dict | None = None):
    """(pool, exclusions, weights). Deterministic given the pinned inputs."""
    ra = routerarena_rows(fx)
    weights = shares(ra)
    excl = Exclusion(ra)
    caps = {k: max(CAP_FLOOR, math.ceil(CAP_TOTAL * w / 1_000_000)) for k, w in weights.items()}
    by_key: dict[str, dict[str, dict]] = defaultdict(dict)   # bounded: the cap smallest orders
    excluded: dict[str, list[str]] = defaultdict(list)
    loaded, dup, seen = Counter(), Counter(), defaultdict(set)

    def keep(key, item):
        items = by_key[key]
        items[item["question_id"]] = item
        if len(items) > 2 * caps.get(key, CAP_FLOOR):            # prune: memory stays ~2 x caps
            for q in sorted(items, key=_order)[caps.get(key, CAP_FLOOR):]:
                del items[q]

    for conv in CONVERTERS:
        for row in conv(fx):
            if not row["Question"] or row["Answer"] in ("", "None"):
                continue
            key = ra_dataset_of(row["ra_name"])
            if key not in weights:
                continue
            qid = question_id(row)
            loaded[key] += 1
            if excl.hit(row):
                excluded[key].append(qid)
                continue
            if qid in seen[key]:
                dup[key] += 1
                continue
            seen[key].add(qid)
            keep(key, to_item(row))
    for item in code_items(v3_pool):
        loaded["LiveCodeBench"] += 1
        keep("LiveCodeBench", item)
    pool, kept = [], {}
    for key, items in sorted(by_key.items()):
        chosen = sorted(items, key=_order)[:caps.get(key, CAP_FLOOR)]
        kept[key] = len(chosen)
        pool.extend(items[q] for q in chosen)
    pool.sort(key=lambda q: q["question_id"])
    if report is not None:
        report.update({"loaded": dict(loaded), "excluded": {k: len(v) for k, v in excluded.items()},
                       "duplicates": dict(dup), "kept": kept})
    return pool, {k: sorted(v) for k, v in sorted(excluded.items())}, weights


MANIFEST_NAME = "pool_manifest_v5.json"
_MANIFEST: dict | None = None


def manifest() -> dict:
    """The pinned pool manifest (pool_manifest_v5.json, pinned by check_safety_invariants)."""
    global _MANIFEST
    if _MANIFEST is None:
        with open(str(files("fugal_subnet.benchmarks") / MANIFEST_NAME), encoding="utf-8") as f:
            _MANIFEST = json.load(f)
    return _MANIFEST


def weights_ppm() -> dict[str, int]:
    """The exam mix: each RouterArena dataset's share of RouterArena, in ppm (the manifest's).
    A dataset with no questions in the pool (ArcMMLU) simply draws nothing; the draw spreads
    its share over the rest (exam_v5.draw_exam)."""
    return {str(k): int(v) for k, v in sorted(manifest()["weights_ppm"].items())}


def dataset_key(item: dict) -> str:
    """A pool item's RouterArena dataset key: the exam's stratum and its mix key."""
    return str((item.get("metadata") or {}).get("ra_dataset", ""))


def in_coverage(question_id: str) -> bool:
    """True when the question is in the coverage part of the v5 pool, False when in the exam
    part: sha256("<POOL_SALT>:partition:<id>")'s first 8 bytes, as an integer h, against
    config.COVERAGE_PARTITION_V5 (a fraction p): coverage iff h < p * 2^64. Fixed per id."""
    from fractions import Fraction

    from fugal_subnet import config
    p = Fraction(config.COVERAGE_PARTITION_V5)
    h = int.from_bytes(hashlib.sha256(f"{POOL_SALT}:partition:{question_id}".encode()).digest()[:8],
                       "big")
    return h * p.denominator < p.numerator * (1 << 64)


def partition(pool) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """({dataset: exam-part ids}, {dataset: coverage-part ids}), ids sorted."""
    exam: dict[str, list[str]] = defaultdict(list)
    cover: dict[str, list[str]] = defaultdict(list)
    for q in pool:
        qid = str(q["question_id"])
        (cover if in_coverage(qid) else exam)[dataset_key(q)].append(qid)
    return ({k: sorted(v) for k, v in sorted(exam.items())},
            {k: sorted(v) for k, v in sorted(cover.items())})


def content_hash(pool: list[dict]) -> str:
    """Every field the grader, the prompt or the router reads, in id order (as
    loader.content_hash_v3)."""
    h = hashlib.sha256()
    for q in sorted(pool, key=lambda x: x["question_id"]):
        meta = q.get("metadata") or {}
        h.update(json.dumps([q["question_id"], q["prompt"], q["route_text"],
                             json.dumps(q["gold"], sort_keys=True, default=str), q["grader_id"],
                             q["benchmark"], meta.get("ra_dataset", ""),
                             json.dumps(meta.get("checker"), sort_keys=True, default=str)],
                            separators=(",", ":"), sort_keys=True).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()
