#!/usr/bin/env python3
#
# THE GRADING CONTRACT
#
# A grader is a CONSENSUS RULE, not an eval script:
#   grade(task, reply_text) -> {0, 1}
# must be a pure function — same bits in, same bit out, on any machine, any OS,
# any year. No network, no wall-clock dependence, no RNG. Every divergence
# between two validators' grades is a consensus failure; every grader bug is a
# paid exploit. These checkers are battle-tested (trailing-period numeric fix,
# typing-import stubs, brace-aware \boxed parsing, kill-tree exec timeout) —
# do not "improve" them casually: a semantic change is a new grader version
# that applies from the NEXT epoch only (past epochs stay re-gradeable under
# their committed hash).
#
# VERSIONING: the grader version is sha256 of this file's bytes (grader_hash()).
# Epoch commitments include it; scores are only valid under the committed hash.
# NOTE the hash does NOT pin third-party deps (math_verify/sympy for
# boxed_math) — epoch reports should record those versions.
#
# SECURITY: exec_io / exec_unittest EXECUTE model-generated Python. Off unless
# allow_exec=True. Sandbox: subprocess in a new process group/session,
# kill-the-whole-tree on timeout (Windows pipe-reaping hang), DEVNULL pipes
# (output cap), program-size cap, and on POSIX rlimits (memory / CPU /
# file-size / nproc). There is NO network isolation inside the sandbox —
# run the validator inside a container/VM (see docs/VALIDATOR_GUIDE.md).
"""fugal_subnet.graders — pure, versioned checkers for the Fugal validator.

Checker inventory:
  numeric_final  last number in reply vs gold (float-safe)      gsm8k
  boxed_math     math_verify + brace-aware \\boxed fallback      math
  integer_exact  integer answer match                           aime
  letter_mcq     'answer is <letter>' extraction, A–J           mmlu, gpqa
  constraint_set mechanical instruction-following constraints   ifeval
  exec_io        run candidate code, parent-verified outputs    humaneval, livecode
  exec_unittest  legacy exit-code contract (forgeable; unused for new epochs)

Entry point:  grade(task, reply, allow_exec=False) -> int  (0/1, never raises)
Version:      grader_hash() -> "sha256:..."
Attack suite: python -m fugal_subnet.attacks.run_attacks
"""
from __future__ import annotations
import hashlib, json, os, re, subprocess, sys, tempfile

# ---------- sandbox limits (exec_unittest only) ----------
EXEC_TIMEOUT_SECS = 10
EXEC_PROGRAM_CAP = 512_000      # bytes of assembled program; larger grades 0
RLIMIT_AS_BYTES = 1 << 30       # 1 GiB address space (POSIX only)
RLIMIT_CPU_SECS = EXEC_TIMEOUT_SECS + 5
RLIMIT_FSIZE_BYTES = 16 << 20   # 16 MiB max file a snippet can write (POSIX only)
RLIMIT_NPROC = 256              # fork-bomb cap (POSIX only)


# ---------- shared extraction helpers ----------
def numeric_answer(text):
    t = (text or "").replace(",", "").replace("−", "-")   # unicode minus
    nums = re.findall(r"-?\d+\.?\d*", t)
    return nums[-1].rstrip(".") if nums else None              # "18." -> "18"


def extract_code(text):
    """Pull the python from a model reply (largest ``` block, else raw)."""
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text or "", re.DOTALL)
    return max(blocks, key=len).strip() if blocks else (text or "").strip()


def last_boxed(text):
    """Content of the last \\boxed{...}, brace-aware (handles \\frac{a}{b})."""
    i = (text or "").rfind("\\boxed{")
    if i < 0:
        return None
    j, depth, start = i + len("\\boxed{"), 1, i + len("\\boxed{")
    while j < len(text) and depth:
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
        j += 1
    return text[start:j - 1] if depth == 0 else None


def _norm_latex(s):
    return re.sub(r"\\left|\\right|\\,|\\!|\s+", "", s or "")


# ---------- exec sandbox ----------
def _posix_preexec():
    """Child-side hardening: own session (kill-tree target) + resource limits."""
    import resource
    os.setsid()
    resource.setrlimit(resource.RLIMIT_AS, (RLIMIT_AS_BYTES, RLIMIT_AS_BYTES))
    resource.setrlimit(resource.RLIMIT_CPU, (RLIMIT_CPU_SECS, RLIMIT_CPU_SECS))
    resource.setrlimit(resource.RLIMIT_FSIZE, (RLIMIT_FSIZE_BYTES, RLIMIT_FSIZE_BYTES))
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (RLIMIT_NPROC, RLIMIT_NPROC))
    except (ValueError, OSError):
        pass                      # some platforms disallow lowering it; non-fatal


def _run_sandboxed(program, timeout=EXEC_TIMEOUT_SECS):
    """Run an assembled program in the sandbox. Returns the exit code, or None
    on timeout/oversize/launch failure.

    Hardened against a timeout-reaping deadlock: generated code that spawns a
    child holding the stdout pipe makes post-timeout reaping block forever
    (seen wedging worker threads on Windows). We launch in a new process
    group/session with DEVNULL pipes and hard-kill the whole tree on timeout
    instead of reaping. POSIX children additionally get rlimits
    (memory/CPU/fsize/nproc); Windows has no rlimit equivalent — the kill-tree
    timeout is the only cap there.
    """
    if len(program.encode("utf-8", "replace")) > EXEC_PROGRAM_CAP:
        return None
    fd, path = tempfile.mkstemp(suffix=".py")
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kw["preexec_fn"] = _posix_preexec
    try:
        os.write(fd, program.encode("utf-8", "replace")); os.close(fd)
        p = subprocess.Popen([sys.executable, path],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kw)
        try:
            return p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                import signal
                try: os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except Exception: p.kill()
            return None
    except Exception:
        return None
    finally:
        try: os.remove(path)
        except OSError: pass


def run_unit_test(code, test_src, entry_point, timeout=EXEC_TIMEOUT_SECS):
    """Legacy exit-code contract: 1 iff the program exits 0. KNOWN FORGEABLE
    by candidate code that calls sys.exit(0)/os._exit(0) at import time,
    before the tests run (documented in fugal_subnet/attacks/run_attacks.py;
    honest-model replies are unaffected). Kept ONLY as a fallback for tasks
    without extractable I/O pairs — everything else uses exec_io, whose
    parent-side output comparison is immune: a forged pass requires actually
    computing the correct outputs.
    """
    return int(_run_sandboxed(f"{code}\n\n{test_src}\n\ncheck({entry_point})\n",
                              timeout) == 0)


# ---------- checkers (task, reply, allow_exec) -> {0,1} ----------
def numeric_final(task, reply, _allow=False):
    n = numeric_answer(reply)
    try:
        return int(n is not None and float(n) == float(task["gold"]))
    except ValueError:
        return 0


def integer_exact(task, reply, _allow=False):
    n = numeric_answer(reply)
    try:
        return int(n is not None and int(float(n)) == int(task["gold"]))
    except ValueError:
        return 0


def boxed_math(task, reply, _allow=False):
    # 1) semantic check via math_verify (timeouts off: broken on Windows)
    try:
        from math_verify import parse, verify
        gold = parse("$" + task["gold"] + "$", parsing_timeout=None)
        pred = parse(reply or "", parsing_timeout=None)
        if verify(gold, pred, timeout_seconds=None):
            return 1
    except Exception:
        pass
    # 2) fallback: normalized string match on the last \boxed{...}
    b = last_boxed(reply)
    return int(b is not None and _norm_latex(b) == _norm_latex(task["gold"]))


def letter_mcq(task, reply, _allow=False):
    m = re.findall(r"answer is\s*:?\s*[\*_\(]*([A-J])[\*_\)\.]*", reply or "", re.IGNORECASE)
    if not m:
        m = re.findall(r"\b([A-J])\b", (reply or "").strip()[-40:])
    return int(bool(m) and m[-1].upper() == task["gold"])


def _strip_fence(text):
    """Deterministically strip ONE wrapping ``` fence pair if the whole reply
    is fenced (models fence JSON despite instructions; the rule is mechanical
    and stated here, not judged)."""
    t = (text or "").strip()
    m = re.match(r"^```[a-zA-Z]*\s*\n(.*?)\n?\s*```$", t, re.DOTALL)
    return m.group(1).strip() if m else t


_JSON_TYPES = {"string": str, "number": (int, float), "boolean": bool,
               "array": list, "object": dict}


def _check_constraint(c, reply):
    k = c["kind"]
    if k == "word_count_range":
        return c["min"] <= len(reply.split()) <= c["max"]
    if k == "line_count_exact":
        return len([l for l in reply.splitlines() if l.strip()]) == c["n"]
    if k == "must_include":            # substring, at least k times
        return reply.count(c["s"]) >= c.get("k", 1)
    if k == "must_exclude":
        return c["s"] not in reply
    if k == "starts_with":
        return reply.lstrip().startswith(c["s"])
    if k == "ends_with":
        return reply.rstrip().endswith(c["s"])
    if k == "lowercase_only":
        return reply == reply.lower()
    if k == "no_digits":
        return re.search(r"\d", reply) is None
    if k == "bullet_lines":            # every non-empty line is a "- " bullet
        lines = [l for l in reply.splitlines() if l.strip()]
        return bool(lines) and all(l.lstrip().startswith("- ") for l in lines)
    if k == "json_object":             # exactly these keys, with these types
        try:
            obj = json.loads(reply)
        except Exception:
            return False
        if not isinstance(obj, dict) or set(obj) != set(c["required"]):
            return False
        return all(isinstance(obj[key], _JSON_TYPES[t])
                   and not (t == "number" and isinstance(obj[key], bool))
                   for key, t in c["required"].items())
    return False                       # unknown kind NEVER passes


def constraint_set(task, reply, _allow=False):
    """F1 instruction-following: every constraint in checker.params must hold
    on the fence-stripped reply. The constraints ARE the gold answer."""
    r = _strip_fence(reply)
    if not r:
        return 0
    cons = task["checker"]["params"]["constraints"]
    return int(bool(cons) and all(_check_constraint(c, r) for c in cons))


def exec_io(task, reply, allow=False):
    """F3 code: run the candidate on checker.params.inputs in the sandbox; the
    child writes its outputs to a file; the PARENT compares them to the gold
    outputs, which the child never sees. Immune to exit-code forgery: passing
    requires computing the correct outputs, i.e. solving the task. Inputs are
    JSON-safe by generator contract (ints/strs/bools/lists/dicts, no floats)."""
    if not allow:
        return 0
    params = task["checker"]["params"]
    code = extract_code(reply)
    fd, outpath = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    driver = (
        f"\n\nif __name__ == '__main__':\n"
        f"    import json as _json\n"
        f"    _inputs = _json.loads({json.dumps(json.dumps(params['inputs']))})\n"
        f"    _res = [{params['func']}(*_a) for _a in _inputs]\n"
        f"    with open({outpath!r}, 'w', encoding='utf-8') as _f:\n"
        f"        _f.write(_json.dumps(_res))\n")
    try:
        rc = _run_sandboxed(code + driver)
        if rc != 0:
            return 0
        with open(outpath, encoding="utf-8") as f:
            raw = f.read(1 << 20)                 # 1 MiB output cap
        got = json.loads(raw)
        want = task["gold"]
        return int(json.dumps(got, sort_keys=True) == json.dumps(want, sort_keys=True))
    except Exception:
        return 0
    finally:
        try: os.remove(outpath)
        except OSError: pass


def exec_unittest(task, reply, allow=False):
    if not allow:
        return 0
    # official-harness parity: the stub's imports (typing etc.) are part of the
    # program context; a model that returns only the function must not fail on them
    imports = "\n".join(l for l in task.get("stub", "").splitlines()
                        if l.startswith(("import ", "from ")))
    code = ("from typing import List, Dict, Tuple, Optional, Any\n"
            + imports + "\n" + extract_code(reply))
    return run_unit_test(code, task["test"], task["entry"])


CHECKERS = {
    "numeric_final":  numeric_final,
    "integer_exact":  integer_exact,
    "boxed_math":     boxed_math,
    "letter_mcq":     letter_mcq,
    "exec_unittest":  exec_unittest,   # legacy cache re-grading ONLY (forgeable)
    "constraint_set": constraint_set,  # F1 instruction-following
    "exec_io":        exec_io,         # F3 code (forgery-proof, parent-verified)
}

# Legacy domain -> checker map, for tasks that predate the explicit
# task["checker"]["id"] schema.
DOMAIN_CHECKER = {
    "gsm8k":     "numeric_final",
    "humaneval": "exec_unittest",
    "math500":   "boxed_math",
    "aime25":    "integer_exact",
    "mmlupro":   "letter_mcq",
    "math_hard": "boxed_math",
    "aime":      "integer_exact",
    "hmmt":      "boxed_math",
    "gpqa":      "letter_mcq",
}


def grade(task, reply, allow_exec=False):
    """Grade one reply. Returns 0 or 1; never raises (failure grades 0).

    Checker resolution: task["checker"]["id"] if
    present, else DOMAIN_CHECKER[task["domain"]] (legacy gate-harness tasks).
    """
    try:
        cid = (task.get("checker") or {}).get("id") or DOMAIN_CHECKER[task["domain"]]
        return int(CHECKERS[cid](task, reply, allow_exec))
    except Exception:
        return 0


def grader_hash():
    """Version of this grading contract: sha256 of this file's bytes with
    line endings normalized to LF — git's CRLF translation would otherwise
    make the SAME grader hash differently on Windows vs Linux checkouts,
    breaking cross-validator hash agreement for identical semantics."""
    with open(os.path.abspath(__file__), "rb") as f:
        return "sha256:" + hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


if __name__ == "__main__":
    print(grader_hash())
