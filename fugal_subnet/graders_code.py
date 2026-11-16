#!/usr/bin/env python3
#
# THE V3 CODE GRADING CONTRACT
#
# Like graders.py this file is a CONSENSUS RULE: grade(task, reply) -> {0, 1}
# must give the same bit on every miner TD. Its hash (grader_hash()) is pinned
# in scripts/check_safety_invariants.py; any byte change is a new grader
# version. graders.py stays immutable; this file adds two checkers for
# protocol v3 code questions:
#
#   exec_stdio  a whole program: one run per test, that test's input on stdin,
#               stdout compared to the expected output by whitespace tokens
#               (float tolerance only when the task sets one).
#   exec_func   a function: one run, the inputs as JSON on stdin, the results
#               compared as canonical JSON, like graders.exec_io, but with a
#               LeetCode-style prelude (typing, collections, ...) so a reply
#               that omits `from typing import List` is not failed for it.
#
# SECURITY: both EXECUTE model-written Python. Off unless allow_exec=True.
# The sandbox is built from kernel features every miner TD has (x86_64 Linux),
# not from Python-level tricks:
#   - a seccomp filter installed in the child before the candidate's code runs:
#     no sockets, no fork/exec (threads are allowed), no file creation,
#     deletion, rename, chmod or open-for-write, no ptrace, no signals to any
#     other process, no namespaces, no io_uring, no bpf;
#   - rlimits: CPU seconds, address space, output file size, open files, stack;
#   - its own session, killed as a group; env cleared except the pins below;
#     -S (stdlib only, so the image's site-packages cannot change a grade);
#     PYTHONHASHSEED pinned; int->str digit limit off; time.sleep refused;
#   - the grading process makes itself non-dumpable (prctl), so a child cannot
#     read the parent's memory (which holds every gold answer) or environment;
#   - if running as root, the child drops to uid/gid 65534.
# The PARENT decides the grade from the child's exit status, its measured CPU
# time and its stdout; the expected outputs never leave the parent.
#
# DETERMINISM: the limits are CPU time (rusage), not wall clock. A wall-clock
# backstop (WALL_FACTOR x the CPU limit + WALL_EXTRA_SECS) only ends programs
# that block without using CPU. The pool keeps only tests whose reference
# solution needs <= 25% of the CPU limit (scripts/calibrate_code_tests.py), so
# an efficient answer passes with a wide margin on any TD; answers near a limit
# can differ between TDs, and shared grades (the median across coldkeys) absorb
# that. See docs/decisions/CODE_GRADER_V3.md.
"""fugal_subnet.graders_code — the v3 code checkers (exec_stdio, exec_func).

Entry point:  grade(task, reply, allow_exec=False) -> int  (0/1, never raises)
Version:      grader_hash() -> "sha256:..."
Self-test:    sandbox_selftest() -> (ok, reason)  — run once at harness start
Attack suite: python -m fugal_subnet.attacks.run_code_attacks
"""
import hashlib
import json
import math
import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

from fugal_subnet.graders import extract_code  # same extraction as exec_io

# ---------- limits ----------
STDIO_CPU_SECS = 3.0            # CPU per test (exec_stdio)
FUNC_CPU_SECS = 6.0             # CPU for the one run of all inputs (exec_func)
QUESTION_CPU_SECS = 24.0        # CPU budget for all tests of one question
WALL_FACTOR = 3.0               # wall backstop = factor x CPU limit + extra
WALL_EXTRA_SECS = 2.0
PROGRAM_CAP = 512_000           # bytes of assembled program (graders.EXEC_PROGRAM_CAP)
OUTPUT_CAP = 1 << 20            # bytes of stdout per run; more grades 0
AS_BYTES = 1 << 30              # address space
STACK_BYTES = 64 << 20          # deep recursion behaves the same on every TD
NOFILE = 64
MAX_TESTS = 64                  # a task with more tests grades 0 (pool caps it lower)
FLOAT_TOKEN = re.compile(rb"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
FUNC_NAME = re.compile(r"^[A-Za-z_]\w*(\(\))?(\.[A-Za-z_]\w*)*$")
RESULT_MARKER = "@@FUGAL_RESULT@@"
SANDBOX_UNAVAILABLE = 97        # the child's exit code when it cannot install the filter

# The child's whole environment. -S keeps site-packages out; no -E/-I, because
# those would also drop PYTHONHASHSEED.
CHILD_ENV = {
    "PYTHONHASHSEED": "0",
    "PYTHONIOENCODING": "utf-8",
    "PYTHONDONTWRITEBYTECODE": "1",
    "LC_ALL": "C.UTF-8",
}

# LeetCodeDataset's own prompt prelude, so reference solutions and model
# replies run in the context the problems were written for.
FUNC_PRELUDE = (
    "import random\nimport functools\nimport collections\nimport string\n"
    "import math\nimport datetime\n\nfrom typing import *\nfrom functools import *\n"
    "from collections import *\nfrom itertools import *\nfrom heapq import *\n"
    "from bisect import *\nfrom string import *\nfrom operator import *\n"
    "from math import *\n\ninf = float('inf')\n\n"
)

# ---------- the child ----------
# Runs as `python -S -s -B -c RUNNER <program>`. It installs the seccomp filter
# with ctypes BEFORE reading the candidate's program, then runs it as __main__.
RUNNER = r'''
import sys
def _fugal_sandbox():
    import ctypes, os
    ARCH_X86_64 = 0xC000003E
    ALLOW, KILL = 0x7FFF0000, 0x80000000
    EPERM, ENOSYS = 0x00050000 | 1, 0x00050000 | 38
    LD, JEQ, JSET, JGE, RET = 0x20, 0x15, 0x45, 0x35, 0x06
    WRITE_FLAGS = 1 | 2 | 0o100 | 0o1000 | 0o2000   # O_WRONLY|O_RDWR|O_CREAT|O_TRUNC|O_APPEND
    CLONE_THREAD = 0x10000
    me = os.getpid()
    deny = [
        41, 53, 42, 49, 50, 43, 288,             # socket socketpair connect bind listen accept accept4
        57, 58, 59, 322,                         # fork vfork execve execveat
        101, 310, 311,                           # ptrace process_vm_readv/writev
        200, 424, 434, 438,                      # tkill pidfd_send_signal pidfd_open pidfd_getfd
        85, 87, 263, 82, 264, 316, 83, 258, 84,  # creat unlink unlinkat rename renameat renameat2 mkdir mkdirat rmdir
        88, 266, 86, 265, 90, 268, 452,          # symlink symlinkat link linkat chmod fchmodat fchmodat2
        92, 94, 260, 76, 132, 235, 261, 280,     # chown lchown fchownat truncate utime utimes futimesat utimensat
        133, 259, 165, 166, 155, 161,            # mknod mknodat mount umount2 pivot_root chroot
        188, 189, 190, 197, 198, 199,            # setxattr lsetxattr fsetxattr removexattr lremovexattr fremovexattr
        303, 304, 428, 429, 430, 431, 432, 433, 442,   # handles, new mount API
        425, 426, 427,                           # io_uring
        272, 308, 321, 298, 323, 250, 248, 249,  # unshare setns bpf perf_event_open userfaultfd keyctl add_key request_key
        246, 320, 175, 313, 176, 169, 167, 168, 163, 172, 173,  # kexec init/finit/delete_module reboot swap acct iopl ioperm
    ]
    f = [(LD, 0, 0, 4), (JEQ, 1, 0, ARCH_X86_64), (RET, 0, 0, KILL),
         (LD, 0, 0, 0), (JGE, 0, 1, 0x40000000), (RET, 0, 0, EPERM)]
    def arg(i):
        return 16 + 8 * i
    # open(path, flags) / openat(dirfd, path, flags): read-only only
    for nr, a in ((2, 1), (257, 2)):
        f += [(JEQ, 0, 4, nr), (LD, 0, 0, arg(a)), (JSET, 0, 1, WRITE_FLAGS),
              (RET, 0, 0, EPERM), (RET, 0, 0, ALLOW), (LD, 0, 0, 0)]
    # clone: threads only
    f += [(JEQ, 0, 4, 56), (LD, 0, 0, arg(0)), (JSET, 1, 0, CLONE_THREAD),
          (RET, 0, 0, EPERM), (RET, 0, 0, ALLOW), (LD, 0, 0, 0)]
    # Calls that name a process may name only this one (its pid, or 0 = self):
    # kill, tgkill, rt_sigqueueinfo, rt_tgsigqueueinfo (signals), prlimit64,
    # sched_setaffinity/setscheduler/setparam/setattr, setpriority, ioprio_set
    # (so it cannot starve or re-limit the grader). kill(0) is its own group,
    # which holds only itself.
    for nr, a in ((62, 0), (234, 0), (129, 0), (297, 0), (302, 0), (203, 0),
                  (144, 0), (142, 0), (314, 0), (141, 1), (251, 1)):
        f += [(JEQ, 0, 5, nr), (LD, 0, 0, arg(a)), (JEQ, 2, 0, me), (JEQ, 1, 0, 0),
              (RET, 0, 0, EPERM), (RET, 0, 0, ALLOW), (LD, 0, 0, 0)]
    deny += [256, 279, 312]                      # migrate_pages move_pages kcmp
    # clone3 and openat2 carry their flags in a struct the filter cannot read:
    # ENOSYS, so libc falls back to clone / openat
    f += [(JEQ, 0, 1, 435), (RET, 0, 0, ENOSYS), (JEQ, 0, 1, 437), (RET, 0, 0, ENOSYS)]
    for nr in deny:
        f += [(JEQ, 0, 1, nr), (RET, 0, 0, EPERM)]
    f.append((RET, 0, 0, ALLOW))

    class Ins(ctypes.Structure):
        _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte),
                    ("jf", ctypes.c_ubyte), ("k", ctypes.c_uint)]
    class Prog(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(Ins))]
    ins = (Ins * len(f))(*[Ins(*x) for x in f])
    prog = Prog(len(f), ins)
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
                           ctypes.c_ulong, ctypes.c_ulong]
    if libc.prctl(38, 1, 0, 0, 0) != 0:                                 # PR_SET_NO_NEW_PRIVS
        os._exit(97)
    if libc.prctl(22, 2, ctypes.addressof(prog), 0, 0) != 0:            # PR_SET_SECCOMP, FILTER
        os._exit(97)

    # Sleeping uses no CPU, so only the wall-clock backstop would end it, and a
    # sleep near that backstop could pass on one TD and fail on another. Refuse
    # it outright: the audit event (3.12+) and the function itself (3.11).
    def _no_sleep(*_a, **_k):
        raise PermissionError("sleep is not allowed")
    def _audit(event, args):
        if event == "time.sleep":
            raise PermissionError("sleep is not allowed")
    sys.addaudithook(_audit)
    import time as _time
    _time.sleep = _no_sleep
    if hasattr(sys, "set_int_max_str_digits"):
        sys.set_int_max_str_digits(0)

_fugal_sandbox()
del _fugal_sandbox
_path = sys.argv[1]
with open(_path, "rb") as _f:
    _src = _f.read()
sys.argv = ["solution.py"]
if sys.path and sys.path[0] in ("", "."):
    del sys.path[0]
_code = compile(_src, "solution.py", "exec")
del _f, _src, _path
exec(_code, {"__name__": "__main__", "__builtins__": __builtins__})
'''


@dataclass
class RunResult:
    exit_code: int | None       # None: killed by the wall-clock backstop
    cpu_secs: float
    stdout: bytes
    output_overflow: bool

    @property
    def clean(self) -> bool:
        return self.exit_code == 0 and not self.output_overflow


_parent_hardened = False


def _harden_parent():
    """Make the grading process non-dumpable once, so the candidate's process
    (same uid) cannot open /proc/<parent>/mem or /environ. Idempotent."""
    global _parent_hardened
    if _parent_hardened:
        return
    try:
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl(4, 0, 0, 0, 0)                                # PR_SET_DUMPABLE, 0
    except Exception:
        pass
    _parent_hardened = True


def _preexec(cpu_secs: float):
    # Everything is computed and imported HERE, in the parent: the function
    # below runs in the forked child before exec, where importing (or anything
    # that takes a lock another thread of the grader might hold) can deadlock.
    soft = int(math.ceil(cpu_secs)) + 1
    as_lim, fsize, nofile, stack = AS_BYTES, OUTPUT_CAP + 1, NOFILE, STACK_BYTES
    drop_root = os.geteuid() == 0

    def fn():
        resource.setrlimit(resource.RLIMIT_CPU, (soft, soft + 1))
        resource.setrlimit(resource.RLIMIT_AS, (as_lim, as_lim))
        resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
        resource.setrlimit(resource.RLIMIT_NOFILE, (nofile, nofile))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        try:
            resource.setrlimit(resource.RLIMIT_STACK, (stack, stack))
        except (ValueError, OSError):
            pass
        if drop_root:
            os.setgroups([])
            os.setgid(65534)
            os.setuid(65534)
    return fn


def _run_once(prog_path: str, stdin_bytes: bytes, workdir: str, cpu_secs: float) -> RunResult:
    in_path = os.path.join(workdir, "stdin")
    out_path = os.path.join(workdir, "stdout")
    with open(in_path, "wb") as f:
        f.write(stdin_bytes)
    with open(in_path, "rb") as fin, open(out_path, "wb") as fout:
        if os.geteuid() == 0:
            os.chmod(prog_path, 0o644)
        p = subprocess.Popen(
            [sys.executable, "-S", "-s", "-B", "-c", RUNNER, prog_path],
            stdin=fin, stdout=fout, stderr=subprocess.DEVNULL, cwd=workdir,
            env=dict(CHILD_ENV), close_fds=True, start_new_session=True,
            preexec_fn=_preexec(cpu_secs))
    deadline = time.monotonic() + cpu_secs * WALL_FACTOR + WALL_EXTRA_SECS
    status, ru, killed = None, None, False
    while True:
        pid, st, r = os.wait4(p.pid, os.WNOHANG)
        if pid:
            status, ru = st, r
            break
        if time.monotonic() > deadline:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except OSError:
                pass
            _, status, ru = os.wait4(p.pid, 0)
            killed = True
            break
        time.sleep(0.002)
    try:
        os.killpg(p.pid, signal.SIGKILL)       # threads are in-process; this is belt and braces
    except OSError:
        pass
    p.returncode = os.waitstatus_to_exitcode(status)
    cpu = (ru.ru_utime + ru.ru_stime) if ru else float("inf")
    size = os.path.getsize(out_path)
    with open(out_path, "rb") as f:
        out = f.read(OUTPUT_CAP + 1)
    return RunResult(exit_code=None if killed else p.returncode, cpu_secs=cpu,
                     stdout=out, output_overflow=size > OUTPUT_CAP)


def run_program(program: str, stdins: list, cpu_secs: float = STDIO_CPU_SECS,
                stop_on_failure: bool = False) -> list:
    """Run `program` once per stdin in the sandbox. The single code path both
    the checkers and scripts/calibrate_code_tests.py use. Returns RunResults;
    an over-cap program returns []."""
    data = program.encode("utf-8", "replace")
    if len(data) > PROGRAM_CAP:
        return []
    _harden_parent()
    workdir = tempfile.mkdtemp(prefix="fugal-code-")
    try:
        os.chmod(workdir, 0o755 if os.geteuid() == 0 else 0o700)
        prog_path = os.path.join(workdir, "solution.py")
        with open(prog_path, "wb") as f:
            f.write(data)
        results, spent = [], 0.0
        for s in stdins:
            b = s if isinstance(s, bytes) else str(s).encode("utf-8")
            r = _run_once(prog_path, b, workdir, cpu_secs)
            results.append(r)
            spent += r.cpu_secs
            if stop_on_failure and (not r.clean or r.cpu_secs > cpu_secs
                                    or spent > QUESTION_CPU_SECS):
                break
        return results
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# ---------- comparison ----------
def _tokens_match(got: bytes, want: bytes, float_tol) -> bool:
    a, b = got.split(), want.split()
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x == y:
            continue
        if float_tol is None or not (FLOAT_TOKEN.match(x) and FLOAT_TOKEN.match(y)):
            return False
        fx, fy = float(x), float(y)
        if not (math.isfinite(fx) and math.isfinite(fy)):
            return False
        if abs(fx - fy) > float_tol * max(1.0, abs(fy)):
            return False
    return True


def _json_match(got, want, float_tol) -> bool:
    if float_tol is None:
        return json.dumps(got, sort_keys=True) == json.dumps(want, sort_keys=True)
    if isinstance(want, bool) or isinstance(got, bool):
        return got is want
    if isinstance(want, (int, float)) and isinstance(got, (int, float)):
        if not (math.isfinite(got) and math.isfinite(want)):
            return False
        return abs(got - want) <= float_tol * max(1.0, abs(want))
    if isinstance(want, list) and isinstance(got, list):
        return len(got) == len(want) and all(_json_match(g, w, float_tol)
                                             for g, w in zip(got, want))
    if isinstance(want, dict) and isinstance(got, dict):
        return got.keys() == want.keys() and all(_json_match(got[k], want[k], float_tol)
                                                 for k in want)
    return got == want and type(got) is type(want)


def _tol(params):
    t = params.get("float_tol")
    if t is None:
        return None
    t = float(t)
    if not (math.isfinite(t) and t >= 0):
        raise ValueError("bad float_tol")
    return t


# ---------- checkers (task, reply, allow_exec) -> {0,1} ----------
def exec_stdio(task, reply, allow=False):
    """Every test passes: exit 0, CPU within STDIO_CPU_SECS, stdout within
    OUTPUT_CAP, and stdout's tokens equal the expected output's."""
    if not allow:
        return 0
    params = task["checker"]["params"]
    tests = params["tests"]
    if not isinstance(tests, list) or not tests or len(tests) > MAX_TESTS:
        return 0
    tol = _tol(params)
    stdins = [t["input"] for t in tests]
    wants = [str(t["output"]).encode("utf-8") for t in tests]
    results = run_program(extract_code(reply), stdins, STDIO_CPU_SECS, stop_on_failure=True)
    if len(results) != len(tests):
        return 0
    spent = 0.0
    for r, want in zip(results, wants):
        spent += r.cpu_secs
        if not r.clean or r.cpu_secs > STDIO_CPU_SECS or spent > QUESTION_CPU_SECS:
            return 0
        if not _tokens_match(r.stdout, want, tol):
            return 0
    return 1


def build_func_program(code: str, func: str) -> str:
    """The exec_func program: prelude, the candidate's code, then a driver that
    reads the inputs (JSON) from stdin and prints the results after a marker."""
    if not isinstance(func, str) or not FUNC_NAME.match(func):
        raise ValueError("func must be a dotted name")
    driver = (
        "\n\nif __name__ == '__main__':\n"
        "    import json as _json, sys as _sys\n"
        "    _inputs = _json.loads(_sys.stdin.read())\n"
        f"    _fn = {func}\n"
        "    _res = [_fn(**_a) if isinstance(_a, dict) else _fn(*_a) for _a in _inputs]\n"
        f"    _sys.stdout.write('\\n{RESULT_MARKER} ' + _json.dumps(_res) + '\\n')\n"
        "    _sys.stdout.flush()\n")
    return FUNC_PRELUDE + code + driver


def parse_func_result(stdout: bytes):
    """The results after the LAST marker line; a ValueError if there is none."""
    text = stdout.decode("utf-8", "replace")
    i = text.rfind("\n" + RESULT_MARKER + " ")
    if i < 0:
        raise ValueError("no result")
    return json.loads(text[i + len(RESULT_MARKER) + 2:])


def exec_func(task, reply, allow=False):
    """The function's results on all inputs equal the gold (canonical JSON,
    as graders.exec_io; float tolerance only when the task sets one)."""
    if not allow:
        return 0
    params = task["checker"]["params"]
    inputs = params["inputs"]
    if not isinstance(inputs, list) or not inputs:
        return 0
    tol = _tol(params)
    program = build_func_program(extract_code(reply), params["func"])
    results = run_program(program, [json.dumps(inputs)], FUNC_CPU_SECS)
    if len(results) != 1:
        return 0
    r = results[0]
    if not r.clean or r.cpu_secs > FUNC_CPU_SECS:
        return 0
    got = parse_func_result(r.stdout)
    return int(_json_match(got, task["gold"], tol))


CHECKERS = {
    "exec_stdio": exec_stdio,
    "exec_func": exec_func,
}


def grade(task, reply, allow_exec=False):
    """Grade one reply. Returns 0 or 1; never raises (failure grades 0)."""
    try:
        return int(CHECKERS[task["checker"]["id"]](task, reply, allow_exec))
    except Exception:
        return 0


def sandbox_selftest():
    """(ok, reason). The harness runs this once at start and refuses to grade
    code if it fails, so a TD that cannot build the sandbox says so loudly
    instead of grading every code question 0."""
    if not (sys.platform.startswith("linux") and os.uname().machine == "x86_64"):
        return False, "the code sandbox needs x86_64 Linux"
    probe = ("import socket\n"
             "try:\n    socket.socket(); print('net-open')\n"
             "except OSError:\n    print('net-closed')\n"
             "import os\n"
             "try:\n    os.fork(); print('fork-open')\n"
             "except OSError:\n    print('fork-closed')\n"
             "print(input())\n")
    try:
        r = run_program(probe, ["ping\n"])[0]
    except Exception as e:                       # noqa: BLE001
        return False, f"could not start the sandbox: {type(e).__name__}"
    if r.exit_code == SANDBOX_UNAVAILABLE:
        return False, "seccomp is unavailable (prctl refused)"
    if r.stdout.split() != [b"net-closed", b"fork-closed", b"ping"]:
        return False, f"sandbox probe gave {r.stdout[:80]!r} (exit {r.exit_code})"
    return True, "ok"


def grader_hash():
    """sha256 of this file's bytes, line endings normalised to LF (as graders.py)."""
    with open(os.path.abspath(__file__), "rb") as f:
        return "sha256:" + hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


if __name__ == "__main__":
    print(grader_hash())
