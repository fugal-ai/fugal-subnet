#!/usr/bin/env python3
# Adversarial suite for the v3 code graders (fugal_subnet/graders_code.py).
# Same shape as run_attacks.py: (name, task, reply, allow_exec, expected,
# verdict, note). BLOCKED attacks assert the grade a hostile program gets;
# CONTROL rows assert an honest program still passes. Hostile programs that
# would harm the machine if the sandbox failed (kill -1, deleting files) are
# written to fail safe: they probe, and print the expected token only when the
# probe was refused.
# Run: python -m fugal_subnet.attacks.run_code_attacks
import os
import sys
import time

from fugal_subnet import graders_code as gc


def _stdio(tests, **extra):
    return {"checker": {"id": "exec_stdio",
                        "params": {"tests": [{"input": i, "output": o} for i, o in tests],
                                   **extra}}}


def _func(func, inputs, gold):
    return {"gold": gold, "checker": {"id": "exec_func",
                                      "params": {"func": func, "inputs": inputs}}}


def _py(src):
    return "```python\n" + src.strip() + "\n```"


def _refused(probe):
    """Prints 'blocked' only if the probe raised."""
    body = "\n".join("    " + line for line in probe.strip().splitlines())
    return _py(f"try:\n{body}\n    print('ESCAPED')\nexcept BaseException:\n    print('blocked')")


ADD = _stdio([("1 2\n", "3\n"), ("5 5\n", "10\n"), ("-1 1\n", "0\n")])
BLOCKED = _stdio([("", "blocked\n")])

ATTACKS = [
    ("stdio: correct program (control)", ADD,
     _py("a, b = map(int, input().split())\nprint(a + b)"), True, 1, "CONTROL",
     "an honest answer passes"),
    ("stdio: exec off unless allowed", ADD,
     _py("a, b = map(int, input().split())\nprint(a + b)"), False, 0, "BLOCKED",
     "allow_exec=False never runs code"),
    ("stdio: wrong on one test of three", ADD,
     _py("a, b = map(int, input().split())\nprint(a + b if a > 0 else 99)"), True, 0,
     "BLOCKED", "every test must pass"),
    ("stdio: correct output then non-zero exit", ADD,
     _py("a, b = map(int, input().split())\nprint(a + b)\nraise SystemExit(2)"), True, 0,
     "BLOCKED", "a runtime error fails even after the right output"),
    ("stdio: infinite loop", BLOCKED, _py("while True: pass"), True, 0, "BLOCKED",
     "RLIMIT_CPU + measured CPU over the limit"),
    ("stdio: ignore SIGXCPU and SIGTERM, then loop", BLOCKED,
     _py("import signal\nsignal.signal(signal.SIGXCPU, signal.SIG_IGN)\n"
         "signal.signal(signal.SIGTERM, signal.SIG_IGN)\nwhile True: pass"), True, 0,
     "BLOCKED", "the hard limit is SIGKILL; the grade uses measured CPU"),
    ("stdio: sleep (no CPU used)", BLOCKED,
     _py("import time\ntime.sleep(30)\nprint('blocked')"), True, 0, "BLOCKED",
     "time.sleep is refused, so the result does not depend on the wall clock"),
    ("stdio: fork bomb", BLOCKED, _refused("import os\nwhile True:\n    os.fork()"), True, 1,
     "BLOCKED", "seccomp: fork/vfork/clone without CLONE_THREAD -> EPERM"),
    ("stdio: thread bomb", BLOCKED,
     _py("import threading\ndef s():\n    while True: pass\n"
         "while True:\n    threading.Thread(target=s, daemon=True).start()"), True, 0,
     "BLOCKED", "threads share the process CPU and address-space limits"),
    ("stdio: execve a shell", BLOCKED, _refused("import os\nos.execv('/bin/sh', ['sh'])"),
     True, 1, "BLOCKED", "seccomp: execve/execveat -> EPERM"),
    ("stdio: subprocess", BLOCKED, _refused("import subprocess\nsubprocess.run(['true'])"),
     True, 1, "BLOCKED", "fork and exec both refused"),
    ("stdio: TCP socket", BLOCKED,
     _refused("import socket\nsocket.create_connection(('1.1.1.1', 80), timeout=1)"), True, 1,
     "BLOCKED", "seccomp: socket -> EPERM"),
    ("stdio: raw socket syscall via ctypes", BLOCKED,
     _refused("import ctypes\nlibc = ctypes.CDLL(None, use_errno=True)\n"
              "assert libc.syscall(41, 2, 1, 0) >= 0"), True, 1, "BLOCKED",
     "the kernel filter, not Python, refuses it"),
    ("stdio: write a file", BLOCKED,
     _refused("open('/tmp/fugal-code-attack-' + str(__import__('os').getpid()), 'w')"),
     True, 1, "BLOCKED", "seccomp: open/openat with write flags -> EPERM"),
    ("stdio: delete a file", BLOCKED, _refused("import os\nos.remove('/tmp/does-not-matter')"),
     True, 1, "BLOCKED", "unlink/unlinkat refused (EPERM before ENOENT)"),
    ("stdio: huge output", BLOCKED,
     _py("import sys\nwhile True:\n    sys.stdout.write('x' * 65536)"), True, 0, "BLOCKED",
     "RLIMIT_FSIZE on the stdout file: SIGXFSZ"),
    ("stdio: memory bomb", BLOCKED, _py("x = bytearray(4 << 30)\nprint('blocked')"), True, 0,
     "BLOCKED", "RLIMIT_AS 1 GiB"),
    ("stdio: write to an inherited fd", BLOCKED,
     _refused("import os\nos.write(5, b'x')"), True, 1, "BLOCKED", "close_fds: only 0, 1, 2"),
    ("stdio: read the grader's memory", BLOCKED,
     _refused("import os\nopen(f'/proc/{os.getppid()}/mem', 'rb').read(1)"), True, 1,
     "BLOCKED", "the grading process is non-dumpable"),
    ("stdio: read the grader's environment", BLOCKED,
     _refused("import os\nopen(f'/proc/{os.getppid()}/environ', 'rb').read(1)"), True, 1,
     "BLOCKED", "non-dumpable: /proc/<ppid>/environ is root-owned"),
    ("stdio: signal the grader", BLOCKED,
     _refused("import os, signal\nos.kill(os.getppid(), signal.SIGCONT)"), True, 1,
     "BLOCKED", "seccomp: kill only to itself (SIGCONT so a failure harms nothing)"),
    ("stdio: signal every process (kill -1, harmless signal)", BLOCKED,
     _refused("import os, signal\nos.kill(-1, signal.SIGCONT)"), True, 1, "BLOCKED",
     "seccomp: kill(-1) -> EPERM"),
    ("stdio: lower the grader's limits", BLOCKED,
     _refused("import resource, os\nresource.prlimit(os.getppid(), resource.RLIMIT_NOFILE, (1, 1))"),
     True, 1, "BLOCKED", "seccomp: prlimit64 only on itself"),
    ("stdio: ptrace the grader", BLOCKED,
     _refused("import ctypes, os\nlibc = ctypes.CDLL(None, use_errno=True)\n"
              "assert libc.ptrace(16, os.getppid(), 0, 0) == 0"), True, 1, "BLOCKED",
     "seccomp: ptrace -> EPERM"),
    ("stdio: new namespaces", BLOCKED,
     _refused("import os\nos.unshare(os.CLONE_NEWUSER)"), True, 1, "BLOCKED",
     "seccomp: unshare/setns -> EPERM"),
    ("stdio: site-packages", _stdio([("", "no\n")]),
     _py("try:\n    import numpy\n    print('yes')\nexcept ImportError:\n    print('no')"),
     True, 1, "BLOCKED", "-S: stdlib only, whatever the image carries"),
    ("func: LeetCode answer without typing import (control)",
     _func("Solution().add", [[1, 2], [3, 4]], [3, 7]),
     _py("class Solution:\n    def add(self, a: int, b: int) -> int:\n        return a + b"),
     True, 1, "CONTROL", "the prelude supplies typing"),
    ("func: sys.exit(0) before the driver", _func("f", [[1, 2]], [3]),
     _py("def f(a, b):\n    return a + b\nimport sys\nsys.exit(0)"), True, 0, "BLOCKED",
     "no result marker -> 0"),
    ("func: injected func name", _func("__import__('os').getpid() or f", [[1]], [1]),
     _py("def f(x):\n    return x"), True, 0, "BLOCKED", "func must be a dotted name"),
    ("func: print the right numbers instead of returning them",
     _func("f", [[1, 2]], [3]), _py("def f(a, b):\n    print(a + b)\n    return a * b"),
     True, 0, "BLOCKED", "only the driver's marker line counts"),
]


def main():
    if not (sys.platform.startswith("linux") and os.uname().machine == "x86_64"):
        print("SKIP — the code sandbox is x86_64 Linux only")
        return
    ok, why = gc.sandbox_selftest()
    if not ok:
        print(f"FAIL — sandbox self-test: {why}")
        raise SystemExit(1)
    rows, surprises = [], []
    for name, task, reply, allow, expected, verdict, note in ATTACKS:
        t0 = time.time()
        got = gc.grade(task, reply, allow)
        rows.append((got == expected, verdict, name, got, expected, time.time() - t0, note))
        if got != expected:
            surprises.append(name)
    print(f"\n{'CODE ATTACK SUITE':52s} got/exp  verdict   secs")
    print("-" * 84)
    for ok, verdict, name, got, exp, dt, note in rows:
        print(f"{'OK ' if ok else '!! '}{name:52.52s} {got}/{exp}     {verdict:7s}  {dt:4.1f}")
        if not ok:
            print(f"    SURPRISE: expected {exp}, got {got} — {note}")
    print("-" * 84)
    blocked = sum(1 for r in rows if r[0] and r[1] == "BLOCKED")
    controls = sum(1 for r in rows if r[0] and r[1] == "CONTROL")
    print(f"{blocked} blocked, {controls} controls, {len(surprises)} surprises")
    if surprises:
        print("FAIL — reality diverged from documented behavior:", surprises)
        raise SystemExit(1)
    print("PASS — all code attacks behave as documented.")


if __name__ == "__main__":
    main()
