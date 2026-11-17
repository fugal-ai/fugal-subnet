"""graders_code after the v3 review (2026-11-16), finding 5.

Every probe here is harmless to the host by construction: a priority call
that sets the process's own priority to the value it already has, a chmod or
chown of the sandbox's own scratch directory to what it already is. Never
test the denial of PRIO_USER / IOPRIO_WHO_USER with a real value: one review
probe reniced every process of the user that way.

5a. setpriority/ioprio_set/sched_* and fchmod/fchown were allowed.
5b. Every thread reserved RLIMIT_STACK (64 MiB) under the 1 GiB address
    space, so only 8 threads started, and ThreadPoolExecutor()'s default size
    follows os.cpu_count(): a correct reply graded 1 or 0 by the TD's vCPUs.
5c. The wall backstop is max(3 x CPU + 2 s, 20 s) per run and 60 s per question.
"""
import os
import textwrap

import pytest

from fugal_subnet import graders_code as gc

pytestmark = pytest.mark.skipif(not gc.sandbox_selftest()[0], reason="sandbox unavailable here")


def _run(code: str) -> str:
    r = gc.run_program(textwrap.dedent(code), [""])[0]
    assert r.exit_code == 0, r
    return r.stdout.decode().strip()


@pytest.mark.parametrize("call", [
    "os.setpriority(os.PRIO_PROCESS, 0, os.getpriority(os.PRIO_PROCESS, 0))",
    "os.sched_setaffinity(0, {0})",
    "os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))",
    "os.sched_setparam(0, os.sched_param(0))",
    "os.fchmod(os.open('.', os.O_RDONLY), os.stat('.').st_mode & 0o7777)",
    "os.fchown(os.open('.', os.O_RDONLY), -1, -1)",
])
def test_priority_scheduling_and_fd_mode_calls_are_refused(call):
    out = _run(f"""
        import os, errno
        try:
            {call}
            print("ALLOWED")
        except OSError as e:
            print("refused", e.errno == errno.EPERM)
    """)
    assert out == "refused True"


def test_64_threads_start_and_the_cpu_count_is_fixed():
    out = _run("""
        import os, threading
        ev = threading.Event(); ts = []
        for i in range(64):
            t = threading.Thread(target=ev.wait); t.start(); ts.append(t)
        ev.set()
        for t in ts: t.join()
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor() as ex:
            n = ex._max_workers
        print(len(ts), os.cpu_count(), n)
    """)
    assert out == f"64 {gc.CHILD_CPU_COUNT} {min(32, gc.CHILD_CPU_COUNT + 4)}"


def test_the_main_thread_keeps_its_deep_recursion():
    # The calibration ran with STACK_BYTES for the main thread; it must not shrink.
    out = _run("""
        import sys
        sys.setrecursionlimit(200000)
        def f(n): return 0 if n == 0 else 1 + f(n - 1)
        print(f(100000))
    """)
    assert out == "100000"


def test_a_thread_that_asks_for_a_bigger_stack_gets_it():
    out = _run("""
        import sys, threading
        sys.setrecursionlimit(300000)
        threading.stack_size(256 << 20)
        res = []
        def f(n): return 0 if n == 0 else 1 + f(n - 1)
        t = threading.Thread(target=lambda: res.append(f(150000))); t.start(); t.join()
        print(res[0])
    """)
    assert out == "150000"


@pytest.mark.parametrize("host_cpus", [2, 64])
def test_a_default_thread_pool_reply_grades_the_same_whatever_the_host(monkeypatch, host_cpus):
    monkeypatch.setattr(os, "cpu_count", lambda: host_cpus)
    task = {"checker": {"id": "exec_stdio", "params": {"tests": [
        {"input": "40\n", "output": str(sum(i * i for i in range(40)))}]}}}
    for workers in ("None", "16", "32"):
        reply = "```python\n" + textwrap.dedent(f"""
            from concurrent.futures import ThreadPoolExecutor
            n = int(input())
            with ThreadPoolExecutor(max_workers={workers}) as ex:
                print(sum(ex.map(lambda i: i * i, range(n))))
        """) + "```"
        assert gc.grade(task, reply, allow_exec=True) == 1, workers


def test_the_wall_backstop_and_question_cap_constants():
    assert max(gc.STDIO_CPU_SECS * gc.WALL_FACTOR + gc.WALL_EXTRA_SECS, gc.WALL_MIN_SECS) == 20.0
    assert max(gc.FUNC_CPU_SECS * gc.WALL_FACTOR + gc.WALL_EXTRA_SECS, gc.WALL_MIN_SECS) == 20.0
    assert gc.QUESTION_WALL_SECS == 60.0


def test_a_question_past_its_wall_cap_grades_zero(monkeypatch):
    task = {"checker": {"id": "exec_stdio", "params": {"tests": [
        {"input": "3\n", "output": "6"}, {"input": "4\n", "output": "8"}]}}}
    real = gc._run_once

    def slow(*a, **k):
        r = real(*a, **k)
        r.wall_secs = 31.0                              # two of these pass the 60 s cap
        return r

    reply = "```python\nprint(int(input()) * 2)\n```"
    assert gc.grade(task, reply, allow_exec=True) == 1
    monkeypatch.setattr(gc, "_run_once", slow)
    assert gc.grade(task, reply, allow_exec=True) == 0


@pytest.mark.slow
def test_a_wait_under_the_wider_backstop_still_answers():
    task = {"checker": {"id": "exec_stdio", "params": {"tests": [{"input": "3\n", "output": "6"}]}}}
    reply = "```python\nimport select\nselect.select([], [], [], 12)\nprint(int(input()) * 2)\n```"
    assert gc.grade(task, reply, allow_exec=True) == 1      # 11 s backstop before the review
