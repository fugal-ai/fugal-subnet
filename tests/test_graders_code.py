"""The v3 code graders (fugal_subnet/graders_code.py): exec_stdio and exec_func.

Each test is a behaviour the grade depends on. The hostile ones prove the
sandbox by running real programs; nothing here is mocked.
"""
import json
import os
import subprocess
import sys
import textwrap
import time
import uuid

import pytest

pytestmark = pytest.mark.skipif(
    not (sys.platform.startswith("linux") and os.uname().machine == "x86_64"),
    reason="the v3 code sandbox is x86_64 Linux only (the miner TD)")

from fugal_subnet import graders_code as gc  # noqa: E402


def _stdio(tests, **extra):
    params = {"tests": [{"input": i, "output": o} for i, o in tests], **extra}
    return {"checker": {"id": "exec_stdio", "params": params}}


def _func(func, inputs, gold, **extra):
    return {"gold": gold, "checker": {"id": "exec_func",
                                      "params": {"func": func, "inputs": inputs, **extra}}}


def _py(src):
    return "```python\n" + textwrap.dedent(src).strip() + "\n```"


ADD = _stdio([("1 2\n", "3\n"), ("10 -4\n", "6\n"), ("0 0\n", "0\n")])
ADD_OK = _py("""
    a, b = map(int, input().split())
    print(a + b)
""")


# ---------- exec_stdio: correctness ----------
def test_a_correct_program_passes():
    assert gc.grade(ADD, ADD_OK, allow_exec=True) == 1


def test_execution_is_off_unless_allowed():
    assert gc.grade(ADD, ADD_OK, allow_exec=False) == 0


def test_a_wrong_program_fails():
    assert gc.grade(ADD, _py("a, b = map(int, input().split())\nprint(a * b)"), True) == 0


def test_every_test_must_pass():
    # right on the first two, wrong on 0 0
    assert gc.grade(ADD, _py("""
        a, b = map(int, input().split())
        print(a + b if a else 1)
    """), True) == 0


def test_whitespace_is_normalised_to_tokens():
    task = _stdio([("3\n", "1 2 3\n")])
    assert gc.grade(task, _py("n=int(input())\nprint('  1\\n2\\t3  ')\nprint()"), True) == 1


def test_tokens_must_match_exactly():
    task = _stdio([("3\n", "1 2 3\n")])
    assert gc.grade(task, _py("print('1 2 3 4')"), True) == 0
    assert gc.grade(task, _py("print('1 2')"), True) == 0


def test_floats_compare_exactly_unless_the_task_sets_a_tolerance():
    exact = _stdio([("", "0.3333333333\n")])
    tol = _stdio([("", "0.3333333333\n")], float_tol=1e-6)
    reply = _py("print(1/3)")                      # 0.3333333333333333
    assert gc.grade(exact, reply, True) == 0
    assert gc.grade(tol, reply, True) == 1
    assert gc.grade(tol, _py("print(0.34)"), True) == 0
    # a tolerance never lets a non-number through
    assert gc.grade(_stdio([("", "abc\n")], float_tol=1e-6), _py("print('abd')"), True) == 0


def test_a_crash_after_correct_output_fails():
    assert gc.grade(ADD, _py("""
        a, b = map(int, input().split())
        print(a + b)
        raise SystemExit(3)
    """), True) == 0


def test_the_program_is_the_largest_fenced_block_as_in_exec_io():
    reply = ("Here is a helper:\n```python\nprint(0)\n```\nand the answer:\n"
             + ADD_OK)
    assert gc.grade(ADD, reply, True) == 1
    # no fence: the whole reply is the program
    assert gc.grade(ADD, "a, b = map(int, input().split())\nprint(a + b)", True) == 1


def test_an_oversized_program_fails():
    reply = ("```python\n" + "# pad\n" * 100_000
             + "a, b = map(int, input().split())\nprint(a + b)\n```")
    assert len(reply) > gc.PROGRAM_CAP
    assert gc.grade(ADD, reply, True) == 0


def test_a_malformed_task_grades_zero_and_never_raises():
    for task in ({}, {"checker": {"id": "exec_stdio"}}, {"checker": {"id": "exec_stdio",
                 "params": {"tests": "nope"}}}, {"checker": {"id": "nope"}}, None):
        assert gc.grade(task, ADD_OK, True) == 0
    assert gc.grade(_stdio([]), ADD_OK, True) == 0          # no tests never passes


def test_big_integers_print_in_full():
    old = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(0)
    try:
        task = _stdio([("", str(7 ** 9000) + "\n")])
    finally:
        sys.set_int_max_str_digits(old)
    assert gc.grade(task, _py("print(7 ** 9000)"), True) == 1


def test_set_order_is_the_same_every_run():
    # string hashing is seeded per process unless PYTHONHASHSEED is pinned
    src = _py("print(' '.join(set('abcdefghijklmnopqrstuvwxyz')))")
    first = gc.run_program(gc.extract_code(src), [""])[0].stdout
    for _ in range(3):
        assert gc.run_program(gc.extract_code(src), [""])[0].stdout == first


def test_the_environment_is_only_what_the_grader_sets():
    want = " ".join(sorted(gc.CHILD_ENV))
    assert gc.grade(_stdio([("", want + "\n")]),
                    _py("import os\nprint(' '.join(sorted(os.environ)))"), True) == 1


def test_threads_still_work():
    assert gc.grade(_stdio([("", "45\n")]), _py("""
        import threading
        out = []
        ts = [threading.Thread(target=out.append, args=(i,)) for i in range(10)]
        [t.start() for t in ts]; [t.join() for t in ts]
        print(sum(out))
    """), True) == 1


def test_stdlib_only_no_site_packages():
    # numpy is installed in this venv; the sandbox must not see it, so every
    # miner TD grades the same whatever else its image carries
    assert gc.grade(_stdio([("", "no\n")]), _py("""
        try:
            import numpy
            print('yes')
        except ImportError:
            print('no')
    """), True) == 1


def test_cpu_time_is_measured_per_test():
    results = gc.run_program("s=0\nfor i in range(3_000_000): s+=i\nprint(s)", ["", ""])
    assert all(r.exit_code == 0 for r in results)
    assert all(0.05 < r.cpu_secs < gc.STDIO_CPU_SECS for r in results)


# ---------- exec_stdio: hostile programs ----------
def _blocked_or_expected(probe):
    """A program that prints the expected token only if `probe` fails."""
    return _py(f"""
        try:
{textwrap.indent(textwrap.dedent(probe).strip(), ' ' * 12)}
            print('ESCAPED')
        except BaseException:
            print('blocked')
    """)


BLOCKED = _stdio([("", "blocked\n")])


def test_an_infinite_loop_fails_quickly():
    t0 = time.monotonic()
    assert gc.grade(BLOCKED, _py("while True: pass"), True) == 0
    assert time.monotonic() - t0 < gc.STDIO_CPU_SECS * gc.WALL_FACTOR + gc.WALL_EXTRA_SECS + 2


def test_ignoring_the_cpu_signal_does_not_help():
    t0 = time.monotonic()
    assert gc.grade(BLOCKED, _py("""
        import signal
        signal.signal(signal.SIGXCPU, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        while True: pass
    """), True) == 0
    assert time.monotonic() - t0 < 20


def test_sleeping_fails_deterministically():
    t0 = time.monotonic()
    assert gc.grade(BLOCKED, _py("import time\ntime.sleep(0.01)\nprint('blocked')"), True) == 0
    assert time.monotonic() - t0 < 5


def test_a_fork_is_refused():
    assert gc.grade(BLOCKED, _blocked_or_expected("""
        import os
        pid = os.fork()
        if pid == 0:
            os._exit(0)
    """), True) == 1


def test_exec_and_shell_are_refused():
    assert gc.grade(BLOCKED, _blocked_or_expected("""
        import os
        os.execv('/bin/echo', ['echo', 'hi'])
    """), True) == 1
    assert gc.grade(BLOCKED, _blocked_or_expected("""
        import subprocess
        subprocess.run(['true'], check=True)
    """), True) == 1


def test_network_is_refused_even_through_ctypes():
    assert gc.grade(BLOCKED, _blocked_or_expected("""
        import socket
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    """), True) == 1
    # a raw syscall, not the socket module: the kernel filter, not Python, refuses
    assert gc.grade(BLOCKED, _blocked_or_expected("""
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.syscall(41, 2, 1, 0) < 0:      # socket(AF_INET, SOCK_STREAM, 0)
            raise OSError(ctypes.get_errno(), 'refused')
    """), True) == 1


def test_writing_files_is_refused(tmp_path):
    target = tmp_path / f"pwn-{uuid.uuid4().hex}"
    assert gc.grade(BLOCKED, _blocked_or_expected(f"""
        open({str(target)!r}, 'w').write('x')
    """), True) == 1
    assert not target.exists()
    # deleting, renaming, chmod: the file survives untouched
    victim = tmp_path / "victim"
    victim.write_text("keep")
    for probe in (f"import os; os.remove({str(victim)!r})",
                  f"import os; os.rename({str(victim)!r}, {str(victim)!r} + '2')",
                  f"import os; os.chmod({str(victim)!r}, 0o777)",
                  f"import os; os.truncate({str(victim)!r}, 0)"):
        assert gc.grade(BLOCKED, _blocked_or_expected(probe), True) == 1
    assert victim.read_text() == "keep"


def test_huge_output_fails():
    assert gc.grade(BLOCKED, _py("""
        import sys
        chunk = 'x' * 65536
        while True:
            sys.stdout.write(chunk)
    """), True) == 0


def test_a_memory_bomb_fails():
    assert gc.grade(BLOCKED, _py("x = bytearray(4 << 30)\nprint('blocked')"), True) == 0


def test_a_thread_bomb_fails_without_harm():
    assert gc.grade(BLOCKED, _py("""
        import threading, time
        def spin():
            while True: pass
        while True:
            threading.Thread(target=spin, daemon=True).start()
    """), True) == 0


def test_other_file_descriptors_are_closed():
    assert gc.grade(BLOCKED, _blocked_or_expected("""
        import os
        for fd in range(3, 64):
            try:
                os.write(fd, b'x')
                break
            except OSError:
                continue
        else:
            raise OSError('none open')
    """), True) == 1


def test_the_parents_memory_and_environment_are_unreadable():
    probe = """
        import os
        ok = 0
        for name in ('mem', 'environ'):
            try:
                with open(f'/proc/{os.getppid()}/{name}', 'rb') as f:
                    f.read(1)
                ok += 1
            except OSError:
                pass
        if ok:
            raise SystemExit
        print('blocked')
    """
    assert gc.grade(BLOCKED, _py(probe), True) == 1


def test_ptrace_is_refused():
    canary = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        assert gc.grade(BLOCKED, _blocked_or_expected(f"""
            import ctypes
            libc = ctypes.CDLL(None, use_errno=True)
            if libc.ptrace(16, {canary.pid}, 0, 0) != 0:   # PTRACE_ATTACH
                raise OSError(ctypes.get_errno(), 'refused')
        """), True) == 1
        assert canary.poll() is None
    finally:
        canary.kill(); canary.wait()


def test_signalling_other_processes_is_refused():
    canary = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        assert gc.grade(BLOCKED, _blocked_or_expected(f"""
            import os, signal
            os.kill({canary.pid}, signal.SIGKILL)
        """), True) == 1
        assert gc.grade(BLOCKED, _blocked_or_expected(f"""
            import os, signal
            os.killpg(os.getpgid({canary.pid}), signal.SIGKILL)
        """), True) == 1
        time.sleep(0.2)
        assert canary.poll() is None
    finally:
        canary.kill(); canary.wait()


def test_a_program_cannot_kill_the_grader():
    """The grader runs in its own process here, so a successful kill shows up as
    that process dying instead of taking pytest with it."""
    task = json.dumps(BLOCKED)
    reply = json.dumps(_blocked_or_expected("""
        import os, signal
        os.kill(os.getppid(), signal.SIGKILL)
    """))
    out = subprocess.run(
        [sys.executable, "-c",
         "import json,sys; from fugal_subnet import graders_code as gc; "
         f"print('GRADE', gc.grade(json.loads({task!r}), json.loads({reply!r}), True))"],
        capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert "GRADE 1" in out.stdout


def test_no_process_outlives_a_test():
    marker = uuid.uuid4().hex
    gc.grade(BLOCKED, _py(f"""
        import threading
        # {marker}
        def spin():
            while True: pass
        threading.Thread(target=spin).start()
        while True: pass
    """), True)
    time.sleep(0.2)
    ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    assert marker not in ps


# ---------- exec_func ----------
TWO_SUM = _func("Solution().twoSum", [[[2, 7, 11, 15], 9], [[3, 2, 4], 6]], [[0, 1], [1, 2]])
TWO_SUM_OK = _py("""
    class Solution:
        def twoSum(self, nums: List[int], target: int) -> List[int]:
            seen = {}
            for i, x in enumerate(nums):
                if target - x in seen:
                    return [seen[target - x], i]
                seen[x] = i
""")


def test_exec_func_passes_a_leetcode_style_answer_without_its_imports():
    # the reply uses List and no import: the prelude supplies typing
    assert gc.grade(TWO_SUM, TWO_SUM_OK, True) == 1


def test_exec_func_fails_a_wrong_answer():
    assert gc.grade(TWO_SUM, _py("""
        class Solution:
            def twoSum(self, nums, target):
                return [0, 1]
    """), True) == 0


def test_exec_func_accepts_keyword_inputs_and_tuples_as_lists():
    task = _func("f", [{"a": 1, "b": 2}], [[1, 2]])
    assert gc.grade(task, _py("def f(a, b):\n    return (a, b)"), True) == 1


def test_exec_func_ignores_the_programs_own_prints():
    task = _func("f", [[1]], [2])
    assert gc.grade(task, _py("def f(x):\n    print('noise', 99)\n    return x + 1"), True) == 1


def test_exec_func_float_tolerance_only_when_set():
    exact = _func("f", [[]], [0.3333333333])
    tol = _func("f", [[]], [0.3333333333], float_tol=1e-6)
    reply = _py("def f():\n    return 1/3")
    assert gc.grade(exact, reply, True) == 0
    assert gc.grade(tol, reply, True) == 1


def test_exec_func_refuses_a_func_that_is_not_a_name():
    task = _func("__import__('os').system('true') or f", [[1]], [2])
    assert gc.grade(task, _py("def f(x):\n    return x + 1"), True) == 0


def test_exec_func_fails_a_program_that_exits_before_the_driver():
    task = _func("f", [[1]], [2])
    for exit_call in ("import sys; sys.exit(0)", "import os; os._exit(0)"):
        assert gc.grade(task, _py(f"def f(x):\n    return x * 2\n{exit_call}"), True) == 0


def test_the_sandbox_selftest_passes_here():
    ok, why = gc.sandbox_selftest()
    assert ok, why


def test_checker_ids_do_not_collide_with_graders_py():
    from fugal_subnet import graders
    assert not set(gc.CHECKERS) & set(graders.CHECKERS)


def test_the_grader_hash_is_of_this_files_bytes():
    import hashlib
    path = os.path.abspath(gc.__file__)
    with open(path, "rb") as f:
        want = "sha256:" + hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()
    assert gc.grader_hash() == want
