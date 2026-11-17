# The v3 code grader (`fugal_subnet/graders_code.py`)

Status: built 2026-11-16 for protocol v3; not yet wired into the harness.

## Why a second file

`graders.py` is immutable and hash-pinned. Its `exec_io` grades function calls
only, and its sandbox has no network isolation, inherits the environment (so
string-set order varies between runs), and puts no `typing` import in front of a
LeetCode-style answer. Protocol v3 adds code questions, so it adds a second
pinned grader beside the first: `graders_code.py`, pinned in
`scripts/check_safety_invariants.py` (`CODE_GRADER_SHA256`). Any byte change is
a new grader version, exactly as for `graders.py`.

| Checker | Question | How it is graded |
|---|---|---|
| `exec_stdio` | a whole program (AtCoder, Codeforces, CodeContests) | one run per test, the test's input on stdin; every test must exit 0, stay within the CPU limit, and print the expected output token for token (float tolerance only if the task sets `float_tol`) |
| `exec_func` | a function or `Solution().method` (LeetCode, MBPP+, HumanEval+) | one run of all inputs (positional lists or keyword dicts, as JSON on stdin); the results after the driver's marker must equal the gold as canonical JSON, as in `exec_io` |

The program is taken from the reply the way `exec_io` takes it today:
`graders.extract_code`, the **largest** fenced block, else the whole reply.
(It is imported, not copied, so the two cannot drift; both files are pinned.)

`exec_func` runs the candidate after LeetCodeDataset's own prompt prelude
(`from typing import *`, `collections`, `heapq`, `bisect`, `math`, ...), so an
answer that uses `List[int]` without importing it is not failed for it, and
reference solutions run in the context the problems were written for.

## Limits and why

| Limit | Value | Why |
|---|---|---|
| CPU per test (`exec_stdio`) | 3.0 s measured (rusage) | CPU time, not wall clock, so a busy TD does not fail answers. Rlimit soft = 4 s (SIGXCPU), hard = 5 s (SIGKILL) as the backstop |
| CPU per run (`exec_func`) | 6.0 s | one process runs every input |
| CPU per question | 24 s | bounds a hostile question's cost to the miner |
| Wall-clock backstop | max(3 × CPU limit + 2 s, 20 s) per run; 60 s per question | only ends programs that block without CPU; never the deciding limit for a program that computes |
| `time.sleep` | refused | sleeping uses no CPU; without this the wall backstop would decide, and that differs between TDs |
| Address space | 1 GiB | memory bombs fail the same way everywhere |
| Stack | 64 MiB main thread; 8 MiB per new thread | deep recursion behaves the same on every TD; 64+ threads fit the address space (a program that calls `threading.stack_size()` gets what it asks for) |
| Malloc arenas | 2 (`MALLOC_ARENA_MAX`) | glibc reserves 64 MiB of address space per thread arena, which capped a program at ~20 threads |
| CPUs the program sees | 4 (`os.cpu_count`, `process_cpu_count`, `sched_getaffinity`) | default pool sizes (`ThreadPoolExecutor()`) do not follow the TD's vCPUs |
| Output | 1 MiB per run (`RLIMIT_FSIZE` on the stdout file) | huge output dies with SIGXFSZ |
| Open files | 64 | |
| Program | 512,000 bytes | as `graders.EXEC_PROGRAM_CAP` |
| Tests per question | 64 at most (the pool keeps ≤ 16) | |

## The sandbox

The child is `python -S -s -B -c RUNNER solution.py`, started in its own
session with `close_fds` (only stdin, stdout and stderr), an environment of
exactly `PYTHONHASHSEED=0`, `PYTHONIOENCODING=utf-8`,
`PYTHONDONTWRITEBYTECODE=1`, `LC_ALL=C.UTF-8`, and the rlimits above. `-S`
means stdlib only, so whatever else a TD image carries cannot change a grade.
The int-to-str digit limit is turned off, as in competitive judges.

Before it reads the candidate's program, the runner installs a **seccomp
filter** (x86_64; `PR_SET_NO_NEW_PRIVS`, then `PR_SET_SECCOMP`). It returns
EPERM for:

- sockets of every kind (`socket`, `socketpair`, `connect`, `bind`, ...) and
  `io_uring`;
- `fork`, `vfork`, `execve`, `execveat`, and `clone` without `CLONE_THREAD`
  (threads work; `clone3` gets ENOSYS so libc falls back to `clone`);
- `open`/`openat` with any of `O_WRONLY|O_RDWR|O_CREAT|O_TRUNC|O_APPEND`
  (`openat2` gets ENOSYS), and every call that creates, deletes, renames,
  links, chmods, chowns, truncates or re-times a path, `mknod`, `mount` and
  the new mount API, `chroot`, `pivot_root`, xattrs, file handles;
- `ptrace`, `process_vm_readv/writev`, `kcmp`, `migrate_pages`, `move_pages`;
- calls that name another process: `kill`, `tgkill`, `rt_sigqueueinfo`,
  `rt_tgsigqueueinfo`, `prlimit64`, the `sched_set*` calls, `setpriority`,
  `ioprio_set` are allowed only on itself (pid 0 or its own pid); `tkill` and
  the pidfd calls are refused;
- `unshare`, `setns`, `bpf`, `perf_event_open`, `userfaultfd`, keyrings,
  module loading, `kexec`, `reboot`, swap, `acct`, `iopl`, `ioperm`;
- the x32 ABI, and any other architecture (killed).

The grading process makes itself **non-dumpable** (`PR_SET_DUMPABLE 0`) once,
so a child running under the same uid cannot open `/proc/<parent>/mem` (which
holds every gold answer of the pool) or `/proc/<parent>/environ` (which can
hold the API key). If the grader runs as root, the child drops to uid/gid 65534.

Reads are allowed (the child must import the stdlib). Nothing it reads can
leave: it has no network, cannot write files, and its stdout goes only to the
parent's comparison; it is never logged or stored.

The parent alone decides the grade, from the exit status, the child's measured
CPU time (`wait4` rusage) and its stdout. Expected outputs never leave the
parent's memory.

`sandbox_selftest()` runs a probe (socket refused, fork refused, stdin echoed).
**The harness must run it at start and refuse to grade code questions if it
fails**, so a TD that cannot build the sandbox says so loudly instead of
grading every code answer 0.

## Determinism and shared grades

Two TDs give the same grade when the answer's behaviour does not depend on the
machine. The grader removes the usual sources:

- hash seed pinned, so `set`/`dict`-of-`str` order is identical;
- stdlib only (`-S`) and the image's one pinned Python build;
- no network, no clock dependence (sleep refused; the CPU limit, not the wall
  clock, decides), no other process to race with (no fork);
- comparisons are byte tokens, or exact canonical JSON, with tolerance only when
  the task sets it.

What remains is CPU speed: an answer whose CPU time is close to a limit can pass
on one TD and fail on another. Two things contain that:

1. **Calibration** (`scripts/calibrate_code_tests.py`) keeps only tests that a
   reference solution passes in ≤ 25% of the limit, on two runs, through this
   same sandbox. An efficient answer then has a 4× margin, which covers the
   spread between TD CPUs.
2. **Shared grades.** A cell's grade is the median across coldkeys, so a single
   TD that grades a borderline answer differently does not move it.

Known residuals, accepted: output that depends on object addresses (`id()`,
default `object.__hash__` ordering) varies with ASLR; ASLR cannot be turned off
in a container. Such answers are wrong by any judge's standard.

**Waits that use no CPU (accepted residual, v3 review 2026-11-16).** Only
`time.sleep` is refused. `select`, `Event.wait`, `Lock.acquire(timeout=…)` and
other blocking waits use no CPU, so for them the wall clock decides: a reply
that waits just under the backstop passes on a quiet TD and can fail on a busy
one. Only a reply that deliberately waits near the backstop is affected; a
program that computes is ended by its CPU limit long before. The review widened
the gap: the per-run backstop is max(3 × CPU + 2 s, 20 s), so 20 s for both
checkers (it was 11 s for `exec_stdio`), and a question's tests together end at
60 s of wall time.

Grading budget per epoch (the harness grades serially). A miner grades about 50
code questions an epoch (the code share, 0.15, of the 300-question slice, plus
its share of the exploration calls).
- Typical: a test costs ~0.05 s to start Python plus its CPU (median 0.05 s in
  the calibration), and a question keeps ≤ 16 tests: ~1.6 s a question, ~80 s an
  epoch, well inside the 2,160 s to the collection point.
- Worst case, every code reply deliberately waiting: each question ends by
  60 s plus one last run's 20 s backstop, 80 s; 50 questions are 4,000 s, past
  the collection point. Before the review it was worse (64 tests × 11 s, ~700 s
  a question). Replies come from the models through the attested proxy, not
  from the miner, so a whole slice of waiting replies is not something a miner
  can arrange; the cap bounds the damage of the few a model might write.

## What the miner TD image must contain

- x86_64 Linux with seccomp filters available to an unprivileged process
  (Docker's default profile allows `prctl(PR_SET_SECCOMP)` and
  `PR_SET_NO_NEW_PRIVS`), and `/proc`.
- The image's single pinned CPython (3.11 today) with its stdlib, including
  `ctypes` (the runner uses it to install the filter).
- Nothing else: no compilers, no extra site-packages are visible to the child.
- The harness must call `sandbox_selftest()` at start and route checker ids
  `exec_stdio` / `exec_func` to `graders_code.grade`, and the v3 grader
  identity in commitments must cover both files' hashes.

## Not covered

- BigCodeBench (third-party libraries, `unittest`, files): a later release.
- Float answers in CodeContests: calibration leaves them out until each task
  carries a stated tolerance.
- Multi-answer problems: calibration keeps a CodeContests problem only if two
  independent Python references pass, or every kept output is a single token.
