# Putting the code benchmarks back

HumanEval and LiveCodeBench are currently **excluded from the pool**. This
describes why, and what has to be true before they return.

## What happened

The harness grades with `allow_exec=False`. The checkers for those benchmarks —
`exec_io` and `exec_unittest` — both begin `if not allow: return 0`. So every
code question scored zero for every miner, regardless of the answer. Confirmed
live: a correct HumanEval solution from a real model graded 0, and two very
different models both scored exactly 0/8.

The size of it was hidden by an averaging artefact. On the full 21,717-question
pool the code benchmarks are 164 questions, 0.8% — which reads like a rounding
error. But `select_slice` **stratifies by benchmark**, giving each an equal share
of every slice. Measured over five real slices: **50 of every 300 questions, with
no variance.** A full sixth of everything actually graded.

And it was not merely dead weight. Miners paid real API cost on those 50
questions for a structurally impossible score, so routing them to a capable model
was punished on thrift with no reachable quality. **The scoring was training heads
to route code questions to the cheapest model available** — the opposite of the
policy this subnet exists to discover, learned from a grading artefact rather
than from anything true about the models.

## Why `allow_exec=False` is right, for now

It is a deliberate security choice, not an oversight.

`graders.py`'s sandbox is process-level: rlimits, a new process group, DEVNULL
pipes, a program-size cap and a kill-tree timeout. It has **no filesystem
isolation, no namespace isolation and no network isolation.**

Under `--live` the miner controls where the metering proxy points, and therefore
controls what "model output" the harness receives. Executing that inside the TD
which produces the attested proof lets a miner run chosen code **inside its own
enclave**, where it could rewrite results to all-correct, read the nonce, or
tamper with the metering records — and leave with a genuine Intel signature over
forged content. That is a total break of I8, not a contained risk.

So the requirement is not "limit the blast radius". It is: **the executed code
must not be able to observe or touch anything the proof depends on.**

## The constraint that shapes the fix

`graders.py` is hash-pinned and immutable, and `_run_sandboxed` lives inside it.
Hardening it in place changes the grader hash and breaks consensus for everyone.

So the sandbox goes in `harness.py`, wrapping the whole `grade()` call for
execution checkers. `graders.py` stays byte-identical. That also nests the
protection usefully: `exec_io` already keeps the gold answer out of the
candidate's process and compares parent-side, so a forged pass requires actually
computing the right output. The new layer keeps the *grader* away from the proof.
Only one bit crosses back.

## Phases

### 1. Threat model, written down — half a day

Enumerate what executed code must not reach: the proof object, gold answers, the
epoch nonce, the metering proxy, the head, the filesystem, the network, the
parent process. Record it in `INVARIANTS.md` under I8. Without this written down,
the next person simplifies the sandbox and nobody notices — which is exactly how
the current defect survived.

### 2. Build the sandbox — ~2 days

`bubblewrap`, with two corrections to an earlier draft of this plan that were
found by trying it on an actual TDX guest rather than on a laptop:

- **It is not installed** on a stock Ubuntu 24.04 TDX image; it has to be added
  to whatever image is built.
- **It does not run unprivileged there.** 24.04 sets
  `kernel.apparmor_restrict_unprivileged_userns=1`, and unprivileged bwrap fails
  with `loopback: Failed RTM_NEWADDR: Operation not permitted`. It works as
  root, which the miner already is — it needs root for configfs quote
  generation — so this is workable, but "no root required" was wrong.

The bind list below is also fuller than the earlier draft, which omitted `/bin`,
`/etc` and the virtualenv, so `python3` did not resolve at all:

```
bwrap --unshare-all --die-with-parent \
      --ro-bind /usr /usr --ro-bind /bin /bin --ro-bind /etc /etc \
      --ro-bind /lib /lib --ro-bind /lib64 /lib64 \
      --ro-bind <venv> <venv> \
      --tmpfs /tmp --proc /proc --dev /dev \
      --chdir /tmp \
      <venv>/bin/python3 program.py
```

`--unshare-all` gives network, PID, IPC, UTS and mount namespaces in one flag.
**The network namespace is the highest-value line**: it is what stops executed
code reaching the metering proxy and injecting API records. Nothing from the repo
is bound in, so the proof is simply not present on the child's filesystem.

If `bwrap` is unavailable, **refuse to grade those questions loudly**. Returning
0 silently is the defect being fixed; reproducing it as a fallback would be
worse than not shipping.

### 3. Prove it with attacks — ~2 days

This is the phase that decides whether the fix is real. Everything else is
plumbing. Add to `fugal_subnet/attacks/run_tee_attacks.py`, candidate code that:

| attack | must not |
|---|---|
| read `results/`, the head, or the proof | observe anything the proof depends on |
| write anywhere outside its own tmpfs | alter grading inputs |
| open a socket to the metering proxy | inject API records or costs |
| `ptrace` / `/proc/<parent>/mem` | reach the harness process |
| fork-bomb, allocate unboundedly, spin | stop the epoch (I6) |
| `sys.exit(0)` at import, before tests | forge a pass |

The last one is why `exec_unittest` cannot simply be re-enabled — see phase 4.

### 4. `exec_unittest` is forgeable and sandboxing does not fix it — ~1 hour

Its verdict is the child's exit code, and `graders.py` documents it: *"KNOWN
FORGEABLE by candidate code that calls `sys.exit(0)`/`os._exit(0)` at import
time, before the tests run."* Isolation does not help — the child is *supposed*
to control its exit code.

Those 9 of 164 code questions must be dropped or re-tagged to `exec_io`, whose
parent-side output comparison is immune: a forged pass there requires computing
the correct outputs, i.e. solving the task.

### 5. Re-enable and measure — half a day

Set `FUGAL_HARNESS_ALLOW_EXEC=1`, rebuild `data/pool_manifest.json`, and confirm
`tests/test_grader_policy.py` goes green on its own. The pool exclusion is
derived from the flag, so the benchmarks return automatically — there is no
second place to edit. Then run a live epoch and confirm HumanEval scores
non-zero for a competent model, which is the observation that started this.

### 6. Roll out — NOT gated, for a reason that is itself the problem

An earlier revision of this plan said phase 6 was blocked on measurement
rotation, because changing `harness.py` would change the TD image and force a
coordinated miner upgrade.

**That premise is wrong, and it was tested rather than reasoned about.** A line
was appended to `harness.py` on a live TD and the measurement was identical
before and after. `measurement_id` covers MRTD and RTMR0-2 — firmware,
bootloader, kernel, initrd — and the repo is git-cloned onto an unmeasured
filesystem at runtime. Changing the harness changes nothing the attestation sees.

So this phase needs no coordination and is not blocked. It is also a warning
about the thing this plan is protecting: if changing the harness does not change
the measurement, then the approved-measurement check never bound the harness,
and a miner can edit grading code inside a genuinely approved TD. See I8 in
`INVARIANTS.md`.

That does not stop phases 1-5, which are worth doing on their own merits — a
sandbox is right whether or not the measurement covers it. But **the sandbox
protects an honest miner's enclave from model output; it does not protect the
subnet from a dishonest miner**, and only a measured code image does that.

## Sequencing

The exclusion shipped first, on purpose. It is one derived flag, fully reversible,
and it stops the scoring actively mis-training heads on every epoch. Building the
sandbox first would have meant running a known mis-training defect for another
week or two to avoid one reversible pool change — the wrong trade.

## Cost of the exclusion

The subnet currently measures routing on maths, reasoning, knowledge and
instruction-following, and not on code. That is a real loss of signal: code is a
domain where model prices and capabilities diverge sharply, so it is exactly
where a router should be able to prove itself. It is a loss worth taking briefly
and not worth taking permanently.
