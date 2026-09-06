"""CPU kernel dispatch pinning. Must be imported before numpy and torch.

Consensus rests on every honest validator computing the same scores from the
same inputs. Both numeric libraries in the scoring path select CPU kernels from
the host's widest SIMD extension, so an AVX-512 host and an AVX2 host reduce in
different orders and produce different float results:

  * torch  — the backbone embedding (ATen, MKL, oneDNN)
  * numpy  — the scoring arithmetic in head_eval (W @ h + b, via OpenBLAS)

Both read these variables once, at import, so setting them afterwards is a
silent no-op. That is why this module carries no heavy imports and must be the
first thing an entry point imports. `scripts/check_safety_invariants.py`
enforces the ordering.

Thread counts are pinned too: multi-threaded reductions vary in order run to
run, which is nondeterminism on a single machine, not just across machines.

THE BACKBONE IS THE EXCEPTION, AND ONLY FOR THE MINER. Under the TEE design a
validator never runs the backbone: it scores routing decisions read out of an
attested proof, and `scripts/check_determinism.py --perturb` asserts that the
whole scoring path is identical with thread pins removed. Embeddings shape only
the miner's own routing choices — which are exactly what it is scored on, and
which it already controls through its head — so they are not consensus state.
Pinning them to one thread bought the miner nothing and cost it ~13 hours of
startup on a 4-vCPU TD (measured, 21,553 questions at 0.46 q/s). So the torch
thread count is settable with FUGAL_BACKBONE_THREADS (default 1, unchanged; 0
means every core). numpy's OpenBLAS stays at one thread regardless: the miner's
routing arithmetic (W @ h + b) runs there, and there is no reason to spend
determinism where nothing is gained.
"""
from __future__ import annotations

import os


def backbone_threads() -> int:
    """torch intra-op threads for the backbone. 1 unless FUGAL_BACKBONE_THREADS
    says otherwise; 0 means all cores; anything unparsable means 1."""
    raw = os.getenv("FUGAL_BACKBONE_THREADS", "1").strip()
    try:
        n = int(raw)
    except ValueError:
        return 1
    if n <= 0:
        return max(1, os.cpu_count() or 1)
    return n


def determinism_env(threads: int = 1) -> dict[str, str]:
    return {
        # torch: ATen kernels, MKL BLAS, and oneDNN each dispatch separately.
        "ATEN_CPU_CAPABILITY": "avx2",
        "MKL_CBWR": "AVX2",
        "DNNL_MAX_CPU_ISA": "AVX2",
        "MKL_NUM_THREADS": str(threads),
        "OMP_NUM_THREADS": str(threads),
        # numpy: PyPI wheels bundle OpenBLAS, which dispatches by detected CPU
        # just as ATen does. OPENBLAS_NUM_THREADS is set explicitly rather than
        # relying on OpenBLAS's fallback to OMP_NUM_THREADS, which only applies
        # to OpenMP builds. Haswell is the AVX2-era kernel family, matching the
        # torch pin. Always 1 — see the module docstring.
        "OPENBLAS_CORETYPE": "Haswell",
        "OPENBLAS_NUM_THREADS": "1",
    }


DETERMINISM_ENV = determinism_env(backbone_threads())


def pin_cpu_dispatch() -> None:
    """Pin kernel dispatch and thread counts. Idempotent; setdefault semantics.

    An operator who has deliberately exported one of these keeps their value —
    the environment fingerprint in the reveal records what was actually in
    effect, so a divergence caused by an override stays diagnosable.
    """
    for key, value in DETERMINISM_ENV.items():
        os.environ.setdefault(key, value)


pin_cpu_dispatch()
