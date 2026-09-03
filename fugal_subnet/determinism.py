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
"""
from __future__ import annotations

import os

DETERMINISM_ENV = {
    # torch: ATen kernels, MKL BLAS, and oneDNN each dispatch separately.
    "ATEN_CPU_CAPABILITY": "avx2",
    "MKL_CBWR": "AVX2",
    "DNNL_MAX_CPU_ISA": "AVX2",
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    # numpy: PyPI wheels bundle OpenBLAS, which dispatches by detected CPU just
    # as ATen does. OPENBLAS_NUM_THREADS is set explicitly rather than relying
    # on OpenBLAS's fallback to OMP_NUM_THREADS, which only applies to OpenMP
    # builds. Haswell is the AVX2-era kernel family, matching the torch pin.
    "OPENBLAS_CORETYPE": "Haswell",
    "OPENBLAS_NUM_THREADS": "1",
}


def pin_cpu_dispatch() -> None:
    """Pin kernel dispatch and thread counts. Idempotent; setdefault semantics.

    An operator who has deliberately exported one of these keeps their value —
    the environment fingerprint in the reveal records what was actually in
    effect, so a divergence caused by an override stays diagnosable.
    """
    for key, value in DETERMINISM_ENV.items():
        os.environ.setdefault(key, value)


pin_cpu_dispatch()
