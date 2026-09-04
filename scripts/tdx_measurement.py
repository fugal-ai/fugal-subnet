#!/usr/bin/env python3
"""Print this TD's measurement, and verify the two checks only real TDX can.

Run this ON a confidential VM. See docs/TDX_VALIDATION.md.

    python scripts/tdx_measurement.py            # print measurement_id
    python scripts/tdx_measurement.py --verify   # assert the two live checks

Everything else in this repo is exercised by scripts/dress_rehearsal.py against
a real chain. These two cannot be: they need genuine Intel-signed quotes and
real measurement registers, which a CPU without TDX cannot produce. Mock mode
does not weaken them — it is their absence.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _require_tdx() -> None:
    if not os.path.exists("/dev/tdx_guest"):
        raise SystemExit(
            "No /dev/tdx_guest — this is not an Intel TDX guest.\n"
            "Nothing here can run without one. See docs/TDX_VALIDATION.md for "
            "provisioning a GCP c3 or Azure DCesv5 confidential VM."
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true",
                    help="Assert DCAP passes and an unapproved image is rejected")
    args = ap.parse_args()

    _require_tdx()

    from fugal_subnet.tee.attestation import measurement_id, parse_quote, verify_dcap
    from fugal_subnet.tee.runtime import TEERuntime

    runtime = TEERuntime(mock=False)
    quote_bytes = runtime.generate_attestation(b"\x00" * 64)
    quote = parse_quote(quote_bytes)
    measurement = measurement_id(quote)

    print(f"mrtd            : {quote.mrtd}")
    print(f"rtmr0           : {quote.rtmr0}")
    print(f"rtmr1           : {quote.rtmr1}")
    print(f"rtmr2           : {quote.rtmr2}")
    print(f"\nmeasurement_id  : {measurement}")
    print("\nApprove this image with:")
    print(f"  export FUGAL_TEE_MEASUREMENTS={measurement}")

    if not args.verify:
        return 0

    print("\n--- the two checks no other environment can make ---")
    failures = []

    ok = verify_dcap(quote_bytes)
    print(f"  [{'PASS' if ok else 'FAIL'}] a genuine quote passes DCAP verification")
    if not ok:
        failures.append("DCAP verification failed on a quote from this machine")

    # An unapproved image must be rejected even though its quote is genuine.
    # This is the attack DCAP alone does not stop: real hardware, modified code.
    from fugal_subnet.tee.proof import BenchmarkProof, compute_questions_hash
    from fugal_subnet.tee.verify import verify_proof

    proof = BenchmarkProof(
        epoch_id="probe", nonce="n" * 64,
        questions_hash=compute_questions_hash([]), weights_hash="w" * 64,
        source_hash="claims-to-be-approved", results=[],
        total_cost_usd=0.0, per_model_costs={}, attestation_quote=b"", timestamp=0.0,
    )
    proof.attestation_quote = runtime.generate_attestation(
        bytes.fromhex(proof.content_hash())
    )
    result = verify_proof(
        proof,
        approved_measurements={"0" * 64},      # deliberately NOT this image
        expected_questions_hash=proof.questions_hash,
        expected_nonce=proof.nonce, gold_answers={},
        mock=False,
    )
    rejected = (not result.valid) and "Unapproved runtime image" in result.reason
    print(f"  [{'PASS' if rejected else 'FAIL'}] an unapproved image is rejected "
          f"despite a genuine quote")
    if not rejected:
        failures.append(
            f"unapproved image was NOT rejected: valid={result.valid} "
            f"reason={result.reason!r}"
        )

    if failures:
        print("\nFAIL — I8 is not enforced on this machine:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nPASS — attestation is genuinely verified, and a genuine quote from "
          "an unapproved image is still rejected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
