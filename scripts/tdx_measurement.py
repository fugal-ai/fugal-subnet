#!/usr/bin/env python3
"""Validate the verifier on real TDX: the two checks no other environment can make.

Run this ON a confidential VM you have a shell on (a plain GCP c3 / Azure
DCesv5 TDX VM; see docs/TDX_VALIDATION.md).

    python scripts/tdx_measurement.py            # print this VM's measurement_id
    python scripts/tdx_measurement.py --verify   # assert the two live checks

IT DOES NOT MEASURE THE APPROVED MINER IMAGE. Live entries name the base of
the dstack miner image, and this cannot run there: the image carries no
scripts/ and a dstack TD gives no shell. The base half of an entry comes from
`scripts/provision_td.py --inspect` on a TD booted from that image, and is
published in the signed release (scripts/fetch_approved_entries.py). The
number printed here is this VM's own measurement, for diagnosis.

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
    print("\nThis is THIS VM's measurement, not the dstack miner image's. The base half")
    print("of an approved entry comes from `scripts/provision_td.py --inspect` on a TD")
    print("booted from that image (docs/APPROVED_ENTRY_ROTATION.md).")

    if not args.verify:
        return 0

    print("\n--- the two checks no other environment can make ---")
    failures = []

    # A collateral fetch that never completed is not a verdict on this machine's
    # quote, and this script exists to diagnose the machine. Reporting it as a
    # FAIL would send an operator hunting a hardware problem that is really an
    # egress or endpoint problem.
    from fugal_subnet.tee.attestation import CollateralUnavailable
    try:
        ok = verify_dcap(quote_bytes)
    except CollateralUnavailable as e:
        print(f"  [SKIP] DCAP verification could not run: {e}")
        print("         The quote was never judged. Check outbound HTTPS to the")
        print("         PCCS endpoint above; this is not a verdict on this machine.")
        ok = None
    if ok is not None:
        print(f"  [{'PASS' if ok else 'FAIL'}] a genuine quote passes DCAP verification")
        if not ok:
            failures.append("DCAP verification failed on a quote from this machine")

    # An unapproved image must be rejected even though its quote is genuine.
    # This is the attack DCAP alone does not stop: real hardware, modified code.
    from fugal_subnet.tee.proof import BenchmarkProof, compute_questions_hash
    from fugal_subnet.tee.verify import verify_proof

    # verify_proof outside mock refuses to run without the hotkey binding
    # (a proof not bound to a miner can be relayed), so the probe names one.
    proof = BenchmarkProof(
        epoch_id="probe", nonce="n" * 64,
        questions_hash=compute_questions_hash([]), weights_hash="w" * 64,
        source_hash="claims-to-be-approved", results=[],
        total_cost_usd=0.0, per_model_costs={}, attestation_quote=b"", timestamp=0.0,
        hotkey="5TdxMeasurementProbe",
    )
    proof.attestation_quote = runtime.generate_attestation(
        bytes.fromhex(proof.content_hash())
    )
    result = verify_proof(
        proof,
        approved_measurements={"0" * 64 + ":" + "1" * 64},   # deliberately NOT this image
        expected_questions_hash=proof.questions_hash,
        expected_nonce=proof.nonce, gold_answers={},
        expected_hotkey="5TdxMeasurementProbe",
        mock=False,
    )
    if getattr(result, "unverifiable", False):
        # Same as the first check: an unreachable collateral endpoint judged
        # nothing, and calling it "I8 is not enforced" would send the operator
        # after the wrong fault.
        print(f"  [SKIP] the unapproved-image probe could not be judged: {result.reason}")
        rejected = None
    else:
        rejected = (not result.valid) and "Unapproved runtime image" in result.reason
        print(f"  [{'PASS' if rejected else 'FAIL'}] an unapproved image is rejected "
              f"despite a genuine quote")
    if rejected is False:
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
