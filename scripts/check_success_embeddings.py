#!/usr/bin/env python3
"""Check actual core and subnet backbone wrappers, with CPU float32 only."""
import argparse
import gc
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--core", required=True)
    p.add_argument("--backbone", required=True)
    p.add_argument("--output")
    args = p.parse_args()
    os.environ.setdefault("FUGAL_BACKBONE_THREADS", "4")
    # isort: off
    import fugal_subnet.determinism  # noqa: F401
    import numpy as np
    # isort: on

    from fugal_subnet.backbone import (
        compute_hidden_states,
        configure_determinism,
        get_backbone,
        release_backbone,
    )
    from fugal_subnet.vendor import success_contract as subnet

    sys.path.insert(0, args.core)
    from fugal import success_contract as core
    from fugal.router import FugalRouter

    configure_determinism()
    fixture = json.loads((ROOT / "fugal_subnet/vendor/conformance.json").read_text())
    questions = [q if isinstance(q, str) else q["repeat"] * q["count"] for q in fixture["questions"]]
    router = FugalRouter(args.backbone, success_profile=True)
    core_tokens = core.tokenize(router.tok, questions)["input_ids"].numpy()
    core_hidden = router.embed_questions(questions, batch_size=2)
    del router
    gc.collect()
    release_backbone()
    tok, model = get_backbone(args.backbone)
    sub_tokens = subnet.tokenize(tok, questions)["input_ids"].numpy()
    np.testing.assert_array_equal(core_tokens, sub_tokens)
    batched = compute_hidden_states(questions, model_name=args.backbone, batch_size=2)
    single = np.concatenate([compute_hidden_states([q], model_name=args.backbone, batch_size=1) for q in questions])
    np.testing.assert_allclose(core_hidden, batched, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(single, batched, atol=1e-5, rtol=1e-5)
    from fugal_subnet.head_eval import load_head_from_npz
    from fugal_subnet.routing_protocol import benchmark_costs
    from fugal_subnet.tee.harness import _route_question
    head = load_head_from_npz((ROOT / "tests/fixtures/success_bundle/head.npz").read_bytes())
    costs = benchmark_costs(head)
    expected = core.rank(core.predictions(head.W, head.b, core_hidden), costs, 1)[:, 0]
    actual = [_route_question(head, h, costs) for h in batched]
    np.testing.assert_array_equal(expected, actual)
    result = {"core_commit": json.loads((ROOT / "fugal_subnet/vendor/SOURCE.json").read_text())["commit"],
              "profile_id": core.PROFILE_ID, "token_ids_equal": True,
              "core_subnet_max_abs": float(np.max(np.abs(core_hidden-batched))),
              "batch_single_max_abs": float(np.max(np.abs(single-batched))),
              "atol": 1e-5, "rtol": 1e-5, "selected_indices": actual}
    result["selected_indices"] = [int(i) for i in actual]
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    del tok, model
    release_backbone()


if __name__ == "__main__":
    main()
