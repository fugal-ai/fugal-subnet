"""Summarize an interrupted 512 pass without claiming held-out routing quality."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_success_contract import c, load_progress, np  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--backbone", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    src, out = Path(args.data), Path(args.results)
    report = json.loads((out / "report-2048.json").read_text())
    for name, expected in report["sources"].items():
        if c.file_hash(src / name) != expected:
            raise ValueError("source extract mismatch")
    prompts = json.loads((src / "sprout_prompts.json").read_text())["prompts"]
    questions = [prompts[i] for i in report["selected_indices"]]
    profile = c.digest(dict(c.PROFILE, max_length=512, version="experimental-512"))
    short, completed, seconds, peak = load_progress(out / "embedding-progress-512.npz", questions, profile)
    full = c.load_cache(out / "embeddings-2048.npz", questions)
    from transformers import AutoTokenizer
    c.check_backbone(args.backbone)
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, local_files_only=True)
    lengths = np.array([len(tokenizer(c.format_question(q))["input_ids"]) for q in questions])
    order = np.argsort(lengths, kind="stable")[:completed]
    unchanged = order[lengths[order] <= 512]
    delta = float(np.max(np.abs(short[unchanged] - full[unchanged]))) if len(unchanged) else None
    if len(unchanged):
        np.testing.assert_allclose(short[unchanged], full[unchanged], atol=1e-5, rtol=1e-5)
    summary = {"status": "stopped at user request; no 512 head trained or evaluated",
               "sources": report["sources"], "profile_id": profile, "total_questions": len(questions),
               "completed_questions": completed, "completed_embedding_seconds": seconds,
               "partial_peak_rss_kib": peak, "order": "ascending untruncated token length; not a random subset",
               "completed_max_input_tokens": int(lengths[order].max()) if completed else None,
               "would_truncate_at_512": int((lengths > 512).sum()),
               "would_truncate_at_2048": int((lengths > 2048).sum()),
               "completed_inputs_exceeding_512": int((lengths[order] > 512).sum()),
               "untruncated_embedding_comparisons": len(unchanged), "untruncated_max_abs_difference": delta,
               "conclusion": "Compatible 2048 routing is established. No conclusion about relative 512/2048 held-out quality or full-pass performance is supported by this partial run."}
    Path(args.output).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
