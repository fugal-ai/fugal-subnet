#!/usr/bin/env python3
"""Build the pinned v3 embeddings file (one row per pool question), sharded.

A row is the pinned success-profile embedding of ONE question's prompt text,
computed alone (batch size 1, one torch thread, the pinned CPU dispatch), so
it depends on that prompt and nothing else: not on which machine, shard or
neighbour computed it. That is what makes a sharded build exact, and what
`verify` checks by recomputing a sample bit for bit.

    # 1. the prompts: a JSON list of {"question_id", "prompt", "sha"} (sha =
    #    sha256 of the prompt's UTF-8 bytes), e.g. from load_pool(3)
    # 2. on each machine, one shard (resumable; writes chunk .npz files):
    python scripts/build_embeddings_v3.py run --prompts P.json.gz --shard 0/3 --workers 56 --out chunks/
    # 3. anywhere, from every machine's chunks, against the FINAL pool's prompts:
    python scripts/build_embeddings_v3.py assemble --prompts FINAL.json.gz --chunks chunks/ --out pool_v3.npz
    # 4. recompute a random sample and require bitwise equality:
    python scripts/build_embeddings_v3.py verify --prompts FINAL.json.gz --file pool_v3.npz --sample 24

`assemble` refuses a row whose recorded prompt hash differs from the final
prompt's (the text changed after it was embedded) and names every missing
question, so nothing stale or absent reaches the pinned file. The output is
written with fixed zip timestamps, so the same rows give the same bytes.
"""
import argparse
import glob
import gzip
import hashlib
import json
import os
import sys
import time
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _read_prompts(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        rows = json.load(f)
    for r in rows:
        if hashlib.sha256(r["prompt"].encode("utf-8")).hexdigest() != r["sha"]:
            raise SystemExit(f"prompt hash mismatch in the input for {r['question_id']}")
    return rows


def _shard_of(question_id, n):
    return int(hashlib.sha256(question_id.encode("utf-8")).hexdigest(), 16) % n


def _write_npz(path, arrays):
    """An uncompressed .npz with fixed timestamps: the same arrays, the same bytes."""
    import numpy as np

    tmp = path + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
        for name, arr in arrays.items():
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            with zf.open(info, "w", force_zip64=True) as f:
                np.lib.format.write_array(f, np.ascontiguousarray(arr), allow_pickle=False)
    os.replace(tmp, path)


def _worker(args):
    """Embed `rows` one at a time in this process (one torch thread), writing a
    chunk every `every` rows. Skips ids already in this worker's chunks."""
    rows, out_dir, tag, every = args
    import numpy as np

    from fugal_subnet.backbone import compute_hidden_states  # model already loaded by the parent

    done = set()
    for p in glob.glob(os.path.join(out_dir, f"chunk-{tag}-*.npz")):
        with np.load(p, allow_pickle=False) as z:
            done.update(str(x) for x in z["question_ids"].tolist())
    todo = [r for r in rows if r["question_id"] not in done]
    k = len(glob.glob(os.path.join(out_dir, f"chunk-{tag}-*.npz")))
    for start in range(0, len(todo), every):
        part = todo[start:start + every]
        H = np.concatenate([compute_hidden_states([r["prompt"]], batch_size=1) for r in part])
        _write_npz(os.path.join(out_dir, f"chunk-{tag}-{k:05d}.npz"), {
            "question_ids": np.array([r["question_id"] for r in part]),
            "sha": np.array([r["sha"] for r in part]),
            "H": H.astype(np.float32)})
        k += 1
    return len(todo)


def _load_backbone_once():
    os.environ["FUGAL_BACKBONE_THREADS"] = "1"
    import fugal_subnet.determinism  # noqa: F401  (pins dispatch before numpy/torch)
    from fugal_subnet.backbone import configure_determinism, get_backbone

    configure_determinism()
    get_backbone()


def cmd_run(a):
    import multiprocessing as mp

    i, n = (int(x) for x in a.shard.split("/"))
    rows = [r for r in _read_prompts(a.prompts) if _shard_of(r["question_id"], n) == i]
    os.makedirs(a.out, exist_ok=True)
    # Longest first, dealt round-robin, so the workers finish together.
    rows.sort(key=lambda r: -len(r["prompt"]))
    groups = [rows[w::a.workers] for w in range(a.workers)]
    jobs = [(g, a.out, f"s{i}of{n}-w{w:03d}", a.every) for w, g in enumerate(groups) if g]
    t0 = time.time()
    print(f"shard {i}/{n}: {len(rows)} questions, {len(jobs)} workers", flush=True)
    # Load the backbone once in the parent (one torch thread, pinned dispatch),
    # then fork: the workers share its weights copy-on-write, so a machine holds
    # one 2.4 GB model instead of one per worker. Nothing has run a forward pass
    # in the parent, so no thread pool is forked mid-flight.
    _load_backbone_once()
    with mp.get_context("fork").Pool(len(jobs)) as pool:
        made = pool.map(_worker, jobs, chunksize=1)
    print(f"shard {i}/{n}: embedded {sum(made)} in {time.time() - t0:.0f} s", flush=True)


def _load_chunks(dirs):
    import numpy as np

    rows = {}
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "chunk-*.npz"))):
            with np.load(p, allow_pickle=False) as z:
                for qid, sha, h in zip(z["question_ids"].tolist(), z["sha"].tolist(), z["H"]):
                    rows[str(qid)] = (str(sha), np.array(h, dtype=np.float32))
    return rows


def cmd_assemble(a):
    import numpy as np

    prompts = _read_prompts(a.prompts)
    have = _load_chunks(a.chunks)
    missing = [r["question_id"] for r in prompts if r["question_id"] not in have]
    stale = [r["question_id"] for r in prompts
             if r["question_id"] in have and have[r["question_id"]][0] != r["sha"]]
    if missing or stale:
        with open(a.out + ".todo.json", "w") as f:
            json.dump({"missing": missing, "stale": stale}, f)
        raise SystemExit(f"not assembled: {len(missing)} missing, {len(stale)} stale (prompt text "
                         f"changed); ids written to {a.out}.todo.json")
    ids = [r["question_id"] for r in prompts]
    H = np.stack([have[q][1] for q in ids]).astype(np.float32)
    from fugal_subnet.embeddings_file import Embeddings

    Embeddings(ids, H)  # the loader's own shape, uniqueness and bounds checks
    _write_npz(a.out, {"question_ids": np.array(ids), "H": H})
    h = hashlib.sha256()
    with open(a.out, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    print(f"{a.out}: {len(ids)} rows x {H.shape[1]}, sha256 {h.hexdigest()}")


def cmd_verify(a):
    import random

    os.environ["FUGAL_BACKBONE_THREADS"] = "1"
    import fugal_subnet.determinism  # noqa: F401
    import numpy as np

    from fugal_subnet.backbone import compute_hidden_states
    from fugal_subnet.embeddings_file import Embeddings

    prompts = {r["question_id"]: r for r in _read_prompts(a.prompts)}
    emb = Embeddings.load(a.file, None)
    rng = random.Random(a.seed)
    sample = rng.sample(sorted(prompts), a.sample)
    bad = 0
    for q in sample:
        h = compute_hidden_states([prompts[q]["prompt"]], batch_size=1)[0]
        same = np.array_equal(h, emb.of(q))
        bad += not same
        print(f"{q}: {'identical' if same else 'DIFFERENT max|d|=%g' % float(np.abs(h - emb.of(q)).max())}",
              flush=True)
    print(f"verify: {a.sample - bad}/{a.sample} bitwise identical")
    if bad:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--prompts", required=True)
    r.add_argument("--shard", default="0/1")
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--every", type=int, default=25)
    r.add_argument("--out", required=True)
    s = sub.add_parser("assemble")
    s.add_argument("--prompts", required=True)
    s.add_argument("--chunks", nargs="+", required=True)
    s.add_argument("--out", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--prompts", required=True)
    v.add_argument("--file", required=True)
    v.add_argument("--sample", type=int, default=24)
    v.add_argument("--seed", default="fugal-embeddings-v3")
    a = ap.parse_args()
    {"run": cmd_run, "assemble": cmd_assemble, "verify": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    main()
