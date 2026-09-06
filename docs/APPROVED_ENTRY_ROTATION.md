# Rotating the approved entry

What a `--live` validator accepts is a list of entries, each
`<base_measurement>:<compose_hash>`. The base is what booted (the dstack image
on a given cloud platform); the compose hash is what ran (the raw bytes of the
miner's `app-compose.json`, which embed the compose file and therefore the
image digest). Either half changes on a schedule set by someone else — a dstack
release, a kernel update, a rebuilt miner image — and when it does, every
miner's proof stops verifying until validators approve the new value.

This is the procedure for doing that without a gap in which honest miners are
scored as absent, or a window in which anyone can slip in an unreviewed image.
It uses nothing the code does not already have: `FUGAL_TEE_MEASUREMENTS` is
comma-separated and `parse_approved` keeps several entries per base.

## What triggers a rotation

| Change | Half that moves | Who controls the timing |
|---|---|---|
| Miner image rebuilt (any commit to `main` that reaches the image) | compose hash | us |
| Any edit to the compose or `app.json` fields that land in the compose | compose hash | us |
| dstack guest OS release | base (MRTD/RTMR1/RTMR2) and compose hash (`os_image` name) | dstack |
| Cloud platform firmware/TDVF update | base | the cloud provider |

Measured 2026-09-06: instance size does **not** move the base (RTMR0 is
excluded), and the instance id does not move the compose hash (it is an event
payload, not a compose field). Two honest miners on the same compose share one
entry regardless of how many times they redeploy.

## Computing the new entry

1. Build the miner image from a clean checkout of the exact commit, by digest.
2. Put the digest into `deploy/dstack/docker-compose.yaml`; run
   `dstack-cloud prepare` in a project whose `app.json` has the documented
   field values (`deploy/dstack/README.md`); do not touch `prelaunch.sh`.
3. `python scripts/compute_app_identity.py --compose shared/app-compose.json
   --base <base>` prints the entry. Hash the **raw bytes**; never a
   re-serialisation.
4. Commit the frozen `app-compose.json` under `deploy/dstack/` next to the
   compose, so anyone can recompute the hash from the repository.
5. A second person recomputes the hash from the committed bytes before it is
   published. On 2026-09-06 two sessions did this independently and matched.

## Publishing

An entry is a consensus parameter. Publish it where every validator operator
already looks and where it cannot be edited silently: a signed GitHub release
note naming the commit, the image digest, the compose hash and the base, plus
a row in the table below. A validator operator who cannot trace an entry back
to a commit and a digest should not approve it.

| Entry (base:compose) | Image digest | Commit | Approved | Retired |
|---|---|---|---|---|
| `12a1f2f5…9141:cf31a1a4…69d2` | `3430da97…541f` | `e284963` | 2026-09-06 | — |

## The transition window

1. **Approve both.** Every validator adds the new entry *alongside* the old
   one in `FUGAL_TEE_MEASUREMENTS` and restarts. A restart mid-epoch is safe:
   the validator resumes the same epoch from chain state (measured 2026-09-06,
   16 s, same slice and commit hash). Validators must all carry both entries
   before any miner moves, or the first miner to move is scored as invalid by
   the validators that have not.
2. **Miners move.** Each miner redeploys onto the new image (`dstack-cloud
   deploy --delete`), which recreates the instance, discards its data disk and
   sealed seed, and re-pays the backbone pass (about 5 hours at 4 vCPUs). Then
   the operator re-provisions over the sealed channel. A miner is unscored
   for the epochs it is down; nothing in the window changes that, so miners
   should move at a time of their choosing, not all at once.
3. **Retire the old entry** once no verified proof has carried it for a full
   day of epochs, and never sooner than a published date. Retiring means every
   validator removes it and restarts. Until then, both are valid, and a proof
   from either image verifies.

The window should be long enough for a miner operator to notice, redeploy and
re-embed with margin: **48 hours minimum** for a compose-hash rotation, longer
for a base rotation, because that one also needs the dstack image re-uploaded
to the cloud project (about 8 minutes, measured, but it is a new moving part).

## What must not happen

- **A bare base entry.** `<base>` alone approves any application on the image.
  `provision_td.py` refuses to push to one; validators should never carry one
  in production.
- **An entry nobody can reproduce.** If the image behind a compose hash is not
  publicly pullable, or the frozen compose is not committed, the entry is an
  assertion, not evidence.
- **Retiring before every validator carries the new entry.** Two validators
  with different approved sets disagree about identical bytes, which is a
  consensus fork that looks like miner misbehaviour.

## Verifying a rotation

From any machine holding a registered hotkey:

```bash
python scripts/verify_live_miner.py --netuid <N> --uid <uid> \
    --coldkey <wallet> --measurements <new entry>          # PASS
python scripts/verify_live_miner.py ... --measurements <old entry> --expect-reject   # PASS after the miner moved
```

and `scripts/watch_subnet.py` for the epochs that follow: validators must keep
agreeing byte-for-byte across the window. If they stop agreeing, one of them
does not carry both entries.
