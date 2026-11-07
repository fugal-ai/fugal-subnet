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
2. Put the digest into the compose for the target network
   (`deploy/dstack/docker-compose.yaml` for finney/70, `docker-compose.test.yaml`
   for test/552 — one digest, one entry per network); run
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

The channel, built 2026-10-14 (diligence finding 7):

1. **Image.** Push a tag `image-v<date>`; `.github/workflows/image.yml` builds
   the image, pushes it to `ghcr.io/<org>/fugal-miner` **by digest** (public,
   so any validator can pull what it approves), writes the digest into a copy
   of `deploy/dstack/docker-compose.yaml`, and drafts a release carrying both.
   Miners deploy the published digest; an image anyone builds themselves has a
   different digest, hence a different compose hash, and is refused.
2. **Compose.** Run `dstack-cloud prepare` with that compose (README in
   `deploy/dstack/`), commit the frozen `app-compose.json`, and have a second
   person recompute its hash from the committed bytes.
3. **Note and tag message.** `python scripts/publish_approved_entry.py
   --digest <digest> --commit <sha> --base <base> --tag image-v<date>
   --compose deploy/dstack/app-compose.frozen-<tag>-finney.json
   --compose deploy/dstack/app-compose.frozen-<tag>-test.json
   --tag-message tag-message.txt > notes.md` writes both networks' entries:
   one `approved-entry <network>: <base>:<compose_hash>` line each in the tag
   message, mirrored in the note.
4. **Sign.** `git tag -s image-v<date> -F tag-message.txt` with a key listed
   in `deploy/keys/allowed_signers` (SSH signing; the file is the trust root
   and is added to by reviewed pull request), push the tag, and
   `gh release create image-v<date> --notes-file notes.md --verify-tag`. The
   entries are what the tag signs; the note is for people.
5. **Validators.** `python scripts/fetch_approved_entries.py --tag
   image-v<date> --network <finney|test>` verifies the tag signature against
   the committed signers, reads that network's entry from the signed message
   (refusing a note that disagrees), and prints the `FUGAL_TEE_MEASUREMENTS`
   line. It never edits the environment; adding the entry alongside the
   current one, restarting, and retiring the old one after the window below
   is the operator's act. Releases whose tag message carries no entry lines
   (`image-v2026.11.03b`, `03c`) are refused by it; validators keep the
   entries they already run from those.

A validator does not check GitHub at startup. Its environment is
authoritative, on purpose: a validator that cannot start because GitHub is
down is a worse failure than one that needs an operator to run step 5.

| Entry (base:compose) | Image digest | Commit | Approved | Retired |
|---|---|---|---|---|
| `12a1f2f5…9141:cf31a1a4…69d2` | `3430da97…541f` | `e284963` | 2026-09-06 | 2026-09-07 (no instance ever verified on it; both TDs recreated) |
| `12a1f2f5…9141:12d945d2…3910` | `59039d01…692a` | `45729c2` | 2026-09-06 | 2026-09-07 (never verified; TDs recreated) |
| `12a1f2f5…9141:00354525…ab0d` | `9d7916c3…e0a9` | `d9fa3a5` | 2026-09-07 | 2026-11-03 (private-registry image; pre-success-contract protocol; testnet only) |
| `12a1f2f5…9141:41de1e4c…9e10` | `e4c2deff…6ca4` | `37495ae` | 2026-09-07 | 2026-11-03 (same) |
| `f4be9383…aa427:dcd0dd8d…2d7f` (**test / 552**, warm-up; compose sets `FUGAL_EXPLORE_FRACTION=0.5`) | `3f6e3e2d…fda12` (`image-v2026.11.03b`, public ghcr) | `bac2a33` (image), `46507ce` (compose) | 2026-11-03 | **2026-11-06 00:00 UTC** (hard date, published; see below) |
| `f4be9383…aa427:ed2be85f…85c4` (**finney / 70**, mainnet) | `3f6e3e2d…fda12` (same image) | `bac2a33` (image), `46507ce` (compose) | 2026-11-03 | **2026-11-06 00:00 UTC** (hard date, published; see below) |
| `f4be9383…aa427:c9b5c43b…0dad8` (**test / 552**, warm-up twin) | `843460e6…4a6f8` (`image-v2026.11.03c`, public ghcr; #121 retries) | `1a0bce0` (image), `79ab299` (compose) | 2026-11-03 | — |
| `f4be9383…aa427:9ab870cf…f166` (**finney / 70**, mainnet) | `843460e6…4a6f8` (same image) | `1a0bce0` (image), `79ab299` (compose) | 2026-11-03 (base re-measured on the first 03c TD, 23:16 UTC: unchanged) | — (the only live entries since 03b retired on 2026-11-06) |

**03b retirement is a hard date.** The two `image-v2026.11.03b` entries retire at
**2026-11-06 00:00 UTC**, published 2026-11-04, 48 h after the 03c release plus
margin. Operators on 03b should be on 03c by **2026-11-05 00:00 UTC**. For this
rotation the published date overrides the "no verified proof for a full day"
condition in step 3 of the transition window: a miner still on 03b at the date
stops verifying, which is the notice this date gives.

**The base moved between September and November without the image changing.**
Measured 2026-11-03 on two fresh `c3-standard-4` TDs booted from the
sha256-verified dstack 0.6.0-rc0 UKI: RTMR1 and RTMR2 (kernel, initrd, the
image) are byte-identical to the September fixtures; MRTD and RTMR0 (the TDX
virtual firmware and its configuration, which GCP controls) changed. So
`measurement_id` went from `12a1f2f5…9141` to `f4be9383…aa427` and every
entry above the line was re-based before publication. The rule this adds:
**measure the base on a live TD the day an entry is published, and treat a
GCP firmware roll as a base rotation** (both bases approved across the window,
or every honest miner that reboots onto the new firmware is refused).

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

**Protocol switches are the exception (v2).** A compose-hash rotation inside one
protocol version runs both entries side by side for the window above. A
protocol switch can't: from the activation block every validator scores under
the new rules, so a proof from the old image can't be scored after it. For v2
the notice before the activation block is therefore also the whole overlap:
at least 24 hours, with the v2 code, model list and pool public on
`release/v2` ahead of the announcement so operators can prepare early. The old
entries retire at the activation block. The activation block itself travels in
the v2 compose (`FUGAL_V2_ACTIVATION_BLOCK` in `deploy/dstack/docker-compose.yaml`,
which the miner image reads), so choosing it is what fixes the v2 compose hash:
freeze the compose after the block is chosen, not before, and publish that
entry. `scripts/check_safety_invariants.py` reports a live compose that no
frozen app-compose embeds (a warning in CI, a failure with `--release`, the
launch gate), so an edit after a freeze cannot ship unnoticed.

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
