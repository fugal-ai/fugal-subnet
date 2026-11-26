# Release signing trust root

`allowed_signers` is an OpenSSH allowed-signers file: one line per person who
may sign an approved-entry release tag, `<email> <key type> <public key>`.
It is committed here so the trust root lives in the same reviewed history as
the code it approves. `scripts/fetch_approved_entries.py` verifies a tag
against it with `git tag -v` and nothing else, and reads the approved entries
from the signed tag message.

To add a signer: append their SSH signing public key, in a reviewed pull
request. To sign a tag: `git config gpg.format ssh`, `git config
user.signingkey ~/.ssh/<key>.pub`, then `git tag -s <tag> -F tag-message.txt`
with the message `scripts/publish_approved_entry.py --tag-message` writes.

The first key was added 2026-11-03: the release key `release@fugalai.com`,
ED25519, fingerprint `SHA256:MpaQWZxhrW77AJg8HAR2TxgFrag5HAo+i9oDnUiIBO8`,
held by the project owner. Its line carried `namespaces="git"`, so the key
was accepted for git signatures only. Until that line landed the file was
absent on purpose, so the fetch script failed closed rather than verifying
against nothing; removing the last line restores that state.

The second key was added 2026-11-09: `release2@fugalai.com`, ED25519,
fingerprint `SHA256:ncBkRnJ8aHy3AfChppE9KhATrM8eFkiFTOrxvNyEzpg`, generated
and held on a separate device by the project owner, passphrase-protected,
also `namespaces="git"` then. It is the backup: either key alone verifies a tag,
so losing one machine no longer leaves the project unable to sign a release.
Verified before it was added: a `git`-namespace signature by it over a known
message verifies against this file, and the same signature over a different
message does not.

## Validator release tags (`validator-v*`)

The opt-in validator auto-updater (`deploy/autoupdate/`) installs a release
only from an annotated tag `validator-vX.Y.Z` signed by a key in this file, in
the `git` namespace, as the file stood when the operator installed the updater
(a copy in `/etc/fugal/autoupdate/allowed_signers`; it never reads the file
from fetched code). A key added or removed here reaches those validators only
when their operators re-run `deploy/autoupdate/install.sh`; after a lost key,
say so in the release announcement. Signing one, on the merged release commit:

    git -c gpg.format=ssh -c user.signingkey=$HOME/.ssh/fugal_release.pub \
        tag -s validator-vX.Y.Z -m "validator vX.Y.Z" <commit>
    python3 deploy/autoupdate/fugal_autoupdate.py verify-tag validator-vX.Y.Z
    git push origin validator-vX.Y.Z

`docs/VALIDATOR_GUIDE.md`, "Auto-update (optional)", has the rules.

## Signed lists (protocol v4; nothing reads one before a v4 activation block)

Since 2026-11-18 both lines read `namespaces="git,fugal-list@fugalai.com"`:
the same two keys, accepted for git tags as before and for one more thing, a
**signed list**. Protocol v4 keeps two lists as signed data, so the owner can
change them without a release (`docs/decisions/LIVE_MENU_V4.md`, sections 6
and 13.4): the lab and host allowlist a menu entry must satisfy, and the
device list of a second attestation path. A signature names its namespace
inside the signed bytes, so a tag signature is never a list signature and a
list signature is never a tag signature, though one key makes both.

The neurons do not run `ssh-keygen` to check a list. `fugal_subnet/menu_rules.py`
verifies the OpenSSH signature in process, and reads the signers from
`fugal_subnet/signed_lists/allowed_signers`: a byte-identical copy of this
file (this directory is not in the miner image), pinned by hash in
`menu_rules.LIST_SIGNERS_SHA256` and in `scripts/check_safety_invariants.py`.
It reads one form of line only,

    <principal> namespaces="<name>[,<name>...]" ssh-ed25519 <key> [comment]

with single spaces and plain namespace names. A line without `namespaces`, with
a pattern (`*`), with another option or with another key type grants nothing
for lists, and the safety check refuses a file that has one, so the file
means the same to OpenSSH and to the neurons. **A change to this file is a
change to the copy and to both pins, in the same pull request.**

### Signing a list

A list is one JSON file: a `kind`, a `serial`, and sorted arrays. A newer
list must carry a higher serial than the one a node holds; serial 0 marks a
draft and is refused whatever signs it.

Shipped: `fugal_subnet/signed_lists/menu_allowlist.json`, serial 1 (v0.7.0,
2026-11-22). The next list carries serial 2: write its draft from the shipped
list's content (`--draft`), review it, and follow the same steps.

    # 1. Review the draft. Before serial 1 it was data/menu_allowlist.draft.json,
    #    first derived from the labs and hosts data/models_v3b.json pins, then
    #    extended by review. Do not regenerate it (plain `draft` refuses to
    #    replace a draft that differs from the derivation; `--overwrite` forces).
    #    Edit it if needed (keep the arrays sorted; `prepare` refuses a file it
    #    cannot read).
    uv run python scripts/signed_list.py draft --check    # derivation, or reviewed draft that contains it?

    # 2. Write the bytes to sign, with the next serial (1 for the first list).
    uv run python scripts/signed_list.py prepare --serial 1
    #    -> fugal_subnet/signed_lists/menu_allowlist.json

    # 3. Sign. The key holder runs this; nothing else touches the private key.
    ssh-keygen -Y sign -f ~/.ssh/<release key> -n fugal-list@fugalai.com \
        fugal_subnet/signed_lists/menu_allowlist.json
    #    -> fugal_subnet/signed_lists/menu_allowlist.json.sig

    # 4. Verify, both ways. The first is what every neuron does.
    uv run python scripts/signed_list.py verify fugal_subnet/signed_lists/menu_allowlist.json
    ssh-keygen -Y verify -f deploy/keys/allowed_signers -I release@fugalai.com \
        -n fugal-list@fugalai.com -s fugal_subnet/signed_lists/menu_allowlist.json.sig \
        < fugal_subnet/signed_lists/menu_allowlist.json

    # 5. In one commit: the list, its .sig, its hash in
    #    PINNED_SIGNED_LISTS_SHA256 (scripts/check_safety_invariants.py; `verify`
    #    prints it), and the draft removed. Then:
    uv run python scripts/check_safety_invariants.py

Do not pass `-O hashalg=...` when signing: the verifier accepts the default
(SHA-512) only. `-I` is `release2@fugalai.com` when the backup key signed. A
device list is the same with `--kind devices` and `device_list.json`.

## Signing a protocol schedule

A protocol schedule (`docs/decisions/SIGNED_SCHEDULE.md`) is signed with the
same keys under the lists' namespace, `fugal-list@fugalai.com`; its `kind`
(`fugal-protocol-schedule`) inside the signed bytes keeps it apart from a
list. `scripts/release.py` writes the bytes and checks everything else:

    uv run python scripts/release.py schedule draft --set <switch>=<block> --apply
    ssh-keygen -Y sign -f ~/.ssh/<release key> -n fugal-list@fugalai.com release-out/schedule-<N>.json
    uv run python scripts/release.py schedule verify release-out/schedule-<N>.json --install

`verify` prints the two `gsutil cp` commands that publish the pair to the
root's bucket. Sign only with a key every deployed release lists: the neurons
check against the signers compiled into their release and image. Never sign
two documents with the same serial; a newer one takes the next serial.

## Rotating a key

- **Planned:** add the new key's line in a reviewed pull request signed off by
  the other key's holder; sign the next release with the new key; then remove
  the old line in a second reviewed pull request. Validators pick the change
  up with the checkout they already update for each release.
- **Lost or compromised key:** remove its line at once in a reviewed pull
  request and cut a release signed by the remaining key, so validators stop
  trusting tags the lost key could sign. Any tag signed by it after the loss
  must not be fetched; announce the removal with the release.
- Never let the file reach zero lines while a release is pending: the fetch
  script then fails closed, and no validator can verify a new entry.
- **Lists follow the release, not the checkout.** The list signers are compiled
  into the validator release and into the measured miner image (the pinned
  copy above). Adding or removing a key therefore changes whose lists are
  accepted only from the release, and the image, that carries the new file.
  After a lost key: remove the line and both pins, release, and sign the next
  list (a higher serial) with the remaining key. A list is adopted only at a
  higher serial than the one held, so an older list cannot come back.
