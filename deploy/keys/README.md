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
held by the project owner. Its line carries `namespaces="git"`, so the key
is accepted for git signatures only. Until that line landed the file was
absent on purpose, so the fetch script failed closed rather than verifying
against nothing; removing the last line restores that state.

The second key was added 2026-11-09: `release2@fugalai.com`, ED25519,
fingerprint `SHA256:ncBkRnJ8aHy3AfChppE9KhATrM8eFkiFTOrxvNyEzpg`, generated
and held on a separate device by the project owner, passphrase-protected,
also `namespaces="git"`. It is the backup: either key alone verifies a tag,
so losing one machine no longer leaves the project unable to sign a release.
Verified before it was added: a `git`-namespace signature by it over a known
message verifies against this file, and the same signature over a different
message does not.

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
