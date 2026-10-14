# Release signing trust root

`allowed_signers` is an OpenSSH allowed-signers file: one line per person who
may sign an approved-entry release tag, `<email> <key type> <public key>`.
It is committed here so the trust root lives in the same reviewed history as
the code it approves. `scripts/fetch_approved_entries.py` verifies a tag
against it with `git tag -v` and nothing else.

To add a signer: append their SSH signing public key, in a reviewed pull
request. To sign a tag: `git config gpg.format ssh`, `git config
user.signingkey ~/.ssh/<key>.pub`, then `git tag -s <tag>`.

The file is intentionally absent until the first key is added, so the
fetch script fails closed rather than verifying against nothing.
