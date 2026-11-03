# Release signing trust root

`allowed_signers` is an OpenSSH allowed-signers file: one line per person who
may sign an approved-entry release tag, `<email> <key type> <public key>`.
It is committed here so the trust root lives in the same reviewed history as
the code it approves. `scripts/fetch_approved_entries.py` verifies a tag
against it with `git tag -v` and nothing else.

To add a signer: append their SSH signing public key, in a reviewed pull
request. To sign a tag: `git config gpg.format ssh`, `git config
user.signingkey ~/.ssh/<key>.pub`, then `git tag -s <tag>`.

The first key was added 2026-11-03: the release key `release@fugalai.com`,
ED25519, fingerprint `SHA256:MpaQWZxhrW77AJg8HAR2TxgFrag5HAo+i9oDnUiIBO8`,
held by the project owner. Its line carries `namespaces="git"`, so the key
is accepted for git signatures only. Until that line landed the file was
absent on purpose, so the fetch script failed closed rather than verifying
against nothing; removing the last line restores that state.
