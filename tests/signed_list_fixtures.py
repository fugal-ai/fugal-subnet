"""Throwaway signers and list bytes for the signed-list tests.

The signer here builds an OpenSSH signature (PROTOCOL.sshsig) on its own, with
no code from fugal_subnet.menu_rules, so the tests check the verifier against
an independent writer. Keys are generated per test and written, as OpenSSH
files, to the temporary directory the test passes in.
"""
import base64
import hashlib
import os

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from fugal_subnet import menu_rules as mr

NS = mr.LIST_NAMESPACE


def _string(b: bytes) -> bytes:
    return len(b).to_bytes(4, "big") + b


class Signer:
    """One throwaway Ed25519 key, as OpenSSH files in a temporary directory."""

    def __init__(self, directory, name):
        self.key = Ed25519PrivateKey.generate()
        self.path = directory / name
        self.path.write_bytes(self.key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.OpenSSH,
            serialization.NoEncryption()))
        os.chmod(self.path, 0o600)
        self.public_line = self.key.public_key().public_bytes(
            serialization.Encoding.OpenSSH, serialization.PublicFormat.OpenSSH).decode()
        (directory / f"{name}.pub").write_text(self.public_line + "\n")
        self.raw = self.key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)

    def signers_line(self, namespaces=NS, principal="release@example.test"):
        options = f'namespaces="{namespaces}" ' if namespaces is not None else ""
        return f"{principal} {options}{self.public_line}\n"

    def blob(self, payload: bytes, namespace: str = NS, hash_name: str = "sha512",
             reserved: bytes = b"", version: int = 1, tail: bytes = b"") -> bytes:
        digest = hashlib.new(hash_name, payload).digest()
        signed = (b"SSHSIG" + _string(namespace.encode()) + _string(reserved)
                  + _string(hash_name.encode()) + _string(digest))
        public = _string(b"ssh-ed25519") + _string(self.raw)
        signature = _string(b"ssh-ed25519") + _string(self.key.sign(signed))
        return (b"SSHSIG" + version.to_bytes(4, "big") + _string(public)
                + _string(namespace.encode()) + _string(reserved)
                + _string(hash_name.encode()) + _string(signature) + tail)

    def sign(self, payload: bytes, **kw) -> bytes:
        return armor(self.blob(payload, **kw))


def armor(blob: bytes) -> bytes:
    b64 = base64.b64encode(blob).decode()
    lines = [b64[i:i + 70] for i in range(0, len(b64), 70)]
    return ("-----BEGIN SSH SIGNATURE-----\n" + "\n".join(lines)
            + "\n-----END SSH SIGNATURE-----\n").encode()


def allowlist_bytes(serial=1, labs=("deepseek", "openai"), hosts=("azure", "baidu", "openai"),
                    first_party=None):
    first_party = {"openai": ("azure",)} if first_party is None else first_party
    return mr.canonical_list_bytes(mr.KIND_MENU_ALLOWLIST, serial, labs=labs, hosts=hosts,
                                   first_party=first_party)


def device_bytes(serial=1, devices=("ab" * 32, "cd" * 32)):
    return mr.canonical_list_bytes(mr.KIND_DEVICE_LIST, serial, devices=devices)


def list_in_use_bytes(root) -> bytes:
    """The lab and host list's payload a release uses: the signed list it
    ships (fugal_subnet/signed_lists/menu_allowlist.json), or, before the
    first one was signed, the draft under review (data/menu_allowlist.draft.json)."""
    from pathlib import Path

    shipped = Path(root) / "fugal_subnet" / "signed_lists" / "menu_allowlist.json"
    draft = Path(root) / "data" / "menu_allowlist.draft.json"
    return (shipped if shipped.exists() else draft).read_bytes()
