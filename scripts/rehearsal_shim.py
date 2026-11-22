#!/usr/bin/env python3
"""Run a shipped neuron for the local-chain v4 rehearsal, with the two
stand-ins a local chain cannot do without, and nothing else:

    python scripts/rehearsal_shim.py neurons/validator.py --network ws://... --mock

1. The public store. The root publishes to a GCS bucket (reveal_publisher's
   uploader and menu_publish.GcsStore, through google-cloud-storage or the
   gcloud CLI) and every reader fetches over HTTPS from
   storage.googleapis.com (reveal_sync.GCS, compiled). A rehearsal has no
   bucket, no credentials and no network, so here the library the uploaders
   import first is a local one that writes create-only into
   FUGAL_REHEARSAL_STORE_DIR, and reveal_sync.GCS is the rehearsal's own
   HTTP server (scripts/dress_rehearsal_v4.LocalStore) serving that directory
   the way GCS serves a public bucket (objects, and the JSON listing with
   size and md5Hash). Every upload and read path of the product runs as on
   finney; only where the bytes land changes.
2. The published-set seed (memo_v4.SEED_PATH, compiled). The release's seed
   is observed on the pinned pool through a finney epoch (e00026509), later
   than any epoch a local chain reaches, so no local chain can run v4 on it;
   FUGAL_REHEARSAL_PUBLISHED_SEED names a seed observed on the rehearsal's
   fixture pool through one of the local chain's epochs.
3. The signed lab and host list (menu_rules.SIGNED_LISTS_DIR and the pinned
   signers file). No release ships one yet (the v4 release gate requires
   it), and without one the root's menu step refuses every epoch, so no menu
   ever changes and no commitment carries a menu hash.
   FUGAL_REHEARSAL_SIGNED_LISTS names a directory holding the draft list
   (data/menu_allowlist.draft.json) at serial 1, signed by a throwaway key,
   and FUGAL_REHEARSAL_LIST_SIGNERS that key's signers file.

4. The listing the root reads for a proposal (join_step.LISTING_URL,
   compiled to openrouter.ai): FUGAL_REHEARSAL_LISTING_URL points it at the
   rehearsal's stub, as every TD's listing read already goes to its
   FUGAL_OPENROUTER_BASE.

Two observation hooks, no behaviour: when the SDK's retrying substrate
switches a connection to an archive node after a state-discarded error, a
line saying so goes to stderr (the SDK's own log line does not reach the
neuron's log); and a block-hash read that comes back empty for a block
above the chain's head (a read of a block not yet produced, which the SDK
then caches: tests/v4_world.Chain.future_reads) is noted the same way.

The process then runs the neuron's own file as __main__, with its own
command line. No other module attribute is touched. The gcloud CLI is also
put out of reach (PATH and CLOUDSDK_CONFIG, set by the rehearsal), so a
regression in (1) fails rather than reaching a real bucket.
"""
import os
import runpy
import sys
import types

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class PreconditionFailed(Exception):
    """google.api_core.exceptions.PreconditionFailed: the object exists."""


def _install_store(store_dir: str) -> None:
    def put(bucket: str, key: str, data: bytes, create_only: bool) -> None:
        if not key or key.startswith("/") or ".." in key.split("/"):
            raise ValueError(f"not an object name: {key!r}")
        path = os.path.join(store_dir, bucket, key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "wb") as f:
            f.write(bytes(data))
        if create_only:
            try:
                os.link(tmp, path)            # atomic create-only, as if_generation_match=0
            except FileExistsError:
                raise PreconditionFailed(key) from None
            finally:
                os.remove(tmp)
        else:
            os.replace(tmp, path)

    class Blob:
        def __init__(self, bucket, key):
            self.bucket, self.key = bucket, key

        def upload_from_filename(self, path, content_type=None, if_generation_match=None,
                                 timeout=None):
            with open(path, "rb") as f:
                put(self.bucket, self.key, f.read(), if_generation_match == 0)

        def upload_from_string(self, data, content_type=None, if_generation_match=None,
                               timeout=None):
            put(self.bucket, self.key, data if isinstance(data, bytes) else data.encode(),
                if_generation_match == 0)

    class Bucket:
        def __init__(self, name):
            self.name = name

        def blob(self, key):
            return Blob(self.name, key)

    class Client:
        def __init__(self, *a, **k):
            pass

        def bucket(self, name):
            return Bucket(name)

    google = sys.modules.get("google") or types.ModuleType("google")
    google.__path__ = getattr(google, "__path__", [])
    cloud = types.ModuleType("google.cloud")
    cloud.__path__ = []
    storage = types.ModuleType("google.cloud.storage")
    storage.Client = Client
    cloud.storage = storage
    api_core = types.ModuleType("google.api_core")
    api_core.__path__ = []
    exceptions = types.ModuleType("google.api_core.exceptions")
    exceptions.PreconditionFailed = PreconditionFailed
    api_core.exceptions = exceptions
    google.cloud, google.api_core = cloud, api_core
    sys.modules.update({"google": google, "google.cloud": cloud, "google.cloud.storage": storage,
                        "google.api_core": api_core, "google.api_core.exceptions": exceptions})


def _note_archive_switches() -> None:
    try:
        from async_substrate_interface import substrate_addons as sa
    except ImportError:
        return
    inner = sa.RetrySyncSubstrate._reinstantiate_substrate

    def noted(self, e=None, use_archive=False):
        if use_archive:
            print(f"REHEARSAL: archive node takes this connection (state discarded at "
                  f"{getattr(e, 'block_hash', '?')})", file=sys.stderr, flush=True)
        return inner(self, e, use_archive=use_archive)

    sa.RetrySyncSubstrate._reinstantiate_substrate = noted


def _note_future_reads() -> None:
    try:
        from async_substrate_interface import sync_substrate as ss
    except ImportError:
        return
    inner = ss.SubstrateInterface.get_block_hash

    def noted(self, block_id):
        got = inner(self, block_id)
        if block_id is not None and not got:
            try:
                head = self.get_block_number(self.get_chain_head())
            except Exception:  # noqa: BLE001 - the note is best effort
                head = None
            print(f"REHEARSAL: FUTURE READ block {block_id} answered empty (chain head {head})",
                  file=sys.stderr, flush=True)
        return got

    ss.SubstrateInterface.get_block_hash = noted


def main() -> None:
    script, argv = sys.argv[1], sys.argv[2:]
    sys.path.insert(0, ROOT)
    _install_store(os.environ["FUGAL_REHEARSAL_STORE_DIR"])
    from fugal_subnet import reveal_sync

    reveal_sync.GCS = os.environ["FUGAL_REHEARSAL_STORE_URL"].rstrip("/")
    seed = os.environ.get("FUGAL_REHEARSAL_PUBLISHED_SEED")
    if seed:
        from fugal_subnet import memo_v4

        memo_v4.SEED_PATH = seed
        memo_v4.seed_doc.cache_clear()
    lists = os.environ.get("FUGAL_REHEARSAL_SIGNED_LISTS")
    if lists:
        from pathlib import Path

        from fugal_subnet import menu_rules

        with open(os.environ["FUGAL_REHEARSAL_LIST_SIGNERS"], "rb") as f:
            signers = f.read()
        menu_rules.SIGNED_LISTS_DIR = Path(lists)
        menu_rules.list_signers = lambda: signers
    listing = os.environ.get("FUGAL_REHEARSAL_LISTING_URL")
    if listing:
        from fugal_subnet import join_step

        join_step.LISTING_URL = listing
    _note_archive_switches()
    _note_future_reads()
    sys.argv = [script, *argv]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
