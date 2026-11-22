"""The protocol v4 release gate (scripts/check_safety_invariants.py,
`v4_release_blockers`, `--release-v4`).

A release that schedules v4 on finney cannot be cut unless every condition
holds: the block, the signed list, the pinned refusal inputs (with the
published-set seed shipped, matching its pin, and observed no more than
PUBLISHED_V4_SEED_MAX_AGE_EPOCHS epochs before the activation epoch: the
nodes fold the public reveals after it), the frozen finney compose as the
miner reads it, the files in the image, the root's hotkeys. Each test below builds a tree where every
condition holds and breaks exactly one; the gate run on today's tree lists
what is still open, and the plain check does not fail on it.
"""
import hashlib
import importlib.util
import json
import pathlib
import shutil
import types

import pytest

from fugal_subnet import menu_rules as mr
from tests.signed_list_fixtures import Signer, allowlist_bytes, list_in_use_bytes

ROOT = pathlib.Path(__file__).resolve().parents[1]
BPE = 360
BLOCK = 9_600_000 - 9_600_000 % BPE + BPE          # an epoch boundary after every finney block
FIRST = BLOCK // BPE
GATE_KINDS = ("block", "list", "pinned", "compose", "image", "root")


def _checker():
    spec = importlib.util.spec_from_file_location(
        "check_safety_invariants", ROOT / "scripts" / "check_safety_invariants.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cfg(**kw):
    from fugal_subnet import config

    base = dict(
        FINNEY_V2_ACTIVATION_BLOCK=config.FINNEY_V2_ACTIVATION_BLOCK,
        FINNEY_V3_ACTIVATION_BLOCK=config.FINNEY_V3_ACTIVATION_BLOCK,
        FINNEY_ODDS_SEAT_BLOCK=config.FINNEY_ODDS_SEAT_BLOCK,
        FINNEY_MENU_B_BLOCK=config.FINNEY_MENU_B_BLOCK,
        FINNEY_V4_ACTIVATION_BLOCK=BLOCK,
        PROPOSAL_MIN_CATEGORY_ACCURACY_BP_V4=5000,
        CHAMPION_WINDOW_EPOCHS=config.CHAMPION_WINDOW_EPOCHS,
        EPOCH_INTERVAL=4320,
        FINNEY_ROOT_HOTKEYS_V4=((0, config.FINNEY_ROOT_HOTKEY),),
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


def _compose_text(env: dict[str, str]) -> str:
    lines = "\n".join(f"      - {k}={v}" for k, v in env.items())
    return ("services:\n  miner:\n    image: ghcr.io/x/y@sha256:" + "a" * 64 + "\n"
            "    environment:\n" + lines + "\n"
            "    entrypoint:\n      - python\n      - --network\n      - finney\n")


def published_doc(through: int) -> dict:
    """A published-set seed the validator's reader accepts (memo_v4.check_seed),
    observed through epoch `through`."""
    return {"format": "published-v4-v1", "through": f"e{through:08d}",
            "observations_sha256": "0" * 64,
            "pool": {"pool_hash": "0" * 64, "n_questions": 0, "per_set": {}},
            "unpublished": {}}


def seed_list_bytes(serial=1):
    """The reviewed draft's labs and hosts as a list with `serial`: every entry
    of the pinned seed passes it."""
    draft = mr.parse_unsigned_list(list_in_use_bytes(ROOT),
                                   mr.KIND_MENU_ALLOWLIST)
    return mr.canonical_list_bytes(mr.KIND_MENU_ALLOWLIST, serial, labs=draft.labs,
                                   hosts=draft.hosts, first_party=draft.first_party_dict())


GOOD_ENV = {"FUGAL_NETWORK": "finney", "FUGAL_EPOCH_INTERVAL": "4320",
            "FUGAL_V2_ACTIVATION_BLOCK": "9485640", "FUGAL_V3_ACTIVATION_BLOCK": "9527760",
            "FUGAL_V4_ACTIVATION_BLOCK": str(BLOCK)}


class Tree:
    """A release tree where every condition holds: the shipped config and
    miner sources, a signed list under a throwaway signer, the seed naming
    it, the published-set seed through FIRST - 2 (pinned), a frozen finney compose
    equal to the live one, the Dockerfile and .dockerignore."""

    def __init__(self, tmp_path):
        self.root = tmp_path / "tree"
        self.signer = Signer(tmp_path, "release")
        self.signers = self.signer.signers_line().encode()
        self.signers_pin = hashlib.sha256(self.signers).hexdigest()      # fixed when built
        pkg = self.root / "fugal_subnet"
        (pkg / "routing_data").mkdir(parents=True)
        (pkg / "signed_lists").mkdir()
        (self.root / "neurons").mkdir()
        (self.root / "data").mkdir()
        (self.root / "deploy" / "dstack").mkdir(parents=True)
        shutil.copy(ROOT / "fugal_subnet" / "config.py", pkg / "config.py")
        shutil.copy(ROOT / "neurons" / "miner.py", self.root / "neurons" / "miner.py")
        shutil.copy(ROOT / "Dockerfile", self.root / "Dockerfile")
        (pkg / "tee").mkdir()
        # The shipped provisioning channel and the miner's v4 options: every
        # operator's own option has a path into a TD (_td_option_gaps).
        for name in ("provision.py", "v4_options.py"):
            shutil.copy(ROOT / "fugal_subnet" / "tee" / name, pkg / "tee" / name)
        shutil.copy(ROOT / ".dockerignore", self.root / ".dockerignore")
        shutil.copy(ROOT / "fugal_subnet" / "routing_data" / "pool_mix_v4.json",
                    pkg / "routing_data" / "pool_mix_v4.json")
        (pkg / "signed_lists" / "allowed_signers").write_bytes(self.signers)
        shutil.copy(ROOT / "fugal_subnet" / "routing_data" / "menu_v4_seed.json",
                    pkg / "routing_data" / "menu_v4_seed.json")
        self.ship_list(seed_list_bytes(serial=1))
        self.write_published(FIRST - 2)
        self.write_compose(GOOD_ENV)

    def ship_list(self, payload: bytes, signature: bytes | None = None):
        lists = self.root / "fugal_subnet" / "signed_lists"
        name = mr.SHIPPED_LISTS[mr.KIND_MENU_ALLOWLIST]
        (lists / name).write_bytes(payload)
        (lists / f"{name}.sig").write_bytes(signature or self.signer.sign(payload))

    def seed_names(self, serial: int, sha256: str):
        path = self.root / "fugal_subnet" / "routing_data" / "menu_v4_seed.json"
        seed = json.loads(path.read_text())
        seed["allowlists"]["labs"] = {"serial": serial, "sha256": sha256}
        path.write_text(json.dumps(seed))

    def write_published(self, through: int):
        doc = published_doc(through)
        path = self.root / "fugal_subnet" / "routing_data" / "published_v4.json"
        path.write_text(json.dumps(doc))
        self.published_pin = hashlib.sha256(path.read_bytes()).hexdigest()

    def write_compose(self, env: dict[str, str], live: str | None = None, **doc):
        text = _compose_text(env)
        frozen = {"allowed_envs": [], "docker_compose_file": text, **doc}
        d = self.root / "deploy" / "dstack"
        (d / "app-compose.frozen-2099-01-01-finney.json").write_text(json.dumps(frozen))
        (d / "docker-compose.yaml").write_text(text if live is None else live)

    def blockers(self, cfg=None, probe=None):
        def selection(_root, env):
            # The miner's selection under the compose's environment, as the
            # real probe computes it on a finney release that compiles BLOCK.
            v4 = env.get("FUGAL_V4_ACTIVATION_BLOCK")
            if v4 is not None and v4 != str(BLOCK):
                return {"error": "ValueError: FUGAL_V4_ACTIVATION_BLOCK is fixed by the release"}
            return {"protocol": 4, "v4_block": BLOCK}

        return _checker().v4_release_blockers(self.root, cfg or _cfg(), signers=self.signers,
                                              probe=probe or selection,
                                              published_pin=self.published_pin,
                                              signers_pin=self.signers_pin)


@pytest.fixture
def tree(tmp_path):
    return Tree(tmp_path)


def test_a_release_with_every_condition_passes(tree):
    assert tree.blockers() == []


@pytest.mark.parametrize("block, why", [
    (None, "is not set"),
    (BLOCK + 1, "not an epoch boundary"),
    (9_531_000 - 9_531_000 % BPE, "is not after FINNEY_MENU_B_BLOCK"),
])
def test_the_block_must_be_set_aligned_and_after_every_earlier_block(tree, block, why):
    got = tree.blockers(_cfg(FINNEY_V4_ACTIVATION_BLOCK=block))
    assert any(b.startswith("block:") and why in b for b in got), got


def test_no_signed_list_is_a_blocker(tree):
    name = mr.SHIPPED_LISTS[mr.KIND_MENU_ALLOWLIST]
    (tree.root / "fugal_subnet" / "signed_lists" / f"{name}.sig").unlink()
    assert any(b.startswith("list: no signed lab and host list") for b in tree.blockers())


def test_a_list_signed_by_another_key_is_a_blocker(tree, tmp_path):
    other = Signer(tmp_path, "other")
    payload = seed_list_bytes(serial=1)
    tree.ship_list(payload, signature=other.sign(payload))
    assert any(b.startswith("list:") and "does not verify" in b for b in tree.blockers())


def test_the_draft_is_not_a_list(tree):
    tree.ship_list(seed_list_bytes(serial=mr.DRAFT_SERIAL))
    assert any(b.startswith("list:") and "draft" in b for b in tree.blockers())
    tree.ship_list(seed_list_bytes(serial=1))
    (tree.root / "data" / "menu_allowlist.draft.json").write_text("{}")
    assert any("draft.json still ships" in b for b in tree.blockers())


def test_every_seed_entry_must_pass_the_shipped_list(tree):
    # A list without the seed's labs: the seed was not rebuilt under it.
    tree.ship_list(allowlist_bytes(serial=1))
    assert any(b.startswith("list: the pinned seed is not checked under the shipped list")
               for b in tree.blockers())


def test_a_seed_that_names_a_list_must_name_the_shipped_one(tree):
    tree.seed_names(7, "ab" * 32)
    assert any(b.startswith("list: the pinned seed names list serial 7") for b in tree.blockers())
    payload = seed_list_bytes(serial=1)
    tree.seed_names(1, hashlib.sha256(payload).hexdigest())
    assert tree.blockers() == []


def test_an_unpinned_category_floor_is_a_blocker(tree):
    got = tree.blockers(_cfg(PROPOSAL_MIN_CATEGORY_ACCURACY_BP_V4=None))
    assert any(b.startswith("pinned: PROPOSAL_MIN_CATEGORY_ACCURACY_BP_V4") for b in got)


def test_a_summary_window_the_menu_step_refuses_is_a_blocker(tree):
    got = tree.blockers(_cfg(CHAMPION_WINDOW_EPOCHS=60))
    assert any(b.startswith("pinned: the menu step refuses the summary window") for b in got)


def test_the_published_set_seed_may_be_older_than_the_block_within_the_compiled_age(tree):
    """The nodes bring the published set forward from the seed by folding the
    public reveals after it, so a seed observed well before the activation is
    the intended release: anything from the activation epoch minus the
    compiled maximum age to the epoch the first v4 epoch reads passes."""
    age = _checker().PUBLISHED_V4_SEED_MAX_AGE_EPOCHS
    assert age == 150
    for through in (FIRST - 2, FIRST - 40, FIRST - age):
        tree.write_published(through)
        assert tree.blockers() == [], through


def test_the_published_set_seed_must_ship_match_its_pin_and_not_be_too_old(tree):
    age = _checker().PUBLISHED_V4_SEED_MAX_AGE_EPOCHS
    tree.write_published(FIRST - age - 1)
    got = tree.blockers()
    assert any(b.startswith("pinned: the published-set seed is observed through") and
               f"more than {age}" in b for b in got), got
    tree.write_published(FIRST - 1)            # a reveal the first v4 epoch does not read
    got = tree.blockers()
    assert any(b.startswith("pinned: the published-set seed is observed through") and
               "READ_LAG" in b for b in got), got
    tree.write_published(FIRST - 2)
    tree.published_pin = "0" * 64
    assert any(b.startswith("pinned: the published-set seed does not match") for b in
               tree.blockers())
    (tree.root / "fugal_subnet" / "routing_data" / "published_v4.json").unlink()
    got = tree.blockers()
    assert any("published-set seed" in b and "not in this tree" in b for b in got)


def test_the_shipped_published_set_seed_matches_its_pin():
    path = ROOT / "fugal_subnet" / "routing_data" / "published_v4.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == _checker().PINNED_PUBLISHED_V4_SHA256


def test_a_frozen_compose_without_the_v4_block_is_a_blocker(tree):
    env = {k: v for k, v in GOOD_ENV.items() if k != "FUGAL_V4_ACTIVATION_BLOCK"}
    tree.write_compose(env)
    got = tree.blockers()
    assert any(b.startswith("compose:") and "does not set FUGAL_V4_ACTIVATION_BLOCK" in b
               for b in got), got


def test_a_frozen_compose_without_the_v2_block_is_a_blocker(tree):
    # The miner reads protocol v2's block only from its compose.
    tree.write_compose({k: v for k, v in GOOD_ENV.items() if k != "FUGAL_V2_ACTIVATION_BLOCK"})
    assert any("does not set FUGAL_V2_ACTIVATION_BLOCK" in b for b in tree.blockers())


def test_a_frozen_compose_with_another_block_is_a_blocker(tree):
    tree.write_compose({**GOOD_ENV, "FUGAL_V4_ACTIVATION_BLOCK": str(BLOCK + BPE)})
    got = tree.blockers()
    assert any("sets FUGAL_V4_ACTIVATION_BLOCK" in b for b in got)
    assert any("the miner refuses to start" in b for b in got)


def test_a_frozen_compose_on_which_the_miner_serves_v3_is_a_blocker(tree):
    got = tree.blockers(probe=lambda _r, _e: {"protocol": 3, "v4_block": None})
    assert any(b.startswith("compose: under") and "serves protocol 3" in b for b in got)


def test_the_live_compose_must_be_the_frozen_one(tree):
    tree.write_compose(GOOD_ENV, live=_compose_text({**GOOD_ENV, "FUGAL_EXTRA": "1"}))
    assert any("is not app-compose.frozen" in b or "docker-compose.yaml is not" in b
               for b in tree.blockers())


@pytest.mark.parametrize("env, doc, why", [
    ({**GOOD_ENV, "FUGAL_OPENROUTER_BASE": "https://x"}, {}, "FUGAL_OPENROUTER_BASE"),
    ({**GOOD_ENV, "FUGAL_V4_PROPOSAL": "{}"}, {}, "FUGAL_V4_PROPOSAL, one operator's own"),
    (GOOD_ENV, {"allowed_envs": ["FUGAL_V4_CHALLENGE"]}, "allowed_envs"),
])
def test_what_the_compose_checks_forbid_is_a_blocker(tree, env, doc, why):
    tree.write_compose(env, **doc)
    assert any(b.startswith("compose:") and why in b for b in tree.blockers())


def _edit(path, old, new):
    text = path.read_text()
    assert old in text, old
    path.write_text(text.replace(old, new))


def _cannot_set(tree):
    got = [b for b in tree.blockers() if "cannot set" in b]
    assert len(got) <= 1
    return got[0] if got else ""


def test_the_miners_own_v4_options_reach_a_td_in_the_release(tree):
    """The shipped provisioning, options module and miner: no gap."""
    assert _cannot_set(tree) == ""
    gate = _checker()
    assert gate._td_option_gaps(ROOT, {"FUGAL_V4_PROPOSAL", "FUGAL_V4_CHALLENGE"}) == {}


def test_a_receiver_without_the_v4_fields_is_a_blocker(tree):
    # The provisioning channel before protocol v4: five fields.
    (tree.root / "fugal_subnet" / "tee" / "provision.py").write_text(
        "ALLOWED_FIELDS = frozenset({'head_b64', 'hotkey_ss58', 'openrouter_api_key', "
        "'hotkey_keyfile_b64', 'coldkeypub_b64'})\n")
    got = _cannot_set(tree)
    assert "FUGAL_V4_PROPOSAL" in got and "FUGAL_V4_CHALLENGE" in got
    assert "ALLOWED_FIELDS) has no field v4_proposal" in got
    assert "FUGAL_V4_STATE_DIR" not in got and "FUGAL_V4_ANCHOR" not in got


def test_a_future_v4_option_without_a_path_into_a_td_is_a_blocker(tree):
    miner = tree.root / "neurons" / "miner.py"
    scored = '@click.option("--v4-challenge", default=lambda: os.getenv("FUGAL_V4_CHALLENGE")'
    _edit(miner, scored, '@click.option("--v4-later", default=lambda: '
                         'os.getenv("FUGAL_V4_LATER") or None)\n' + scored)
    got = _cannot_set(tree)
    assert got.startswith("compose: a miner in a TD cannot set FUGAL_V4_LATER:")
    assert "FIELDS names no provisioning field" in got
    assert "FUGAL_V4_PROPOSAL" not in got


@pytest.mark.parametrize("where, old, new, why", [
    ("tee/v4_options.py", '"FUGAL_V4_CHALLENGE": "v4_challenge",\n', "",
     "FUGAL_V4_CHALLENGE: tee/v4_options.FIELDS names no provisioning field"),
    ("tee/v4_options.py", "def parse_proposal(", "def parse_proposal_later(",
     "PARSERS has no parser for v4_proposal"),
    ("tee/v4_options.py", "def parse_challenge(", "def parse_challenge_later(",
     "PARSERS has no parser for v4_challenge"),
    ("miner", 'if "FUGAL_V4_CHALLENGE" in given:', 'if "FUGAL_V4_CHALLENGE_" in given:',
     "FUGAL_V4_CHALLENGE: the miner does not apply it from the push"),
    ("miner", "wallet_path=wallet_path, v4_options=pushed,", "wallet_path=wallet_path,",
     "the miner does not hand the pushed options on"),
    ("miner", "v4_options: \"dict | None\" = None):", "v4_extra: \"dict | None\" = None):",
     "the miner does not hand the pushed options on"),
])
def test_each_link_of_the_path_into_a_td_is_required(tree, where, old, new, why):
    path = (tree.root / "neurons" / "miner.py" if where == "miner"
            else tree.root / "fugal_subnet" / where)
    _edit(path, old, new)
    assert why in _cannot_set(tree)


def test_a_v4_file_left_out_of_the_image_is_a_blocker(tree):
    (tree.root / ".dockerignore").write_text("fugal_subnet/signed_lists/\n")
    got = tree.blockers()
    assert any(b.startswith("image:") and "excluded by .dockerignore" in b for b in got)
    (tree.root / ".dockerignore").write_text("")
    (tree.root / "fugal_subnet" / "routing_data" / "pool_mix_v4.json").unlink()
    assert any(b.startswith("image:") and "pool_mix_v4.json" in b for b in tree.blockers())


def test_a_dockerfile_that_does_not_copy_the_package_is_a_blocker(tree):
    (tree.root / "Dockerfile").write_text("FROM scratch\n")
    assert any(b.startswith("image: the Dockerfile") for b in tree.blockers())


HK_A = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
HK_B = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"


@pytest.mark.parametrize("table", [
    (), ((FIRST + 1, HK_A),), ((5, HK_A), (3, HK_B)),
    # two rows at one first epoch (the row in force would be chosen by position)
    ((0, HK_A), (0, HK_B)), ((0, HK_A), (7, HK_B), (7, HK_A)),
    # a hotkey that is not one canonical string
    ((0, " " + HK_A),), ((0, HK_A + " "),), ((0, " "),), ((0, ""),), ((0, HK_A[:-1] + "0"),),
    ((0, HK_A[:10] + " " + HK_A[10:]),), ((0, HK_A + "\n"),), ((0, HK_A[:20]),),
    # a first epoch that is not a non-negative integer, a row of another shape
    ((-1, HK_A),), ((True, HK_A),), ((1.0, HK_A),), (("0", HK_A),), ((0, None),),
    ((0, HK_A, 1),), ("not a table",), (None,),
])
def test_the_root_hotkeys_must_be_a_table_the_menu_reads_can_use(tree, table):
    got = tree.blockers(_cfg(FINNEY_ROOT_HOTKEYS_V4=table))
    assert any(b.startswith("root:") for b in got), got


def test_a_root_table_with_rotated_rows_passes(tree):
    assert tree.blockers(_cfg(FINNEY_ROOT_HOTKEYS_V4=((0, HK_A), (FIRST + 50, HK_B)))) == []
    assert tree.blockers(_cfg(FINNEY_ROOT_HOTKEYS_V4=[[0, HK_A], [FIRST + 50, HK_B]])) == []


@pytest.mark.parametrize("block", ["9600120", True, 9_600_120.0, [9_600_120], b"x"])
def test_a_block_that_is_not_an_integer_is_a_named_blocker_not_a_crash(tree, block):
    got = tree.blockers(_cfg(FINNEY_V4_ACTIVATION_BLOCK=block))
    assert any(b.startswith("block:") and "is not an integer" in b for b in got), got


def test_signers_that_are_not_the_pinned_file_are_a_blocker(tree, tmp_path):
    """The list verifies under whatever signers file ships, so the file itself
    is held to the pin in menu_rules (the gate alone, not the plain check)."""
    other = Signer(tmp_path, "swapped")
    (tree.root / "fugal_subnet" / "signed_lists" / "allowed_signers").write_bytes(
        other.signers_line().encode())
    tree.signers = other.signers_line().encode()
    tree.ship_list(seed_list_bytes(1), other.sign(seed_list_bytes(1)))
    got = tree.blockers()
    assert any(b.startswith("list:") and "allowed_signers is not the signers file" in b
               for b in got), got


def test_a_published_seed_the_validators_reader_refuses_is_a_blocker(tree):
    from fugal_subnet import memo_v4

    path = tree.root / "fugal_subnet" / "routing_data" / "published_v4.json"
    good = published_doc(FIRST - 2)
    bad = {
        "a wrong format": {**good, "format": "nonsense"},
        "no through": {k: v for k, v in good.items() if k != "through"},
        "an unreadable through": {**good, "through": "yesterday"},
        "no pool": {k: v for k, v in good.items() if k != "pool"},
        "a pool without its sets": {**good, "pool": {"pool_hash": "0" * 64}},
        "unpublished ids unsorted": {**good, "pool": {**good["pool"], "per_set": {"a": 2}},
                                     "unpublished": {"a": ["b", "a"]}},
        "unpublished beyond the set": {**good, "pool": {**good["pool"], "per_set": {"a": 1}},
                                       "unpublished": {"a": ["a", "b"]}},
        "an extra key": {**good, "extra": 1},
    }
    for label, doc in bad.items():
        with pytest.raises((ValueError, KeyError, TypeError)):
            memo_v4.check_seed(doc)                  # the reader refuses it
            memo_v4.seed_through(doc)
        path.write_text(json.dumps(doc))
        tree.published_pin = hashlib.sha256(path.read_bytes()).hexdigest()
        got = tree.blockers()
        assert any(b.startswith("pinned: the published-set seed is not a seed the validator "
                                "reads") for b in got), (label, got)
    path.write_text("{")                                # not JSON
    tree.published_pin = hashlib.sha256(path.read_bytes()).hexdigest()
    assert any(b.startswith("pinned: the published-set seed is not a seed") for b in
               tree.blockers())


def test_only_a_lab_the_seed_uses_matters_to_the_list_and_only_a_window_past_the_gap(tree):
    """Two breaks that look like gaps and are not: a shipped list that lacks
    a lab no seed entry uses still holds every entry, and a summary window of
    one epoch is one the menu step accepts. Dropping a lab the seed does use,
    or a window past the gap, is a blocker."""
    draft = mr.parse_unsigned_list(list_in_use_bytes(ROOT),
                                   mr.KIND_MENU_ALLOWLIST)
    seed = json.loads((tree.root / "fugal_subnet" / "routing_data" / "menu_v4_seed.json"
                       ).read_text())
    used = {e["model"].split("/")[0] for e in seed["entries"]}
    unused = sorted(set(draft.labs) - used)
    assert unused and used

    def ship(labs):
        tree.ship_list(mr.canonical_list_bytes(
            mr.KIND_MENU_ALLOWLIST, 1, labs=sorted(labs), hosts=draft.hosts,
            first_party={lab: hs for lab, hs in draft.first_party if lab in labs}))

    ship(set(draft.labs) - {unused[0]})
    assert tree.blockers() == []
    ship(set(draft.labs) - {sorted(used)[0]})
    assert any(b.startswith("list: the pinned seed is not checked under the shipped list")
               for b in tree.blockers())
    ship(draft.labs)
    assert not any(b.startswith("pinned:") for b in tree.blockers(_cfg(CHAMPION_WINDOW_EPOCHS=1)))


def test_a_frozen_compose_with_another_epoch_interval_is_a_compose_blocker(tree):
    tree.write_compose({**GOOD_ENV, "FUGAL_EPOCH_INTERVAL": "3600"})
    got = tree.blockers()
    assert any(b.startswith("compose:") and "FUGAL_EPOCH_INTERVAL=3600" in b for b in got), got


def test_the_gate_today_lists_exactly_the_open_conditions():
    got = _checker().v4_release_blockers()
    kinds = {b.split(":", 1)[0] for b in got}
    assert kinds <= set(GATE_KINDS)
    # The release clears the conditions one at a time: the block is compiled
    # (scripts/set_v4_activation_block.py), then the signed list ships, then
    # the composes are frozen on the release image. Each step's blocker is
    # listed exactly while its step is not done.
    from fugal_subnet import config

    compiled = config.FINNEY_V4_ACTIVATION_BLOCK is not None
    assert any(b.startswith("block: FINNEY_V4_ACTIVATION_BLOCK is not set") for b in got) \
        is not compiled
    ships = (ROOT / "fugal_subnet" / "signed_lists" / "menu_allowlist.json").exists()
    assert any(b.startswith("list: no signed lab and host list") for b in got) is not ships


def test_the_plain_check_does_not_fail_on_the_v4_gate_and_release_v4_does(capsys):
    mod = _checker()
    mod.main([])
    assert "Protocol v4 release gate:" in capsys.readouterr().out
    if not mod.v4_release_blockers():
        return                                    # the release tree: --release-v4 passes
    with pytest.raises(SystemExit) as e:
        mod.main(["--release-v4"])
    assert "v4 release gate: " in str(e.value)


def test_the_probe_runs_the_miners_selection_on_the_shipped_code():
    # With no finney block compiled the miner serves protocol 3 under a
    # compose without the variable; with one compiled it serves protocol 4
    # from that block. Either way a compose that sets another v4 block on
    # finney is refused by the miner itself.
    from fugal_subnet import config

    compiled = config.FINNEY_V4_ACTIVATION_BLOCK
    assert compiled != BLOCK
    mod = _checker()
    env = dict(GOOD_ENV)
    env.pop("FUGAL_V4_ACTIVATION_BLOCK")
    assert mod._miner_selection(ROOT, env)["protocol"] == (3 if compiled is None else 4)
    got = mod._miner_selection(ROOT, GOOD_ENV)
    assert "FUGAL_V4_ACTIVATION_BLOCK" in got.get("error", ""), got
    if compiled is not None:
        got = mod._miner_selection(ROOT, {**GOOD_ENV, "FUGAL_V4_ACTIVATION_BLOCK": str(compiled)})
        assert got.get("protocol") == 4 and got.get("v4_block") == compiled, got


def test_the_compose_variables_are_derived_from_the_miners_protocol_selection():
    mod = _checker()
    got = mod._protocol_selection_variables((ROOT / "fugal_subnet" / "config.py").read_text())
    assert ("FINNEY_V4_ACTIVATION_BLOCK", "FUGAL_V4_ACTIVATION_BLOCK") in got
    assert ("FINNEY_V2_ACTIVATION_BLOCK", "FUGAL_V2_ACTIVATION_BLOCK") in got
    assert ("FINNEY_V3_ACTIVATION_BLOCK", "FUGAL_V3_ACTIVATION_BLOCK") in got


def test_the_list_in_use_admits_every_entry_of_the_pinned_seed():
    """The shipped signed list when one ships, else the reviewed draft:
    every entry of the pinned seed passes it, so the first signed list is
    adopted at the first step and never held back for leaving the menu under
    MENU_MIN_ENTRIES; and every seed entry is within the price ceiling."""
    from fugal_subnet import menu_state as ms

    try:
        allow = mr.shipped_list(mr.KIND_MENU_ALLOWLIST)
    except FileNotFoundError:
        allow = mr.parse_unsigned_list(list_in_use_bytes(ROOT),
                                       mr.KIND_MENU_ALLOWLIST)
    seed = json.loads((ROOT / "fugal_subnet" / "routing_data" / "menu_v4_seed.json").read_text())
    assert len(seed["entries"]) >= ms.MENU_MIN_ENTRIES
    assert [e["id"] for e in seed["entries"] if mr.check_entry(e, allow) is not None] == []
    assert _checker()._seed_strand_blockers(ROOT, None, seed["entries"]) == []


def test_a_list_that_would_strand_the_menu_is_a_blocker(tree):
    from fugal_subnet import menu_state as ms

    draft = mr.parse_unsigned_list(list_in_use_bytes(ROOT),
                                   mr.KIND_MENU_ALLOWLIST)
    seed = json.loads((tree.root / "fugal_subnet" / "routing_data" / "menu_v4_seed.json"
                       ).read_text())
    by_lab: dict[str, int] = {}
    for e in seed["entries"]:
        by_lab[e["model"].split("/")[0]] = by_lab.get(e["model"].split("/")[0], 0) + 1
    # Drop the labs of the most entries until fewer than the minimum are left.
    drop, left = set(), len(seed["entries"])
    for lab, n in sorted(by_lab.items(), key=lambda kv: -kv[1]):
        if left < ms.MENU_MIN_ENTRIES:
            break
        drop.add(lab)
        left -= n
    labs = sorted(set(draft.labs) - drop)
    tree.ship_list(mr.canonical_list_bytes(
        mr.KIND_MENU_ALLOWLIST, 1, labs=labs, hosts=draft.hosts,
        first_party={lab: hs for lab, hs in draft.first_party if lab in labs}))
    got = tree.blockers()
    assert any(b.startswith("list: the shipped list refuses") and "fewer than MENU_MIN_ENTRIES"
               in b for b in got), got
    # Before a list ships, the draft is held to the same.
    name = mr.SHIPPED_LISTS[mr.KIND_MENU_ALLOWLIST]
    (tree.root / "fugal_subnet" / "signed_lists" / name).unlink()
    (tree.root / "data" / "menu_allowlist.draft.json").write_bytes(mr.canonical_list_bytes(
        mr.KIND_MENU_ALLOWLIST, mr.DRAFT_SERIAL, labs=labs, hosts=draft.hosts,
        first_party={lab: hs for lab, hs in draft.first_party if lab in labs}))
    assert any(b.startswith("list: the draft list") and "refuses" in b
               for b in tree.blockers())


def test_a_seed_entry_above_the_price_ceiling_is_a_blocker(tree):
    path = tree.root / "fugal_subnet" / "routing_data" / "menu_v4_seed.json"
    seed = json.loads(path.read_text())
    seed["entries"][0]["price_out"] = mr.MAX_PRICE_OUT_PICO + 1
    path.write_text(json.dumps(seed))
    assert any("priced above the ceiling" in b for b in tree.blockers())
