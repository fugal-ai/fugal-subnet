#!/usr/bin/env python3
"""Protocol v2 flip rehearsal: the in-process harness and the shared probes.

The chain half lives in `scripts/dress_rehearsal.py --scenario v2_flip`: the
shipped neurons against the local Docker chain across an activation block. This
file is the half that needs no chain and no Docker. It drives the real v2
modules (activation, shared grades, update_scores with the E'' miss rule, the
champion seat, the 85/10/5 payout, and the validator's v2 epoch) through a scripted timeline that crosses the flip:

    python scripts/rehearsal_v2.py            # offline, seconds, no spend

Timeline (epoch F is the activation boundary; the slice is 40 questions):

    F-3..F-1  v1 epochs. Five miners serve v1 heads.
    F         the flip. Miners serve new heads (v2 is a new model list, so
              every head is new and its evidence starts from zero). m2 and m3
              arrive later, as an operator who retrains after the flip would.
    F+1       m4 (young, below the accuracy floor) misses: E'' resets it.
    F+2       m1 (the most senior) completes burn-in and takes the empty seat.
    F+5       m2 completes burn-in and takes the seat from m1 by the margin.
    F+8       m3 completes burn-in and takes it from m2: seats 85/10/5.
    F+10      m3 (burned in) misses: the evidence decays (no zeros added), burn-in
              restarts, a ghost bars the seat, the champion share burns.
    F+11      m3 returns and resumes its seat, paid at once without burn-in.

Three validators run it: two with the activation block (the second sees every
proof in the opposite order and is killed mid-epoch at F+8 and restarted from
its persisted state), and a control without it.

SHORTENED FOR THE REHEARSAL, and only here: burn-in is 120 questions (3
epochs at 40) instead of V2_BURN_IN_QUESTIONS (1,500; v1's BURN_IN_QUESTIONS,
3,000, is patched to the same 120), and a take needs 80 paired
questions instead of CHAMPION_MIN_PAIRED_QUESTIONS (1,500). Every other v2
constant is the shipped default.

Every v2 piece is on this branch, so a piece the probe cannot find means a
rename or a break: its checks are SKIPPED with the reason, and a skip fails
the run (dress_rehearsal.Report.summary).
"""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from types import SimpleNamespace

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

# ── which v2 pieces are on this branch ─────────────────────────────────────────

# (name, "module:attribute" that proves it is present, where it comes from)
V2_PIECES = (
    ("activation", "fugal_subnet.benchmarks.slicer:protocol_version_for_boundary",
     "activation block (#133)"),
    ("shared_grades", "fugal_subnet.shared_grades:shared_cell_grades",
     "shared coldkey-median grades (#134)"),
    ("champion", "fugal_subnet.champion:seat_step", "champion scoring module (#136)"),
    ("payout", "fugal_subnet.payout:seat_weights", "85/10/5 payout module (#137)"),
    ("miss_rule", "fugal_subnet.evidence:apply_miss_v2", "E'' miss rule (#138)"),
    ("wiring", "fugal_subnet.v2_epoch:run_v2_epoch",
     "the validator's v2 epoch wiring (#141, fugal_subnet/v2_epoch.py)"),
)


@dataclass(frozen=True)
class Piece:
    name: str
    present: bool
    source: str


def probe_features(repo: str = REPO) -> dict[str, Piece]:
    """What of protocol v2 this checkout contains. Imports only; runs nothing."""
    out: dict[str, Piece] = {}
    for name, target, source in V2_PIECES:
        mod_name, attr = target.split(":")
        try:
            present = hasattr(importlib.import_module(mod_name), attr)
        except ImportError:
            present = False
        out[name] = Piece(name, present, source)
    # The miner image holds its axon until the flip by reading the activation
    # block. A text probe: the miner imports bittensor, which this must not.
    try:
        with open(os.path.join(repo, "neurons", "miner.py"), encoding="utf-8") as f:
            holds = "V2_ACTIVATION_BLOCK" in f.read()
    except OSError:
        holds = False
    out["miner_hold"] = Piece("miner_hold", holds,
                              "the v2 miner image holding its axon until the flip")
    return out


def missing(features: dict[str, Piece], *names: str) -> str | None:
    """None if every named piece is present, else a skip reason naming what is not."""
    gone = [features[n].source for n in names if not features[n].present]
    return ("not on this branch: " + "; ".join(gone)) if gone else None


# ── comparisons shared with the chain scenario ────────────────────────────────

def canonical_digest(obj) -> str:
    """sha256 of canonical JSON, floats by repr so a last-bit difference shows."""
    def canon(x):
        if isinstance(x, float):
            return repr(x)
        if isinstance(x, dict):
            return {str(k): canon(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
        if isinstance(x, (list, tuple, set, frozenset)):
            seq = sorted(x, key=repr) if isinstance(x, (set, frozenset)) else x
            return [canon(v) for v in seq]
        if hasattr(x, "tolist"):
            return canon(x.tolist())
        return x

    payload = json.dumps(canon(obj), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# Keys a reveal gains from the activation boundary. A v1 reveal has none.
V2_REVEAL_KEYS = ("protocol_version", "shared_grades", "champion", "failed")


def reveal_v2_keys(reveal: dict) -> list[str]:
    return sorted(k for k in V2_REVEAL_KEYS if k in reveal)


def comparable_reveal(reveal: dict) -> dict:
    """The part of a reveal two agreeing validators must publish identically.

    Drops `environment`, which legitimately differs (platform, thread counts);
    its score-affecting subset is `consensus_digest`, which is kept.
    """
    return {k: v for k, v in reveal.items() if k != "environment"}


def seat_split(weights: dict, seat_uids: list[int | None], burn_uid: int) -> dict:
    """{'seats': [w of each seat uid], 'burn': w, 'other': w, 'sum': w} from a weight map."""
    w = {int(k): float(v) for k, v in weights.items()}
    seats = [w.get(u, 0.0) if u is not None else 0.0 for u in seat_uids]
    named = {u for u in seat_uids if u is not None} | {burn_uid}
    return {
        "seats": seats,
        "burn": w.get(burn_uid, 0.0),
        "other": sum(v for u, v in w.items() if u not in named),
        "sum": sum(w.values()),
    }


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


# ── the simulated world ───────────────────────────────────────────────────────

MODELS = ("sim/cheap", "sim/mid", "sim/strong")
SKILL = {"sim/cheap": 0.15, "sim/mid": 0.60, "sim/strong": 0.85}
# $/token. strong is 2x mid: at CHAMPION_ACCURACY_VALUE (30) the cost term is
# a few points, so accuracy decides, as it should on a real list.
PRICES = {"sim/cheap": (1e-8, 4e-8), "sim/mid": (1e-7, 4e-7), "sim/strong": (2e-7, 8e-7)}
COMPLETION_TOKENS = {"sim/cheap": 200, "sim/mid": 300, "sim/strong": 400}
BENCH_CATEGORY = {"mmlu": "knowledge", "math": "math", "aime": "hard", "ifeval": "instruction"}
SLICE = 40
BURN_IN = 120              # rehearsal only; shipped V2_BURN_IN_QUESTIONS is 1500 (v1: 3000)
MIN_PAIRED = 80            # rehearsal only; shipped CHAMPION_MIN_PAIRED_QUESTIONS is 1500
BURN_UID = 0
EPOCH_INTERVAL = 3600      # 300 blocks
FLIP = 1000                # epoch index of the activation boundary
V1_EPOCHS = 3
V2_EPOCHS = 13             # F .. F+12
CRASH_AT = FLIP + 8        # val2 is killed mid-epoch here and restarts


def _h(*parts) -> bytes:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()


def correct(qid: str, model: str) -> bool:
    return _h(qid, model)[0] / 255.0 < SKILL[model]


def build_pool(n: int = 800) -> list[dict]:
    benches = sorted(BENCH_CATEGORY)
    return [{
        "question_id": f"sim_{i:04d}",
        "prompt": f"Simulated question {i}: what is {i} + {i}?",
        "gold": str(2 * i),
        "grader_id": "numeric_final",
        "benchmark": benches[i % len(benches)],
        "metadata": {},
    } for i in range(n)]


def route(policy: str, qid: str) -> str:
    if policy.startswith("const:"):
        return policy.split(":", 1)[1]
    if policy.startswith("mix:"):
        return MODELS[_h(policy, qid)[0] % len(MODELS)]
    if policy == "oracle":
        return next((m for m in MODELS if correct(qid, m)), "sim/strong")
    raise ValueError(policy)


@dataclass(frozen=True)
class SimMiner:
    name: str
    uid: int
    coldkey: str
    reg_block: int
    v1_policy: str
    v2_policy: str
    v2_from: int = 0                         # epochs after the flip it starts serving
    misses: frozenset = frozenset()          # absolute epoch indices with no proof

    @property
    def hotkey(self) -> str:
        return f"hk-{self.name}"

    def head(self, epoch: int) -> str | None:
        """The head hash this miner serves in `epoch`, or None (no proof)."""
        if epoch in self.misses:
            return None
        if epoch < FLIP:
            return hashlib.sha256(f"v1|{self.v1_policy}".encode()).hexdigest()
        if epoch < FLIP + self.v2_from:
            return None  # retraining for the new list; v1 heads retire at the flip
        return hashlib.sha256(f"v2|{self.v2_policy}".encode()).hexdigest()

    def policy(self, epoch: int) -> str:
        return self.v1_policy if epoch < FLIP else self.v2_policy

    def commit_epoch(self) -> int:
        # The v2 head is committed in the epoch before it serves.
        return FLIP + self.v2_from - 1

    def committed(self, epoch: int, bpe: int) -> tuple[str, int]:
        """(head hash, block) on chain at this epoch's boundary. A missed epoch
        does not withdraw the commitment; only committing another head does."""
        if epoch <= self.commit_epoch():
            return hashlib.sha256(f"v1|{self.v1_policy}".encode()).hexdigest(), self.reg_block
        return (hashlib.sha256(f"v2|{self.v2_policy}".encode()).hexdigest(),
                self.commit_block(bpe))

    def commit_block(self, bpe: int) -> int:
        # Inside that epoch, miners commit in registration order, so m1
        # commits its head before m5 commits the same bytes.
        return self.commit_epoch() * bpe + 1 + self.reg_block


MINERS = (
    SimMiner("m1", 2, "ck-a", 10, "mix:1", "const:sim/mid"),
    SimMiner("m2", 3, "ck-b", 11, "mix:2", "const:sim/strong", v2_from=3),
    SimMiner("m3", 4, "ck-c", 12, "mix:3", "oracle", v2_from=6,
             misses=frozenset({FLIP + 10})),
    SimMiner("m4", 5, "ck-d", 13, "const:sim/cheap", "const:sim/cheap",
             misses=frozenset({FLIP + 1})),
    # An exact copy of m1's v2 head under another operator, registered later.
    SimMiner("m5", 6, "ck-e", 14, "mix:5", "const:sim/mid"),
)


def epoch_questions(pool: list[dict], epoch: int) -> list[dict]:
    from fugal_subnet.benchmarks.slicer import derive_nonce, epoch_id_for_block, select_slice

    nonce = derive_nonce(epoch_id_for_block(epoch), f"0x{epoch:064x}")
    return select_slice(nonce, pool, size=SLICE)


def proofs_for(epoch: int, questions: list[dict]) -> dict[int, SimpleNamespace]:
    """Verified 'proofs' (what the validator holds after verification), by uid."""
    from fugal_subnet.pricing import policy_cost

    out = {}
    for m in MINERS:
        h = m.head(epoch)
        if h is None:
            continue
        results = []
        for q in questions:
            model = route(m.policy(epoch), q["question_id"])
            results.append(SimpleNamespace(
                question_id=q["question_id"], routed_model=model,
                correct=correct(q["question_id"], model), failed=False,
                prompt_tokens=0, completion_tokens=COMPLETION_TOKENS[model],
                cost=policy_cost(q["prompt"], model, PRICES, COMPLETION_TOKENS[model]),
            ))
        out[m.uid] = SimpleNamespace(scored_results=results, weights_hash=h)
    return out


# ── one simulated validator ───────────────────────────────────────────────────

@contextlib.contextmanager
def patched(obj, **values):
    old = {k: getattr(obj, k) for k in values if hasattr(obj, k)}
    try:
        for k, v in values.items():
            setattr(obj, k, v)
        yield
    finally:
        for k in values:
            if k in old:
                setattr(obj, k, old[k])
            else:
                delattr(obj, k)


def _seat_params():
    from fugal_subnet import config
    from fugal_subnet.champion import SeatParams

    given = dict(
        margin_points=config.CHAMPION_MARGIN, confidence=config.CHAMPION_CONFIDENCE,
        resamples=config.CHAMPION_BOOTSTRAP_RESAMPLES,
        accuracy_value=config.CHAMPION_ACCURACY_VALUE,
        hard_weight=config.CHAMPION_HARD_WEIGHT,
        hard_gate_min_questions=config.CHAMPION_HARD_GATE_MIN_QUESTIONS,
        ghost_off=config.CHAMPION_GHOST_OFF, ghost_retire=config.CHAMPION_GHOST_RETIRE,
        wilson_confidence=config.WILSON_CONFIDENCE,
        min_accuracy=getattr(config, "CHAMPION_MIN_ACCURACY", 0.30),
        min_paired_questions=MIN_PAIRED,
    )
    names = {f.name for f in dataclasses.fields(SeatParams)}
    return SeatParams(**{k: v for k, v in given.items() if k in names})


def _burned_in(rec) -> bool:
    ev = getattr(rec, "evidence", None)
    try:
        from fugal_subnet.evidence import burn_in_complete
    except ImportError:  # before the E'' rule: burn-in reads n_total
        return ev is not None and ev.n_total >= BURN_IN
    return burn_in_complete(ev, BURN_IN)


_VALIDATOR_MODULE = None


class SimValidator:
    """The validator's epoch, minus the chain: verified proofs in, state and weights out.

    State is persisted after every committed epoch with the validator's own
    save_state/load_state (and the v2 window/seat with the wiring's helpers
    when present), so a restart reloads exactly what a real one would.
    """

    def __init__(self, name: str, activation: int | None, features: dict[str, Piece],
                 workdir: str, reverse: bool = False):
        self.name = name
        self.activation = activation
        self.f = features
        self.reverse = reverse
        self.path = os.path.join(workdir, name, "state.json")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.outputs: dict[int, dict] = {}
        self.restored_from_disk = False
        self._fresh()

    # -- state ---------------------------------------------------------------
    def _fresh(self):
        from fugal_subnet.scoring import ScoringState

        self.scoring = ScoringState()
        self.seat = None
        self.window = None
        self.copy = None   # copy_rule.CopyState once the wiring runs

    def _validator_module(self):
        """neurons/validator.py, for its own save_state/load_state (no chain import)."""
        global _VALIDATOR_MODULE
        if _VALIDATOR_MODULE is None:
            spec = importlib.util.spec_from_file_location(
                "_rehearsal_validator", os.path.join(REPO, "neurons", "validator.py"))
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            _VALIDATOR_MODULE = mod
        return _VALIDATOR_MODULE

    def persist(self, last_epoch: int):
        v = self._validator_module()
        kwargs = {}
        params = inspect.signature(v.save_state).parameters
        if self.f["wiring"].present and "v2" in params and self.seat is not None:
            from fugal_subnet import copy_rule
            from fugal_subnet.v2_epoch import seat_to_dict
            copy_state = self.copy or copy_rule.CopyState()
            kwargs = {"v2": {"seat": seat_to_dict(self.seat),
                             copy_rule.REVEAL_KEY: copy_state.to_reveal()},
                      "v2_window": self.window}
        v.save_state(self.scoring.records, [], [], last_epoch, None, path=self.path, **kwargs)
        if not kwargs:
            # Before the wiring there is no serializer for the window and the
            # seat; keep an in-memory copy and say so in the restart check.
            self._mem = copy.deepcopy((self.seat, self.window, self.copy))

    def restart(self):
        """Process death: drop everything in memory and reload what is on disk."""
        from fugal_subnet.scoring import MinerRecord, ScoringState

        v = self._validator_module()
        raw = v.load_state(MinerRecord, path=self.path)
        self._fresh()
        self.scoring = ScoringState(records=raw["records"])
        v2raw = raw.get("v2")
        if self.f["wiring"].present and v2raw is not None:
            from fugal_subnet import copy_rule
            from fugal_subnet.config import CHAMPION_WINDOW_EPOCHS
            from fugal_subnet.v2_epoch import load_window, seat_from_dict

            self.seat = seat_from_dict(v2raw.get("seat"))
            block = v2raw.get(copy_rule.REVEAL_KEY)
            self.copy = copy_rule.CopyState.from_reveal(block) if block else None
            self.window = load_window(self.path + ".v2window.npz", CHAMPION_WINDOW_EPOCHS)
            self.restored_from_disk = True
        elif hasattr(self, "_mem"):
            self.seat, self.window, self.copy = copy.deepcopy(self._mem)

    # -- one epoch -----------------------------------------------------------
    def digest(self) -> str:
        from fugal_subnet import config
        from fugal_subnet.fingerprint import consensus_digest

        with patched(config, V2_ACTIVATION_BLOCK=self.activation):
            return consensus_digest()

    def run_epoch(self, epoch: int, pool: list[dict], commit: bool = True) -> dict:
        import numpy as np

        from fugal_subnet import scoring
        from fugal_subnet.benchmarks.slicer import (
            blocks_per_epoch,
            epoch_id_for_block,
            protocol_version_for_boundary,
        )
        from fugal_subnet.frontier import build_frontier
        from fugal_subnet.head_eval import HeadScore
        from fugal_subnet.pricing import question_input_tokens
        from fugal_subnet.reference_frame import ReferenceFrame
        from fugal_subnet.rewards import compute_weights

        bpe = blocks_per_epoch(EPOCH_INTERVAL)
        boundary = epoch * bpe
        pv = protocol_version_for_boundary(boundary, self.activation)
        epoch_id = epoch_id_for_block(epoch)
        questions = epoch_questions(pool, epoch)
        proofs = proofs_for(epoch, questions)
        order = sorted(proofs, reverse=self.reverse)
        proofs = {u: proofs[u] for u in order}

        # Evidence, through the real update_scores (E'' from v2 when present).
        ref = sum(_policy_cost(q, "sim/mid") for q in questions)
        epoch_scores = {}
        for uid, p in proofs.items():
            res = p.scored_results
            n_ok = sum(r.correct for r in res)
            epoch_scores[uid] = HeadScore(
                accuracy=n_ok / len(res), cost_efficiency=0.0, kl_score=0.0,
                routing_decisions=np.array([], dtype=np.int32),
                correct_mask=np.array([r.correct for r in res], dtype=bool),
                n_correct=n_ok, n_scored=len(res),
                total_head_cost=sum(r.cost for r in res), total_oracle_cost=ref,
            )
        frontier = build_frontier(
            ReferenceFrame(), PRICES,
            sum(question_input_tokens(q["prompt"]) for q in questions), len(questions), 300.0,
        )
        kwargs = {}
        if "protocol_version" in inspect.signature(scoring.update_scores).parameters:
            kwargs["protocol_version"] = pv
        hotkeys = {m.uid: m.hotkey for m in MINERS}
        with patched(scoring, BURN_IN_QUESTIONS=BURN_IN, V2_BURN_IN_QUESTIONS=BURN_IN):
            state = scoring.update_scores(
                self.scoring, epoch_scores, {u: p.weights_hash for u, p in proofs.items()},
                frontier=frontier, hotkeys=hotkeys, n_questions=len(questions),
                pool_size=len(pool), **kwargs,
            )

        out = {
            "epoch": epoch, "protocol_version": pv, "digest": self.digest(),
            "records": {u: dataclasses.asdict(r) for u, r in sorted(state.records.items())},
        }
        seat, window, copy_state = self.seat, self.window, self.copy
        if pv < 2:
            uids, weights = compute_weights(state.records, set(),
                                            paid_fraction=frontier.confidence,
                                            burn_uid=BURN_UID)
            out["weights"] = dict(zip(uids, weights))
        else:
            out["shared_grades"] = _shared_grades(proofs)
            if self.f["champion"].present and self.f["payout"].present:
                seat, window, copy_state, v2 = self._v2_epoch(epoch, epoch_id, questions,
                                                              proofs, state.records)
                out.update(v2)

        if commit:
            self.scoring, self.seat, self.window, self.copy = state, seat, window, copy_state
            self.outputs[epoch] = out
            self.persist(epoch)
        return out

    def _v2_epoch(self, epoch, epoch_id, questions, proofs, records):
        from fugal_subnet.champion import SeatState, Window

        params = _seat_params()
        seat = self.seat if self.seat is not None else SeatState()
        window = self.window if self.window is not None else Window.empty(
            getattr(importlib.import_module("fugal_subnet.config"), "CHAMPION_WINDOW_EPOCHS", 48))
        from fugal_subnet.benchmarks.slicer import blocks_per_epoch

        by_uid = {m.uid: m for m in MINERS}
        bpe = blocks_per_epoch(EPOCH_INTERVAL)
        # v2 seniority among heads is the commitment block (the first
        # committer of a hash owns it), as the validator reads it from chain.
        seniority = {u: by_uid[u].commit_block(bpe) for u in proofs}
        commit_epoch = {u: by_uid[u].commit_epoch() for u in proofs}
        # Exact copies: a junior uid serving a senior uid's bytes (the
        # validator's dedup flags these; here by hash equality).
        copies = set()
        first_by_hash: dict[str, int] = {}
        for u in sorted(proofs, key=lambda u: (seniority[u], u)):
            h = proofs[u].weights_hash
            if h in first_by_hash:
                copies.add(u)
            else:
                first_by_hash[h] = u

        if self.f["wiring"].present:
            from fugal_subnet import copy_rule
            from fugal_subnet.v2_epoch import run_v2_epoch

            # The boundary snapshot, as the validator reads it from chain:
            # every miner's commitment with its owning coldkey. Ownership,
            # seniority and copies are the copy rule's (copy_rule.classify).
            snapshot = [
                copy_rule.Commitment(m.hotkey, m.coldkey, *m.committed(epoch, bpe))
                for m in MINERS
            ]
            r = run_v2_epoch(
                epoch_index=epoch, epoch_id=epoch_id, boundary_block=epoch * bpe,
                blocks_per_epoch=bpe, questions=questions,
                pool_models=list(MODELS), prices=PRICES, verified_proofs=proofs,
                head_hashes={u: p.weights_hash for u, p in proofs.items()},
                coldkey_of={u: by_uid[u].coldkey for u in proofs},
                hotkey_of={u: by_uid[u].hotkey for u in proofs},
                records=records, commitments=snapshot,
                window=window, seat=seat, copy_state=self.copy or copy_rule.CopyState(),
                burn_uid=BURN_UID, params=params, shares=_shares(),
                burn_in_questions=BURN_IN, bench_category=BENCH_CATEGORY,
            )
            decision, new_window, copy_state = r.decision, r.window, r.copy_state
            weights = dict(zip(r.uids, r.weights))
            path = "validator wiring (v2_epoch.run_v2_epoch)"
        else:
            decision, new_window, weights = _compose_v2(
                epoch, epoch_id, questions, proofs, records, seniority, commit_epoch, copies,
                window, seat, params)
            copy_state = self.copy
            path = "module level (champion.seat_step + payout.seat_weights; wiring absent)"
        st = decision.state
        # Head hash -> the most senior uid serving it (m1 before its copy m5),
        # including heads that are offline this epoch (a ghost, a former seat).
        uid_of: dict[str, int] = {}
        for m in sorted(MINERS, key=lambda m: m.reg_block):
            uid_of.setdefault(hashlib.sha256(f"v2|{m.v2_policy}".encode()).hexdigest(), m.uid)
        v2 = {
            "path": path,
            "event": decision.event,
            "champion": decision.champion,
            "champion_uid": uid_of.get(decision.champion),
            "history_uids": [uid_of.get(h) for h in st.history],
            "ghosts": [(g.head_hash, g.last_epoch, g.reason) for g in st.ghosts],
            "high_water": st.high_water,
            "weights": weights,
            "burned_in": {u: _burned_in(records.get(u)) for u in sorted(proofs)},
        }
        return st, new_window, copy_state, v2


def _policy_cost(q: dict, model: str) -> float:
    from fugal_subnet.pricing import policy_cost

    return policy_cost(q["prompt"], model, PRICES, COMPLETION_TOKENS[model])


def _shares():
    from fugal_subnet.config import PAYOUT_SEAT_SHARES

    return PAYOUT_SEAT_SHARES


def _answers(proofs):
    from fugal_subnet.shared_grades import ScoredAnswer

    names = {f.name for f in dataclasses.fields(ScoredAnswer)}
    coldkey = {m.uid: m.coldkey for m in MINERS}
    for uid, p in proofs.items():
        for r in p.scored_results:
            given = dict(question_id=r.question_id, model=r.routed_model,
                         coldkey=coldkey[uid], correct=bool(r.correct),
                         failed=bool(r.failed), completion_tokens=int(r.completion_tokens))
            yield ScoredAnswer(**{k: v for k, v in given.items() if k in names})


def _shared_grades(proofs) -> list[dict]:
    from fugal_subnet.shared_grades import cells_for_reveal, shared_cell_grades

    cells = shared_cell_grades(_answers(proofs))
    try:
        return cells_for_reveal(cells)
    except TypeError:  # the wiring's variant also publishes tokens
        return cells_for_reveal(cells, {})


def _compose_v2(epoch, epoch_id, questions, proofs, records, seniority, commit_epoch,
                copies, window, seat, params):
    """The v2 epoch from the pure modules, for a branch without the wiring.

    The same composition the wiring performs (one uid per head hash, shared
    cells, the window, seat_step, payout eligibility = the hold rule), kept
    deliberately small. When fugal_subnet.v2_epoch exists the harness uses it
    instead and this is not called.
    """
    import numpy as np

    from fugal_subnet.champion import Candidate, EpochSlice, HeadEpoch, seat_step, window_accuracy
    from fugal_subnet.payout import seat_weights
    from fugal_subnet.shared_grades import shared_cell_grades

    cells = shared_cell_grades(_answers(proofs))
    q, m = len(questions), len(MODELS)
    grades = np.full((q, m), np.nan)
    spends = np.full((q, m), np.nan)
    for i, qq in enumerate(questions):
        for j, model in enumerate(MODELS):
            g = cells.get((qq["question_id"], model))
            if g is not None:
                grades[i, j] = g
                spends[i, j] = _policy_cost(qq, model)
    cats = tuple(BENCH_CATEGORY[qq["benchmark"]] for qq in questions)
    sl = EpochSlice(epoch=epoch, question_ids=tuple(qq["question_id"] for qq in questions),
                    categories=cats, hard=tuple(c == "hard" for c in cats),
                    grades=grades, spends=spends)
    index = {mm: j for j, mm in enumerate(MODELS)}
    serving: dict[str, int] = {}
    for u in sorted(proofs, key=lambda u: (seniority[u], u)):
        serving.setdefault(proofs[u].weights_hash, u)
    answers = {}
    for h, u in serving.items():
        by_q = {r.question_id: r for r in proofs[u].scored_results}
        answers[h] = HeadEpoch(
            routed=np.array([index[by_q[x].routed_model] for x in sl.question_ids], dtype=np.int64),
            failed=np.array([bool(by_q[x].failed) for x in sl.question_ids], dtype=bool),
        )
    window = window.add_epoch(sl, answers)
    names = {f.name for f in dataclasses.fields(Candidate)}
    candidates = {}
    for h, u in serving.items():
        given = dict(head_hash=h, seniority=(seniority[u], h), eligible=u not in copies,
                     since_epoch=commit_epoch[u], burned_in=_burned_in(records.get(u)))
        candidates[h] = Candidate(**{k: v for k, v in given.items() if k in names})
    decision = seat_step(seat, epoch, epoch_id, window, candidates, params=params)
    payable = set()
    for h, c in candidates.items():
        acc, n = window_accuracy(window, h)
        if c.eligible and n > 0 and acc >= getattr(params, "min_accuracy", 0.0):
            payable.add(h)
    uids, weights = seat_weights(decision.champion, decision.state.history, serving, payable,
                                 shares=_shares(), paid_fraction=1.0, burn_uid=BURN_UID)
    return decision, window, dict(zip(uids, weights))


# ── the offline scenario ──────────────────────────────────────────────────────

def _v1_pipeline_digests(activation: int | None) -> dict:
    """check_determinism's real v1 pipeline, in a child process, with or without the block."""
    env = os.environ.copy()
    env.pop("FUGAL_V2_ACTIVATION_BLOCK", None)
    if activation is not None:
        env["FUGAL_V2_ACTIVATION_BLOCK"] = str(activation)
    env["PYTHONHASHSEED"] = "0"
    proc = subprocess.run(
        [sys.executable, os.path.join(REPO, "scripts", "check_determinism.py"), "--child", "7"],
        capture_output=True, text=True, env=env, timeout=600, cwd=REPO,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"check_determinism child failed: {proc.stderr[-800:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _uid_weight(out: dict, uid: int) -> float:
    return float(out.get("weights", {}).get(uid, 0.0))


def _evidence(out: dict, uid: int) -> dict:
    return (out["records"].get(uid) or {}).get("evidence") or {}


def run_offline(report, *, run_v1_pipeline: bool = True) -> None:
    """The v2 flip, in process. Records PASS/FAIL/SKIP rows on `report`."""
    S = "v2_offline"
    from fugal_subnet.benchmarks.slicer import blocks_per_epoch

    f = probe_features()
    print("\n[v2 offline] protocol v2 pieces on this branch:", flush=True)
    for p in f.values():
        print(f"    {'present' if p.present else 'absent '}  {p.name:<14} {p.source}", flush=True)
    if not f["activation"].present:
        report.skip(S, "the whole v2 flip rehearsal", missing(f, "activation"))
        return

    from fugal_subnet.benchmarks.slicer import validate_activation_block

    bpe = blocks_per_epoch(EPOCH_INTERVAL)
    act = FLIP * bpe
    ok_valid = True
    try:
        validate_activation_block(act, bpe)
    except ValueError:
        ok_valid = False
    try:
        validate_activation_block(act + 1, bpe)
        refused = False
    except ValueError:
        refused = True
    report.check(S, "the activation block is an epoch boundary and a mid-epoch one is refused",
                 ok_valid and refused, f"block {act} (epoch {FLIP}), {act + 1} refused={refused}")

    pool = build_pool()
    epochs = list(range(FLIP - V1_EPOCHS, FLIP + V2_EPOCHS))
    with tempfile.TemporaryDirectory(prefix="fugal-v2-rehearsal-") as work:
        val1 = SimValidator("val1", act, f, work)
        val2 = SimValidator("val2", act, f, work, reverse=True)
        ctl = SimValidator("control", None, f, work)
        crashed = None
        for e in epochs:
            val1.run_epoch(e, pool)
            if e == CRASH_AT:
                # Killed mid-epoch: the epoch was computed but never committed,
                # then the process restarts from what it persisted.
                crashed = val2.run_epoch(e, pool, commit=False)
                val2.restart()
            val2.run_epoch(e, pool)
            if e <= FLIP:
                ctl.run_epoch(e, pool)

        _check_flip(report, S, val1, val2, ctl, epochs)
        _check_restart(report, S, val1, val2, crashed, f)
        if run_v1_pipeline:
            _check_v1_pipeline(report, S, act)
        _check_seat(report, S, val1, f)
        _check_payout(report, S, val1, f)
        _check_miss(report, S, val1, f)


def _check_flip(report, S, val1, val2, ctl, epochs):
    o1, o2, oc = val1.outputs, val2.outputs, ctl.outputs
    versions = [o1[e]["protocol_version"] for e in epochs]
    want = [1 if e < FLIP else 2 for e in epochs]
    report.check(S, "v1 before the activation boundary, v2 from it (both validators)",
                 versions == want and versions == [o2[e]["protocol_version"] for e in epochs],
                 f"flip at epoch {FLIP}: {versions[:V1_EPOCHS + 2]}...")
    report.check(S, "validators with the same block publish the same consensus digest",
                 o1[epochs[0]]["digest"] == o2[epochs[0]]["digest"])
    report.check(S, "a validator without the block shows a different digest from epoch one",
                 oc[epochs[0]]["digest"] != o1[epochs[0]]["digest"])
    v1e = [e for e in epochs if e < FLIP]
    same_v1 = all(
        canonical_digest({k: v for k, v in o1[e].items() if k != "digest"})
        == canonical_digest({k: v for k, v in oc[e].items() if k != "digest"})
        for e in v1e
    )
    report.check(S, "v1 epochs are unchanged by setting the block (records and weights)",
                 same_v1, f"epochs {v1e[0]}..{v1e[-1]} vs the control validator")
    report.check(S, "v1 epochs carry no v2 fields; v2 epochs publish shared grades",
                 all("shared_grades" not in o1[e] for e in v1e)
                 and all(o1[e].get("shared_grades") for e in epochs if e >= FLIP))
    report.check(S, "the control keeps v1 rules at the flip (it is visibly on the wrong side)",
                 oc[FLIP]["protocol_version"] == 1 and o1[FLIP]["protocol_version"] == 2)
    diverged = [e for e in epochs if canonical_digest(o1[e]) != canonical_digest(o2[e])]
    report.check(S, "two validators agree on every epoch across the flip (proofs in "
                    "opposite order)", not diverged,
                 f"diverged at {diverged[:3]}" if diverged else f"{len(epochs)} epochs identical")


def _check_restart(report, S, val1, val2, crashed, f):
    ok = canonical_digest(val1.outputs[CRASH_AT]) == canonical_digest(val2.outputs[CRASH_AT])
    later = all(canonical_digest(val1.outputs[e]) == canonical_digest(val2.outputs[e])
                for e in val1.outputs if e > CRASH_AT)
    how = ("seat and window reloaded from disk (v2_epoch helpers)" if val2.restored_from_disk
           else "records reloaded from disk; seat and window from an in-memory copy "
                "(no serializer before the wiring)")
    report.check(S, "a validator killed mid-epoch resumes identically",
                 ok and later and crashed is not None, f"killed at epoch {CRASH_AT}; {how}")


def _check_v1_pipeline(report, S, act):
    try:
        unset, at = _v1_pipeline_digests(None), _v1_pipeline_digests(act)
    except Exception as e:  # noqa: BLE001 - reported, not raised
        report.check(S, "the real v1 pipeline is identical with the block set", False, str(e))
        return
    stages = [k for k in unset if not k.startswith("_")]
    diff = [k for k in stages if unset[k] != at[k]]
    report.check(S, "the real v1 pipeline (check_determinism) is identical with the block set",
                 not diff, f"{len(stages)} stages; differing: {diff}" if diff
                 else f"{len(stages)} stages identical; digest differs as designed: "
                      f"{unset['_consensus_digest'] != at['_consensus_digest']}")


def _check_seat(report, S, val1, f):
    why = missing(f, "champion", "payout")
    names = ("the seat is empty and the share burns until a head completes burn-in",
             "the first seat goes to the eligible head after burn-in (m1: the only one)",
             "a head below the accuracy floor and an exact copy are never seated or paid",
             "a challenger takes the seat only after its own burn-in, by the margin")
    if why:
        for n in names:
            report.skip(S, n, why)
        return
    o = val1.outputs
    m1, m2, m3, m4, m5 = (m.uid for m in MINERS)
    early = [o[e] for e in (FLIP, FLIP + 1)]
    report.check(S, names[0],
                 all(x["event"] == "empty" and close(_uid_weight(x, BURN_UID), 1.0) for x in early),
                 f"{[(x['epoch'], x['event'], x['weights']) for x in early]}")
    first = o[FLIP + 2]
    report.check(S, names[1],
                 first["event"] == "first" and first["champion_uid"] == m1,
                 f"epoch {FLIP + 2}: event={first['event']} champion uid={first['champion_uid']} "
                 f"(expected m1 uid {m1}; m5 serves the same bytes, m4 is below the floor)"
                 f" [{first['path']}]")
    v2e = [e for e in o if e >= FLIP]
    never = all(_uid_weight(o[e], m4) == 0.0 and _uid_weight(o[e], m5) == 0.0
                and o[e].get("champion_uid") not in (m4, m5) for e in v2e)
    report.check(S, names[2], never, f"m4 uid {m4} (accuracy {SKILL['sim/cheap']}), "
                                     f"m5 uid {m5} (copy of m1)")
    t2, t3 = o[FLIP + 5], o[FLIP + 8]
    before = [o[e].get("champion_uid") for e in (FLIP + 3, FLIP + 4, FLIP + 6, FLIP + 7)]
    report.check(S, names[3],
                 t2["event"] == "take" and t2["champion_uid"] == m2
                 and t3["event"] == "take" and t3["champion_uid"] == m3
                 and before == [m1, m1, m2, m2],
                 f"F+5: {t2['event']} -> uid {t2['champion_uid']}; F+8: {t3['event']} -> "
                 f"uid {t3['champion_uid']}; seats before burn-in: {before}")


def _check_payout(report, S, val1, f):
    why = missing(f, "champion", "payout")
    names = ("payout weights sum to 1 in every v2 epoch",
             "one champion: 100%; two: 90/10; three: 85/10/5 (empty seats to the champion)")
    if why:
        for n in names:
            report.skip(S, n, why)
        return
    o = val1.outputs
    m1, m2, m3 = (m.uid for m in MINERS[:3])
    v2e = sorted(e for e in o if e >= FLIP)
    sums = [round(sum(o[e]["weights"].values()), 12) for e in v2e]
    report.check(S, names[0], all(close(s, 1.0, 1e-9) for s in sums), f"sums={sorted(set(sums))}")
    one = seat_split(o[FLIP + 3]["weights"], [m1], BURN_UID)
    two = seat_split(o[FLIP + 6]["weights"], [m2, m1], BURN_UID)
    three = seat_split(o[FLIP + 9]["weights"], [m3, m2, m1], BURN_UID)
    ok = (close(one["seats"][0], 1.0)
          and close(two["seats"][0], 0.90) and close(two["seats"][1], 0.10)
          and [round(x, 9) for x in three["seats"]] == [0.85, 0.10, 0.05]
          and close(three["burn"], 0.0))
    report.check(S, names[1], ok, f"one={one['seats']} two={two['seats']} three={three['seats']}")


def _check_miss(report, S, val1, f):
    why = missing(f, "miss_rule")
    names = ("E'': a miss before burn-in resets the evidence",
             "E'': a miss after burn-in decays the evidence (no zeros added) and restarts burn-in")
    o = val1.outputs
    m3, m4 = MINERS[2].uid, MINERS[3].uid
    if why:
        for n in names:
            report.skip(S, n, why)
    else:
        ev = _evidence(o[FLIP + 1], m4)
        report.check(S, names[0],
                     ev.get("n_total") == 0 and ev.get("burn_in_n") == 0.0,
                     f"m4 after missing F+1: n_total={ev.get('n_total')} "
                     f"burn_in_n={ev.get('burn_in_n')}")
        from fugal_subnet.config import EVIDENCE_HALF_LIFE
        from fugal_subnet.evidence import decay_factor

        before, after = _evidence(o[FLIP + 9], m3), _evidence(o[FLIP + 10], m3)
        alpha = decay_factor(EVIDENCE_HALF_LIFE)
        report.check(S, names[1],
                     close(after.get("n_total", -1), before.get("n_total", 0) * alpha)
                     and close(after.get("n_correct", -1), before.get("n_correct", 0) * alpha)
                     and after.get("burn_in_n") == 0.0 and before.get("burn_in_n", 0) >= BURN_IN,
                     f"m3: n_total {before.get('n_total')} -> {after.get('n_total')} "
                     f"(x{alpha:.5f}), burn_in_n {before.get('burn_in_n')} -> "
                     f"{after.get('burn_in_n')}")

    names2 = ("a champion that misses leaves a ghost bar and its share burns",
              "the champion resumes on return and is paid at once, before re-burning-in")
    why2 = missing(f, "champion", "payout", "miss_rule")
    if why2:
        for n in names2:
            report.skip(S, n, why2)
        return
    g, r = o[FLIP + 10], o[FLIP + 11]
    m1, m2 = MINERS[0].uid, MINERS[1].uid
    report.check(S, names2[0],
                 g["event"] == "ghost" and g["champion"] is None
                 and any(x[2] == "missed" for x in g["ghosts"])
                 and _uid_weight(g, BURN_UID) >= 0.85 - 1e-9,
                 f"F+10: event={g['event']} ghosts={[(x[1], x[2]) for x in g['ghosts']]} "
                 f"burn={_uid_weight(g, BURN_UID):.2f} weights={g['weights']}")
    split = seat_split(r["weights"], [m3, m2, m1], BURN_UID)
    report.check(S, names2[1],
                 r["champion_uid"] == m3 and not r["burned_in"].get(m3, True)
                 and [round(x, 9) for x in split["seats"]] == [0.85, 0.10, 0.05],
                 f"F+11: event={r['event']} champion uid={r['champion_uid']} "
                 f"burned_in={r['burned_in'].get(m3)} seats={split['seats']}")


def main() -> int:
    import argparse

    from scripts.dress_rehearsal import Report

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-v1-pipeline", action="store_true",
                    help="Skip the two check_determinism child runs (saves ~10s)")
    args = ap.parse_args()
    print("=" * 78)
    print("FUGAL v2 FLIP REHEARSAL (offline) — real v2 modules, no chain, no spend")
    print("=" * 78)
    report = Report(pass_message="the v2 flip holds in process.")
    run_offline(report, run_v1_pipeline=not args.no_v1_pipeline)
    return report.summary()


if __name__ == "__main__":
    raise SystemExit(main())
