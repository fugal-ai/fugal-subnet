"""The benchmark pool must be derived by exactly one function.

The pool is consensus state. When it has two code paths, a change lands on one
of them and the two sides disagree — with a questions_hash mismatch that names
neither the pool nor the path. That is not hypothetical: the miner's
--benchmark-pool flag json.load()ed the file directly, so when the loader
learned to drop questions the harness cannot grade, the validator served 125
questions against the miner's 150 and the live subnet set no weights for three
epochs before anyone noticed.

Structural, because a behavioural test cannot catch it: a test that loads the
pool once and hands it to both sides agrees with itself by construction. This is
the same reason the epoch id has a structural check.
"""
import ast
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent


def _neuron_sources():
    for name in ("miner.py", "validator.py"):
        path = REPO / "neurons" / name
        yield name, ast.parse(path.read_text(encoding="utf-8"))


def test_both_neurons_get_their_pool_from_load_all():
    for name, tree in _neuron_sources():
        calls = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "load_all" in calls, (
            f"neurons/{name} does not call load_all() — if it derives the pool "
            "some other way, the two neurons can disagree about consensus state"
        )


def test_no_neuron_parses_a_pool_file_itself():
    """json.load on an open file inside a neuron is the shape of the bug."""
    offenders = []
    for name, tree in _neuron_sources():
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if node.func.attr != "load":
                continue
            if getattr(node.func.value, "id", "") != "json":
                continue
            # json.load(f) where f came from open() is pool-shaped; json.load of
            # a state file is fine, so only flag it inside a `with open(...)`.
            offenders.append(f"{name}:{node.lineno}")
    # The validator legitimately reads its own state file this way; the pool
    # must not be among them. Assert on the count being zero in the miner, which
    # is where the second path lived.
    miner_offenders = [o for o in offenders if o.startswith("miner.py")]
    assert not miner_offenders, (
        "neurons/miner.py parses a JSON file directly at "
        f"{miner_offenders} — if that is the benchmark pool it is a second "
        "code path for consensus state. Route it through load_all()."
    )
