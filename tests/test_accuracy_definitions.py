"""There is ONE definition of miner accuracy on the consensus path.

Two exist in this tree and they differ by 7 points of apparent score on real
data:

  proof.accuracy   n_correct / len(scored_results) -- EVERY scored question
                   counts, including ones no model could answer. The validator
                   cannot do otherwise: it holds one proof and does not know
                   what the other models would have said.
  evaluate_head    drops questions no model in the pool answered ("carry no
                   routing signal"), so it divides by the answerable subset.

Scoring uses the first. `_proof_to_head_score` reads accuracy, n_correct and
n_total straight off the proof, and `evaluate_head` has no caller under
fugal_subnet/ or neurons/ -- it is a training and analysis metric.

Mixing them -- numerator over the answerable subset, denominator over all
questions -- inflated a reference model's score from 0.994 to 1.065 in an
experiment harness, found by someone who reached for the function whose NAME
looks like the scoring function.

This pins the RELATIONSHIP rather than banning a call. Banning `evaluate_head`
would forbid one route to the problem and not the problem: someone who needs an
accuracy in a new code path and finds the function banned writes a third
definition inline, and the ban passes green while the tree gets worse. Pinning
the relationship catches a third definition too, survives a rename, and records
the divergence as a measured quantity rather than a prohibition.

It also fails if anyone "fixes" `evaluate_head` to match the proof. That would
be a silent consensus change: SCORE_QUALITY_EXPONENT is calibrated against the
accuracy scale the proof produces.
"""
import numpy as np

from fugal_subnet.config import HEAD_HIDDEN_DIM
from fugal_subnet.head_eval import evaluate_head, load_head_from_npz
from fugal_subnet.soft_targets import compute_soft_targets


def _npz(W, b, models):
    import io
    buf = io.BytesIO()
    np.savez(buf, W=W, b=b, models=np.array(models))
    return buf.getvalue()


def _setup(n_unanswerable, seed=0, n_q=40, n_models=5):
    """A head, and a matrix with exactly `n_unanswerable` all-zero rows."""
    rng = np.random.RandomState(seed)
    models = [f"vendor/m{i}" for i in range(n_models)]
    W = (rng.randn(n_models, HEAD_HIDDEN_DIM) * 0.05).astype(np.float32)
    b = (rng.randn(n_models) * 0.05).astype(np.float32)
    head = load_head_from_npz(_npz(W, b, models))
    hidden = rng.randn(n_q, HEAD_HIDDEN_DIM).astype(np.float32)
    hidden /= np.linalg.norm(hidden, axis=1, keepdims=True)
    matrix = (rng.rand(n_q, n_models) > 0.4).astype(np.int8)
    matrix[matrix.sum(axis=1) == 0] = 1          # start fully answerable
    matrix[:n_unanswerable] = 0                  # then make exactly k impossible
    costs = {m: 0.001 * (i + 1) for i, m in enumerate(models)}
    return head, hidden, matrix, models, compute_soft_targets(matrix), costs


def _proof_accuracy(score, n_q):
    """What proof.accuracy would report: correct over EVERY scored question."""
    return score.n_correct / n_q


def test_the_two_definitions_agree_when_every_question_is_answerable():
    n_q = 40
    score = evaluate_head(*_setup(0, n_q=n_q))
    assert score.n_scored == n_q, (
        "evaluate_head dropped a question when none were unanswerable")
    assert score.accuracy == _proof_accuracy(score, n_q)


def test_evaluate_head_reads_high_by_exactly_the_unanswerable_fraction():
    """The divergence is not a bug in either function -- it is the whole
    difference between them, and it is this size."""
    n_q = 40
    for k in (1, 5, 10):
        score = evaluate_head(*_setup(k, n_q=n_q))
        answerable = n_q - k
        assert score.n_scored == answerable, (
            f"{k} unanswerable questions should leave {answerable} scored, "
            f"got {score.n_scored}")
        proof_acc = _proof_accuracy(score, n_q)
        assert score.accuracy > proof_acc, (
            "evaluate_head must read HIGH -- it divides by a smaller "
            "denominator over the same numerator")
        # The exact relationship, which is the thing being pinned.
        #
        # It is exact -- not approximate, not an inequality -- for ONE reason:
        # a routing decision on an unanswerable question is wrong by
        # construction, because no model is correct there. So the two
        # definitions share the SAME numerator (the same integer n_correct) and
        # differ only in the denominator. Without that fact the identity looks
        # like a coincidence between two ratios, and the temptation on a future
        # failure is to weaken it to `>`. Do not. If this assertion breaks,
        # one of the two denominators moved, and one of them is consensus.
        assert np.isclose(proof_acc, score.accuracy * answerable / n_q), (
            f"the gap is no longer the unanswerable fraction ({k}/{n_q}). "
            "Either evaluate_head's exclusion rule changed, or proof.accuracy's "
            "denominator did -- the second is a consensus change")


def _result(qid, correct, is_exploration=False):
    from fugal_subnet.tee.proof import QuestionResult
    return QuestionResult(
        question_id=qid, routed_model="vendor/m0", correct=correct,
        cost_usd=0.001, response_hash="a" * 64,
        prompt_tokens=1, completion_tokens=1, is_exploration=is_exploration,
    )


def test_the_consensus_definition_counts_every_scored_question():
    """The property that matters: two validators must divide by the same thing,
    and the only thing both can see is the proof's own scored_results.

    Asserted BEHAVIOURALLY. An earlier version of this test read
    `proof.accuracy`'s source and looked for "len(scored)" in the text, which
    is the wrong kind of assertion twice over — a cosmetic refactor breaks it
    while nothing has changed, and a real change that keeps the substring
    passes it. The denominator is observable; observe it.
    """
    from fugal_subnet.tee.proof import BenchmarkProof

    # 10 scored questions, 3 correct, plus 2 exploration routes that must not
    # reach either side of the fraction.
    results = [_result(f"q{i}", i < 3) for i in range(10)]
    results += [_result(f"x{i}", True, is_exploration=True) for i in range(2)]
    proof = BenchmarkProof(
        hotkey="5T" + "A" * 46, epoch_id="e1", nonce="ab" * 32,
        questions_hash="c" * 64, weights_hash="d" * 64, source_hash="e" * 64,
        results=results, total_cost_usd=0.012, per_model_costs={"vendor/m0": 0.012},
        attestation_quote=b"", timestamp=1.0,
    )

    assert len(proof.scored_results) == 10
    assert proof.n_correct == 3
    assert proof.accuracy == 0.3, (
        f"accuracy is {proof.accuracy}, not 3/10 — the consensus denominator "
        "changed, which re-scores every miner in the subnet")

    # And the divergence this file exists for: dropping the 7 questions nobody
    # got right would report 3/3 = 1.0 for the same proof.
    unanswerable_style = proof.n_correct / max(1, proof.n_correct)
    assert unanswerable_style == 1.0
    assert proof.accuracy < unanswerable_style


def test_a_misrouted_answerable_question_is_counted_the_same_by_both():
    """The case that makes the relation hold for the RIGHT reason.

    The other tests only ever exercise questions that are impossible (wrong for
    everyone) or routed correctly. Neither distinguishes "wrong because the
    question was impossible" from "wrong because the head misrouted it" -- and
    that distinction is the entire reason `accuracy_on_answerable` exists as a
    separate diagnostic.

    An answerable-but-misrouted question is in BOTH denominators and neither
    numerator, so both definitions treat it identically. That is what leaves
    only the unanswerable rows to separate them. Without this test the suite
    could pass vacuously if a change made every routed model correct, and the
    relation would then hold for a reason narrower than the property wanted.
    """
    n_q = 40
    head, hidden, matrix, models, soft, costs = _setup(5, n_q=n_q)
    score = evaluate_head(head, hidden, matrix, models, soft, costs)

    answerable = matrix.sum(axis=1) > 0
    routed = score.routing_decisions
    misrouted = [
        q for q in range(n_q)
        if answerable[q] and matrix[q][routed[q]] == 0
    ]
    assert misrouted, (
        "no answerable question was misrouted, so this suite is not exercising "
        "the case that distinguishes the two definitions — the relation above "
        "may be holding vacuously")

    # Counted in the denominator by both, in the numerator by neither.
    assert score.n_scored == int(answerable.sum())
    n_right = sum(
        1 for q in range(n_q) if answerable[q] and matrix[q][routed[q]] == 1
    )
    assert score.n_correct == n_right, (
        "a misrouted answerable question leaked into the numerator")
    # And the numerator really is shared: proof.accuracy would report the same
    # integer over a larger denominator, which is the whole divergence.
    assert _proof_accuracy(score, n_q) == n_right / n_q
