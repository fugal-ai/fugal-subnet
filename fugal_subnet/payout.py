"""Protocol v2 payout: the miner share split across the last three champions.

From the activation block the miner share (the frame-confidence fraction of
emission, as in v1) is split across SEATS, not across the field:

    seat 0  the current champion          PAYOUT_SEAT_SHARES[0]  (0.85)
    seat 1  the previous champion         PAYOUT_SEAT_SHARES[1]  (0.10)
    seat 2  the one before                PAYOUT_SEAT_SHARES[2]  (0.05)

Two rules:

1. A seat pays only while its head is live this epoch and eligible (it
   verified, passes the quality floor and burn-in, and is not an exact copy).
   Otherwise its share goes to the current champion. With no current
   champion (a ghost bar stands, or nothing beats the virtual champion) the
   champion's share and any share that would have gone to it burn.
2. Seats are paid to the UID serving the seat's head. A hotkey keeps every
   seat it holds; one operator can hold several only by having won each in
   turn, so extra registrations buy nothing.

Seats are earned by promotion, at most three exist, and none can be bought by
registering more hotkeys, so the split does not reward running many UIDs
(I3). The weight-change cap (MAX_WEIGHT_DELTA) still applies to the vector
afterwards, as in v1.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence


def validate_seat_shares(shares: Sequence[float]) -> None:
    """Refuse a seat split that is not a non-increasing distribution."""
    if len(shares) < 1:
        raise ValueError("at least one seat share is required")
    if any(s < 0 for s in shares):
        raise ValueError(f"seat shares must be non-negative, got {tuple(shares)}")
    if abs(sum(shares) - 1.0) > 1e-9:
        raise ValueError(f"seat shares must sum to 1, got {sum(shares)!r}")
    if any(a < b for a, b in zip(shares, shares[1:])):
        raise ValueError(f"seat shares must not increase down the seats, got {tuple(shares)}")


def seat_weights(
    champion: str | None,
    history: Sequence[str],
    serving_uid: Mapping[str, int],
    eligible: set[str] | frozenset[str],
    *,
    shares: Sequence[float],
    paid_fraction: float,
    burn_uid: int,
) -> tuple[list[int], list[float]]:
    """The v2 weight vector: (uids, weights), summing to 1, unpaid weight to burn_uid.

    `champion` is the current champion's head hash (None when the seat is
    empty); `history` the former champions, most recent first. `serving_uid`
    maps each head verified this epoch to the UID that served it; `eligible`
    is the set of those heads that pass the gates.
    """
    validate_seat_shares(shares)
    paid_fraction = min(1.0, max(0.0, float(paid_fraction)))
    burn_uid = int(burn_uid)

    seats: list[str | None] = [champion]
    for h in history:
        if len(seats) >= len(shares):
            break
        if h != champion and h not in seats:
            seats.append(h)
    # A seat no champion has filled yet (fewer than three champions so far)
    # is an empty seat, and its share goes to the current champion.
    seats += [None] * (len(shares) - len(seats))

    def paying(h: str | None) -> int | None:
        if h is None or h not in eligible or h not in serving_uid:
            return None
        uid = int(serving_uid[h])
        return None if uid == burn_uid else uid

    champion_uid = paying(champion)
    weights: dict[int, float] = {}
    for i, h in enumerate(seats):
        share = paid_fraction * float(shares[i])
        uid = paying(h)
        target = uid if uid is not None else champion_uid
        if target is None:
            continue  # burns
        weights[target] = weights.get(target, 0.0) + share

    assigned = sum(weights.values())
    burn = 1.0 - assigned
    if burn > 1e-12 or not weights:
        weights[burn_uid] = weights.get(burn_uid, 0.0) + max(burn, 0.0)
    uids = sorted(weights)
    return uids, [weights[u] for u in uids]
