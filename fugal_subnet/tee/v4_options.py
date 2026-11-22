"""Protocol v4: a miner's own options, as its operator sets them for a TD.

A miner running in a TD (the only kind whose proofs a live validator
accepts) cannot read one operator's choice from its compose: the frozen
compose is shared by every miner, its hash is the approved identity, and it
allows no environment of its own (`allowed_envs: []`). Its two v4 options
therefore travel the way the head and the key do: over the attested,
sealed provisioning channel (tee/provision.py), as the fields named in
`FIELDS`, each a string:

    v4_proposal          FUGAL_V4_PROPOSAL: the entry this miner proposes,
                         the JSON object --v4-proposal takes
    v4_challenge         FUGAL_V4_CHALLENGE: the challenge the TD answers
                         each epoch, 0 (none), 500 or 1000 questions [PAID]
    v4_challenge_usd     FUGAL_V4_CHALLENGE_USD: the TD's dollar ceiling on
                         one challenge's realised spend, whole dollars, 1 to
                         config.CHALLENGE_SPEND_MAX_USD_V4 (default
                         config.CHALLENGE_SPEND_USD_V4)

None of them is an input of the grader, the pool, the slice, the menu or
the cost model, and none changes what the TD measures: the compose hash and
the image do not depend on them. Their effect is the documented effect of
the option and nothing else:

- a proposal is a document the miner serves from its box and whose sha256 it
  commits on chain beside its head pointer (chain_proposals, FAST_CROWN_V4.md
  18); the root decides whether it joins, under the list in force and the
  listing it reads. Nothing in the TD is called or spent for it. Here it
  passes every rule that does not depend on a list (`listed_refusal` says,
  for the READY line, whether a given list allows it);
- the challenge size decides whether, and on how many questions, the TD
  answers a challenge (td_v4.ChallengePolicy), which a validator sees in
  the proof; the holder it pairs with is the epoch's, never the
  operator's;
- the dollar ceiling bounds what one challenge spends: the TD makes no
  challenge whose estimate at the epoch's pick costs exceeds it and starts
  no challenge call once the challenge's realised spend plus the estimate
  of its calls in flight reaches it; what that leaves out is in the proof,
  judged by the rules every challenge is.

Each value is checked here strictly (type, size, characters, form, range),
and the receiver refuses a push with any value that does not pass: the TD
then stays unprovisioned, never half-provisioned with a default. A refusal
names the field and the rule, never the value. The same parsers read the
options from the environment (`--v4-proposal` and the rest) for a miner
outside a TD, so the two paths accept the same values.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass

# The operator's v4 options a TD takes over provisioning: {the option's
# environment variable: its provisioning field}. A field is the variable's
# name without FUGAL_ in lower case. scripts/check_safety_invariants.py
# (--release-v4) reads this literal: every FUGAL_V4_ option the miner reads
# must be here (or have a stated image default), be in
# provision.ALLOWED_FIELDS and have a parser in PARSERS.
FIELDS = {
    "FUGAL_V4_PROPOSAL": "v4_proposal",
    "FUGAL_V4_CHALLENGE": "v4_challenge",
    "FUGAL_V4_CHALLENGE_USD": "v4_challenge_usd",
}
# {field: the function in this module that checks and parses its value}.
PARSERS = {
    "v4_proposal": "parse_proposal",
    "v4_challenge": "parse_challenge",
    "v4_challenge_usd": "parse_challenge_usd",
}

# The longest proposal accepted, in characters: a model id (200), two host
# tags (96 each), a reasoning setting and a quantization, as JSON, fit in
# well under half of it.
MAX_PROPOSAL_CHARS = 1024
MAX_CHALLENGE_CHARS = 4
MAX_CHALLENGE_USD_CHARS = 6
# Printable ASCII only: no control character, no byte a terminal or a log
# would read as something else.
_PRINTABLE = re.compile(r"[\x20-\x7e]*")
_PROPOSAL_KEYS = frozenset({"model", "reasoning", "provider_order", "quantization"})


@dataclass(frozen=True)
class ProposalSpec:
    """What an operator proposes: the entry's model, reasoning setting,
    pinned host order and quantization. The miner turns it into its proposal
    document (chain_proposals.ProposalDoc) with its own hotkey."""

    model: str
    reasoning: tuple[str, str | bool]
    provider_order: tuple[str, ...]
    quantization: str

    @property
    def id(self) -> str:
        from fugal_subnet import menu_rules

        return menu_rules.entry_id(self.model, {self.reasoning[0]: self.reasoning[1]})


class OptionRefused(ValueError):
    """An operator's v4 option that does not pass. The message names the
    option and the rule, never the value."""


def _text(value, what: str, limit: int) -> str:
    if type(value) is not str:
        raise OptionRefused(f"{what} must be a string")
    if not value:
        raise OptionRefused(f"{what} is empty; leave the option out instead")
    if len(value) > limit:
        raise OptionRefused(f"{what} is longer than {limit} characters")
    if not _PRINTABLE.fullmatch(value):
        raise OptionRefused(f"{what} holds a character that is not printable ASCII")
    return value


def _object(pairs):
    seen = {}
    for key, value in pairs:
        if key in seen:
            raise OptionRefused("FUGAL_V4_PROPOSAL names a key twice")
        seen[key] = value
    return seen


def _no_constant(name):
    raise OptionRefused("FUGAL_V4_PROPOSAL holds a number JSON does not have")


def proposal_request(spec) -> dict:
    """The request a proposal would join with, the answer limit at its
    compiled cap: what the compiled entry rules read, before any listing is
    read."""
    from fugal_subnet import menu_rules as mr

    return {"model": spec.model, "reasoning": {spec.reasoning[0]: spec.reasoning[1]},
            "provider": {"order": list(spec.provider_order), "allow_fallbacks": False},
            "quantization": spec.quantization, "max_tokens": mr.MAX_ANSWER_TOKENS}


def parse_proposal(text):
    """`ProposalSpec` from a proposal's JSON text, checked strictly: a
    string of printable ASCII of at most MAX_PROPOSAL_CHARS; JSON with no key
    twice and no NaN or infinity; an object of exactly `model`, `reasoning`
    ([key, value]), `provider_order` (a non-empty list of host tags) and
    `quantization`; and every compiled entry rule that does not depend on
    the signed list (menu_rules.check_request: the model's form, the
    reasoning setting, at most MENU_MAX_HOSTS hosts, none twice, no service
    tier, one quantization across them). Whether a list allows its lab and
    hosts is decided by the root at admission, under the list in force there
    (`listed_refusal`). OptionRefused otherwise."""
    from fugal_subnet import menu_rules as mr

    text = _text(text, "FUGAL_V4_PROPOSAL", MAX_PROPOSAL_CHARS)
    try:
        doc = json.loads(text, object_pairs_hook=_object, parse_constant=_no_constant)
    except OptionRefused:
        raise
    except ValueError:
        raise OptionRefused("FUGAL_V4_PROPOSAL is not JSON") from None
    if not isinstance(doc, dict) or set(doc) != _PROPOSAL_KEYS:
        raise OptionRefused("FUGAL_V4_PROPOSAL must be a JSON object with exactly the keys "
                            f"{sorted(_PROPOSAL_KEYS)}")
    reasoning, order = doc["reasoning"], doc["provider_order"]
    if (type(doc["model"]) is not str or type(doc["quantization"]) is not str
            or type(reasoning) is not list or len(reasoning) != 2
            or type(reasoning[0]) is not str or type(reasoning[1]) not in (str, bool)
            or type(order) is not list or not order
            or not all(type(h) is str and h for h in order)):
        raise OptionRefused("FUGAL_V4_PROPOSAL: model and quantization are strings, reasoning "
                            "is [key, value], provider_order a non-empty list of host tags")
    spec = ProposalSpec(model=doc["model"], reasoning=(reasoning[0], reasoning[1]),
                              provider_order=tuple(order), quantization=doc["quantization"])
    # The list-independent rules: under a list that holds exactly this
    # proposal's lab and hosts, every refusal is a rule of the request's own.
    lab = spec.model.split("/", 1)[0]
    own = mr.MenuAllowlist(serial=1, labs=frozenset({lab}),
                           hosts=frozenset(mr.host_identity(h) for h in spec.provider_order),
                           first_party=mr.first_party_map(
                               {lab: [mr.host_identity(h) for h in spec.provider_order]}))
    refusal = mr.check_request(proposal_request(spec), own)
    if refusal is not None:
        raise OptionRefused(f"FUGAL_V4_PROPOSAL fails the compiled entry rule {refusal.code}")
    return spec


def listed_refusal(spec, allowlist) -> str | None:
    """Why `allowlist` (a verified menu_rules.MenuAllowlist, or None) does not
    allow the proposal `spec`, naming the rule (its lab and every host must
    be on it, and a proposal that pins no quantization may name only its
    lab's own host or one the list marks first-party), or None when it
    does. Informational: what the miner shows on its READY line and the
    operator's push script prints. The decision is the root's at admission,
    under the list in force there, and nothing refuses a push for it."""
    from fugal_subnet import menu_rules as mr

    if allowlist is None:
        return "no signed lab and host list"
    refusal = mr.check_request(proposal_request(spec), allowlist)
    return None if refusal is None else f"the compiled entry rule {refusal.code}"


def parse_challenge(text) -> int:
    """The challenge size: exactly the decimal text of one of
    td_v4.CHALLENGE_CHOICES (0, 500 or 1000), as an int. No sign, space,
    leading zero or other number."""
    from fugal_subnet.tee import td_v4

    text = _text(text, "FUGAL_V4_CHALLENGE", MAX_CHALLENGE_CHARS)
    choices = {str(n): n for n in td_v4.CHALLENGE_CHOICES}
    if text not in choices:
        raise OptionRefused("FUGAL_V4_CHALLENGE is one of "
                            + ", ".join(str(n) for n in td_v4.CHALLENGE_CHOICES))
    return choices[text]


def parse_challenge_usd(text) -> int:
    """The TD's dollar budget for one challenge: the decimal text of a whole
    number of dollars from 1 to config.CHALLENGE_SPEND_MAX_USD_V4, as an
    int. No sign, space, leading zero, fraction or exponent."""
    from fugal_subnet import config

    text = _text(text, "FUGAL_V4_CHALLENGE_USD", MAX_CHALLENGE_USD_CHARS)
    top = int(config.CHALLENGE_SPEND_MAX_USD_V4)
    if not re.fullmatch(r"[1-9][0-9]*", text) or int(text) > top:
        raise OptionRefused(f"FUGAL_V4_CHALLENGE_USD is a whole number of dollars from 1 to {top}")
    return int(text)


def check_pushed(values: Mapping[str, str]) -> dict:
    """{field: parsed value} of the v4 fields in a provisioning payload,
    each checked by its parser (`PARSERS`): every rule that does not depend
    on a signed list. No list is read here (see the module's docstring).
    Fields that are not v4 options are not looked at. OptionRefused for the
    first that fails."""
    out = {}
    for field in sorted(set(values) & set(PARSERS)):
        out[field] = globals()[PARSERS[field]](values[field])
    return out


def pushed_options(values: Mapping[str, str]) -> dict[str, str]:
    """{environment variable: value} of the v4 options a push carried
    (`FIELDS`), as the miner applies them in place of its own options."""
    return {var: values[field] for var, field in FIELDS.items() if values.get(field)}
