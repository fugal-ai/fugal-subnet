#!/usr/bin/env python3
"""Minimal OCI worker. It contains no wallet, secrets, benchmark corpus, or gold cases."""

import json
import os
import re
import sys

MAX_INPUT_BYTES = 1536 * 1024
MAX_OUTPUT_BYTES = 1024 * 1024


def _canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _code_io(payload):
    code = payload["code"]
    function = payload["function"]
    inputs = payload["inputs"]
    namespace = {"__name__": "fugal_candidate", "__file__": "<candidate>"}
    exec(compile(code, "<candidate>", "exec"), namespace, namespace)
    candidate = namespace.get(function)
    if not callable(candidate):
        raise ValueError("candidate function is missing")
    outputs = [candidate(*arguments) for arguments in inputs]
    encoded = _canonical({"outputs": outputs}).encode("utf-8")
    if len(encoded) > MAX_OUTPUT_BYTES:
        raise ValueError("candidate output is oversized")
    os.write(1, encoded + b"\n")


def _symbolic_math(payload):
    from math_verify import parse, verify

    gold_text = payload["gold"]
    reply = payload["reply"]
    try:
        gold = parse("$" + gold_text + "$", parsing_timeout=None)
        prediction = parse(reply, parsing_timeout=None)
        passed = bool(verify(gold, prediction, timeout_seconds=None))
    except Exception:
        passed = False
    if not passed:
        marker = "\\boxed{"
        start = reply.rfind(marker)
        boxed = None
        if start >= 0:
            index = start + len(marker)
            content_start = index
            depth = 1
            while index < len(reply) and depth:
                if reply[index] == "{":
                    depth += 1
                elif reply[index] == "}":
                    depth -= 1
                index += 1
            if depth == 0:
                boxed = reply[content_start:index - 1]
        normalize = lambda value: re.sub(r"\\left|\\right|\\,|\\!|\s+", "", value or "")
        passed = boxed is not None and normalize(boxed) == normalize(gold_text)
    os.write(1, _canonical({"passed": passed}).encode("utf-8") + b"\n")


def main():
    raw = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValueError("worker input is oversized")
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("invalid worker payload")
    if payload.get("operation") == "code_io":
        _code_io(payload)
    elif payload.get("operation") == "symbolic_math":
        _symbolic_math(payload)
    else:
        raise ValueError("unsupported worker operation")


if __name__ == "__main__":
    main()
