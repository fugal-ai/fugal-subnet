#!/usr/bin/env python3
"""Verify the byte-identical v2 golden vector on the current Python runtime."""

from __future__ import annotations

import platform

from fugal_subnet.v2.golden import assert_golden, golden_sha256


def main() -> int:
    assert_golden()
    print(
        f"v2 golden vector passed on Python {platform.python_version()}: "
        f"{golden_sha256()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
