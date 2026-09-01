#!/usr/bin/env python3
"""Compatibility wrapper for the installed ``fugal-train`` command."""

from fugal_subnet.training import main

if __name__ == "__main__":
    raise SystemExit(main())
