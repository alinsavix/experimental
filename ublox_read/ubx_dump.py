#!/usr/bin/env python3
"""Dump or split a u-blox UBX log file.

Default (dump) mode prints one line per UBX packet: the message name
(e.g. NAV-PVT) and a short summary of the most useful fields. A per-type
tally is printed at the end.

Split mode (--split-by-date) writes the input out as one file per UTC
calendar date, keeping every navigation epoch intact.

Requires pyubx2 (install into a uv venv: `uv pip install pyubx2`).

Usage:
    python ubx_dump.py FILE [--limit N] [--no-summary] [--include-nmea]
    python ubx_dump.py FILE --split-by-date [--outdir DIR] [--split-boundary eoe|pvt]
    python ubx_dump.py FILE --decode-sfrbx [--gnss gps|glo|...] [--sigid N] [--sv PRN] [--subframe 1-5]
    python ubx_dump.py FILE --decode [--type IDENT ...] [--decode-reserved]

This is a thin launcher; the implementation lives in the ``ubxread`` package
(``ubxread.cli.main``).
"""
import sys

from ubxread.cli import main

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        # Fallback for an interrupt outside the per-mode handlers.
        print("\nAborted.", file=sys.stderr)
        raise SystemExit(130)
