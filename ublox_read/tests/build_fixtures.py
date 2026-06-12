#!/usr/bin/env python3
"""Regenerate the committed RXM-SFRBX test fixtures.

Builds a small, deterministic, multi-constellation sample by keeping the first
few frames of each distinct (gnssId, sigId, message-type) seen in the raw
u-blox logs, then records the current decoder output as a golden file.

The raw ``.ubx`` source logs are large and NOT committed; pass them as
arguments (or rely on the defaults below if they're present locally)::

    python tests/build_fixtures.py [SOURCE.ubx ...]

Outputs (committed):
    tests/fixtures/sfrbx_sample.ubx           - the small binary sample
    tests/fixtures/sfrbx_sample.expected.txt  - golden decoder output
"""
import io
import pathlib
import sys
from contextlib import redirect_stdout

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

import ubxread as ud  # noqa: E402
from pyubx2 import ERR_LOG, UBXReader  # noqa: E402

DEFAULT_SOURCES = [
    REPO / "data-9t-1-nav-20260304T080907.ubx",
    REPO / "data9-nav-20260304T202022.ubx",
]
PER_KEY = 3  # frames kept per distinct (gnssId, sigId, sub-type)

FIXDIR = HERE / "fixtures"
SAMPLE = FIXDIR / "sfrbx_sample.ubx"
GOLDEN = FIXDIR / "sfrbx_sample.expected.txt"


def _subtype(parsed, gid, sid):
    """Cheap message-type label so the sample captures variety."""
    words = ud.sfrbx_data_words(parsed)
    raw = ud.sfrbx_raw_words(parsed)
    try:
        if gid in (0, 5) and sid == 0:
            return ("LNAV", ud.decode_gps_lnav(words)[0])
        if (gid == 0 and sid in (4, 6)) or (gid == 5 and sid in (4, 8)):
            if len(raw) == 10:
                return ("CNAV", ud.decode_gps_cnav(raw)[0])
        if gid == 2 and len(raw) >= 8:
            if sid == 3:
                return ("FNAV", ud.decode_gal_fnav(raw)[0])
            return ("INAV", ud.decode_gal_inav(raw)[0])
        if gid == 3 and sid in (0, 2) and len(raw) == 10:
            if ud._g(parsed, "svId") <= 5:
                return ("D2", ud.decode_bds_d2(raw)[0])
            return ("D1", ud.decode_bds_d1(raw)[0])
        if gid == 3 and sid == 6:
            return ("BC1", ud.decode_bds_cnav1(raw)[0])
        if gid == 3 and sid == 8 and len(raw) == 9:
            return ("BC2", ud.decode_bds_cnav2(raw)[0])
        if gid == 6 and len(raw) == 4:
            return ("GLO", ud.decode_glonass(raw)[0])
        if gid == 1 and len(raw) >= 8:
            return ("SBAS", ud.decode_sbas(raw)[0])
    except Exception:
        return ("ERR", -1)
    return ("RAW", len(words))


def build_sample(sources):
    counts, chunks = {}, []
    for src in sources:
        with open(src, "rb") as stream:
            ubr = UBXReader(stream, protfilter=2, quitonerror=ERR_LOG)
            for raw_bytes, parsed in ubr:
                if getattr(parsed, "identity", None) != "RXM-SFRBX":
                    continue
                gid = ud._g(parsed, "gnssId")
                sid = ud._g(parsed, "sigId")
                key = (gid, sid) + _subtype(parsed, gid, sid)
                if counts.get(key, 0) >= PER_KEY:
                    continue
                counts[key] = counts.get(key, 0) + 1
                chunks.append(bytes(raw_bytes))
    FIXDIR.mkdir(parents=True, exist_ok=True)
    SAMPLE.write_bytes(b"".join(chunks))
    return len(chunks), counts


def build_golden():
    buf = io.StringIO()
    with redirect_stdout(buf):
        ud.decode_sfrbx(str(SAMPLE))
    GOLDEN.write_text(buf.getvalue())
    return len(buf.getvalue().splitlines())


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    sources = [pathlib.Path(a) for a in argv] or DEFAULT_SOURCES
    missing = [str(s) for s in sources if not pathlib.Path(s).exists()]
    if missing:
        sys.exit("missing source log(s): " + ", ".join(missing))

    frames, counts = build_sample(sources)
    lines = build_golden()
    print(f"sample : {frames} frames, {SAMPLE.stat().st_size} bytes, "
          f"{len(counts)} categories")
    print(f"golden : {lines} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
