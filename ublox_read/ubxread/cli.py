"""Command-line interface: packet dump, date split, and message decoding.

Two decode paths share one rich, columnar output format:
  * RXM-SFRBX nav subframes are dispatched per constellation (see the
    ``_h_*`` handlers / :func:`decode_sfrbx`);
  * every other message type is decoded via the per-identity registry in
    :mod:`ubxread.messages` (e.g. MON-HW, MON-HW2).  :func:`decode_messages`
    drives both and falls back to the one-line :func:`summarize` for any
    identity without a dedicated decoder.
"""

import argparse
import os
import sys
from collections import namedtuple
from datetime import date, datetime, timedelta, timezone

from pyubx2 import ERR_LOG, UBXReader, sigid2str

from .gps import CNAV_A_REF, CNAV_MSGTYPE, QZSS_CNAV_A_REF, QZSS_CNAV_MSGTYPE, SUBFRAME_NAME, decode_gps_cnav, decode_gps_lnav
from .galileo import GAL_FNAV_PAGETYPE, GAL_WORDTYPE, decode_gal_fnav, decode_gal_inav
from .beidou import BDS_CNAV2_MSGTYPE, BDS_SUBFRAME, decode_bds_cnav1, decode_bds_cnav2, decode_bds_d1, decode_bds_d2
from .glonass import GLO_STRING, decode_glonass
from .sbas import SBAS_MSGTYPE, decode_sbas
from .messages import MSG_DECODERS, Line, Section


GNSS = {0: "GPS", 1: "SBAS", 2: "GAL", 3: "BDS", 4: "IMES", 5: "QZSS", 6: "GLO", 7: "NavIC"}

# constellation name -> gnssId (for the --gnss CLI flag)
GNSS_BY_NAME = {"gps": 0, "sbas": 1, "gal": 2, "bds": 3, "qzss": 5, "glo": 6, "navic": 7}

# Human-readable fix types for NAV-PVT / NAV-STATUS
FIXTYPE = {0: "no-fix", 1: "dead-reckoning", 2: "2D", 3: "3D", 4: "GNSS+DR", 5: "time-only"}


def _g(msg, name, default=None):
    """Safe attribute getter for a parsed UBX message."""
    return getattr(msg, name, default)


def summarize(msg):
    """Return a short, human-readable summary string for a parsed UBX message."""
    ident = msg.identity

    if ident == "NAV-PVT":
        try:
            ts = f"{msg.year:04d}-{msg.month:02d}-{msg.day:02d}T{msg.hour:02d}:{msg.min:02d}:{msg.second:02d}"
        except Exception:
            ts = "?"
        fix = FIXTYPE.get(_g(msg, "fixType"), _g(msg, "fixType"))
        return (f"{ts} fix={fix} numSV={_g(msg,'numSV')} "
                f"lat={_g(msg,'lat'):.7f} lon={_g(msg,'lon'):.7f} "
                f"hMSL={_g(msg,'hMSL',0)/1000:.1f}m")

    if ident == "NAV-POSLLH":
        return (f"iTOW={_g(msg,'iTOW',0)/1000:.3f}s lat={_g(msg,'lat'):.7f} "
                f"lon={_g(msg,'lon'):.7f} hMSL={_g(msg,'hMSL',0)/1000:.1f}m")

    if ident in ("NAV-POSECEF", "NAV-HPPOSECEF"):
        return (f"iTOW={_g(msg,'iTOW',0)/1000:.3f}s "
                f"ecefX={_g(msg,'ecefX')} ecefY={_g(msg,'ecefY')} ecefZ={_g(msg,'ecefZ')}")

    if ident == "NAV-HPPOSLLH":
        return f"iTOW={_g(msg,'iTOW',0)/1000:.3f}s lat={_g(msg,'lat'):.7f} lon={_g(msg,'lon'):.7f}"

    if ident == "NAV-STATUS":
        return (f"iTOW={_g(msg,'iTOW',0)/1000:.3f}s gpsFix={FIXTYPE.get(_g(msg,'gpsFix'),_g(msg,'gpsFix'))} "
                f"ttff={_g(msg,'ttff',0)}ms")

    if ident == "NAV-DOP":
        return (f"iTOW={_g(msg,'iTOW',0)/1000:.3f}s pDOP={_g(msg,'pDOP')} "
                f"hDOP={_g(msg,'hDOP')} vDOP={_g(msg,'vDOP')}")

    if ident == "NAV-SAT":
        return f"iTOW={_g(msg,'iTOW',0)/1000:.3f}s numSvs={_g(msg,'numSvs')}"

    if ident == "NAV-SIG":
        return f"iTOW={_g(msg,'iTOW',0)/1000:.3f}s numSigs={_g(msg,'numSigs')}"

    if ident == "NAV-ORB":
        return f"iTOW={_g(msg,'iTOW',0)/1000:.3f}s numSv={_g(msg,'numSv')}"

    if ident == "NAV-CLOCK":
        return f"iTOW={_g(msg,'iTOW',0)/1000:.3f}s clkB={_g(msg,'clkB')}ns clkD={_g(msg,'clkD')}ns/s"

    if ident == "NAV-TIMEUTC":
        try:
            ts = f"{msg.year:04d}-{msg.month:02d}-{msg.day:02d}T{msg.hour:02d}:{msg.min:02d}:{msg.sec:02d}"
        except Exception:
            ts = "?"
        return f"iTOW={_g(msg,'iTOW',0)/1000:.3f}s utc={ts} validUTC={_g(msg,'validUTC')}"

    if ident == "NAV-TIMELS":
        return f"iTOW={_g(msg,'iTOW',0)/1000:.3f}s curLs={_g(msg,'currLs')} srcOfCurrLs={_g(msg,'srcOfCurrLs')}"

    if ident.startswith("NAV-TIME"):
        return f"iTOW={_g(msg,'iTOW',0)/1000:.3f}s week={_g(msg,'week', _g(msg,'galWno', _g(msg,'galWno')))}"

    if ident == "NAV-EOE":
        return f"iTOW={_g(msg,'iTOW',0)/1000:.3f}s (epoch end)"

    if ident == "RXM-RAWX":
        return f"rcvTow={_g(msg,'rcvTow'):.3f}s week={_g(msg,'week')} numMeas={_g(msg,'numMeas')}"

    if ident == "RXM-MEASX":
        return f"gpsTOW={_g(msg,'gpsTOW')} numSv={_g(msg,'numSv')}"

    if ident == "RXM-SFRBX":
        gid = _g(msg, "gnssId")
        sid = _g(msg, "sigId")
        return (f"{GNSS.get(gid, gid)} svId={_g(msg,'svId')} "
                f"sigId={sid} ({sigid2str(gid, sid)}) "
                f"numWords={_g(msg,'numWords')} (nav subframe)")

    if ident == "MON-HW":
        return (f"noisePerMS={_g(msg,'noisePerMS')} agcCnt={_g(msg,'agcCnt')} "
                f"jammingState={_g(msg,'jammingState')}")

    if ident == "MON-RF":
        return f"nBlocks={_g(msg,'nBlocks')}"

    if ident == "MON-HW2":
        return (f"I(ofs={_g(msg,'ofsI')},mag={_g(msg,'magI')}) "
                f"Q(ofs={_g(msg,'ofsQ')},mag={_g(msg,'magQ')}) "
                f"cfgSource={_g(msg,'cfgSource')}")

    # Generic fallback: show whichever common scalar fields are present.
    parts = []
    for f in ("iTOW", "version", "numSV", "numSvs", "numSigs", "numMeas", "numSv", "numWords"):
        v = _g(msg, f)
        if v is not None:
            if f == "iTOW":
                parts.append(f"iTOW={v/1000:.3f}s")
            else:
                parts.append(f"{f}={v}")
    return " ".join(parts) if parts else f"payload={len(msg.payload)}B"


def sfrbx_data_words(msg):
    """Return the list of 24-bit data words from an RXM-SFRBX message."""
    nw = _g(msg, "numWords", 0)
    return [(getattr(msg, f"dwrd_{i:02d}") >> 6) & 0xFFFFFF for i in range(1, nw + 1)]


def sfrbx_raw_words(msg):
    """Return the list of raw 32-bit words from an RXM-SFRBX message."""
    nw = _g(msg, "numWords", 0)
    return [getattr(msg, f"dwrd_{i:02d}") & 0xFFFFFFFF for i in range(1, nw + 1)]


def _fmt(x, unit="", sig=True):
    if isinstance(x, float):
        s = f"{x:.6g}"
    else:
        s = str(x)
    return f"{s}{unit}"


def _columns(cells, indent="      ", maxline=108):
    """Lay out (label, value) cells into aligned columns.

    Picks the largest column count (up to 4) whose rendered width fits maxline,
    aligning the '=' and values within each grid column.
    """
    if not cells:
        return []
    for ncols in range(min(4, len(cells)), 0, -1):
        rows = [cells[i:i + ncols] for i in range(0, len(cells), ncols)]
        name_w = [0] * ncols
        val_w = [0] * ncols
        for r in rows:
            for j, (lab, val) in enumerate(r):
                name_w[j] = max(name_w[j], len(lab))
                val_w[j] = max(val_w[j], len(val))
        width = len(indent) + sum(name_w[j] + 3 + val_w[j] for j in range(ncols)) + 3 * (ncols - 1)
        if width <= maxline or ncols == 1:
            lines = []
            for r in rows:
                parts = [f"{lab:<{name_w[j]}} = {val:<{val_w[j]}}" for j, (lab, val) in enumerate(r)]
                lines.append((indent + "   ".join(parts)).rstrip())
            return lines
    return []


def _emit(header_line, fields):
    """Print a decoded message: a header line plus an aligned field grid.

    ``fields`` is a list of (label, value, unit) cells accumulated into a grid,
    optionally interspersed with ``Section``/``Line`` markers that flush the
    current grid and emit a sub-header or a verbatim detail line (used by the
    repeated-group decoders, e.g. MON-RF blocks, MON-HW3 pins).
    """
    print(header_line)
    cells = []

    def flush():
        for line in _columns(cells):
            print(line)
        cells.clear()

    for item in fields:
        if isinstance(item, Section):
            flush()
            print("    " + item.title)
        elif isinstance(item, Line):
            flush()
            print("      " + item.text)
        else:
            fn, val, unit = item
            cells.append((fn, _fmt(val, unit)))
    flush()


# ---------------------------------------------------------------------------
# RXM-SFRBX dispatch: one handler per constellation (keyed by gnssId).  Each
# handler inspects the signal/message and returns a `_Decoded(header, fields)`
# to emit, `_SKIP` to drop the frame silently (filtered out), or None to fall
# through to the raw-words dump.  `header` is the text after the "#<n>  " index.
# ---------------------------------------------------------------------------
_SfrbxCtx = namedtuple("_SfrbxCtx", "gid sid prn words raw tag reserved subframe")
_Decoded = namedtuple("_Decoded", "header fields")
_SKIP = object()


def _h_gps(ctx):
    """GPS / QZSS (gnssId 0 / 5): L1 C/A LNAV, L2C / L5 CNAV."""
    if ctx.sid == 0:                                   # L1 C/A -> LNAV
        sfid, tow_s, fields = decode_gps_lnav(ctx.words, reserved=ctx.reserved)
        if ctx.subframe is not None and sfid != ctx.subframe:
            return _SKIP
        sfname = SUBFRAME_NAME.get(sfid, "?")
        return _Decoded(f"{ctx.tag} LNAV  SF{sfid} {sfname:<13}  "
                        f"TOW={tow_s}s (next subframe)", fields)
    # CNAV rides L2 CM / L5 I; QZSS numbers L5 I as sigId 8.
    is_cnav = (ctx.gid == 0 and ctx.sid in (4, 6)) or \
              (ctx.gid == 5 and ctx.sid in (4, 8))
    if is_cnav and len(ctx.raw) == 10:
        if ctx.subframe is not None:                   # LNAV-only filter active
            return _SKIP
        a_ref = QZSS_CNAV_A_REF if ctx.gid == 5 else CNAV_A_REF
        mtype, tow_s, fields = decode_gps_cnav(
            ctx.raw, reserved=ctx.reserved, a_ref=a_ref)
        mname = CNAV_MSGTYPE.get(mtype) or \
            (QZSS_CNAV_MSGTYPE.get(mtype) if ctx.gid == 5 else None) or "?"
        return _Decoded(f"{ctx.tag} CNAV  MT{mtype} {mname:<26}  TOW={tow_s}s",
                        fields)
    return None


def _h_galileo(ctx):
    """Galileo (gnssId 2): E1-B / E5b-I I/NAV and E5a-I F/NAV pages."""
    if ctx.subframe is not None:
        return _SKIP
    if ctx.sid == 3 and len(ctx.raw) >= 8:             # E5a-I -> F/NAV
        pt, _tow, fields = decode_gal_fnav(ctx.raw)
        pname = GAL_FNAV_PAGETYPE.get(pt, "spare/other")
        return _Decoded(f"{ctx.tag} F/NAV  PT{pt} {pname}", fields)
    if len(ctx.raw) >= 8:                              # E1-B / E5b-I -> I/NAV
        wt, _tow, fields = decode_gal_inav(ctx.raw)
        wname = GAL_WORDTYPE.get(wt, "almanac/other")
        return _Decoded(f"{ctx.tag} I/NAV  WT{wt} {wname}", fields)
    return None


def _h_beidou(ctx):
    """BeiDou (gnssId 3): D1/D2 NAV (B1I/B2I), B-CNAV1 (B1C), B-CNAV2 (B2a)."""
    if ctx.subframe is not None:
        return _SKIP
    if ctx.sid in (0, 2) and len(ctx.raw) == 10:
        if ctx.prn <= 5:                               # GEO satellites use D2
            fraid, pnum, sow, fields = decode_bds_d2(ctx.raw)
            return _Decoded(f"{ctx.tag} D2NAV  SF{fraid} P{pnum:<2}  "
                            f"SOW={sow}s", fields)
        fraid, sow, fields = decode_bds_d1(ctx.raw)    # MEO/IGSO use D1
        sfname = BDS_SUBFRAME.get(fraid, "?")
        return _Decoded(f"{ctx.tag} D1NAV  SF{fraid} {sfname:<18}  "
                        f"SOW={sow}s", fields)
    if ctx.sid == 6:                                   # B1C -> B-CNAV1
        part, crc_ok, fields = decode_bds_cnav1(ctx.raw)
        crc = "" if crc_ok is None else f"  CRC={'OK' if crc_ok else 'BAD'}"
        return _Decoded(f"{ctx.tag} B-CNAV1  {part}{crc}", fields)
    if ctx.sid == 8 and len(ctx.raw) == 9:             # B2a -> B-CNAV2
        mt, crc_ok, sow, _prn, fields = decode_bds_cnav2(ctx.raw)
        mname = BDS_CNAV2_MSGTYPE.get(mt, "?")
        crc = "OK" if crc_ok else "BAD"
        return _Decoded(f"{ctx.tag} B-CNAV2  MT{mt} {mname:<28}  "
                        f"SOW={sow}s CRC={crc}", fields)
    return None


def _h_glonass(ctx):
    """GLONASS (gnssId 6): L1OF / L2OF strings."""
    if ctx.subframe is not None:
        return _SKIP
    if len(ctx.raw) == 4:
        m, _none, fields = decode_glonass(ctx.raw)
        sname = GLO_STRING.get(m, "almanac")
        return _Decoded(f"{ctx.tag} GLO  String {m:<2} {sname}", fields)
    return None


def _h_sbas(ctx):
    """SBAS (gnssId 1): L1 messages."""
    if ctx.subframe is not None:
        return _SKIP
    if len(ctx.raw) >= 8:
        mt, crc_ok, pre, fields = decode_sbas(ctx.raw)
        mname = SBAS_MSGTYPE.get(mt, "?")
        crc = "OK" if crc_ok else "BAD"
        return _Decoded(f"{ctx.tag} SBAS  MT{mt} {mname:<24}  "
                        f"pre=0x{pre:02X} CRC={crc}", fields)
    return None


SFRBX_HANDLERS = {
    0: _h_gps, 5: _h_gps,      # GPS, QZSS
    1: _h_sbas,                # SBAS
    2: _h_galileo,             # Galileo
    3: _h_beidou,              # BeiDou
    6: _h_glonass,             # GLONASS
}


def sfrbx_decode_one(parsed, reserved=False, subframe=None):
    """Decode a single parsed RXM-SFRBX message.

    Returns ``(result, words, tag, gid)`` where ``result`` is a ``_Decoded``,
    ``_SKIP`` (frame filtered out), or ``None`` (no decoder -> raw-words dump).
    Shared by :func:`decode_sfrbx` and the generic :func:`decode_messages`.
    """
    gid = _g(parsed, "gnssId")
    sid = _g(parsed, "sigId")
    prn = _g(parsed, "svId")
    cname = GNSS.get(gid, str(gid))
    signame = sigid2str(gid, sid)
    words = sfrbx_data_words(parsed)
    raw = sfrbx_raw_words(parsed)
    tag = f"{cname} SV {prn:<3} {signame:<7}"

    ctx = _SfrbxCtx(gid, sid, prn, words, raw, tag, reserved, subframe)
    handler = SFRBX_HANDLERS.get(gid)
    result = handler(ctx) if handler else None
    return result, words, tag, gid


def decode_sfrbx(path, gnss_id=None, sigid=None, sv=None, subframe=None, limit=0,
                 reserved=False):
    """Print decoded RXM-SFRBX nav-message contents.

    Decoders implemented:
      * GPS L1 C/A (sigId 0)        -> LNAV subframes 1-5
      * GPS L2C/L5 (sigId 4/6)      -> CNAV (types 10/11/30/33 + clock block)
      * QZSS L1 C/A (sigId 0)       -> LNAV subframes 1-5 (GPS-compatible)
      * QZSS L2C/L5 (sigId 4/8)     -> CNAV (GPS-compatible types + QZSS 60/61)
      * Galileo E1-B/E5b-I (sigId 1/5) -> I/NAV pages (word types 0-10, 16),
                                       CRC-24Q checked
      * Galileo E5a-I   (sigId 3)   -> F/NAV pages 1-6 (eph, clock, iono, BGD,
                                       GST-UTC, almanac), CRC-24Q checked
      * BeiDou B1I/B2I  (sigId 0/2) -> D1 NAV (MEO/IGSO: clock+iono, ephemeris,
                                       almanac, health, UTC), D2 NAV (GEO),
                                       preamble checked
      * BeiDou B1C      (sigId 6)   -> B-CNAV1 SF1/SF2/SF3 (eph+clock, iono, UTC,
                                       EOP, reduced/midi almanac), CRC-24Q checked
      * BeiDou B2a      (sigId 8)   -> B-CNAV2 MT10/11 eph, MT30-34 clock+system
                                       (iono, EOP, UTC, almanac), MT40 midi almanac
      * GLONASS L1OF/L2OF           -> strings 1-15, Hamming checked
      * SBAS L1 (gnssId 1)          -> MT 1/2-5/7/9/10/17/18/25/26, CRC-24Q checked
    Anything without a decoder is shown as a labelled dump of its raw words.

    gnss_id: restrict to this gnssId (None = all). sigid: restrict to this
    sigId (None = all). sv: restrict to this svId. subframe: LNAV subframe id.
    """
    n = 0
    interrupted = False

    try:
        with open(path, "rb") as stream:
            ubr = UBXReader(stream, protfilter=2, quitonerror=ERR_LOG)
            for _raw, parsed in ubr:
                if getattr(parsed, "identity", None) != "RXM-SFRBX":
                    continue
                gid = _g(parsed, "gnssId")
                sid = _g(parsed, "sigId")
                prn = _g(parsed, "svId")
                if gnss_id is not None and gid != gnss_id:
                    continue
                if sigid is not None and sid != sigid:
                    continue
                if sv is not None and prn != sv:
                    continue

                result, words, tag, gid = sfrbx_decode_one(
                    parsed, reserved=reserved, subframe=subframe)

                if result is _SKIP:
                    continue
                if result is None:
                    # Filtered (LNAV-only) or no decoder: nothing, or raw dump.
                    if subframe is not None:
                        continue
                    n += 1
                    note = "D2 NAV (GEO)" if gid == 3 else "no decoder"
                    print(f"#{n}  {tag} numWords={len(words)}  (raw words, {note})")
                    print("      " + " ".join(f"0x{w:06X}" for w in words))
                else:
                    n += 1
                    _emit(f"#{n}  {result.header}", result.fields)

                if limit and n >= limit:
                    break
    except KeyboardInterrupt:
        print(f"\nInterrupted after {n} subframes.", file=sys.stderr)
        interrupted = True

    print(f"\nDecoded {n} RXM-SFRBX subframe(s).", file=sys.stderr)
    return 130 if interrupted else 0


def decode_messages(path, types=None, limit=0, reserved=False, include_nmea=False):
    """Decode every message that has a detailed decoder; one block per packet.

    RXM-SFRBX is routed through the per-constellation nav-subframe decoders;
    any other identity registered in :data:`ubxread.messages.MSG_DECODERS`
    (e.g. MON-HW, MON-HW2) gets its own rich field decode.  Messages without a
    dedicated decoder fall back to the one-line :func:`summarize` output so
    nothing is hidden.

    types: optional iterable of identities to keep (e.g. ["MON-HW", "NAV-PVT"]);
        None decodes everything.  reserved: also show reserved/spare fields.
    """
    protfilter = 7 if include_nmea else 2
    want = set(types) if types else None
    n = 0
    decoded = 0
    interrupted = False
    counts = {}

    try:
        with open(path, "rb") as stream:
            ubr = UBXReader(stream, protfilter=protfilter, quitonerror=ERR_LOG)
            for _raw, parsed in ubr:
                if parsed is None:
                    continue
                ident = getattr(parsed, "identity", parsed.__class__.__name__)
                if want is not None and ident not in want:
                    continue
                counts[ident] = counts.get(ident, 0) + 1

                if ident == "RXM-SFRBX":
                    result, words, tag, gid = sfrbx_decode_one(parsed, reserved=reserved)
                    if result is _SKIP or result is None:
                        n += 1
                        note = "D2 NAV (GEO)" if gid == 3 else "no decoder"
                        print(f"#{n}  {ident:<12} {tag} numWords={len(words)}  "
                              f"(raw words, {note})")
                        print("      " + " ".join(f"0x{w:06X}" for w in words))
                    else:
                        n += 1
                        decoded += 1
                        _emit(f"#{n}  {ident:<12} {result.header}", result.fields)
                elif ident in MSG_DECODERS:
                    try:
                        dec = MSG_DECODERS[ident](parsed, reserved=reserved)
                    except Exception as exc:  # a malformed packet must not stop the run
                        n += 1
                        print(f"#{n}  {ident:<12} <decode error: {exc}>")
                        continue
                    n += 1
                    decoded += 1
                    _emit(f"#{n}  {ident:<12} {dec.header}", dec.fields)
                else:
                    n += 1
                    try:
                        info = summarize(parsed)
                    except Exception as exc:
                        info = f"<summary error: {exc}>"
                    print(f"#{n}  {ident:<12} {info}")

                if limit and n >= limit:
                    break
    except KeyboardInterrupt:
        print(f"\nInterrupted after {n} messages.", file=sys.stderr)
        interrupted = True

    print(f"\nDecoded {decoded} of {n} message(s) with detailed decoders.",
          file=sys.stderr)
    for k in sorted(counts):
        mark = "  (decoder)" if (k == "RXM-SFRBX" or k in MSG_DECODERS) else ""
        print(f"  {k:<14} {counts[k]}{mark}", file=sys.stderr)
    return 130 if interrupted else 0


# Messages that can establish the UTC calendar date of an epoch.
DATE_IDENTS = ("NAV-PVT", "NAV-TIMEUTC", "RXM-RAWX")

GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
SECONDS_PER_WEEK = 604800


def epoch_date(msg):
    """Return the UTC calendar date (datetime.date) carried by a message, or None.

    NAV-PVT and NAV-TIMEUTC report UTC year/month/day directly. RXM-RAWX is used
    as a fallback for raw-only logs: its GPS week + receiver time-of-week are
    converted to UTC using the leap-second count in the same message.
    """
    ident = getattr(msg, "identity", None)

    if ident == "NAV-PVT":
        if _g(msg, "validDate") and _g(msg, "year", 0) >= 2000:
            return date(msg.year, msg.month, msg.day)
        return None

    if ident == "NAV-TIMEUTC":
        if _g(msg, "validUTC") and _g(msg, "year", 0) >= 2000:
            return date(msg.year, msg.month, msg.day)
        return None

    if ident == "RXM-RAWX":
        week = _g(msg, "week", 0)
        tow = _g(msg, "rcvTow")
        if not week or tow is None:
            return None
        leap = _g(msg, "leapS", 0) if _g(msg, "leapSec") else 0
        t = GPS_EPOCH + timedelta(seconds=week * SECONDS_PER_WEEK + tow - leap)
        return t.date()

    return None


class DateSplitter:
    """Writes raw UBX frames into one output file per UTC calendar date."""

    def __init__(self, inpath, outdir=None):
        root, ext = os.path.splitext(os.path.basename(inpath))
        self.root = root
        self.ext = ext or ".ubx"
        base = outdir or os.path.dirname(os.path.abspath(inpath))
        # Place the per-date files in a subdirectory named after the input file.
        self.outdir = os.path.join(base, root)
        os.makedirs(self.outdir, exist_ok=True)
        self.fh = None
        self.cur_key = None
        self.seen_keys = set()
        # key -> {"path", "bytes", "epochs"}
        self.stats = {}

    def _path_for(self, key):
        return os.path.join(self.outdir, f"{self.root}.{key}{self.ext}")

    def write_epoch(self, d, data):
        """Append one complete epoch's raw bytes under calendar date `d`."""
        if d is not None:
            key = d.strftime("%Y%m%d")
        else:
            # No date learned yet/for this epoch: keep it with the current file.
            key = self.cur_key or "unknown-date"

        if key != self.cur_key:
            if self.fh is not None:
                self.fh.close()
            path = self._path_for(key)
            mode = "ab" if key in self.seen_keys else "wb"
            self.fh = open(path, mode)
            self.cur_key = key
            self.seen_keys.add(key)
            self.stats.setdefault(key, {"path": path, "bytes": 0, "epochs": 0})

        self.fh.write(data)
        self.stats[key]["bytes"] += len(data)
        self.stats[key]["epochs"] += 1

    def close(self):
        if self.fh is not None:
            self.fh.close()
            self.fh = None


def split_by_date(path, outdir=None, boundary="eoe"):
    """Split a UBX log into per-UTC-date files, keeping epochs intact.

    boundary="eoe": flush each epoch when its NAV-EOE (End-Of-Epoch) marker is
        seen; the date is learned from a NAV-PVT/NAV-TIMEUTC/RXM-RAWX earlier in
        the same epoch. This is the recommended mode for logs that emit NAV-EOE.
    boundary="pvt": treat each NAV-PVT as the start of a new epoch and flush the
        preceding epoch. Use for logs that lack NAV-EOE.
    """
    splitter = DateSplitter(path, outdir)
    buffer = bytearray()
    buf_date = None
    saw_eoe = False
    interrupted = False
    pending = []  # leading epochs seen before any valid date is known

    def commit(d, data):
        # Hold leading undated epochs until the first real date is known, then
        # attach them to that date instead of writing an "unknown-date" file.
        if d is None and splitter.cur_key is None:
            pending.append(data)
            return
        if pending and d is not None:
            for chunk in pending:
                splitter.write_epoch(d, chunk)
            pending.clear()
        splitter.write_epoch(d, data)

    try:
        with open(path, "rb") as stream:
            ubr = UBXReader(stream, protfilter=7, quitonerror=ERR_LOG)
            for raw, parsed in ubr:
                ident = getattr(parsed, "identity", None)

                if boundary == "pvt":
                    if ident in DATE_IDENTS:
                        d = epoch_date(parsed)
                        if d is not None:
                            if buf_date is not None:
                                commit(buf_date, bytes(buffer))
                                buffer.clear()
                            buf_date = d
                    buffer += raw
                else:  # eoe
                    buffer += raw
                    if ident in DATE_IDENTS:
                        d = epoch_date(parsed)
                        if d is not None:
                            buf_date = d
                    if ident == "NAV-EOE":
                        saw_eoe = True
                        commit(buf_date, bytes(buffer))
                        buffer.clear()
                        buf_date = None

        if buffer:
            commit(buf_date, bytes(buffer))
        # Any epochs that never had a valid date anywhere in the log.
        for chunk in pending:
            splitter.write_epoch(None, chunk)
    except KeyboardInterrupt:
        print("\nInterrupted; flushing and closing current output file.", file=sys.stderr)
        interrupted = True
    finally:
        splitter.close()

    if boundary == "eoe" and not saw_eoe and not interrupted:
        print("Warning: no NAV-EOE markers found; output was not split into "
              "epochs. Re-run with --split-boundary pvt.", file=sys.stderr)

    print("\n=== Split by UTC date ===", file=sys.stderr)
    for key in sorted(splitter.stats):
        s = splitter.stats[key]
        print(f"  {key}  epochs={s['epochs']:<8} bytes={s['bytes']:<12} {s['path']}",
              file=sys.stderr)
    return 130 if interrupted else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Dump or split a u-blox UBX log file.")
    ap.add_argument("file", help="path to .ubx log file")
    ap.add_argument("--limit", type=int, default=0, help="stop after N packets (0 = all)")
    ap.add_argument("--no-summary", action="store_true", help="don't print the per-type tally at the end")
    ap.add_argument("--include-nmea", action="store_true", help="also report NMEA/RTCM frames, not just UBX")
    ap.add_argument("--split-by-date", action="store_true",
                    help="split the input into one file per UTC calendar date (keeps epochs intact)")
    ap.add_argument("--outdir", default=None,
                    help="parent directory for --split-by-date output; files go in a "
                         "subdirectory named after the input file (default: input file's directory)")
    ap.add_argument("--split-boundary", choices=("eoe", "pvt"), default="eoe",
                    help="epoch boundary marker for splitting: NAV-EOE (default) or NAV-PVT")
    ap.add_argument("--decode-sfrbx", action="store_true",
                    help="decode RXM-SFRBX nav subframe contents (GPS/QZSS LNAV & "
                         "CNAV, Galileo I/NAV & F/NAV, BeiDou D1/D2/B-CNAV1/B-CNAV2, "
                         "GLONASS, SBAS); other signals are shown as raw words")
    ap.add_argument("--decode", action="store_true",
                    help="decode every message type that has a detailed decoder "
                         "(RXM-SFRBX nav subframes plus MON-HW, MON-HW2, ...); "
                         "messages without one are shown as a one-line summary")
    ap.add_argument("--type", nargs="+", metavar="IDENT", default=None,
                    help="with --decode: only decode these message identities "
                         "(e.g. MON-HW MON-HW2 NAV-PVT)")
    ap.add_argument("--gnss", default="gps",
                    choices=("gps", "sbas", "gal", "bds", "qzss", "glo", "navic", "all"),
                    help="with --decode-sfrbx: which constellation to decode (default: gps)")
    ap.add_argument("--sigid", default="all",
                    help="with --decode-sfrbx: which signal id to decode, as an integer, "
                         "or 'all' (default)")
    ap.add_argument("--sv", type=int, default=None,
                    help="with --decode-sfrbx: only decode this satellite (svId)")
    ap.add_argument("--subframe", type=int, default=None, choices=(1, 2, 3, 4, 5),
                    help="with --decode-sfrbx: only decode this LNAV subframe id (1-5)")
    ap.add_argument("--decode-reserved", action="store_true",
                    help="with --decode-sfrbx: also show GPS reserved/spare fields "
                         "(hex, plus ASCII when printable); off by default")
    args = ap.parse_args(argv)

    if args.split_by_date:
        return split_by_date(args.file, outdir=args.outdir, boundary=args.split_boundary)

    if args.decode:
        return decode_messages(args.file, types=args.type, limit=args.limit,
                               reserved=args.decode_reserved,
                               include_nmea=args.include_nmea)

    if args.decode_sfrbx:
        gnss_id = None if args.gnss == "all" else GNSS_BY_NAME[args.gnss]
        if args.sigid == "all":
            sigid = None
        else:
            try:
                sigid = int(args.sigid)
            except ValueError:
                ap.error("--sigid must be an integer or 'all'")
        return decode_sfrbx(args.file, gnss_id=gnss_id, sigid=sigid,
                            sv=args.sv, subframe=args.subframe, limit=args.limit,
                            reserved=args.decode_reserved)

    # protfilter: 2 = UBX only; 7 = UBX + NMEA + RTCM
    protfilter = 7 if args.include_nmea else 2

    counts = {}
    n = 0
    interrupted = False
    try:
        with open(args.file, "rb") as stream:
            ubr = UBXReader(stream, protfilter=protfilter, quitonerror=ERR_LOG)
            for _raw, parsed in ubr:
                if parsed is None:
                    continue
                ident = getattr(parsed, "identity", parsed.__class__.__name__)
                counts[ident] = counts.get(ident, 0) + 1
                n += 1
                try:
                    info = summarize(parsed)
                except Exception as exc:  # never let one odd packet stop the dump
                    info = f"<summary error: {exc}>"
                print(f"{n:>8}  {ident:<14} {info}")
                if args.limit and n >= args.limit:
                    break
    except KeyboardInterrupt:
        print(f"\nInterrupted after {n} packets.", file=sys.stderr)
        interrupted = True

    if not args.no_summary:
        print("\n=== Summary ===", file=sys.stderr)
        print(f"Total packets: {n}", file=sys.stderr)
        for k in sorted(counts):
            print(f"  {k:<14} {counts[k]}", file=sys.stderr)

    return 130 if interrupted else 0
