"""Generic, per-message-type decoders (everything that is *not* RXM-SFRBX).

RXM-SFRBX carries raw GNSS navigation subframes and has its own constellation
dispatch in :mod:`ubxread.cli`.  Every *other* UBX message is already fully
parsed into named fields by ``pyubx2``; the decoders here turn those fields into
the same rich, multi-line, CRC-style output the SFRBX decoders produce so any
message type can be inspected with one consistent format.

Adding support for a new message type is a one-function change: write a
``decode_xxx(msg, reserved=False)`` that returns a :class:`Decoded`, and tag it
with ``@register("MON-XYZ")``.  The dump/decode driver picks it up automatically
via :data:`MSG_DECODERS`.

A decoder returns ``Decoded(header, fields)`` where:
  * ``header`` is a short one-line summary (the driver prefixes it with the
    packet index and message identity), and
  * ``fields`` is a list of ``(label, value, unit)`` triples laid out as an
    aligned grid.  ``value`` may be an int/float (formatted numerically) or a
    pre-formatted string (e.g. a hex bitfield), in which case ``unit`` is
    usually "".
"""
from collections import namedtuple

Decoded = namedtuple("Decoded", "header fields")

# Output markers a decoder may place in a Decoded.fields list, alongside plain
# (label, value, unit) cells, to structure grouped/repeated data:
#   Section(title) -> a sub-group header line (e.g. "block 0", "pins (34)")
#   Line(text)     -> a verbatim detail line (e.g. one row per GPIO pin)
# Plain tuples are accumulated into an aligned grid; a Section/Line flushes it.
Section = namedtuple("Section", "title")
Line = namedtuple("Line", "text")

# identity (e.g. "MON-HW") -> decoder(msg, reserved=False) -> Decoded
MSG_DECODERS = {}


def register(*idents):
    """Register a decoder function for one or more message identities."""
    def deco(fn):
        for ident in idents:
            MSG_DECODERS[ident] = fn
        return fn
    return deco


def _enum(value, table, unknown="?"):
    """Render an enumerated field as ``value (name)``."""
    return f"{value} ({table.get(value, unknown)})"


def _grp(msg, name, i):
    """Read the i-th (1-based) field of a repeated group, e.g. agcCnt_01."""
    return getattr(msg, f"{name}_{i:02d}", None)


def _num(v, scale=1.0, nd=3):
    """Scale a numeric field and format with a fixed number of decimals.

    Pre-formatting (rather than relying on the grid's %g) keeps full precision
    for coordinates and avoids exponent notation for large ECEF distances.
    """
    if v is None:
        return "?"
    return f"{v * scale:.{nd}f}"


def _hexle(b):
    """Format a little-endian byte field as an MSB-first hex literal."""
    if not isinstance(b, (bytes, bytearray)):
        return f"0x{int(b):X}"
    return "0x" + (bytes(b)[::-1].hex().upper() or "0")


def _g(msg, name, default=None):
    return getattr(msg, name, default)


# --- MON-HW: hardware status (antenna, AGC, noise, CW/broadband jamming) ----

_HW_ASTATUS = {0: "INIT", 1: "DONTKNOW", 2: "OK", 3: "SHORT", 4: "OPEN"}
_HW_APOWER = {0: "OFF", 1: "ON", 2: "DONTKNOW"}
_HW_JAMMING = {0: "unknown/disabled", 1: "ok", 2: "warning", 3: "critical"}


@register("MON-HW")
def decode_mon_hw(msg, reserved=False):
    agc = _g(msg, "agcCnt", 0) or 0
    jam = _g(msg, "jamInd", 0) or 0
    astatus = _g(msg, "aStatus")
    apower = _g(msg, "aPower")
    jstate = _g(msg, "jammingState")

    fields = [
        ("noisePerMS", _g(msg, "noisePerMS"), ""),
        ("agcCnt", agc, " (0-8191)"),
        ("agc", agc / 8191 * 100, "%"),
        ("aStatus", _enum(astatus, _HW_ASTATUS), ""),
        ("aPower", _enum(apower, _HW_APOWER), ""),
        ("jammingState", _enum(jstate, _HW_JAMMING), ""),
        ("jamInd", jam, " (0-255)"),
        ("jam", jam / 255 * 100, "%"),
        ("rtcCalib", _g(msg, "rtcCalib"), ""),
        ("safeBoot", _g(msg, "safeBoot"), ""),
        ("xtalAbsent", _g(msg, "xtalAbsent"), ""),
        ("usedMask", _hexle(_g(msg, "usedMask", b"")), ""),
        ("pinSel", _hexle(_g(msg, "pinSel", b"")), ""),
        ("pinBank", _hexle(_g(msg, "pinBank", b"")), ""),
        ("pinDir", _hexle(_g(msg, "pinDir", b"")), ""),
        ("pinVal", _hexle(_g(msg, "pinVal", b"")), ""),
        ("pullH", _hexle(_g(msg, "pullH", b"")), ""),
        ("pullL", _hexle(_g(msg, "pullL", b"")), ""),
    ]
    if reserved:
        fields += [
            ("reserved0", _g(msg, "reserved0"), ""),
            ("reserved1", _g(msg, "reserved1"), ""),
            ("pinIrq", _hexle(_g(msg, "pinIrq", b"")), ""),
        ]

    header = (f"ant={_HW_ASTATUS.get(astatus, astatus)}/"
              f"{_HW_APOWER.get(apower, apower)}  "
              f"jam={_HW_JAMMING.get(jstate, jstate)}  "
              f"noise={_g(msg, 'noisePerMS')}  agc={agc}")
    return Decoded(header, fields)


# --- MON-HW2: extended hardware status (I/Q front-end, config source) -------

_HW2_CFGSOURCE = {0: "undefined", 102: "flash image", 111: "OTP",
                  112: "config pins", 114: "ROM"}


@register("MON-HW2")
def decode_mon_hw2(msg, reserved=False):
    ofsI = _g(msg, "ofsI")
    magI = _g(msg, "magI")
    ofsQ = _g(msg, "ofsQ")
    magQ = _g(msg, "magQ")
    cfg = _g(msg, "cfgSource")

    fields = [
        ("ofsI", ofsI, " (I imbalance)"),
        ("magI", magI, " (I magnitude, 0-255)"),
        ("ofsQ", ofsQ, " (Q imbalance)"),
        ("magQ", magQ, " (Q magnitude, 0-255)"),
        ("cfgSource", _enum(cfg, _HW2_CFGSOURCE), ""),
        ("postStatus", _hexle(_g(msg, "postStatus", 0)), ""),
    ]
    if reserved:
        fields += [
            ("lowLevCfg", _hexle(_g(msg, "lowLevCfg", 0)), ""),
            ("reserved0", _g(msg, "reserved0"), ""),
            ("reserved1", _g(msg, "reserved1"), ""),
            ("reserved2", _g(msg, "reserved2"), ""),
        ]

    header = (f"I: ofs={ofsI} mag={magI}   Q: ofs={ofsQ} mag={magQ}   "
              f"cfgSource={_HW2_CFGSOURCE.get(cfg, cfg)}")
    return Decoded(header, fields)


# --- MON-RF: per-RF-band status (antenna, AGC, noise, jamming, I/Q) ---------

@register("MON-RF")
def decode_mon_rf(msg, reserved=False):
    n = _g(msg, "nBlocks", 0) or 0
    body = []
    for i in range(1, n + 1):
        jam = _grp(msg, "jammingState", i)
        ast = _grp(msg, "antStatus", i)
        apw = _grp(msg, "antPower", i)
        noise = _grp(msg, "noisePerMS", i)
        agc = _grp(msg, "agcCnt", i) or 0
        jind = _grp(msg, "jamInd", i) or 0
        body.append(Section(
            f"block {_grp(msg, 'blockId', i)}: "
            f"jam={_HW_JAMMING.get(jam, jam)} "
            f"ant={_HW_ASTATUS.get(ast, ast)}/{_HW_APOWER.get(apw, apw)} "
            f"noise={noise} agc={agc}"))
        body += [
            ("noisePerMS", noise, ""),
            ("agcCnt", agc, " (0-8191)"),
            ("agc", agc / 8191 * 100, "%"),
            ("jammingState", _enum(jam, _HW_JAMMING), ""),
            ("antStatus", _enum(ast, _HW_ASTATUS), ""),
            ("antPower", _enum(apw, _HW_APOWER), ""),
            ("jamInd", jind, " (0-255)"),
            ("jam", jind / 255 * 100, "%"),
            ("ofsI", _grp(msg, "ofsI", i), ""),
            ("magI", _grp(msg, "magI", i), ""),
            ("ofsQ", _grp(msg, "ofsQ", i), ""),
            ("magQ", _grp(msg, "magQ", i), ""),
            ("postStatus", _hexle(_grp(msg, "postStatus", i) or 0), ""),
        ]
        if reserved:
            body += [
                ("reserved1", _grp(msg, "reserved1", i), ""),
                ("reserved2", _grp(msg, "reserved2", i), ""),
            ]
    header = f"version={_g(msg, 'version')} nBlocks={n}"
    return Decoded(header, body)


# --- MON-SYS: system status (CPU/memory/IO load, uptime, temperature) -------

_SYS_BOOT = {0: "unknown", 1: "cold start", 2: "watchdog", 3: "hardware reset",
             4: "hardware backup", 5: "software backup", 6: "software reset",
             7: "VIO fail", 8: "VDD_X fail", 9: "VDD_RF fail",
             10: "V_CORE_HIGH fail"}


@register("MON-SYS")
def decode_mon_sys(msg, reserved=False):
    boot = _g(msg, "bootType")
    fields = [
        ("msgVer", _g(msg, "msgVer"), ""),
        ("bootType", _enum(boot, _SYS_BOOT), ""),
        ("cpuLoad", _g(msg, "cpuLoad"), "%"),
        ("cpuLoadMax", _g(msg, "cpuLoadMax"), "%"),
        ("memUsage", _g(msg, "memUsage"), "%"),
        ("memUsageMax", _g(msg, "memUsageMax"), "%"),
        ("ioUsage", _g(msg, "ioUsage"), "%"),
        ("ioUsageMax", _g(msg, "ioUsageMax"), "%"),
        ("runTime", _g(msg, "runTime"), " s"),
        ("noticeCount", _g(msg, "noticeCount"), ""),
        ("warnCount", _g(msg, "warnCount"), ""),
        ("errorCount", _g(msg, "errorCount"), ""),
        ("tempValue", _g(msg, "tempValue"), " C"),
    ]
    if reserved:
        fields.append(("reserved0", _g(msg, "reserved0"), ""))
    header = (f"boot={_SYS_BOOT.get(boot, boot)} cpu={_g(msg, 'cpuLoad')}% "
              f"mem={_g(msg, 'memUsage')}% temp={_g(msg, 'tempValue')}C "
              f"up={_g(msg, 'runTime')}s err={_g(msg, 'errorCount')}")
    return Decoded(header, fields)


# --- MON-TXBUF: transmit buffer usage per port ------------------------------

@register("MON-TXBUF")
def decode_mon_txbuf(msg, reserved=False):
    body = []
    for i in range(1, 7):
        body.append(Line(
            f"port {i - 1}: pending={_grp(msg, 'pending', i)} "
            f"usage={_grp(msg, 'usage', i)}% "
            f"peak={_grp(msg, 'peakUsage', i)}%"))
    body.append(Section("totals"))
    body += [
        ("tUsage", _g(msg, "tUsage"), "%"),
        ("tPeakUsage", _g(msg, "tPeakUsage"), "%"),
        ("errLimit", _g(msg, "limit"), ""),
        ("errMem", _g(msg, "lem"), ""),
        ("errAlloc", _g(msg, "alloc"), ""),
    ]
    if reserved:
        body.append(("reserved0", _g(msg, "reserved0"), ""))
    header = (f"tUsage={_g(msg, 'tUsage')}% tPeak={_g(msg, 'tPeakUsage')}% "
              f"errs(limit={_g(msg, 'limit')},mem={_g(msg, 'lem')},"
              f"alloc={_g(msg, 'alloc')})")
    return Decoded(header, body)


# --- MON-HW3: extended hardware status, one row per GPIO pin ----------------

@register("MON-HW3")
def decode_mon_hw3(msg, reserved=False):
    n = _g(msg, "nPins", 0) or 0
    hwver = _g(msg, "hwVersion", b"")
    if isinstance(hwver, (bytes, bytearray)):
        hwver = bytes(hwver).split(b"\x00")[0].decode("ascii", "replace")

    body = [
        ("version", _g(msg, "version"), ""),
        ("nPins", n, ""),
        ("hwVersion", hwver, ""),
        ("rtcCalib", _g(msg, "rtcCalib"), ""),
        ("safeBoot", _g(msg, "safeBoot"), ""),
        ("xtalAbsent", _g(msg, "xtalAbsent"), ""),
    ]
    if reserved:
        body.append(("reserved0", _g(msg, "reserved0"), ""))

    body.append(Section(f"pins ({n})"))
    for i in range(1, n + 1):
        pull = ("H" if _grp(msg, "pioPullHigh", i)
                else "L" if _grp(msg, "pioPullLow", i) else "-")
        body.append(Line(
            f"pin {_grp(msg, 'pinId', i):>4}: "
            f"{'PIO' if _grp(msg, 'periphPIO', i) else 'periph'} "
            f"bank={_grp(msg, 'pinBank', i)} "
            f"dir={'out' if _grp(msg, 'direction', i) else 'in'} "
            f"val={_grp(msg, 'pinValue', i)} "
            f"pull={pull} "
            f"vpMgr={_grp(msg, 'vpManager', i)} "
            f"irq={_grp(msg, 'pioIrq', i)} "
            f"VP={_grp(msg, 'VP', i)}"))

    header = (f"version={_g(msg, 'version')} hwVersion={hwver} nPins={n} "
              f"rtcCalib={_g(msg, 'rtcCalib')}")
    return Decoded(header, body)


# --- MON-COMMS: communication port statistics ------------------------------

_COMMS_PORT = {0x000: "I2C", 0x100: "UART1", 0x200: "UART2",
               0x300: "USB", 0x400: "SPI"}
_COMMS_PROTO = {0: "UBX", 1: "NMEA", 2: "RTCM2", 5: "RTCM3", 255: "none"}


@register("MON-COMMS")
def decode_mon_comms(msg, reserved=False):
    n = _g(msg, "nPorts", 0) or 0
    # 4 fixed protocol slots; pyubx2 names them protIds_01..04
    proto_names = [_COMMS_PROTO.get(getattr(msg, f"protIds_{j:02d}", None), "?")
                   for j in range(1, 5)]
    err_parts = []
    if _g(msg, "mem"):   err_parts.append("mem")
    if _g(msg, "alloc"): err_parts.append("alloc")
    op = _g(msg, "outputPort")
    if op: err_parts.append(f"outPort={op}")

    body = [
        ("version",    _g(msg, "version"), ""),
        ("nPorts",     n,                  ""),
        ("txErrors",   ", ".join(err_parts) or "none", ""),
        ("protocols",  ", ".join(proto_names), ""),
    ]
    if reserved:
        body.append(("reserved0", _g(msg, "reserved0"), ""))

    body.append(Section(f"ports ({n})"))
    for i in range(1, n + 1):
        pid = _grp(msg, "portId", i)
        txu = _grp(msg, "txUsage", i) or 0
        rxu = _grp(msg, "rxUsage", i) or 0
        msg_counts = [getattr(msg, f"msgs_{i:02d}_{j:02d}", 0) or 0
                      for j in range(1, 5)]
        msg_str = " ".join(f"{proto_names[j]}={msg_counts[j]}"
                           for j in range(4))
        body.append(Section(
            f"port {_COMMS_PORT.get(pid, pid)}: "
            f"tx={txu}% rx={rxu}% "
            f"overrunErrs={_grp(msg, 'overrunErrs', i)} "
            f"skipped={_grp(msg, 'skipped', i)}"))
        body += [
            ("portId",       _enum(pid, _COMMS_PORT), ""),
            ("txPending",    _grp(msg, "txPending",    i), " B"),
            ("txBytes",      _grp(msg, "txBytes",      i), " B"),
            ("txUsage",      txu,                         "%"),
            ("txPeakUsage",  _grp(msg, "txPeakUsage",  i), "%"),
            ("rxPending",    _grp(msg, "rxPending",    i), " B"),
            ("rxBytes",      _grp(msg, "rxBytes",      i), " B"),
            ("rxUsage",      rxu,                         "%"),
            ("rxPeakUsage",  _grp(msg, "rxPeakUsage",  i), "%"),
            ("overrunErrs",  _grp(msg, "overrunErrs",  i), ""),
            ("skipped",      _grp(msg, "skipped",      i), ""),
            ("msgs",         msg_str,                     ""),
        ]
        if reserved:
            body.append(("reserved1", getattr(msg, f"reserved1_{i:02d}", None), ""))

    header = (f"version={_g(msg, 'version')} nPorts={n} "
              f"txErrors={'|'.join(err_parts) or 'none'}")
    return Decoded(header, body)


# ===========================================================================
# NAV-* navigation solution / status messages
# ===========================================================================

_GNSS = {0: "GPS", 1: "SBAS", 2: "GAL", 3: "BDS", 4: "IMES", 5: "QZSS",
         6: "GLO", 7: "NavIC"}

_PVT_FIX = {0: "no-fix", 1: "dead-reckoning", 2: "2D", 3: "3D",
            4: "GNSS+DR", 5: "time-only"}
_PVT_CARR = {0: "none", 1: "float", 2: "fixed"}
_PVT_PSM = {0: "not active", 1: "enabled", 2: "acquisition", 3: "tracking",
            4: "power-optimised tracking", 5: "inactive"}

_ORB_HEALTH = {0: "unknown", 1: "healthy", 2: "not healthy"}
_ORB_VIS = {0: "unknown", 1: "below horizon", 2: "above horizon", 3: "visible"}


# --- NAV-CLOCK: receiver clock bias / drift --------------------------------

@register("NAV-CLOCK")
def decode_nav_clock(msg, reserved=False):
    itow = _g(msg, "iTOW", 0)
    fields = [
        ("iTOW", _num(itow, 1e-3, 3), " s"),
        ("clkB", _g(msg, "clkB"), " ns"),
        ("clkD", _g(msg, "clkD"), " ns/s"),
        ("tAcc", _g(msg, "tAcc"), " ns"),
        ("fAcc", _g(msg, "fAcc"), " ps/s"),
    ]
    header = (f"iTOW={_num(itow, 1e-3, 3)}s clkB={_g(msg, 'clkB')}ns "
              f"clkD={_g(msg, 'clkD')}ns/s tAcc={_g(msg, 'tAcc')}ns")
    return Decoded(header, fields)


# --- NAV-DOP: dilution of precision ----------------------------------------

@register("NAV-DOP")
def decode_nav_dop(msg, reserved=False):
    itow = _g(msg, "iTOW", 0)
    fields = [("iTOW", _num(itow, 1e-3, 3), " s")]
    for k in ("gDOP", "pDOP", "tDOP", "vDOP", "hDOP", "nDOP", "eDOP"):
        fields.append((k, _g(msg, k), ""))
    header = (f"iTOW={_num(itow, 1e-3, 3)}s pDOP={_g(msg, 'pDOP')} "
              f"hDOP={_g(msg, 'hDOP')} vDOP={_g(msg, 'vDOP')}")
    return Decoded(header, fields)


# --- NAV-EOE: end of epoch -------------------------------------------------

@register("NAV-EOE")
def decode_nav_eoe(msg, reserved=False):
    itow = _g(msg, "iTOW", 0)
    header = f"iTOW={_num(itow, 1e-3, 3)}s (end of epoch)"
    return Decoded(header, [("iTOW", _num(itow, 1e-3, 3), " s")])


# --- NAV-POSECEF: position in Earth-centred Earth-fixed frame --------------

@register("NAV-POSECEF")
def decode_nav_posecef(msg, reserved=False):
    itow = _g(msg, "iTOW", 0)
    x, y, z = _g(msg, "ecefX"), _g(msg, "ecefY"), _g(msg, "ecefZ")
    fields = [
        ("iTOW", _num(itow, 1e-3, 3), " s"),
        ("ecefX", _num(x, 1e-2, 2), " m"),
        ("ecefY", _num(y, 1e-2, 2), " m"),
        ("ecefZ", _num(z, 1e-2, 2), " m"),
        ("pAcc", _num(_g(msg, "pAcc"), 1e-2, 2), " m"),
    ]
    header = (f"iTOW={_num(itow, 1e-3, 3)}s "
              f"ecef=({_num(x, 1e-2, 2)}, {_num(y, 1e-2, 2)}, "
              f"{_num(z, 1e-2, 2)}) m  pAcc={_num(_g(msg, 'pAcc'), 1e-2, 2)}m")
    return Decoded(header, fields)


# --- NAV-POSLLH: geodetic position -----------------------------------------

@register("NAV-POSLLH")
def decode_nav_posllh(msg, reserved=False):
    itow = _g(msg, "iTOW", 0)
    lat, lon = _g(msg, "lat"), _g(msg, "lon")
    fields = [
        ("iTOW", _num(itow, 1e-3, 3), " s"),
        ("lat", _num(lat, 1, 7), " deg"),
        ("lon", _num(lon, 1, 7), " deg"),
        ("height", _num(_g(msg, "height"), 1e-3, 3), " m"),
        ("hMSL", _num(_g(msg, "hMSL"), 1e-3, 3), " m"),
        ("hAcc", _num(_g(msg, "hAcc"), 1e-3, 3), " m"),
        ("vAcc", _num(_g(msg, "vAcc"), 1e-3, 3), " m"),
    ]
    header = (f"iTOW={_num(itow, 1e-3, 3)}s lat={_num(lat, 1, 7)} "
              f"lon={_num(lon, 1, 7)} hMSL={_num(_g(msg, 'hMSL'), 1e-3, 3)}m")
    return Decoded(header, fields)


# --- NAV-HPPOSECEF: high-precision ECEF position ---------------------------

# pyubx2 folds the high-precision (_HP*) component into the base attribute, so
# ecefX/Y/Z already carry sub-cm resolution (cm units) and pAcc is in mm.

@register("NAV-HPPOSECEF")
def decode_nav_hpposecef(msg, reserved=False):
    itow = _g(msg, "iTOW", 0)
    x, y, z = _g(msg, "ecefX"), _g(msg, "ecefY"), _g(msg, "ecefZ")
    invalid = _g(msg, "invalidEcef")
    fields = [
        ("iTOW",        _num(itow, 1e-3, 3),          " s"),
        ("version",     _g(msg, "version"),           ""),
        ("ecefX",       _num(x, 1e-2, 4),             " m"),
        ("ecefY",       _num(y, 1e-2, 4),             " m"),
        ("ecefZ",       _num(z, 1e-2, 4),             " m"),
        ("pAcc",        _num(_g(msg, "pAcc"), 1e-3, 4), " m"),
        ("invalidEcef", invalid,                       ""),
    ]
    if reserved:
        fields.append(("reserved0", _g(msg, "reserved0"), ""))
    header = (f"iTOW={_num(itow, 1e-3, 3)}s "
              f"ecef=({_num(x, 1e-2, 4)}, {_num(y, 1e-2, 4)}, "
              f"{_num(z, 1e-2, 4)}) m  "
              f"pAcc={_num(_g(msg, 'pAcc'), 1e-3, 4)}m"
              f"{'  INVALID' if invalid else ''}")
    return Decoded(header, fields)


# --- NAV-HPPOSLLH: high-precision geodetic position ------------------------

# lat/lon carry the folded HP component (1e-9 deg resolution); height/hMSL are
# in mm with 0.1 mm resolution; hAcc/vAcc are in mm.

@register("NAV-HPPOSLLH")
def decode_nav_hpposllh(msg, reserved=False):
    itow = _g(msg, "iTOW", 0)
    lat, lon = _g(msg, "lat"), _g(msg, "lon")
    invalid = _g(msg, "invalidLlh")
    fields = [
        ("iTOW",       _num(itow, 1e-3, 3),            " s"),
        ("version",    _g(msg, "version"),             ""),
        ("lat",        _num(lat, 1, 9),                " deg"),
        ("lon",        _num(lon, 1, 9),                " deg"),
        ("height",     _num(_g(msg, "height"), 1e-3, 4), " m"),
        ("hMSL",       _num(_g(msg, "hMSL"),   1e-3, 4), " m"),
        ("hAcc",       _num(_g(msg, "hAcc"),   1e-3, 4), " m"),
        ("vAcc",       _num(_g(msg, "vAcc"),   1e-3, 4), " m"),
        ("invalidLlh", invalid,                         ""),
    ]
    if reserved:
        fields.append(("reserved0", _g(msg, "reserved0"), ""))
    header = (f"iTOW={_num(itow, 1e-3, 3)}s lat={_num(lat, 1, 9)} "
              f"lon={_num(lon, 1, 9)} hMSL={_num(_g(msg, 'hMSL'), 1e-3, 4)}m"
              f"{'  INVALID' if invalid else ''}")
    return Decoded(header, fields)


# --- NAV-PVT: position / velocity / time solution --------------------------

@register("NAV-PVT")
def decode_nav_pvt(msg, reserved=False):
    try:
        ts = (f"{msg.year:04d}-{msg.month:02d}-{msg.day:02d}T"
              f"{msg.hour:02d}:{msg.min:02d}:{msg.second:02d}")
    except Exception:
        ts = "?"
    fix = _g(msg, "fixType")
    lat, lon = _g(msg, "lat"), _g(msg, "lon")
    fields = [
        ("time", ts, ""),
        ("validDate", _g(msg, "validDate"), ""),
        ("validTime", _g(msg, "validTime"), ""),
        ("fullyResolved", _g(msg, "fullyResolved"), ""),
        ("nano", _g(msg, "nano"), " ns"),
        ("tAcc", _g(msg, "tAcc"), " ns"),
        ("fixType", _enum(fix, _PVT_FIX), ""),
        ("gnssFixOk", _g(msg, "gnssFixOk"), ""),
        ("diffSoln", _g(msg, "diffSoln"), ""),
        ("carrSoln", _enum(_g(msg, "carrSoln"), _PVT_CARR), ""),
        ("psmState", _enum(_g(msg, "psmState"), _PVT_PSM), ""),
        ("numSV", _g(msg, "numSV"), ""),
        ("lat", _num(lat, 1, 7), " deg"),
        ("lon", _num(lon, 1, 7), " deg"),
        ("height", _num(_g(msg, "height"), 1e-3, 3), " m"),
        ("hMSL", _num(_g(msg, "hMSL"), 1e-3, 3), " m"),
        ("hAcc", _num(_g(msg, "hAcc"), 1e-3, 3), " m"),
        ("vAcc", _num(_g(msg, "vAcc"), 1e-3, 3), " m"),
        ("velN", _num(_g(msg, "velN"), 1e-3, 3), " m/s"),
        ("velE", _num(_g(msg, "velE"), 1e-3, 3), " m/s"),
        ("velD", _num(_g(msg, "velD"), 1e-3, 3), " m/s"),
        ("gSpeed", _num(_g(msg, "gSpeed"), 1e-3, 3), " m/s"),
        ("headMot", _g(msg, "headMot"), " deg"),
        ("sAcc", _num(_g(msg, "sAcc"), 1e-3, 3), " m/s"),
        ("headAcc", _g(msg, "headAcc"), " deg"),
        ("pDOP", _g(msg, "pDOP"), ""),
    ]
    if _g(msg, "validMag"):
        fields += [("magDec", _g(msg, "magDec"), " deg"),
                   ("magAcc", _g(msg, "magAcc"), " deg")]
    if reserved:
        fields.append(("reserved0", _g(msg, "reserved0"), ""))
    header = (f"{ts} fix={_PVT_FIX.get(fix, fix)} numSV={_g(msg, 'numSV')} "
              f"lat={_num(lat, 1, 7)} lon={_num(lon, 1, 7)} "
              f"hMSL={_num(_g(msg, 'hMSL'), 1e-3, 3)}m")
    return Decoded(header, fields)


# --- NAV-ORB: orbit/ephemeris/almanac availability, one row per satellite --

@register("NAV-ORB")
def decode_nav_orb(msg, reserved=False):
    itow = _g(msg, "iTOW", 0)
    n = _g(msg, "numSv", 0) or 0
    body = [
        ("iTOW", _num(itow, 1e-3, 3), " s"),
        ("version", _g(msg, "version"), ""),
        ("numSv", n, ""),
    ]
    if reserved:
        body.append(("reserved0", _g(msg, "reserved0"), ""))
    body.append(Section(f"satellites ({n})"))
    for i in range(1, n + 1):
        gid = _grp(msg, "gnssId", i)
        body.append(Line(
            f"{_GNSS.get(gid, gid):<5} SV {_grp(msg, 'svId', i):>3}: "
            f"health={_ORB_HEALTH.get(_grp(msg, 'health', i), '?')} "
            f"vis={_ORB_VIS.get(_grp(msg, 'visibility', i), '?')} "
            f"eph(use={_grp(msg, 'ephUsability', i)},"
            f"src={_grp(msg, 'ephSource', i)}) "
            f"alm(use={_grp(msg, 'almUsability', i)},"
            f"src={_grp(msg, 'almSource', i)}) "
            f"ano={_grp(msg, 'anoAopUsability', i)}"))
    header = (f"iTOW={_num(itow, 1e-3, 3)}s version={_g(msg, 'version')} "
              f"numSv={n}")
    return Decoded(header, body)

# --- NAV-STATUS: receiver navigation status --------------------------------

_STATUS_PSM = {0: "acquisition", 1: "tracking",
               2: "power-optimised tracking", 3: "inactive"}
_SPOOF = {0: "unknown/disabled", 1: "none", 2: "indicated",
          3: "indicated (multiple)"}


@register("NAV-STATUS")
def decode_nav_status(msg, reserved=False):
    itow = _g(msg, "iTOW", 0)
    fix = _g(msg, "gpsFix")
    fields = [
        ("iTOW", _num(itow, 1e-3, 3), " s"),
        ("gpsFix", _enum(fix, _PVT_FIX), ""),
        ("gpsFixOk", _g(msg, "gpsFixOk"), ""),
        ("diffSoln", _g(msg, "diffSoln"), ""),
        ("wknSet", _g(msg, "wknSet"), ""),
        ("towSet", _g(msg, "towSet"), ""),
        ("diffCorr", _g(msg, "diffCorr"), ""),
        ("carrSoln", _enum(_g(msg, "carrSoln"), _PVT_CARR), ""),
        ("mapMatching", _g(msg, "mapMatching"), ""),
        ("psmState", _enum(_g(msg, "psmState"), _STATUS_PSM), ""),
        ("spoofDetState", _enum(_g(msg, "spoofDetState"), _SPOOF), ""),
        ("ttff", _num(_g(msg, "ttff"), 1e-3, 3), " s"),
        ("msss", _num(_g(msg, "msss"), 1e-3, 3), " s"),
    ]
    header = (f"iTOW={_num(itow, 1e-3, 3)}s fix={_PVT_FIX.get(fix, fix)} "
              f"fixOk={_g(msg, 'gpsFixOk')} "
              f"ttff={_num(_g(msg, 'ttff'), 1e-3, 3)}s "
              f"spoof={_SPOOF.get(_g(msg, 'spoofDetState'), '?')}")
    return Decoded(header, fields)


# --- NAV-VELECEF: velocity in Earth-centred Earth-fixed frame --------------

@register("NAV-VELECEF")
def decode_nav_velecef(msg, reserved=False):
    itow = _g(msg, "iTOW", 0)
    vx, vy, vz = _g(msg, "ecefVX"), _g(msg, "ecefVY"), _g(msg, "ecefVZ")
    fields = [
        ("iTOW",   _num(itow, 1e-3, 3),     " s"),
        ("ecefVX", _num(vx,   1e-2, 3),     " m/s"),
        ("ecefVY", _num(vy,   1e-2, 3),     " m/s"),
        ("ecefVZ", _num(vz,   1e-2, 3),     " m/s"),
        ("sAcc",   _num(_g(msg, "sAcc"), 1e-2, 3), " m/s"),
    ]
    header = (f"iTOW={_num(itow, 1e-3, 3)}s "
              f"vel=({_num(vx, 1e-2, 3)}, {_num(vy, 1e-2, 3)}, "
              f"{_num(vz, 1e-2, 3)}) m/s  "
              f"sAcc={_num(_g(msg, 'sAcc'), 1e-2, 3)}m/s")
    return Decoded(header, fields)


# --- NAV-VELNED: velocity in North/East/Down frame -------------------------

@register("NAV-VELNED")
def decode_nav_velned(msg, reserved=False):
    itow = _g(msg, "iTOW", 0)
    vn, ve, vd = _g(msg, "velN"), _g(msg, "velE"), _g(msg, "velD")
    fields = [
        ("iTOW",    _num(itow, 1e-3, 3),              " s"),
        ("velN",    _num(vn,   1e-3, 3),              " m/s"),
        ("velE",    _num(ve,   1e-3, 3),              " m/s"),
        ("velD",    _num(vd,   1e-3, 3),              " m/s"),
        ("speed",   _num(_g(msg, "speed"),   1e-3, 3), " m/s"),
        ("gSpeed",  _num(_g(msg, "gSpeed"),  1e-3, 3), " m/s"),
        ("heading", _num(_g(msg, "heading"), 1, 5), " deg"),
        ("sAcc",    _num(_g(msg, "sAcc"),    1e-3, 3), " m/s"),
        ("cAcc",    _num(_g(msg, "cAcc"),    1, 5), " deg"),
    ]
    header = (f"iTOW={_num(itow, 1e-3, 3)}s "
              f"N={_num(vn, 1e-3, 3)} E={_num(ve, 1e-3, 3)} "
              f"D={_num(vd, 1e-3, 3)} m/s  "
              f"gSpeed={_num(_g(msg, 'gSpeed'), 1e-3, 3)}m/s "
              f"hdg={_num(_g(msg, 'heading'), 1, 5)}deg")
    return Decoded(header, fields)


# --- NAV-SAT: per-satellite signal/quality info, one row per satellite -----

_SAT_QUAL = {0: "no-signal", 1: "searching", 2: "acquired", 3: "unusable",
             4: "code+time", 5: "code+carrier+time", 6: "code+carrier+time",
             7: "code+carrier+time"}
_SAT_HEALTH = {0: "unknown", 1: "healthy", 2: "unhealthy"}
_ORBSRC = {0: "none", 1: "eph", 2: "alm", 3: "AOP-offline",
           4: "AOP-auto", 5: "other", 6: "other", 7: "other"}


@register("NAV-SAT")
def decode_nav_sat(msg, reserved=False):
    itow = _g(msg, "iTOW", 0)
    n = _g(msg, "numSvs", 0) or 0
    used = sum(1 for i in range(1, n + 1) if _grp(msg, "svUsed", i))
    body = [
        ("iTOW", _num(itow, 1e-3, 3), " s"),
        ("version", _g(msg, "version"), ""),
        ("numSvs", n, ""),
        ("svsUsed", used, ""),
    ]
    if reserved:
        body.append(("reserved0", _g(msg, "reserved0"), ""))
    body.append(Section(f"satellites ({n})"))
    for i in range(1, n + 1):
        gid = _grp(msg, "gnssId", i)
        body.append(Line(
            f"{_GNSS.get(gid, gid):<5} SV {_grp(msg, 'svId', i):>3}: "
            f"cno={_grp(msg, 'cno', i):>2} elev={_grp(msg, 'elev', i):>3} "
            f"azim={_grp(msg, 'azim', i):>3} "
            f"qual={_SAT_QUAL.get(_grp(msg, 'qualityInd', i), '?')} "
            f"used={_grp(msg, 'svUsed', i)} "
            f"health={_SAT_HEALTH.get(_grp(msg, 'health', i), '?')} "
            f"prRes={_num(_grp(msg, 'prRes', i), 1, 1)}m "
            f"orb={_ORBSRC.get(_grp(msg, 'orbitSource', i), '?')}"))
    header = (f"iTOW={_num(itow, 1e-3, 3)}s numSvs={n} (used {used})")
    return Decoded(header, body)


# --- NAV-SIG: per-signal info, one row per tracked signal ------------------

_SIG_CORR = {0: "none", 1: "SBAS", 2: "BeiDou", 3: "RTCM2", 4: "RTCM3-OSR",
             5: "RTCM3-SSR", 6: "QZSS-SLAS", 7: "SPARTN", 8: "CLAS"}
_SIG_IONO = {0: "none", 1: "Klob-GPS", 2: "SBAS", 3: "Klob-BDS",
             8: "dual-freq"}


@register("NAV-SIG")
def decode_nav_sig(msg, reserved=False):
    itow = _g(msg, "iTOW", 0)
    n = _g(msg, "numSigs", 0) or 0
    used = sum(1 for i in range(1, n + 1) if _grp(msg, "prUsed", i))
    body = [
        ("iTOW", _num(itow, 1e-3, 3), " s"),
        ("version", _g(msg, "version"), ""),
        ("numSigs", n, ""),
        ("sigsUsed", used, ""),
    ]
    if reserved:
        body.append(("reserved0", _g(msg, "reserved0"), ""))
    body.append(Section(f"signals ({n})"))
    for i in range(1, n + 1):
        gid = _grp(msg, "gnssId", i)
        body.append(Line(
            f"{_GNSS.get(gid, gid):<5} SV {_grp(msg, 'svId', i):>3} "
            f"sig {_grp(msg, 'sigId', i):>2} (fr {_grp(msg, 'freqId', i):>2}): "
            f"cno={_grp(msg, 'cno', i):>2} "
            f"qual={_SAT_QUAL.get(_grp(msg, 'qualityInd', i), '?')} "
            f"used={_grp(msg, 'prUsed', i)} "
            f"health={_SAT_HEALTH.get(_grp(msg, 'health', i), '?')} "
            f"corr={_SIG_CORR.get(_grp(msg, 'corrSource', i), '?')} "
            f"prRes={_num(_grp(msg, 'prRes', i), 1, 1)}m"))
    header = (f"iTOW={_num(itow, 1e-3, 3)}s numSigs={n} (used {used})")
    return Decoded(header, body)


# --- NAV-SBAS: SBAS status, one row per ranging satellite ------------------

_SBAS_MODE = {0: "disabled", 1: "enabled (integrity)", 3: "enabled (testmode)"}
_SBAS_SYS = {-1: "unknown", 0: "WAAS", 1: "EGNOS", 2: "MSAS", 3: "GAGAN",
             16: "GPS"}


@register("NAV-SBAS")
def decode_nav_sbas(msg, reserved=False):
    itow = _g(msg, "iTOW", 0)
    n = _g(msg, "cnt", 0) or 0
    svc = ",".join(k for k in ("Ranging", "Corrections", "Integrity",
                               "Testmode", "Bad") if _g(msg, k))
    body = [
        ("iTOW", _num(itow, 1e-3, 3), " s"),
        ("geo", _g(msg, "geo"), " (PRN)"),
        ("mode", _enum(_g(msg, "mode"), _SBAS_MODE), ""),
        ("sys", _enum(_g(msg, "sys"), _SBAS_SYS), ""),
        ("service", svc or "none", ""),
        ("integrityUsed", _g(msg, "integrityUsed"), ""),
        ("cnt", n, ""),
    ]
    if reserved:
        body.append(("reserved1", _g(msg, "reserved1"), ""))
    body.append(Section(f"ranging satellites ({n})"))
    for i in range(1, n + 1):
        body.append(Line(
            f"PRN {_grp(msg, 'svid', i):>3}: "
            f"udre={_grp(msg, 'udre', i):>2} "
            f"prc={_grp(msg, 'prc', i):>4} cm "
            f"ic={_grp(msg, 'ic', i):>4} cm "
            f"svSys={_grp(msg, 'svSys', i)} "
            f"svService={_grp(msg, 'svService', i)} "
            f"flags={_hexle(_grp(msg, 'flags', i) or 0)}"))
    header = (f"iTOW={_num(itow, 1e-3, 3)}s geo={_g(msg, 'geo')} "
              f"sys={_SBAS_SYS.get(_g(msg, 'sys'), '?')} "
              f"mode={_SBAS_MODE.get(_g(msg, 'mode'), '?')} cnt={n}")
    return Decoded(header, body)


# ===========================================================================
# RXM-*: receiver manager / raw measurement messages
# ===========================================================================

_MPATH = {0: "not measured", 1: "low", 2: "medium", 3: "high"}


# --- RXM-MEASX: satellite measurements for RRLP ---------------------------

@register("RXM-MEASX")
def decode_rxm_measx(msg, reserved=False):
    n = _g(msg, "numSv", 0) or 0
    gpstow  = _g(msg, "gpsTOW",  0)
    glotow  = _g(msg, "gloTOW",  0)
    bdstow  = _g(msg, "bdsTOW",  0)

    body = [
        ("version",    _g(msg, "version"),           ""),
        ("gpsTOW",     gpstow,                       " ms"),
        ("gpsTOWacc",  _g(msg, "gpsTOWacc"),         " ms"),
        ("gloTOW",     glotow,                       " ms"),
        ("gloTOWacc",  _g(msg, "gloTOWacc"),         " ms"),
        ("bdsTOW",     bdstow,                       " ms"),
        ("bdsTOWacc",  _g(msg, "bdsTOWacc"),         " ms"),
        ("qzssTOW",    _g(msg, "qzssTOW"),           " ms"),
        ("qzssTOWacc", _g(msg, "qzssTOWacc"),        " ms"),
        ("numSv",      n,                            ""),
        ("towSet",     _g(msg, "towSet"),             ""),
    ]
    if reserved:
        body += [
            ("reserved0", _g(msg, "reserved0"), ""),
            ("reserved1", _g(msg, "reserved1"), ""),
            ("reserved2", _g(msg, "reserved2"), ""),
            ("reserved3", _g(msg, "reserved3"), ""),
        ]
    body.append(Section(f"satellites ({n})"))
    for i in range(1, n + 1):
        gid  = _grp(msg, "gnssId", i)
        mpi  = _grp(msg, "mpathIndic", i)
        dms  = _grp(msg, "dopplerMS", i)
        dhz  = _grp(msg, "dopplerHz", i)
        cp   = _grp(msg, "codePhase", i)
        body.append(Line(
            f"{_GNSS.get(gid, gid):<5} SV {_grp(msg, 'svId', i):>3}: "
            f"cNo={_grp(msg, 'cNo', i):>2} "
            f"mpath={_MPATH.get(mpi, mpi)} "
            f"dopplerMS={_num(dms, 1, 3)}m/s "
            f"dopplerHz={_num(dhz, 1, 2)}Hz "
            f"wChips={_grp(msg, 'wholeChips', i)} "
            f"fChips={_grp(msg, 'fracChips', i)} "
            f"codePhase={_num(cp, 1, 9)}s "
            f"prrmsErr={_grp(msg, 'pseuRangeRMSErr', i)}"))
    header = (f"version={_g(msg, 'version')} numSv={n} towSet={_g(msg, 'towSet')} "
              f"gpsTOW={gpstow}ms gloTOW={glotow}ms bdsTOW={bdstow}ms")
    return Decoded(header, body)


# --- RXM-RAWX: multi-GNSS raw measurements ---------------------------------

@register("RXM-RAWX")
def decode_rxm_rawx(msg, reserved=False):
    n    = _g(msg, "numMeas", 0) or 0
    tow  = _g(msg, "rcvTow",  0.0)
    week = _g(msg, "week")

    body = [
        ("rcvTow",   _num(tow, 1, 6), " s"),
        ("week",     week,            ""),
        ("leapS",    _g(msg, "leapS"), " s"),
        ("numMeas",  n,               ""),
        ("leapSec",  _g(msg, "leapSec"), ""),
        ("clkReset", _g(msg, "clkReset"), ""),
    ]
    if reserved:
        body.append(("reserved1", _g(msg, "reserved1"), ""))
    body.append(Section(f"measurements ({n})"))
    for i in range(1, n + 1):
        gid  = _grp(msg, "gnssId", i)
        pr   = _grp(msg, "prMes",  i)
        cp   = _grp(msg, "cpMes",  i)
        do   = _grp(msg, "doMes",  i)
        pv   = _grp(msg, "prValid", i)
        cv   = _grp(msg, "cpValid", i)
        body.append(Line(
            f"{_GNSS.get(gid, gid):<5} SV {_grp(msg, 'svId', i):>3} "
            f"sig {_grp(msg, 'sigId', i):>2} "
            f"(fr {_grp(msg, 'freqId', i):>2}): "
            f"cno={_grp(msg, 'cno', i):>2} "
            f"lock={_grp(msg, 'locktime', i)}ms "
            f"pr={_num(pr, 1, 3)}m (valid={pv},std={_grp(msg, 'prStd', i)}) "
            f"cp={_num(cp, 1, 3)}cy (valid={cv},std={_grp(msg, 'cpStd', i)}) "
            f"do={_num(do, 1, 3)}Hz"))
    header = (f"rcvTow={_num(tow, 1, 6)}s week={week} "
              f"leapS={_g(msg, 'leapS')}s numMeas={n} "
              f"leapSec={_g(msg, 'leapSec')} clkReset={_g(msg, 'clkReset')}")
    return Decoded(header, body)


# ===========================================================================
# NAV-TIME*: GNSS-specific and UTC time solutions
# ===========================================================================

# --- NAV-TIMEBDS: BeiDou time solution -------------------------------------

@register("NAV-TIMEBDS")
def decode_nav_timebds(msg, reserved=False):
    itow = _g(msg, "iTOW", 0)
    sow = _g(msg, "SOW", 0)
    fields = [
        ("iTOW",       _num(itow, 1e-3, 3), " s"),
        ("SOW",        sow,                 " s"),
        ("fSOW",       _g(msg, "fSOW"),     " ns"),
        ("week",       _g(msg, "week"),     ""),
        ("leapS",      _g(msg, "leapS"),    " s"),
        ("sowValid",   _g(msg, "sowValid"), ""),
        ("weekValid",  _g(msg, "weekValid"), ""),
        ("leapSValid", _g(msg, "leapSValid"), ""),
        ("tAcc",       _g(msg, "tAcc"),     " ns"),
    ]
    header = (f"iTOW={_num(itow, 1e-3, 3)}s BDS SOW={sow}s "
              f"week={_g(msg, 'week')} leapS={_g(msg, 'leapS')}s "
              f"valid(sow={_g(msg, 'sowValid')},"
              f"wk={_g(msg, 'weekValid')},"
              f"ls={_g(msg, 'leapSValid')})")
    return Decoded(header, fields)


# --- NAV-TIMEGAL: Galileo time solution ------------------------------------

@register("NAV-TIMEGAL")
def decode_nav_timegal(msg, reserved=False):
    itow = _g(msg, "iTOW", 0)
    tow = _g(msg, "galTow", 0)
    fields = [
        ("iTOW",        _num(itow, 1e-3, 3),    " s"),
        ("galTow",      tow,                     " s"),
        ("fGalTow",     _g(msg, "fGalTow"),      " ns"),
        ("galWno",      _g(msg, "galWno"),        ""),
        ("leapS",       _g(msg, "leapS"),         " s"),
        ("galTowValid", _g(msg, "galTowValid"),   ""),
        ("galWnoValid", _g(msg, "galWnoValid"),   ""),
        ("leapSValid",  _g(msg, "leapSValid"),    ""),
        ("tAcc",        _g(msg, "tAcc"),          " ns"),
    ]
    header = (f"iTOW={_num(itow, 1e-3, 3)}s GAL TOW={tow}s "
              f"WNO={_g(msg, 'galWno')} leapS={_g(msg, 'leapS')}s "
              f"valid(tow={_g(msg, 'galTowValid')},"
              f"wno={_g(msg, 'galWnoValid')},"
              f"ls={_g(msg, 'leapSValid')})")
    return Decoded(header, fields)


# --- NAV-TIMEGLO: GLONASS time solution ------------------------------------

@register("NAV-TIMEGLO")
def decode_nav_timeglo(msg, reserved=False):
    itow = _g(msg, "iTOW", 0)
    tod = _g(msg, "TOD", 0)
    fields = [
        ("iTOW",      _num(itow, 1e-3, 3), " s"),
        ("TOD",       tod,                 " ms"),
        ("fTOD",      _g(msg, "fTOD"),     " ns"),
        ("Nt",        _g(msg, "Nt"),       " day"),
        ("N4",        _g(msg, "N4"),       ""),
        ("todValid",  _g(msg, "todValid"), ""),
        ("dateValid", _g(msg, "dateValid"), ""),
        ("tAcc",      _g(msg, "tAcc"),     " ns"),
    ]
    header = (f"iTOW={_num(itow, 1e-3, 3)}s GLO TOD={tod}ms "
              f"Nt={_g(msg, 'Nt')} N4={_g(msg, 'N4')} "
              f"valid(tod={_g(msg, 'todValid')},"
              f"date={_g(msg, 'dateValid')})")
    return Decoded(header, fields)


# --- NAV-TIMEGPS: GPS time solution ----------------------------------------

@register("NAV-TIMEGPS")
def decode_nav_timegps(msg, reserved=False):
    itow = _g(msg, "iTOW", 0)
    fields = [
        ("iTOW",       _num(itow, 1e-3, 3), " s"),
        ("fTOW",       _g(msg, "fTOW"),     " ns"),
        ("week",       _g(msg, "week"),     ""),
        ("leapS",      _g(msg, "leapS"),    " s"),
        ("towValid",   _g(msg, "towValid"), ""),
        ("weekValid",  _g(msg, "weekValid"), ""),
        ("leapSValid", _g(msg, "leapSValid"), ""),
        ("tAcc",       _g(msg, "tAcc"),     " ns"),
    ]
    header = (f"iTOW={_num(itow, 1e-3, 3)}s fTOW={_g(msg, 'fTOW')}ns "
              f"week={_g(msg, 'week')} leapS={_g(msg, 'leapS')}s "
              f"valid(tow={_g(msg, 'towValid')},"
              f"wk={_g(msg, 'weekValid')},"
              f"ls={_g(msg, 'leapSValid')})")
    return Decoded(header, fields)


# --- NAV-TIMELS: GPS leap-second event information -------------------------

_LS_SRC = {0: "default/factory", 1: "GPS/SBAS health", 2: "GPS", 3: "SBAS",
           4: "BeiDou", 5: "Galileo", 6: "GLONASS", 7: "LORAN-C"}


@register("NAV-TIMELS")
def decode_nav_timels(msg, reserved=False):
    itow = _g(msg, "iTOW", 0)
    curr = _g(msg, "currLs")
    chg = _g(msg, "lsChange")
    fields = [
        ("iTOW",               _num(itow, 1e-3, 3),              " s"),
        ("version",            _g(msg, "version"),               ""),
        ("srcOfCurrLs",        _enum(_g(msg, "srcOfCurrLs"), _LS_SRC), ""),
        ("currLs",             curr,                             " s"),
        ("srcOfLsChange",      _enum(_g(msg, "srcOfLsChange"), _LS_SRC), ""),
        ("lsChange",           chg,                             " s"),
        ("timeToLsEvent",      _g(msg, "timeToLsEvent"),        " s"),
        ("dateOfLsGpsWn",      _g(msg, "dateOfLsGpsWn"),        ""),
        ("dateOfLsGpsDn",      _g(msg, "dateOfLsGpsDn"),        ""),
        ("validCurrLs",        _g(msg, "validCurrLs"),          ""),
        ("validTimeToLsEvent", _g(msg, "validTimeToLsEvent"),   ""),
    ]
    if reserved:
        fields += [
            ("reserved0", _g(msg, "reserved0"), ""),
            ("reserved1", _g(msg, "reserved1"), ""),
        ]
    chg_s = f"{chg:+d}" if isinstance(chg, int) else str(chg)
    header = (f"iTOW={_num(itow, 1e-3, 3)}s currLs={curr}s "
              f"lsChange={chg_s}s ttl={_g(msg, 'timeToLsEvent')}s "
              f"valid(curr={_g(msg, 'validCurrLs')},"
              f"ttl={_g(msg, 'validTimeToLsEvent')})")
    return Decoded(header, fields)


# --- NAV-TIMENAVIC: NavIC time solution ------------------------------------

@register("NAV-TIMENAVIC")
def decode_nav_timenavic(msg, reserved=False):
    itow = _g(msg, "iTOW", 0)
    tow = _g(msg, "NavICTow", 0)
    fields = [
        ("iTOW",         _num(itow, 1e-3, 3),     " s"),
        ("NavICTow",     tow,                      " s"),
        ("fNavICTow",    _g(msg, "fNavICTow"),     " ns"),
        ("NavICWno",     _g(msg, "NavICWno"),      ""),
        ("leapS",        _g(msg, "leapS"),         " s"),
        ("NavICTowValid", _g(msg, "NavICTowValid"), ""),
        ("NavICWnoValid", _g(msg, "NavICWnoValid"), ""),
        ("leapSValid",   _g(msg, "leapSValid"),    ""),
        ("tAcc",         _g(msg, "tAcc"),          " ns"),
    ]
    header = (f"iTOW={_num(itow, 1e-3, 3)}s NavIC TOW={tow}s "
              f"WNO={_g(msg, 'NavICWno')} leapS={_g(msg, 'leapS')}s "
              f"valid(tow={_g(msg, 'NavICTowValid')},"
              f"wno={_g(msg, 'NavICWnoValid')},"
              f"ls={_g(msg, 'leapSValid')})")
    return Decoded(header, fields)


# --- NAV-TIMEUTC: UTC time solution ----------------------------------------

_UTC_STD = {0: "not available", 1: "CRL/NICT", 2: "NIST", 3: "USNO",
            4: "BIPM", 5: "EU", 6: "SU", 7: "NTSC", 15: "receiver internal"}


@register("NAV-TIMEUTC")
def decode_nav_timeutc(msg, reserved=False):
    itow = _g(msg, "iTOW", 0)
    try:
        ts = (f"{msg.year:04d}-{msg.month:02d}-{msg.day:02d}T"
              f"{msg.hour:02d}:{msg.min:02d}:{msg.sec:02d}")
    except Exception:
        ts = "?"
    std = _g(msg, "utcStandard")
    fields = [
        ("iTOW",        _num(itow, 1e-3, 3), " s"),
        ("tAcc",        _g(msg, "tAcc"),     " ns"),
        ("nano",        _g(msg, "nano"),     " ns"),
        ("time",        ts,                  ""),
        ("utcStandard", _enum(std, _UTC_STD), ""),
        ("validTOW",    _g(msg, "validTOW"), ""),
        ("validWKN",    _g(msg, "validWKN"), ""),
        ("validUTC",    _g(msg, "validUTC"), ""),
    ]
    header = (f"iTOW={_num(itow, 1e-3, 3)}s {ts} "
              f"tAcc={_g(msg, 'tAcc')}ns "
              f"utcStd={_UTC_STD.get(std, std)} "
              f"valid(tow={_g(msg, 'validTOW')},"
              f"wkn={_g(msg, 'validWKN')},"
              f"utc={_g(msg, 'validUTC')})")
    return Decoded(header, fields)


# ===========================================================================
# SEC-*: security messages
# ===========================================================================

# --- SEC-SIG: signal security status (jamming and spoofing detection) ------

@register("SEC-SIG")
def decode_sec_sig(msg, reserved=False):
    jam = _g(msg, "jammingState")
    spf = _g(msg, "spoofingState")
    fields = [
        ("version",        _g(msg, "version"),       ""),
        ("jammingState",   _enum(jam, _HW_JAMMING),  ""),
        ("jamDetEnabled",  _g(msg, "jamDetEnabled"),  ""),
        ("jamNumCentFreqs", _g(msg, "jamNumCentFreqs"), ""),
        ("spoofingState",  _enum(spf, _SPOOF),        ""),
        ("spfDetEnabled",  _g(msg, "spfDetEnabled"),  ""),
    ]
    if reserved:
        fields.append(("reserved0", _g(msg, "reserved0"), ""))
    header = (f"jam={_HW_JAMMING.get(jam, jam)} "
              f"(det={'on' if _g(msg, 'jamDetEnabled') else 'off'}, "
              f"freqs={_g(msg, 'jamNumCentFreqs')})  "
              f"spoof={_SPOOF.get(spf, spf)} "
              f"(det={'on' if _g(msg, 'spfDetEnabled') else 'off'})")
    return Decoded(header, fields)


# ===========================================================================
# TIM-*: timing messages
# ===========================================================================

# --- TIM-SVIN: survey-in data ----------------------------------------------

@register("TIM-SVIN")
def decode_tim_svin(msg, reserved=False):
    x  = _g(msg, "meanX",  0)
    y  = _g(msg, "meanY",  0)
    z  = _g(msg, "meanZ",  0)
    mv = _g(msg, "meanV",  0)
    fields = [
        ("dur",    _g(msg, "dur"),          " s"),
        ("meanX",  _num(x,  1e-2, 2),       " m"),
        ("meanY",  _num(y,  1e-2, 2),       " m"),
        ("meanZ",  _num(z,  1e-2, 2),       " m"),
        ("meanV",  mv,                      " mm²"),
        ("obs",    _g(msg, "obs"),           ""),
        ("valid",  _g(msg, "valid"),         ""),
        ("active", _g(msg, "active"),        ""),
    ]
    if reserved:
        fields.append(("reserved1", _g(msg, "reserved1"), ""))
    header = (f"dur={_g(msg, 'dur')}s obs={_g(msg, 'obs')} "
              f"pos=({_num(x, 1e-2, 2)}, {_num(y, 1e-2, 2)}, "
              f"{_num(z, 1e-2, 2)}) m  "
              f"meanV={mv}mm²  "
              f"valid={_g(msg, 'valid')} active={_g(msg, 'active')}")
    return Decoded(header, fields)


# --- TIM-TP: time pulse time data ------------------------------------------

_TP_TIMEBASE  = {0: "GNSS", 1: "UTC"}
_TP_RAIM      = {0: "not available", 1: "inactive", 2: "active", 3: "unknown"}
_TP_REFGNSS   = {0: "GPS", 1: "GLONASS", 2: "BeiDou", 3: "Galileo",
                 4: "NavIC", 15: "unknown"}


@register("TIM-TP")
def decode_tim_tp(msg, reserved=False):
    tow_ms  = _g(msg, "towMS",    0)
    sub_ms  = _g(msg, "towSubMS", 0.0)
    qerr    = _g(msg, "qErr",     0)
    week    = _g(msg, "week")
    tb      = _g(msg, "timeBase")
    raim    = _g(msg, "raim")
    refgnss = _g(msg, "timeRefGnss")
    utcstd  = _g(msg, "utcStandard")
    fields = [
        ("towMS",       tow_ms,                         " ms"),
        ("towSubMS",    _num(sub_ms, 1, 9),             " ms"),
        ("qErr",        qerr,                           " ps"),
        ("week",        week,                           ""),
        ("timeBase",    _enum(tb,      _TP_TIMEBASE),   ""),
        ("utc",         _g(msg, "utc"),                 ""),
        ("raim",        _enum(raim,    _TP_RAIM),        ""),
        ("qErrInvalid", _g(msg, "qErrInvalid"),         ""),
        ("TpNotLocked", _g(msg, "TpNotLocked"),         ""),
        ("timeRefGnss", _enum(refgnss, _TP_REFGNSS),    ""),
        ("utcStandard", _enum(utcstd,  _UTC_STD),        ""),
    ]
    # Full TOW in seconds = (towMS + towSubMS) / 1000; towSubMS is sub-ms.
    tow_s = (tow_ms + (sub_ms or 0.0)) / 1000.0
    header = (f"TOW={tow_s:.9f}s week={week} "
              f"qErr={qerr}ps "
              f"base={_TP_TIMEBASE.get(tb, tb)} "
              f"ref={_TP_REFGNSS.get(refgnss, refgnss)} "
              f"raim={_TP_RAIM.get(raim, raim)} "
              f"locked={not _g(msg, 'TpNotLocked')}")
    return Decoded(header, fields)
