"""SBAS RXM-SFRBX decoder: L1 messages, CRC-24Q checked."""

import math

from .bits import _bs, _bu, _crc24q_head, _words_int


# ---------------------------------------------------------------------------
# SBAS L1 (RXM-SFRBX, gnssId=1) decoder, per RTCA DO-229 / SBAS MOPS.  Each
# message is 250 bits: 8-bit preamble + 6-bit message type + 212-bit data +
# 24-bit CRC-24Q computed over the first 226 bits.  The preamble cycles through
# 0x53, 0x9A, 0xC6 over three successive one-second messages.  Field layouts and
# scale factors follow RTKLIB's sbas.c (types 1-7, 9, 18, 25, 26) and DO-229
# (types 10, 17).  u-blox delivers the 250-bit message MSB-first in numWords=9
# 32-bit words (288 bits); the message is the top 250 bits.
# ---------------------------------------------------------------------------

SBAS_PREAMBLES = (0x53, 0x9A, 0xC6)

SBAS_MSGTYPE = {
    0:  "Test mode (do not use)",
    1:  "PRN mask",
    2:  "Fast corrections 2",
    3:  "Fast corrections 3",
    4:  "Fast corrections 4",
    5:  "Fast corrections 5",
    6:  "Integrity info",
    7:  "Fast-correction degradation",
    9:  "GEO navigation (ephemeris)",
    10: "Degradation parameters",
    12: "SBAS network time/UTC",
    17: "GEO almanacs",
    18: "Ionospheric grid mask",
    24: "Mixed fast/long-term corrections",
    25: "Long-term satellite corrections",
    26: "Ionospheric delay corrections",
    27: "SBAS service message",
    28: "Clock-ephemeris covariance",
    62: "Internal test message",
    63: "Null message",
}


def _sbas_prn_label(i):
    """Map a 1-based PRN-mask slot (DO-229) to a readable satellite id."""
    if 1 <= i <= 37:
        return f"G{i:02d}"          # GPS
    if 38 <= i <= 61:
        return f"R{i - 37:02d}"     # GLONASS
    if 120 <= i <= 138:
        return f"S{i:d}"            # SBAS/GEO
    if 183 <= i <= 192:
        return f"J{i - 182:02d}"    # QZSS (SBAS ranging ref)
    if 193 <= i <= 202:
        return f"J{i - 192:02d}"    # QZSS
    return f"#{i}"


def decode_sbas(raw):
    """Decode one SBAS L1 message from RXM-SFRBX raw 32-bit words.

    Returns (mt, crc_ok, preamble, fields) where fields is a list of
    (name, value, unit) cells ready for emit()/_columns().
    """
    nbits = len(raw) * 32
    v = _words_int(raw) >> (nbits - 250)        # top 250 bits, MSB-first
    pre = _bu(v, 250, 0, 8)
    mt = _bu(v, 250, 8, 6)

    # CRC-24Q over the first 226 bits (preamble+type+data) vs bits 226-249.
    crc_ok = _crc24q_head(v, 250, 226) == _bu(v, 250, 226, 24)

    def bu(p, l):
        return _bu(v, 250, p, l)

    def bs(p, l):
        return _bs(v, 250, p, l)

    out = []

    if mt == 1:                                  # PRN mask assignments
        prns = [_sbas_prn_label(i) for i in range(1, 211) if bu(13 + i, 1)]
        out += [("IODP", bu(224, 2), ""), ("nSat", len(prns), ""),
                ("PRNs", " ".join(prns), "")]

    elif mt in (0, 2, 3, 4, 5):                  # fast corrections
        t = 2 if mt == 0 else mt
        prc = [bs(18 + i * 12, 12) * 0.125 for i in range(13)]
        udrei = [bu(174 + i * 4, 4) for i in range(13)]
        out += [("IODF", bu(14, 2), ""), ("IODP", bu(16, 2), ""),
                ("slotBase", 13 * (t - 2), ""),
                ("PRC(m)", " ".join(f"{x:+g}" for x in prc), ""),
                ("UDREI", " ".join(str(x) for x in udrei), "")]

    elif mt == 7:                                # fast-correction degradation
        ai = [bu(22 + i * 4, 4) for i in range(51)]
        out += [("tlat", bu(14, 4), " s"), ("IODP", bu(18, 2), ""),
                ("ai[0:51]", " ".join(str(x) for x in ai), "")]

    elif mt == 9:                                # GEO navigation (ephemeris)
        x, y, z = bs(39, 30) * 0.08, bs(69, 30) * 0.08, bs(99, 25) * 0.4
        out += [
            ("t0", bu(22, 13) * 16, " s"), ("URA", bu(35, 4), ""),
            ("X", x, " m"), ("Y", y, " m"), ("Z", z, " m"),
            ("Vx", bs(124, 17) * 0.000625, " m/s"),
            ("Vy", bs(141, 17) * 0.000625, " m/s"),
            ("Vz", bs(158, 18) * 0.004, " m/s"),
            ("Ax", bs(176, 10) * 0.0000125, " m/s2"),
            ("Ay", bs(186, 10) * 0.0000125, " m/s2"),
            ("Az", bs(196, 10) * 0.0000625, " m/s2"),
            ("af0", bs(206, 12) * 2 ** -31, " s"),
            ("af1", bs(218, 8) * 2 ** -39 / 2.0, " s/s"),
            ("radius", math.sqrt(x * x + y * y + z * z), " m"),
        ]

    elif mt == 10:                               # degradation parameters
        for nm, p, l, sc, u in (
            ("Brrc", 14, 10, 0.002, " m"), ("Cltc_lsb", 24, 10, 0.002, " m"),
            ("Cltc_v1", 34, 10, 0.00005, " m/s"), ("Iltc_v1", 44, 9, 1.0, " s"),
            ("Cltc_v0", 53, 10, 0.002, " m"), ("Iltc_v0", 63, 9, 1.0, " s"),
            ("Cgeo_lsb", 72, 10, 0.0005, " m"), ("Cgeo_v", 82, 10, 0.00005, " m/s"),
            ("Igeo", 92, 9, 1.0, " s"), ("Cer", 101, 6, 0.5, " m"),
            ("Ciono_step", 107, 10, 0.001, " m"), ("Iiono", 117, 9, 1.0, " s"),
            ("Ciono_ramp", 126, 10, 0.000005, " m/s"),
        ):
            out.append((nm, bu(p, l) * sc, u))
        out += [("RSS_UDRE", bu(136, 1), ""), ("RSS_iono", bu(137, 1), "")]

    elif mt == 17:                               # GEO almanacs (up to 3)
        for k in range(3):
            p = 14 + k * 67
            prn = bu(p + 2, 8)
            if prn == 0:
                continue
            ax, ay = bs(p + 18, 15) * 2600.0, bs(p + 33, 15) * 2600.0
            az = bs(p + 48, 9) * 26000.0
            out += [
                (f"PRN{k}", prn, ""), (f"health{k}", f"0x{bu(p + 10, 8):02X}", ""),
                (f"X{k}", ax, " m"), (f"Y{k}", ay, " m"), (f"Z{k}", az, " m"),
                (f"Vx{k}", bs(p + 57, 3) * 10.0, " m/s"),
                (f"Vy{k}", bs(p + 60, 3) * 10.0, " m/s"),
                (f"Vz{k}", bs(p + 63, 4) * 60.0, " m/s"),
                (f"radius{k}", math.sqrt(ax * ax + ay * ay + az * az), " m"),
            ]
        out.append(("t0", bu(14 + 3 * 67, 11) * 64, " s"))
        if len(out) == 1:
            out.insert(0, ("note", "no almanac entries (all PRN=0)", ""))

    elif mt == 18:                               # ionospheric grid-point mask
        igps = [i for i in range(1, 202) if bu(23 + i, 1)]
        out += [("nBands", bu(14, 4), ""), ("band", bu(18, 4), ""),
                ("IODI", bu(22, 2), ""), ("nIGP", len(igps), ""),
                ("IGPset", " ".join(str(i) for i in igps), "")]

    elif mt == 25:                               # long-term satellite corrections
        for p in (14, 120):                      # two half-message blocks
            if bu(p, 1) == 0:                    # velocity code 0: two sats, no velocity
                for sub in (p + 1, p + 52):
                    n = bu(sub, 6)
                    if n == 0:
                        continue
                    out += [
                        ("PRNmask", n, ""), ("IODE", bu(sub + 6, 8), ""),
                        ("dX", bs(sub + 14, 9) * 0.125, " m"),
                        ("dY", bs(sub + 23, 9) * 0.125, " m"),
                        ("dZ", bs(sub + 32, 9) * 0.125, " m"),
                        ("daf0", bs(sub + 41, 10) * 2 ** -31, " s"),
                    ]
            else:                                # velocity code 1: one sat with velocity
                n = bu(p + 1, 6)
                if n != 0:
                    t = bu(p + 91, 13) * 16
                    out += [
                        ("PRNmask", n, ""), ("IODE", bu(p + 7, 8), ""),
                        ("dX", bs(p + 15, 11) * 0.125, " m"),
                        ("dY", bs(p + 26, 11) * 0.125, " m"),
                        ("dZ", bs(p + 37, 11) * 0.125, " m"),
                        ("dVx", bs(p + 59, 8) * 2 ** -11, " m/s"),
                        ("dVy", bs(p + 67, 8) * 2 ** -11, " m/s"),
                        ("dVz", bs(p + 75, 8) * 2 ** -11, " m/s"),
                        ("daf0", bs(p + 48, 11) * 2 ** -31, " s"),
                        ("daf1", bs(p + 83, 8) * 2 ** -39, " s/s"),
                        ("t0", t, " s"),
                    ]
        if not out:
            out.append(("note", "no satellite corrections in this message", ""))

    elif mt == 26:                               # ionospheric delay corrections
        delays, gives = [], []
        for i in range(15):
            d = bu(22 + i * 13, 9)
            delays.append("n/a" if d == 0x1FF else f"{d * 0.125:g}")
            gives.append(str(bu(22 + i * 13 + 9, 4)))
        out += [("band", bu(14, 4), ""), ("block", bu(18, 4), ""),
                ("IODI", bu(217, 2), ""),
                ("delay(m)", " ".join(delays), ""),
                ("GIVEI", " ".join(gives), "")]

    elif mt == 63:                               # null message
        out.append(("note", "null message (no data)", ""))

    else:
        out.append(("note",
                    f"{SBAS_MSGTYPE.get(mt, f'type {mt}')} (payload not decoded)", ""))

    return mt, crc_ok, pre, out
