"""GPS / QZSS RXM-SFRBX decoders: L1 C/A LNAV and L2C/L5 CNAV."""

from .bits import SC, _cs, _cu, _s, _twos, _u, cnav_crc_ok, cnav_message_int


SUBFRAME_NAME = {
    1: "clock/health",
    2: "ephemeris I",
    3: "ephemeris II",
    4: "almanac/other",
    5: "almanac",
}

# Nominal URA (user range accuracy) in metres, indexed by the 4-bit URA index.
URA_M = {0: 2.0, 1: 2.8, 2: 4.0, 3: 5.7, 4: 8.0, 5: 11.3, 6: 16.0, 7: 32.0,
         8: 64.0, 9: 128.0, 10: 256.0, 11: 512.0, 12: 1024.0, 13: 2048.0,
         14: 4096.0, 15: float("inf")}


def _ascii_maybe(val, nbits, minbytes=2):
    """Return printable-ASCII text if the field's bytes look like text, else None.

    The field is read as whole bytes the way its hex is shown: right-aligned over
    its full width (e.g. the 23-bit 0x3B463F -> bytes 3B 46 3F -> ';F?').  When the
    width isn't a multiple of 8, the low whole bytes are also tried (dropping the
    partial high bits, so text packed into e.g. the low 48 bits of a 51-bit field
    is still found).  The first candidate that is all printable ASCII (>= minbytes
    bytes, 0x20-0x7E) is returned.
    """
    candidates = [val.to_bytes((nbits + 7) // 8, "big")]
    nlow = nbits // 8
    if nlow and nlow != (nbits + 7) // 8:
        candidates.append((val & ((1 << (nlow * 8)) - 1)).to_bytes(nlow, "big"))
    for bs in candidates:
        if len(bs) >= minbytes and all(0x20 <= b <= 0x7E for b in bs):
            return bs.decode("ascii")
    return None


def _hexbits(val, nbits):
    """Format a reserved/raw bit field as zero-padded hex tagged with its width.

    When the contents look like printable ASCII, the text is appended in quotes.
    """
    s = f"0x{val:0{(nbits + 3) // 4}X} ({nbits}b)"
    text = _ascii_maybe(val, nbits)
    if text is not None:
        s += f" '{text}'"
    return s


def decode_gps_lnav(words, reserved=False):
    """Decode one GPS L1 C/A subframe (10 x 24-bit data words).

    Returns (subframe_id, tow_seconds, list_of_("field", value, unit) tuples).
    Reserved/spare fields are included only when `reserved` is true.
    """
    if len(words) < 10:
        return None, None, [("error", f"only {len(words)} words", "")]

    preamble = _u(words, 0, 1, 8)
    tlm = _u(words, 0, 9, 14)
    tow_count = _u(words, 1, 1, 17)      # truncated TOW of NEXT subframe (x6 s)
    alert = _u(words, 1, 18, 1)
    antispoof = _u(words, 1, 19, 1)
    sfid = _u(words, 1, 20, 3)
    tow_s = tow_count * 6

    out = [("preamble", f"0x{preamble:02X}{'' if preamble == 0x8B else ' (BAD)'}", ""),
           ("TLM", f"0x{tlm:04X}", ""),
           ("alert", alert, ""), ("A-S", antispoof, "")]

    if sfid == 1:  # clock and health
        wn = _u(words, 2, 1, 10)
        codeL2 = _u(words, 2, 11, 2)
        ura = _u(words, 2, 13, 4)
        health = _u(words, 2, 17, 6)
        iodc = (_u(words, 2, 23, 2) << 8) | _u(words, 7, 1, 8)
        l2p = _u(words, 3, 1, 1)
        tgd = _s(words, 6, 17, 8) * 2 ** -31
        toc = _u(words, 7, 9, 16) * 2 ** 4
        af2 = _s(words, 8, 1, 8) * 2 ** -55
        af1 = _s(words, 8, 9, 16) * 2 ** -43
        af0 = _s(words, 9, 1, 22) * 2 ** -31
        out += [
            ("WN", wn, " (mod 1024)"), ("codeOnL2", codeL2, ""),
            ("URAindex", f"{ura} (~{URA_M[ura]:g} m)", ""),
            ("SVhealth", f"0x{health:02X}", ""), ("IODC", iodc, ""), ("L2Pdata", l2p, ""),
            ("Tgd", tgd, " s"), ("toc", toc, " s"),
            ("af0", af0, " s"), ("af1", af1, " s/s"), ("af2", af2, " s/s^2"),
        ]
        if reserved:
            out += [
                ("reserved_w4", _hexbits(_u(words, 3, 2, 23), 23), ""),
                ("reserved_w5", _hexbits(_u(words, 4, 1, 24), 24), ""),
                ("reserved_w6", _hexbits(_u(words, 5, 1, 24), 24), ""),
                ("reserved_w7", _hexbits(_u(words, 6, 1, 16), 16), ""),
            ]

    elif sfid == 2:  # ephemeris I
        iode = _u(words, 2, 1, 8)
        crs = _s(words, 2, 9, 16) * 2 ** -5
        dn = _s(words, 3, 1, 16) * 2 ** -43 * SC
        m0 = _twos((_u(words, 3, 17, 8) << 24) | _u(words, 4, 1, 24), 32) * 2 ** -31 * SC
        cuc = _s(words, 5, 1, 16) * 2 ** -29
        ecc = ((_u(words, 5, 17, 8) << 24) | _u(words, 6, 1, 24)) * 2 ** -33
        cus = _s(words, 7, 1, 16) * 2 ** -29
        sqrta = ((_u(words, 7, 17, 8) << 24) | _u(words, 8, 1, 24)) * 2 ** -19
        toe = _u(words, 9, 1, 16) * 2 ** 4
        fit = _u(words, 9, 17, 1)
        aodo = _u(words, 9, 18, 5) * 900
        out += [
            ("IODE", iode, ""), ("Crs", crs, " m"), ("dn", dn, " rad/s"),
            ("M0", m0, " rad"), ("Cuc", cuc, " rad"), ("e", ecc, ""),
            ("Cus", cus, " rad"), ("sqrtA", sqrta, " m^0.5"),
            ("toe", toe, " s"), ("fitInterval", fit, ""), ("AODO", aodo, " s"),
        ]

    elif sfid == 3:  # ephemeris II
        cic = _s(words, 2, 1, 16) * 2 ** -29
        omega0 = _twos((_u(words, 2, 17, 8) << 24) | _u(words, 3, 1, 24), 32) * 2 ** -31 * SC
        cis = _s(words, 4, 1, 16) * 2 ** -29
        i0 = _twos((_u(words, 4, 17, 8) << 24) | _u(words, 5, 1, 24), 32) * 2 ** -31 * SC
        crc = _s(words, 6, 1, 16) * 2 ** -5
        argp = _twos((_u(words, 6, 17, 8) << 24) | _u(words, 7, 1, 24), 32) * 2 ** -31 * SC
        omegadot = _s(words, 8, 1, 24) * 2 ** -43 * SC
        iode = _u(words, 9, 1, 8)
        idot = _s(words, 9, 9, 14) * 2 ** -43 * SC
        out += [
            ("Cic", cic, " rad"), ("Omega0", omega0, " rad"), ("Cis", cis, " rad"),
            ("i0", i0, " rad"), ("Crc", crc, " m"), ("omega", argp, " rad"),
            ("OmegaDot", omegadot, " rad/s"), ("IODE", iode, ""), ("IDOT", idot, " rad/s"),
        ]

    elif sfid in (4, 5):  # almanac and special pages
        data_id = _u(words, 2, 1, 2)
        sv_id = _u(words, 2, 3, 6)   # page / almanac PRN selector
        out += [("dataID", data_id, ""), ("pageSVID", sv_id, "")]
        if 1 <= sv_id <= 32:  # almanac for a single SV
            ecc = _u(words, 2, 9, 16) * 2 ** -21
            toa = _u(words, 3, 1, 8) * 2 ** 12
            deltai = _s(words, 3, 9, 16) * 2 ** -19 * SC
            omegadot = _s(words, 4, 1, 16) * 2 ** -38 * SC
            health = _u(words, 4, 17, 8)
            sqrta = _u(words, 5, 1, 24) * 2 ** -11
            omega0 = _s(words, 6, 1, 24) * 2 ** -23 * SC
            argp = _s(words, 7, 1, 24) * 2 ** -23 * SC
            m0 = _s(words, 8, 1, 24) * 2 ** -23 * SC
            af0 = _twos((_u(words, 9, 1, 8) << 3) | _u(words, 9, 20, 3), 11) * 2 ** -20
            af1 = _s(words, 9, 9, 11) * 2 ** -38
            out += [
                ("almanacPRN", sv_id, ""), ("e", ecc, ""), ("toa", toa, " s"),
                ("deltai", deltai, " rad"), ("OmegaDot", omegadot, " rad/s"),
                ("SVhealth", f"0x{health:02X}", ""), ("sqrtA", sqrta, " m^0.5"),
                ("Omega0", omega0, " rad"), ("omega", argp, " rad"), ("M0", m0, " rad"),
                ("af0", af0, " s"), ("af1", af1, " s/s"),
            ]
        elif sfid == 4 and sv_id == 56:  # ionosphere + UTC parameters
            a0 = _s(words, 2, 9, 8) * 2 ** -30
            a1 = _s(words, 2, 17, 8) * 2 ** -27
            a2 = _s(words, 3, 1, 8) * 2 ** -24
            a3 = _s(words, 3, 9, 8) * 2 ** -24
            b0 = _s(words, 3, 17, 8) * 2 ** 11
            b1 = _s(words, 4, 1, 8) * 2 ** 14
            b2 = _s(words, 4, 9, 8) * 2 ** 16
            b3 = _s(words, 4, 17, 8) * 2 ** 16
            A1 = _s(words, 5, 1, 24) * 2 ** -50
            A0 = _twos((_u(words, 6, 1, 24) << 8) | _u(words, 7, 1, 8), 32) * 2 ** -30
            tot = _u(words, 7, 9, 8) * 2 ** 12
            wnt = _u(words, 7, 17, 8)
            dtls = _s(words, 8, 1, 8)
            wnlsf = _u(words, 8, 9, 8)
            dn = _u(words, 8, 17, 8)
            dtlsf = _s(words, 9, 1, 8)
            out += [
                ("iono_alpha", f"[{a0:.3g}, {a1:.3g}, {a2:.3g}, {a3:.3g}]", ""),
                ("iono_beta", f"[{b0:.3g}, {b1:.3g}, {b2:.3g}, {b3:.3g}]", ""),
                ("UTC_A0", A0, " s"), ("UTC_A1", A1, " s/s"), ("tot", tot, " s"),
                ("WNt", wnt, ""), ("dtLS", dtls, " s"), ("WNLSF", wnlsf, ""),
                ("DN", dn, ""), ("dtLSF", dtlsf, " s"),
            ]
            if reserved:
                out += [("reserved_w10", _hexbits(_u(words, 9, 9, 14), 14), "")]
        else:
            out += [("note", "reserved/other page (not decoded)", "")]

    return sfid, tow_s, out


# ---------------------------------------------------------------------------
# GPS / QZSS CNAV decoder (RXM-SFRBX on L2 CM / L5, e.g. GPS sigId=4), per
# IS-GPS-200M Appendix III.  Unlike LNAV, CNAV uses the full 32 bits of each
# word, left-justified: the 10 words form a 320-bit string whose top 300 bits
# are the CNAV message (the bottom 20 are padding).  Bit fields below are
# 1-indexed from the MSB of that 300-bit message.  A 24-bit CRC (CRC-24Q,
# bits 277-300) protects the whole message.
# ---------------------------------------------------------------------------

CNAV_MSGTYPE = {
    0: "Default (no data)",
    10: "Ephemeris 1",
    11: "Ephemeris 2",
    12: "Reduced Almanac",
    13: "Clock Differential Correction",
    14: "Ephemeris Differential Correction",
    15: "Text",
    30: "Clock, IONO & Group Delay",
    31: "Clock & Reduced Almanac",
    32: "Clock & EOP",
    33: "Clock & UTC",
    34: "Clock & Differential Correction",
    35: "Clock & GGTO",
    36: "Clock & Text",
    37: "Clock & Midi Almanac",
}

CNAV_A_REF = 26559710.0          # m  (semi-major axis reference, IS-GPS-200M)
QZSS_CNAV_A_REF = 42164200.0     # m  (geosynchronous reference, IS-QZSS-PNT-004)
CNAV_OMEGA_DOT_REF = -2.6e-9     # semicircles/s (rate-of-RAAN reference; same QZSS)

# QZSS reuses the GPS CNAV message types above and adds its own (IS-QZSS-PNT).
# Their bodies are QZSS-proprietary; only the shared CNAV header + CRC decode.
QZSS_CNAV_MSGTYPE = {
    60: "QZSS-specific",
    61: "QZSS-specific",
}


def _cnav_clock_block(m):
    """Decode the clock block (bits 39-127) shared by message types 30-37."""
    top = _cu(m, 39, 11) * 300
    ned0 = _cs(m, 50, 5)
    ned1 = _cu(m, 55, 3)
    ned2 = _cu(m, 58, 3)
    toc = _cu(m, 61, 11) * 300
    af0 = _cs(m, 72, 26) * 2 ** -35
    af1 = _cs(m, 98, 20) * 2 ** -48
    af2 = _cs(m, 118, 10) * 2 ** -60
    return [
        ("t_op", top, " s"), ("URA_NED0", ned0, ""),
        ("URA_NED1", ned1, ""), ("URA_NED2", ned2, ""),
        ("toc", toc, " s"), ("af0", af0, " s"),
        ("af1", af1, " s/s"), ("af2", af2, " s/s^2"),
    ]


def decode_gps_cnav(dwrds, reserved=False, a_ref=CNAV_A_REF):
    """Decode one GPS/QZSS CNAV message (10 x 32-bit words).

    Returns (msg_type, tow_seconds, list_of_("field", value, unit) tuples).
    Reserved/spare fields are included only when `reserved` is true.
    `a_ref` is the semi-major-axis reference (GPS vs QZSS differ; MT10).
    """
    if len(dwrds) != 10:
        return None, None, [("error", f"need 10 words, got {len(dwrds)}", "")]

    m = cnav_message_int(dwrds)
    preamble = _cu(m, 1, 8)
    prn = _cu(m, 9, 6)
    mtype = _cu(m, 15, 6)
    tow_s = _cu(m, 21, 17) * 6
    alert = _cu(m, 38, 1)
    crc_ok = cnav_crc_ok(m)

    out = [("preamble", f"0x{preamble:02X}{'' if preamble == 0x8B else ' (BAD)'}", ""),
           ("PRN", prn, ""), ("alert", alert, ""),
           ("CRC", "OK" if crc_ok else "FAIL", "")]

    if mtype == 10:  # Ephemeris 1
        wn = _cu(m, 39, 13)
        health = _cu(m, 52, 3)
        top = _cu(m, 55, 11) * 300
        ura_ed = _cs(m, 66, 5)
        toe = _cu(m, 71, 11) * 300
        dA = _cs(m, 82, 26) * 2 ** -9
        adot = _cs(m, 108, 25) * 2 ** -21
        dn0 = _cs(m, 133, 17) * 2 ** -44 * SC
        dn0dot = _cs(m, 150, 23) * 2 ** -57 * SC
        m0 = _cs(m, 173, 33) * 2 ** -32 * SC
        ecc = _cu(m, 206, 33) * 2 ** -34
        omega = _cs(m, 239, 33) * 2 ** -32 * SC
        integ = _cu(m, 272, 1)
        l2phase = _cu(m, 273, 1)
        hl = f"L1={'bad' if health & 4 else 'ok'} " \
             f"L2={'bad' if health & 2 else 'ok'} L5={'bad' if health & 1 else 'ok'}"
        out += [
            ("WN", wn, ""), ("health", hl, ""), ("t_op", top, " s"),
            ("URA_ED", ura_ed, ""), ("toe", toe, " s"),
            ("A", a_ref + dA, " m"), ("dA", dA, " m"), ("Adot", adot, " m/s"),
            ("dn0", dn0, " rad/s"), ("dn0dot", dn0dot, " rad/s^2"),
            ("M0", m0, " rad"), ("e", ecc, ""), ("omega", omega, " rad"),
            ("integrity", integ, ""), ("L2phasing", l2phase, ""),
        ]
        if reserved:
            out += [("reserved", _hexbits(_cu(m, 274, 3), 3), "")]

    elif mtype == 11:  # Ephemeris 2
        toe = _cu(m, 39, 11) * 300
        omega0 = _cs(m, 50, 33) * 2 ** -32 * SC
        i0 = _cs(m, 83, 33) * 2 ** -32 * SC
        domega_dot = _cs(m, 116, 17) * 2 ** -44
        omega_dot = (CNAV_OMEGA_DOT_REF + domega_dot) * SC
        i0dot = _cs(m, 133, 15) * 2 ** -44 * SC
        cis = _cs(m, 148, 16) * 2 ** -30
        cic = _cs(m, 164, 16) * 2 ** -30
        crs = _cs(m, 180, 24) * 2 ** -8
        crc = _cs(m, 204, 24) * 2 ** -8
        cus = _cs(m, 228, 21) * 2 ** -30
        cuc = _cs(m, 249, 21) * 2 ** -30
        out += [
            ("toe", toe, " s"), ("Omega0", omega0, " rad"), ("i0", i0, " rad"),
            ("OmegaDot", omega_dot, " rad/s"), ("i0dot", i0dot, " rad/s"),
            ("Cis", cis, " rad"), ("Cic", cic, " rad"),
            ("Crs", crs, " m"), ("Crc", crc, " m"),
            ("Cus", cus, " rad"), ("Cuc", cuc, " rad"),
        ]
        if reserved:
            out += [("reserved", _hexbits(_cu(m, 270, 7), 7), "")]

    elif mtype == 30:  # Clock, IONO & Group Delay
        out += _cnav_clock_block(m)
        tgd = _cs(m, 128, 13) * 2 ** -35
        isc_l1 = _cs(m, 141, 13) * 2 ** -35
        isc_l2 = _cs(m, 154, 13) * 2 ** -35
        isc_l5i = _cs(m, 167, 13) * 2 ** -35
        isc_l5q = _cs(m, 180, 13) * 2 ** -35
        a0 = _cs(m, 193, 8) * 2 ** -30
        a1 = _cs(m, 201, 8) * 2 ** -27
        a2 = _cs(m, 209, 8) * 2 ** -24
        a3 = _cs(m, 217, 8) * 2 ** -24
        b0 = _cs(m, 225, 8) * 2 ** 11
        b1 = _cs(m, 233, 8) * 2 ** 14
        b2 = _cs(m, 241, 8) * 2 ** 16
        b3 = _cs(m, 249, 8) * 2 ** 16
        wnop = _cu(m, 257, 8)
        out += [
            ("Tgd", tgd, " s"), ("ISC_L1CA", isc_l1, " s"), ("ISC_L2C", isc_l2, " s"),
            ("ISC_L5I5", isc_l5i, " s"), ("ISC_L5Q5", isc_l5q, " s"),
            ("alpha0", a0, " s"), ("alpha1", a1, " s/sc"),
            ("alpha2", a2, " s/sc^2"), ("alpha3", a3, " s/sc^3"),
            ("beta0", b0, " s"), ("beta1", b1, " s/sc"),
            ("beta2", b2, " s/sc^2"), ("beta3", b3, " s/sc^3"),
            ("WNop", wnop, ""),
        ]
        if reserved:
            out += [("reserved", _hexbits(_cu(m, 265, 12), 12), "")]

    elif mtype == 33:  # Clock & UTC
        out += _cnav_clock_block(m)
        a0 = _cs(m, 128, 16) * 2 ** -35
        a1 = _cs(m, 144, 13) * 2 ** -51
        a2 = _cs(m, 157, 7) * 2 ** -68
        dtls = _cs(m, 164, 8)
        tot = _cu(m, 172, 16) * 2 ** 4
        wnot = _cu(m, 188, 13)
        wnlsf = _cu(m, 201, 13)
        dn = _cu(m, 214, 4)
        dtlsf = _cs(m, 218, 8)
        out += [
            ("UTC_A0", a0, " s"), ("UTC_A1", a1, " s/s"), ("UTC_A2", a2, " s/s^2"),
            ("dtLS", dtls, " s"), ("tot", tot, " s"), ("WNot", wnot, ""),
            ("WN_LSF", wnlsf, ""), ("DN", dn, ""), ("dtLSF", dtlsf, " s"),
        ]
        if reserved:
            out += [("reserved", _hexbits(_cu(m, 226, 51), 51), "")]

    elif 31 <= mtype <= 37:  # other clock-family messages: decode shared block
        out += _cnav_clock_block(m)
        out += [("note", "type-specific fields not decoded", "")]

    else:
        out += [("note", "message body not decoded", "")]

    return mtype, tow_s, out
