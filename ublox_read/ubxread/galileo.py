"""Galileo RXM-SFRBX decoders: E1-B/E5b-I I/NAV and E5a-I F/NAV."""

from .bits import SC, _bs, _bu, _crc24q, _crc24q_head, _words_int


# ---------------------------------------------------------------------------
# Galileo E1-B / E5b-I I/NAV decoder (RXM-SFRBX, gnssId=2), per the Galileo OS
# SIS ICD.  u-blox delivers one nominal I/NAV page as eight 32-bit words: the
# first four are the even page-part, the next four the odd part (256 bits).
# Each page carries one "word type"; its 128-bit payload is the even page bits
# [2:114] followed by odd page bits [2:18].  A CRC-24Q over even[0:114] +
# odd[0:82] (with 4 pad bits) protects the page.
# ---------------------------------------------------------------------------

GAL_WORDTYPE = {
    0: "Spare / GST time",
    1: "Ephemeris (1/4)",
    2: "Ephemeris (2/4)",
    3: "Ephemeris (3/4)",
    4: "Ephemeris (4/4) & clock",
    5: "Iono, BGD, health & GST",
    6: "GST-UTC conversion",
    7: "Almanac (SV1)",
    8: "Almanac (SV1 clk, SV2)",
    9: "Almanac (SV2, SV3)",
    10: "Almanac (SV3, GST-GPS)",
    16: "Reduced CED",
    17: "FEC2 outer code (CED)",
    18: "FEC2 outer code (CED)",
    19: "FEC2 outer code (CED)",
    20: "FEC2 outer code (CED)",
    63: "Dummy page",
}

# Galileo almanac references (Galileo OS SIS ICD): nominal sqrt(A) and the
# reference inclination of 56 deg, to which the broadcast deltas are applied.
GAL_SQRTA_REF = 5440.588203494
GAL_INC_REF = 56.0 / 180.0  # semicircles


def decode_gal_inav(words):
    """Decode one Galileo I/NAV page (>= 8 x 32-bit words).

    Returns (word_type, tow_or_None, list_of_("field", value, unit) tuples).
    """
    if len(words) < 8:
        return None, None, [("error", f"need 8 words, got {len(words)}", "")]

    val = _words_int(words[:8])  # 256 bits
    part1 = _bu(val, 256, 0, 1)
    page1 = _bu(val, 256, 1, 1)
    part2 = _bu(val, 256, 128, 1)
    page2 = _bu(val, 256, 129, 1)

    even = _bu(val, 256, 0, 128)
    odd = _bu(val, 256, 128, 128)
    payload = (_bu(val, 256, 2, 112) << 16) | _bu(val, 256, 130, 16)  # 128 bits
    wt = _bu(payload, 128, 0, 6)

    if page1 == 1 or page2 == 1:
        page_status = "alert page"
        crc_ok = None
    elif part1 != 0 or part2 != 1:
        page_status = "even/odd error"
        crc_ok = None
    else:
        crc_bits = [0, 0, 0, 0]
        crc_bits += [(even >> (127 - b)) & 1 for b in range(114)]
        crc_bits += [(odd >> (127 - b)) & 1 for b in range(82)]
        stored = _bu(odd, 128, 82, 24)
        crc_ok = _crc24q(crc_bits) == stored
        page_status = "OK" if crc_ok else "FAIL"

    out = [("page", page_status, ""), ("CRC", "OK" if crc_ok else
            ("-" if crc_ok is None else "FAIL"), "")]
    tow = None

    if crc_ok is False or crc_ok is None:
        return wt, tow, out

    if wt == 0:  # spare + GST time
        time_f = _bu(payload, 128, 6, 2)
        wn = _bu(payload, 128, 96, 12)
        tow = _bu(payload, 128, 108, 20)
        out += [("timeFlag", time_f, ""), ("WN", wn, ""), ("TOW", tow, " s")]

    elif wt == 1:  # ephemeris 1
        iodnav = _bu(payload, 128, 6, 10)
        toe = _bu(payload, 128, 16, 14) * 60
        m0 = _bs(payload, 128, 30, 32) * 2 ** -31 * SC
        ecc = _bu(payload, 128, 62, 32) * 2 ** -33
        sqrta = _bu(payload, 128, 94, 32) * 2 ** -19
        out += [("IODnav", iodnav, ""), ("toe", toe, " s"), ("M0", m0, " rad"),
                ("e", ecc, ""), ("sqrtA", sqrta, " m^0.5"), ("A", sqrta ** 2, " m")]

    elif wt == 2:  # ephemeris 2
        iodnav = _bu(payload, 128, 6, 10)
        omg0 = _bs(payload, 128, 16, 32) * 2 ** -31 * SC
        i0 = _bs(payload, 128, 48, 32) * 2 ** -31 * SC
        omg = _bs(payload, 128, 80, 32) * 2 ** -31 * SC
        idot = _bs(payload, 128, 112, 14) * 2 ** -43 * SC
        out += [("IODnav", iodnav, ""), ("Omega0", omg0, " rad"), ("i0", i0, " rad"),
                ("omega", omg, " rad"), ("idot", idot, " rad/s")]

    elif wt == 3:  # ephemeris 3
        iodnav = _bu(payload, 128, 6, 10)
        omgd = _bs(payload, 128, 16, 24) * 2 ** -43 * SC
        deln = _bs(payload, 128, 40, 16) * 2 ** -43 * SC
        cuc = _bs(payload, 128, 56, 16) * 2 ** -29
        cus = _bs(payload, 128, 72, 16) * 2 ** -29
        crc = _bs(payload, 128, 88, 16) * 2 ** -5
        crs = _bs(payload, 128, 104, 16) * 2 ** -5
        sva = _bu(payload, 128, 120, 8)
        out += [("IODnav", iodnav, ""), ("OmegaDot", omgd, " rad/s"),
                ("deltaN", deln, " rad/s"), ("Cuc", cuc, " rad"), ("Cus", cus, " rad"),
                ("Crc", crc, " m"), ("Crs", crs, " m"), ("SISA", sva, "")]

    elif wt == 4:  # ephemeris 4 + clock
        iodnav = _bu(payload, 128, 6, 10)
        svid = _bu(payload, 128, 16, 6)
        cic = _bs(payload, 128, 22, 16) * 2 ** -29
        cis = _bs(payload, 128, 38, 16) * 2 ** -29
        toc = _bu(payload, 128, 54, 14) * 60
        af0 = _bs(payload, 128, 68, 31) * 2 ** -34
        af1 = _bs(payload, 128, 99, 21) * 2 ** -46
        af2 = _bs(payload, 128, 120, 6) * 2 ** -59
        out += [("IODnav", iodnav, ""), ("SVID", svid, ""), ("Cic", cic, " rad"),
                ("Cis", cis, " rad"), ("toc", toc, " s"), ("af0", af0, " s"),
                ("af1", af1, " s/s"), ("af2", af2, " s/s^2")]

    elif wt == 5:  # iono, BGD, health, GST
        ai0 = _bu(payload, 128, 6, 11) * 2 ** -2
        ai1 = _bs(payload, 128, 17, 11) * 2 ** -8
        ai2 = _bs(payload, 128, 28, 14) * 2 ** -15
        bgd_a = _bs(payload, 128, 47, 10) * 2 ** -32
        bgd_b = _bs(payload, 128, 57, 10) * 2 ** -32
        e5b_hs = _bu(payload, 128, 67, 2)
        e1b_hs = _bu(payload, 128, 69, 2)
        e5b_dvs = _bu(payload, 128, 71, 1)
        e1b_dvs = _bu(payload, 128, 72, 1)
        wn = _bu(payload, 128, 73, 12)
        tow = _bu(payload, 128, 85, 20)
        out += [("ai0", ai0, " sfu"), ("ai1", ai1, " sfu/deg"),
                ("ai2", ai2, " sfu/deg^2"), ("BGD_E1E5a", bgd_a, " s"),
                ("BGD_E1E5b", bgd_b, " s"), ("E5b_health", e5b_hs, ""),
                ("E1B_health", e1b_hs, ""), ("E5b_DVS", e5b_dvs, ""),
                ("E1B_DVS", e1b_dvs, ""), ("WN", wn, ""), ("TOW", tow, " s")]

    elif wt == 6:  # GST-UTC conversion
        a0 = _bs(payload, 128, 6, 32) * 2 ** -30
        a1 = _bs(payload, 128, 38, 24) * 2 ** -50
        dtls = _bs(payload, 128, 62, 8)
        tot = _bu(payload, 128, 70, 8) * 3600
        wnot = _bu(payload, 128, 78, 8)
        wnlsf = _bu(payload, 128, 86, 8)
        dn = _bu(payload, 128, 94, 3)
        dtlsf = _bs(payload, 128, 97, 8)
        tow = _bu(payload, 128, 105, 20)
        out += [("UTC_A0", a0, " s"), ("UTC_A1", a1, " s/s"), ("dtLS", dtls, " s"),
                ("tot", tot, " s"), ("WNot", wnot, ""), ("WN_LSF", wnlsf, ""),
                ("DN", dn, ""), ("dtLSF", dtlsf, " s"), ("TOW", tow, " s")]

    elif wt == 7:  # almanac: SV1 orbit + reference time
        ioda = _bu(payload, 128, 6, 4)
        wna = _bu(payload, 128, 10, 2)
        t0a = _bu(payload, 128, 12, 10) * 600
        svid1 = _bu(payload, 128, 22, 6)
        sqrta = GAL_SQRTA_REF + _bs(payload, 128, 28, 13) * 2 ** -9
        ecc = _bu(payload, 128, 41, 11) * 2 ** -16
        omg = _bs(payload, 128, 52, 16) * 2 ** -15 * SC
        i0 = (GAL_INC_REF + _bs(payload, 128, 68, 11) * 2 ** -14) * SC
        omg0 = _bs(payload, 128, 79, 16) * 2 ** -15 * SC
        omgd = _bs(payload, 128, 95, 11) * 2 ** -33 * SC
        m0 = _bs(payload, 128, 106, 16) * 2 ** -15 * SC
        out += [("IODa", ioda, ""), ("WNa", wna, ""), ("t0a", t0a, " s"),
                ("SVID1", svid1, ""), ("sqrtA", sqrta, " m^0.5"), ("e", ecc, ""),
                ("omega", omg, " rad"), ("i0", i0, " rad"), ("Omega0", omg0, " rad"),
                ("OmegaDot", omgd, " rad/s"), ("M0", m0, " rad")]

    elif wt == 8:  # almanac: SV1 clock + SV2 orbit
        ioda = _bu(payload, 128, 6, 4)
        af0 = _bs(payload, 128, 10, 16) * 2 ** -19
        af1 = _bs(payload, 128, 26, 13) * 2 ** -38
        e5b_hs = _bu(payload, 128, 39, 2)
        e1b_hs = _bu(payload, 128, 41, 2)
        svid2 = _bu(payload, 128, 43, 6)
        sqrta = GAL_SQRTA_REF + _bs(payload, 128, 49, 13) * 2 ** -9
        ecc = _bu(payload, 128, 62, 11) * 2 ** -16
        omg = _bs(payload, 128, 73, 16) * 2 ** -15 * SC
        i0 = (GAL_INC_REF + _bs(payload, 128, 89, 11) * 2 ** -14) * SC
        omg0 = _bs(payload, 128, 100, 16) * 2 ** -15 * SC
        omgd = _bs(payload, 128, 116, 11) * 2 ** -33 * SC
        out += [("IODa", ioda, ""), ("SV1_af0", af0, " s"), ("SV1_af1", af1, " s/s"),
                ("SV1_E5bHS", e5b_hs, ""), ("SV1_E1bHS", e1b_hs, ""),
                ("SVID2", svid2, ""), ("sqrtA", sqrta, " m^0.5"), ("e", ecc, ""),
                ("omega", omg, " rad"), ("i0", i0, " rad"),
                ("Omega0", omg0, " rad"), ("OmegaDot", omgd, " rad/s")]

    elif wt == 9:  # almanac: SV2 remainder + SV3 orbit
        ioda = _bu(payload, 128, 6, 4)
        wna = _bu(payload, 128, 10, 2)
        t0a = _bu(payload, 128, 12, 10) * 600
        m0 = _bs(payload, 128, 22, 16) * 2 ** -15 * SC
        af0 = _bs(payload, 128, 38, 16) * 2 ** -19
        af1 = _bs(payload, 128, 54, 13) * 2 ** -38
        e5b_hs = _bu(payload, 128, 67, 2)
        e1b_hs = _bu(payload, 128, 69, 2)
        svid3 = _bu(payload, 128, 71, 6)
        sqrta = GAL_SQRTA_REF + _bs(payload, 128, 77, 13) * 2 ** -9
        ecc = _bu(payload, 128, 90, 11) * 2 ** -16
        omg = _bs(payload, 128, 101, 16) * 2 ** -15 * SC
        i0 = (GAL_INC_REF + _bs(payload, 128, 117, 11) * 2 ** -14) * SC
        out += [("IODa", ioda, ""), ("WNa", wna, ""), ("t0a", t0a, " s"),
                ("SV2_M0", m0, " rad"), ("SV2_af0", af0, " s"), ("SV2_af1", af1, " s/s"),
                ("SV2_E5bHS", e5b_hs, ""), ("SV2_E1bHS", e1b_hs, ""),
                ("SVID3", svid3, ""), ("sqrtA", sqrta, " m^0.5"), ("e", ecc, ""),
                ("omega", omg, " rad"), ("i0", i0, " rad")]

    elif wt == 10:  # almanac: SV3 remainder + GST-GPS conversion
        ioda = _bu(payload, 128, 6, 4)
        omg0 = _bs(payload, 128, 10, 16) * 2 ** -15 * SC
        omgd = _bs(payload, 128, 26, 11) * 2 ** -33 * SC
        m0 = _bs(payload, 128, 37, 16) * 2 ** -15 * SC
        af0 = _bs(payload, 128, 53, 16) * 2 ** -19
        af1 = _bs(payload, 128, 69, 13) * 2 ** -38
        e5b_hs = _bu(payload, 128, 82, 2)
        e1b_hs = _bu(payload, 128, 84, 2)
        a0g = _bs(payload, 128, 86, 16) * 2 ** -35
        a1g = _bs(payload, 128, 102, 12) * 2 ** -51
        t0g = _bu(payload, 128, 114, 8) * 3600
        wn0g = _bu(payload, 128, 122, 6)
        out += [("IODa", ioda, ""), ("SV3_Omega0", omg0, " rad"),
                ("SV3_OmegaDot", omgd, " rad/s"), ("SV3_M0", m0, " rad"),
                ("SV3_af0", af0, " s"), ("SV3_af1", af1, " s/s"),
                ("SV3_E5bHS", e5b_hs, ""), ("SV3_E1bHS", e1b_hs, ""),
                ("GGTO_A0G", a0g, " s"), ("GGTO_A1G", a1g, " s/s"),
                ("GGTO_t0G", t0g, " s"), ("GGTO_WN0G", wn0g, "")]

    elif wt == 16:  # reduced clock & ephemeris data (reduced CED)
        da = _bs(payload, 128, 6, 5) * 2 ** 8
        ex = _bs(payload, 128, 11, 13) * 2 ** -22
        ey = _bs(payload, 128, 24, 13) * 2 ** -22
        di0 = _bs(payload, 128, 37, 17) * 2 ** -22 * SC
        omg0 = _bs(payload, 128, 54, 23) * 2 ** -22 * SC
        lam0 = _bs(payload, 128, 77, 23) * 2 ** -22 * SC
        af0 = _bs(payload, 128, 100, 22) * 2 ** -26
        af1 = _bs(payload, 128, 122, 6) * 2 ** -35
        out += [("dAred", da, " m"), ("ex_red", ex, ""), ("ey_red", ey, ""),
                ("di0red", di0, " rad"), ("Omega0red", omg0, " rad"),
                ("lambda0red", lam0, " rad"), ("af0red", af0, " s"),
                ("af1red", af1, " s/s")]

    elif 17 <= wt <= 20:  # FEC2 outer (Reed-Solomon) code for CED recovery
        out += [("note", "Reed-Solomon FEC2 redundancy (no plain nav data)", "")]

    elif wt == 63:  # dummy page broadcast when no data is scheduled
        out += [("note", "dummy page (no data)", "")]

    else:
        out += [("note", f"word type {wt} (not decoded)", "")]

    return wt, tow, out


GAL_FNAV_PAGETYPE = {
    1: "Clock, iono, BGD, GST & health",
    2: "Ephemeris (1/3)",
    3: "Ephemeris (2/3) & GST",
    4: "Ephemeris (3/3) & GST-UTC",
    5: "Almanac (SV1, SV2)",
    6: "Almanac (SV2, SV3)",
}


def _gal_fnav_alm_sv(page, r, prefix):
    """One Galileo F/NAV almanac satellite block (131 bits) at bit offset r.

    Same field order and scaling as the I/NAV almanac word (the per-SV blocks
    cross-validate bit-for-bit against I/NAV WT7-10).  The health field is the
    single 2-bit E5a health (F/NAV), where I/NAV instead carries E5b + E1B.
    """
    T = 244
    svid = _bu(page, T, r, 6)
    sqrta = GAL_SQRTA_REF + _bs(page, T, r + 6, 13) * 2 ** -9
    ecc = _bu(page, T, r + 19, 11) * 2 ** -16
    omg = _bs(page, T, r + 30, 16) * 2 ** -15 * SC
    di = _bs(page, T, r + 46, 11) * 2 ** -14
    i0 = (GAL_INC_REF + di) * SC
    omg0 = _bs(page, T, r + 57, 16) * 2 ** -15 * SC
    omgd = _bs(page, T, r + 73, 11) * 2 ** -33 * SC
    m0 = _bs(page, T, r + 84, 16) * 2 ** -15 * SC
    af0 = _bs(page, T, r + 100, 16) * 2 ** -19
    af1 = _bs(page, T, r + 116, 13) * 2 ** -38
    e5a_hs = _bu(page, T, r + 129, 2)
    return [(f"{prefix}_SVID", svid, ""), (f"{prefix}_sqrtA", sqrta, " m^0.5"),
            (f"{prefix}_e", ecc, ""), (f"{prefix}_omega", omg, " rad"),
            (f"{prefix}_i0", i0, " rad"), (f"{prefix}_Omega0", omg0, " rad"),
            (f"{prefix}_OmegaDot", omgd, " rad/s"), (f"{prefix}_M0", m0, " rad"),
            (f"{prefix}_af0", af0, " s"), (f"{prefix}_af1", af1, " s/s"),
            (f"{prefix}_E5aHS", e5a_hs, "")]


def decode_gal_fnav(words):
    """Decode one Galileo F/NAV page (E5a-I, sigId 3; 8 x 32-bit words).

    u-blox delivers F/NAV as eight 32-bit words whose top 244 bits hold the
    page: type(6) + data(208) + CRC-24Q(24) + tail(6).  The CRC-24Q is computed
    over the first 214 bits.  Returns (page_type, tow_or_None, fields).

    Page layouts were pinned empirically and cross-validated against the I/NAV
    stream broadcast by the same satellites (ephemeris keyed on IODnav, almanac
    keyed on (SVID, WNa, t0a)); every shared field matches bit-for-bit.
    """
    if len(words) < 8:
        return None, None, [("error", f"need 8 words, got {len(words)}", "")]

    T = 244
    page = _words_int(words[:8]) >> (8 * 32 - T)
    pt = _bu(page, T, 0, 6)
    crc_ok = _crc24q_head(page, T, 214) == _bu(page, T, 214, 24)

    out = [("page", "OK" if crc_ok else "FAIL", ""),
           ("CRC", "OK" if crc_ok else "FAIL", "")]
    tow = None
    if not crc_ok:
        return pt, tow, out

    if pt == 1:  # clock, iono, BGD, GST & health
        svid = _bu(page, T, 6, 6)
        iodnav = _bu(page, T, 12, 10)
        toc = _bu(page, T, 22, 14) * 60
        af0 = _bs(page, T, 36, 31) * 2 ** -34
        af1 = _bs(page, T, 67, 21) * 2 ** -46
        af2 = _bs(page, T, 88, 6) * 2 ** -59
        sisa = _bu(page, T, 94, 8)
        ai0 = _bu(page, T, 102, 11) * 2 ** -2
        ai1 = _bs(page, T, 113, 11) * 2 ** -8
        ai2 = _bs(page, T, 124, 14) * 2 ** -15
        region = _bu(page, T, 138, 5)
        bgd = _bs(page, T, 143, 10) * 2 ** -32
        e5a_hs = _bu(page, T, 153, 2)
        wn = _bu(page, T, 155, 12)
        tow = _bu(page, T, 167, 20)
        e5a_dvs = _bu(page, T, 187, 1)
        out += [("SVID", svid, ""), ("IODnav", iodnav, ""), ("toc", toc, " s"),
                ("af0", af0, " s"), ("af1", af1, " s/s"), ("af2", af2, " s/s^2"),
                ("SISA", sisa, ""), ("ai0", ai0, " sfu"), ("ai1", ai1, " sfu/deg"),
                ("ai2", ai2, " sfu/deg^2"), ("regionFlags", region, ""),
                ("BGD_E1E5a", bgd, " s"), ("E5aHS", e5a_hs, ""), ("WN", wn, ""),
                ("TOW", tow, " s"), ("E5aDVS", e5a_dvs, "")]

    elif pt == 2:  # ephemeris 1/3
        iodnav = _bu(page, T, 6, 10)
        m0 = _bs(page, T, 16, 32) * 2 ** -31 * SC
        omgd = _bs(page, T, 48, 24) * 2 ** -43 * SC
        ecc = _bu(page, T, 72, 32) * 2 ** -33
        sqrta = _bu(page, T, 104, 32) * 2 ** -19
        omg0 = _bs(page, T, 136, 32) * 2 ** -31 * SC
        idot = _bs(page, T, 168, 14) * 2 ** -43 * SC
        out += [("IODnav", iodnav, ""), ("M0", m0, " rad"),
                ("OmegaDot", omgd, " rad/s"), ("e", ecc, ""),
                ("sqrtA", sqrta, " m^0.5"), ("A", sqrta ** 2, " m"),
                ("Omega0", omg0, " rad"), ("idot", idot, " rad/s")]

    elif pt == 3:  # ephemeris 2/3 + GST
        iodnav = _bu(page, T, 6, 10)
        i0 = _bs(page, T, 16, 32) * 2 ** -31 * SC
        omg = _bs(page, T, 48, 32) * 2 ** -31 * SC
        deln = _bs(page, T, 80, 16) * 2 ** -43 * SC
        cuc = _bs(page, T, 96, 16) * 2 ** -29
        cus = _bs(page, T, 112, 16) * 2 ** -29
        crc_c = _bs(page, T, 128, 16) * 2 ** -5
        crs = _bs(page, T, 144, 16) * 2 ** -5
        toe = _bu(page, T, 160, 14) * 60
        wn = _bu(page, T, 174, 12)
        tow = _bu(page, T, 186, 20)
        out += [("IODnav", iodnav, ""), ("i0", i0, " rad"), ("omega", omg, " rad"),
                ("deltaN", deln, " rad/s"), ("Cuc", cuc, " rad"), ("Cus", cus, " rad"),
                ("Crc", crc_c, " m"), ("Crs", crs, " m"), ("toe", toe, " s"),
                ("WN", wn, ""), ("TOW", tow, " s")]

    elif pt == 4:  # ephemeris 3/3 + GST-UTC
        iodnav = _bu(page, T, 6, 10)
        cic = _bs(page, T, 16, 16) * 2 ** -29
        cis = _bs(page, T, 32, 16) * 2 ** -29
        a0 = _bs(page, T, 48, 32) * 2 ** -30
        a1 = _bs(page, T, 80, 24) * 2 ** -50
        dtls = _bs(page, T, 104, 8)
        tot = _bu(page, T, 112, 8) * 3600
        wnot = _bu(page, T, 120, 8)
        wnlsf = _bu(page, T, 128, 8)
        dn = _bu(page, T, 136, 3)
        dtlsf = _bs(page, T, 139, 8)
        tow = _bu(page, T, 147, 20)
        out += [("IODnav", iodnav, ""), ("Cic", cic, " rad"), ("Cis", cis, " rad"),
                ("UTC_A0", a0, " s"), ("UTC_A1", a1, " s/s"), ("dtLS", dtls, " s"),
                ("tot", tot, " s"), ("WNot", wnot, ""), ("WN_LSF", wnlsf, ""),
                ("DN", dn, ""), ("dtLSF", dtlsf, " s"), ("TOW", tow, " s")]

    elif pt == 5:  # almanac: reference time + SV1 (full) + SV2 (orbit start)
        ioda = _bu(page, T, 6, 4)
        wna = _bu(page, T, 10, 2)
        t0a = _bu(page, T, 12, 10) * 600
        out += [("IODa", ioda, ""), ("WNa", wna, ""), ("t0a", t0a, " s")]
        out += _gal_fnav_alm_sv(page, 22, "SV1")
        # SV2's orbit starts here; its clock and Omega0 spill into page 6.
        svid2 = _bu(page, T, 153, 6)
        sqrta2 = GAL_SQRTA_REF + _bs(page, T, 159, 13) * 2 ** -9
        ecc2 = _bu(page, T, 172, 11) * 2 ** -16
        omg2 = _bs(page, T, 183, 16) * 2 ** -15 * SC
        i0_2 = (GAL_INC_REF + _bs(page, T, 199, 11) * 2 ** -14) * SC
        out += [("SV2_SVID", svid2, ""), ("SV2_sqrtA", sqrta2, " m^0.5"),
                ("SV2_e", ecc2, ""), ("SV2_omega", omg2, " rad"),
                ("SV2_i0", i0_2, " rad"),
                ("note", "SV2 Omega0/clock continue in page 6", "")]

    elif pt == 6:  # almanac: SV2 (remainder) + SV3 (full)
        ioda = _bu(page, T, 6, 4)
        out += [("IODa", ioda, ""),
                ("note", "SV2 Omega0/clock continue from page 5", "")]
        out += _gal_fnav_alm_sv(page, 80, "SV3")

    else:
        out += [("note", f"page type {pt} (not decoded)", "")]

    return pt, tow, out
