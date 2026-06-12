"""BeiDou RXM-SFRBX decoders: D1/D2 NAV, B-CNAV1 (B1C), B-CNAV2 (B2a)."""

import math

from .bits import SC, _bds_u, _bs, _bs2, _bu, _bu2, _crc24q_head, _words_int


# ---------------------------------------------------------------------------
# BeiDou B1I / B2I D1 NAV decoder (RXM-SFRBX, gnssId=3), per the BDS-SIS-ICD.
# u-blox delivers ten 30-bit words (parity included); each RXM-SFRBX carries
# one subframe (1-5).  Fields straddle word boundaries because of the BCH
# parity bits, so several are read as two concatenated pieces.  The receiver
# has already validated parity; we confirm the 11-bit preamble (0x712).
# Note: GEO satellites (svId <= 5) use the D2 format, which is not decoded.
# ---------------------------------------------------------------------------

BDS_SUBFRAME = {
    1: "clock/health/iono",
    2: "ephemeris I",
    3: "ephemeris II",
    4: "almanac",
    5: "almanac/time",
}


def decode_bds_d1(words):
    """Decode one BeiDou D1 subframe (10 x 30-bit words).

    Returns (subframe_id, sow_seconds, list_of_("field", value, unit) tuples).
    """
    if len(words) < 10:
        return None, None, [("error", f"need 10 words, got {len(words)}", "")]

    v = 0
    for w in words:
        v = (v << 30) | (w & 0x3FFFFFFF)  # 300 bits
    T = 300
    pre = _bu(v, T, 0, 11)
    fraid = _bu(v, T, 15, 3)
    sow = _bu2(v, T, 18, 8, 30, 12)

    out = [("preamble", f"0x{pre:03X}{'' if pre == 0x712 else ' (BAD)'}", ""),
           ("SOW", sow, " s")]

    if fraid == 1:  # clock, health, iono
        sath1 = _bu(v, T, 42, 1)
        aodc = _bu(v, T, 43, 5)
        urai = _bu(v, T, 48, 4)
        wn = _bu(v, T, 60, 13)
        toc = _bu2(v, T, 73, 9, 90, 8) * 8
        tgd1 = _bs(v, T, 98, 10) * 0.1
        tgd2 = _bs2(v, T, 108, 4, 120, 6) * 0.1
        af2 = _bs(v, T, 214, 11) * 2 ** -66
        af0 = _bs2(v, T, 225, 7, 240, 17) * 2 ** -33
        af1 = _bs2(v, T, 257, 5, 270, 17) * 2 ** -50
        aode = _bu(v, T, 287, 5)
        # Klobuchar ionospheric model (alpha 0-3 in s, beta 0-3 in s)
        a0i = _bs(v, T, 126, 8) * 2 ** -30
        a1i = _bs(v, T, 134, 8) * 2 ** -27
        a2i = _bs(v, T, 150, 8) * 2 ** -24
        a3i = _bs(v, T, 158, 8) * 2 ** -24
        b0i = _bs2(v, T, 166, 6, 180, 2) * 2 ** 11
        b1i = _bs(v, T, 182, 8) * 2 ** 14
        b2i = _bs(v, T, 190, 8) * 2 ** 16
        b3i = _bs2(v, T, 198, 4, 210, 4) * 2 ** 16
        out += [("SatH1", sath1, ""), ("AODC", aodc, ""), ("URAI", urai, ""),
                ("WN", wn, ""), ("toc", toc, " s"), ("TGD1", tgd1, " ns"),
                ("TGD2", tgd2, " ns"), ("af0", af0, " s"), ("af1", af1, " s/s"),
                ("af2", af2, " s/s^2"), ("AODE", aode, ""),
                ("alpha0", a0i, " s"), ("alpha1", a1i, " s/sc"),
                ("alpha2", a2i, " s/sc^2"), ("alpha3", a3i, " s/sc^3"),
                ("beta0", b0i, " s"), ("beta1", b1i, " s/sc"),
                ("beta2", b2i, " s/sc^2"), ("beta3", b3i, " s/sc^3")]

    elif fraid == 2:  # ephemeris I
        deln = _bs2(v, T, 42, 10, 60, 6) * 2 ** -43 * SC
        cuc = _bs2(v, T, 66, 16, 90, 2) * 2 ** -31
        m0 = _bs2(v, T, 92, 20, 120, 12) * 2 ** -31 * SC
        ecc = _bu2(v, T, 132, 10, 150, 22) * 2 ** -33
        cus = _bs(v, T, 180, 18) * 2 ** -31
        crc = _bs2(v, T, 198, 4, 210, 14) * 2 ** -6
        crs = _bs2(v, T, 224, 8, 240, 10) * 2 ** -6
        sqrta = _bu2(v, T, 250, 12, 270, 20) * 2 ** -19
        out += [("deltaN", deln, " rad/s"), ("M0", m0, " rad"), ("e", ecc, ""),
                ("sqrtA", sqrta, " m^0.5"), ("A", sqrta ** 2, " m"),
                ("Cuc", cuc, " rad"), ("Cus", cus, " rad"),
                ("Crc", crc, " m"), ("Crs", crs, " m")]

    elif fraid == 3:  # ephemeris II
        i0 = _bs2(v, T, 65, 17, 90, 15) * 2 ** -31 * SC
        cic = _bs2(v, T, 105, 7, 120, 11) * 2 ** -31
        omgd = _bs2(v, T, 131, 11, 150, 13) * 2 ** -43 * SC
        cis = _bs2(v, T, 163, 9, 180, 9) * 2 ** -31
        idot = _bs2(v, T, 189, 13, 210, 1) * 2 ** -43 * SC
        omg0 = _bs2(v, T, 211, 21, 240, 11) * 2 ** -31 * SC
        omg = _bs2(v, T, 251, 11, 270, 21) * 2 ** -31 * SC
        out += [("i0", i0, " rad"), ("Omega0", omg0, " rad"), ("omega", omg, " rad"),
                ("OmegaDot", omgd, " rad/s"), ("idot", idot, " rad/s"),
                ("Cic", cic, " rad"), ("Cis", cis, " rad")]

    else:  # fraid 4/5: almanac, satellite health, time offsets and UTC
        pnum = _bu(v, T, 43, 7)
        out += [("Pnum", pnum, "")]
        if fraid == 4 or pnum <= 6:
            alm_prn = pnum if fraid == 4 else pnum + 24
            out += _bds_d1_almanac(v, T, alm_prn)
        elif pnum == 7:
            hea = [_bds_u(v, T, s) for s in _BDS_D1_HEA[:19]]
            out += [("AmEpID", "", ""),
                    ("health(C01-C19)", " ".join(str(h) for h in hea), "")]
        elif pnum == 8:
            hea = [_bds_u(v, T, s) for s in _BDS_D1_HEA[:11]]
            wna = _bu(v, T, 189, 8)
            toa2 = _bu2(v, T, 197, 5, 210, 3) * 2 ** 12
            out += [("health(C20-C30)", " ".join(str(h) for h in hea), ""),
                    ("WNa", wna, ""), ("toa", toa2, " s")]
        elif pnum == 9:  # BDT - GNSS time offsets, units of ns
            out += [("A0GPS", _bs(v, T, 96, 14) * 0.1, " ns"),
                    ("A1GPS", _bs2(v, T, 110, 2, 120, 14) * 0.1, " ns/s"),
                    ("A0Gal", _bs2(v, T, 134, 8, 150, 6) * 0.1, " ns"),
                    ("A1Gal", _bs(v, T, 156, 16) * 0.1, " ns/s"),
                    ("A0GLO", _bs(v, T, 180, 14) * 0.1, " ns"),
                    ("A1GLO", _bs2(v, T, 194, 8, 210, 8) * 0.1, " ns/s")]
        elif pnum == 10:  # BDT - UTC parameters
            out += [("dtLS", _bs2(v, T, 50, 2, 60, 6), " s"),
                    ("dtLSF", _bs(v, T, 66, 8), " s"),
                    ("WN_LSF", _bu(v, T, 74, 8), ""),
                    ("A0UTC", _bs2(v, T, 90, 22, 120, 10) * 2 ** -30, " s"),
                    ("A1UTC", _bs2(v, T, 130, 12, 150, 12) * 2 ** -50, " s/s"),
                    ("DN", _bu(v, T, 162, 8), "")]
        else:
            out += [("note", f"subframe {fraid} page {pnum} (not decoded)", "")]

    return fraid, sow, out


# BeiDou D1 satellite-health field positions (SF5 pages 7/8), 0-based MSB.
# Health words straddle word-parity gaps so several use a 4-tuple split spec.
_BDS_D1_HEA = [
    (50, 2, 60, 7), (67, 9), (76, 6, 90, 3), (93, 9), (102, 9),
    (111, 1, 120, 8), (128, 9), (137, 5, 150, 4), (154, 9), (163, 9),
    (180, 9), (189, 9), (198, 4, 210, 5), (215, 9), (224, 8, 240, 1),
    (241, 9), (250, 9), (259, 3, 270, 6), (276, 9),
]


def _bds_d1_almanac(v, T, prn):
    """Decode a BeiDou D1 almanac page (SF4 pages 1-24 / SF5 pages 1-6).

    Reference inclination for MEO/IGSO almanacs is 0.30 semicircles; the
    broadcast delta-i is added to it.  Returns a list of (name, value, unit).
    """
    sqrta = _bu2(v, T, 50, 2, 60, 22) * 2 ** -11
    if sqrta == 0:
        return [("almSV", f"C{prn:02d}", ""), ("note", "empty almanac slot", "")]
    a1 = _bs(v, T, 90, 11) * 2 ** -38
    a0 = _bs(v, T, 101, 11) * 2 ** -20
    omg0 = _bs2(v, T, 120, 22, 150, 2) * SC * 2 ** -23
    ecc = _bu(v, T, 152, 17) * 2 ** -21
    deltai = _bs2(v, T, 169, 3, 180, 13) * SC * 2 ** -19
    inc = 0.3 * SC + deltai
    toa = _bu(v, T, 193, 8) * 2 ** 12
    omgd = _bs2(v, T, 201, 1, 210, 16) * SC * 2 ** -38
    omg = _bs2(v, T, 226, 6, 240, 18) * SC * 2 ** -23
    m0 = _bs2(v, T, 258, 4, 270, 20) * SC * 2 ** -23
    return [("almSV", f"C{prn:02d}", ""), ("toa", toa, " s"),
            ("sqrtA", sqrta, " m^0.5"), ("A", sqrta ** 2, " m"), ("e", ecc, ""),
            ("i0", inc, " rad"), ("Omega0", omg0, " rad"), ("omega", omg, " rad"),
            ("OmegaDot", omgd, " rad/s"), ("M0", m0, " rad"),
            ("af0", a0, " s"), ("af1", a1, " s/s")]


# ---------------------------------------------------------------------------
# BeiDou D2 NAV decoder (RXM-SFRBX, gnssId=3) for GEO satellites (svId <= 5).
# D2 packs the same 300-bit/10-word frame as D1 but at 500 bps, so each frame 1
# is split into ten 30-bit pages (Pnum 1-10).  Each page carries a slice of the
# clock/iono/ephemeris set; pages 1-2 (clock, ionosphere) are self-contained,
# while the ephemeris is spread across pages 3-10 (MSB/LSB halves on different
# pages) and cannot be reassembled from a single RXM-SFRBX.  Field positions
# follow the BDS-SIS-ICD (matching gnss-sdr's Beidou_DNAV tables).  No GEO data
# was present in the sample logs, so the D2 layout is unverified against truth.
# ---------------------------------------------------------------------------

def decode_bds_d2(words):
    """Decode one BeiDou D2 (GEO) page.  Returns (fraid, pnum, sow, fields)."""
    if len(words) < 10:
        return None, None, None, [("error", f"need 10 words, got {len(words)}", "")]
    v = 0
    for w in words:
        v = (v << 30) | (w & 0x3FFFFFFF)
    T = 300
    pre = _bu(v, T, 0, 11)
    fraid = _bu(v, T, 15, 3)
    sow = _bu2(v, T, 18, 8, 30, 12)
    pnum = _bu(v, T, 42, 4)
    out = [("preamble", f"0x{pre:03X}{'' if pre == 0x712 else ' (BAD)'}", ""),
           ("SOW", sow, " s")]
    if fraid == 1:
        out += [("Pnum", pnum, "")]
        if pnum == 1:  # clock / health
            out += [("SatH1", _bu(v, T, 46, 1), ""), ("AODC", _bu(v, T, 47, 5), ""),
                    ("URAI", _bu(v, T, 60, 4), ""), ("WN", _bu(v, T, 64, 13), ""),
                    ("toc", _bu2(v, T, 77, 5, 90, 12) * 8, " s"),
                    ("TGD1", _bs(v, T, 102, 10) * 0.1, " ns"),
                    ("TGD2", _bs(v, T, 120, 10) * 0.1, " ns")]
        elif pnum == 2:  # ionosphere (Klobuchar)
            out += [("alpha0", _bs2(v, T, 46, 6, 60, 2) * 2 ** -30, " s"),
                    ("alpha1", _bs(v, T, 62, 8) * 2 ** -27, " s/sc"),
                    ("alpha2", _bs(v, T, 70, 8) * 2 ** -24, " s/sc^2"),
                    ("alpha3", _bs2(v, T, 78, 4, 90, 4) * 2 ** -24, " s/sc^3"),
                    ("beta0", _bs(v, T, 94, 8) * 2 ** 11, " s"),
                    ("beta1", _bs(v, T, 102, 8) * 2 ** 14, " s/sc"),
                    ("beta2", _bs2(v, T, 110, 2, 120, 6) * 2 ** 16, " s/sc^2"),
                    ("beta3", _bs(v, T, 126, 8) * 2 ** 16, " s/sc^3")]
        elif 3 <= pnum <= 10:
            out += [("note", f"ephemeris page {pnum}/10 (fragment, "
                             "spans multiple pages)", "")]
        else:
            out += [("note", f"page {pnum} (not decoded)", "")]
    else:
        out += [("note", f"subframe {fraid} (almanac/integrity, not decoded)", "")]
    return fraid, pnum, sow, out


# ---------------------------------------------------------------------------
# BeiDou-3 B-CNAV1 (B1C, sigId 6) and B-CNAV2 (B2a, sigId 8) decoders.  Both
# carry the BDS-3 CNAV parameter set (ICD-B1C / ICD-B2a).  No open-source
# decoder publishes the ephemeris bit layout, so the field offsets below were
# recovered empirically from these logs and cross-checked against the physics:
# MEO sqrt(A) ~= 5282.6 (A_ref 27,906,100 m), IGSO ~= 6493 (A_ref 42,162,200 m),
# i0 ~= 55 deg, OmegaDot ~= -7e-9 rad/s, satellites sharing an orbital plane
# share Omega0, and clock toc == ephemeris toe.  All frames pass CRC-24Q.
#   B1C: SF1 (numWords 3) PRN+SOH; SF2 (19) full ephemeris+clock, CRC over 576b;
#        SF3 (9) per-page system data, CRC over 240b.
#   B2a: every message is 288 bits = PRN(6)+MesType(6)+SOW(18)+data+CRC-24Q.
#        MT10 ephemeris-I, MT11 ephemeris-II, MT30-33 clock, MT34 UTC, MT40 alm.
# ---------------------------------------------------------------------------

BDS3_AREF_MEO = 27906100.0
BDS3_AREF_IGSO = 42162200.0
BDS3_SATTYPE = {0: "reserved", 1: "GEO", 2: "IGSO", 3: "MEO"}

BDS_CNAV1_PAGE = {
    1: "iono (BDGIM), BDT-UTC, ISC",
    2: "reduced almanac",
    3: "EOP",
    4: "midi almanac",
    60: "system info",
}

BDS_CNAV2_MSGTYPE = {
    10: "Ephemeris I",
    11: "Ephemeris II",
    30: "Clock, iono (BDGIM), BGD/ISC",
    31: "Clock & reduced almanac",
    32: "Clock & EOP",
    33: "Clock & midi almanac",
    34: "Clock(SISAI) & BDT-UTC",
    40: "SISAI & midi almanac",
}


def _bds_cnav_eph1(v, T, b):
    """BDS-3 B-CNAV ephemeris part I (orbit). b = bit offset of t_oe."""
    st = _bu(v, T, b + 11, 2)
    aref = BDS3_AREF_IGSO if st in (1, 2) else BDS3_AREF_MEO
    A = aref + _bs(v, T, b + 13, 26) * 2 ** -9
    sqrta = math.sqrt(A) if A > 0 else 0.0
    return [("SatType", BDS3_SATTYPE.get(st, st), ""),
            ("toe", _bu(v, T, b, 11) * 300, " s"),
            ("sqrtA", sqrta, " m^0.5"), ("A", A, " m"),
            ("Adot", _bs(v, T, b + 39, 25) * 2 ** -21, " m/s"),
            ("deltaN0", _bs(v, T, b + 64, 17) * SC * 2 ** -44, " rad/s"),
            ("deltaN0dot", _bs(v, T, b + 81, 23) * SC * 2 ** -57, " rad/s^2"),
            ("M0", _bs(v, T, b + 104, 33) * SC * 2 ** -32, " rad"),
            ("e", _bu(v, T, b + 137, 33) * 2 ** -34, ""),
            ("omega", _bs(v, T, b + 170, 33) * SC * 2 ** -32, " rad")]


def _bds_cnav_eph2(v, T, b):
    """BDS-3 B-CNAV ephemeris part II (orbit). b = bit offset of Omega0."""
    return [("Omega0", _bs(v, T, b, 33) * SC * 2 ** -32, " rad"),
            ("i0", _bs(v, T, b + 33, 33) * SC * 2 ** -32, " rad"),
            ("OmegaDot", _bs(v, T, b + 66, 19) * SC * 2 ** -44, " rad/s"),
            ("idot", _bs(v, T, b + 85, 15) * SC * 2 ** -44, " rad/s"),
            ("Cis", _bs(v, T, b + 100, 16) * 2 ** -30, " rad"),
            ("Cic", _bs(v, T, b + 116, 16) * 2 ** -30, " rad"),
            ("Crs", _bs(v, T, b + 132, 24) * 2 ** -8, " m"),
            ("Crc", _bs(v, T, b + 156, 24) * 2 ** -8, " m"),
            ("Cus", _bs(v, T, b + 180, 21) * 2 ** -30, " rad"),
            ("Cuc", _bs(v, T, b + 201, 21) * 2 ** -30, " rad")]


def _bds_cnav_clock(v, T, b):
    """BDS-3 B-CNAV clock correction block. b = bit offset of t_oc."""
    return [("toc", _bu(v, T, b, 11) * 300, " s"),
            ("af0", _bs(v, T, b + 11, 25) * 2 ** -34, " s"),
            ("af1", _bs(v, T, b + 36, 22) * 2 ** -50, " s/s"),
            ("af2", _bs(v, T, b + 58, 11) * 2 ** -66, " s/s^2")]


def _bds_cnav_bdgim(v, T, b):
    """BDGIM ionosphere model (9 alpha coefficients). b = bit offset of alpha1."""
    out = [("ion_a1", _bs(v, T, b, 10) * 2 ** -3, " TECu")]
    for k in range(8):
        out.append((f"ion_a{k + 2}", _bs(v, T, b + 10 + 8 * k, 8) * 2 ** -3,
                    " TECu"))
    return out


def _bds_cnav_utc(v, T, b):
    """BDT-UTC time-offset parameters. b = bit offset of A0UTC."""
    return [("A0UTC", _bs(v, T, b, 16) * 2 ** -35, " s"),
            ("A1UTC", _bs(v, T, b + 16, 13) * 2 ** -51, " s/s"),
            ("A2UTC", _bs(v, T, b + 29, 7) * 2 ** -68, " s/s^2"),
            ("dt_LS", _bs(v, T, b + 36, 8), " s"),
            ("t_ot", _bu(v, T, b + 44, 16) * 16, " s"),
            ("WN_ot", _bu(v, T, b + 60, 13), ""),
            ("WN_LSF", _bu(v, T, b + 73, 13), ""),
            ("DN", _bu(v, T, b + 86, 3), ""),
            ("dt_LSF", _bs(v, T, b + 89, 8), " s")]


def _bds_cnav_eop(v, T, b):
    """Earth orientation parameters. b = bit offset of t_EOP."""
    return [("t_EOP", _bu(v, T, b, 16) * 16, " s"),
            ("PM_X", _bs(v, T, b + 16, 21) * 2 ** -20, " arcsec"),
            ("PM_Xdot", _bs(v, T, b + 37, 15) * 2 ** -21, " arcsec/d"),
            ("PM_Y", _bs(v, T, b + 52, 21) * 2 ** -20, " arcsec"),
            ("PM_Ydot", _bs(v, T, b + 73, 15) * 2 ** -21, " arcsec/d"),
            ("dUT1", _bs(v, T, b + 88, 31) * 2 ** -24, " s"),
            ("dUT1dot", _bs(v, T, b + 119, 19) * 2 ** -25, " s/d")]


def _bds_cnav_ralm(v, T, b):
    """One reduced-almanac packet (38 bits). b = bit offset of PRNa.

    Returns None when the slot is empty (PRNa == 0).
    """
    pr = _bu(v, T, b, 6)
    if pr == 0:
        return None
    st = _bu(v, T, b + 6, 2)
    aref = BDS3_AREF_IGSO if st in (1, 2) else BDS3_AREF_MEO
    return [("almPRN", f"C{pr:02d}", ""),
            ("SatType", BDS3_SATTYPE.get(st, st), ""),
            ("deltaA", _bs(v, T, b + 8, 8) * 512, " m"),
            ("A", aref + _bs(v, T, b + 8, 8) * 512, " m"),
            ("Omega0", _bs(v, T, b + 16, 7) * SC * 2 ** -6, " rad"),
            ("Phi0", _bs(v, T, b + 23, 7) * SC * 2 ** -6, " rad"),
            ("Health", _bu(v, T, b + 30, 8), "")]


def _bds_cnav_midi(v, T, b):
    """Midi almanac for a single satellite (156 bits). b = bit offset of PRNa.

    Returns None when the slot is empty (PRNa == 0).
    """
    pr = _bu(v, T, b, 6)
    if pr == 0:
        return None
    st = _bu(v, T, b + 6, 2)
    di = _bs(v, T, b + 40, 11) * 2 ** -14
    return [("almPRN", f"C{pr:02d}", ""),
            ("SatType", BDS3_SATTYPE.get(st, st), ""),
            ("WNa", _bu(v, T, b + 8, 13), ""),
            ("toa", _bu(v, T, b + 21, 8) * 4096, " s"),
            ("e", _bu(v, T, b + 29, 11) * 2 ** -16, ""),
            ("i0", (0.30 + di) * SC, " rad"),
            ("sqrtA", _bu(v, T, b + 51, 17) * 2 ** -4, " m^0.5"),
            ("Omega0", _bs(v, T, b + 68, 16) * SC * 2 ** -15, " rad"),
            ("OmegaDot", _bs(v, T, b + 84, 11) * SC * 2 ** -33, " rad/s"),
            ("omega", _bs(v, T, b + 95, 16) * SC * 2 ** -15, " rad"),
            ("M0", _bs(v, T, b + 111, 16) * SC * 2 ** -15, " rad"),
            ("af0", _bs(v, T, b + 127, 11) * 2 ** -20, " s"),
            ("af1", _bs(v, T, b + 138, 10) * 2 ** -37, " s/s"),
            ("Health", _bu(v, T, b + 148, 8), "")]


def _bds_ralm_blocks(v, T, offsets):
    """Decode reduced-almanac packets at the given bit offsets, skipping empty
    slots."""
    out = []
    for b in offsets:
        blk = _bds_cnav_ralm(v, T, b)
        if blk:
            out += blk
    return out


def decode_bds_cnav1(raw):
    """Decode a BeiDou B-CNAV1 (B1C) RXM-SFRBX. Returns (part, crc_ok, fields)."""
    nw = len(raw)
    total = nw * 32
    v = _words_int(raw)
    if nw == 3:                                   # Subframe 1: PRN + SOH
        prn = _bu(v, total, 0, 6)
        soh = _bu(v, total, 6, 8)
        return "SF1", None, [("PRN", f"C{prn:02d}", ""), ("SOH", soh * 18, " s")]
    if nw == 19:                                  # Subframe 2: ephemeris + clock
        m = v >> (total - 600)
        crc_ok = _crc24q_head(m, 600, 576) == _bu(m, 600, 576, 24)
        fields = [("WN", _bu(m, 600, 0, 13), "")]
        fields += _bds_cnav_eph1(m, 600, 39)
        fields += _bds_cnav_eph2(m, 600, 242)
        fields += _bds_cnav_clock(m, 600, 464)
        return "SF2", crc_ok, fields
    # Subframe 3 (numWords 9): per-page system data
    m = v >> (total - 264)
    crc_ok = _crc24q_head(m, 264, 240) == _bu(m, 264, 240, 24)
    pageid = _bu(m, 264, 0, 6)
    pname = BDS_CNAV1_PAGE.get(pageid, "system data")
    fields = [("PageID", pageid, ""), ("page", pname, ""),
              ("HS", _bu(m, 264, 6, 2), "")]
    if pageid == 1:                               # iono (BDGIM) + BDT-UTC
        fields += _bds_cnav_bdgim(m, 264, 42)
        fields += _bds_cnav_utc(m, 264, 116)
        fields.append(("note", "+ ISC/Tgd (not decoded)", ""))
    elif pageid == 2:                             # reduced almanac (x4)
        fields += _bds_ralm_blocks(m, 264, (58, 96, 134, 172))
    elif pageid == 3:                             # EOP (+ SISAI)
        fields += _bds_cnav_eop(m, 264, 20)
    elif pageid == 4:                             # midi almanac
        blk = _bds_cnav_midi(m, 264, 37)
        if blk:
            fields += blk
    return "SF3", crc_ok, fields


def decode_bds_cnav2(raw):
    """Decode a BeiDou B-CNAV2 (B2a) RXM-SFRBX.

    Returns (msgType, crc_ok, sow, prn, fields).
    """
    total = len(raw) * 32
    v = _words_int(raw) >> (total - 288)
    T = 288
    prn = _bu(v, T, 0, 6)
    mt = _bu(v, T, 6, 6)
    sow = _bu(v, T, 12, 18)
    crc_ok = _crc24q_head(v, T, 264) == _bu(v, T, 264, 24)
    if mt == 10:
        fields = _bds_cnav_eph1(v, T, 61)
    elif mt == 11:
        fields = _bds_cnav_eph2(v, T, 42)
    elif mt in (30, 31, 32, 33, 34):
        fields = _bds_cnav_clock(v, T, 42)
        if mt == 30:
            fields += _bds_cnav_bdgim(v, T, 145)
            fields.append(("note", "+ ISC/Tgd, SISAI (not decoded)", ""))
        elif mt == 31:
            fields += _bds_ralm_blocks(v, T, (142, 180, 218))
        elif mt == 32:
            fields += _bds_cnav_eop(v, T, 121)
        elif mt == 33:
            fields += _bds_ralm_blocks(v, T, (141, 179, 217))
        else:                                      # mt == 34
            fields += _bds_cnav_utc(v, T, 143)
    elif mt == 40:
        blk = _bds_cnav_midi(v, T, 69)
        fields = blk if blk else []
        fields.append(("note", "+ SISAI (not decoded)", ""))
    else:
        fields = [("note", BDS_CNAV2_MSGTYPE.get(mt, f"type {mt}") +
                   " (not decoded)", "")]
    return mt, crc_ok, sow, prn, fields
