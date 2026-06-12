"""GLONASS RXM-SFRBX decoder: L1OF/L2OF strings, Hamming-checked."""

from .bits import SC, _bg, _bu, _words_int


# ---------------------------------------------------------------------------
# GLONASS L1OF / L2OF decoder (RXM-SFRBX, gnssId=6), per the GLONASS ICD.
# u-blox delivers one 85-bit string as four 32-bit words (bit 0 = string bit
# 85).  Strings 1-4 carry ephemeris/clock; 5 carries frame time; 6-15 carry
# almanac.  GLONASS uses sign-magnitude integers.  A Hamming check validates
# each string.  Position/velocity/acceleration are scaled to SI (m, m/s, m/s^2).
# ---------------------------------------------------------------------------

GLO_STRING = {
    1: "ephemeris (x, tk)",
    2: "ephemeris (y, tb, health)",
    3: "ephemeris (z, gamma)",
    4: "clock (tau, n, FT, NT)",
    5: "frame time / UTC (tauC, N4, tauGPS)",
    6: "almanac A (slot, lambda, di, e)",
    7: "almanac B (omega, t_lambda, dT)",
    8: "almanac A (slot, lambda, di, e)",
    9: "almanac B (omega, t_lambda, dT)",
    10: "almanac A (slot, lambda, di, e)",
    11: "almanac B (omega, t_lambda, dT)",
    12: "almanac A (slot, lambda, di, e)",
    13: "almanac B (omega, t_lambda, dT)",
    14: "almanac A (slot, lambda, di, e)",
    15: "almanac B (omega, t_lambda, dT)",
}

_GLO_XOR8 = [bin(i).count("1") & 1 for i in range(256)]
_GLO_HAMMING_MASK = (
    (0x55, 0x55, 0x5A, 0xAA, 0xAA, 0xAA, 0xB5, 0x55, 0x6A, 0xD8, 0x08),
    (0x66, 0x66, 0x6C, 0xCC, 0xCC, 0xCC, 0xD9, 0x99, 0xB3, 0x68, 0x10),
    (0x87, 0x87, 0x8F, 0x0F, 0x0F, 0x0F, 0x1E, 0x1E, 0x3C, 0x70, 0x20),
    (0x07, 0xF8, 0x0F, 0xF0, 0x0F, 0xF0, 0x1F, 0xE0, 0x3F, 0x80, 0x40),
    (0xF8, 0x00, 0x0F, 0xFF, 0xF0, 0x00, 0x1F, 0xFF, 0xC0, 0x00, 0x80),
    (0x00, 0x00, 0x0F, 0xFF, 0xFF, 0xFF, 0xE0, 0x00, 0x00, 0x01, 0x00),
    (0xFF, 0xFF, 0xF0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00),
    (0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xF8),
)


def _glo_hamming_ok(buff):
    cs = 0
    n = 0
    for i in range(8):
        cs = 0
        for j in range(11):
            cs ^= _GLO_XOR8[buff[j] & _GLO_HAMMING_MASK[i][j]]
        if cs:
            n += 1
    return n == 0 or (n == 2 and cs)


def decode_glonass(words):
    """Decode one GLONASS nav string (4 x 32-bit words).

    Returns (string_number, None, list_of_("field", value, unit) tuples).
    """
    if len(words) < 4:
        return None, None, [("error", f"need 4 words, got {len(words)}", "")]

    val = _words_int(words[:4])  # 128 bits
    buff = val.to_bytes(16, "big")
    T = 128
    m = _bu(val, T, 1, 4)
    ham = "OK" if _glo_hamming_ok(buff) else "FAIL"
    out = [("Hamming", ham, "")]

    if m == 1:  # x, tk, velocity/accel x
        p1 = _bu(val, T, 7, 2)
        tk_h = _bu(val, T, 9, 5)
        tk_m = _bu(val, T, 14, 6)
        tk_s = _bu(val, T, 20, 1) * 30
        xdot = _bg(val, T, 21, 24) * 2 ** -20 * 1e3
        xddot = _bg(val, T, 45, 5) * 2 ** -30 * 1e3
        x = _bg(val, T, 50, 27) * 2 ** -11 * 1e3
        out += [("P1", p1, ""), ("tk", f"{tk_h:02d}:{tk_m:02d}:{tk_s:02d}", ""),
                ("x", x, " m"), ("xdot", xdot, " m/s"), ("xddot", xddot, " m/s^2")]

    elif m == 2:  # y, tb, health
        bn = _bu(val, T, 5, 3)
        p2 = _bu(val, T, 8, 1)
        tb = _bu(val, T, 9, 7)
        ydot = _bg(val, T, 21, 24) * 2 ** -20 * 1e3
        yddot = _bg(val, T, 45, 5) * 2 ** -30 * 1e3
        y = _bg(val, T, 50, 27) * 2 ** -11 * 1e3
        out += [("Bn(health)", bn, ""), ("P2", p2, ""),
                ("tb", tb, " (x15min)"), ("toe", tb * 900, " s"),
                ("y", y, " m"), ("ydot", ydot, " m/s"), ("yddot", yddot, " m/s^2")]

    elif m == 3:  # z, gamma
        p3 = _bu(val, T, 5, 1)
        gamma = _bg(val, T, 6, 11) * 2 ** -40
        pp = _bu(val, T, 18, 2)
        ln = _bu(val, T, 20, 1)
        zdot = _bg(val, T, 21, 24) * 2 ** -20 * 1e3
        zddot = _bg(val, T, 45, 5) * 2 ** -30 * 1e3
        z = _bg(val, T, 50, 27) * 2 ** -11 * 1e3
        out += [("P3", p3, ""), ("gammaN", gamma, ""), ("P", pp, ""),
                ("ln(health)", ln, ""), ("z", z, " m"),
                ("zdot", zdot, " m/s"), ("zddot", zddot, " m/s^2")]

    elif m == 4:  # clock, slot, age
        taun = _bg(val, T, 5, 22) * 2 ** -30
        dtaun = _bg(val, T, 27, 5) * 2 ** -30
        en = _bu(val, T, 32, 5)
        p4 = _bu(val, T, 51, 1)
        ft = _bu(val, T, 52, 4)
        nt = _bu(val, T, 59, 11)
        slot = _bu(val, T, 70, 5)
        mtype = _bu(val, T, 75, 2)
        out += [("tauN", taun, " s"), ("dTauN", dtaun, " s"), ("En(age)", en, " days"),
                ("P4", p4, ""), ("FT(URA)", ft, ""), ("NT(day)", nt, ""),
                ("slot", slot, ""), ("type", mtype, "")]

    elif m == 5:  # frame time / UTC: NA, tauC, N4, tauGPS
        na = _bu(val, T, 5, 11)
        tau_c = _bg(val, T, 16, 32) * 2 ** -31
        n4 = _bu(val, T, 49, 5)
        tau_gps = _bg(val, T, 54, 22) * 2 ** -30
        ln5 = _bu(val, T, 76, 1)
        out += [("NA(day)", na, ""), ("tauC", tau_c, " s"), ("N4", n4, ""),
                ("tauGPS", tau_gps, " s"), ("ln(health)", ln5, "")]

    elif m in (6, 8, 10, 12, 14):  # almanac part A: orbit geometry of one SV
        cn = _bu(val, T, 5, 1)
        mna = _bu(val, T, 6, 2)
        na = _bu(val, T, 8, 5)
        tau_na = _bg(val, T, 13, 10) * 2 ** -18
        lam_na = _bg(val, T, 23, 21) * 2 ** -20 * SC
        di_na = _bg(val, T, 44, 18) * 2 ** -20 * SC
        eps_na = _bu(val, T, 62, 15) * 2 ** -20
        out += [("Cn(health)", cn, ""), ("MnA(type)", mna, ""), ("nA(slot)", na, ""),
                ("taunA", tau_na, " s"), ("lambdanA", lam_na, " rad"),
                ("dInA", di_na, " rad"), ("eps", eps_na, "")]

    elif m in (7, 9, 11, 13, 15):  # almanac part B: timing of the same SV
        omg_na = _bg(val, T, 5, 16) * 2 ** -15 * SC
        tlam_na = _bu(val, T, 21, 21) * 2 ** -5
        dt_na = _bg(val, T, 42, 22) * 2 ** -9
        dtdot_na = _bg(val, T, 64, 7) * 2 ** -14
        hna = _bu(val, T, 71, 5)
        out += [("omeganA", omg_na, " rad"), ("t_lambdanA", tlam_na, " s"),
                ("dTnA", dt_na, " s"), ("dTdotnA", dtdot_na, " s/period"),
                ("HnA(freq)", hna, "")]

    else:
        out += [("note", f"string {m} (not decoded)", "")]

    return m, None, out
