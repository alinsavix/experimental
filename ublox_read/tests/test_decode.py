"""Regression tests for the RXM-SFRBX decoders in ``ubx_dump``.

These run against a small, committed multi-constellation fixture
(``tests/fixtures/sfrbx_sample.ubx``) built by ``tests/build_fixtures.py``.
They turn the empirical cross-checks used while developing the decoders into
permanent guards:

* ``test_golden_output_matches`` pins the *exact* decoded text, so behaviour-
  preserving refactors (CRC consolidation, dispatch registry, ...) can't drift.
* ``test_no_crc_failures`` asserts every decoder we claim to support passes its
  CRC/parity on real data, including the Galileo E5a (F/NAV) pages now routed to
  the dedicated F/NAV decoder.
* ``test_qzss_cnav_semi_major_axis`` is a direct regression for the QZSS CNAV
  ``A_REF`` fix (QZSS is geosynchronous, ~4.2e7 m; GPS is ~2.66e7 m).
* ``test_all_formats_present`` checks the fixture still exercises every decoder.
"""
import io
import pathlib
from contextlib import redirect_stdout

from pyubx2 import ERR_LOG, UBXReader

import ubxread as ud

HERE = pathlib.Path(__file__).parent
FIXTURE = HERE / "fixtures" / "sfrbx_sample.ubx"
EXPECTED = HERE / "fixtures" / "sfrbx_sample.expected.txt"


def _decode_to_str(**kwargs):
    """Run ``decode_sfrbx`` on the fixture and capture its stdout."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        ud.decode_sfrbx(str(FIXTURE), **kwargs)
    return buf.getvalue()


def _iter_sfrbx():
    """Yield each parsed RXM-SFRBX message in the fixture."""
    with open(FIXTURE, "rb") as stream:
        for _raw, parsed in UBXReader(stream, protfilter=2, quitonerror=ERR_LOG):
            if getattr(parsed, "identity", None) == "RXM-SFRBX":
                yield parsed


def _frame_blocks(output):
    """Split decoder output into (header_line, [body_lines]) per frame."""
    blocks = []
    for line in output.splitlines():
        if line.startswith("#"):
            blocks.append((line, []))
        elif blocks:
            blocks[-1][1].append(line)
    return blocks


def test_golden_output_matches():
    assert _decode_to_str() == EXPECTED.read_text()


def test_no_crc_failures():
    blocks = _frame_blocks(_decode_to_str())
    assert blocks, "fixture decoded to no frames"

    failing = [
        header
        for header, body in blocks
        if any(("FAIL" in ln) or ("(BAD)" in ln) or ("CRC=BAD" in ln) for ln in body)
    ]
    # Every signal we claim to decode (now including Galileo E5a F/NAV) must
    # pass its CRC / parity check on real data.
    assert not failing, f"unexpected CRC failures: {failing}"


def test_qzss_cnav_semi_major_axis():
    found_qzss = found_gps = False
    for parsed in _iter_sfrbx():
        gid = ud._g(parsed, "gnssId")
        sid = ud._g(parsed, "sigId")
        raw = ud.sfrbx_raw_words(parsed)
        if len(raw) != 10:
            continue

        if gid == 5 and sid == 8:  # QZSS L5 CNAV
            mtype, _tow, fields = ud.decode_gps_cnav(raw, a_ref=ud.QZSS_CNAV_A_REF)
            if mtype == 10:
                vals = {n: v for n, v, _u in fields}
                assert 4.20e7 < vals["A"] < 4.23e7, vals["A"]  # geosynchronous
                assert 0.05 < vals["e"] < 0.10, vals["e"]      # QZSS eccentricity
                found_qzss = True
        elif gid == 0 and sid in (4, 6):  # GPS L2C/L5 CNAV
            mtype, _tow, fields = ud.decode_gps_cnav(raw)
            if mtype == 10:
                vals = {n: v for n, v, _u in fields}
                assert 2.6e7 < vals["A"] < 2.7e7, vals["A"]    # MEO
                found_gps = True

    assert found_qzss, "no QZSS CNAV MT10 in fixture"
    assert found_gps, "no GPS CNAV MT10 in fixture"


def test_all_formats_present():
    out = _decode_to_str()
    for constellation in ("GPS", "QZSS", "GAL", "BDS", "GLO", "SBAS"):
        assert constellation in out, f"missing constellation {constellation}"
    for fmt in ("LNAV", "CNAV", "I/NAV", "F/NAV", "D1NAV",
                "B-CNAV1", "B-CNAV2", "String", "MT"):
        assert fmt in out, f"missing message format {fmt}"


# --- Generic (non-SFRBX) message decoders ---------------------------------

MSG_FIXTURE = HERE / "fixtures" / "msg_sample.ubx"


def _msg_msgs():
    """Yield each parsed message in the generic-message fixture."""
    with open(MSG_FIXTURE, "rb") as stream:
        for _raw, parsed in UBXReader(stream, protfilter=2, quitonerror=ERR_LOG):
            if parsed is not None:
                yield parsed


def test_registry_dispatch():
    # Every MON/NAV decoder must be wired into the generic registry, and each
    # registry decoder must run on its real message without raising.
    for ident in ("MON-COMMS", "MON-HW", "MON-HW2", "MON-RF", "MON-SYS", "MON-TXBUF", "MON-HW3",
                  "NAV-CLOCK", "NAV-DOP", "NAV-EOE", "NAV-ORB", "NAV-POSECEF",
                  "NAV-POSLLH", "NAV-HPPOSECEF", "NAV-HPPOSLLH",
                  "NAV-PVT", "NAV-SAT", "NAV-SBAS", "NAV-SIG",
                  "NAV-STATUS",
                  "NAV-TIMEBDS", "NAV-TIMEGAL", "NAV-TIMEGLO", "NAV-TIMEGPS",
                  "NAV-TIMELS", "NAV-TIMENAVIC", "NAV-TIMEUTC",
                  "NAV-VELECEF", "NAV-VELNED",
                  "RXM-MEASX", "RXM-RAWX",
                  "SEC-SIG", "TIM-SVIN", "TIM-TP"):
        assert ident in ud.MSG_DECODERS, f"{ident} not registered"
    for msg in _msg_msgs():
        ident = msg.identity
        if ident in ud.MSG_DECODERS:
            dec = ud.MSG_DECODERS[ident](msg)
            assert dec.header and dec.fields


def test_mon_hw_decode_fields():
    hw = next(m for m in _msg_msgs() if m.identity == "MON-HW")
    dec = ud.decode_mon_hw(hw)
    vals = {label: val for label, val, _u in dec.fields}
    # antenna/jamming enums render as "<n> (<name>)"
    assert "(" in str(vals["aStatus"]) and "(" in str(vals["jammingState"])
    # AGC percentage is derived from agcCnt / 8191
    agc = ud._g(hw, "agcCnt")
    assert abs(vals["agc"] - agc / 8191 * 100) < 1e-6
    # bitfields are rendered as MSB-first hex
    assert str(vals["usedMask"]).startswith("0x")
    # reserved fields appear only when requested
    assert "reserved0" not in vals
    assert "reserved0" in {label for label, _v, _u in ud.decode_mon_hw(hw, reserved=True).fields}


def test_mon_hw2_decode_fields():
    hw2 = next(m for m in _msg_msgs() if m.identity == "MON-HW2")
    dec = ud.decode_mon_hw2(hw2)
    vals = {label: val for label, val, _u in dec.fields}
    assert vals["magI"] == ud._g(hw2, "magI")
    assert vals["ofsQ"] == ud._g(hw2, "ofsQ")
    assert "(" in str(vals["cfgSource"])  # enum-rendered


def test_decode_messages_smoke():
    # The generic driver decodes registered types and never raises on a mix.
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = ud.decode_messages(str(MSG_FIXTURE))
    out = buf.getvalue()
    assert rc == 0
    assert "MON-HW" in out and "MON-HW2" in out
    assert "aStatus" in out and "cfgSource" in out


def test_mon_rf_grouped_decode():
    rf = next(m for m in _msg_msgs() if m.identity == "MON-RF")
    dec = ud.decode_mon_rf(rf)
    # One Section per RF block, each followed by its field cells.
    sections = [it for it in dec.fields if isinstance(it, ud.Section)]
    assert len(sections) == ud._g(rf, "nBlocks")
    assert all("block" in s.title for s in sections)
    labels = {it[0] for it in dec.fields
              if not isinstance(it, (ud.Section, ud.Line))}
    assert {"agcCnt", "jammingState", "ofsI", "magQ"} <= labels


def test_mon_sys_decode_fields():
    sys = next(m for m in _msg_msgs() if m.identity == "MON-SYS")
    dec = ud.decode_mon_sys(sys)
    vals = {label: val for label, val, _u in dec.fields}
    assert "(" in str(vals["bootType"])              # enum-rendered
    assert vals["runTime"] == ud._g(sys, "runTime")
    assert vals["tempValue"] == ud._g(sys, "tempValue")


def test_mon_txbuf_ports():
    tx = next(m for m in _msg_msgs() if m.identity == "MON-TXBUF")
    dec = ud.decode_mon_txbuf(tx)
    lines = [it for it in dec.fields if isinstance(it, ud.Line)]
    assert len(lines) == 6                            # six tx ports
    assert all("port" in ln.text for ln in lines)


def test_mon_hw3_pins():
    hw3 = next(m for m in _msg_msgs() if m.identity == "MON-HW3")
    dec = ud.decode_mon_hw3(hw3)
    pins = [it for it in dec.fields if isinstance(it, ud.Line)]
    assert len(pins) == ud._g(hw3, "nPins")          # one row per GPIO pin
    cells = {it[0]: it[1] for it in dec.fields
             if not isinstance(it, (ud.Section, ud.Line))}
    assert isinstance(cells["hwVersion"], str)        # decoded from bytes to ASCII
    assert "pin" in pins[0].text and "VP=" in pins[0].text


def _cells(dec):
    """Plain (label -> value) cells of a Decoded, ignoring Section/Line markers."""
    return {it[0]: it[1] for it in dec.fields
            if not isinstance(it, (ud.Section, ud.Line))}


def test_nav_pvt_decode():
    pvt = next(m for m in _msg_msgs() if m.identity == "NAV-PVT")
    vals = _cells(ud.decode_nav_pvt(pvt))
    assert "(" in str(vals["fixType"])                # enum-rendered
    # lat/lon keep full 7-decimal precision (not %g-truncated)
    assert vals["lat"] == f"{ud._g(pvt, 'lat'):.7f}"
    # height is converted from mm to metres
    assert vals["hMSL"] == f"{ud._g(pvt, 'hMSL') / 1000:.3f}"


def test_nav_posecef_metres():
    pos = next(m for m in _msg_msgs() if m.identity == "NAV-POSECEF")
    vals = _cells(ud.decode_nav_posecef(pos))
    assert vals["ecefX"] == f"{ud._g(pos, 'ecefX') / 100:.2f}"   # cm -> m


def test_nav_posllh_precision():
    pos = next(m for m in _msg_msgs() if m.identity == "NAV-POSLLH")
    vals = _cells(ud.decode_nav_posllh(pos))
    assert vals["lat"] == f"{ud._g(pos, 'lat'):.7f}"
    assert vals["lon"] == f"{ud._g(pos, 'lon'):.7f}"


def test_nav_hpposecef_precision():
    pos = next(m for m in _msg_msgs() if m.identity == "NAV-HPPOSECEF")
    dec = ud.decode_nav_hpposecef(pos)
    vals = _cells(dec)
    # ecefX carries the folded HP component (cm); rendered to metres at 4 dp
    assert vals["ecefX"] == f"{ud._g(pos, 'ecefX') / 100:.4f}"
    assert vals["ecefY"] == f"{ud._g(pos, 'ecefY') / 100:.4f}"
    assert vals["ecefZ"] == f"{ud._g(pos, 'ecefZ') / 100:.4f}"
    # pAcc is in mm -> metres at 4 dp
    assert vals["pAcc"] == f"{ud._g(pos, 'pAcc') / 1000:.4f}"
    assert vals["invalidEcef"] == ud._g(pos, "invalidEcef")
    # reserved0 hidden by default
    assert "reserved0" not in vals
    assert "reserved0" in _cells(ud.decode_nav_hpposecef(pos, reserved=True))


def test_nav_hpposllh_precision():
    pos = next(m for m in _msg_msgs() if m.identity == "NAV-HPPOSLLH")
    dec = ud.decode_nav_hpposllh(pos)
    vals = _cells(dec)
    # lat/lon keep 9-decimal high precision
    assert vals["lat"] == f"{ud._g(pos, 'lat'):.9f}"
    assert vals["lon"] == f"{ud._g(pos, 'lon'):.9f}"
    # height/hMSL are in mm -> metres at 4 dp
    assert vals["height"] == f"{ud._g(pos, 'height') / 1000:.4f}"
    assert vals["hMSL"] == f"{ud._g(pos, 'hMSL') / 1000:.4f}"
    assert vals["hAcc"] == f"{ud._g(pos, 'hAcc') / 1000:.4f}"
    assert vals["invalidLlh"] == ud._g(pos, "invalidLlh")
    assert "lat=" in dec.header


def test_nav_clock_and_dop():
    clk = next(m for m in _msg_msgs() if m.identity == "NAV-CLOCK")
    cvals = _cells(ud.decode_nav_clock(clk))
    assert cvals["clkB"] == ud._g(clk, "clkB")        # ns, passed through
    dop = next(m for m in _msg_msgs() if m.identity == "NAV-DOP")
    dvals = _cells(ud.decode_nav_dop(dop))
    assert dvals["pDOP"] == ud._g(dop, "pDOP")        # already scaled by pyubx2


def test_nav_orb_satellites():
    orb = next(m for m in _msg_msgs() if m.identity == "NAV-ORB")
    dec = ud.decode_nav_orb(orb)
    sats = [it for it in dec.fields if isinstance(it, ud.Line)]
    assert len(sats) == ud._g(orb, "numSv")           # one row per satellite
    assert "health=" in sats[0].text and "vis=" in sats[0].text


def test_nav_eoe():
    eoe = next(m for m in _msg_msgs() if m.identity == "NAV-EOE")
    dec = ud.decode_nav_eoe(eoe)
    assert "end of epoch" in dec.header


# --- NAV-TIME* decoder tests -----------------------------------------------

def test_nav_timegps_fields():
    msg = next(m for m in _msg_msgs() if m.identity == "NAV-TIMEGPS")
    vals = _cells(ud.decode_nav_timegps(msg))
    assert vals["fTOW"] == ud._g(msg, "fTOW")
    assert vals["week"] == ud._g(msg, "week")
    assert vals["leapS"] == ud._g(msg, "leapS")
    assert vals["towValid"] == ud._g(msg, "towValid")
    assert "week=" in ud.decode_nav_timegps(msg).header


def test_nav_timeutc_fields():
    msg = next(m for m in _msg_msgs() if m.identity == "NAV-TIMEUTC")
    dec = ud.decode_nav_timeutc(msg)
    vals = _cells(dec)
    assert "(" in str(vals["utcStandard"])   # enum-rendered
    assert vals["nano"] == ud._g(msg, "nano")
    assert vals["validTOW"] == ud._g(msg, "validTOW")
    assert str(msg.year) in dec.header


def test_nav_timebds_fields():
    msg = next(m for m in _msg_msgs() if m.identity == "NAV-TIMEBDS")
    vals = _cells(ud.decode_nav_timebds(msg))
    assert vals["SOW"] == ud._g(msg, "SOW")
    assert vals["leapS"] == ud._g(msg, "leapS")
    assert "BDS" in ud.decode_nav_timebds(msg).header


def test_nav_timegal_fields():
    msg = next(m for m in _msg_msgs() if m.identity == "NAV-TIMEGAL")
    vals = _cells(ud.decode_nav_timegal(msg))
    assert vals["galTow"] == ud._g(msg, "galTow")
    assert vals["galWno"] == ud._g(msg, "galWno")
    assert "GAL" in ud.decode_nav_timegal(msg).header


def test_nav_timeglo_fields():
    msg = next(m for m in _msg_msgs() if m.identity == "NAV-TIMEGLO")
    vals = _cells(ud.decode_nav_timeglo(msg))
    assert vals["TOD"] == ud._g(msg, "TOD")
    assert vals["Nt"] == ud._g(msg, "Nt")
    assert "GLO" in ud.decode_nav_timeglo(msg).header


def test_nav_timels_fields():
    msg = next(m for m in _msg_msgs() if m.identity == "NAV-TIMELS")
    dec = ud.decode_nav_timels(msg)
    vals = _cells(dec)
    assert "(" in str(vals["srcOfCurrLs"])   # enum-rendered
    assert vals["currLs"] == ud._g(msg, "currLs")
    assert vals["validCurrLs"] == ud._g(msg, "validCurrLs")
    # reserved fields hidden by default
    assert "reserved0" not in vals
    assert "reserved0" in _cells(ud.decode_nav_timels(msg, reserved=True))


def test_nav_timenavic_fields():
    msg = next(m for m in _msg_msgs() if m.identity == "NAV-TIMENAVIC")
    vals = _cells(ud.decode_nav_timenavic(msg))
    assert vals["NavICTow"] == ud._g(msg, "NavICTow")
    assert vals["NavICWno"] == ud._g(msg, "NavICWno")
    assert "NavIC" in ud.decode_nav_timenavic(msg).header


def test_nav_velecef_fields():
    msg = next(m for m in _msg_msgs() if m.identity == "NAV-VELECEF")
    vals = _cells(ud.decode_nav_velecef(msg))
    # ecefVX is raw cm/s; decoder converts to m/s
    assert vals["ecefVX"] == f"{ud._g(msg, 'ecefVX') / 100:.3f}"
    assert vals["ecefVY"] == f"{ud._g(msg, 'ecefVY') / 100:.3f}"
    assert vals["ecefVZ"] == f"{ud._g(msg, 'ecefVZ') / 100:.3f}"
    assert "m/s" in ud.decode_nav_velecef(msg).header


def test_nav_velned_fields():
    msg = next(m for m in _msg_msgs() if m.identity == "NAV-VELNED")
    vals = _cells(ud.decode_nav_velned(msg))
    # velN/E/D raw in mm/s; decoder converts to m/s
    assert vals["velN"] == f"{ud._g(msg, 'velN') / 1000:.3f}"
    assert vals["gSpeed"] == f"{ud._g(msg, 'gSpeed') / 1000:.3f}"
    # heading already in degrees from pyubx2
    assert vals["heading"] == f"{ud._g(msg, 'heading'):.5f}"
    assert "hdg=" in ud.decode_nav_velned(msg).header


def test_sec_sig_fields():
    msg = next(m for m in _msg_msgs() if m.identity == "SEC-SIG")
    dec = ud.decode_sec_sig(msg)
    vals = _cells(dec)
    assert "(" in str(vals["jammingState"])   # enum-rendered
    assert "(" in str(vals["spoofingState"])  # enum-rendered
    assert vals["jamDetEnabled"] == ud._g(msg, "jamDetEnabled")
    # reserved0 hidden by default
    assert "reserved0" not in vals
    assert "reserved0" in _cells(ud.decode_sec_sig(msg, reserved=True))


def test_tim_svin_fields():
    msg = next(m for m in _msg_msgs() if m.identity == "TIM-SVIN")
    vals = _cells(ud.decode_tim_svin(msg))
    # meanX raw cm; decoder converts to m
    assert vals["meanX"] == f"{ud._g(msg, 'meanX') / 100:.2f}"
    assert vals["obs"] == ud._g(msg, "obs")
    assert vals["valid"] == ud._g(msg, "valid")
    assert "active=" in ud.decode_tim_svin(msg).header


def test_rxm_rawx_measurements():
    msg = next(m for m in _msg_msgs() if m.identity == "RXM-RAWX")
    dec = ud.decode_rxm_rawx(msg)
    meas_lines = [it for it in dec.fields if isinstance(it, ud.Line)]
    n = ud._g(msg, "numMeas")
    assert len(meas_lines) == n
    vals = _cells(dec)
    assert vals["numMeas"] == n
    assert vals["leapS"] == ud._g(msg, "leapS")
    assert vals["leapSec"] == ud._g(msg, "leapSec")
    assert vals["clkReset"] == ud._g(msg, "clkReset")
    # rcvTow passed through (formatted)
    assert vals["rcvTow"] == f"{ud._g(msg, 'rcvTow'):.6f}"
    # each row mentions cno= and pr=
    assert all("cno=" in ln.text and "pr=" in ln.text for ln in meas_lines)
    # header mentions week
    assert f"week={ud._g(msg, 'week')}" in dec.header


def test_rxm_measx_satellites():
    msg = next(m for m in _msg_msgs() if m.identity == "RXM-MEASX")
    dec = ud.decode_rxm_measx(msg)
    sv_lines = [it for it in dec.fields if isinstance(it, ud.Line)]
    n = ud._g(msg, "numSv")
    assert len(sv_lines) == n
    vals = _cells(dec)
    assert vals["numSv"] == n
    assert vals["gpsTOW"] == ud._g(msg, "gpsTOW")
    assert vals["gpsTOWacc"] == ud._g(msg, "gpsTOWacc")
    # each row mentions dopplerMS= and cNo=
    assert all("cNo=" in ln.text and "dopplerMS=" in ln.text for ln in sv_lines)
    assert "gpsTOW=" in dec.header


def test_mon_comms_ports():
    msg = next(m for m in _msg_msgs() if m.identity == "MON-COMMS")
    dec = ud.decode_mon_comms(msg)
    n = ud._g(msg, "nPorts")
    # One Section per port (plus the outer "ports (n)" section)
    sections = [it for it in dec.fields if isinstance(it, ud.Section)]
    assert len(sections) == n + 1
    vals = _cells(dec)
    assert vals["nPorts"] == n
    # protocols field rendered as named protocols (not "?")
    assert isinstance(vals["protocols"], str) and "?" not in vals["protocols"]
    assert "UBX" in vals["protocols"] and "RTCM3" in vals["protocols"]
    # composite portId (0x000/0x100/...) resolves to a named port enum
    assert "(" in str(vals["portId"])
    # txBytes and msgs appear in the per-port cells
    labels = {it[0] for it in dec.fields if not isinstance(it, (ud.Section, ud.Line))}
    assert {"txBytes", "rxBytes", "overrunErrs", "msgs"} <= labels
    # reserved1 hidden by default, present when requested
    assert "reserved1" not in labels
    assert "reserved1" in {it[0] for it in ud.decode_mon_comms(msg, reserved=True).fields
                           if not isinstance(it, (ud.Section, ud.Line))}


def test_tim_tp_fields():
    msg = next(m for m in _msg_msgs() if m.identity == "TIM-TP")
    dec = ud.decode_tim_tp(msg)
    vals = _cells(dec)
    assert vals["towMS"] == ud._g(msg, "towMS")
    assert vals["week"] == ud._g(msg, "week")
    assert vals["qErr"] == ud._g(msg, "qErr")
    assert "(" in str(vals["timeBase"])    # enum-rendered
    assert "(" in str(vals["raim"])        # enum-rendered
    assert "(" in str(vals["timeRefGnss"]) # enum-rendered
    assert "(" in str(vals["utcStandard"]) # enum-rendered
    # full TOW combines towMS and sub-ms towSubMS: (towMS + towSubMS)/1000
    expected_tow = (ud._g(msg, "towMS") + ud._g(msg, "towSubMS")) / 1000.0
    assert f"TOW={expected_tow:.9f}s" in dec.header
    assert f"week={ud._g(msg, 'week')}" in dec.header
    assert "locked=" in dec.header

