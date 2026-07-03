#!/usr/bin/env python3
"""
mixerlint.py — Lint checker for Behringer XR18 (and X32-family) mixers.

Reads parameters via OSC and reports things that look wrong, based on rules
defined in mixerlint.yaml. Makes no changes to the mixer.

Usage:
    uv run mixerlint.py [--config mixerlint.yaml] [--host 192.168.0.28]
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "rich>=13.0",
#   "pyyaml>=6.0",
# ]
# ///

import argparse
import math
import socket
import sys
import threading
import time
import yaml
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.table import Table
from rich import print as rprint

console = Console()

# ---------------------------------------------------------------------------
# XR18 OSC constants
# ---------------------------------------------------------------------------

MIXER_PORT = 10024
LOCAL_PORT = 10025
QUERY_TIMEOUT = 0.3   # seconds to wait per OSC response
BATCH_PAUSE  = 0.01   # small sleep between batched sends to avoid flooding

NUM_CHANNELS = 16
NUM_BUSES    = 6
NUM_FX       = 4


# ---------------------------------------------------------------------------
# dB / fader conversion  (X32 / XR18 scale)
# ---------------------------------------------------------------------------

def fader_to_db(val: float) -> float:
    """Convert XR18 fader float (0.0–1.0) to approximate dB."""
    if val <= 0.0:
        return -math.inf
    # Piecewise linear approximation matching X32 curve
    # 0.00 → -inf, 0.25 → -40, 0.50 → -10, 0.75 → 0, 1.00 → +10
    if val < 0.25:
        return -40.0 + (val / 0.25) * 30.0   # -inf..−40 compressed; good enough
    elif val < 0.50:
        return -40.0 + ((val - 0.25) / 0.25) * 30.0
    elif val < 0.75:
        return -10.0 + ((val - 0.50) / 0.25) * 10.0
    else:
        return 0.0   + ((val - 0.75) / 0.25) * 10.0


def db_str(val: float) -> str:
    if math.isinf(val):
        return "-inf dB"
    return f"{val:+.1f} dB"


def pan_to_str(val: float) -> str:
    """0.0=L, 0.5=C, 1.0=R → human string."""
    if abs(val - 0.5) < 0.01:
        return "C"
    pct = int(round(abs(val - 0.5) * 200))
    side = "L" if val < 0.5 else "R"
    return f"{side}{pct}"


# ---------------------------------------------------------------------------
# OSC query layer
# ---------------------------------------------------------------------------

def _build_osc_get(addr: str) -> bytes:
    """Build a minimal OSC message with no arguments (a 'get' request)."""
    import struct
    # Pad address to 4-byte boundary (including null terminator)
    addr_b = addr.encode('ascii') + b'\x00'
    addr_b += b'\x00' * ((4 - len(addr_b) % 4) % 4)
    # Type tag string: no args → ","
    tags = b',\x00\x00\x00'
    return addr_b + tags


def _parse_osc_packet(data: bytes) -> tuple[str, Any] | None:
    """Parse a single OSC message. Returns (address, value) or None."""
    import struct
    try:
        end = data.index(b'\x00')
        addr = data[:end].decode('ascii')
        offset = (end + 4) & ~3
        if offset >= len(data) or data[offset:offset+1] != b',':
            return None
        end2 = data.index(b'\x00', offset)
        typetags = data[offset+1:end2].decode('ascii')
        offset = (end2 + 4) & ~3

        values = []
        for t in typetags:
            if t == 'f':
                val = struct.unpack('>f', data[offset:offset+4])[0]
                values.append(val)
                offset += 4
            elif t == 'i':
                val = struct.unpack('>i', data[offset:offset+4])[0]
                values.append(val)
                offset += 4
            elif t == 's':
                end3 = data.index(b'\x00', offset)
                val = data[offset:end3].decode('ascii', errors='replace')
                values.append(val)
                offset = (end3 + 4) & ~3

        if not values:
            return None
        return addr, (values[0] if len(values) == 1 else values)
    except Exception:
        return None


class MixerQuery:
    """
    Sends OSC get-requests to the mixer and collects responses.

    Uses a single UDP socket bound to local_port for both sending and receiving,
    so the XR18 replies go back to the port we are listening on.
    """

    def __init__(self, host: str, port: int = MIXER_PORT, local_port: int = LOCAL_PORT):
        self.host = host
        self.port = port
        self._dest = (host, port)
        self._results: dict[str, Any] = {}
        self._lock = threading.Lock()

        # Single socket bound to local_port — send and receive on the same fd
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("0.0.0.0", local_port))
        self._sock.settimeout(0.1)

        self._running = True
        self._listener = threading.Thread(target=self._recv_loop, daemon=True)
        self._listener.start()

    def _recv_loop(self):
        while self._running:
            try:
                data, _ = self._sock.recvfrom(4096)
                parsed = _parse_osc_packet(data)
                if parsed:
                    addr, val = parsed
                    with self._lock:
                        self._results[addr] = val
            except socket.timeout:
                pass
            except Exception:
                pass

    def _send(self, addr: str):
        self._sock.sendto(_build_osc_get(addr), self._dest)

    def get(self, addr: str) -> Any:
        """Query a single address; wait up to QUERY_TIMEOUT for a response."""
        self._send(addr)
        deadline = time.monotonic() + QUERY_TIMEOUT
        while time.monotonic() < deadline:
            with self._lock:
                if addr in self._results:
                    return self._results[addr]
            time.sleep(0.005)
        return None

    def get_many(self, addrs: list[str]) -> dict[str, Any]:
        """Send all requests, then wait for all responses."""
        for addr in addrs:
            self._send(addr)
            time.sleep(BATCH_PAUSE)
        deadline = time.monotonic() + QUERY_TIMEOUT + len(addrs) * BATCH_PAUSE
        while time.monotonic() < deadline:
            with self._lock:
                if all(a in self._results for a in addrs):
                    break
            time.sleep(0.01)
        with self._lock:
            return {a: self._results.get(a) for a in addrs}

    def close(self):
        self._running = False
        self._sock.close()


# ---------------------------------------------------------------------------
# Mixer state snapshot
# ---------------------------------------------------------------------------

@dataclass
class ChannelState:
    num: int          # 1-based
    name: str = ""
    fader: float = 0.0       # main LR fader value
    pan: float = 0.5
    on: int = 1               # 1=active, 0=muted
    linked: bool = False      # is this ch part of a stereo link?
    bus_levels: dict[int, float] = field(default_factory=dict)   # bus# → level value
    bus_on:     dict[int, int]   = field(default_factory=dict)   # bus# → on value


@dataclass
class BusState:
    num: int
    name: str = ""
    fader: float = 0.0
    on: int = 1


class MixerUnreachable(Exception):
    """
    Raised when the mixer returns no OSC responses at all — almost always means
    it is powered off, unplugged, or the host/port is wrong. Without this guard
    the linter would run against all-default state and emit a flood of
    meaningless warnings (e.g. every stereo pair reported as mis-panned).
    """
    def __init__(self, host: str, port: int, num_queries: int):
        self.host = host
        self.port = port
        self.num_queries = num_queries
        super().__init__(
            f"no response from {host}:{port} "
            f"({num_queries} queries sent, 0 replies)"
        )


def fetch_mixer_state(q: MixerQuery) -> tuple[list[ChannelState], list[BusState], dict]:
    """Read all relevant state from the mixer in as few round-trips as practical."""

    channels = [ChannelState(num=i) for i in range(1, NUM_CHANNELS + 1)]
    buses    = [BusState(num=i)     for i in range(1, NUM_BUSES + 1)]

    # --- Build address list ---
    addrs = []

    # Channel basics
    for ch in channels:
        n = f"{ch.num:02d}"
        addrs += [
            f"/ch/{n}/config/name",
            f"/ch/{n}/mix/fader",
            f"/ch/{n}/mix/pan",
            f"/ch/{n}/mix/on",
        ]
        for bus in range(1, NUM_BUSES + 1):
            b = f"{bus:02d}"
            addrs += [
                f"/ch/{n}/mix/{b}/level",
                f"/ch/{n}/mix/{b}/on",
            ]

    # Bus basics
    # NOTE: XR18 bus *master* objects use single-digit numbers (/bus/1/...),
    # unlike channel send sub-addresses (/ch/01/mix/01/level) which are 2-digit.
    for bus in buses:
        n = f"{bus.num}"
        addrs += [
            f"/bus/{n}/config/name",
            f"/bus/{n}/mix/fader",
            f"/bus/{n}/mix/on",
        ]

    # Channel link config (16-bit bitmask, one bit per pair)
    addrs.append("/config/chlink")

    console.print(f"[dim]Querying {len(addrs)} OSC addresses...[/dim]")
    results = q.get_many(addrs)

    # If the mixer sent nothing back at all, it's offline/unreachable. Bail out
    # rather than lint the all-default snapshot (which produces spurious warnings).
    if not any(v is not None for v in results.values()):
        raise MixerUnreachable(q.host, q.port, len(addrs))

    # --- Decode channel link state ---
    # The XR18 returns /config/chlink as a LIST of per-pair flags, one per
    # stereo pair: [1/2, 3/4, 5/6, 7/8, 9/10, 11/12, 13/14, 15/16].
    # Some firmware/devices may instead return a single integer bitmask, so
    # handle both: list -> index is pair number; int -> bit N is pair N.
    chlink_raw = results.get("/config/chlink", 0)
    if chlink_raw is None:
        chlink_raw = 0

    linked_channels: set[int] = set()
    if isinstance(chlink_raw, list):
        for idx, flag in enumerate(chlink_raw):
            if int(flag):
                linked_channels.add(idx * 2 + 1)   # 1-based left
                linked_channels.add(idx * 2 + 2)   # 1-based right
    else:
        # Fallback: treat as a bitmask (bit N -> pair 2N+1 / 2N+2)
        for bit in range(8):
            if int(chlink_raw) & (1 << bit):
                linked_channels.add(bit * 2 + 1)
                linked_channels.add(bit * 2 + 2)

    def _f(v, default: float) -> float:
        """Float with explicit None check — avoids `0.0 or default` pitfall."""
        return float(v) if v is not None else default

    def _i(v, default: int) -> int:
        """Int with explicit None check — avoids `0 or default` pitfall."""
        return int(v) if v is not None else default

    # --- Populate channel state ---
    for ch in channels:
        n = f"{ch.num:02d}"
        ch.name   = str(results.get(f"/ch/{n}/config/name") or "").strip()
        ch.fader  = _f(results.get(f"/ch/{n}/mix/fader"), 0.0)
        ch.pan    = _f(results.get(f"/ch/{n}/mix/pan"),   0.5)
        ch.on     = _i(results.get(f"/ch/{n}/mix/on"),    1)
        ch.linked = ch.num in linked_channels
        for bus in range(1, NUM_BUSES + 1):
            b = f"{bus:02d}"
            ch.bus_levels[bus] = _f(results.get(f"/ch/{n}/mix/{b}/level"), 0.0)
            ch.bus_on[bus]     = _i(results.get(f"/ch/{n}/mix/{b}/on"),    1)

    # --- Populate bus state ---
    for bus in buses:
        n = f"{bus.num}"
        bus.name  = str(results.get(f"/bus/{n}/config/name") or "").strip()
        bus.fader = _f(results.get(f"/bus/{n}/mix/fader"), 0.0)
        bus.on    = _i(results.get(f"/bus/{n}/mix/on"),    1)

    meta = {
        "linked_channels": linked_channels,
        "chlink_raw": chlink_raw,
    }
    return channels, buses, meta


# ---------------------------------------------------------------------------
# Lint issue
# ---------------------------------------------------------------------------

@dataclass
class Issue:
    severity: str   # "error" | "warn" | "info"
    check: str
    subject: str
    detail: str


@dataclass
class CheckResult:
    """Outcome of running a single lint check (used for verbose reporting)."""
    name: str
    status: str                              # "ok" | "issues" | "skipped"
    issues: list[Issue] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Individual lint checks
# ---------------------------------------------------------------------------

def check_pan_center(channels: list[ChannelState], cfg: dict) -> list[Issue]:
    """Warn when a non-linked channel's pan is not centered."""
    issues = []
    tolerance = float(cfg.get("pan_tolerance", 0.03))
    skip_muted = bool(cfg.get("skip_muted_channels", True))
    skip_nums  = set(cfg.get("skip_channels", []))

    for ch in channels:
        if ch.num in skip_nums:
            continue
        if skip_muted and ch.on == 0:
            continue
        if ch.linked:
            continue   # stereo-linked pairs are expected to be panned
        if abs(ch.pan - 0.5) > tolerance:
            issues.append(Issue(
                severity="warn",
                check="pan_center",
                subject=f"CH{ch.num:02d} {ch.name!r}",
                detail=f"Pan is {pan_to_str(ch.pan)} (expected center)",
            ))
    return issues


def check_stereo_pair_balance(channels: list[ChannelState], cfg: dict) -> list[Issue]:
    """
    For explicitly configured L/R pairs, warn if the main fader levels
    differ by more than the tolerance.
    """
    issues = []
    tolerance_db = float(cfg.get("tolerance_db", 3.0))
    pairs = cfg.get("pairs", [])

    ch_by_num = {c.num: c for c in channels}

    for pair in pairs:
        left_n  = int(pair["left"])
        right_n = int(pair["right"])
        name    = pair.get("name", f"CH{left_n:02d}/CH{right_n:02d}")

        l = ch_by_num.get(left_n)
        r = ch_by_num.get(right_n)
        if l is None or r is None:
            continue

        l_db = fader_to_db(l.fader)
        r_db = fader_to_db(r.fader)

        if math.isinf(l_db) and math.isinf(r_db):
            continue  # both off, fine

        diff = abs(l_db - r_db) if not (math.isinf(l_db) or math.isinf(r_db)) else math.inf

        if diff > tolerance_db:
            issues.append(Issue(
                severity="warn",
                check="stereo_balance",
                subject=name,
                detail=(
                    f"L (CH{left_n:02d}) = {db_str(l_db)}, "
                    f"R (CH{right_n:02d}) = {db_str(r_db)}, "
                    f"diff = {diff:.1f} dB (tolerance {tolerance_db} dB)"
                ),
            ))

    return issues


def check_stereo_pair_pan(channels: list[ChannelState], cfg: dict) -> list[Issue]:
    """
    For explicitly configured L/R pairs, verify that the left channel is panned
    fully left and the right channel is fully right (within tolerance).
    Expected pan values are configurable; defaults are L100 / R100.
    """
    issues = []
    pairs         = cfg.get("pairs", [])
    tolerance     = float(cfg.get("pan_tolerance", 0.03))   # same units as pan (0–1)
    expected_left  = float(cfg.get("expected_left_pan",  0.0))   # 0.0 = full left
    expected_right = float(cfg.get("expected_right_pan", 1.0))   # 1.0 = full right
    skip_muted    = bool(cfg.get("skip_muted_channels", True))

    ch_by_num = {c.num: c for c in channels}

    for pair in pairs:
        left_n  = int(pair["left"])
        right_n = int(pair["right"])
        name    = pair.get("name", f"CH{left_n:02d}/CH{right_n:02d}")

        # Allow per-pair overrides
        p_tol        = float(pair.get("pan_tolerance",     tolerance))
        p_exp_left   = float(pair.get("expected_left_pan",  expected_left))
        p_exp_right  = float(pair.get("expected_right_pan", expected_right))

        for ch_n, expected, side_label in (
            (left_n,  p_exp_left,  "L"),
            (right_n, p_exp_right, "R"),
        ):
            ch = ch_by_num.get(ch_n)
            if ch is None:
                continue
            if skip_muted and ch.on == 0:
                continue
            if abs(ch.pan - expected) > p_tol:
                exp_str = pan_to_str(expected)
                issues.append(Issue(
                    severity="warn",
                    check="stereo_pan",
                    subject=f"{name} {side_label} (CH{ch_n:02d})",
                    detail=(
                        f"Pan is {pan_to_str(ch.pan)}, expected {exp_str} "
                        f"(tolerance {p_tol:.2f})"
                    ),
                ))

    return issues


def check_linked_pair_balance(channels: list[ChannelState], cfg: dict) -> list[Issue]:
    """
    For hardware-linked stereo pairs, check that fader levels match.
    On the XR18 linked pairs usually mirror automatically, but it's worth confirming.
    """
    issues = []
    tolerance_db = float(cfg.get("tolerance_db", 1.0))
    skip_muted   = bool(cfg.get("skip_muted_channels", True))

    ch_by_num = {c.num: c for c in channels}

    seen = set()
    for ch in channels:
        if not ch.linked or ch.num in seen:
            continue
        # Find partner: odd → partner is odd+1, even → partner is even-1
        if ch.num % 2 == 1:
            partner_num = ch.num + 1
        else:
            partner_num = ch.num - 1
        seen.add(ch.num)
        seen.add(partner_num)

        partner = ch_by_num.get(partner_num)
        if partner is None:
            continue
        if skip_muted and (ch.on == 0 or partner.on == 0):
            continue

        l_db = fader_to_db(ch.fader)
        r_db = fader_to_db(partner.fader)
        if math.isinf(l_db) and math.isinf(r_db):
            continue
        diff = abs(l_db - r_db) if not (math.isinf(l_db) or math.isinf(r_db)) else math.inf
        if diff > tolerance_db:
            issues.append(Issue(
                severity="warn",
                check="linked_pair_balance",
                subject=f"CH{ch.num:02d}/{partner_num:02d}",
                detail=(
                    f"Hardware-linked pair has mismatched faders: "
                    f"CH{ch.num:02d}={db_str(l_db)}, CH{partner_num:02d}={db_str(r_db)}, "
                    f"diff={diff:.1f} dB"
                ),
            ))

    return issues


def check_aux_send_balance(channels: list[ChannelState], buses: list[BusState],
                           cfg: dict) -> list[Issue]:
    """
    For input stereo pairs routed to a non-main (aux) output bus that is itself
    a stereo pair, verify the send levels feeding the two halves of the output
    pair are balanced. Two wiring patterns are accepted:

      * Stereo (diagonal): left input -> left output, right input -> right output
        (both nonzero, cross-sends off). The two direct sends must match.
      * Dual-mono: each input channel feeds BOTH output halves at the same level;
        all four sends must match.

    Additionally, the two output bus *master* faders of each pair must match
    within tolerance (the "output fader" of both halves should be the same).

    Input pairs default to stereo_balance.pairs (injected as cfg["pairs"]).
    Output buses come from cfg["output_pairs"]. Pan is not considered. Pairs
    that send nothing to an output bus are skipped (not routed there).
    """
    issues = []
    tolerance_db = float(cfg.get("tolerance_db", 1.5))
    floor_db     = float(cfg.get("floor_db", -60.0))
    skip_muted   = bool(cfg.get("skip_muted_channels", False))
    input_pairs  = cfg.get("pairs", [])
    output_pairs = cfg.get("output_pairs", [])

    ch_by_num  = {c.num: c for c in channels}
    bus_by_num = {b.num: b for b in buses}

    def send_db(ch: ChannelState, bus: int) -> float:
        return fader_to_db(ch.bus_levels.get(bus, 0.0))

    def is_active(ch: ChannelState, bus: int) -> bool:
        """A send counts as routed if it's on and above the floor."""
        if ch.bus_on.get(bus, 1) == 0:
            return False
        db = send_db(ch, bus)
        return not (math.isinf(db) or db <= floor_db)

    for opair in output_pairs:
        busL  = int(opair["left"])
        busR  = int(opair["right"])
        oname = opair.get("name", f"BUS{busL:02d}/{busR:02d}")

        # --- Output bus master fader balance ---
        oL = bus_by_num.get(busL)
        oR = bus_by_num.get(busR)
        if oL is not None and oR is not None:
            l_db = fader_to_db(oL.fader)
            r_db = fader_to_db(oR.fader)
            if not (math.isinf(l_db) and math.isinf(r_db)):
                diff = abs(l_db - r_db) if not (math.isinf(l_db) or math.isinf(r_db)) else math.inf
                if diff > tolerance_db:
                    issues.append(Issue(
                        severity="warn",
                        check="aux_output_fader",
                        subject=f"{oname} output masters",
                        detail=(
                            f"Output fader mismatch: BUS{busL:02d}={db_str(l_db)}, "
                            f"BUS{busR:02d}={db_str(r_db)}, "
                            f"diff={diff:.1f} dB (tolerance {tolerance_db} dB)"
                        ),
                    ))

        for ipair in input_pairs:
            cL_n  = int(ipair["left"])
            cR_n  = int(ipair["right"])
            iname = ipair.get("name", f"CH{cL_n:02d}/{cR_n:02d}")

            cL = ch_by_num.get(cL_n)
            cR = ch_by_num.get(cR_n)
            if cL is None or cR is None:
                continue
            if skip_muted and (cL.on == 0 or cR.on == 0):
                continue

            subject = f"{iname} -> {oname}"

            # Four sends: direct (LL, RR) and cross-feed (LR, RL)
            ll_db, ll_a = send_db(cL, busL), is_active(cL, busL)   # L in -> L out
            lr_db, lr_a = send_db(cL, busR), is_active(cL, busR)   # L in -> R out
            rl_db, rl_a = send_db(cR, busL), is_active(cR, busL)   # R in -> L out
            rr_db, rr_a = send_db(cR, busR), is_active(cR, busR)   # R in -> R out

            # Not routed to this output pair at all — nothing to check.
            if not (ll_a or lr_a or rl_a or rr_a):
                continue

            cross_active = lr_a or rl_a

            if not cross_active:
                # --- Stereo (diagonal) pattern: only LL and RR should be used ---
                if not (ll_a and rr_a):
                    missing = f"left (CH{cL_n:02d}->BUS{busL:02d})" if not ll_a \
                        else f"right (CH{cR_n:02d}->BUS{busR:02d})"
                    issues.append(Issue(
                        severity="warn",
                        check="aux_send_balance",
                        subject=subject,
                        detail=(
                            f"Stereo send routed to only one side; {missing} is off "
                            f"(CH{cL_n:02d}->BUS{busL:02d}={db_str(ll_db)}, "
                            f"CH{cR_n:02d}->BUS{busR:02d}={db_str(rr_db)})"
                        ),
                    ))
                    continue
                diff = abs(ll_db - rr_db)
                if diff > tolerance_db:
                    issues.append(Issue(
                        severity="warn",
                        check="aux_send_balance",
                        subject=subject,
                        detail=(
                            f"Send imbalance: CH{cL_n:02d}->BUS{busL:02d}={db_str(ll_db)}, "
                            f"CH{cR_n:02d}->BUS{busR:02d}={db_str(rr_db)}, "
                            f"diff={diff:.1f} dB (tolerance {tolerance_db} dB)"
                        ),
                    ))
            else:
                # --- Dual-mono pattern: all four sends should be on and equal ---
                sends = [
                    (f"CH{cL_n:02d}->BUS{busL:02d}", ll_db, ll_a),
                    (f"CH{cL_n:02d}->BUS{busR:02d}", lr_db, lr_a),
                    (f"CH{cR_n:02d}->BUS{busL:02d}", rl_db, rl_a),
                    (f"CH{cR_n:02d}->BUS{busR:02d}", rr_db, rr_a),
                ]
                inactive = [label for (label, _, a) in sends if not a]
                if inactive:
                    issues.append(Issue(
                        severity="warn",
                        check="aux_send_balance",
                        subject=subject,
                        detail=(
                            f"Dual-mono send is partially routed; off: "
                            f"{', '.join(inactive)}"
                        ),
                    ))
                    continue
                dbs = [d for (_, d, _) in sends]
                spread = max(dbs) - min(dbs)
                if spread > tolerance_db:
                    parts = ", ".join(f"{label}={db_str(d)}" for (label, d, _) in sends)
                    issues.append(Issue(
                        severity="warn",
                        check="aux_send_balance",
                        subject=subject,
                        detail=(
                            f"Dual-mono send imbalance: {parts}; "
                            f"spread={spread:.1f} dB (tolerance {tolerance_db} dB)"
                        ),
                    ))

    return issues


def check_bus_levels(channels: list[ChannelState], cfg: dict) -> list[Issue]:
    """
    Check that specified channels have expected send levels on specified buses.
    Supports:
      expect = "zero"    → level should be at or below max_level_db (default -60)
      expect = "nonzero" → level should be above min_level_db (default -60)
      expect = "off"     → bus send should be switched off (on == 0)
    """
    issues = []
    rules = cfg.get("rules", [])
    ch_by_num = {c.num: c for c in channels}

    for rule in rules:
        desc        = rule.get("description", "")
        ch_nums     = rule.get("channels", [])
        bus_nums    = rule.get("buses", [])
        expect      = rule.get("expect", "zero")
        max_lvl     = float(rule.get("max_level_db", -60.0))
        min_lvl     = float(rule.get("min_level_db", -60.0))
        skip_muted  = bool(rule.get("skip_muted_channels", False))

        for ch_n in ch_nums:
            ch = ch_by_num.get(int(ch_n))
            if ch is None:
                continue
            if skip_muted and ch.on == 0:
                continue
            for bus_n in bus_nums:
                bus_n = int(bus_n)
                level_val = ch.bus_levels.get(bus_n, 0.0)
                on_val    = ch.bus_on.get(bus_n, 1)
                level_db  = fader_to_db(level_val)

                if expect == "zero":
                    if not math.isinf(level_db) and level_db > max_lvl:
                        issues.append(Issue(
                            severity="warn",
                            check="bus_level",
                            subject=f"CH{ch_n:02d}->BUS{bus_n:02d}",
                            detail=f"{desc}: send level is {db_str(level_db)} (expected <= {max_lvl} dB)",
                        ))
                elif expect == "nonzero":
                    if math.isinf(level_db) or level_db < min_lvl:
                        issues.append(Issue(
                            severity="info",
                            check="bus_level",
                            subject=f"CH{ch_n:02d}->BUS{bus_n:02d}",
                            detail=f"{desc}: send level is {db_str(level_db)} (expected > {min_lvl} dB)",
                        ))
                elif expect == "off":
                    if on_val != 0:
                        issues.append(Issue(
                            severity="warn",
                            check="bus_level",
                            subject=f"CH{ch_n:02d}->BUS{bus_n:02d}",
                            detail=f"{desc}: bus send is ON (expected OFF)",
                        ))

    return issues


def check_fader_range(channels: list[ChannelState], buses: list[BusState], cfg: dict) -> list[Issue]:
    """Warn when a channel or bus fader is suspiciously high or low."""
    issues = []
    max_db   = float(cfg.get("max_db",  6.0))
    min_db   = float(cfg.get("min_db", -40.0))
    skip_muted = bool(cfg.get("skip_muted_channels", True))
    skip_nums  = set(cfg.get("skip_channels", []))

    targets: list[tuple[str, float, int]] = []
    for ch in channels:
        if ch.num not in skip_nums:
            targets.append((f"CH{ch.num:02d} {ch.name!r}", ch.fader, ch.on))
    for bus in buses:
        targets.append((f"BUS{bus.num:02d} {bus.name!r}", bus.fader, bus.on))

    for label, fader, on in targets:
        if skip_muted and on == 0:
            continue
        db = fader_to_db(fader)
        if math.isinf(db):
            continue   # fully off — handled elsewhere if needed
        if db > max_db:
            issues.append(Issue(
                severity="warn",
                check="fader_range",
                subject=label,
                detail=f"Fader is {db_str(db)} (above max {max_db:+.1f} dB)",
            ))
        elif db < min_db:
            issues.append(Issue(
                severity="info",
                check="fader_range",
                subject=label,
                detail=f"Fader is {db_str(db)} (below min {min_db:+.1f} dB — maybe intentional)",
            ))

    return issues


def check_unnamed_channels(channels: list[ChannelState], cfg: dict) -> list[Issue]:
    """Flag channels that have no name assigned."""
    issues = []
    skip_nums   = set(cfg.get("skip_channels", []))
    only_active = bool(cfg.get("only_active_channels", False))

    for ch in channels:
        if ch.num in skip_nums:
            continue
        if only_active and fader_to_db(ch.fader) < -60:
            continue   # channel is off, skip
        if not ch.name:
            issues.append(Issue(
                severity="info",
                check="unnamed_channel",
                subject=f"CH{ch.num:02d}",
                detail="Channel has no name",
            ))

    return issues


def check_muted_but_fader_up(channels: list[ChannelState], cfg: dict) -> list[Issue]:
    """Flag channels that are muted but have their fader up — possible confusion."""
    issues = []
    min_db    = float(cfg.get("min_fader_db", -10.0))
    skip_nums = set(cfg.get("skip_channels", []))

    for ch in channels:
        if ch.num in skip_nums:
            continue
        if ch.on == 0:
            db = fader_to_db(ch.fader)
            if not math.isinf(db) and db >= min_db:
                issues.append(Issue(
                    severity="info",
                    check="muted_fader_up",
                    subject=f"CH{ch.num:02d} {ch.name!r}",
                    detail=f"Channel is MUTED but fader is {db_str(db)}",
                ))

    return issues


# ---------------------------------------------------------------------------
# Run all checks
# ---------------------------------------------------------------------------

CHECKS = {
    "pan_center":        check_pan_center,
    "stereo_balance":    check_stereo_pair_balance,
    "stereo_pan":        check_stereo_pair_pan,
    "linked_balance":    check_linked_pair_balance,
    "aux_send_balance":  check_aux_send_balance,
    "bus_levels":        check_bus_levels,
    "fader_range":       check_fader_range,
    "unnamed_channels":  check_unnamed_channels,
    "muted_fader_up":    check_muted_but_fader_up,
}


def run_checks(channels: list[ChannelState], buses: list[BusState],
               cfg: dict) -> list[CheckResult]:
    results: list[CheckResult] = []
    checks_cfg = cfg.get("checks", {})

    # Collect all channel numbers that appear in any configured stereo pair
    pair_members: set[int] = set()
    for pair in checks_cfg.get("stereo_balance", {}).get("pairs", []):
        pair_members.add(int(pair["left"]))
        pair_members.add(int(pair["right"]))

    for name, fn in CHECKS.items():
        check_cfg = checks_cfg.get(name, {})
        if not check_cfg:
            results.append(CheckResult(name=name, status="skipped"))
            continue  # not configured — skip
        if check_cfg is True:
            check_cfg = {}
        if isinstance(check_cfg, dict) and not check_cfg.get("enabled", True):
            results.append(CheckResult(name=name, status="skipped"))
            continue

        if name == "pan_center":
            # Inject pair_members so the check can skip stereo pair sides
            merged = dict(check_cfg)
            existing_skip = set(merged.get("skip_channels", []))
            merged["skip_channels"] = list(existing_skip | pair_members)
            found = fn(channels, merged)
        elif name == "stereo_pan":
            # Inherit pairs list from stereo_balance if not overridden locally
            sb_cfg = checks_cfg.get("stereo_balance", {})
            merged = {"pairs": sb_cfg.get("pairs", [])}
            merged.update(check_cfg)
            found = fn(channels, merged)
        elif name == "aux_send_balance":
            # Inherit input pairs from stereo_balance unless overridden locally
            sb_cfg = checks_cfg.get("stereo_balance", {})
            merged = {"pairs": sb_cfg.get("pairs", [])}
            merged.update(check_cfg)
            found = fn(channels, buses, merged)
        elif name == "fader_range":
            found = fn(channels, buses, check_cfg)
        else:
            found = fn(channels, check_cfg)

        results.append(CheckResult(
            name=name,
            status="issues" if found else "ok",
            issues=found,
        ))

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

SEVERITY_COLOR = {
    "error": "bold red",
    "warn":  "yellow",
    "info":  "dim",
}
SEVERITY_ICON = {
    "error": "X",
    "warn":  "!",
    "info":  "i",
}


def print_issues(issues: list[Issue], channels: list[ChannelState],
                 buses: list[BusState]):
    if not issues:
        console.print("[bold green]OK - No issues found.[/bold green]")
        return

    by_sev: dict[str, list[Issue]] = defaultdict(list)
    for iss in issues:
        by_sev[iss.severity].append(iss)

    table = Table(title="Mixer Lint Results", show_lines=False)
    table.add_column("Sev",     style="bold", width=6, no_wrap=True)
    table.add_column("Check",   style="cyan", width=18, no_wrap=True)
    table.add_column("Subject", width=24, no_wrap=True)
    table.add_column("Detail")

    for sev in ("error", "warn", "info"):
        for iss in by_sev.get(sev, []):
            color = SEVERITY_COLOR[sev]
            table.add_row(
                f"[{color}]{sev}[/{color}]",
                iss.check,
                iss.subject,
                iss.detail,
            )

    console.print(table)
    console.print(
        f"[bold]{len(issues)} issue(s)[/bold]: "
        f"[red]{len(by_sev['error'])} error(s)[/red], "
        f"[yellow]{len(by_sev['warn'])} warning(s)[/yellow], "
        f"[dim]{len(by_sev['info'])} info[/dim]"
    )


def print_check_runs(results: list[CheckResult]):
    """
    Verbose report: list every check and its result — including checks that
    passed cleanly (OK) and checks that were skipped because they are disabled
    or absent from the config.
    """
    table = Table(title="Checks Run", show_lines=True)
    table.add_column("Check",  style="cyan", width=20, no_wrap=True)
    table.add_column("Result", width=10,     no_wrap=True)
    table.add_column("Detail")

    for r in results:
        if r.status == "ok":
            result_cell = "[green]OK[/green]"
            detail = "[dim]no issues[/dim]"
        elif r.status == "issues":
            result_cell = f"[yellow]{len(r.issues)} issue(s)[/yellow]"
            lines = []
            for iss in r.issues:
                color = SEVERITY_COLOR[iss.severity]
                icon  = SEVERITY_ICON[iss.severity]
                lines.append(f"[{color}]{icon}[/{color}] {iss.subject}: {iss.detail}")
            detail = "\n".join(lines)
        else:
            result_cell = "[dim]skipped[/dim]"
            detail = "[dim]disabled or not in config[/dim]"
        table.add_row(r.name, result_cell, detail)

    console.print(table)


def print_state_summary(channels: list[ChannelState], buses: list[BusState],
                        verbose: bool = False):
    """Print a compact overview of the mixer state."""
    table = Table(title="Channel Summary", show_header=True)
    table.add_column("CH",    width=4)
    table.add_column("Name",  width=14)
    table.add_column("Fader", width=9)
    table.add_column("Pan",   width=5)
    if verbose:
        table.add_column("Pan raw", width=8)
    table.add_column("On",    width=4)
    table.add_column("Linked",width=6)

    for ch in channels:
        db    = fader_to_db(ch.fader)
        on_s  = "[green]ON[/green]" if ch.on else "[red]MUTE[/red]"
        lnk_s = "[cyan]L[/cyan]" if ch.linked else ""
        row = [
            f"{ch.num:02d}",
            ch.name or "[dim]--[/dim]",
            db_str(db),
            pan_to_str(ch.pan),
        ]
        if verbose:
            row.append(f"{ch.pan:.4f}")
        row += [on_s, lnk_s]
        table.add_row(*row)

    console.print(table)

    bus_table = Table(title="Bus Summary")
    bus_table.add_column("BUS",   width=4)
    bus_table.add_column("Name",  width=14)
    bus_table.add_column("Fader", width=9)
    bus_table.add_column("On",    width=4)
    for bus in buses:
        db   = fader_to_db(bus.fader)
        on_s = "[green]ON[/green]" if bus.on else "[red]OFF[/red]"
        bus_table.add_row(f"{bus.num:02d}", bus.name or "[dim]--[/dim]", db_str(db), on_s)

    console.print(bus_table)


# ---------------------------------------------------------------------------
# Simulation (sample data that triggers every check)
# ---------------------------------------------------------------------------

def build_simulation() -> tuple[list[ChannelState], list[BusState], dict]:
    """
    Construct a synthetic mixer snapshot and a matching config crafted so that
    every lint check fires at least once. Used by --simulate to preview the
    full range of report output without querying a real mixer.

    Fader float reference (XR18 curve): 0.75 = 0 dB, 0.99 ~ +9.6 dB,
    0.625 = -5 dB, 0.55 ~ -8 dB, 0.375 = -25 dB.
    """
    DB0, HOT, DOWN5, DOWN8, LOW25 = 0.75, 0.99, 0.625, 0.55, 0.375

    def mk(num, name, fader, pan, on=1, linked=False, sends=None):
        ch = ChannelState(num=num, name=name, fader=fader, pan=pan,
                          on=on, linked=linked)
        for bus, (lvl, son) in (sends or {}).items():
            ch.bus_levels[bus] = lvl
            ch.bus_on[bus] = son
        return ch

    channels = [
        mk(1,  "PanOff",   DB0,   0.70),                                     # pan_center
        mk(2,  "Vox",      DB0,   0.50),
        mk(3,  "LinkL",    DB0,   0.00, linked=True),                        # linked_balance (L)
        mk(4,  "LinkR",    DOWN5, 1.00, linked=True),                        # linked_balance (R mismatched)
        mk(5,  "BalL",     DB0,   0.00),                                     # stereo_balance (L)
        mk(6,  "BalR",     DOWN5, 1.00),                                     # stereo_balance (R mismatched)
        mk(7,  "PanPairL", DB0,   0.50),                                     # stereo_pan (L mis-panned)
        mk(8,  "PanPairR", DB0,   1.00),
        mk(9,  "AuxL",     DB0,   0.00, sends={1: (DB0, 1), 2: (0.0, 1)}),   # aux_send (L->busL)
        mk(10, "AuxR",     DB0,   1.00, sends={1: (0.0, 1), 2: (DOWN8, 1)}), # aux_send (R->busR imbalanced)
        mk(11, "HotFader", HOT,   0.50),                                     # fader_range (high)
        mk(12, "LowFader", LOW25, 0.50),                                     # fader_range (low / info)
        mk(13, "",         DB0,   0.50),                                     # unnamed_channels (info)
        mk(14, "MutedUp",  DB0,   0.50, on=0),                               # muted_fader_up (info)
        mk(15, "BusSend",  DB0,   0.50, sends={4: (DB0, 1)}),               # bus_levels rule
        mk(16, "Spare",    DB0,   0.50),
    ]

    buses = [
        BusState(num=1, name="AuxOutL", fader=DB0,   on=1),   # aux_output_fader mismatch (L)
        BusState(num=2, name="AuxOutR", fader=DOWN8, on=1),   # aux_output_fader mismatch (R)
        BusState(num=3, name="Bus3",    fader=DB0,   on=1),
        BusState(num=4, name="Bus4",    fader=DB0,   on=1),
        BusState(num=5, name="Bus5",    fader=DB0,   on=1),
        BusState(num=6, name="Bus6",    fader=DB0,   on=1),
    ]

    cfg = {
        "checks": {
            "pan_center": {
                "enabled": True, "pan_tolerance": 0.02,
                "skip_muted_channels": False,
            },
            "stereo_balance": {
                "enabled": True, "tolerance_db": 1.5,
                "pairs": [
                    {"left": 5, "right": 6,  "name": "BalPair"},
                    {"left": 7, "right": 8,  "name": "PanPair"},
                    {"left": 9, "right": 10, "name": "AuxIn"},
                ],
            },
            "stereo_pan": {
                "enabled": True, "pan_tolerance": 0.03,
                "expected_left_pan": 0.0, "expected_right_pan": 1.0,
                "skip_muted_channels": False,
            },
            "linked_balance": {
                "enabled": True, "tolerance_db": 1.0,
                "skip_muted_channels": False,
            },
            "aux_send_balance": {
                "enabled": True, "tolerance_db": 1.5,
                "output_pairs": [{"left": 1, "right": 2, "name": "AuxOut"}],
            },
            "bus_levels": {
                "enabled": True,
                "rules": [{
                    "description": "Spare send should be clean",
                    "channels": [15], "buses": [4],
                    "expect": "zero", "max_level_db": -60,
                }],
            },
            "fader_range": {
                "enabled": True, "max_db": 9.0, "min_db": -20.0,
                "skip_muted_channels": True,
            },
            "unnamed_channels": {"enabled": True, "only_active_channels": False},
            "muted_fader_up": {"enabled": True, "min_fader_db": -10.0},
        }
    }

    return channels, buses, cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Lint a Behringer XR18 mixer via OSC")
    parser.add_argument("--config",  default="mixerlint.yaml",  help="Config file path")
    parser.add_argument("--host",    default=None,              help="Mixer IP (overrides config)")
    parser.add_argument("--port",    type=int, default=None,    help="Mixer OSC port (overrides config)")
    parser.add_argument("--summary", action="store_true",       help="Also print channel/bus summary table")
    parser.add_argument("--verbose", action="store_true",       help="Show every check run and its result (even OK checks); also adds raw OSC values to the summary")
    parser.add_argument("--simulate", action="store_true",      help="Generate a sample report from synthetic data that triggers every check (no mixer needed)")
    parser.add_argument("--timeout", type=float, default=None,  help="Per-query timeout in seconds")
    args = parser.parse_args()

    # --- Simulation mode: synthetic data, no mixer or config file needed ---
    if args.simulate:
        console.print(
            "[bold]mixerlint[/bold] - [magenta]SIMULATION[/magenta] "
            "(synthetic data; no mixer queried)"
        )
        channels, buses, cfg = build_simulation()
        if args.summary or args.verbose:
            print_state_summary(channels, buses, verbose=args.verbose)
        results = run_checks(channels, buses, cfg)
        issues = [iss for r in results for iss in r.issues]
        print_check_runs(results)
        print_issues(issues, channels, buses)
        has_problems = any(i.severity in ("error", "warn") for i in issues)
        sys.exit(1 if has_problems else 0)

    # Load config
    try:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        console.print(f"[red]Config file not found: {args.config}[/red]")
        sys.exit(1)

    mixer_cfg = cfg.get("mixer", {})
    host = args.host or mixer_cfg.get("host", "192.168.0.28")
    port = args.port or int(mixer_cfg.get("port", MIXER_PORT))

    global QUERY_TIMEOUT
    if args.timeout:
        QUERY_TIMEOUT = args.timeout
    elif "timeout" in mixer_cfg:
        QUERY_TIMEOUT = float(mixer_cfg["timeout"])

    console.print(f"[bold]mixerlint[/bold] - connecting to [cyan]{host}:{port}[/cyan]")

    q = MixerQuery(host, port)

    try:
        channels, buses, meta = fetch_mixer_state(q)
    except MixerUnreachable as e:
        console.print(f"[bold red]Mixer not responding[/bold red] - {e}")
        console.print(
            "[dim]Check that the mixer is powered on and reachable, and that "
            "the host/port are correct.[/dim]"
        )
        sys.exit(2)
    finally:
        q.close()

    if args.summary or args.verbose:
        print_state_summary(channels, buses, verbose=args.verbose)

    results = run_checks(channels, buses, cfg)
    issues = [iss for r in results for iss in r.issues]

    if args.verbose:
        print_check_runs(results)

    print_issues(issues, channels, buses)

    # Exit code: 1 if any errors/warnings, 0 if clean
    has_problems = any(i.severity in ("error", "warn") for i in issues)
    sys.exit(1 if has_problems else 0)


if __name__ == "__main__":
    main()
