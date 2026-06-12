"""Bit-field extractors and CRC/parity helpers shared by all decoders.

Three indexing conventions coexist (kept ICD-faithful): ``_u/_s`` operate
on a list of 32-bit words; ``_cu/_cs`` index a single int 1-based from the
MSB (CNAV); ``_bu/_bs/_bg`` index a single int 0-based from the MSB
(Galileo/BeiDou/GLONASS/SBAS)."""

import math


# ---------------------------------------------------------------------------
# GPS L1 C/A LNAV decoder (RXM-SFRBX, gnssId=0, sigId=0), per IS-GPS-200.
#
# u-blox reports each of the 10 words as a 32-bit value with the 30-bit GPS
# word right-justified. The 24 data bits are (dwrd >> 6) & 0xFFFFFF, MSB first,
# with the parity inversion already removed by the receiver (word-1 preamble
# reads 0x8B directly), so no D30* handling is required.
# ---------------------------------------------------------------------------

SC = math.pi  # 1 semicircle = pi radians


def _u(words, w, start, length):
    """Unsigned field: word index w (0-based), bit `start` (1-based, MSB=1), length bits."""
    shift = 25 - start - length
    return (words[w] >> shift) & ((1 << length) - 1)


def _twos(val, nbits):
    return val - (1 << nbits) if val & (1 << (nbits - 1)) else val


def _s(words, w, start, length):
    """Signed (two's complement) single-word field."""
    return _twos(_u(words, w, start, length), length)


def cnav_message_int(dwrds):
    """Concatenate 10 x 32-bit words and return the top 300 bits as an int."""
    full = 0
    for w in dwrds:
        full = (full << 32) | (w & 0xFFFFFFFF)
    return full >> 20  # drop the 20 padding bits below the 300-bit message


def _crc24q(bits):
    """CRC-24Q remainder (poly 0x1864CFB) over an iterable of MSB-first bits.

    Shared by GPS/QZSS CNAV, Galileo I/NAV, BeiDou B-CNAV1/2 and SBAS.
    """
    crc = 0
    for b in bits:
        msb = (crc >> 23) & 1
        crc = (crc << 1) & 0xFFFFFF
        if msb ^ b:
            crc ^= 0x864CFB
    return crc


def _crc24q_head(value, total_bits, data_bits):
    """CRC-24Q over the most-significant `data_bits` of a `total_bits`-wide int."""
    return _crc24q((value >> (total_bits - 1 - i)) & 1 for i in range(data_bits))


def cnav_crc_ok(m):
    """Validate the CRC-24Q over a 300-bit CNAV message integer."""
    return _crc24q_head(m, 300, 300) == 0


def _cu(m, start, length):
    """Unsigned field, 1-indexed bit `start` from the MSB of the 300-bit message."""
    return (m >> (301 - start - length)) & ((1 << length) - 1)


def _cs(m, start, length):
    return _twos(_cu(m, start, length), length)


# ---------------------------------------------------------------------------
# Generic MSB-first bit-field extractors operating on a single integer that
# holds `total` bits (bit 0 = MSB).  Used by the Galileo / BeiDou / GLONASS
# decoders below.  _bu = unsigned, _bs = two's complement, _bg = sign-magnitude
# (GLONASS), _bu2/_bs2 = two non-adjacent fields concatenated (parity gaps).
# ---------------------------------------------------------------------------

def _bu(val, total, pos, length):
    return (val >> (total - pos - length)) & ((1 << length) - 1)


def _bs(val, total, pos, length):
    return _twos(_bu(val, total, pos, length), length)


def _bg(val, total, pos, length):
    mag = _bu(val, total, pos + 1, length - 1)
    return -mag if _bu(val, total, pos, 1) else mag


def _bu2(val, total, p1, l1, p2, l2):
    return (_bu(val, total, p1, l1) << l2) | _bu(val, total, p2, l2)


def _bs2(val, total, p1, l1, p2, l2):
    return (_bs(val, total, p1, l1) << l2) | _bu(val, total, p2, l2)


def _bu3(val, total, p1, l1, p2, l2, p3, l3):
    return (_bu2(val, total, p1, l1, p2, l2) << l3) | _bu(val, total, p3, l3)


def _bs3(val, total, p1, l1, p2, l2, p3, l3):
    return (_bs2(val, total, p1, l1, p2, l2) << l3) | _bu(val, total, p3, l3)


def _bds_u(val, total, spec):
    """Read an unsigned BeiDou field given a (pos,len) or (p1,l1,p2,l2) spec."""
    return _bu(val, total, *spec) if len(spec) == 2 else _bu2(val, total, *spec)


def _bds_s(val, total, spec):
    """Read a signed (two's complement) BeiDou field from a 2- or 4-tuple spec."""
    return _bs(val, total, *spec) if len(spec) == 2 else _bs2(val, total, *spec)


def _words_int(words):
    """Concatenate 32-bit words MSB-first into one integer."""
    v = 0
    for w in words:
        v = (v << 32) | (w & 0xFFFFFFFF)
    return v
