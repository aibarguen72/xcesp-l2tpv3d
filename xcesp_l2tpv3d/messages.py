"""L2TPv3 control-message wire encoding (RFC 3931 §4.1.1).

A control message is:

    +----------------+------------------+
    | 12-byte header | AVP list (0..N)  |
    +----------------+------------------+

The 12-byte header carries the T/L/S flag bits, protocol version (=3),
total length, Control Connection ID, and sequence numbers Ns/Nr.
An AVP-less message with only the header is a ZLB (Zero-Length Body)
per §5.8 — used to acknowledge received messages when the sender has
no data to piggyback the ack on.

Typed message builders (SCCRQ, SCCRP, ...) live on top of this class
and are added in 0.3.0.  This module deliberately stays wire-level so
the reliable-transport layer (transport.py) can reason about Ns/Nr
without knowing message semantics.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import List

from .avp import AVP, AVP_HEADER_LEN


#: Fixed control-message header length in bytes (RFC 3931 §4.1.1).
HEADER_LEN = 12

#: L2TP protocol version for L2TPv3.
VERSION = 3

#: Byte-0 flag mask for control messages: T=1 (control), L=1 (Length
#: present, mandatory for control), S=1 (Sequence present, mandatory
#: for control).  All other bits in byte 0 are reserved (must be 0).
_CTRL_FLAGS_BYTE0 = 0b11001000

#: Byte-1 mask carrying the 4-bit Version in the low nibble.  The
#: high nibble is reserved (must be 0).
_VERSION_BYTE1_MASK = 0x0F

#: Max control-message length per the 16-bit Length field.
MAX_MESSAGE_LEN = 0xFFFF


@dataclass
class ControlMessage:
    """A parsed or to-be-serialised L2TPv3 control message.

    Attributes match the wire header directly (control_connection_id,
    ns, nr) plus a list of AVPs that make up the message body.  Length
    is derived from the encoded size and not stored explicitly.
    """

    control_connection_id: int
    ns: int
    nr: int
    avps: List[AVP] = field(default_factory=list)

    # ---- properties -------------------------------------------------------

    @property
    def is_zlb(self) -> bool:
        """A ZLB (Zero-Length Body) has header only, no AVPs.

        ZLBs are ack-only messages: the receiver sends one to
        acknowledge received messages when it has nothing else to
        piggyback the ack on.  Per §5.8, a ZLB itself is NOT
        acknowledged — sending a ZLB doesn't consume an Ns.
        """
        return not self.avps

    # ---- encoding ---------------------------------------------------------

    def encode(
        self,
        *,
        shared_secret: bytes | None = None,
        random_vector: bytes | None = None,
    ) -> bytes:
        """Serialise the message to wire bytes.

        Hidden-AVP encoding parameters are forwarded to every AVP in
        the body; individual AVPs choose to hide themselves based on
        their own ``hidden`` flag.  Callers with any hidden AVPs must
        include a Random Vector AVP earlier in the ``avps`` list —
        that's the AVP whose value is used as the IV.
        """
        self._validate_fields()

        body = b"".join(
            avp.encode(shared_secret=shared_secret, random_vector=random_vector)
            for avp in self.avps
        )
        total_len = HEADER_LEN + len(body)
        if total_len > MAX_MESSAGE_LEN:
            raise ValueError(
                f"control message too long: {total_len} > {MAX_MESSAGE_LEN}"
            )

        header = struct.pack(
            "!BBHIHH",
            _CTRL_FLAGS_BYTE0,
            VERSION & _VERSION_BYTE1_MASK,
            total_len,
            self.control_connection_id,
            self.ns,
            self.nr,
        )
        return header + body

    # ---- decoding ---------------------------------------------------------

    @classmethod
    def decode(
        cls,
        buf: bytes,
        *,
        shared_secret: bytes | None = None,
    ) -> "ControlMessage":
        """Decode a control message from wire bytes.

        Validates the T/L/S flag bits, reserved bits, and version.
        Rejects data messages (T=0) — this decoder is control-only.
        """
        if len(buf) < HEADER_LEN:
            raise ValueError(
                f"buffer too short for control-message header: {len(buf)} < "
                f"{HEADER_LEN}"
            )

        b0, b1, total_len, ccid, ns, nr = struct.unpack_from(
            "!BBHIHH", buf, 0
        )

        # Enforce the fixed control-message flag byte in one comparison so
        # any reserved-bit set or T/L/S off is caught with an actionable msg.
        if b0 != _CTRL_FLAGS_BYTE0:
            # Pinpoint the specific problem for the caller.
            if not (b0 & 0x80):
                raise ValueError(
                    "T bit clear — data messages are handled elsewhere"
                )
            if not (b0 & 0x40):
                raise ValueError("L bit clear — control messages must have Length")
            if not (b0 & 0x08):
                raise ValueError(
                    "S bit clear — control messages must have Sequence"
                )
            raise ValueError(
                f"reserved bit set in flag byte 0 (got 0x{b0:02x}, "
                f"expected 0x{_CTRL_FLAGS_BYTE0:02x})"
            )
        if b1 & 0xF0:
            raise ValueError(
                f"reserved high nibble non-zero in byte 1 (got 0x{b1:02x})"
            )
        version = b1 & _VERSION_BYTE1_MASK
        if version != VERSION:
            raise ValueError(f"unsupported L2TP version {version}, expected {VERSION}")
        if total_len != len(buf):
            raise ValueError(
                f"header Length={total_len} does not match buffer size {len(buf)}"
            )

        avps = AVP.decode_sequence(
            buf, HEADER_LEN, shared_secret=shared_secret
        )
        return cls(
            control_connection_id=ccid,
            ns=ns,
            nr=nr,
            avps=avps,
        )

    # ---- internals --------------------------------------------------------

    def _validate_fields(self) -> None:
        if self.control_connection_id < 0 or self.control_connection_id > 0xFFFFFFFF:
            raise ValueError(
                f"control_connection_id out of range: {self.control_connection_id}"
            )
        if self.ns < 0 or self.ns > 0xFFFF:
            raise ValueError(f"ns out of range: {self.ns}")
        if self.nr < 0 or self.nr > 0xFFFF:
            raise ValueError(f"nr out of range: {self.nr}")


# ---------------------------------------------------------------------------
# Sequence-number arithmetic (RFC 3931 §5.8)
# ---------------------------------------------------------------------------
#
# L2TP Ns/Nr are 16-bit and wrap at 65535.  Comparisons use signed
# 16-bit arithmetic on the difference: values in (0, 32767] mean "a
# is ahead of b", values in [-32768, 0) mean "a is behind b".
#
# This is subtle enough that every user of it goes through the two
# helpers below.

def seq_delta(a: int, b: int) -> int:
    """Return signed 16-bit distance from ``b`` to ``a``, per RFC 3931 §5.8.

    Positive  → a is ahead of b (higher sequence).
    Zero      → a == b.
    Negative  → a is behind b (lower sequence).
    """
    diff = (a - b) & 0xFFFF
    if diff & 0x8000:
        diff -= 0x10000
    return diff


def seq_advance(seq: int, by: int = 1) -> int:
    """Add ``by`` to a 16-bit sequence number, wrapping at 65535."""
    return (seq + by) & 0xFFFF
