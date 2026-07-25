"""Unit tests for xcesp_l2tpv3d.messages — ControlMessage + seq helpers."""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from xcesp_l2tpv3d.avp import (   # noqa: E402
    AttrType, MessageType, build_message_type, build_router_id,
)
from xcesp_l2tpv3d.messages import (  # noqa: E402
    HEADER_LEN, VERSION, ControlMessage, seq_advance, seq_delta,
)


# ---------------------------------------------------------------------------
# ControlMessage encode / decode round-trip
# ---------------------------------------------------------------------------

def test_zlb_roundtrip_header_only():
    zlb = ControlMessage(control_connection_id=42, ns=3, nr=7)
    wire = zlb.encode()
    assert len(wire) == HEADER_LEN
    decoded = ControlMessage.decode(wire)
    assert decoded.control_connection_id == 42
    assert decoded.ns == 3
    assert decoded.nr == 7
    assert decoded.is_zlb


def test_message_with_avps_roundtrip():
    msg = ControlMessage(
        control_connection_id=0xDEADBEEF,
        ns=100,
        nr=200,
        avps=[
            build_message_type(MessageType.SCCRQ),
            build_router_id(0x0A0A0A01),
        ],
    )
    wire = msg.encode()
    decoded = ControlMessage.decode(wire)
    assert decoded.control_connection_id == 0xDEADBEEF
    assert decoded.ns == 100
    assert decoded.nr == 200
    assert len(decoded.avps) == 2
    assert decoded.avps[0].attribute_type == int(AttrType.MESSAGE_TYPE)
    assert decoded.avps[1].attribute_type == int(AttrType.ROUTER_ID)
    assert not decoded.is_zlb


def test_header_flag_byte_is_C8():
    # T=1, L=1, S=1, all other bits 0 → 0xC8
    msg = ControlMessage(control_connection_id=1, ns=0, nr=0)
    wire = msg.encode()
    assert wire[0] == 0xC8


def test_header_version_is_3():
    msg = ControlMessage(control_connection_id=1, ns=0, nr=0)
    wire = msg.encode()
    assert (wire[1] & 0x0F) == VERSION


def test_header_length_matches_actual():
    msg = ControlMessage(
        control_connection_id=1, ns=0, nr=0,
        avps=[build_message_type(MessageType.HELLO)],
    )
    wire = msg.encode()
    length_from_header = struct.unpack_from("!H", wire, 2)[0]
    assert length_from_header == len(wire)


# ---------------------------------------------------------------------------
# Malformed input rejection
# ---------------------------------------------------------------------------

def test_short_buffer_rejected():
    with pytest.raises(ValueError, match="buffer too short"):
        ControlMessage.decode(b"\x00" * (HEADER_LEN - 1))


def test_data_message_t_bit_clear_rejected():
    # Craft a header with T=0
    wire = bytearray(HEADER_LEN)
    wire[0] = 0x48   # L=1, S=1, T=0
    wire[1] = 0x03
    struct.pack_into("!H", wire, 2, HEADER_LEN)
    with pytest.raises(ValueError, match="T bit clear"):
        ControlMessage.decode(bytes(wire))


def test_l_bit_clear_rejected():
    # T=1, L=0, S=1
    wire = bytearray(HEADER_LEN)
    wire[0] = 0x88   # T=1, L=0, S=1
    wire[1] = 0x03
    struct.pack_into("!H", wire, 2, HEADER_LEN)
    with pytest.raises(ValueError, match="L bit clear"):
        ControlMessage.decode(bytes(wire))


def test_s_bit_clear_rejected():
    # T=1, L=1, S=0
    wire = bytearray(HEADER_LEN)
    wire[0] = 0xC0   # T=1, L=1, S=0
    wire[1] = 0x03
    struct.pack_into("!H", wire, 2, HEADER_LEN)
    with pytest.raises(ValueError, match="S bit clear"):
        ControlMessage.decode(bytes(wire))


def test_reserved_bit_in_byte0_rejected():
    # Set bit-2 of byte 0 (a reserved bit)
    wire = bytearray(HEADER_LEN)
    wire[0] = 0xCC   # T=1, L=1, x=1, x=0, S=1, ...
    wire[1] = 0x03
    struct.pack_into("!H", wire, 2, HEADER_LEN)
    with pytest.raises(ValueError, match="reserved bit set"):
        ControlMessage.decode(bytes(wire))


def test_reserved_high_nibble_in_byte1_rejected():
    wire = bytearray(HEADER_LEN)
    wire[0] = 0xC8
    wire[1] = 0x13   # high nibble non-zero
    struct.pack_into("!H", wire, 2, HEADER_LEN)
    with pytest.raises(ValueError, match="reserved high nibble"):
        ControlMessage.decode(bytes(wire))


def test_wrong_version_rejected():
    wire = bytearray(HEADER_LEN)
    wire[0] = 0xC8
    wire[1] = 0x02   # version=2
    struct.pack_into("!H", wire, 2, HEADER_LEN)
    with pytest.raises(ValueError, match="unsupported L2TP version 2"):
        ControlMessage.decode(bytes(wire))


def test_length_mismatch_rejected():
    # Header says length is HEADER_LEN, but we hand a buffer with an
    # extra byte — decode should catch the mismatch.
    msg = ControlMessage(control_connection_id=1, ns=0, nr=0)
    wire = msg.encode() + b"\x00"
    with pytest.raises(ValueError, match="does not match buffer size"):
        ControlMessage.decode(wire)


def test_ccid_out_of_range_rejected_on_encode():
    with pytest.raises(ValueError, match="control_connection_id"):
        ControlMessage(control_connection_id=0x1_00000000, ns=0, nr=0).encode()


def test_ns_out_of_range_rejected_on_encode():
    with pytest.raises(ValueError, match="ns"):
        ControlMessage(control_connection_id=1, ns=0x10000, nr=0).encode()


def test_nr_out_of_range_rejected_on_encode():
    with pytest.raises(ValueError, match="nr"):
        ControlMessage(control_connection_id=1, ns=0, nr=0x10000).encode()


# ---------------------------------------------------------------------------
# Sequence-number arithmetic
# ---------------------------------------------------------------------------

def test_seq_delta_zero():
    assert seq_delta(100, 100) == 0
    assert seq_delta(0, 0) == 0
    assert seq_delta(0xFFFF, 0xFFFF) == 0


def test_seq_delta_positive_no_wrap():
    assert seq_delta(200, 100) == 100
    assert seq_delta(1, 0) == 1


def test_seq_delta_negative_no_wrap():
    assert seq_delta(100, 200) == -100
    assert seq_delta(0, 1) == -1


def test_seq_delta_positive_across_wrap():
    # 0 is one ahead of 65535
    assert seq_delta(0, 0xFFFF) == 1
    # 5 is 6 ahead of 65535
    assert seq_delta(5, 0xFFFF) == 6


def test_seq_delta_negative_across_wrap():
    # 65535 is 1 behind 0
    assert seq_delta(0xFFFF, 0) == -1
    # 65530 is 6 behind 0
    assert seq_delta(0xFFF0 - 5, 0) == -21


def test_seq_delta_at_half_window():
    # Boundary at 32768 (0x8000).
    # 32768 is 32768 ahead of 0 (still positive).
    # Actually: (32768 - 0) & 0xFFFF = 32768; MSB set → subtract 0x10000
    # → -32768.  So exactly at the boundary it's considered behind.
    # That matches the RFC 3931 §5.8 half-window recommendation.
    assert seq_delta(0x8000, 0) == -0x8000
    assert seq_delta(0x7FFF, 0) == 0x7FFF


def test_seq_advance_default_by_one():
    assert seq_advance(0) == 1
    assert seq_advance(100) == 101


def test_seq_advance_wraps_at_65535():
    assert seq_advance(0xFFFF) == 0
    assert seq_advance(0xFFFE, by=3) == 1


def test_seq_advance_can_go_by_arbitrary_amount():
    assert seq_advance(1000, by=500) == 1500
