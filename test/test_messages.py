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
    AttrType, DigestHash, MessageType, PseudowireType,
    build_message_type, build_router_id,
)
from xcesp_l2tpv3d.messages import (  # noqa: E402
    HEADER_LEN, VERSION, ControlMessage,
    build_hello_avps, build_sccrp_avps, build_sccrq_avps,
    build_scccn_avps, build_stopccn_avps,
    get_message_type, parse_sccrx_fields, parse_stopccn_fields,
    seq_advance, seq_delta,
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


# ---------------------------------------------------------------------------
# Typed message builders (0.3.0)
# ---------------------------------------------------------------------------

def _sccrq_kwargs(**overrides):
    base = dict(
        router_id=0x0A0A0A01,
        assigned_ccid=0x1234,
        host_name="xcespfc",
        pw_capabilities=[int(PseudowireType.ETHERNET),
                          int(PseudowireType.ETHERNET_VLAN)],
        receive_window=4,
    )
    base.update(overrides)
    return base


def test_sccrq_avps_contains_all_mandatory():
    avps = build_sccrq_avps(**_sccrq_kwargs())
    types_in_message = {a.attribute_type for a in avps}
    assert int(AttrType.MESSAGE_TYPE) in types_in_message
    assert int(AttrType.ROUTER_ID) in types_in_message
    assert int(AttrType.ASSIGNED_CONTROL_CONNECTION_ID) in types_in_message
    assert int(AttrType.PSEUDOWIRE_CAPABILITIES_LIST) in types_in_message
    assert int(AttrType.HOST_NAME) in types_in_message
    assert int(AttrType.RECEIVE_WINDOW_SIZE) in types_in_message
    assert get_message_type(avps) == int(MessageType.SCCRQ)


def test_sccrq_avps_includes_optional_when_provided():
    avps = build_sccrq_avps(**_sccrq_kwargs(
        vendor_name="XCESP",
        firmware_revision=42,
        auth_nonce=b"\x01" * 32,
    ))
    types = {a.attribute_type for a in avps}
    assert int(AttrType.VENDOR_NAME) in types
    assert int(AttrType.FIRMWARE_REVISION) in types
    assert int(AttrType.CTL_MSG_AUTH_NONCE) in types


def test_sccrq_avps_omits_optional_when_absent():
    avps = build_sccrq_avps(**_sccrq_kwargs())
    types = {a.attribute_type for a in avps}
    assert int(AttrType.VENDOR_NAME) not in types
    assert int(AttrType.FIRMWARE_REVISION) not in types
    assert int(AttrType.CTL_MSG_AUTH_NONCE) not in types


def test_sccrp_mirrors_sccrq_with_different_message_type():
    kwargs = _sccrq_kwargs()
    sccrq = build_sccrq_avps(**kwargs)
    sccrp = build_sccrp_avps(**kwargs)
    assert get_message_type(sccrq) == int(MessageType.SCCRQ)
    assert get_message_type(sccrp) == int(MessageType.SCCRP)
    # Everything else identical
    sccrq_no_mt = [a for a in sccrq if a.attribute_type != int(AttrType.MESSAGE_TYPE)]
    sccrp_no_mt = [a for a in sccrp if a.attribute_type != int(AttrType.MESSAGE_TYPE)]
    assert [(a.attribute_type, a.value) for a in sccrq_no_mt] == \
           [(a.attribute_type, a.value) for a in sccrp_no_mt]


def test_scccn_is_message_type_only():
    avps = build_scccn_avps()
    assert len(avps) == 1
    assert get_message_type(avps) == int(MessageType.SCCCN)


def test_hello_is_message_type_only():
    avps = build_hello_avps()
    assert len(avps) == 1
    assert get_message_type(avps) == int(MessageType.HELLO)


def test_stopccn_carries_result_code_and_ccid():
    avps = build_stopccn_avps(assigned_ccid=0xABCD, result_code=1)
    assert get_message_type(avps) == int(MessageType.StopCCN)
    types = {a.attribute_type for a in avps}
    assert int(AttrType.ASSIGNED_CONTROL_CONNECTION_ID) in types
    assert int(AttrType.RESULT_CODE) in types


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def test_parse_sccrx_extracts_all_mandatory_fields():
    avps = build_sccrq_avps(**_sccrq_kwargs(
        router_id=0x0A0A0A02, assigned_ccid=42, host_name="peer",
        pw_capabilities=[int(PseudowireType.ETHERNET)],
        receive_window=8,
    ))
    fields = parse_sccrx_fields(avps)
    assert fields.router_id == 0x0A0A0A02
    assert fields.assigned_ccid == 42
    assert fields.host_name == b"peer"
    assert fields.pw_capabilities == [int(PseudowireType.ETHERNET)]
    assert fields.receive_window == 8
    assert fields.auth_nonce is None


def test_parse_sccrx_extracts_optional_nonce():
    nonce = b"\xa5" * 32
    avps = build_sccrq_avps(**_sccrq_kwargs(auth_nonce=nonce))
    fields = parse_sccrx_fields(avps)
    assert fields.auth_nonce == nonce


def test_parse_sccrx_rejects_missing_mandatory():
    # Build SCCRQ, then drop the Router ID AVP.
    avps = [a for a in build_sccrq_avps(**_sccrq_kwargs())
            if a.attribute_type != int(AttrType.ROUTER_ID)]
    with pytest.raises(ValueError, match="Router ID"):
        parse_sccrx_fields(avps)


def test_parse_stopccn_extracts_fields():
    avps = build_stopccn_avps(
        assigned_ccid=99, result_code=1, error_code=2,
        error_message="teardown",
    )
    fields = parse_stopccn_fields(avps)
    assert fields.assigned_ccid == 99
    assert fields.result_code == 1
    assert fields.error_code == 2
    assert fields.error_message == b"teardown"


def test_parse_stopccn_rejects_missing_result_code():
    avps = [a for a in build_stopccn_avps(assigned_ccid=1, result_code=0)
            if a.attribute_type != int(AttrType.RESULT_CODE)]
    with pytest.raises(ValueError, match="Result Code"):
        parse_stopccn_fields(avps)


# ---------------------------------------------------------------------------
# ControlMessage.encode_signed / decode_and_verify (0.3.0)
# ---------------------------------------------------------------------------

_SIGNING_SECRET = b"unit-test-shared-secret"


def test_encode_signed_appends_message_digest_avp():
    avps = build_hello_avps()
    msg = ControlMessage(control_connection_id=1, ns=0, nr=0, avps=avps)
    wire = msg.encode_signed(DigestHash.HMAC_MD5, _SIGNING_SECRET)
    # Wire should be longer than the un-signed encoding by exactly
    # the size of a Message Digest AVP (header + hash_type + digest).
    unsigned = msg.encode()
    md_avp_size = 6 + 1 + 16   # AVP header + hash_type + md5 digest
    assert len(wire) == len(unsigned) + md_avp_size

    # And the wire should now be decodable, with the digest AVP present.
    decoded = ControlMessage.decode(wire)
    types = {a.attribute_type for a in decoded.avps}
    assert int(AttrType.MESSAGE_DIGEST) in types


@pytest.mark.parametrize("hash_type", [DigestHash.HMAC_MD5, DigestHash.HMAC_SHA1])
def test_encode_signed_and_decode_and_verify_roundtrip(hash_type):
    avps = build_sccrq_avps(**_sccrq_kwargs())
    msg = ControlMessage(control_connection_id=1, ns=5, nr=3, avps=avps)
    wire = msg.encode_signed(hash_type, _SIGNING_SECRET)

    verified = ControlMessage.decode_and_verify(
        wire, hash_type, _SIGNING_SECRET
    )
    assert verified.control_connection_id == 1
    assert verified.ns == 5
    assert verified.nr == 3
    # The original AVPs are preserved (plus the digest AVP appended).
    original_types = [a.attribute_type for a in msg.avps]
    verified_types = [a.attribute_type for a in verified.avps]
    assert verified_types[:len(original_types)] == original_types


def test_decode_and_verify_rejects_tampered_body():
    avps = build_hello_avps()
    msg = ControlMessage(control_connection_id=1, ns=0, nr=0, avps=avps)
    wire = bytearray(msg.encode_signed(DigestHash.HMAC_MD5, _SIGNING_SECRET))
    # Flip the LOW byte of the Message Type AVP's value.  The Message
    # Type AVP is 8 bytes total (6-byte header + 2-byte value); its
    # value byte 1 sits at offset HEADER_LEN+7 in the wire buffer.
    # Flipping this changes the message meaning but keeps the AVP
    # length field intact, so the decode succeeds and the digest
    # check is what should reject.
    wire[HEADER_LEN + 7] ^= 0x01
    with pytest.raises(ValueError, match="verification failed"):
        ControlMessage.decode_and_verify(
            bytes(wire), DigestHash.HMAC_MD5, _SIGNING_SECRET
        )


def test_decode_and_verify_rejects_wrong_secret():
    avps = build_hello_avps()
    msg = ControlMessage(control_connection_id=1, ns=0, nr=0, avps=avps)
    wire = msg.encode_signed(DigestHash.HMAC_MD5, _SIGNING_SECRET)
    with pytest.raises(ValueError, match="verification failed"):
        ControlMessage.decode_and_verify(
            wire, DigestHash.HMAC_MD5, b"WRONG"
        )


def test_decode_and_verify_rejects_missing_digest_avp():
    # Encode without signing, then try to verify.
    unsigned = ControlMessage(
        control_connection_id=1, ns=0, nr=0, avps=build_hello_avps()
    ).encode()
    with pytest.raises(ValueError, match="missing Message Digest AVP"):
        ControlMessage.decode_and_verify(
            unsigned, DigestHash.HMAC_MD5, _SIGNING_SECRET
        )


def test_decode_and_verify_rejects_wrong_hash_type():
    avps = build_hello_avps()
    msg = ControlMessage(control_connection_id=1, ns=0, nr=0, avps=avps)
    wire = msg.encode_signed(DigestHash.HMAC_MD5, _SIGNING_SECRET)
    with pytest.raises(ValueError, match="hash type mismatch"):
        ControlMessage.decode_and_verify(
            wire, DigestHash.HMAC_SHA1, _SIGNING_SECRET
        )
