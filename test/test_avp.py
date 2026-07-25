"""Unit tests for xcesp_l2tpv3d.avp — 0.1.0 milestone verification.

Coverage:
  - AVP header round-trip: M/H bits, mandatory-only, hidden-only, both.
  - Length boundaries: empty value, 1 byte, 256 bytes, max value length.
  - Per-type value codecs (Message Type, Result Code, Router ID, Assigned
    CCID, PW Capabilities List, Local/Remote Session ID, Assigned Cookie,
    Remote End ID, Pseudowire Type, L2-Specific Sublayer, Data Sequencing,
    Circuit Status, Preferred Language, Tx/Rx Connect Speed, Message
    Digest, Host Name).
  - Hidden-AVP path (§4.3): round-trip with a shared-secret fixture +
    fixed Random Vector.
  - Malformed input rejection: short header, over-long length,
    non-zero reserved bits, hidden AVP without secret/RV, hidden AVP
    with corrupt ciphertext length, PW capabilities with odd byte count.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest

# Allow tests to run without pip install (dev-loop): add the package
# parent to sys.path if needed.
_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from xcesp_l2tpv3d.avp import (  # noqa: E402
    AVP,
    AVP_HEADER_LEN,
    AVP_MAX_VALUE_LEN,
    AttrType,
    DigestHash,
    MessageType,
    PseudowireType,
    VENDOR_IETF,
    build_assigned_control_connection_id,
    build_assigned_cookie,
    build_circuit_status,
    build_data_sequencing,
    build_host_name,
    build_l2_specific_sublayer,
    build_local_session_id,
    build_message_digest,
    build_message_type,
    build_preferred_language,
    build_pseudowire_capabilities_list,
    build_pseudowire_type,
    build_random_vector,
    build_remote_end_id,
    build_remote_session_id,
    build_result_code,
    build_router_id,
    build_rx_connect_speed,
    build_tx_connect_speed,
    decode_message_type,
    decode_pseudowire_capabilities_list,
    decode_result_code,
    decode_router_id,
)


# ---------------------------------------------------------------------------
# Header round-trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mandatory", [True, False])
@pytest.mark.parametrize("hidden", [False])   # hidden covered separately
def test_header_roundtrip_all_bit_combos(mandatory, hidden):
    original = AVP(
        attribute_type=AttrType.MESSAGE_TYPE,
        value=struct.pack("!H", 1),
        mandatory=mandatory,
        hidden=hidden,
    )
    wire = original.encode()
    decoded, offset = AVP.decode_one(wire, 0)
    assert offset == len(wire)
    assert decoded.mandatory is mandatory
    assert decoded.hidden is hidden
    assert decoded.attribute_type == int(AttrType.MESSAGE_TYPE)
    assert decoded.vendor_id == VENDOR_IETF
    assert decoded.value == original.value


def test_header_encodes_length_correctly():
    # Length field must be total AVP length (header + value).
    value = b"\x01\x02\x03\x04\x05\x06\x07\x08"
    a = AVP(attribute_type=AttrType.ROUTER_ID, value=value)
    wire = a.encode()
    assert len(wire) == AVP_HEADER_LEN + len(value)
    b0, b1 = wire[0], wire[1]
    length_from_header = ((b0 & 0x03) << 8) | b1
    assert length_from_header == len(wire)


def test_vendor_id_out_of_range_rejected():
    with pytest.raises(ValueError, match="vendor_id"):
        AVP(attribute_type=1, value=b"", vendor_id=0x10000).encode()


def test_attribute_type_out_of_range_rejected():
    with pytest.raises(ValueError, match="attribute_type"):
        AVP(attribute_type=0x10000, value=b"").encode()


# ---------------------------------------------------------------------------
# Length boundaries
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value_len",
    [0, 1, 15, 16, 17, 100, 255, 256, 512, AVP_MAX_VALUE_LEN - 1, AVP_MAX_VALUE_LEN],
)
def test_boundary_lengths_roundtrip(value_len):
    payload = bytes(range(256)) * ((value_len // 256) + 1)
    payload = payload[:value_len]
    a = AVP(attribute_type=AttrType.REMOTE_END_ID, value=payload)
    wire = a.encode()
    b, _ = AVP.decode_one(wire, 0)
    assert b.value == payload
    assert len(b.value) == value_len


def test_value_too_long_rejected():
    payload = b"\x00" * (AVP_MAX_VALUE_LEN + 1)
    with pytest.raises(ValueError, match="too long"):
        AVP(attribute_type=AttrType.REMOTE_END_ID, value=payload).encode()


# ---------------------------------------------------------------------------
# Per-type value codecs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "mt",
    [MessageType.SCCRQ, MessageType.SCCRP, MessageType.SCCCN,
     MessageType.StopCCN, MessageType.HELLO, MessageType.ICRQ,
     MessageType.ICRP, MessageType.ICCN, MessageType.CDN,
     MessageType.WEN, MessageType.SLI],
)
def test_message_type_roundtrip(mt):
    a = build_message_type(mt)
    assert a.mandatory
    d, _ = AVP.decode_one(a.encode())
    assert decode_message_type(d) == int(mt)


def test_result_code_result_only():
    a = build_result_code(2)
    d, _ = AVP.decode_one(a.encode())
    r, e, msg = decode_result_code(d)
    assert (r, e, msg) == (2, None, b"")


def test_result_code_with_error_and_message():
    a = build_result_code(2, error_code=6, error_message="unable to allocate tunnel")
    d, _ = AVP.decode_one(a.encode())
    r, e, msg = decode_result_code(d)
    assert r == 2
    assert e == 6
    assert msg == b"unable to allocate tunnel"


def test_result_code_error_message_without_code_rejected():
    with pytest.raises(ValueError, match="error_message requires error_code"):
        build_result_code(2, error_message=b"nope")


def test_router_id_roundtrip():
    # Router ID is arbitrary uint32; often a dotted-quad in host order.
    a = build_router_id(0x0A0A0A01)  # 10.10.10.1
    d, _ = AVP.decode_one(a.encode())
    assert decode_router_id(d) == 0x0A0A0A01


def test_router_id_out_of_range():
    with pytest.raises(ValueError, match="router_id"):
        build_router_id(-1)
    with pytest.raises(ValueError, match="router_id"):
        build_router_id(0x1_00000000)


def test_assigned_control_connection_id_roundtrip():
    a = build_assigned_control_connection_id(0xDEADBEEF)
    d, _ = AVP.decode_one(a.encode())
    (val,) = struct.unpack("!I", d.value)
    assert val == 0xDEADBEEF


def test_pseudowire_capabilities_list_roundtrip():
    caps = [int(PseudowireType.ETHERNET), int(PseudowireType.ETHERNET_VLAN)]
    a = build_pseudowire_capabilities_list(caps)
    d, _ = AVP.decode_one(a.encode())
    assert decode_pseudowire_capabilities_list(d) == caps


def test_pseudowire_capabilities_list_empty_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        build_pseudowire_capabilities_list([])


def test_pseudowire_capabilities_list_odd_length_rejected():
    # Craft an AVP with a value that's 3 bytes (odd) — should decode-fail.
    bad = AVP(
        attribute_type=AttrType.PSEUDOWIRE_CAPABILITIES_LIST,
        value=b"\x00\x05\xff",
    )
    d, _ = AVP.decode_one(bad.encode())
    with pytest.raises(ValueError, match="multiple of 2"):
        decode_pseudowire_capabilities_list(d)


def test_local_and_remote_session_id_roundtrip():
    la = build_local_session_id(123)
    ra = build_remote_session_id(456)
    ld, _ = AVP.decode_one(la.encode())
    rd, _ = AVP.decode_one(ra.encode())
    assert struct.unpack("!I", ld.value)[0] == 123
    assert struct.unpack("!I", rd.value)[0] == 456


@pytest.mark.parametrize("cookie", [b"", b"\x01\x02\x03\x04",
                                     b"\x01\x02\x03\x04\x05\x06\x07\x08"])
def test_assigned_cookie_valid_lengths(cookie):
    a = build_assigned_cookie(cookie)
    d, _ = AVP.decode_one(a.encode())
    assert d.value == cookie


@pytest.mark.parametrize("cookie", [b"\x01", b"\x01\x02\x03",
                                     b"\x01\x02\x03\x04\x05"])
def test_assigned_cookie_invalid_lengths(cookie):
    with pytest.raises(ValueError, match="0/4/8"):
        build_assigned_cookie(cookie)


def test_remote_end_id_roundtrip():
    a = build_remote_end_id("100")
    d, _ = AVP.decode_one(a.encode())
    assert d.value == b"100"


def test_pseudowire_type_roundtrip():
    a = build_pseudowire_type(PseudowireType.ETHERNET_VLAN)
    d, _ = AVP.decode_one(a.encode())
    (pt,) = struct.unpack("!H", d.value)
    assert pt == int(PseudowireType.ETHERNET_VLAN)


def test_l2_specific_sublayer_roundtrip():
    a = build_l2_specific_sublayer(1)   # default
    d, _ = AVP.decode_one(a.encode())
    (v,) = struct.unpack("!H", d.value)
    assert v == 1


def test_data_sequencing_roundtrip():
    a = build_data_sequencing(2)
    d, _ = AVP.decode_one(a.encode())
    (v,) = struct.unpack("!H", d.value)
    assert v == 2


def test_circuit_status_roundtrip():
    # bit 0 = active, bit 1 = new
    a = build_circuit_status(0b11)
    d, _ = AVP.decode_one(a.encode())
    (v,) = struct.unpack("!H", d.value)
    assert v == 0b11


def test_preferred_language_roundtrip():
    a = build_preferred_language("en-US")
    d, _ = AVP.decode_one(a.encode())
    assert d.value == b"en-US"
    assert not d.mandatory     # explicitly optional per RFC


def test_tx_and_rx_connect_speed_roundtrip():
    tx = build_tx_connect_speed(10_000_000_000)   # 10 Gbps
    rx = build_rx_connect_speed(1_000_000_000)    # 1 Gbps
    td, _ = AVP.decode_one(tx.encode())
    rd, _ = AVP.decode_one(rx.encode())
    (t,) = struct.unpack("!Q", td.value)
    (r,) = struct.unpack("!Q", rd.value)
    assert t == 10_000_000_000
    assert r == 1_000_000_000


def test_message_digest_roundtrip():
    digest = b"\xaa" * 16   # dummy 16-byte MD5-like digest
    a = build_message_digest(DigestHash.HMAC_MD5, digest)
    d, _ = AVP.decode_one(a.encode())
    assert d.value[0] == int(DigestHash.HMAC_MD5)
    assert d.value[1:] == digest


def test_host_name_roundtrip():
    a = build_host_name("xcespfc")
    d, _ = AVP.decode_one(a.encode())
    assert d.value == b"xcespfc"


# ---------------------------------------------------------------------------
# Hidden-AVP path (RFC 3931 §4.3)
# ---------------------------------------------------------------------------

_SHARED_SECRET = b"this-is-a-shared-secret"
_RANDOM_VECTOR = b"\x00\x11\x22\x33\x44\x55\x66\x77\x88\x99\xAA\xBB\xCC\xDD\xEE\xFF"


def test_hidden_avp_roundtrip_short_value():
    plain = b"secret-payload"
    a = AVP(
        attribute_type=AttrType.CHALLENGE,
        value=plain,
        hidden=True,
    )
    wire = a.encode(
        shared_secret=_SHARED_SECRET,
        random_vector=_RANDOM_VECTOR,
    )
    # Wire value field must be an integer number of MD5 blocks.
    assert (len(wire) - AVP_HEADER_LEN) % 16 == 0
    # Ciphertext must not equal the plaintext prefix.
    assert wire[AVP_HEADER_LEN:] != plain.ljust(len(wire) - AVP_HEADER_LEN, b"\x00")
    d, _ = AVP.decode_one(
        wire,
        shared_secret=_SHARED_SECRET,
        random_vector=_RANDOM_VECTOR,
    )
    assert d.hidden is True
    assert d.value == plain


def test_hidden_avp_roundtrip_long_value_multi_block():
    plain = bytes(range(48))   # 48 bytes → spans 4 blocks after 2-byte prefix
    a = AVP(
        attribute_type=AttrType.CHALLENGE_RESPONSE,
        value=plain,
        hidden=True,
    )
    wire = a.encode(
        shared_secret=_SHARED_SECRET,
        random_vector=_RANDOM_VECTOR,
    )
    d, _ = AVP.decode_one(
        wire,
        shared_secret=_SHARED_SECRET,
        random_vector=_RANDOM_VECTOR,
    )
    assert d.value == plain


def test_hidden_avp_missing_secret_encode_error():
    a = AVP(attribute_type=AttrType.CHALLENGE, value=b"x", hidden=True)
    with pytest.raises(ValueError, match="requires shared_secret"):
        a.encode()


def test_hidden_avp_missing_rv_encode_error():
    a = AVP(attribute_type=AttrType.CHALLENGE, value=b"x", hidden=True)
    with pytest.raises(ValueError, match="requires shared_secret"):
        a.encode(shared_secret=_SHARED_SECRET)


def test_hidden_avp_missing_secret_decode_error():
    a = AVP(attribute_type=AttrType.CHALLENGE, value=b"x", hidden=True)
    wire = a.encode(shared_secret=_SHARED_SECRET, random_vector=_RANDOM_VECTOR)
    with pytest.raises(ValueError, match="hidden AVP encountered"):
        AVP.decode_one(wire, 0)


def test_hidden_avp_sequence_uses_rv_from_earlier_avp():
    """Sequence decoder must pick up the RV from the RV AVP encountered
    earlier in the same sequence and apply it to subsequent hidden AVPs."""
    rv_avp = build_random_vector(_RANDOM_VECTOR)
    hidden = AVP(
        attribute_type=AttrType.CHALLENGE,
        value=b"top-secret",
        hidden=True,
    )
    wire = rv_avp.encode() + hidden.encode(
        shared_secret=_SHARED_SECRET,
        random_vector=_RANDOM_VECTOR,
    )
    avps = AVP.decode_sequence(wire, shared_secret=_SHARED_SECRET)
    assert len(avps) == 2
    assert avps[0].attribute_type == int(AttrType.RANDOM_VECTOR)
    assert avps[0].value == _RANDOM_VECTOR
    assert avps[1].hidden is True
    assert avps[1].value == b"top-secret"


def test_hidden_avp_corrupt_length_prefix_rejected():
    """If someone tampers with the ciphertext, the recovered length prefix
    may claim more bytes than exist — decode must reject."""
    a = AVP(attribute_type=AttrType.CHALLENGE, value=b"abc", hidden=True)
    wire = bytearray(
        a.encode(shared_secret=_SHARED_SECRET, random_vector=_RANDOM_VECTOR)
    )
    # Flip the FIRST ciphertext byte (which XORs against the high byte of
    # the length prefix, so a single bit-flip inflates the claimed length
    # past the ciphertext body).
    wire[AVP_HEADER_LEN] ^= 0x80
    with pytest.raises(ValueError, match="exceeds decrypted body"):
        AVP.decode_one(
            bytes(wire),
            shared_secret=_SHARED_SECRET,
            random_vector=_RANDOM_VECTOR,
        )


# ---------------------------------------------------------------------------
# Malformed input rejection
# ---------------------------------------------------------------------------

def test_short_buffer_rejected():
    with pytest.raises(ValueError, match="buffer too short"):
        AVP.decode_one(b"\x00\x06\x00\x00\x00", 0)   # 5 bytes < 6


def test_length_extends_past_buffer_rejected():
    # Encode a valid AVP, then chop the last byte.
    a = AVP(attribute_type=AttrType.MESSAGE_TYPE, value=b"\x00\x01")
    wire = a.encode()[:-1]
    with pytest.raises(ValueError, match="extends past buffer"):
        AVP.decode_one(wire, 0)


def test_length_smaller_than_header_rejected():
    # Craft raw header with length=5 (< 6).
    # Byte layout: M=1,H=0,rsvd=0,length=5 → b0=0x80, b1=0x05, vendor=0,
    # attr=0. Nothing to decode beyond that; but the length check fires.
    wire = struct.pack("!BBHH", 0x80, 5, 0, 0)
    with pytest.raises(ValueError, match="< header size"):
        AVP.decode_one(wire, 0)


def test_reserved_bits_nonzero_rejected():
    # Set one of the reserved bits (bit 5 of byte 0).
    a = AVP(attribute_type=AttrType.MESSAGE_TYPE, value=b"\x00\x01")
    wire = bytearray(a.encode())
    wire[0] |= 0x04     # a reserved bit
    with pytest.raises(ValueError, match="reserved bits non-zero"):
        AVP.decode_one(bytes(wire), 0)


def test_message_type_decode_wrong_type_rejected():
    # Wrap a non-MessageType AVP and try to decode it as Message Type.
    a = build_router_id(0x01020304)
    d, _ = AVP.decode_one(a.encode())
    with pytest.raises(ValueError, match="expected IETF AVP type 0"):
        decode_message_type(d)


def test_router_id_wrong_length_rejected():
    # Craft a Router ID AVP with a 3-byte value (invalid).
    bad = AVP(attribute_type=AttrType.ROUTER_ID, value=b"\x01\x02\x03")
    d, _ = AVP.decode_one(bad.encode())
    with pytest.raises(ValueError, match="expected value length 4"):
        decode_router_id(d)


# ---------------------------------------------------------------------------
# Sequence decoding
# ---------------------------------------------------------------------------

def test_decode_sequence_multiple_avps():
    a1 = build_message_type(MessageType.SCCRQ)
    a2 = build_host_name("xcespfc")
    a3 = build_router_id(0x0A0A0A01)
    wire = a1.encode() + a2.encode() + a3.encode()
    seq = AVP.decode_sequence(wire)
    assert len(seq) == 3
    assert seq[0].attribute_type == int(AttrType.MESSAGE_TYPE)
    assert seq[1].attribute_type == int(AttrType.HOST_NAME)
    assert seq[2].attribute_type == int(AttrType.ROUTER_ID)
    assert decode_message_type(seq[0]) == int(MessageType.SCCRQ)
    assert seq[1].value == b"xcespfc"
    assert decode_router_id(seq[2]) == 0x0A0A0A01


def test_decode_sequence_empty_buffer_returns_empty_list():
    assert AVP.decode_sequence(b"") == []


def test_decode_sequence_stops_at_end_exactly():
    # Two AVPs, no trailing bytes, no truncation.
    seq = AVP.decode_sequence(
        build_message_type(MessageType.HELLO).encode()
        + build_router_id(1).encode()
    )
    assert len(seq) == 2
