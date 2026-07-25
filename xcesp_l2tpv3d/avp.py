"""AVP (Attribute-Value Pair) encoding and decoding per RFC 3931 §5.

This module handles:
  - the 6-byte AVP header (M/H bits, reserved, 10-bit length, vendor ID,
    attribute type) at §5.2
  - per-type value serialisation for the mandatory AVP set used in the
    control messages this daemon builds (SCCRQ / SCCRP / SCCCN / HELLO /
    StopCCN / ICRQ / ICRP / ICCN / CDN / WEN / SLI)
  - the hidden-AVP encryption path at §4.3 (MD5 keystream over
    Attribute Type + Shared Secret + Random Vector AVP)

The AVP class is deliberately minimal: it carries wire-representable
fields (mandatory, hidden, vendor_id, attribute_type, value) as bytes.
Higher-level modules (messages.py, tunnel_fsm.py) build typed AVPs via
the ``build_*`` factories and decode with ``AVP.decode_sequence``.
"""

from __future__ import annotations

import enum
import hashlib
import struct
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: IETF vendor ID (all standard AVPs)
VENDOR_IETF = 0

#: Fixed AVP header length in bytes (M/H/rsvd/length/vendor/attr).
AVP_HEADER_LEN = 6

#: Max total AVP length per RFC 3931 §5.2 (10-bit Length field → 1023).
AVP_MAX_LEN = 0x03FF

#: Max value length = max total - header.
AVP_MAX_VALUE_LEN = AVP_MAX_LEN - AVP_HEADER_LEN

#: MD5 digest size in bytes; used for the hidden-AVP keystream.
_MD5_BLOCK = 16


# ---------------------------------------------------------------------------
# Attribute-type enums (subset for 0.1.0; extends per release)
# ---------------------------------------------------------------------------

class AttrType(enum.IntEnum):
    """IETF-vendor AVP attribute types used by RFC 3931 control messages.

    Values match the IANA L2TPv3 Attribute Value Pair registry.  This
    enum covers 0-16 (mostly L2TPv2 legacy but a handful still used in
    v3) and the v3-specific block 46-75.
    """

    # --- 0-16: general / v2-legacy, some still valid for v3 ---
    MESSAGE_TYPE                = 0
    RESULT_CODE                 = 1
    PROTOCOL_VERSION            = 2   # v2 legacy; v3 uses version=3 in header
    FRAMING_CAPABILITIES        = 3
    BEARER_CAPABILITIES         = 4
    TIE_BREAKER                 = 5
    FIRMWARE_REVISION           = 6
    HOST_NAME                   = 7   # ("Host Name" - see also ROUTER_ID)
    VENDOR_NAME                 = 8
    ASSIGNED_TUNNEL_ID          = 9   # v2 legacy (v3 uses ASSIGNED_CONTROL_CONNECTION_ID)
    RECEIVE_WINDOW_SIZE         = 10
    CHALLENGE                   = 11
    Q931_CAUSE_CODE             = 12
    CHALLENGE_RESPONSE          = 13
    ASSIGNED_SESSION_ID_V2      = 14  # v2 legacy
    CALL_SERIAL_NUMBER          = 15
    MINIMUM_BPS                 = 16

    # --- 36: random vector for hidden-AVP path ---
    RANDOM_VECTOR               = 36

    # --- 46-75: L2TPv3-specific block (subset used at 0.1.0-0.4.0) ---
    EXTENDED_VENDOR_ID          = 58
    MESSAGE_DIGEST              = 59
    ROUTER_ID                   = 60
    ASSIGNED_CONTROL_CONNECTION_ID = 61
    PSEUDOWIRE_CAPABILITIES_LIST   = 62
    LOCAL_SESSION_ID            = 63
    REMOTE_SESSION_ID           = 64
    ASSIGNED_COOKIE             = 65
    REMOTE_END_ID               = 66
    PSEUDOWIRE_TYPE             = 68
    L2_SPECIFIC_SUBLAYER        = 69
    DATA_SEQUENCING             = 70
    CIRCUIT_STATUS              = 71
    PREFERRED_LANGUAGE          = 72
    CTL_MSG_AUTH_NONCE          = 73
    TX_CONNECT_SPEED            = 74
    RX_CONNECT_SPEED            = 75


class MessageType(enum.IntEnum):
    """L2TP control-message types carried in the Message Type AVP (§5.4.1)."""

    SCCRQ    = 1    # Start-Control-Connection-Request
    SCCRP    = 2    # Start-Control-Connection-Reply
    SCCCN    = 3    # Start-Control-Connection-Connected
    StopCCN  = 4    # Stop-Control-Connection-Notification
    HELLO    = 6    # Hello (keepalive)
    ICRQ     = 10   # Incoming-Call-Request
    ICRP     = 11   # Incoming-Call-Reply
    ICCN     = 12   # Incoming-Call-Connected
    CDN      = 14   # Call-Disconnect-Notify
    WEN      = 15   # WAN-Error-Notify
    SLI      = 16   # Set-Link-Info


class PseudowireType(enum.IntEnum):
    """Pseudowire Type AVP values (§5.4.11 / IANA registry).

    Only the two we ship in xcesp-l2tpv3d 1.0 are listed as first-class
    enum members; the rest are addressable by raw int.
    """

    ETHERNET_VLAN = 4     # Ethernet VLAN
    ETHERNET      = 5     # Ethernet


class DigestHash(enum.IntEnum):
    """Hash Type field of Message Digest AVP (§5.4.3)."""

    HMAC_MD5  = 0
    HMAC_SHA1 = 1


# ---------------------------------------------------------------------------
# The AVP data class
# ---------------------------------------------------------------------------

@dataclass
class AVP:
    """A single Attribute-Value Pair.

    ``value`` is always the *decoded* value bytes (i.e. after hidden-AVP
    decryption, if applicable).  Whether an AVP was hidden on the wire
    is preserved in the ``hidden`` flag so an outgoing AVP can be
    re-hidden with the same setting.
    """

    attribute_type: int          # 16-bit; AttrType enum values are ints
    value: bytes = b""
    mandatory: bool = True       # M bit; default True per most §5.4 AVPs
    hidden: bool = False         # H bit; encoder hides if True and secret set
    vendor_id: int = VENDOR_IETF

    # ---- encoding ---------------------------------------------------------

    def encode(
        self,
        *,
        shared_secret: Optional[bytes] = None,
        random_vector: Optional[bytes] = None,
    ) -> bytes:
        """Serialise the AVP to wire bytes.

        If ``hidden`` is True, both ``shared_secret`` and ``random_vector``
        must be provided; the value field is encrypted per RFC 3931 §4.3.
        The Random Vector AVP itself must be present *before* this hidden
        AVP in the containing message (that's the caller's responsibility).
        """
        if self.vendor_id < 0 or self.vendor_id > 0xFFFF:
            raise ValueError(f"vendor_id out of range: {self.vendor_id}")
        if self.attribute_type < 0 or self.attribute_type > 0xFFFF:
            raise ValueError(f"attribute_type out of range: {self.attribute_type}")

        if self.hidden:
            if shared_secret is None or random_vector is None:
                raise ValueError(
                    "hidden AVP requires shared_secret and random_vector"
                )
            value_field = _hide_value(
                self.attribute_type, self.value, shared_secret, random_vector
            )
        else:
            value_field = self.value

        total_len = AVP_HEADER_LEN + len(value_field)
        if total_len > AVP_MAX_LEN:
            raise ValueError(
                f"AVP too long: {total_len} > {AVP_MAX_LEN} (attribute_type="
                f"{self.attribute_type})"
            )

        # Byte 0: M(1) | H(1) | rsvd(4)=0 | length_high(2)
        # Byte 1: length_low(8)
        b0 = ((1 if self.mandatory else 0) << 7) \
             | ((1 if self.hidden else 0) << 6) \
             | ((total_len >> 8) & 0x03)
        b1 = total_len & 0xFF
        header = struct.pack("!BBHH", b0, b1, self.vendor_id, self.attribute_type)
        return header + value_field

    # ---- decoding ---------------------------------------------------------

    @classmethod
    def decode_one(
        cls,
        buf: bytes,
        offset: int = 0,
        *,
        shared_secret: Optional[bytes] = None,
        random_vector: Optional[bytes] = None,
    ) -> Tuple["AVP", int]:
        """Decode one AVP starting at ``offset`` in ``buf``.

        Returns ``(avp, next_offset)``.  If the decoded AVP is hidden,
        the caller must have provided ``shared_secret`` and the Random
        Vector value seen earlier in the same message; the returned
        AVP's ``value`` is the decrypted original bytes.
        """
        remaining = len(buf) - offset
        if remaining < AVP_HEADER_LEN:
            raise ValueError(
                f"buffer too short for AVP header at offset {offset}: "
                f"have {remaining}, need {AVP_HEADER_LEN}"
            )

        b0, b1, vendor_id, attr_type = struct.unpack_from(
            "!BBHH", buf, offset
        )
        mandatory = bool(b0 & 0x80)
        hidden    = bool(b0 & 0x40)
        rsvd      = (b0 >> 2) & 0x0F
        if rsvd != 0:
            raise ValueError(
                f"AVP reserved bits non-zero (got 0x{rsvd:x}) at offset {offset}"
            )
        total_len = ((b0 & 0x03) << 8) | b1
        if total_len < AVP_HEADER_LEN:
            raise ValueError(
                f"AVP length {total_len} < header size {AVP_HEADER_LEN} "
                f"at offset {offset}"
            )
        if offset + total_len > len(buf):
            raise ValueError(
                f"AVP length {total_len} extends past buffer "
                f"(offset={offset}, buf_len={len(buf)})"
            )

        value_field = bytes(buf[offset + AVP_HEADER_LEN : offset + total_len])
        if hidden:
            if shared_secret is None or random_vector is None:
                raise ValueError(
                    "hidden AVP encountered but shared_secret / random_vector "
                    "not supplied"
                )
            value = _unhide_value(
                attr_type, value_field, shared_secret, random_vector
            )
        else:
            value = value_field

        avp = cls(
            attribute_type=attr_type,
            value=value,
            mandatory=mandatory,
            hidden=hidden,
            vendor_id=vendor_id,
        )
        return avp, offset + total_len

    @classmethod
    def decode_sequence(
        cls,
        buf: bytes,
        offset: int = 0,
        *,
        shared_secret: Optional[bytes] = None,
    ) -> List["AVP"]:
        """Decode a full sequence of AVPs from ``buf`` starting at ``offset``.

        Walks the buffer to the end, decoding one AVP per iteration.
        Hidden AVPs are decrypted using ``shared_secret`` plus the Random
        Vector AVP found earlier in the sequence (per RFC 3931 §4.3: the
        RV must precede any hidden AVPs it applies to).
        """
        result: List[AVP] = []
        random_vector: Optional[bytes] = None
        pos = offset
        while pos < len(buf):
            avp, pos = cls.decode_one(
                buf, pos,
                shared_secret=shared_secret,
                random_vector=random_vector,
            )
            if avp.vendor_id == VENDOR_IETF \
                    and avp.attribute_type == AttrType.RANDOM_VECTOR:
                random_vector = avp.value
            result.append(avp)
        return result


# ---------------------------------------------------------------------------
# Hidden-AVP keystream (RFC 3931 §4.3)
# ---------------------------------------------------------------------------

def _hide_value(
    attribute_type: int,
    plaintext: bytes,
    shared_secret: bytes,
    random_vector: bytes,
) -> bytes:
    """Encrypt an AVP value using the RFC 3931 §4.3 MD5 keystream.

    Hidden-AVP wire format for the value field:
        - 2-byte Original AV Length (big-endian)
        - Original AV Value bytes
        - Padding to next MD5 block (16 bytes) boundary
    The whole (length + value + padding) is XOR'd with the keystream.
    """
    original_len = len(plaintext)
    if original_len > 0xFFFF:
        raise ValueError(f"hidden AVP value too long: {original_len}")

    body = struct.pack("!H", original_len) + plaintext
    pad = (-len(body)) % _MD5_BLOCK
    if pad:
        body += b"\x00" * pad

    keystream = _generate_keystream(
        attribute_type, shared_secret, random_vector, len(body)
    )
    return bytes(a ^ b for a, b in zip(body, keystream))


def _unhide_value(
    attribute_type: int,
    ciphertext: bytes,
    shared_secret: bytes,
    random_vector: bytes,
) -> bytes:
    """Reverse of _hide_value; strips the length prefix and padding."""
    if len(ciphertext) < _MD5_BLOCK or len(ciphertext) % _MD5_BLOCK != 0:
        raise ValueError(
            f"hidden AVP ciphertext length {len(ciphertext)} is not a "
            f"non-zero multiple of {_MD5_BLOCK}"
        )
    keystream = _generate_keystream(
        attribute_type, shared_secret, random_vector, len(ciphertext)
    )
    body = bytes(a ^ b for a, b in zip(ciphertext, keystream))
    (original_len,) = struct.unpack_from("!H", body, 0)
    if 2 + original_len > len(body):
        raise ValueError(
            f"hidden AVP original length {original_len} exceeds decrypted "
            f"body ({len(body) - 2} bytes available)"
        )
    return body[2 : 2 + original_len]


def _generate_keystream(
    attribute_type: int,
    shared_secret: bytes,
    random_vector: bytes,
    length: int,
) -> bytes:
    """Generate ``length`` bytes of MD5 keystream per RFC 3931 §4.3.

    First block:   p1 = MD5(AttributeType || SharedSecret || RandomVector)
    Subsequent:    pn = MD5(SharedSecret || p_{n-1})
    where AttributeType is the 2-byte network-order attribute-type field.
    """
    if length <= 0:
        return b""
    attr_bytes = struct.pack("!H", attribute_type)
    ks = bytearray()
    prev = hashlib.md5(attr_bytes + shared_secret + random_vector).digest()
    ks.extend(prev)
    while len(ks) < length:
        prev = hashlib.md5(shared_secret + prev).digest()
        ks.extend(prev)
    return bytes(ks[:length])


# ---------------------------------------------------------------------------
# Typed value helpers (encoders + decoders per AVP type used at 0.1.0-0.4.0)
# ---------------------------------------------------------------------------
#
# Each AVP §5.4.x has its own value layout.  The functions below are
# thin wrappers around ``struct`` that make ``messages.py`` and the
# state machines self-documenting: ``build_message_type(MessageType
# .SCCRQ)`` reads better than ``AVP(0, struct.pack('!H', 1))``.

def build_message_type(mt: MessageType | int) -> AVP:
    """§5.4.1 — 2-byte Message Type."""
    return AVP(
        attribute_type=AttrType.MESSAGE_TYPE,
        value=struct.pack("!H", int(mt)),
        mandatory=True,
    )


def decode_message_type(avp: AVP) -> int:
    _require_attr(avp, AttrType.MESSAGE_TYPE, exact_len=2)
    (mt,) = struct.unpack("!H", avp.value)
    return mt


def build_result_code(
    result: int,
    error_code: Optional[int] = None,
    error_message: bytes | str = b"",
) -> AVP:
    """§5.4.2 — 2-byte Result [+ 2-byte Error [+ variable Error Message]]."""
    payload = struct.pack("!H", result)
    if error_code is not None:
        payload += struct.pack("!H", error_code)
        if error_message:
            if isinstance(error_message, str):
                error_message = error_message.encode("utf-8")
            payload += error_message
    elif error_message:
        raise ValueError("error_message requires error_code")
    return AVP(
        attribute_type=AttrType.RESULT_CODE,
        value=payload,
        mandatory=True,
    )


def decode_result_code(avp: AVP) -> Tuple[int, Optional[int], bytes]:
    _require_attr(avp, AttrType.RESULT_CODE, min_len=2)
    val = avp.value
    (result,) = struct.unpack_from("!H", val, 0)
    error_code: Optional[int] = None
    error_message = b""
    if len(val) >= 4:
        (error_code,) = struct.unpack_from("!H", val, 2)
        error_message = val[4:]
    return result, error_code, error_message


def build_random_vector(rv_bytes: bytes) -> AVP:
    """§5.3 — variable-length random bytes used as the hidden-AVP IV."""
    if len(rv_bytes) < _MD5_BLOCK:
        raise ValueError(
            f"random vector should be at least {_MD5_BLOCK} bytes "
            f"(RFC 3931 §4.3 recommends the MD5 block size)"
        )
    return AVP(
        attribute_type=AttrType.RANDOM_VECTOR,
        value=rv_bytes,
        mandatory=True,
    )


def build_host_name(name: str | bytes) -> AVP:
    """§5.4.3 (v2 legacy) — Host Name AVP."""
    if isinstance(name, str):
        name = name.encode("utf-8")
    return AVP(attribute_type=AttrType.HOST_NAME, value=name, mandatory=True)


def build_router_id(router_id: int) -> AVP:
    """§5.4.4 — 4-byte Router ID (typically a 32-bit IPv4-style ID)."""
    if router_id < 0 or router_id > 0xFFFFFFFF:
        raise ValueError(f"router_id out of range: {router_id}")
    return AVP(
        attribute_type=AttrType.ROUTER_ID,
        value=struct.pack("!I", router_id),
        mandatory=True,
    )


def decode_router_id(avp: AVP) -> int:
    _require_attr(avp, AttrType.ROUTER_ID, exact_len=4)
    (rid,) = struct.unpack("!I", avp.value)
    return rid


def build_assigned_control_connection_id(ccid: int) -> AVP:
    """§5.4.5 — 4-byte Assigned Control Connection ID."""
    if ccid < 0 or ccid > 0xFFFFFFFF:
        raise ValueError(f"ccid out of range: {ccid}")
    return AVP(
        attribute_type=AttrType.ASSIGNED_CONTROL_CONNECTION_ID,
        value=struct.pack("!I", ccid),
        mandatory=True,
    )


def build_pseudowire_capabilities_list(pw_types: Iterable[int]) -> AVP:
    """§5.4.6 — list of 2-byte Pseudowire Types the sender supports."""
    pw_list = list(pw_types)
    if not pw_list:
        raise ValueError("PW capabilities list must be non-empty")
    return AVP(
        attribute_type=AttrType.PSEUDOWIRE_CAPABILITIES_LIST,
        value=b"".join(struct.pack("!H", int(t)) for t in pw_list),
        mandatory=True,
    )


def decode_pseudowire_capabilities_list(avp: AVP) -> List[int]:
    _require_attr(avp, AttrType.PSEUDOWIRE_CAPABILITIES_LIST, min_len=2)
    if len(avp.value) % 2 != 0:
        raise ValueError(
            f"PW capabilities list length {len(avp.value)} not a multiple of 2"
        )
    return [
        struct.unpack_from("!H", avp.value, i)[0]
        for i in range(0, len(avp.value), 2)
    ]


def build_local_session_id(sid: int) -> AVP:
    """§5.4.7 — 4-byte Local Session ID."""
    return _u32_avp(AttrType.LOCAL_SESSION_ID, sid, "sid")


def build_remote_session_id(sid: int) -> AVP:
    """§5.4.8 — 4-byte Remote Session ID."""
    return _u32_avp(AttrType.REMOTE_SESSION_ID, sid, "sid")


def build_assigned_cookie(cookie: bytes) -> AVP:
    """§5.4.9 — 0, 4, or 8 bytes Assigned Cookie."""
    if len(cookie) not in (0, 4, 8):
        raise ValueError(
            f"assigned cookie must be 0/4/8 bytes, got {len(cookie)}"
        )
    return AVP(
        attribute_type=AttrType.ASSIGNED_COOKIE,
        value=cookie,
        mandatory=True,
    )


def build_remote_end_id(end_id: bytes | str) -> AVP:
    """§5.4.10 — variable-length Remote End ID (opaque)."""
    if isinstance(end_id, str):
        end_id = end_id.encode("utf-8")
    return AVP(
        attribute_type=AttrType.REMOTE_END_ID,
        value=end_id,
        mandatory=True,
    )


def build_pseudowire_type(pw: PseudowireType | int) -> AVP:
    """§5.4.11 — 2-byte Pseudowire Type."""
    return AVP(
        attribute_type=AttrType.PSEUDOWIRE_TYPE,
        value=struct.pack("!H", int(pw)),
        mandatory=True,
    )


def build_l2_specific_sublayer(sublayer_type: int) -> AVP:
    """§5.4.12 — 2-byte L2-Specific Sublayer Type (0=none, 1=default)."""
    return AVP(
        attribute_type=AttrType.L2_SPECIFIC_SUBLAYER,
        value=struct.pack("!H", sublayer_type),
        mandatory=True,
    )


def build_data_sequencing(level: int) -> AVP:
    """§5.4.13 — 2-byte Data Sequencing (0=disabled, 1=out-of-order OK,
    2=all packets in order)."""
    return AVP(
        attribute_type=AttrType.DATA_SEQUENCING,
        value=struct.pack("!H", level),
        mandatory=True,
    )


def build_circuit_status(status: int) -> AVP:
    """§5.4.14 — 2-byte Circuit Status bit-mask (bit 0 = 'up', bit 1 = 'new')."""
    return AVP(
        attribute_type=AttrType.CIRCUIT_STATUS,
        value=struct.pack("!H", status),
        mandatory=True,
    )


def build_preferred_language(lang_tag: str | bytes) -> AVP:
    """§5.4.15 — RFC 3066 language tag."""
    if isinstance(lang_tag, str):
        lang_tag = lang_tag.encode("utf-8")
    return AVP(
        attribute_type=AttrType.PREFERRED_LANGUAGE,
        value=lang_tag,
        mandatory=False,   # optional per RFC
    )


def build_tx_connect_speed(bps: int) -> AVP:
    """§5.4.16 — 8-byte Tx Connect Speed in bits per second."""
    return _u64_avp(AttrType.TX_CONNECT_SPEED, bps, "bps")


def build_rx_connect_speed(bps: int) -> AVP:
    """§5.4.17 — 8-byte Rx Connect Speed in bits per second."""
    return _u64_avp(AttrType.RX_CONNECT_SPEED, bps, "bps")


def build_message_digest(hash_type: DigestHash | int, digest: bytes) -> AVP:
    """§5.4.3 — Message Digest AVP.  Payload = 1-byte hash type + digest."""
    return AVP(
        attribute_type=AttrType.MESSAGE_DIGEST,
        value=bytes([int(hash_type)]) + digest,
        mandatory=True,
    )


def build_receive_window_size(window: int) -> AVP:
    """§5.4.3 (Receive Window Size, AVP 10) — 2-byte window size."""
    if window < 0 or window > 0xFFFF:
        raise ValueError(f"receive window out of range: {window}")
    return AVP(
        attribute_type=AttrType.RECEIVE_WINDOW_SIZE,
        value=struct.pack("!H", window),
        mandatory=True,
    )


def build_vendor_name(name: str | bytes) -> AVP:
    """§5.4 legacy — Vendor Name AVP, optional descriptive string."""
    if isinstance(name, str):
        name = name.encode("utf-8")
    return AVP(
        attribute_type=AttrType.VENDOR_NAME,
        value=name,
        mandatory=False,   # optional per RFC
    )


def build_firmware_revision(revision: int) -> AVP:
    """§5.4 legacy — Firmware Revision AVP (2-byte uint)."""
    if revision < 0 or revision > 0xFFFF:
        raise ValueError(f"firmware revision out of range: {revision}")
    return AVP(
        attribute_type=AttrType.FIRMWARE_REVISION,
        value=struct.pack("!H", revision),
        mandatory=False,   # optional per RFC
    )


def build_control_message_auth_nonce(nonce: bytes) -> AVP:
    """§4.3 — Control Message Authentication Nonce (AVP 73).

    Sent in SCCRQ (Local Nonce) and SCCRP (Remote Nonce).  These
    nonces MAY be used as HMAC key material for the Message Digest
    AVP; 0.3.0 provides the AVP but the digest algorithm uses the
    shared secret directly (RFC-compatible message-integrity path).
    """
    if not nonce:
        raise ValueError("nonce must not be empty")
    return AVP(
        attribute_type=AttrType.CTL_MSG_AUTH_NONCE,
        value=nonce,
        mandatory=True,
    )


def decode_control_message_auth_nonce(avp: AVP) -> bytes:
    _require_attr(avp, AttrType.CTL_MSG_AUTH_NONCE, min_len=1)
    return avp.value


def decode_assigned_control_connection_id(avp: AVP) -> int:
    _require_attr(avp, AttrType.ASSIGNED_CONTROL_CONNECTION_ID, exact_len=4)
    (ccid,) = struct.unpack("!I", avp.value)
    return ccid


def decode_host_name(avp: AVP) -> bytes:
    _require_attr(avp, AttrType.HOST_NAME, min_len=0)
    return avp.value


def decode_receive_window_size(avp: AVP) -> int:
    _require_attr(avp, AttrType.RECEIVE_WINDOW_SIZE, exact_len=2)
    (w,) = struct.unpack("!H", avp.value)
    return w


def find_avp(
    avps: list["AVP"],
    attribute_type: int,
    vendor_id: int = VENDOR_IETF,
) -> Optional["AVP"]:
    """Return the first AVP matching ``(vendor_id, attribute_type)``,
    or None.  Useful when message parsers want a single AVP without
    iterating manually."""
    for avp in avps:
        if avp.vendor_id == vendor_id and avp.attribute_type == int(attribute_type):
            return avp
    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _u32_avp(attr: int, value: int, name: str) -> AVP:
    if value < 0 or value > 0xFFFFFFFF:
        raise ValueError(f"{name} out of range: {value}")
    return AVP(attribute_type=attr, value=struct.pack("!I", value), mandatory=True)


def _u64_avp(attr: int, value: int, name: str) -> AVP:
    if value < 0 or value > 0xFFFFFFFFFFFFFFFF:
        raise ValueError(f"{name} out of range: {value}")
    return AVP(attribute_type=attr, value=struct.pack("!Q", value), mandatory=True)


def _require_attr(
    avp: AVP,
    expected_type: int,
    *,
    exact_len: Optional[int] = None,
    min_len: Optional[int] = None,
) -> None:
    if avp.vendor_id != VENDOR_IETF or avp.attribute_type != int(expected_type):
        raise ValueError(
            f"expected IETF AVP type {int(expected_type)}, got vendor="
            f"{avp.vendor_id} type={avp.attribute_type}"
        )
    if exact_len is not None and len(avp.value) != exact_len:
        raise ValueError(
            f"AVP type {int(expected_type)}: expected value length "
            f"{exact_len}, got {len(avp.value)}"
        )
    if min_len is not None and len(avp.value) < min_len:
        raise ValueError(
            f"AVP type {int(expected_type)}: expected value length >= "
            f"{min_len}, got {len(avp.value)}"
        )
