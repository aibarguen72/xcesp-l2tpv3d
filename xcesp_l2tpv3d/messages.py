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
from typing import Iterable, List, Optional

from .avp import (
    AVP,
    AVP_HEADER_LEN,
    AttrType,
    DigestHash,
    MessageType,
    PseudowireType,
    VENDOR_IETF,
    build_assigned_control_connection_id,
    build_assigned_cookie,
    build_circuit_status,
    build_control_message_auth_nonce,
    build_data_sequencing,
    build_firmware_revision,
    build_host_name,
    build_l2_specific_sublayer,
    build_local_session_id,
    build_message_digest,
    build_message_type,
    build_pseudowire_capabilities_list,
    build_pseudowire_type,
    build_receive_window_size,
    build_remote_end_id,
    build_remote_session_id,
    build_result_code,
    build_router_id,
    build_rx_connect_speed,
    build_tx_connect_speed,
    build_vendor_name,
    decode_assigned_control_connection_id,
    decode_assigned_cookie,
    decode_circuit_status,
    decode_control_message_auth_nonce,
    decode_data_sequencing,
    decode_host_name,
    decode_l2_specific_sublayer,
    decode_local_session_id,
    decode_message_type,
    decode_pseudowire_capabilities_list,
    decode_pseudowire_type,
    decode_receive_window_size,
    decode_remote_end_id,
    decode_remote_session_id,
    decode_result_code,
    decode_router_id,
    decode_rx_connect_speed,
    decode_tx_connect_speed,
    find_avp,
)


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


# ---------------------------------------------------------------------------
# Typed message builders — return List[AVP] ready for wrapping in a
# ControlMessage by the caller (Ns/Nr/CCID come from the Peer context).
# ---------------------------------------------------------------------------

def build_sccrq_avps(
    *,
    router_id: int,
    assigned_ccid: int,
    host_name: str,
    pw_capabilities: Iterable[int],
    receive_window: int = 4,
    vendor_name: Optional[str] = None,
    firmware_revision: Optional[int] = None,
    auth_nonce: Optional[bytes] = None,
) -> List[AVP]:
    """Assemble the AVP list for a Start-Control-Connection-Request.

    Mandatory AVPs per RFC 3931 §6.1: Message Type, Router ID,
    Assigned Control Connection ID, Pseudowire Capabilities List,
    Host Name.  Receive Window Size is recommended (RFC 3931 §5.4.3)
    and required in practice for interop.  Vendor Name, Firmware
    Revision, and Control Message Authentication Nonce are optional.
    """
    avps: List[AVP] = [
        build_message_type(MessageType.SCCRQ),
        build_router_id(router_id),
        build_assigned_control_connection_id(assigned_ccid),
        build_pseudowire_capabilities_list(pw_capabilities),
        build_host_name(host_name),
        build_receive_window_size(receive_window),
    ]
    if vendor_name is not None:
        avps.append(build_vendor_name(vendor_name))
    if firmware_revision is not None:
        avps.append(build_firmware_revision(firmware_revision))
    if auth_nonce is not None:
        avps.append(build_control_message_auth_nonce(auth_nonce))
    return avps


def build_sccrp_avps(
    *,
    router_id: int,
    assigned_ccid: int,
    host_name: str,
    pw_capabilities: Iterable[int],
    receive_window: int = 4,
    vendor_name: Optional[str] = None,
    firmware_revision: Optional[int] = None,
    auth_nonce: Optional[bytes] = None,
) -> List[AVP]:
    """Assemble the AVP list for a Start-Control-Connection-Reply.

    Mirrors SCCRQ (§6.2 mandatory AVPs are identical) but with
    Message Type = SCCRP.
    """
    avps: List[AVP] = [
        build_message_type(MessageType.SCCRP),
        build_router_id(router_id),
        build_assigned_control_connection_id(assigned_ccid),
        build_pseudowire_capabilities_list(pw_capabilities),
        build_host_name(host_name),
        build_receive_window_size(receive_window),
    ]
    if vendor_name is not None:
        avps.append(build_vendor_name(vendor_name))
    if firmware_revision is not None:
        avps.append(build_firmware_revision(firmware_revision))
    if auth_nonce is not None:
        avps.append(build_control_message_auth_nonce(auth_nonce))
    return avps


def build_scccn_avps() -> List[AVP]:
    """Assemble the AVP list for a Start-Control-Connection-Connected.

    RFC 3931 §6.3: only Message Type is mandatory.  Message Digest AVP
    is added by encode_signed if authentication is in use.
    """
    return [build_message_type(MessageType.SCCCN)]


def build_hello_avps() -> List[AVP]:
    """Assemble the AVP list for a HELLO (RFC 3931 §6.5).

    Message Type only.  Message Digest AVP is added by encode_signed
    if authentication is in use.
    """
    return [build_message_type(MessageType.HELLO)]


def build_stopccn_avps(
    *,
    assigned_ccid: int,
    result_code: int,
    error_code: Optional[int] = None,
    error_message: str = "",
) -> List[AVP]:
    """Assemble the AVP list for a StopCCN (RFC 3931 §6.4).

    Mandatory: Message Type, Assigned Control Connection ID, Result Code.
    """
    return [
        build_message_type(MessageType.StopCCN),
        build_assigned_control_connection_id(assigned_ccid),
        build_result_code(result_code, error_code, error_message),
    ]


# ---------------------------------------------------------------------------
# Session-level typed message builders (RFC 3931 §6.6–§6.9)
# ---------------------------------------------------------------------------

def build_icrq_avps(
    *,
    local_sid: int,
    pseudowire_type: PseudowireType | int,
    remote_end_id: Optional[str] = None,
    l2_specific_sublayer: int = 1,          # 1 = default sublayer, 0 = none
    data_sequencing: int = 0,               # 0 = disabled
    circuit_status: int = 0b11,             # bit0=up, bit1=new
    assigned_cookie: Optional[bytes] = None,
    tx_connect_speed: Optional[int] = None,
    rx_connect_speed: Optional[int] = None,
) -> List[AVP]:
    """Assemble the AVP list for an Incoming-Call-Request (§6.6).

    Mandatory: Message Type, Local Session ID, Pseudowire Type.  L2-
    Specific Sublayer + Data Sequencing + Circuit Status are strongly
    recommended for RFC-conformant handshake.  Remote End ID identifies
    the session at the peer (used for demultiplexing when the peer has
    multiple pre-configured sessions).
    """
    avps: List[AVP] = [
        build_message_type(MessageType.ICRQ),
        build_local_session_id(local_sid),
        build_pseudowire_type(pseudowire_type),
        build_l2_specific_sublayer(l2_specific_sublayer),
        build_data_sequencing(data_sequencing),
        build_circuit_status(circuit_status),
    ]
    if remote_end_id is not None:
        avps.append(build_remote_end_id(remote_end_id))
    if assigned_cookie is not None:
        avps.append(build_assigned_cookie(assigned_cookie))
    if tx_connect_speed is not None:
        avps.append(build_tx_connect_speed(tx_connect_speed))
    if rx_connect_speed is not None:
        avps.append(build_rx_connect_speed(rx_connect_speed))
    return avps


def build_icrp_avps(
    *,
    local_sid: int,
    remote_sid: int,
    l2_specific_sublayer: int = 1,
    data_sequencing: int = 0,
    circuit_status: int = 0b11,
    assigned_cookie: Optional[bytes] = None,
    tx_connect_speed: Optional[int] = None,
    rx_connect_speed: Optional[int] = None,
) -> List[AVP]:
    """Assemble the AVP list for an Incoming-Call-Reply (§6.7).

    Mandatory: Message Type, Local Session ID.  Remote Session ID is
    included so the initiator can correlate this ICRP with its earlier
    ICRQ (whose Local Session ID equals our Remote Session ID here).
    """
    avps: List[AVP] = [
        build_message_type(MessageType.ICRP),
        build_local_session_id(local_sid),
        build_remote_session_id(remote_sid),
        build_l2_specific_sublayer(l2_specific_sublayer),
        build_data_sequencing(data_sequencing),
        build_circuit_status(circuit_status),
    ]
    if assigned_cookie is not None:
        avps.append(build_assigned_cookie(assigned_cookie))
    if tx_connect_speed is not None:
        avps.append(build_tx_connect_speed(tx_connect_speed))
    if rx_connect_speed is not None:
        avps.append(build_rx_connect_speed(rx_connect_speed))
    return avps


def build_iccn_avps(
    *,
    local_sid: int,
    remote_sid: int,
    l2_specific_sublayer: int = 1,
    data_sequencing: int = 0,
    circuit_status: int = 0b11,
    tx_connect_speed: Optional[int] = None,
    rx_connect_speed: Optional[int] = None,
) -> List[AVP]:
    """Assemble the AVP list for an Incoming-Call-Connected (§6.8).

    Mandatory: Message Type.  Also carries Local + Remote Session ID
    so the responder can finalise binding.  L2-Specific Sublayer and
    Data Sequencing must be present per RFC (they finalise the choice
    from ICRQ+ICRP negotiation).
    """
    avps: List[AVP] = [
        build_message_type(MessageType.ICCN),
        build_local_session_id(local_sid),
        build_remote_session_id(remote_sid),
        build_l2_specific_sublayer(l2_specific_sublayer),
        build_data_sequencing(data_sequencing),
        build_circuit_status(circuit_status),
    ]
    if tx_connect_speed is not None:
        avps.append(build_tx_connect_speed(tx_connect_speed))
    if rx_connect_speed is not None:
        avps.append(build_rx_connect_speed(rx_connect_speed))
    return avps


def build_cdn_avps(
    *,
    local_sid: int,
    remote_sid: int,
    result_code: int,
    error_code: Optional[int] = None,
    error_message: str = "",
) -> List[AVP]:
    """Assemble the AVP list for a Call-Disconnect-Notify (§6.9).

    Mandatory: Message Type, Result Code, Local Session ID.  Remote
    Session ID identifies the session at the recipient side (which is
    the one being torn down).
    """
    return [
        build_message_type(MessageType.CDN),
        build_local_session_id(local_sid),
        build_remote_session_id(remote_sid),
        build_result_code(result_code, error_code, error_message),
    ]


# ---------------------------------------------------------------------------
# Session-level parsers
# ---------------------------------------------------------------------------

@dataclass
class SessionFields:
    """Fields extracted from any of ICRQ / ICRP / ICCN.

    Not every AVP appears in every message — for ICRQ there's no
    Remote Session ID (initiator doesn't know it yet); for ICRP+ICCN
    it's mandatory.  Fields absent from the message are set to their
    natural zero / None.
    """

    local_sid: int              # sender's Local Session ID
    remote_sid: int             # sender's view of recipient's SID (0 for ICRQ)
    pseudowire_type: Optional[int] = None    # present in ICRQ; usually omitted in ICRP/ICCN
    l2_specific_sublayer: Optional[int] = None
    data_sequencing: Optional[int] = None
    circuit_status: Optional[int] = None
    assigned_cookie: Optional[bytes] = None
    remote_end_id: Optional[bytes] = None
    tx_connect_speed: Optional[int] = None
    rx_connect_speed: Optional[int] = None


def parse_session_fields(avps: List[AVP]) -> SessionFields:
    """Extract session-level fields from an AVP list.

    Enforces that Local Session ID is present (mandatory in every
    session-level message).  All other fields are optional at the
    parser level; callers enforce message-type-specific requirements.
    """
    local_avp  = find_avp(avps, AttrType.LOCAL_SESSION_ID)
    if local_avp is None:
        raise ValueError("session message missing Local Session ID AVP")

    remote_avp = find_avp(avps, AttrType.REMOTE_SESSION_ID)
    pw_avp     = find_avp(avps, AttrType.PSEUDOWIRE_TYPE)
    l2s_avp    = find_avp(avps, AttrType.L2_SPECIFIC_SUBLAYER)
    ds_avp     = find_avp(avps, AttrType.DATA_SEQUENCING)
    cs_avp     = find_avp(avps, AttrType.CIRCUIT_STATUS)
    cookie_avp = find_avp(avps, AttrType.ASSIGNED_COOKIE)
    eid_avp    = find_avp(avps, AttrType.REMOTE_END_ID)
    tx_avp     = find_avp(avps, AttrType.TX_CONNECT_SPEED)
    rx_avp     = find_avp(avps, AttrType.RX_CONNECT_SPEED)

    return SessionFields(
        local_sid=decode_local_session_id(local_avp),
        remote_sid=decode_remote_session_id(remote_avp) if remote_avp else 0,
        pseudowire_type=decode_pseudowire_type(pw_avp) if pw_avp else None,
        l2_specific_sublayer=decode_l2_specific_sublayer(l2s_avp) if l2s_avp else None,
        data_sequencing=decode_data_sequencing(ds_avp) if ds_avp else None,
        circuit_status=decode_circuit_status(cs_avp) if cs_avp else None,
        assigned_cookie=decode_assigned_cookie(cookie_avp) if cookie_avp else None,
        remote_end_id=decode_remote_end_id(eid_avp) if eid_avp else None,
        tx_connect_speed=decode_tx_connect_speed(tx_avp) if tx_avp else None,
        rx_connect_speed=decode_rx_connect_speed(rx_avp) if rx_avp else None,
    )


@dataclass
class CdnFields:
    """Fields extracted from a CDN message."""

    local_sid:     int
    remote_sid:    int
    result_code:   int
    error_code:    Optional[int]
    error_message: bytes


def parse_cdn_fields(avps: List[AVP]) -> CdnFields:
    local_avp  = find_avp(avps, AttrType.LOCAL_SESSION_ID)
    remote_avp = find_avp(avps, AttrType.REMOTE_SESSION_ID)
    rc_avp     = find_avp(avps, AttrType.RESULT_CODE)
    if local_avp is None:
        raise ValueError("CDN missing Local Session ID AVP")
    if rc_avp is None:
        raise ValueError("CDN missing Result Code AVP")
    result, error, err_msg = decode_result_code(rc_avp)
    return CdnFields(
        local_sid=decode_local_session_id(local_avp),
        remote_sid=decode_remote_session_id(remote_avp) if remote_avp else 0,
        result_code=result,
        error_code=error,
        error_message=err_msg,
    )


# ---------------------------------------------------------------------------
# Parsed-message dataclasses for the typed decoders
# ---------------------------------------------------------------------------

@dataclass
class SccrxFields:
    """Fields extracted from a parsed SCCRQ or SCCRP.

    Both messages carry the same field set per RFC 3931 §6.1/§6.2.
    """

    router_id: int
    assigned_ccid: int
    host_name: bytes
    pw_capabilities: List[int]
    receive_window: Optional[int]
    auth_nonce: Optional[bytes]


def parse_sccrx_fields(avps: List[AVP]) -> SccrxFields:
    """Extract common SCCRQ/SCCRP fields from an AVP list.

    Raises ValueError if any of the mandatory AVPs (Router ID,
    Assigned CCID, Host Name, PW Capabilities) is missing or malformed.
    """
    router_id_avp = find_avp(avps, AttrType.ROUTER_ID)
    ccid_avp      = find_avp(avps, AttrType.ASSIGNED_CONTROL_CONNECTION_ID)
    host_avp      = find_avp(avps, AttrType.HOST_NAME)
    pwc_avp       = find_avp(avps, AttrType.PSEUDOWIRE_CAPABILITIES_LIST)
    if router_id_avp is None:
        raise ValueError("SCCRQ/SCCRP missing Router ID AVP")
    if ccid_avp is None:
        raise ValueError("SCCRQ/SCCRP missing Assigned Control Connection ID AVP")
    if host_avp is None:
        raise ValueError("SCCRQ/SCCRP missing Host Name AVP")
    if pwc_avp is None:
        raise ValueError("SCCRQ/SCCRP missing Pseudowire Capabilities List AVP")

    win_avp = find_avp(avps, AttrType.RECEIVE_WINDOW_SIZE)
    nonce_avp = find_avp(avps, AttrType.CTL_MSG_AUTH_NONCE)

    return SccrxFields(
        router_id=decode_router_id(router_id_avp),
        assigned_ccid=decode_assigned_control_connection_id(ccid_avp),
        host_name=decode_host_name(host_avp),
        pw_capabilities=decode_pseudowire_capabilities_list(pwc_avp),
        receive_window=decode_receive_window_size(win_avp) if win_avp else None,
        auth_nonce=decode_control_message_auth_nonce(nonce_avp) if nonce_avp else None,
    )


@dataclass
class StopCcnFields:
    """Fields extracted from a StopCCN."""

    assigned_ccid: int
    result_code: int
    error_code: Optional[int]
    error_message: bytes


def parse_stopccn_fields(avps: List[AVP]) -> StopCcnFields:
    ccid_avp = find_avp(avps, AttrType.ASSIGNED_CONTROL_CONNECTION_ID)
    rc_avp   = find_avp(avps, AttrType.RESULT_CODE)
    if ccid_avp is None:
        raise ValueError("StopCCN missing Assigned Control Connection ID AVP")
    if rc_avp is None:
        raise ValueError("StopCCN missing Result Code AVP")
    result, error, err_msg = decode_result_code(rc_avp)
    return StopCcnFields(
        assigned_ccid=decode_assigned_control_connection_id(ccid_avp),
        result_code=result,
        error_code=error,
        error_message=err_msg,
    )


def get_message_type(avps: List[AVP]) -> Optional[int]:
    """Return the numeric Message Type from an AVP list, or None if absent."""
    mt = find_avp(avps, AttrType.MESSAGE_TYPE)
    if mt is None:
        return None
    return decode_message_type(mt)


# ---------------------------------------------------------------------------
# Signing / verification for authenticated control messages
# (RFC 3931 §5.4.3 Message Digest AVP + §4.4 Message Integrity Check)
# ---------------------------------------------------------------------------
#
# The signing model:
#   1. Caller assembles the AVP list for a message (excluding Message Digest).
#   2. Caller wraps it in a ControlMessage with the right Ns/Nr/CCID.
#   3. Caller invokes ``msg.encode_signed(hash_type, shared_secret)`` — which
#      appends a Message Digest AVP with a zeroed digest, encodes, computes
#      HMAC over the encoded bytes, and patches the digest in place.
#   4. The returned wire bytes are ready for the transport layer.
#
# Verification is the mirror image — see ``ControlMessage.decode_and_verify``.
# The digest position is discovered at decode time by parsing the AVPs, so
# a peer's Message Digest AVP can be anywhere in the AVP list; we don't
# require it to be last.

def _encode_signed_impl(
    ccid: int,
    ns: int,
    nr: int,
    avps: List[AVP],
    hash_type: DigestHash | int,
    shared_secret: bytes,
) -> bytes:
    """Build a signed ControlMessage's wire bytes.

    Extracted as a free function so ``ControlMessage.encode_signed`` can
    stay a thin wrapper without an import cycle to auth.py.
    """
    from .auth import compute_message_digest, digest_size

    dsize = digest_size(hash_type)
    placeholder = build_message_digest(hash_type, b"\x00" * dsize)
    signed_avps = list(avps) + [placeholder]

    stub = ControlMessage(
        control_connection_id=ccid,
        ns=ns,
        nr=nr,
        avps=signed_avps,
    )
    encoded_zeroed = stub.encode()
    digest = compute_message_digest(hash_type, shared_secret, encoded_zeroed)
    # The Message Digest AVP is last, so the digest bytes are the final
    # dsize bytes of the encoded output.  Patch in place.
    return encoded_zeroed[:-dsize] + digest


def _decode_and_verify_impl(
    buf: bytes,
    hash_type: DigestHash | int,
    shared_secret: bytes,
) -> "ControlMessage":
    from .auth import digest_size, verify_message_digest

    msg = ControlMessage.decode(buf)
    md_avp = find_avp(msg.avps, AttrType.MESSAGE_DIGEST)
    if md_avp is None:
        raise ValueError("received message missing Message Digest AVP")
    if len(md_avp.value) < 1:
        raise ValueError("Message Digest AVP value too short (missing hash type)")

    received_hash_type = md_avp.value[0]
    if received_hash_type != int(hash_type):
        raise ValueError(
            f"hash type mismatch: got {received_hash_type}, expected "
            f"{int(hash_type)}"
        )
    dsize = digest_size(hash_type)
    if len(md_avp.value) != 1 + dsize:
        raise ValueError(
            f"Message Digest AVP wrong size for hash type {received_hash_type}: "
            f"got {len(md_avp.value)}, expected {1 + dsize}"
        )
    received_digest = md_avp.value[1:]

    # Rebuild the "digest-zeroed" wire bytes by re-encoding the same
    # AVPs with the digest replaced by zeros.  Requires deterministic
    # AVP encoding (we have it: no reserved-bit randomisation, no
    # optional padding elsewhere).
    zeroed_avps: List[AVP] = []
    for a in msg.avps:
        if a.vendor_id == VENDOR_IETF \
                and a.attribute_type == int(AttrType.MESSAGE_DIGEST):
            zeroed_avps.append(build_message_digest(hash_type, b"\x00" * dsize))
        else:
            zeroed_avps.append(a)
    zeroed_msg = ControlMessage(
        control_connection_id=msg.control_connection_id,
        ns=msg.ns,
        nr=msg.nr,
        avps=zeroed_avps,
    )
    zeroed_bytes = zeroed_msg.encode()
    if not verify_message_digest(
        hash_type, shared_secret, zeroed_bytes, received_digest
    ):
        raise ValueError("message digest verification failed")
    return msg


# Bolt the signing methods onto ControlMessage after definition to keep
# imports cycle-free (auth.py imports from avp.py; we don't want messages.py
# to be a hard dep of auth.py just because ControlMessage carries an
# encode_signed method).
def _install_signing_methods() -> None:
    def encode_signed(
        self: "ControlMessage",
        hash_type: DigestHash | int,
        shared_secret: bytes,
    ) -> bytes:
        """Encode with a Message Digest AVP appended and HMAC filled in."""
        return _encode_signed_impl(
            self.control_connection_id, self.ns, self.nr, self.avps,
            hash_type, shared_secret,
        )
    ControlMessage.encode_signed = encode_signed  # type: ignore[attr-defined]

    @classmethod   # type: ignore[misc]
    def decode_and_verify(
        cls,
        buf: bytes,
        hash_type: DigestHash | int,
        shared_secret: bytes,
    ) -> "ControlMessage":
        """Decode and verify the Message Digest AVP.

        Raises ValueError with 'verification failed' on any mismatch.
        """
        return _decode_and_verify_impl(buf, hash_type, shared_secret)
    ControlMessage.decode_and_verify = decode_and_verify  # type: ignore[attr-defined]


_install_signing_methods()
