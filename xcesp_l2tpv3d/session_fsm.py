"""L2TPv3 session state machine (RFC 3931 §7.4).

One SessionFSM instance per configured pseudowire.  Same design
principles as ``TunnelFSM``: time-agnostic, no I/O, callers drive it
via ``on_*`` events and execute the returned ``Action`` objects.

## States

    IDLE                  — no session in flight.
    WAIT_SESSION_REPLY    — sent ICRQ, awaiting ICRP (initiator role).
    WAIT_SESSION_CONNECT  — sent ICRP, awaiting ICCN (responder role).
    ESTABLISHED           — session up, dataplane netdev created.
    SEND_CDN              — CDN sent, awaiting cleanup.

## Session-ID exchange (§6.6–§6.9)

  * Initiator sends ICRQ with its **Local Session ID** only (it doesn't
    know the peer's yet).
  * Responder replies with ICRP containing ITS Local Session ID plus
    a Remote Session ID = the initiator's Local Session ID (so the
    initiator can match this reply to its earlier ICRQ).
  * Initiator sends ICCN with both — full binding on both sides.

## Actions

    SendMessage(avps)          — hand to transport for reliable delivery.
    DataplaneAddSession(params) — kernel add-session (netdev born).
    DataplaneDelSession(local_ccid, local_sid) — netdev + kernel state gone.
    SessionEstablished         — signal to outer world.
    SessionTornDown(reason)    — signal to outer world.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import List, Optional

from .avp import AVP, MessageType, PseudowireType
from .dataplane import SessionParams
from .messages import (
    ControlMessage,
    build_cdn_avps,
    build_iccn_avps,
    build_icrp_avps,
    build_icrq_avps,
    get_message_type,
    parse_cdn_fields,
    parse_session_fields,
)
from .tunnel_fsm import ResultCode


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

class SessionState(enum.Enum):
    IDLE                  = "idle"
    WAIT_SESSION_REPLY    = "wait-session-reply"
    WAIT_SESSION_CONNECT  = "wait-session-connect"
    ESTABLISHED           = "established"
    SEND_CDN              = "send-cdn"


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SendMessage:
    """Session-level SendMessage (semantically identical to the tunnel's
    SendMessage, kept separate so the two FSMs can grow independently)."""
    avps: List[AVP]


@dataclass(frozen=True)
class DataplaneAddSession:
    """Ask the dataplane to create the kernel session + netdev."""
    params: SessionParams


@dataclass(frozen=True)
class DataplaneDelSession:
    """Ask the dataplane to tear down the kernel session + netdev."""
    local_ccid: int
    local_sid: int


@dataclass(frozen=True)
class SessionEstablished:
    """The session just came up (ICCN sent or received)."""
    pass


@dataclass(frozen=True)
class SessionTornDown:
    """The session just went down."""
    reason: str


Action = object


# ---------------------------------------------------------------------------
# Per-session config
# ---------------------------------------------------------------------------

@dataclass
class SessionConfig:
    """Per-session config — everything the FSM needs to build ICRQ/etc.

    The initiator flag decides whether ``on_tunnel_established`` triggers
    an outbound ICRQ or waits for the peer's.
    """

    name:              str
    local_ccid:        int              # inherited from tunnel; kept here so
                                         # actions can reference it without a
                                         # back-pointer to the TunnelFSM
    local_sid:         int              # our assigned Local Session ID
    pseudowire_type:   int              # e.g. PseudowireType.ETHERNET
    initiator:         bool = True

    # Optional negotiation knobs — sane defaults for Ethernet PW.
    l2_specific_sublayer: int = 1       # 1 = default, 0 = none
    data_sequencing:      int = 0       # 0 = disabled
    circuit_status:       int = 0b11    # bit0=up, bit1=new
    cookie:               Optional[bytes] = None       # our cookie
    peer_cookie:          Optional[bytes] = None       # peer's cookie (learned or configured)
    tx_connect_speed:     Optional[int] = None
    rx_connect_speed:     Optional[int] = None

    # Local netdev naming — the daemon derives one if not set.
    ifname:               Optional[str] = None


# ---------------------------------------------------------------------------
# The FSM
# ---------------------------------------------------------------------------

class SessionFSM:
    """Per-session state machine.  See module docstring for design."""

    def __init__(self, config: SessionConfig) -> None:
        self.config = config
        self._state: SessionState = SessionState.IDLE
        # Peer's Local Session ID — our Remote Session ID.  Filled in on
        # ICRQ (responder) or ICRP (initiator).
        self.peer_sid: int = 0
        # Peer's cookie learned from the negotiation, if any.
        self.peer_cookie_learned: Optional[bytes] = None

    # ---- state --------------------------------------------------------

    @property
    def state(self) -> SessionState:
        return self._state

    def _transition(self, new_state: SessionState) -> None:
        self._state = new_state

    # ---- lifecycle events --------------------------------------------

    def on_tunnel_established(self) -> List[Action]:
        """Called by TunnelFSM.Established → session should initiate.

        No-op for responder-role sessions; they wait for peer's ICRQ.
        """
        if not self.config.initiator:
            return []
        if self._state != SessionState.IDLE:
            return []
        avps = build_icrq_avps(
            local_sid=self.config.local_sid,
            pseudowire_type=self.config.pseudowire_type,
            remote_end_id=self.config.name,
            l2_specific_sublayer=self.config.l2_specific_sublayer,
            data_sequencing=self.config.data_sequencing,
            circuit_status=self.config.circuit_status,
            assigned_cookie=self.config.cookie,
            tx_connect_speed=self.config.tx_connect_speed,
            rx_connect_speed=self.config.rx_connect_speed,
        )
        self._transition(SessionState.WAIT_SESSION_REPLY)
        return [SendMessage(avps)]

    def on_local_close(
        self, result_code: int = int(ResultCode.ADMIN_SHUTDOWN)
    ) -> List[Action]:
        """Config-driven session teardown."""
        if self._state in (SessionState.IDLE, SessionState.SEND_CDN):
            return []
        avps = build_cdn_avps(
            local_sid=self.config.local_sid,
            remote_sid=self.peer_sid,
            result_code=result_code,
        )
        prior = self._state
        self._transition(SessionState.SEND_CDN)
        actions: List[Action] = [SendMessage(avps)]
        if prior == SessionState.ESTABLISHED:
            actions.append(DataplaneDelSession(
                local_ccid=self.config.local_ccid,
                local_sid=self.config.local_sid,
            ))
            actions.append(SessionTornDown("local close"))
        return actions

    def on_tunnel_down(self) -> List[Action]:
        """The parent tunnel went down; all sessions in it are dead too."""
        if self._state == SessionState.IDLE:
            return []
        prior = self._state
        self._transition(SessionState.IDLE)
        actions: List[Action] = []
        if prior == SessionState.ESTABLISHED:
            actions.append(DataplaneDelSession(
                local_ccid=self.config.local_ccid,
                local_sid=self.config.local_sid,
            ))
            actions.append(SessionTornDown("tunnel down"))
        return actions

    # ---- inbound-message dispatch ------------------------------------

    def on_message(self, msg: ControlMessage) -> List[Action]:
        mt = get_message_type(msg.avps)
        if mt == int(MessageType.ICRQ):
            return self._on_icrq(msg)
        if mt == int(MessageType.ICRP):
            return self._on_icrp(msg)
        if mt == int(MessageType.ICCN):
            return self._on_iccn(msg)
        if mt == int(MessageType.CDN):
            return self._on_cdn(msg)
        return []

    # ---- per-message handlers ----------------------------------------

    def _on_icrq(self, msg: ControlMessage) -> List[Action]:
        """Responder role: peer wants to open a session with us."""
        if self._state != SessionState.IDLE:
            # Duplicate / race: RFC 3931 says the responder handling
            # depends on collision-resolution tiebreaker.  0.4.0's
            # simple model rejects with CDN.
            return self._teardown_cdn("received ICRQ in non-IDLE state",
                                       result_code=int(ResultCode.GENERAL_ERROR))
        try:
            fields = parse_session_fields(msg.avps)
        except ValueError:
            return []   # malformed → drop
        self.peer_sid = fields.local_sid
        if fields.assigned_cookie is not None:
            self.peer_cookie_learned = fields.assigned_cookie

        avps = build_icrp_avps(
            local_sid=self.config.local_sid,
            remote_sid=self.peer_sid,
            l2_specific_sublayer=self.config.l2_specific_sublayer,
            data_sequencing=self.config.data_sequencing,
            circuit_status=self.config.circuit_status,
            assigned_cookie=self.config.cookie,
            tx_connect_speed=self.config.tx_connect_speed,
            rx_connect_speed=self.config.rx_connect_speed,
        )
        self._transition(SessionState.WAIT_SESSION_CONNECT)
        return [SendMessage(avps)]

    def _on_icrp(self, msg: ControlMessage) -> List[Action]:
        """Initiator role: peer replied to our ICRQ."""
        if self._state != SessionState.WAIT_SESSION_REPLY:
            return []
        try:
            fields = parse_session_fields(msg.avps)
        except ValueError:
            return self._teardown_cdn("malformed ICRP")
        self.peer_sid = fields.local_sid
        if fields.assigned_cookie is not None:
            self.peer_cookie_learned = fields.assigned_cookie

        avps = build_iccn_avps(
            local_sid=self.config.local_sid,
            remote_sid=self.peer_sid,
            l2_specific_sublayer=self.config.l2_specific_sublayer,
            data_sequencing=self.config.data_sequencing,
            circuit_status=self.config.circuit_status,
            tx_connect_speed=self.config.tx_connect_speed,
            rx_connect_speed=self.config.rx_connect_speed,
        )
        dp_params = self._dataplane_params()
        self._transition(SessionState.ESTABLISHED)
        return [
            SendMessage(avps),
            DataplaneAddSession(dp_params),
            SessionEstablished(),
        ]

    def _on_iccn(self, msg: ControlMessage) -> List[Action]:
        """Responder role: initiator finalised — session up."""
        if self._state != SessionState.WAIT_SESSION_CONNECT:
            return []
        # Confirm peer_sid from the ICCN's Local Session ID (still their SID).
        try:
            fields = parse_session_fields(msg.avps)
            if fields.local_sid != self.peer_sid:
                # Peer changed their SID mid-handshake — reject.
                return self._teardown_cdn("ICCN Local SID mismatch")
        except ValueError:
            return self._teardown_cdn("malformed ICCN")

        dp_params = self._dataplane_params()
        self._transition(SessionState.ESTABLISHED)
        return [
            DataplaneAddSession(dp_params),
            SessionEstablished(),
        ]

    def _on_cdn(self, msg: ControlMessage) -> List[Action]:
        """Peer is tearing down the session."""
        if self._state == SessionState.IDLE:
            return []
        try:
            fields = parse_cdn_fields(msg.avps)
            reason = (
                f"peer CDN, result={fields.result_code}"
                + (f" error={fields.error_code}" if fields.error_code else "")
            )
        except ValueError:
            reason = "peer CDN (unparseable)"
        prior = self._state
        self._transition(SessionState.IDLE)
        actions: List[Action] = []
        if prior == SessionState.ESTABLISHED:
            actions.append(DataplaneDelSession(
                local_ccid=self.config.local_ccid,
                local_sid=self.config.local_sid,
            ))
            actions.append(SessionTornDown(reason))
        return actions

    # ---- internals ---------------------------------------------------

    def _dataplane_params(self) -> SessionParams:
        """Build the SessionParams the dataplane needs.

        Uses the peer's cookie if we learned one during negotiation,
        otherwise the one configured.  Default netdev name is
        ``l2tpeth-<local_ccid>-<local_sid>``.
        """
        ifname = self.config.ifname or \
            f"l2tpeth-{self.config.local_ccid}-{self.config.local_sid}"
        return SessionParams(
            local_ccid=self.config.local_ccid,
            local_sid=self.config.local_sid,
            remote_sid=self.peer_sid,
            ifname=ifname,
            cookie=self.config.cookie,
            peer_cookie=self.peer_cookie_learned or self.config.peer_cookie,
            l2spec_type=("default" if self.config.l2_specific_sublayer == 1
                         else "none"),
            pseudowire_type=(
                "ethernet"
                if self.config.pseudowire_type == int(PseudowireType.ETHERNET)
                else "ethernet-vlan"
                if self.config.pseudowire_type == int(PseudowireType.ETHERNET_VLAN)
                else "ethernet"
            ),
        )

    def _teardown_cdn(
        self, reason: str,
        result_code: int = int(ResultCode.GENERAL_ERROR),
    ) -> List[Action]:
        avps = build_cdn_avps(
            local_sid=self.config.local_sid,
            remote_sid=self.peer_sid,
            result_code=result_code,
        )
        prior = self._state
        self._transition(SessionState.SEND_CDN)
        actions: List[Action] = [SendMessage(avps)]
        if prior == SessionState.ESTABLISHED:
            actions.append(DataplaneDelSession(
                local_ccid=self.config.local_ccid,
                local_sid=self.config.local_sid,
            ))
            actions.append(SessionTornDown(reason))
        return actions
