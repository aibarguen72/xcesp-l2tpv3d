"""L2TPv3 tunnel state machine (RFC 3931 §7.2.1).

The tunnel FSM tracks control-connection lifecycle from IDLE through
tunnel establishment to teardown.  It does NOT own sockets, timers,
or coroutines — callers drive it by feeding events (received message,
local open/close, HELLO-timer fire, peer-dead notification) and
executing the ``Action`` objects the FSM returns.  This matches the
design of ``transport.Peer``: everything time- and I/O-agnostic, so
tests exercise the state matrix without asyncio or real sleeps.

## States

    IDLE           — no control connection.
    WAIT_CTL_REPLY — we sent SCCRQ, waiting for the peer's SCCRP.
    WAIT_CTL_CONN  — we sent SCCRP, waiting for the peer's SCCCN.
    ESTABLISHED    — tunnel is up; HELLO exchange keeps it alive.
    SEND_STOPCCN   — StopCCN sent, awaiting cleanup / transport drain.

## Events

    LOCAL_OPEN     — config says "bring this tunnel up".
    LOCAL_CLOSE    — config or operator says "tear this tunnel down".
    RX_SCCRQ       — peer initiated (they want us to be responder).
    RX_SCCRP       — peer replied to our SCCRQ.
    RX_SCCCN       — peer confirmed the tunnel established.
    RX_HELLO       — peer's keepalive (no state change; refreshes idle).
    RX_STOPCCN     — peer is tearing down.
    HELLO_TIMER    — our HELLO periodic timer fired.
    PEER_DEAD      — transport layer exhausted retransmit budget.

## Actions (returned to caller)

    SendMessage(avps)     — hand these AVPs to the transport for delivery.
    SetHelloTimer(sec)    — arm the HELLO periodic timer at ``sec``.
    ClearHelloTimer()     — cancel any pending HELLO timer.
    Established()         — the tunnel just came up.
    TornDown(reason)      — the tunnel just went down.

The caller is expected to consume actions in order.  For example,
``on_message(sccrp)`` in state WAIT_CTL_REPLY typically returns
``[SendMessage(scccn_avps), Established(), SetHelloTimer(60)]`` —
the caller must send SCCCN before treating the tunnel as up.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import List, Optional

from .avp import AVP, MessageType, PseudowireType
from .messages import (
    ControlMessage,
    build_hello_avps,
    build_sccrp_avps,
    build_sccrq_avps,
    build_scccn_avps,
    build_stopccn_avps,
    get_message_type,
    parse_sccrx_fields,
    parse_stopccn_fields,
)


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

class TunnelState(enum.Enum):
    IDLE           = "idle"
    WAIT_CTL_REPLY = "wait-ctl-reply"    # sent SCCRQ, awaiting SCCRP
    WAIT_CTL_CONN  = "wait-ctl-conn"     # sent SCCRP, awaiting SCCCN
    ESTABLISHED    = "established"
    SEND_STOPCCN   = "send-stopccn"      # StopCCN sent, cleaning up


# ---------------------------------------------------------------------------
# Result codes (RFC 3931 §5.4.2, subset used at 0.3.0)
# ---------------------------------------------------------------------------

class ResultCode(enum.IntEnum):
    #: General-request result codes for StopCCN
    RESERVED               = 0
    GENERAL_REQUEST        = 1     # "General request to clear control connection"
    GENERAL_ERROR          = 2     # "General error"
    ADMIN_SHUTDOWN         = 6     # "Local shutdown by administrator"
    NORMAL_STOP            = 0     # Some texts use 0 for normal — we use GENERAL_REQUEST.


# ---------------------------------------------------------------------------
# Actions returned by the FSM
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SendMessage:
    """Hand these AVPs to the transport for reliable delivery."""
    avps: List[AVP]


@dataclass(frozen=True)
class SetHelloTimer:
    """Arm the HELLO periodic timer to fire in ``seconds``."""
    seconds: float


@dataclass(frozen=True)
class ClearHelloTimer:
    """Cancel any pending HELLO timer."""
    pass


@dataclass(frozen=True)
class Established:
    """The tunnel just became ESTABLISHED — inform the outer world."""
    pass


@dataclass(frozen=True)
class TornDown:
    """The tunnel just went down — inform the outer world."""
    reason: str


Action = object   # union type; kept loose since dataclasses aren't a proper union


# ---------------------------------------------------------------------------
# Config carried by the FSM (immutable after init)
# ---------------------------------------------------------------------------

@dataclass
class TunnelConfig:
    """Per-tunnel config: everything the FSM needs to build outgoing messages.

    The Peer/transport-layer things (retransmit interval, receive window
    for the transport layer) live on ``transport.Peer``.  What lives
    here is protocol-visible config — AVP payloads.
    """

    host_name: str
    router_id: int
    local_ccid: int              # our own Assigned Control Connection ID
    pw_capabilities: List[int] = field(
        default_factory=lambda: [int(PseudowireType.ETHERNET),
                                 int(PseudowireType.ETHERNET_VLAN)]
    )
    receive_window: int = 4
    hello_interval: float = 60.0
    vendor_name: Optional[str] = None
    firmware_revision: Optional[int] = None


# ---------------------------------------------------------------------------
# The FSM itself
# ---------------------------------------------------------------------------

class TunnelFSM:
    """Per-tunnel state machine.

    Each configured remote tunnel gets its own instance.  Callers
    drive it by invoking the ``on_*`` methods and executing the
    returned action list against the transport layer.
    """

    def __init__(self, config: TunnelConfig) -> None:
        self.config = config
        self._state: TunnelState = TunnelState.IDLE
        # Filled in when we learn the peer's Assigned CCID (from SCCRQ or SCCRP).
        self.peer_ccid: int = 0
        self.peer_host_name: bytes = b""
        self.peer_router_id: int = 0

    # ---- state --------------------------------------------------------

    @property
    def state(self) -> TunnelState:
        return self._state

    def _transition(self, new_state: TunnelState) -> None:
        self._state = new_state

    # ---- event: local open --------------------------------------------

    def on_local_open(self) -> List[Action]:
        """Config-driven: bring the tunnel up (we initiate)."""
        if self._state != TunnelState.IDLE:
            return []   # already up or in flight; nothing to do
        avps = build_sccrq_avps(
            router_id=self.config.router_id,
            assigned_ccid=self.config.local_ccid,
            host_name=self.config.host_name,
            pw_capabilities=self.config.pw_capabilities,
            receive_window=self.config.receive_window,
            vendor_name=self.config.vendor_name,
            firmware_revision=self.config.firmware_revision,
        )
        self._transition(TunnelState.WAIT_CTL_REPLY)
        return [SendMessage(avps)]

    # ---- event: local close -------------------------------------------

    def on_local_close(
        self, result_code: int = int(ResultCode.ADMIN_SHUTDOWN)
    ) -> List[Action]:
        """Config-driven: tear the tunnel down (we initiate)."""
        if self._state in (TunnelState.IDLE, TunnelState.SEND_STOPCCN):
            return []
        avps = build_stopccn_avps(
            assigned_ccid=self.config.local_ccid,
            result_code=result_code,
        )
        prior = self._state
        self._transition(TunnelState.SEND_STOPCCN)
        actions: List[Action] = [SendMessage(avps), ClearHelloTimer()]
        if prior == TunnelState.ESTABLISHED:
            actions.append(TornDown("local close"))
        return actions

    # ---- event: incoming control message ------------------------------

    def on_message(self, msg: ControlMessage) -> List[Action]:
        """Dispatch on Message Type AVP.  Unknown or wrong-state
        messages are silently ignored (RFC 3931 §7.2.1 "discarded")
        — a strict-mode option can be added later.
        """
        mt = get_message_type(msg.avps)
        if mt == int(MessageType.SCCRQ):
            return self._on_sccrq(msg)
        if mt == int(MessageType.SCCRP):
            return self._on_sccrp(msg)
        if mt == int(MessageType.SCCCN):
            return self._on_scccn(msg)
        if mt == int(MessageType.HELLO):
            return self._on_hello(msg)
        if mt == int(MessageType.StopCCN):
            return self._on_stopccn(msg)
        # ICRQ / ICRP / ICCN / CDN / WEN / SLI — session-layer, deferred to 0.4.0.
        # Silently ignore for now.
        return []

    # ---- event: HELLO timer fired -------------------------------------

    def on_hello_timer(self) -> List[Action]:
        """Send HELLO + rearm the timer."""
        if self._state != TunnelState.ESTABLISHED:
            return []
        return [
            SendMessage(build_hello_avps()),
            SetHelloTimer(self.config.hello_interval),
        ]

    # ---- event: transport says peer is dead ---------------------------

    def on_peer_dead(self) -> List[Action]:
        if self._state == TunnelState.IDLE:
            return []
        prior = self._state
        self._transition(TunnelState.IDLE)
        actions: List[Action] = [ClearHelloTimer()]
        if prior == TunnelState.ESTABLISHED:
            actions.append(TornDown("peer dead (retransmit exhausted)"))
        return actions

    # ---- per-message handlers -----------------------------------------

    def _on_sccrq(self, msg: ControlMessage) -> List[Action]:
        if self._state != TunnelState.IDLE:
            # Peer restart while we thought the tunnel was up.  Simplest
            # correct handling per §7.2.1: StopCCN the old, drop to IDLE.
            # Fuller re-open handling comes in 0.6.0 for Cisco interop.
            return self._teardown_and_notify(reason="peer restart (SCCRQ in non-IDLE)")

        try:
            fields = parse_sccrx_fields(msg.avps)
        except ValueError:
            # Malformed SCCRQ — refuse without moving off IDLE.
            return []

        # Capture peer identity.
        self.peer_ccid = fields.assigned_ccid
        self.peer_host_name = fields.host_name
        self.peer_router_id = fields.router_id

        avps = build_sccrp_avps(
            router_id=self.config.router_id,
            assigned_ccid=self.config.local_ccid,
            host_name=self.config.host_name,
            pw_capabilities=self.config.pw_capabilities,
            receive_window=self.config.receive_window,
            vendor_name=self.config.vendor_name,
            firmware_revision=self.config.firmware_revision,
        )
        self._transition(TunnelState.WAIT_CTL_CONN)
        return [SendMessage(avps)]

    def _on_sccrp(self, msg: ControlMessage) -> List[Action]:
        if self._state != TunnelState.WAIT_CTL_REPLY:
            return []

        try:
            fields = parse_sccrx_fields(msg.avps)
        except ValueError:
            return self._teardown_and_notify(reason="malformed SCCRP")

        self.peer_ccid = fields.assigned_ccid
        self.peer_host_name = fields.host_name
        self.peer_router_id = fields.router_id

        avps = build_scccn_avps()
        self._transition(TunnelState.ESTABLISHED)
        return [
            SendMessage(avps),
            SetHelloTimer(self.config.hello_interval),
            Established(),
        ]

    def _on_scccn(self, msg: ControlMessage) -> List[Action]:
        if self._state != TunnelState.WAIT_CTL_CONN:
            return []
        self._transition(TunnelState.ESTABLISHED)
        return [
            SetHelloTimer(self.config.hello_interval),
            Established(),
        ]

    def _on_hello(self, msg: ControlMessage) -> List[Action]:
        # HELLO is just a keepalive.  The transport layer already
        # consumed our Nr from the header; the FSM doesn't need to
        # do anything except signal "peer is alive".  The HELLO timer
        # rearms on its own tick; we don't reset it here (that would
        # collide with our own scheduled sends).
        return []

    def _on_stopccn(self, msg: ControlMessage) -> List[Action]:
        if self._state == TunnelState.IDLE:
            return []
        try:
            fields = parse_stopccn_fields(msg.avps)
            reason = (
                f"peer StopCCN, result={fields.result_code}"
                + (f" error={fields.error_code}" if fields.error_code else "")
                + (
                    f" msg={fields.error_message.decode('utf-8', errors='replace')!r}"
                    if fields.error_message else ""
                )
            )
        except ValueError:
            reason = "peer StopCCN (unparseable)"
        prior = self._state
        self._transition(TunnelState.IDLE)
        actions: List[Action] = [ClearHelloTimer()]
        if prior == TunnelState.ESTABLISHED:
            actions.append(TornDown(reason))
        return actions

    # ---- internals ----------------------------------------------------

    def _teardown_and_notify(self, reason: str) -> List[Action]:
        """Common helper: send StopCCN, clear HELLO, notify TornDown."""
        avps = build_stopccn_avps(
            assigned_ccid=self.config.local_ccid,
            result_code=int(ResultCode.GENERAL_ERROR),
        )
        prior = self._state
        self._transition(TunnelState.SEND_STOPCCN)
        actions: List[Action] = [SendMessage(avps), ClearHelloTimer()]
        if prior == TunnelState.ESTABLISHED:
            actions.append(TornDown(reason))
        return actions
