"""Reliable-delivery transport for L2TPv3 control messages (RFC 3931 §5.8).

## Layering

The daemon has three layers stacked here:

  ┌───────────────────────────────────────────────────────────┐
  │  Peer  — per-remote reliable delivery, Ns/Nr, retransmit  │
  ├───────────────────────────────────────────────────────────┤
  │  Transport  — send(addr, bytes) / receive() → (addr, bytes)│
  │              (UdpTransport for real, LoopbackTransport    │
  │               for tests)                                   │
  ├───────────────────────────────────────────────────────────┤
  │  Wire — UDP datagrams / in-memory queue                    │
  └───────────────────────────────────────────────────────────┘

## Peer semantics

``Peer`` is a pure state machine: it maintains Ns/Nr counters, a
retransmit queue, a reorder buffer, and receive-window book-keeping,
but does not own an event loop, timers, or sockets.  Callers drive
it by calling ``send()``, ``receive()``, and ``tick(now)`` at the
appropriate moments — the outer event loop in main.py knows about
asyncio; Peer knows only about a monotonic-clock float that the
caller passes in.  This makes the state machine deterministically
testable against a mocked clock.

Retransmit behaviour follows RFC 3931 §5.8:
  * Initial RTO = ``retransmit_interval`` seconds (typ. 1s).
  * Exponential back-off: 1s, 2s, 4s, 8s, 16s (cap).
  * After ``max_retries`` unsuccessful retransmits, ``tick()``
    returns ``dead=True`` and the caller shuts the peer down.

Receive semantics (§5.8):
  * ``ns == expected_ns``:      accept, deliver AVPs, bump expected_ns.
  * ``ns < expected_ns`` (wrap-safe): duplicate — return a ZLB to re-ack.
  * ``ns > expected_ns`` and within receive window: reorder-queue.
  * ``ns`` outside receive window: drop.
"""

from __future__ import annotations

import asyncio
import socket
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

from .avp import AVP
from .messages import ControlMessage, seq_advance, seq_delta


#: RFC 3931 §5.8: exponential back-off cap on retransmit interval.
RETRANSMIT_MAX_INTERVAL = 16.0

#: Default UDP port for L2TP control (IANA).
L2TP_UDP_PORT = 1701


SockAddr = Tuple[str, int]   # (host, port)


# ---------------------------------------------------------------------------
# Transport abstract + implementations
# ---------------------------------------------------------------------------

class Transport(ABC):
    """Bidirectional bytes-in/bytes-out with a remote address."""

    @abstractmethod
    def send(self, addr: SockAddr, buf: bytes) -> None:
        """Send ``buf`` to ``addr``.  Non-blocking, best-effort."""

    @abstractmethod
    async def receive(self) -> Tuple[SockAddr, bytes]:
        """Await the next inbound datagram, return ``(sender, buf)``."""

    @abstractmethod
    def close(self) -> None:
        """Release any underlying resources (socket, queues, ...)."""


class UdpTransport(Transport):
    """asyncio-backed UDP transport.

    Binds a single UDP socket at ``local_addr`` and demultiplexes all
    inbound datagrams into an ``asyncio.Queue`` that ``receive()``
    consumes.  Real production path.
    """

    def __init__(self) -> None:
        self._transport: Optional[asyncio.DatagramTransport] = None
        self._protocol: Optional["_UdpProto"] = None
        self._queue: "asyncio.Queue[Tuple[SockAddr, bytes]]" = asyncio.Queue()

    async def start(self, local_addr: SockAddr) -> None:
        loop = asyncio.get_running_loop()
        self._transport, self._protocol = await loop.create_datagram_endpoint(
            lambda: _UdpProto(self._queue),
            local_addr=local_addr,
            reuse_port=False,
            family=socket.AF_INET,
        )

    def send(self, addr: SockAddr, buf: bytes) -> None:
        if self._transport is None:
            raise RuntimeError("UdpTransport.send before start()")
        self._transport.sendto(buf, addr)

    async def receive(self) -> Tuple[SockAddr, bytes]:
        return await self._queue.get()

    def close(self) -> None:
        if self._transport is not None:
            self._transport.close()
            self._transport = None


class _UdpProto(asyncio.DatagramProtocol):
    def __init__(self, queue: "asyncio.Queue[Tuple[SockAddr, bytes]]") -> None:
        self._queue = queue

    def datagram_received(self, data: bytes, addr: SockAddr) -> None:  # type: ignore[override]
        # asyncio delivers addr as (host, port, ...) for IPv6; on IPv4
        # it's a 2-tuple.  Normalise to (host, port).
        self._queue.put_nowait(((addr[0], addr[1]), data))


class LoopbackTransport(Transport):
    """In-memory transport used by the pytest suite.

    A pair of ``LoopbackTransport`` instances share a single dict of
    per-address queues so a ``send()`` from A drops directly into B's
    queue.  Test harness can inject packet loss (drop), reordering
    (swap queue entries), or duplication by tampering with the shared
    fabric before ``receive()`` runs.
    """

    def __init__(
        self,
        local_addr: SockAddr,
        fabric: Optional[Dict[SockAddr, "asyncio.Queue[Tuple[SockAddr, bytes]]"]] = None,
    ) -> None:
        self.local_addr = local_addr
        self.fabric: Dict[SockAddr, "asyncio.Queue[Tuple[SockAddr, bytes]]"] = (
            fabric if fabric is not None else {}
        )
        self._queue: "asyncio.Queue[Tuple[SockAddr, bytes]]" = asyncio.Queue()
        self.fabric[local_addr] = self._queue
        # Test hooks:
        self.drop_next: int = 0   # drop this many outbound datagrams before delivering
        self.on_send = None        # optional callable(dst, buf) called BEFORE queueing

    def send(self, addr: SockAddr, buf: bytes) -> None:
        if self.drop_next > 0:
            self.drop_next -= 1
            return
        if self.on_send is not None:
            self.on_send(addr, buf)
        target = self.fabric.get(addr)
        if target is None:
            # No peer registered — silently discard, same as UDP would.
            return
        target.put_nowait((self.local_addr, buf))

    async def receive(self) -> Tuple[SockAddr, bytes]:
        return await self._queue.get()

    def close(self) -> None:
        self.fabric.pop(self.local_addr, None)


# ---------------------------------------------------------------------------
# Peer — per-remote reliable delivery state machine
# ---------------------------------------------------------------------------

@dataclass
class _PendingSend:
    """One in-flight (or awaiting-retransmit) outbound message."""

    msg: ControlMessage
    encoded: bytes            # cached to avoid re-encoding on retransmit
    sent_at: float            # monotonic clock time of last transmission
    retries: int = 0          # 0 on first send; incremented per retransmit
    next_rto: float = 1.0     # next retransmit-timer duration


class Peer:
    """Per-remote reliable-delivery state machine.

    The daemon creates one Peer per configured remote endpoint.  Peer
    is time-agnostic — callers pass a monotonic ``now`` on every
    call; tests use a mocked clock to make retransmit behaviour
    deterministic.
    """

    def __init__(
        self,
        *,
        remote_addr: SockAddr,
        local_control_connection_id: int,
        remote_control_connection_id: int = 0,
        receive_window: int = 4,
        retransmit_interval: float = 1.0,
        max_retries: int = 5,
    ) -> None:
        self.remote_addr = remote_addr
        self.local_ccid = local_control_connection_id
        # remote_ccid is 0 until SCCRP arrives; 0.3.0 sets it via a setter.
        self.remote_ccid = remote_control_connection_id
        self.receive_window = receive_window
        self.retransmit_interval = retransmit_interval
        self.max_retries = max_retries

        # Ns of the next message we will send.
        self._ns_send: int = 0
        # Ns we expect from the peer next; also the Nr we announce.
        self._nr_expect: int = 0
        # Send queue: unacked messages, in Ns order.
        self._send_queue: Deque[_PendingSend] = deque()
        # Reorder buffer: {ns → ControlMessage} for messages received
        # ahead of nr_expect (within receive window).
        self._reorder: Dict[int, ControlMessage] = {}
        # Whether the peer has been declared dead.
        self._dead: bool = False
        # Set by receive() when a duplicate arrives — a ZLB should be
        # sent to re-acknowledge.
        # (Callers process the return value; this attribute is not used
        # for internal state, just for readability.)

    # ---- properties -------------------------------------------------------

    @property
    def ns_send(self) -> int:
        """Next Ns to use for send (readonly to callers)."""
        return self._ns_send

    @property
    def nr_expect(self) -> int:
        """Ns we expect from the peer next (readonly to callers)."""
        return self._nr_expect

    @property
    def unacked(self) -> int:
        """Number of messages currently awaiting ack from the peer."""
        return len(self._send_queue)

    @property
    def dead(self) -> bool:
        """True if max_retries exceeded on any in-flight message."""
        return self._dead

    def set_remote_ccid(self, ccid: int) -> None:
        """Set the remote-assigned CCID (from SCCRP's Assigned CCID AVP).

        The daemon uses local_ccid as its own address and remote_ccid
        as the recipient address in outgoing messages.  Peers exchange
        CCID assignments during tunnel establishment; before SCCRP
        this stays at the caller-provided initial value (typically 0).
        """
        self.remote_ccid = ccid

    # ---- send side --------------------------------------------------------

    def send(self, avps: List[AVP], now: float) -> ControlMessage:
        """Queue a message for reliable delivery and return it for transmit.

        The returned ``ControlMessage`` is stamped with the current
        Ns / Nr / CCID.  The caller must immediately transmit its
        ``.encode()`` bytes via a Transport.  The message is also
        added to the retransmit queue with a deadline of
        ``now + retransmit_interval``.

        Sending a ZLB (empty ``avps``) does NOT consume an Ns and does
        NOT enter the retransmit queue — per RFC 3931 §5.8, ZLBs are
        not themselves acknowledged.  Callers should use ``ack_only()``
        below to build a ZLB instead of calling ``send([])``.
        """
        if not avps:
            raise ValueError("empty avps — use ack_only() to build a ZLB")

        msg = ControlMessage(
            control_connection_id=self.remote_ccid,
            ns=self._ns_send,
            nr=self._nr_expect,
            avps=list(avps),
        )
        self._send_queue.append(
            _PendingSend(
                msg=msg,
                encoded=msg.encode(),
                sent_at=now,
                retries=0,
                next_rto=self.retransmit_interval,
            )
        )
        self._ns_send = seq_advance(self._ns_send)
        return msg

    def ack_only(self) -> ControlMessage:
        """Build a ZLB ack — Ns is the CURRENT self._ns_send (not
        consumed), Nr is the current expected.  Not added to retransmit
        queue; sent once, no acknowledgement expected.
        """
        return ControlMessage(
            control_connection_id=self.remote_ccid,
            # A ZLB carries the current Ns (i.e. what our NEXT real
            # message would use) — per §5.8, this Ns does not advance.
            ns=self._ns_send,
            nr=self._nr_expect,
            avps=[],
        )

    # ---- receive side -----------------------------------------------------

    def receive(
        self, msg: ControlMessage
    ) -> Tuple[List[AVP], List[ControlMessage]]:
        """Process one inbound control message.

        Returns ``(deliverable_avps_batch, messages_to_send)`` where:

          * ``deliverable_avps_batch`` is a flat list of AVPs that
            became newly deliverable, in Ns order (may span multiple
            messages if a reorder gap has just been filled by the
            newly-arrived message).  Empty list on duplicate or drop.
          * ``messages_to_send`` is a list of response ``ControlMessage``
            objects the caller must transmit (typically at most one
            ZLB — sent to re-ack a duplicate or to acknowledge new
            data when the caller has no piggyback opportunity).
            The caller is free to skip the ZLB if it plans to piggyback
            the ack on an outgoing app-level message immediately.
        """
        # Accept messages addressed to our local_ccid (normal case,
        # post-handshake) or CCID=0 (initial SCCRQ before the peer has
        # learned our Assigned Control Connection ID).  Reject anything
        # else — the caller-level demux should have routed those to the
        # right Peer already.
        if msg.control_connection_id not in (0, self.local_ccid):
            return [], []

        # First: consume any acks piggy-backed on this message's Nr.
        self._consume_acks(msg.nr)

        delivered: List[AVP] = []
        response: List[ControlMessage] = []

        delta = seq_delta(msg.ns, self._nr_expect)
        if msg.is_zlb:
            # ZLB: purely an ack, no Ns consumed by peer.  Never
            # advances our nr_expect, never enters reorder buffer.
            # Already processed the Nr above; nothing else to do.
            return [], []

        if delta == 0:
            # In-order.  Deliver and advance.
            delivered.extend(msg.avps)
            self._nr_expect = seq_advance(self._nr_expect)
            # Drain any reorder-queued messages that are now in order.
            while self._nr_expect in self._reorder:
                queued = self._reorder.pop(self._nr_expect)
                delivered.extend(queued.avps)
                self._nr_expect = seq_advance(self._nr_expect)
            # Send a ZLB ack.  Higher layer may drop it if a real
            # response follows immediately.
            response.append(self.ack_only())
        elif delta < 0:
            # Duplicate — peer didn't see our previous ack.  Re-ack.
            response.append(self.ack_only())
        elif 0 < delta <= self.receive_window:
            # Out-of-order but within window.  Buffer for later.
            self._reorder[msg.ns] = msg
        else:
            # Way outside window — drop silently.  Peer will retransmit.
            pass

        return delivered, response

    # ---- retransmit timer -------------------------------------------------

    def tick(self, now: float) -> Tuple[List[bytes], bool]:
        """Fire any expired retransmit timers.

        Returns ``(datagrams_to_resend, is_dead)``.  Datagrams are
        already-encoded bytes so the caller just hands them to the
        transport.  ``is_dead`` is True if any pending message has
        exceeded ``max_retries``; the caller should tear the peer down.
        """
        if self._dead:
            return [], True

        to_resend: List[bytes] = []
        for pending in self._send_queue:
            if now - pending.sent_at < pending.next_rto:
                continue
            pending.retries += 1
            if pending.retries > self.max_retries:
                self._dead = True
                return [], True
            # Exponential back-off, capped.
            pending.next_rto = min(
                pending.next_rto * 2, RETRANSMIT_MAX_INTERVAL
            )
            pending.sent_at = now
            # Update the encoded message's Nr in case we've received
            # new stuff from the peer since first send.  Cheaper to
            # re-encode than to poke bytes in place, and cheap overall.
            pending.msg.nr = self._nr_expect
            pending.encoded = pending.msg.encode()
            to_resend.append(pending.encoded)

        return to_resend, False

    # ---- internals --------------------------------------------------------

    def _consume_acks(self, peer_nr: int) -> None:
        """Remove any queued messages whose Ns is strictly less than
        peer_nr (i.e. peer has acknowledged them).

        RFC 3931 §5.8: ``Nr`` in a received message means "I expect
        your next Ns to be Nr; I have acked everything with Ns < Nr."
        """
        while self._send_queue:
            head = self._send_queue[0]
            if seq_delta(peer_nr, head.msg.ns) > 0:
                self._send_queue.popleft()
            else:
                break
