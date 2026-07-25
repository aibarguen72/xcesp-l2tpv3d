"""Integration: two Tunnel objects handshake via LoopbackTransport.

This exercises the full 0.3.0 stack together — the config-loaded
Peer + FSM + transport + signing + HELLO timer — but without real
UDP sockets or real wall-clock sleeps.  Two Tunnel objects share an
in-memory fabric; the test drives them through SCCRQ/SCCRP/SCCCN
handshake and a HELLO exchange, both with and without shared-secret
authentication.

Real UDP is deferred to 0.4.0's dataplane spike; this pytest gives
us the confidence that the state-machine + transport composition
works end-to-end without needing root or a second netns.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from xcesp_l2tpv3d.avp import DigestHash, PseudowireType  # noqa: E402
from xcesp_l2tpv3d.config import (                         # noqa: E402
    SessionConfigEntry,
    TunnelConfigEntry,
)
from xcesp_l2tpv3d.dataplane import MockDataplane         # noqa: E402
from xcesp_l2tpv3d.main import Tunnel                     # noqa: E402
from xcesp_l2tpv3d.session_fsm import SessionState        # noqa: E402
from xcesp_l2tpv3d.transport import (                     # noqa: E402
    L2TP_UDP_PORT, LoopbackTransport,
)
from xcesp_l2tpv3d.tunnel_fsm import TunnelState          # noqa: E402


def _cfg(
    name: str, local: str, remote: str, ccid: int, host: str, rid: int,
    *,
    password: bytes | None = None,
    digest_alg: DigestHash = DigestHash.HMAC_MD5,
    sessions: list | None = None,
) -> TunnelConfigEntry:
    return TunnelConfigEntry(
        name=name,
        local_address=local,
        remote_address=remote,
        local_ccid=ccid,
        host_name=host,
        router_id=rid,
        password=password,
        digest_alg=digest_alg,
        hello_interval=0.05,          # tight for tests
        retransmit_interval=0.02,
        max_retries=8,
        sessions=sessions or [],
    )


def _session(
    name: str, sid: int, *, initiator: bool = True,
    pw: PseudowireType = PseudowireType.ETHERNET,
) -> SessionConfigEntry:
    return SessionConfigEntry(
        name=name,
        pseudowire_type=pw,
        local_sid=sid,
        initiator=initiator,
    )


async def _pump_until(
    predicate,
    fabric: dict,
    ta: LoopbackTransport,
    tb: LoopbackTransport,
    a: Tunnel,
    b: Tunnel,
    *,
    max_iterations: int = 100,
) -> None:
    """Alternately deliver any queued datagram to the addressee's Tunnel.

    Exits when ``predicate()`` returns True, or after max_iterations to
    avoid hanging on bugs.  This is a rough scheduler; real production
    uses asyncio's own event loop.
    """
    for _ in range(max_iterations):
        if predicate():
            return
        # Drain both queues in one round.
        progressed = False
        for transport, tunnel in ((ta, a), (tb, b)):
            try:
                _, buf = await asyncio.wait_for(transport.receive(), timeout=0.05)
                tunnel.handle_datagram(buf, now=asyncio.get_event_loop().time())
                progressed = True
            except asyncio.TimeoutError:
                pass
        if not progressed:
            # Nothing to receive — nudge tick to fire retransmit / HELLO
            # timers (asyncio's own call_later still needs the loop to
            # step forward).
            await asyncio.sleep(0.01)
    if not predicate():
        pytest.fail(
            f"pump_until: predicate never satisfied.  "
            f"A.state={a.state.value}  B.state={b.state.value}"
        )


@pytest.mark.asyncio
async def test_two_tunnels_reach_established_without_auth():
    fabric: dict = {}
    ta = LoopbackTransport(("A", L2TP_UDP_PORT), fabric=fabric)
    tb = LoopbackTransport(("B", L2TP_UDP_PORT), fabric=fabric)

    # A initiates by sending to ("B", 1701); B replies to ("A", 1701).
    a = Tunnel(_cfg("t", "A", "B", 100, "xcesp-A", 0x0A0A0A01),
               ta, asyncio.get_event_loop())
    b = Tunnel(_cfg("t", "B", "A", 200, "xcesp-B", 0x0B0B0B01),
               tb, asyncio.get_event_loop())

    a.start()   # SCCRQ

    await _pump_until(
        lambda: a.state == TunnelState.ESTABLISHED
             and b.state == TunnelState.ESTABLISHED,
        fabric, ta, tb, a, b,
    )
    assert a.state == TunnelState.ESTABLISHED
    assert b.state == TunnelState.ESTABLISHED


@pytest.mark.asyncio
async def test_two_tunnels_reach_established_with_shared_secret():
    """Full authenticated handshake: every message signed with HMAC-MD5,
    verified on receive."""
    fabric: dict = {}
    ta = LoopbackTransport(("A", L2TP_UDP_PORT), fabric=fabric)
    tb = LoopbackTransport(("B", L2TP_UDP_PORT), fabric=fabric)

    secret = b"shared-key-for-integration-test"
    a = Tunnel(
        _cfg("t", "A", "B", 100, "xcesp-A", 0x0A0A0A01, password=secret),
        ta, asyncio.get_event_loop(),
    )
    b = Tunnel(
        _cfg("t", "B", "A", 200, "xcesp-B", 0x0B0B0B01, password=secret),
        tb, asyncio.get_event_loop(),
    )

    a.start()

    await _pump_until(
        lambda: a.state == TunnelState.ESTABLISHED
             and b.state == TunnelState.ESTABLISHED,
        fabric, ta, tb, a, b,
    )
    assert a.state == TunnelState.ESTABLISHED
    assert b.state == TunnelState.ESTABLISHED


@pytest.mark.asyncio
async def test_wrong_shared_secret_causes_no_establishment():
    """If A signs with one secret and B verifies with a different one,
    B drops every message A sends and neither side ever reaches
    ESTABLISHED."""
    fabric: dict = {}
    ta = LoopbackTransport(("A", L2TP_UDP_PORT), fabric=fabric)
    tb = LoopbackTransport(("B", L2TP_UDP_PORT), fabric=fabric)

    a = Tunnel(
        _cfg("t", "A", "B", 100, "xcesp-A", 0x0A0A0A01, password=b"A-secret"),
        ta, asyncio.get_event_loop(),
    )
    b = Tunnel(
        _cfg("t", "B", "A", 200, "xcesp-B", 0x0B0B0B01, password=b"B-secret"),
        tb, asyncio.get_event_loop(),
    )
    a.start()

    # Give the pump plenty of iterations; both sides should stay unstuck.
    for _ in range(40):
        for transport, tunnel in ((ta, a), (tb, b)):
            try:
                _, buf = await asyncio.wait_for(transport.receive(), timeout=0.03)
                tunnel.handle_datagram(buf, now=asyncio.get_event_loop().time())
            except asyncio.TimeoutError:
                pass
        await asyncio.sleep(0.005)

    # Neither side should have made it past WAIT_CTL_REPLY / IDLE.
    assert a.state != TunnelState.ESTABLISHED
    assert b.state != TunnelState.ESTABLISHED


@pytest.mark.asyncio
async def test_hello_exchange_keeps_tunnel_up():
    """After ESTABLISHED, HELLO timer fires and peer receives HELLOs.

    Tight hello_interval (~50 ms) so we can observe several exchanges
    without the test running longer than a couple of seconds.
    """
    fabric: dict = {}
    ta = LoopbackTransport(("A", L2TP_UDP_PORT), fabric=fabric)
    tb = LoopbackTransport(("B", L2TP_UDP_PORT), fabric=fabric)

    a = Tunnel(_cfg("t", "A", "B", 100, "xcesp-A", 0x0A0A0A01),
               ta, asyncio.get_event_loop())
    b = Tunnel(_cfg("t", "B", "A", 200, "xcesp-B", 0x0B0B0B01),
               tb, asyncio.get_event_loop())
    a.start()

    await _pump_until(
        lambda: a.state == TunnelState.ESTABLISHED
             and b.state == TunnelState.ESTABLISHED,
        fabric, ta, tb, a, b,
    )

    # Let the HELLO timer fire ~5 times on each side.
    for _ in range(400):
        for transport, tunnel in ((ta, a), (tb, b)):
            try:
                _, buf = await asyncio.wait_for(transport.receive(), timeout=0.02)
                tunnel.handle_datagram(buf, now=asyncio.get_event_loop().time())
            except asyncio.TimeoutError:
                pass
        await asyncio.sleep(0.005)

    # Neither side should have collapsed.
    assert a.state == TunnelState.ESTABLISHED
    assert b.state == TunnelState.ESTABLISHED


@pytest.mark.asyncio
async def test_local_close_transitions_to_send_stopccn():
    fabric: dict = {}
    ta = LoopbackTransport(("A", L2TP_UDP_PORT), fabric=fabric)
    tb = LoopbackTransport(("B", L2TP_UDP_PORT), fabric=fabric)

    a = Tunnel(_cfg("t", "A", "B", 100, "xcesp-A", 0x0A0A0A01),
               ta, asyncio.get_event_loop())
    b = Tunnel(_cfg("t", "B", "A", 200, "xcesp-B", 0x0B0B0B01),
               tb, asyncio.get_event_loop())
    a.start()
    await _pump_until(
        lambda: a.state == TunnelState.ESTABLISHED
             and b.state == TunnelState.ESTABLISHED,
        fabric, ta, tb, a, b,
    )

    a.close()
    assert a.state == TunnelState.SEND_STOPCCN

    # B should receive StopCCN and return to IDLE.
    await _pump_until(
        lambda: b.state == TunnelState.IDLE,
        fabric, ta, tb, a, b,
    )
    assert b.state == TunnelState.IDLE


# ---------------------------------------------------------------------------
# 0.4.0: session over tunnel via LoopbackTransport + MockDataplane
# ---------------------------------------------------------------------------

def _session_states(t: Tunnel) -> dict[int, SessionState]:
    return {sid: s.state for sid, s in t._sessions_by_local.items()}


@pytest.mark.asyncio
async def test_tunnel_with_one_session_reaches_established_end_to_end():
    """A (initiator) + B (responder) tunnel + one Ethernet session
    handshake to full ESTABLISHED and both dataplanes see add_session."""
    fabric: dict = {}
    ta = LoopbackTransport(("A", L2TP_UDP_PORT), fabric=fabric)
    tb = LoopbackTransport(("B", L2TP_UDP_PORT), fabric=fabric)
    dpa = MockDataplane()
    dpb = MockDataplane()

    a_cfg = _cfg("t", "A", "B", 100, "xcesp-A", 0x0A0A0A01,
                 sessions=[_session("eth-xc", 42, initiator=True)])
    b_cfg = _cfg("t", "B", "A", 200, "xcesp-B", 0x0B0B0B01,
                 sessions=[_session("eth-xc", 99, initiator=False)])

    a = Tunnel(a_cfg, ta, asyncio.get_event_loop(), dataplane=dpa)
    b = Tunnel(b_cfg, tb, asyncio.get_event_loop(), dataplane=dpb)
    a.start()

    await _pump_until(
        lambda: (a._sessions_by_local[42].state == SessionState.ESTABLISHED
                 and b._sessions_by_local[99].state == SessionState.ESTABLISHED),
        fabric, ta, tb, a, b,
        max_iterations=200,
    )

    # Tunnels also up.
    assert a.state == TunnelState.ESTABLISHED
    assert b.state == TunnelState.ESTABLISHED

    # Both dataplanes saw add_tunnel + add_session.
    a_kinds = [k for k, _ in dpa.calls]
    b_kinds = [k for k, _ in dpb.calls]
    assert "add_tunnel" in a_kinds and "add_session" in a_kinds
    assert "add_tunnel" in b_kinds and "add_session" in b_kinds

    # Netdev names match the derived pattern.
    a_add = next(v for k, v in dpa.calls if k == "add_session")
    b_add = next(v for k, v in dpb.calls if k == "add_session")
    assert a_add.ifname == "l2tpeth-100-42"
    assert b_add.ifname == "l2tpeth-200-99"

    # Cross-check SIDs match up.
    assert a_add.local_sid == 42 and a_add.remote_sid == 99
    assert b_add.local_sid == 99 and b_add.remote_sid == 42


@pytest.mark.asyncio
async def test_session_local_close_tears_down_dataplane():
    fabric: dict = {}
    ta = LoopbackTransport(("A", L2TP_UDP_PORT), fabric=fabric)
    tb = LoopbackTransport(("B", L2TP_UDP_PORT), fabric=fabric)
    dpa = MockDataplane()
    dpb = MockDataplane()

    a_cfg = _cfg("t", "A", "B", 100, "xcesp-A", 0x0A0A0A01,
                 sessions=[_session("eth-xc", 42, initiator=True)])
    b_cfg = _cfg("t", "B", "A", 200, "xcesp-B", 0x0B0B0B01,
                 sessions=[_session("eth-xc", 99, initiator=False)])

    a = Tunnel(a_cfg, ta, asyncio.get_event_loop(), dataplane=dpa)
    b = Tunnel(b_cfg, tb, asyncio.get_event_loop(), dataplane=dpb)
    a.start()

    await _pump_until(
        lambda: (a._sessions_by_local[42].state == SessionState.ESTABLISHED
                 and b._sessions_by_local[99].state == SessionState.ESTABLISHED),
        fabric, ta, tb, a, b,
        max_iterations=200,
    )

    # A closes only the session (via local session close), not the tunnel.
    a_sess = a._sessions_by_local[42]
    for act in a_sess.on_local_close():
        a._execute_actions([act])

    # Pump until B's session goes IDLE (received CDN, cleaned up dataplane).
    await _pump_until(
        lambda: b._sessions_by_local[99].state == SessionState.IDLE,
        fabric, ta, tb, a, b,
        max_iterations=200,
    )

    # Both dataplanes should show del_session.
    assert any(k == "del_session" for k, _ in dpa.calls)
    assert any(k == "del_session" for k, _ in dpb.calls)


@pytest.mark.asyncio
async def test_tunnel_close_cascades_to_session_dataplane_teardown():
    fabric: dict = {}
    ta = LoopbackTransport(("A", L2TP_UDP_PORT), fabric=fabric)
    tb = LoopbackTransport(("B", L2TP_UDP_PORT), fabric=fabric)
    dpa = MockDataplane()
    dpb = MockDataplane()

    a_cfg = _cfg("t", "A", "B", 100, "xcesp-A", 0x0A0A0A01,
                 sessions=[_session("s", 42, initiator=True)])
    b_cfg = _cfg("t", "B", "A", 200, "xcesp-B", 0x0B0B0B01,
                 sessions=[_session("s", 99, initiator=False)])

    a = Tunnel(a_cfg, ta, asyncio.get_event_loop(), dataplane=dpa)
    b = Tunnel(b_cfg, tb, asyncio.get_event_loop(), dataplane=dpb)
    a.start()
    await _pump_until(
        lambda: (a._sessions_by_local[42].state == SessionState.ESTABLISHED
                 and b._sessions_by_local[99].state == SessionState.ESTABLISHED),
        fabric, ta, tb, a, b,
        max_iterations=200,
    )

    # A closes the tunnel — cascade should also tear down the session on A
    # AND propagate a StopCCN which tears down B's session too.
    a.close()

    await _pump_until(
        lambda: b.state == TunnelState.IDLE,
        fabric, ta, tb, a, b,
        max_iterations=200,
    )

    # A's dataplane: added session, then deleted session, then del_tunnel.
    a_kinds = [k for k, _ in dpa.calls]
    assert a_kinds.count("add_session") == 1
    assert a_kinds.count("del_session") == 1
    assert a_kinds.count("del_tunnel") == 1
    # Ordering: del_session must precede del_tunnel.
    assert a_kinds.index("del_session") < a_kinds.index("del_tunnel")
