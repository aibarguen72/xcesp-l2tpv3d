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

from xcesp_l2tpv3d.avp import DigestHash            # noqa: E402
from xcesp_l2tpv3d.config import TunnelConfigEntry  # noqa: E402
from xcesp_l2tpv3d.main import Tunnel                # noqa: E402
from xcesp_l2tpv3d.transport import (               # noqa: E402
    L2TP_UDP_PORT, LoopbackTransport,
)
from xcesp_l2tpv3d.tunnel_fsm import TunnelState    # noqa: E402


def _cfg(
    name: str, local: str, remote: str, ccid: int, host: str, rid: int,
    *,
    password: bytes | None = None,
    digest_alg: DigestHash = DigestHash.HMAC_MD5,
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
