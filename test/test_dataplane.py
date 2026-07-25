"""Unit tests for xcesp_l2tpv3d.dataplane — MockDataplane semantics.

IpCommandDataplane isn't unit-tested here (needs root + real kernel).
Its subprocess call construction is covered by inspection; real
verification is the 0.4.x manual/integration steps in the plan file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from xcesp_l2tpv3d.dataplane import (   # noqa: E402
    DataplaneError,
    MockDataplane,
    SessionParams,
    TunnelParams,
)


def _tp(ccid: int = 100) -> TunnelParams:
    return TunnelParams(
        local_ccid=ccid,
        remote_ccid=200,
        local_address="10.0.0.1",
        remote_address="10.0.0.2",
        encap="udp",
        udp_sport=1701,
        udp_dport=1701,
    )


def _sp(ccid: int = 100, sid: int = 42) -> SessionParams:
    return SessionParams(
        local_ccid=ccid,
        local_sid=sid,
        remote_sid=99,
        ifname=f"l2tpeth-{ccid}-{sid}",
        cookie=None, peer_cookie=None,
        l2spec_type="default",
        pseudowire_type="ethernet",
    )


# ---------------------------------------------------------------------------
# add_tunnel
# ---------------------------------------------------------------------------

def test_add_tunnel_records_call():
    dp = MockDataplane()
    p = _tp()
    dp.add_tunnel(p)
    assert dp.tunnels[100] == p
    assert dp.calls == [("add_tunnel", p)]


def test_add_tunnel_idempotent_with_same_params():
    dp = MockDataplane()
    p = _tp()
    dp.add_tunnel(p)
    dp.add_tunnel(p)   # second call — no-op
    # Only one recorded call.
    assert dp.calls.count(("add_tunnel", p)) == 1


def test_add_tunnel_rejects_conflicting_params():
    dp = MockDataplane()
    dp.add_tunnel(_tp())
    conflict = TunnelParams(
        local_ccid=100, remote_ccid=999,   # different remote_ccid
        local_address="10.0.0.1", remote_address="10.0.0.2",
    )
    with pytest.raises(DataplaneError, match="different params"):
        dp.add_tunnel(conflict)


# ---------------------------------------------------------------------------
# add_session
# ---------------------------------------------------------------------------

def test_add_session_returns_ifname_and_registers():
    dp = MockDataplane()
    dp.add_tunnel(_tp())
    sp = _sp()
    got = dp.add_session(sp)
    assert got == sp.ifname
    assert dp.sessions[(100, 42)] == sp


def test_add_session_without_tunnel_rejected():
    dp = MockDataplane()
    with pytest.raises(DataplaneError, match="unknown tunnel"):
        dp.add_session(_sp())


def test_add_session_duplicate_rejected():
    dp = MockDataplane()
    dp.add_tunnel(_tp())
    dp.add_session(_sp())
    with pytest.raises(DataplaneError, match="already exists"):
        dp.add_session(_sp())


# ---------------------------------------------------------------------------
# del_session
# ---------------------------------------------------------------------------

def test_del_session_removes_registration():
    dp = MockDataplane()
    dp.add_tunnel(_tp())
    dp.add_session(_sp())
    dp.del_session(100, 42)
    assert (100, 42) not in dp.sessions


def test_del_session_missing_is_noop():
    dp = MockDataplane()
    dp.del_session(100, 42)   # no error


# ---------------------------------------------------------------------------
# del_tunnel
# ---------------------------------------------------------------------------

def test_del_tunnel_with_live_sessions_rejected():
    dp = MockDataplane()
    dp.add_tunnel(_tp())
    dp.add_session(_sp())
    with pytest.raises(DataplaneError, match="sessions still live"):
        dp.del_tunnel(100)


def test_del_tunnel_after_sessions_gone_succeeds():
    dp = MockDataplane()
    dp.add_tunnel(_tp())
    dp.add_session(_sp())
    dp.del_session(100, 42)
    dp.del_tunnel(100)
    assert 100 not in dp.tunnels


def test_del_tunnel_missing_is_noop():
    dp = MockDataplane()
    dp.del_tunnel(999)   # no error


# ---------------------------------------------------------------------------
# Full lifecycle
# ---------------------------------------------------------------------------

def test_full_lifecycle_call_ordering():
    dp = MockDataplane()
    dp.add_tunnel(_tp())
    dp.add_session(_sp(sid=1))
    dp.add_session(_sp(sid=2))
    dp.del_session(100, 1)
    dp.del_session(100, 2)
    dp.del_tunnel(100)

    # Call kinds in the recorded order:
    kinds = [k for k, _ in dp.calls]
    assert kinds == [
        "add_tunnel",
        "add_session",
        "add_session",
        "del_session",
        "del_session",
        "del_tunnel",
    ]
