"""Unit tests for xcesp_l2tpv3d.tunnel_fsm — RFC 3931 §7.2.1 state matrix."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from xcesp_l2tpv3d.avp import MessageType, PseudowireType   # noqa: E402
from xcesp_l2tpv3d.messages import (   # noqa: E402
    ControlMessage,
    build_hello_avps,
    build_sccrp_avps,
    build_sccrq_avps,
    build_scccn_avps,
    build_stopccn_avps,
    get_message_type,
)
from xcesp_l2tpv3d.tunnel_fsm import (   # noqa: E402
    ClearHelloTimer,
    Established,
    ResultCode,
    SendMessage,
    SetHelloTimer,
    TornDown,
    TunnelConfig,
    TunnelFSM,
    TunnelState,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_fsm(**overrides) -> TunnelFSM:
    cfg = TunnelConfig(
        host_name="xcesp-A",
        router_id=0x0A0A0A01,
        local_ccid=100,
        pw_capabilities=[int(PseudowireType.ETHERNET),
                          int(PseudowireType.ETHERNET_VLAN)],
        receive_window=4,
        hello_interval=60.0,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return TunnelFSM(cfg)


def _sccrq_msg(
    router_id: int = 0x0B0B0B01,
    ccid: int = 200,
    host_name: str = "xcesp-B",
) -> ControlMessage:
    return ControlMessage(
        control_connection_id=100, ns=0, nr=0,
        avps=build_sccrq_avps(
            router_id=router_id, assigned_ccid=ccid, host_name=host_name,
            pw_capabilities=[int(PseudowireType.ETHERNET)],
        ),
    )


def _sccrp_msg(
    router_id: int = 0x0B0B0B01,
    ccid: int = 200,
    host_name: str = "xcesp-B",
) -> ControlMessage:
    return ControlMessage(
        control_connection_id=100, ns=0, nr=1,
        avps=build_sccrp_avps(
            router_id=router_id, assigned_ccid=ccid, host_name=host_name,
            pw_capabilities=[int(PseudowireType.ETHERNET)],
        ),
    )


def _scccn_msg() -> ControlMessage:
    return ControlMessage(
        control_connection_id=100, ns=1, nr=1, avps=build_scccn_avps()
    )


def _hello_msg() -> ControlMessage:
    return ControlMessage(
        control_connection_id=100, ns=2, nr=2, avps=build_hello_avps()
    )


def _stopccn_msg() -> ControlMessage:
    return ControlMessage(
        control_connection_id=100, ns=1, nr=1,
        avps=build_stopccn_avps(assigned_ccid=200, result_code=1),
    )


def _find_send(actions):
    for a in actions:
        if isinstance(a, SendMessage):
            return a
    return None


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

def test_initial_state_is_idle():
    fsm = make_fsm()
    assert fsm.state == TunnelState.IDLE


# ---------------------------------------------------------------------------
# IDLE state
# ---------------------------------------------------------------------------

def test_idle_local_open_sends_sccrq_and_transitions():
    fsm = make_fsm()
    actions = fsm.on_local_open()
    assert fsm.state == TunnelState.WAIT_CTL_REPLY
    send = _find_send(actions)
    assert send is not None
    assert get_message_type(send.avps) == int(MessageType.SCCRQ)


def test_idle_receive_sccrq_sends_sccrp_and_transitions():
    fsm = make_fsm()
    actions = fsm.on_message(_sccrq_msg())
    assert fsm.state == TunnelState.WAIT_CTL_CONN
    send = _find_send(actions)
    assert send is not None
    assert get_message_type(send.avps) == int(MessageType.SCCRP)
    # Peer identity captured.
    assert fsm.peer_ccid == 200
    assert fsm.peer_host_name == b"xcesp-B"
    assert fsm.peer_router_id == 0x0B0B0B01


def test_idle_receive_sccrp_is_ignored():
    fsm = make_fsm()
    actions = fsm.on_message(_sccrp_msg())
    assert fsm.state == TunnelState.IDLE
    assert actions == []


def test_idle_receive_scccn_is_ignored():
    fsm = make_fsm()
    actions = fsm.on_message(_scccn_msg())
    assert fsm.state == TunnelState.IDLE
    assert actions == []


def test_idle_receive_hello_is_ignored():
    fsm = make_fsm()
    actions = fsm.on_message(_hello_msg())
    assert fsm.state == TunnelState.IDLE
    assert actions == []


def test_idle_local_close_is_noop():
    fsm = make_fsm()
    actions = fsm.on_local_close()
    assert fsm.state == TunnelState.IDLE
    assert actions == []


def test_idle_peer_dead_is_noop():
    fsm = make_fsm()
    actions = fsm.on_peer_dead()
    assert fsm.state == TunnelState.IDLE
    assert actions == []


# ---------------------------------------------------------------------------
# WAIT_CTL_REPLY state (we're the initiator)
# ---------------------------------------------------------------------------

def test_wait_ctl_reply_receive_sccrp_establishes():
    fsm = make_fsm()
    fsm.on_local_open()
    assert fsm.state == TunnelState.WAIT_CTL_REPLY

    actions = fsm.on_message(_sccrp_msg())
    assert fsm.state == TunnelState.ESTABLISHED
    send = _find_send(actions)
    assert send is not None
    assert get_message_type(send.avps) == int(MessageType.SCCCN)
    # HELLO timer armed.
    assert any(isinstance(a, SetHelloTimer) for a in actions)
    # Established signalled.
    assert any(isinstance(a, Established) for a in actions)


def test_wait_ctl_reply_local_close_sends_stopccn():
    fsm = make_fsm()
    fsm.on_local_open()
    actions = fsm.on_local_close()
    assert fsm.state == TunnelState.SEND_STOPCCN
    send = _find_send(actions)
    assert send is not None
    assert get_message_type(send.avps) == int(MessageType.StopCCN)
    # TornDown should NOT fire because we were not yet ESTABLISHED.
    assert not any(isinstance(a, TornDown) for a in actions)


def test_wait_ctl_reply_peer_dead_returns_to_idle():
    fsm = make_fsm()
    fsm.on_local_open()
    actions = fsm.on_peer_dead()
    assert fsm.state == TunnelState.IDLE
    # No TornDown yet (not ESTABLISHED).
    assert not any(isinstance(a, TornDown) for a in actions)


# ---------------------------------------------------------------------------
# WAIT_CTL_CONN state (we're the responder)
# ---------------------------------------------------------------------------

def test_wait_ctl_conn_receive_scccn_establishes():
    fsm = make_fsm()
    fsm.on_message(_sccrq_msg())
    assert fsm.state == TunnelState.WAIT_CTL_CONN

    actions = fsm.on_message(_scccn_msg())
    assert fsm.state == TunnelState.ESTABLISHED
    # No SendMessage (SCCCN has no explicit reply).
    assert not any(isinstance(a, SendMessage) for a in actions)
    # HELLO timer armed.
    assert any(isinstance(a, SetHelloTimer) for a in actions)
    assert any(isinstance(a, Established) for a in actions)


def test_wait_ctl_conn_local_close_sends_stopccn():
    fsm = make_fsm()
    fsm.on_message(_sccrq_msg())
    actions = fsm.on_local_close()
    assert fsm.state == TunnelState.SEND_STOPCCN
    assert not any(isinstance(a, TornDown) for a in actions)


# ---------------------------------------------------------------------------
# ESTABLISHED state
# ---------------------------------------------------------------------------

def _reach_established_initiator(fsm: TunnelFSM) -> None:
    fsm.on_local_open()
    fsm.on_message(_sccrp_msg())
    assert fsm.state == TunnelState.ESTABLISHED


def _reach_established_responder(fsm: TunnelFSM) -> None:
    fsm.on_message(_sccrq_msg())
    fsm.on_message(_scccn_msg())
    assert fsm.state == TunnelState.ESTABLISHED


def test_established_hello_timer_sends_hello_and_rearms():
    fsm = make_fsm()
    _reach_established_initiator(fsm)
    actions = fsm.on_hello_timer()
    send = _find_send(actions)
    assert send is not None
    assert get_message_type(send.avps) == int(MessageType.HELLO)
    assert any(isinstance(a, SetHelloTimer) for a in actions)
    assert fsm.state == TunnelState.ESTABLISHED


def test_established_receive_hello_is_noop():
    fsm = make_fsm()
    _reach_established_initiator(fsm)
    actions = fsm.on_message(_hello_msg())
    assert actions == []
    assert fsm.state == TunnelState.ESTABLISHED


def test_established_receive_stopccn_tears_down():
    fsm = make_fsm()
    _reach_established_initiator(fsm)
    actions = fsm.on_message(_stopccn_msg())
    assert fsm.state == TunnelState.IDLE
    assert any(isinstance(a, TornDown) for a in actions)
    assert any(isinstance(a, ClearHelloTimer) for a in actions)


def test_established_local_close_sends_stopccn_and_torndown():
    fsm = make_fsm()
    _reach_established_initiator(fsm)
    actions = fsm.on_local_close()
    assert fsm.state == TunnelState.SEND_STOPCCN
    send = _find_send(actions)
    assert send is not None
    assert get_message_type(send.avps) == int(MessageType.StopCCN)
    assert any(isinstance(a, TornDown) for a in actions)


def test_established_peer_dead_tears_down():
    fsm = make_fsm()
    _reach_established_initiator(fsm)
    actions = fsm.on_peer_dead()
    assert fsm.state == TunnelState.IDLE
    assert any(isinstance(a, TornDown) for a in actions)


def test_established_receive_sccrq_tears_down_treated_as_peer_restart():
    """RFC 3931 §7.2.1: SCCRQ in non-IDLE means peer restarted.
    Send StopCCN and go to teardown."""
    fsm = make_fsm()
    _reach_established_initiator(fsm)
    actions = fsm.on_message(_sccrq_msg())
    assert fsm.state == TunnelState.SEND_STOPCCN
    send = _find_send(actions)
    assert send is not None
    assert get_message_type(send.avps) == int(MessageType.StopCCN)
    assert any(isinstance(a, TornDown) for a in actions)


# ---------------------------------------------------------------------------
# SEND_STOPCCN state
# ---------------------------------------------------------------------------

def test_send_stopccn_local_close_is_noop():
    fsm = make_fsm()
    _reach_established_initiator(fsm)
    fsm.on_local_close()
    assert fsm.state == TunnelState.SEND_STOPCCN
    actions = fsm.on_local_close()
    assert actions == []
    assert fsm.state == TunnelState.SEND_STOPCCN


# ---------------------------------------------------------------------------
# Malformed / edge cases
# ---------------------------------------------------------------------------

def test_malformed_sccrq_from_idle_is_dropped_silently():
    fsm = make_fsm()
    # SCCRQ missing Router ID
    from xcesp_l2tpv3d.avp import AttrType
    bad_avps = [
        a for a in build_sccrq_avps(
            router_id=1, assigned_ccid=1, host_name="x",
            pw_capabilities=[5],
        ) if a.attribute_type != int(AttrType.ROUTER_ID)
    ]
    bad = ControlMessage(control_connection_id=100, ns=0, nr=0, avps=bad_avps)
    actions = fsm.on_message(bad)
    assert actions == []
    assert fsm.state == TunnelState.IDLE


def test_unknown_message_type_is_ignored():
    fsm = make_fsm()
    _reach_established_initiator(fsm)
    # Craft a message with no Message Type AVP at all.
    weird = ControlMessage(control_connection_id=100, ns=5, nr=5, avps=[])
    actions = fsm.on_message(weird)
    assert actions == []
    assert fsm.state == TunnelState.ESTABLISHED


def test_hello_timer_in_wrong_state_is_noop():
    fsm = make_fsm()
    # IDLE state: HELLO timer fires but no SendMessage.
    actions = fsm.on_hello_timer()
    assert actions == []


# ---------------------------------------------------------------------------
# Symmetric two-instance handshake (both FSMs, no transport yet)
# ---------------------------------------------------------------------------

def test_two_fsms_handshake_to_established():
    """Wire two FSMs together in-process: A initiates, B responds.

    This is the integration-lite: verifies the FSMs' state transitions
    match up when driven by each other's produced messages.  Real
    transport (Ns/Nr) integration comes in the daemon test at 0.3.0's
    end.
    """
    a = make_fsm()          # host_name=xcesp-A, ccid=100, router_id=0x0A0A0A01
    b_cfg = TunnelConfig(
        host_name="xcesp-B", router_id=0x0B0B0B01, local_ccid=200,
        pw_capabilities=[int(PseudowireType.ETHERNET)],
    )
    b = TunnelFSM(b_cfg)

    # A initiates.
    a_actions = a.on_local_open()
    a_send = _find_send(a_actions)
    assert get_message_type(a_send.avps) == int(MessageType.SCCRQ)

    # Wrap A's SCCRQ AVPs into a message and hand to B.
    sccrq = ControlMessage(
        control_connection_id=b.config.local_ccid, ns=0, nr=0,
        avps=a_send.avps,
    )
    b_actions = b.on_message(sccrq)
    assert b.state == TunnelState.WAIT_CTL_CONN
    b_send = _find_send(b_actions)
    assert get_message_type(b_send.avps) == int(MessageType.SCCRP)

    # B's SCCRP goes to A.
    sccrp = ControlMessage(
        control_connection_id=a.config.local_ccid, ns=0, nr=1,
        avps=b_send.avps,
    )
    a_actions = a.on_message(sccrp)
    assert a.state == TunnelState.ESTABLISHED
    a_send = _find_send(a_actions)
    assert get_message_type(a_send.avps) == int(MessageType.SCCCN)
    assert any(isinstance(x, Established) for x in a_actions)

    # A's SCCCN goes to B.
    scccn = ControlMessage(
        control_connection_id=b.config.local_ccid, ns=1, nr=1,
        avps=a_send.avps,
    )
    b_actions = b.on_message(scccn)
    assert b.state == TunnelState.ESTABLISHED
    assert any(isinstance(x, Established) for x in b_actions)

    # Cross-verify peer identities.
    assert a.peer_ccid == 200
    assert b.peer_ccid == 100
    assert a.peer_host_name == b"xcesp-B"
    assert b.peer_host_name == b"xcesp-A"
