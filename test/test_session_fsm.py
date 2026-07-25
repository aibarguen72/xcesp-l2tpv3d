"""Unit tests for xcesp_l2tpv3d.session_fsm — RFC 3931 §7.4 state matrix."""

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
    build_cdn_avps,
    build_iccn_avps,
    build_icrp_avps,
    build_icrq_avps,
    get_message_type,
)
from xcesp_l2tpv3d.session_fsm import (   # noqa: E402
    DataplaneAddSession,
    DataplaneDelSession,
    SendMessage,
    SessionConfig,
    SessionEstablished,
    SessionFSM,
    SessionState,
    SessionTornDown,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_session(**overrides) -> SessionFSM:
    cfg = SessionConfig(
        name="s1",
        local_ccid=100,
        local_sid=42,
        pseudowire_type=int(PseudowireType.ETHERNET),
        initiator=True,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return SessionFSM(cfg)


def _icrq_msg(peer_sid: int = 99, end_id: str = "s1") -> ControlMessage:
    return ControlMessage(
        control_connection_id=100, ns=1, nr=1,
        avps=build_icrq_avps(
            local_sid=peer_sid,
            pseudowire_type=PseudowireType.ETHERNET,
            remote_end_id=end_id,
        ),
    )


def _icrp_msg(peer_sid: int = 99, remote_sid: int = 42) -> ControlMessage:
    return ControlMessage(
        control_connection_id=100, ns=2, nr=2,
        avps=build_icrp_avps(local_sid=peer_sid, remote_sid=remote_sid),
    )


def _iccn_msg(peer_sid: int = 99, remote_sid: int = 42) -> ControlMessage:
    return ControlMessage(
        control_connection_id=100, ns=3, nr=3,
        avps=build_iccn_avps(local_sid=peer_sid, remote_sid=remote_sid),
    )


def _cdn_msg(peer_sid: int = 99, remote_sid: int = 42) -> ControlMessage:
    return ControlMessage(
        control_connection_id=100, ns=4, nr=4,
        avps=build_cdn_avps(
            local_sid=peer_sid, remote_sid=remote_sid, result_code=1,
        ),
    )


def _find(actions, cls):
    for a in actions:
        if isinstance(a, cls):
            return a
    return None


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

def test_initial_state_is_idle():
    s = make_session()
    assert s.state == SessionState.IDLE


# ---------------------------------------------------------------------------
# Initiator path
# ---------------------------------------------------------------------------

def test_initiator_on_tunnel_established_sends_icrq():
    s = make_session()
    actions = s.on_tunnel_established()
    assert s.state == SessionState.WAIT_SESSION_REPLY
    sm = _find(actions, SendMessage)
    assert sm is not None
    assert get_message_type(sm.avps) == int(MessageType.ICRQ)


def test_initiator_receive_icrp_sends_iccn_and_establishes():
    s = make_session()
    s.on_tunnel_established()   # → WAIT_SESSION_REPLY

    actions = s.on_message(_icrp_msg(peer_sid=99, remote_sid=42))
    assert s.state == SessionState.ESTABLISHED
    assert s.peer_sid == 99

    sm = _find(actions, SendMessage)
    assert sm is not None
    assert get_message_type(sm.avps) == int(MessageType.ICCN)

    dp = _find(actions, DataplaneAddSession)
    assert dp is not None
    assert dp.params.local_sid == 42
    assert dp.params.remote_sid == 99
    assert dp.params.local_ccid == 100
    assert dp.params.pseudowire_type == "ethernet"

    assert _find(actions, SessionEstablished) is not None


def test_initiator_malformed_icrp_triggers_cdn():
    s = make_session()
    s.on_tunnel_established()
    # ICRP missing Local Session ID
    bad = ControlMessage(
        control_connection_id=100, ns=2, nr=2,
        avps=[a for a in build_icrp_avps(local_sid=99, remote_sid=42)
              if not (a.attribute_type == 63)],   # 63 = LOCAL_SESSION_ID
    )
    actions = s.on_message(bad)
    assert s.state == SessionState.SEND_CDN
    sm = _find(actions, SendMessage)
    assert sm is not None
    assert get_message_type(sm.avps) == int(MessageType.CDN)


# ---------------------------------------------------------------------------
# Responder path
# ---------------------------------------------------------------------------

def test_responder_on_icrq_sends_icrp():
    s = make_session(initiator=False)
    actions = s.on_message(_icrq_msg(peer_sid=99))
    assert s.state == SessionState.WAIT_SESSION_CONNECT
    assert s.peer_sid == 99

    sm = _find(actions, SendMessage)
    assert sm is not None
    assert get_message_type(sm.avps) == int(MessageType.ICRP)


def test_responder_on_iccn_establishes_and_adds_dataplane():
    s = make_session(initiator=False)
    s.on_message(_icrq_msg(peer_sid=99))

    actions = s.on_message(_iccn_msg(peer_sid=99, remote_sid=42))
    assert s.state == SessionState.ESTABLISHED
    assert _find(actions, DataplaneAddSession) is not None
    assert _find(actions, SessionEstablished) is not None
    # ICCN has no reply (SendMessage should be None here).
    assert _find(actions, SendMessage) is None


def test_responder_iccn_peer_sid_mismatch_teardown():
    s = make_session(initiator=False)
    s.on_message(_icrq_msg(peer_sid=99))
    # Peer suddenly reports a different Local Session ID in ICCN.
    weird = _iccn_msg(peer_sid=101, remote_sid=42)
    actions = s.on_message(weird)
    assert s.state == SessionState.SEND_CDN
    sm = _find(actions, SendMessage)
    assert sm is not None
    assert get_message_type(sm.avps) == int(MessageType.CDN)


def test_responder_initiator_does_nothing_on_tunnel_established():
    s = make_session(initiator=False)
    actions = s.on_tunnel_established()
    assert actions == []
    assert s.state == SessionState.IDLE


# ---------------------------------------------------------------------------
# Teardown paths
# ---------------------------------------------------------------------------

def _reach_established_initiator(s: SessionFSM) -> None:
    s.on_tunnel_established()
    s.on_message(_icrp_msg())


def _reach_established_responder(s: SessionFSM) -> None:
    s.on_message(_icrq_msg())
    s.on_message(_iccn_msg())


def test_established_local_close_sends_cdn_and_teardowns_dp():
    s = make_session()
    _reach_established_initiator(s)
    actions = s.on_local_close()
    assert s.state == SessionState.SEND_CDN
    sm = _find(actions, SendMessage)
    assert sm is not None
    assert get_message_type(sm.avps) == int(MessageType.CDN)
    assert _find(actions, DataplaneDelSession) is not None
    assert _find(actions, SessionTornDown) is not None


def test_established_receive_cdn_teardowns_dp_and_signals():
    s = make_session()
    _reach_established_initiator(s)
    actions = s.on_message(_cdn_msg())
    assert s.state == SessionState.IDLE
    assert _find(actions, DataplaneDelSession) is not None
    assert _find(actions, SessionTornDown) is not None
    # No CDN reply — peer already CDN'd us.
    assert _find(actions, SendMessage) is None


def test_established_tunnel_down_teardowns_dp():
    s = make_session()
    _reach_established_initiator(s)
    actions = s.on_tunnel_down()
    assert s.state == SessionState.IDLE
    assert _find(actions, DataplaneDelSession) is not None
    assert _find(actions, SessionTornDown) is not None


def test_pre_established_tunnel_down_no_dp_teardown():
    s = make_session()
    s.on_tunnel_established()   # WAIT_SESSION_REPLY
    actions = s.on_tunnel_down()
    assert s.state == SessionState.IDLE
    # No dataplane state to delete — we never got there.
    assert _find(actions, DataplaneDelSession) is None
    assert _find(actions, SessionTornDown) is None


def test_idle_local_close_noop():
    s = make_session()
    actions = s.on_local_close()
    assert actions == []
    assert s.state == SessionState.IDLE


def test_idle_receive_cdn_noop():
    s = make_session()
    actions = s.on_message(_cdn_msg())
    assert actions == []
    assert s.state == SessionState.IDLE


def test_send_cdn_local_close_noop():
    s = make_session()
    _reach_established_initiator(s)
    s.on_local_close()
    assert s.state == SessionState.SEND_CDN
    actions = s.on_local_close()   # already tearing down
    assert actions == []


# ---------------------------------------------------------------------------
# Symmetric two-instance handshake
# ---------------------------------------------------------------------------

def test_two_sessions_handshake_to_established():
    """Wire two SessionFSMs together: A initiates, B responds.  Same
    style as test_two_fsms_handshake_to_established in tunnel_fsm tests.
    """
    a = make_session()      # initiator=True, local_sid=42
    b_cfg = SessionConfig(
        name="s1", local_ccid=100, local_sid=99,
        pseudowire_type=int(PseudowireType.ETHERNET),
        initiator=False,
    )
    b = SessionFSM(b_cfg)

    # A → ICRQ
    a_actions = a.on_tunnel_established()
    a_send = _find(a_actions, SendMessage)
    assert get_message_type(a_send.avps) == int(MessageType.ICRQ)

    icrq = ControlMessage(control_connection_id=100, ns=1, nr=1,
                          avps=a_send.avps)
    b_actions = b.on_message(icrq)
    assert b.state == SessionState.WAIT_SESSION_CONNECT
    assert b.peer_sid == 42
    b_send = _find(b_actions, SendMessage)
    assert get_message_type(b_send.avps) == int(MessageType.ICRP)

    # B → ICRP to A
    icrp = ControlMessage(control_connection_id=100, ns=2, nr=2,
                          avps=b_send.avps)
    a_actions = a.on_message(icrp)
    assert a.state == SessionState.ESTABLISHED
    assert a.peer_sid == 99
    a_send = _find(a_actions, SendMessage)
    assert get_message_type(a_send.avps) == int(MessageType.ICCN)
    assert _find(a_actions, DataplaneAddSession) is not None

    # A → ICCN to B
    iccn = ControlMessage(control_connection_id=100, ns=3, nr=3,
                          avps=a_send.avps)
    b_actions = b.on_message(iccn)
    assert b.state == SessionState.ESTABLISHED
    assert _find(b_actions, DataplaneAddSession) is not None
