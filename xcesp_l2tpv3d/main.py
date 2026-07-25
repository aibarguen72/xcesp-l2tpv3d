"""Daemon entry point.

Composes ``config.load_file`` → per-tunnel ``Peer`` (reliable
delivery from ``transport.py``) + per-tunnel ``TunnelFSM`` (state
machine from ``tunnel_fsm.py``), binds a single UDP socket via
``UdpTransport``, dispatches inbound datagrams to the matching Peer
by Control Connection ID, and runs a periodic tick for retransmit +
HELLO timers.

Systemd integration:
  * ``Type=notify`` in the unit file expects READY=1 on stderr's
    parent socket via ``sd_notify``; this module implements a small
    stdlib-only ``sd_notify`` helper (no external dep).
  * Logs to stderr → journald consumes them by default.
"""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import os
import signal
import socket
import sys
from typing import Dict, List, Optional

from . import __version__
from . import log
from .avp import find_avp, AttrType, MessageType, VENDOR_IETF
from .config import DaemonConfig, TunnelConfigEntry, load_file, ConfigError
from .dataplane import (
    Dataplane, MockDataplane, TunnelParams,
    default_dataplane,
)
from .messages import ControlMessage, get_message_type, parse_session_fields, parse_cdn_fields
from .session_fsm import (
    DataplaneAddSession as SessDataplaneAdd,
    DataplaneDelSession as SessDataplaneDel,
    SendMessage as SessSendMessage,
    SessionConfig,
    SessionEstablished,
    SessionFSM,
    SessionState,
    SessionTornDown,
)
from .transport import (
    L2TP_UDP_PORT,
    LoopbackTransport,
    Peer,
    SockAddr,
    Transport,
    UdpTransport,
)
from .tunnel_fsm import (
    ClearHelloTimer,
    Established,
    SendMessage,
    SetHelloTimer,
    TornDown,
    TunnelConfig,
    TunnelFSM,
    TunnelState,
)


# Set of message-type ints that belong to a session (dispatched to a
# SessionFSM), not to the TunnelFSM.
_SESSION_MSG_TYPES = frozenset({
    int(MessageType.ICRQ),
    int(MessageType.ICRP),
    int(MessageType.ICCN),
    int(MessageType.CDN),
})


_LOG = log.get("main")


# ---------------------------------------------------------------------------
# Tunnel — bundles per-remote transport.Peer + TunnelFSM + config
# ---------------------------------------------------------------------------

class Tunnel:
    """Per-configured-tunnel runtime object.

    Owns the transport-layer Peer (Ns/Nr, retransmit) AND the
    protocol-layer TunnelFSM (states, message dispatch), plus a
    HELLO timer handle and a reference to the signing secret if
    authentication is configured.

    The Daemon drives this object by calling ``handle_datagram`` on
    receive and ``tick`` on a periodic timer.
    """

    def __init__(
        self,
        cfg: TunnelConfigEntry,
        transport: Transport,
        loop: asyncio.AbstractEventLoop,
        dataplane: Optional[Dataplane] = None,
    ) -> None:
        self.name = cfg.name
        self.cfg  = cfg
        self._transport = transport
        self._loop = loop
        # 0.4.0+: dataplane may be None only for the pure-FSM tests that
        # don't exercise session establishment.  Session actions crash
        # loudly if dataplane is None and a session ESTABLISHES.
        self._dataplane = dataplane

        # remote_addr is the (host, port) tuple the transport sends to.
        # 0.3.0 uses IANA L2TP UDP port 1701 on both ends.
        self._remote_addr: SockAddr = (cfg.remote_address, L2TP_UDP_PORT)

        # Reliable-delivery layer.
        self._peer = Peer(
            remote_addr=self._remote_addr,
            local_control_connection_id=cfg.local_ccid,
            remote_control_connection_id=cfg.remote_ccid,
            receive_window=cfg.receive_window,
            retransmit_interval=cfg.retransmit_interval,
            max_retries=cfg.max_retries,
        )

        # Protocol-layer FSM.
        self._fsm = TunnelFSM(TunnelConfig(
            host_name=cfg.host_name,
            router_id=cfg.router_id,
            local_ccid=cfg.local_ccid,
            receive_window=cfg.receive_window,
            hello_interval=cfg.hello_interval,
            vendor_name=cfg.vendor_name,
            firmware_revision=cfg.firmware_revision,
        ))

        # HELLO timer handle (cancellable).
        self._hello_handle: Optional[asyncio.TimerHandle] = None

        # 0.4.0: session state.  Each configured session gets one
        # SessionFSM.  Two lookup maps because inbound session
        # messages arrive with different SID fields depending on
        # direction (see _dispatch_session_message).
        self._sessions_by_local:  Dict[int, SessionFSM] = {}
        self._sessions_by_remote: Dict[int, SessionFSM] = {}
        # Track whether we've told the dataplane about the tunnel yet
        # (added once, when the first session ESTABLISHES).
        self._tunnel_dp_added: bool = False
        for scfg in cfg.sessions:
            sess = SessionFSM(SessionConfig(
                name=scfg.name,
                local_ccid=cfg.local_ccid,
                local_sid=scfg.local_sid,
                pseudowire_type=int(scfg.pseudowire_type),
                initiator=scfg.initiator,
                l2_specific_sublayer=scfg.l2_specific_sublayer,
                data_sequencing=scfg.data_sequencing,
                circuit_status=scfg.circuit_status,
                cookie=scfg.cookie,
                peer_cookie=scfg.peer_cookie,
                tx_connect_speed=scfg.tx_connect_speed,
                rx_connect_speed=scfg.rx_connect_speed,
                ifname=scfg.ifname,
            ))
            self._sessions_by_local[scfg.local_sid] = sess

    @property
    def local_ccid(self) -> int:
        return self.cfg.local_ccid

    @property
    def state(self) -> TunnelState:
        return self._fsm.state

    # ---- lifecycle ----------------------------------------------------

    def start(self) -> None:
        """Kick off local-open — send SCCRQ."""
        _LOG.info("tunnel %s: local open (initiate SCCRQ)", self.name)
        actions = self._fsm.on_local_open()
        self._execute_actions(actions)

    def close(self) -> None:
        """Local close — send StopCCN, cancel HELLO."""
        _LOG.info("tunnel %s: local close", self.name)
        actions = self._fsm.on_local_close()
        self._execute_actions(actions)
        if self._hello_handle is not None:
            self._hello_handle.cancel()
            self._hello_handle = None

    # ---- datagram + tick paths ---------------------------------------

    def handle_datagram(self, buf: bytes, now: float) -> None:
        """Process one inbound datagram addressed to this tunnel.

        Assumes the Daemon-level demux already matched Control
        Connection ID.  Runs authenticated decode when a shared
        secret is configured.
        """
        try:
            if self.cfg.password:
                msg = ControlMessage.decode_and_verify(
                    buf, self.cfg.digest_alg, self.cfg.password
                )
            else:
                msg = ControlMessage.decode(buf)
        except ValueError as exc:
            _LOG.warning("tunnel %s: dropped malformed/unverified msg — %s",
                         self.name, exc)
            return

        # Learn peer's remote CCID once we have one.
        if self._peer.remote_ccid == 0:
            # For a message we RECEIVED, the peer's CCID appears as the
            # message's control_connection_id (they sent it to OUR CCID),
            # but the Assigned Control Connection ID AVP in SCCRQ/SCCRP
            # tells us what to USE when we send to them.  That AVP is
            # extracted here as a convenience.
            ccid_avp = find_avp(msg.avps, AttrType.ASSIGNED_CONTROL_CONNECTION_ID)
            if ccid_avp is not None and len(ccid_avp.value) == 4:
                import struct as _s
                (assigned,) = _s.unpack("!I", ccid_avp.value)
                if assigned:
                    self._peer.set_remote_ccid(assigned)
                    _LOG.debug("tunnel %s: learned peer remote CCID = %d",
                               self.name, assigned)

        delivered, responses = self._peer.receive(msg)
        # Send any transport-layer response (typically a ZLB ack).
        for resp in responses:
            self._transport.send(self._remote_addr, resp.encode())

        # If any AVPs were delivered, dispatch by message type:
        # tunnel-level → TunnelFSM, session-level (ICRQ/ICRP/ICCN/CDN) →
        # the matching SessionFSM.
        if delivered:
            virtual = ControlMessage(
                control_connection_id=self._peer.local_ccid,
                ns=msg.ns, nr=msg.nr, avps=delivered,
            )
            mt = get_message_type(delivered)
            if mt is not None and mt in _SESSION_MSG_TYPES:
                self._dispatch_session_message(virtual, mt)
            else:
                actions = self._fsm.on_message(virtual)
                self._execute_actions(actions)

    def tick(self, now: float) -> bool:
        """Run per-tick retransmit + dead-peer checks.

        Returns True if the tunnel should be removed (peer dead).
        """
        resends, dead = self._peer.tick(now=now)
        for buf in resends:
            self._transport.send(self._remote_addr, buf)
            _LOG.debug("tunnel %s: retransmit (%d bytes)", self.name, len(buf))
        if dead:
            _LOG.warning("tunnel %s: peer dead — retransmit budget exhausted",
                         self.name)
            actions = self._fsm.on_peer_dead()
            self._execute_actions(actions)
            return True
        return False

    # ---- action executor ---------------------------------------------

    def _execute_actions(self, actions: List[object]) -> None:
        for act in actions:
            # Tunnel + session share the SendMessage / other action types
            # by name; isinstance handles both since they're distinct
            # dataclass types from the two FSMs.
            if isinstance(act, SendMessage) or isinstance(act, SessSendMessage):
                self._send(act.avps)
            elif isinstance(act, SetHelloTimer):
                if self._hello_handle is not None:
                    self._hello_handle.cancel()
                self._hello_handle = self._loop.call_later(
                    act.seconds, self._on_hello_timer
                )
            elif isinstance(act, ClearHelloTimer):
                if self._hello_handle is not None:
                    self._hello_handle.cancel()
                    self._hello_handle = None
            elif isinstance(act, Established):
                _LOG.info("tunnel %s: ESTABLISHED (peer_ccid=%d)",
                          self.name, self._fsm.peer_ccid)
                # Now that the tunnel is up, tell every session to kick
                # off (initiator ones send ICRQ; responder-only ones
                # just sit in IDLE waiting for peer's ICRQ).
                for sess in self._sessions_by_local.values():
                    sub_actions = sess.on_tunnel_established()
                    self._execute_actions(sub_actions)
            elif isinstance(act, TornDown):
                _LOG.info("tunnel %s: torn down — %s", self.name, act.reason)
                # Cascade to every session so they clean their dataplane.
                for sess in list(self._sessions_by_local.values()):
                    sub_actions = sess.on_tunnel_down()
                    self._execute_actions(sub_actions)
                # After session cleanup, tear down the tunnel dataplane too.
                if self._tunnel_dp_added and self._dataplane is not None:
                    try:
                        self._dataplane.del_tunnel(self.cfg.local_ccid)
                    except Exception as exc:
                        _LOG.warning("tunnel %s: dataplane del_tunnel failed: %s",
                                     self.name, exc)
                    self._tunnel_dp_added = False
            elif isinstance(act, SessDataplaneAdd):
                self._ensure_tunnel_dataplane()
                if self._dataplane is None:
                    _LOG.error("tunnel %s: session %d wants dataplane add "
                               "but no dataplane configured",
                               self.name, act.params.local_sid)
                    continue
                try:
                    ifname = self._dataplane.add_session(act.params)
                    _LOG.info("tunnel %s: session %d dataplane added → %s",
                              self.name, act.params.local_sid, ifname)
                except Exception as exc:
                    _LOG.error("tunnel %s: session %d dataplane add failed: %s",
                               self.name, act.params.local_sid, exc)
            elif isinstance(act, SessDataplaneDel):
                if self._dataplane is None:
                    continue
                try:
                    self._dataplane.del_session(act.local_ccid, act.local_sid)
                    _LOG.info("tunnel %s: session %d dataplane deleted",
                              self.name, act.local_sid)
                except Exception as exc:
                    _LOG.warning("tunnel %s: session %d dataplane del failed: %s",
                                 self.name, act.local_sid, exc)
            elif isinstance(act, SessionEstablished):
                # Actual info-log lives in the SessDataplaneAdd handler
                # above (once we have the netdev name); nothing to do
                # here beyond noting the transition happened.
                pass
            elif isinstance(act, SessionTornDown):
                _LOG.info("tunnel %s: session torn down — %s",
                          self.name, act.reason)

    # ---- helpers -----------------------------------------------------

    def _send(self, avps: List[object]) -> None:
        """Common send path used by both tunnel + session SendMessage actions."""
        now = self._loop.time()
        msg = self._peer.send(avps, now=now)
        if self.cfg.password:
            wire = msg.encode_signed(self.cfg.digest_alg, self.cfg.password)
        else:
            wire = msg.encode()
        self._transport.send(self._remote_addr, wire)
        _LOG.debug("tunnel %s: sent Ns=%d (%d bytes)",
                   self.name, msg.ns, len(wire))

    def _dispatch_session_message(self, msg: ControlMessage, mt: int) -> None:
        """Route an ICRQ/ICRP/ICCN/CDN to the right SessionFSM.

        Correlation rules:
          * ICRQ (mt=10) — new inbound session; peer's Local SID is in
            the message.  We match against a configured session whose
            Remote End ID matches, or fall back to any responder-role
            session in IDLE with a free local_sid.
          * ICRP (mt=11) — reply to our earlier ICRQ.  Peer's Local SID
            is in the message; their echoed Remote Session ID equals
            our local_sid.  Match by that.
          * ICCN (mt=12) — the earlier ICRQ initiator finalising.
            Match by the message's Remote Session ID (= our local_sid).
          * CDN (mt=14) — teardown.  Match by Remote Session ID
            (= our local_sid) if present; otherwise by Local Session
            ID mapped through our remote-sid index.
        """
        try:
            if mt == int(MessageType.CDN):
                fields = parse_cdn_fields(msg.avps)
                local_target = fields.remote_sid   # our local_sid
                remote_key   = fields.local_sid
            else:
                fields = parse_session_fields(msg.avps)
                local_target = fields.remote_sid
                remote_key   = fields.local_sid
        except ValueError as exc:
            _LOG.warning("tunnel %s: dropping malformed session msg — %s",
                         self.name, exc)
            return

        sess: Optional[SessionFSM] = None
        if local_target and local_target in self._sessions_by_local:
            sess = self._sessions_by_local[local_target]
        elif mt == int(MessageType.ICRQ):
            # Inbound ICRQ: pick a configured session matching Remote
            # End ID (if any), else any responder-role session in IDLE.
            end_id = fields.remote_end_id
            for candidate in self._sessions_by_local.values():
                if candidate.state != SessionState.IDLE:
                    continue
                if end_id is not None and candidate.config.name.encode() != end_id:
                    continue
                sess = candidate
                break
        elif remote_key and remote_key in self._sessions_by_remote:
            # ICRP-late / CDN-with-missing-remote-sid fallback: match by
            # peer's Local Session ID against our known peer_sid map.
            sess = self._sessions_by_remote[remote_key]

        if sess is None:
            _LOG.debug("tunnel %s: no session found for msg type %d "
                       "(local_target=%d, remote_key=%d)",
                       self.name, mt, local_target, remote_key)
            return

        actions = sess.on_message(msg)
        # After the message dispatch, refresh our by-remote-sid index
        # since ICRQ/ICRP just learned the peer's SID.
        if sess.peer_sid and sess.peer_sid not in self._sessions_by_remote:
            self._sessions_by_remote[sess.peer_sid] = sess
        self._execute_actions(actions)

    def _ensure_tunnel_dataplane(self) -> None:
        """Add the kernel tunnel context if not yet done.

        Called lazily on the first session's DataplaneAdd action so we
        don't create a kernel tunnel we'd never use (tunnel could have
        no sessions — pure keep-alive mode).
        """
        if self._tunnel_dp_added or self._dataplane is None:
            return
        params = TunnelParams(
            local_ccid=self.cfg.local_ccid,
            remote_ccid=self._peer.remote_ccid,
            local_address=self.cfg.local_address,
            remote_address=self.cfg.remote_address,
            encap="udp",
            udp_sport=L2TP_UDP_PORT,
            udp_dport=L2TP_UDP_PORT,
        )
        try:
            self._dataplane.add_tunnel(params)
            self._tunnel_dp_added = True
            _LOG.info("tunnel %s: dataplane tunnel added (kernel CCID=%d)",
                      self.name, self.cfg.local_ccid)
        except Exception as exc:
            _LOG.error("tunnel %s: dataplane add_tunnel failed: %s",
                       self.name, exc)

    def _on_hello_timer(self) -> None:
        self._hello_handle = None
        actions = self._fsm.on_hello_timer()
        self._execute_actions(actions)


# ---------------------------------------------------------------------------
# Daemon
# ---------------------------------------------------------------------------

class Daemon:
    """Owns the UDP socket + all configured Tunnels + the main loop."""

    def __init__(
        self,
        cfg: DaemonConfig,
        transport: Transport,
        dataplane: Optional[Dataplane] = None,
    ) -> None:
        self.cfg = cfg
        self._transport = transport
        self._loop = asyncio.get_event_loop()
        # 0.4.0+: dataplane is real (IpCommandDataplane by default) when
        # any tunnel has sessions; None when no sessions configured (a
        # keep-alive-only deployment doesn't need CAP_NET_ADMIN).
        self._dataplane = dataplane
        self._tunnels: Dict[int, Tunnel] = {}   # keyed by local_ccid
        self._stopping = False

    async def start(self) -> None:
        # Instantiate one Tunnel per configured entry.
        for tcfg in self.cfg.tunnels:
            t = Tunnel(tcfg, self._transport, self._loop, dataplane=self._dataplane)
            self._tunnels[t.local_ccid] = t
            t.start()   # send SCCRQ immediately (initiator role)
        _LOG.info("daemon: started with %d tunnel(s)", len(self._tunnels))
        _sd_notify(b"READY=1")

    async def run(self) -> None:
        recv_task = asyncio.create_task(self._recv_loop())
        tick_task = asyncio.create_task(self._tick_loop())
        try:
            await asyncio.gather(recv_task, tick_task)
        finally:
            recv_task.cancel()
            tick_task.cancel()

    async def stop(self) -> None:
        _sd_notify(b"STOPPING=1")
        self._stopping = True
        for t in list(self._tunnels.values()):
            t.close()
        _LOG.info("daemon: stopped")

    # ---- internal loops ----------------------------------------------

    async def _recv_loop(self) -> None:
        while not self._stopping:
            try:
                _sender, buf = await self._transport.receive()
            except asyncio.CancelledError:
                return
            self._dispatch_datagram(buf)

    async def _tick_loop(self) -> None:
        while not self._stopping:
            try:
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                return
            now = self._loop.time()
            for local_ccid in list(self._tunnels):
                t = self._tunnels[local_ccid]
                is_dead = t.tick(now=now)
                if is_dead:
                    # Remove from the peer table so subsequent
                    # datagrams for this CCID don't hit a stopped tunnel.
                    del self._tunnels[local_ccid]

    def _dispatch_datagram(self, buf: bytes) -> None:
        # Peek at the destination CCID (bytes 4-7 of the header).
        if len(buf) < 12:
            _LOG.debug("daemon: dropping runt datagram (%d bytes)", len(buf))
            return
        import struct as _s
        (dest_ccid,) = _s.unpack_from("!I", buf, 4)
        tunnel = self._tunnels.get(dest_ccid)
        if tunnel is None:
            _LOG.debug("daemon: dropping datagram for unknown CCID %d",
                       dest_ccid)
            return
        tunnel.handle_datagram(buf, now=self._loop.time())


# ---------------------------------------------------------------------------
# sd_notify — no external dep
# ---------------------------------------------------------------------------

def _sd_notify(msg: bytes) -> None:
    """Send a systemd sd_notify message.

    No-op when not running under systemd (NOTIFY_SOCKET unset).
    Handles both AF_UNIX and abstract-socket variants.
    """
    addr = os.environ.get("NOTIFY_SOCKET")
    if not addr:
        return
    if addr[0] == "@":
        addr = "\0" + addr[1:]   # abstract socket
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
            s.connect(addr)
            s.sendall(msg)
    except OSError:
        # sd_notify is best-effort; don't crash the daemon over it.
        pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="xcesp-l2tpv3d",
        description="L2TPv3 (RFC 3931) dynamic-mode control-plane daemon "
                    "for XCESP.  First free-software implementation.",
    )
    p.add_argument(
        "--config", "-c",
        default="/etc/xcesp-l2tpv3d.toml",
        help="TOML config file path (default: %(default)s)",
    )
    p.add_argument(
        "--version", action="version", version=f"xcesp-l2tpv3d {__version__}"
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        cfg = load_file(args.config)
    except ConfigError as exc:
        print(f"xcesp-l2tpv3d: config error: {exc}", file=sys.stderr)
        return 2
    log.configure(cfg.global_.log_level)
    _LOG.info("xcesp-l2tpv3d %s starting; config=%s", __version__, args.config)

    return asyncio.run(_async_main(cfg))


async def _async_main(cfg: DaemonConfig) -> int:
    transport = UdpTransport()
    try:
        await transport.start((cfg.global_.listen_address, cfg.global_.listen_port))
    except OSError as exc:
        _LOG.error("bind %s:%d failed: %s",
                   cfg.global_.listen_address, cfg.global_.listen_port, exc)
        return 3

    # Create a dataplane only if any tunnel has session configs — a
    # keep-alive-only deployment doesn't need CAP_NET_ADMIN.
    needs_dp = any(t.sessions for t in cfg.tunnels)
    dp: Optional[Dataplane] = None
    if needs_dp:
        try:
            dp = default_dataplane()
        except Exception as exc:
            _LOG.error("could not initialise dataplane: %s", exc)
            transport.close()
            return 3

    daemon = Daemon(cfg, transport, dataplane=dp)
    stop_event = asyncio.Event()

    def _handle_sig() -> None:
        _LOG.info("signal received; shutting down")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _handle_sig)

    await daemon.start()
    run_task = asyncio.create_task(daemon.run())
    stop_task = asyncio.create_task(stop_event.wait())
    done, pending = await asyncio.wait(
        {run_task, stop_task}, return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()
    await daemon.stop()
    transport.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
