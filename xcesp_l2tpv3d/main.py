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
from .avp import find_avp, AttrType, VENDOR_IETF
from .config import DaemonConfig, TunnelConfigEntry, load_file, ConfigError
from .messages import ControlMessage
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
    ) -> None:
        self.name = cfg.name
        self.cfg  = cfg
        self._transport = transport
        self._loop = loop

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

        # If any AVPs were delivered, run them through the FSM.  A
        # single Ns can bring one message with one Message Type AVP,
        # so we assemble a virtual ControlMessage per delivered batch.
        if delivered:
            virtual = ControlMessage(
                control_connection_id=self._peer.local_ccid,
                ns=msg.ns, nr=msg.nr, avps=delivered,
            )
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
            if isinstance(act, SendMessage):
                # Wrap FSM's AVP list into a ControlMessage via the Peer
                # (assigns Ns/Nr, queues for retransmit) and encode
                # (signed if we have a secret).
                now = self._loop.time()
                msg = self._peer.send(act.avps, now=now)
                if self.cfg.password:
                    wire = msg.encode_signed(
                        self.cfg.digest_alg, self.cfg.password
                    )
                else:
                    wire = msg.encode()
                self._transport.send(self._remote_addr, wire)
                _LOG.debug("tunnel %s: sent Ns=%d (%d bytes)",
                           self.name, msg.ns, len(wire))
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
            elif isinstance(act, TornDown):
                _LOG.info("tunnel %s: torn down — %s", self.name, act.reason)

    def _on_hello_timer(self) -> None:
        self._hello_handle = None
        actions = self._fsm.on_hello_timer()
        self._execute_actions(actions)


# ---------------------------------------------------------------------------
# Daemon
# ---------------------------------------------------------------------------

class Daemon:
    """Owns the UDP socket + all configured Tunnels + the main loop."""

    def __init__(self, cfg: DaemonConfig, transport: Transport) -> None:
        self.cfg = cfg
        self._transport = transport
        self._loop = asyncio.get_event_loop()
        self._tunnels: Dict[int, Tunnel] = {}   # keyed by local_ccid
        self._stopping = False

    async def start(self) -> None:
        # Instantiate one Tunnel per configured entry.
        for tcfg in self.cfg.tunnels:
            t = Tunnel(tcfg, self._transport, self._loop)
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

    daemon = Daemon(cfg, transport)
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
