"""L2TPv3 kernel dataplane abstraction.

## What the dataplane owns

Everything below the L2TP control channel: kernel-side ``l2tp_core`` /
``l2tp_netlink`` / ``l2tp_eth`` state.  Once the SessionFSM finishes
its handshake, the ``ICCN`` action tells the dataplane to:

  1. Create the kernel L2TP tunnel context (once per peer).
  2. Create the L2TP session, which spawns an ``l2tpeth<n>`` netdev.
  3. Bring the netdev up.

On teardown (``CDN`` sent or received, tunnel torn down):

  1. Delete the session (destroys its netdev).
  2. Delete the tunnel if it has no more sessions.

## Implementations

- ``MockDataplane`` — records calls to ``add_tunnel``,
  ``add_session``, ``del_session``, ``del_tunnel``.  Used by pytest.
- ``IpCommandDataplane`` — shells out to ``ip l2tp add tunnel …`` etc.
  Fallback for platforms where pyroute2's L2TP support is incomplete
  or unavailable.  Requires ``CAP_NET_ADMIN``.
- ``PyRoute2Dataplane`` — netlink-direct via pyroute2 (0.4.1 goal;
  0.4.0 provides a stub that falls back to IpCommand if pyroute2
  can't be imported or its L2TP module is absent).

## Interface

All dataplane implementations expose the same four-method surface:

    add_tunnel(TunnelParams)   → None
    add_session(SessionParams) → str  (netdev name)
    del_session(local_ccid, local_sid)
    del_tunnel(local_ccid)

The daemon is expected to serialise calls per tunnel/session
(neither add nor del in flight at the same time for the same object).
"""

from __future__ import annotations

import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import log


_LOG = log.get("dataplane")


@dataclass(frozen=True)
class TunnelParams:
    """Parameters for creating a kernel L2TP tunnel.

    ``encap`` is ``"udp"`` or ``"ip"``.  UDP is what XCESP uses by
    default; IP encap needs additional kernel + capability config.
    ``udp_sport`` / ``udp_dport`` are ignored for IP encap.
    """

    local_ccid:      int             # our tunnel_id in the kernel
    remote_ccid:     int             # peer's tunnel_id
    local_address:   str
    remote_address:  str
    encap:           str = "udp"
    udp_sport:       int = 1701
    udp_dport:       int = 1701


@dataclass(frozen=True)
class SessionParams:
    """Parameters for creating a kernel L2TP session (netdev).

    ``ifname`` is the requested netdev name (e.g. ``l2tpeth-100-42``).
    ``l2spec_type`` is ``"none"`` or ``"default"``.  Cookies are
    optional; when present they must be 4 or 8 bytes.
    """

    local_ccid:        int
    local_sid:         int
    remote_sid:        int
    ifname:            str
    cookie:            Optional[bytes] = None    # our cookie the peer will send
    peer_cookie:       Optional[bytes] = None    # peer's cookie we will send
    l2spec_type:       str = "default"
    pseudowire_type:   str = "ethernet"


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class Dataplane(ABC):
    """Common surface every dataplane implementation exposes."""

    @abstractmethod
    def add_tunnel(self, params: TunnelParams) -> None:
        """Create the kernel L2TP tunnel state."""

    @abstractmethod
    def add_session(self, params: SessionParams) -> str:
        """Create the kernel L2TP session.  Returns the netdev name
        that was actually assigned (may differ from requested if the
        kernel renamed to avoid collisions)."""

    @abstractmethod
    def del_session(self, local_ccid: int, local_sid: int) -> None:
        """Tear down the session (and its netdev)."""

    @abstractmethod
    def del_tunnel(self, local_ccid: int) -> None:
        """Tear down the tunnel (must have no live sessions)."""


class DataplaneError(RuntimeError):
    """Raised when a dataplane operation fails."""


# ---------------------------------------------------------------------------
# MockDataplane — records calls; used by pytest
# ---------------------------------------------------------------------------

@dataclass
class MockDataplane(Dataplane):
    """In-memory dataplane for unit + integration tests.

    Records every call on ``self.calls`` so tests can assert what the
    daemon asked for.  ``add_session`` returns the requested ``ifname``
    unchanged.
    """

    calls:   List[Tuple[str, object]] = field(default_factory=list)
    tunnels: Dict[int, TunnelParams]  = field(default_factory=dict)
    # keyed by (local_ccid, local_sid)
    sessions: Dict[Tuple[int, int], SessionParams] = field(default_factory=dict)

    def add_tunnel(self, params: TunnelParams) -> None:
        if params.local_ccid in self.tunnels:
            # Idempotent — same params re-add is OK, changing is not.
            existing = self.tunnels[params.local_ccid]
            if existing != params:
                raise DataplaneError(
                    f"tunnel {params.local_ccid} already exists with different params"
                )
            return
        self.calls.append(("add_tunnel", params))
        self.tunnels[params.local_ccid] = params

    def add_session(self, params: SessionParams) -> str:
        key = (params.local_ccid, params.local_sid)
        if key in self.sessions:
            raise DataplaneError(f"session {key} already exists")
        if params.local_ccid not in self.tunnels:
            raise DataplaneError(
                f"add_session for unknown tunnel {params.local_ccid}"
            )
        self.calls.append(("add_session", params))
        self.sessions[key] = params
        return params.ifname

    def del_session(self, local_ccid: int, local_sid: int) -> None:
        key = (local_ccid, local_sid)
        if key not in self.sessions:
            # Silent no-op: it's fine to del what's already gone
            # (CDN + on_peer_dead both call this).
            return
        self.calls.append(("del_session", key))
        del self.sessions[key]

    def del_tunnel(self, local_ccid: int) -> None:
        if local_ccid not in self.tunnels:
            return
        # Enforce ordering: sessions must be gone first.
        live = [k for k in self.sessions if k[0] == local_ccid]
        if live:
            raise DataplaneError(
                f"del_tunnel({local_ccid}): sessions still live: {live}"
            )
        self.calls.append(("del_tunnel", local_ccid))
        del self.tunnels[local_ccid]


# ---------------------------------------------------------------------------
# IpCommandDataplane — subprocess to `ip l2tp`
# ---------------------------------------------------------------------------

class IpCommandDataplane(Dataplane):
    """Fallback dataplane that shells out to iproute2's ``ip l2tp``.

    Works on any recent Linux (kernel L2TP driver present, ``ip`` in
    PATH).  Slower than netlink-direct but portable, and this is what
    XCESP's static L2TP mode already uses.  Requires ``CAP_NET_ADMIN``.
    """

    def __init__(self, ip_binary: str = "ip") -> None:
        resolved = shutil.which(ip_binary)
        if resolved is None:
            raise DataplaneError(
                f"IpCommandDataplane: '{ip_binary}' not found in PATH"
            )
        self._ip = resolved
        self._tunnels: Dict[int, TunnelParams] = {}
        self._sessions: Dict[Tuple[int, int], SessionParams] = {}

    def add_tunnel(self, params: TunnelParams) -> None:
        if params.local_ccid in self._tunnels:
            return
        argv = [
            self._ip, "l2tp", "add", "tunnel",
            "remote", params.remote_address,
            "local",  params.local_address,
            "tunnel_id", str(params.local_ccid),
            "peer_tunnel_id", str(params.remote_ccid),
            "encap", params.encap,
        ]
        if params.encap == "udp":
            argv += ["udp_sport", str(params.udp_sport),
                     "udp_dport", str(params.udp_dport)]
        self._run(argv, "add_tunnel")
        self._tunnels[params.local_ccid] = params

    def add_session(self, params: SessionParams) -> str:
        argv = [
            self._ip, "l2tp", "add", "session",
            "name", params.ifname,
            "tunnel_id", str(params.local_ccid),
            "session_id", str(params.local_sid),
            "peer_session_id", str(params.remote_sid),
        ]
        if params.cookie is not None:
            argv += ["cookie", params.cookie.hex()]
        if params.peer_cookie is not None:
            argv += ["peer_cookie", params.peer_cookie.hex()]
        if params.l2spec_type:
            argv += ["l2spec_type", params.l2spec_type]
        self._run(argv, "add_session")
        # Bring the netdev up.
        self._run(
            [self._ip, "link", "set", "dev", params.ifname, "up"],
            "link_up",
        )
        self._sessions[(params.local_ccid, params.local_sid)] = params
        return params.ifname

    def del_session(self, local_ccid: int, local_sid: int) -> None:
        key = (local_ccid, local_sid)
        if key not in self._sessions:
            return
        self._run(
            [self._ip, "l2tp", "del", "session",
             "tunnel_id", str(local_ccid),
             "session_id", str(local_sid)],
            "del_session",
        )
        del self._sessions[key]

    def del_tunnel(self, local_ccid: int) -> None:
        if local_ccid not in self._tunnels:
            return
        live = [k for k in self._sessions if k[0] == local_ccid]
        if live:
            raise DataplaneError(
                f"del_tunnel({local_ccid}): sessions still live: {live}"
            )
        self._run(
            [self._ip, "l2tp", "del", "tunnel", "tunnel_id", str(local_ccid)],
            "del_tunnel",
        )
        del self._tunnels[local_ccid]

    def _run(self, argv: List[str], op: str) -> None:
        _LOG.debug("dataplane %s: %s", op, " ".join(argv))
        r = subprocess.run(argv, capture_output=True, text=True)
        if r.returncode != 0:
            raise DataplaneError(
                f"{op} failed (rc={r.returncode}): "
                f"{r.stderr.strip() or r.stdout.strip() or '<no output>'}"
            )


# ---------------------------------------------------------------------------
# Factory — pick the best dataplane available
# ---------------------------------------------------------------------------

def default_dataplane() -> Dataplane:
    """Return the best available real dataplane on this host.

    0.4.0 uses ``IpCommandDataplane`` (subprocess to iproute2's
    ``ip l2tp``) — universally portable, matches XCESP's existing
    static-L2TP path, no new dependency.  0.4.1 will try a pyroute2
    netlink-direct implementation first and fall back to this.
    """
    return IpCommandDataplane()
