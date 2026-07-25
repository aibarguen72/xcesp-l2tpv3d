"""TOML configuration loader for the L2TPv3 daemon.

## Format

The daemon takes a single TOML file (default ``/etc/xcesp-l2tpv3d.toml``,
overridable with ``--config PATH``).  Structure:

```toml
[global]
# Optional daemon-wide settings.
log_level      = "info"        # debug | info | warning | error
listen_address = "0.0.0.0"     # UDP bind address; default 0.0.0.0
listen_port    = 1701          # UDP bind port; default 1701 (IANA L2TP)

[[tunnel]]
name           = "to-remote"     # operator-visible label
local_address  = "192.0.2.1"     # our local IP (source)
remote_address = "192.0.2.2"     # peer's IP (destination)
local_ccid     = 100             # our Assigned Control Connection ID (u32)
remote_ccid    = 0               # optional; 0 = learned dynamically from SCCRP
host_name      = "xcesp-router"  # our Host Name AVP; peer's for their side
router_id      = "10.0.0.1"      # our Router ID AVP (v4 dotted or u32 int)

# Optional per-tunnel:
password              = "topsecret"   # enables Message Digest AVP auth
digest_alg            = "md5"         # md5 | sha1
hello_interval        = 60            # seconds; default 60
retransmit_interval   = 1             # seconds; default 1
max_retries           = 5             # default 5
receive_window        = 4             # AVP 10; default 4
tx_connect_speed      = 100000000     # AVP 74; default 100 Mbps
rx_connect_speed      = 100000000     # AVP 75; default 100 Mbps
vendor_name           = "XCESP"       # optional AVP 8
firmware_revision     = 42            # optional AVP 6

# Session sub-blocks (0.4.0+, ignored at 0.3.0):
# [[tunnel.session]]
# session_type  = "ethernet-vlan"
# ...
```

## Public surface

- ``load_file(path) -> DaemonConfig``  — parse a TOML file.
- ``load_string(text) -> DaemonConfig`` — parse a TOML string (tests).
- ``DaemonConfig`` / ``TunnelConfigEntry`` — typed dataclasses.

Validation errors raise ``ConfigError`` with a specific message about
what failed.  All type coercion and range checking happens here so the
FSM / daemon never sees malformed input.
"""

from __future__ import annotations

import ipaddress
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from .avp import DigestHash, PseudowireType


class ConfigError(ValueError):
    """Raised on any configuration validation failure."""


@dataclass
class GlobalConfig:
    log_level:      str = "info"
    listen_address: str = "0.0.0.0"
    listen_port:    int = 1701


@dataclass
class SessionConfigEntry:
    """Per-session config inside a tunnel — [[tunnel.session]] block.

    Both ends need a session with matching pseudowire_type, l2spec_type,
    and complementary initiator flag (one true, one false).  The name
    is used as the Remote End ID AVP for peer-side matching.
    """

    name:              str
    pseudowire_type:   PseudowireType    # ethernet | ethernet-vlan
    local_sid:         int
    initiator:         bool = True

    l2_specific_sublayer: int  = 1       # 1 = default, 0 = none
    data_sequencing:      int  = 0
    circuit_status:       int  = 0b11
    cookie:               Optional[bytes] = None      # our cookie
    peer_cookie:          Optional[bytes] = None      # peer's (pre-shared)
    tx_connect_speed:     Optional[int] = None
    rx_connect_speed:     Optional[int] = None
    ifname:               Optional[str] = None        # kernel netdev name


@dataclass
class TunnelConfigEntry:
    """Per-tunnel config as loaded from TOML.

    Fields map 1:1 to TOML keys.  The daemon converts this into
    ``tunnel_fsm.TunnelConfig`` + ``transport.Peer`` parameters at
    startup — this class stays close to the on-disk representation.
    """

    name:           str
    local_address:  str
    remote_address: str
    local_ccid:     int
    host_name:      str
    router_id:      int              # normalised to u32 (dotted-quad → int)

    remote_ccid:    int = 0
    password:       Optional[bytes] = None
    digest_alg:     DigestHash = DigestHash.HMAC_MD5

    hello_interval:      float = 60.0
    retransmit_interval: float = 1.0
    max_retries:         int   = 5
    receive_window:      int   = 4

    tx_connect_speed: int = 100_000_000
    rx_connect_speed: int = 100_000_000

    vendor_name:       Optional[str] = None
    firmware_revision: Optional[int] = None

    sessions:          List[SessionConfigEntry] = field(default_factory=list)


@dataclass
class DaemonConfig:
    global_:  GlobalConfig = field(default_factory=GlobalConfig)
    tunnels:  List[TunnelConfigEntry] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_file(path: Path | str) -> DaemonConfig:
    """Load and validate a TOML configuration from ``path``."""
    p = Path(path)
    try:
        raw = p.read_bytes()
    except OSError as exc:
        raise ConfigError(f"cannot read config file {p}: {exc}") from exc
    return _parse(raw, source=str(p))


def load_string(text: str) -> DaemonConfig:
    """Load and validate a TOML configuration from an in-memory string.

    Convenience for tests and for callers that already have the text.
    """
    return _parse(text.encode("utf-8"), source="<string>")


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _parse(raw: bytes, source: str) -> DaemonConfig:
    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{source}: invalid TOML — {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ConfigError(f"{source}: non-UTF-8 content — {exc}") from exc

    return DaemonConfig(
        global_=_parse_global(data.get("global", {}), source),
        tunnels=_parse_tunnels(data.get("tunnel", []), source),
    )


def _parse_global(section: dict, source: str) -> GlobalConfig:
    if not isinstance(section, dict):
        raise ConfigError(f"{source}: [global] must be a table")
    lvl = str(section.get("log_level", "info")).lower()
    if lvl not in ("debug", "info", "warning", "error"):
        raise ConfigError(f"{source}: [global].log_level {lvl!r} invalid")
    port = int(section.get("listen_port", 1701))
    if not (1 <= port <= 65535):
        raise ConfigError(f"{source}: [global].listen_port {port} out of range")
    addr = str(section.get("listen_address", "0.0.0.0"))
    try:
        ipaddress.ip_address(addr)
    except ValueError as exc:
        raise ConfigError(
            f"{source}: [global].listen_address {addr!r} — {exc}"
        ) from exc
    return GlobalConfig(log_level=lvl, listen_address=addr, listen_port=port)


def _parse_tunnels(section: Any, source: str) -> List[TunnelConfigEntry]:
    if not isinstance(section, list):
        raise ConfigError(f"{source}: [[tunnel]] must be an array of tables")
    tunnels: List[TunnelConfigEntry] = []
    names: set[str] = set()
    for i, t in enumerate(section):
        if not isinstance(t, dict):
            raise ConfigError(
                f"{source}: [[tunnel]] index {i} is not a table"
            )
        entry = _parse_tunnel(t, source, index=i)
        if entry.name in names:
            raise ConfigError(
                f"{source}: duplicate tunnel name {entry.name!r}"
            )
        names.add(entry.name)
        tunnels.append(entry)
    return tunnels


def _parse_tunnel(t: dict, source: str, *, index: int) -> TunnelConfigEntry:
    def req(key: str, kind: type):
        if key not in t:
            raise ConfigError(
                f"{source}: [[tunnel]] index {index} missing required {key!r}"
            )
        v = t[key]
        if not isinstance(v, kind):
            raise ConfigError(
                f"{source}: [[tunnel]] index {index} — {key!r} must be "
                f"{kind.__name__}, got {type(v).__name__}"
            )
        return v

    name = req("name", str)
    local_address  = req("local_address", str)
    remote_address = req("remote_address", str)
    local_ccid     = req("local_ccid", int)
    host_name      = req("host_name", str)
    router_id_raw  = t.get("router_id")
    if router_id_raw is None:
        raise ConfigError(
            f"{source}: tunnel {name!r} missing required router_id"
        )

    # Validate addresses.
    for addr_field, addr in (
        ("local_address", local_address),
        ("remote_address", remote_address),
    ):
        try:
            ipaddress.ip_address(addr)
        except ValueError as exc:
            raise ConfigError(
                f"{source}: tunnel {name!r} — {addr_field} {addr!r} — {exc}"
            ) from exc

    # router_id may be int or dotted-quad string.
    router_id = _parse_router_id(router_id_raw, source, name)

    if not (0 < local_ccid <= 0xFFFFFFFF):
        raise ConfigError(
            f"{source}: tunnel {name!r} — local_ccid must be 1..2^32-1, "
            f"got {local_ccid}"
        )

    remote_ccid = int(t.get("remote_ccid", 0))
    if not (0 <= remote_ccid <= 0xFFFFFFFF):
        raise ConfigError(
            f"{source}: tunnel {name!r} — remote_ccid out of range: "
            f"{remote_ccid}"
        )

    password = t.get("password")
    if password is not None:
        if not isinstance(password, str) or not password:
            raise ConfigError(
                f"{source}: tunnel {name!r} — password must be a non-empty string"
            )
        password = password.encode("utf-8")

    digest_alg_raw = str(t.get("digest_alg", "md5")).lower()
    if digest_alg_raw not in ("md5", "sha1"):
        raise ConfigError(
            f"{source}: tunnel {name!r} — digest_alg must be md5 or sha1, "
            f"got {digest_alg_raw!r}"
        )
    digest_alg = (
        DigestHash.HMAC_MD5 if digest_alg_raw == "md5" else DigestHash.HMAC_SHA1
    )

    hello_interval      = float(t.get("hello_interval", 60.0))
    retransmit_interval = float(t.get("retransmit_interval", 1.0))
    max_retries         = int(t.get("max_retries", 5))
    receive_window      = int(t.get("receive_window", 4))
    tx_connect_speed    = int(t.get("tx_connect_speed", 100_000_000))
    rx_connect_speed    = int(t.get("rx_connect_speed", 100_000_000))

    if not (1 <= hello_interval <= 3600):
        raise ConfigError(
            f"{source}: tunnel {name!r} — hello_interval must be 1..3600 s"
        )
    if not (1 <= retransmit_interval <= 8):
        raise ConfigError(
            f"{source}: tunnel {name!r} — retransmit_interval must be 1..8 s"
        )
    if not (1 <= max_retries <= 10):
        raise ConfigError(
            f"{source}: tunnel {name!r} — max_retries must be 1..10"
        )
    if not (1 <= receive_window <= 1024):
        raise ConfigError(
            f"{source}: tunnel {name!r} — receive_window must be 1..1024"
        )

    vendor_name       = t.get("vendor_name")
    firmware_revision = t.get("firmware_revision")
    if vendor_name is not None and not isinstance(vendor_name, str):
        raise ConfigError(
            f"{source}: tunnel {name!r} — vendor_name must be a string"
        )
    if firmware_revision is not None:
        firmware_revision = int(firmware_revision)
        if not (0 <= firmware_revision <= 0xFFFF):
            raise ConfigError(
                f"{source}: tunnel {name!r} — firmware_revision must be 0..65535"
            )

    # Parse any [[tunnel.session]] sub-blocks — array of tables under
    # this tunnel.  Names must be unique within one tunnel.
    sessions_raw = t.get("session", [])
    if not isinstance(sessions_raw, list):
        raise ConfigError(
            f"{source}: tunnel {name!r} — 'session' must be array of tables"
        )
    sessions: List[SessionConfigEntry] = []
    seen_names: set[str] = set()
    seen_sids:  set[int] = set()
    for si, sraw in enumerate(sessions_raw):
        if not isinstance(sraw, dict):
            raise ConfigError(
                f"{source}: tunnel {name!r} session #{si} is not a table"
            )
        s = _parse_session(sraw, source, tunnel=name, index=si)
        if s.name in seen_names:
            raise ConfigError(
                f"{source}: tunnel {name!r} — duplicate session name {s.name!r}"
            )
        if s.local_sid in seen_sids:
            raise ConfigError(
                f"{source}: tunnel {name!r} — duplicate local_sid {s.local_sid}"
            )
        seen_names.add(s.name)
        seen_sids.add(s.local_sid)
        sessions.append(s)

    return TunnelConfigEntry(
        name=name,
        local_address=local_address,
        remote_address=remote_address,
        local_ccid=local_ccid,
        remote_ccid=remote_ccid,
        host_name=host_name,
        router_id=router_id,
        password=password,
        digest_alg=digest_alg,
        hello_interval=hello_interval,
        retransmit_interval=retransmit_interval,
        max_retries=max_retries,
        receive_window=receive_window,
        tx_connect_speed=tx_connect_speed,
        rx_connect_speed=rx_connect_speed,
        vendor_name=vendor_name,
        firmware_revision=firmware_revision,
        sessions=sessions,
    )


def _parse_session(
    s: dict, source: str, *, tunnel: str, index: int
) -> SessionConfigEntry:
    def req(key: str, kind: type):
        if key not in s:
            raise ConfigError(
                f"{source}: tunnel {tunnel!r} session #{index} — "
                f"missing required {key!r}"
            )
        v = s[key]
        if not isinstance(v, kind):
            raise ConfigError(
                f"{source}: tunnel {tunnel!r} session #{index} — "
                f"{key!r} must be {kind.__name__}, got {type(v).__name__}"
            )
        return v

    name       = req("name", str)
    local_sid  = req("local_sid", int)
    pw_raw     = str(s.get("pseudowire_type", "ethernet")).lower()

    if not (0 < local_sid <= 0xFFFFFFFF):
        raise ConfigError(
            f"{source}: tunnel {tunnel!r} session {name!r} — local_sid "
            f"must be 1..2^32-1"
        )

    if pw_raw == "ethernet":
        pseudowire_type = PseudowireType.ETHERNET
    elif pw_raw == "ethernet-vlan":
        pseudowire_type = PseudowireType.ETHERNET_VLAN
    else:
        raise ConfigError(
            f"{source}: tunnel {tunnel!r} session {name!r} — pseudowire_type "
            f"must be 'ethernet' or 'ethernet-vlan', got {pw_raw!r}"
        )

    initiator = bool(s.get("initiator", True))
    l2s = int(s.get("l2_specific_sublayer", 1))
    if l2s not in (0, 1):
        raise ConfigError(
            f"{source}: tunnel {tunnel!r} session {name!r} — "
            f"l2_specific_sublayer must be 0 (none) or 1 (default)"
        )
    data_seq = int(s.get("data_sequencing", 0))
    if data_seq not in (0, 1, 2):
        raise ConfigError(
            f"{source}: tunnel {tunnel!r} session {name!r} — "
            f"data_sequencing must be 0, 1, or 2"
        )
    circuit_status = int(s.get("circuit_status", 0b11))
    if not (0 <= circuit_status <= 0xFFFF):
        raise ConfigError(
            f"{source}: tunnel {tunnel!r} session {name!r} — "
            f"circuit_status must be a u16"
        )

    cookie = _parse_cookie(s.get("cookie"), source, tunnel, name, "cookie")
    peer_cookie = _parse_cookie(
        s.get("peer_cookie"), source, tunnel, name, "peer_cookie"
    )

    tx = s.get("tx_connect_speed")
    rx = s.get("rx_connect_speed")
    if tx is not None:
        tx = int(tx)
    if rx is not None:
        rx = int(rx)

    ifname = s.get("ifname")
    if ifname is not None and not isinstance(ifname, str):
        raise ConfigError(
            f"{source}: tunnel {tunnel!r} session {name!r} — ifname must be str"
        )

    return SessionConfigEntry(
        name=name,
        pseudowire_type=pseudowire_type,
        local_sid=local_sid,
        initiator=initiator,
        l2_specific_sublayer=l2s,
        data_sequencing=data_seq,
        circuit_status=circuit_status,
        cookie=cookie,
        peer_cookie=peer_cookie,
        tx_connect_speed=tx,
        rx_connect_speed=rx,
        ifname=ifname,
    )


def _parse_cookie(
    raw: Any, source: str, tunnel: str, session: str, key: str
) -> Optional[bytes]:
    """Accept a hex string ("deadbeef") of 0/4/8 bytes → bytes, or None."""
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ConfigError(
            f"{source}: tunnel {tunnel!r} session {session!r} — {key} must be "
            f"a hex string"
        )
    try:
        b = bytes.fromhex(raw)
    except ValueError as exc:
        raise ConfigError(
            f"{source}: tunnel {tunnel!r} session {session!r} — {key} is not "
            f"valid hex: {exc}"
        ) from exc
    if len(b) not in (0, 4, 8):
        raise ConfigError(
            f"{source}: tunnel {tunnel!r} session {session!r} — {key} must "
            f"decode to 0/4/8 bytes, got {len(b)}"
        )
    return b


def _parse_router_id(raw: Any, source: str, tunnel_name: str) -> int:
    """Accept int (u32) or dotted-quad string; return u32."""
    if isinstance(raw, int):
        if not (0 <= raw <= 0xFFFFFFFF):
            raise ConfigError(
                f"{source}: tunnel {tunnel_name!r} — router_id int out of "
                f"range: {raw}"
            )
        return raw
    if isinstance(raw, str):
        try:
            return int(ipaddress.IPv4Address(raw))
        except ValueError as exc:
            raise ConfigError(
                f"{source}: tunnel {tunnel_name!r} — router_id {raw!r} not a "
                f"valid IPv4 dotted-quad — {exc}"
            ) from exc
    raise ConfigError(
        f"{source}: tunnel {tunnel_name!r} — router_id must be int or "
        f"dotted-quad string, got {type(raw).__name__}"
    )
