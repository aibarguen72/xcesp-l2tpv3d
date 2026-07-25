"""Unit tests for xcesp_l2tpv3d.config."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from xcesp_l2tpv3d.avp import DigestHash, PseudowireType   # noqa: E402
from xcesp_l2tpv3d.config import (   # noqa: E402
    ConfigError,
    DaemonConfig,
    SessionConfigEntry,
    load_file,
    load_string,
)


_MINIMAL_TUNNEL = """
[[tunnel]]
name           = "to-remote"
local_address  = "192.0.2.1"
remote_address = "192.0.2.2"
local_ccid     = 100
host_name      = "xcesp-A"
router_id      = "10.0.0.1"
"""


def test_minimal_config_parses():
    cfg = load_string(_MINIMAL_TUNNEL)
    assert isinstance(cfg, DaemonConfig)
    assert len(cfg.tunnels) == 1
    t = cfg.tunnels[0]
    assert t.name == "to-remote"
    assert t.local_address == "192.0.2.1"
    assert t.remote_address == "192.0.2.2"
    assert t.local_ccid == 100
    assert t.host_name == "xcesp-A"
    assert t.router_id == 0x0A000001    # 10.0.0.1


def test_defaults_are_applied():
    cfg = load_string(_MINIMAL_TUNNEL)
    t = cfg.tunnels[0]
    assert t.hello_interval == 60.0
    assert t.retransmit_interval == 1.0
    assert t.max_retries == 5
    assert t.receive_window == 4
    assert t.tx_connect_speed == 100_000_000
    assert t.rx_connect_speed == 100_000_000
    assert t.password is None
    assert t.digest_alg == DigestHash.HMAC_MD5
    assert t.vendor_name is None
    assert t.firmware_revision is None
    # Global defaults.
    assert cfg.global_.log_level == "info"
    assert cfg.global_.listen_address == "0.0.0.0"
    assert cfg.global_.listen_port == 1701


def test_router_id_accepts_integer_form():
    cfg = load_string("""
[[tunnel]]
name = "t1"
local_address = "10.0.0.1"
remote_address = "10.0.0.2"
local_ccid = 1
host_name = "h"
router_id = 12345678
""")
    assert cfg.tunnels[0].router_id == 12345678


def test_password_encoded_to_bytes():
    cfg = load_string(_MINIMAL_TUNNEL + '\npassword = "shhh"\n')
    assert cfg.tunnels[0].password == b"shhh"


def test_digest_alg_sha1():
    cfg = load_string(_MINIMAL_TUNNEL + '\npassword = "s"\ndigest_alg = "sha1"\n')
    assert cfg.tunnels[0].digest_alg == DigestHash.HMAC_SHA1


def test_global_overrides():
    cfg = load_string("""
[global]
log_level = "debug"
listen_address = "127.0.0.1"
listen_port = 2701
""" + _MINIMAL_TUNNEL)
    assert cfg.global_.log_level == "debug"
    assert cfg.global_.listen_address == "127.0.0.1"
    assert cfg.global_.listen_port == 2701


def test_multiple_tunnels():
    cfg = load_string("""
[[tunnel]]
name = "t1"
local_address = "10.0.0.1"
remote_address = "10.0.0.2"
local_ccid = 1
host_name = "A"
router_id = "10.0.0.1"

[[tunnel]]
name = "t2"
local_address = "10.0.0.1"
remote_address = "10.0.0.3"
local_ccid = 2
host_name = "A"
router_id = "10.0.0.1"
""")
    assert len(cfg.tunnels) == 2
    assert [t.name for t in cfg.tunnels] == ["t1", "t2"]


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_invalid_toml_rejected():
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_string("[[tunnel\n")


def test_missing_required_name_rejected():
    with pytest.raises(ConfigError, match="missing required 'name'"):
        load_string("""
[[tunnel]]
local_address = "10.0.0.1"
remote_address = "10.0.0.2"
local_ccid = 1
host_name = "A"
router_id = "10.0.0.1"
""")


def test_missing_router_id_rejected():
    with pytest.raises(ConfigError, match="missing required router_id"):
        load_string("""
[[tunnel]]
name = "t"
local_address = "10.0.0.1"
remote_address = "10.0.0.2"
local_ccid = 1
host_name = "A"
""")


def test_bad_local_address_rejected():
    with pytest.raises(ConfigError, match="local_address"):
        load_string("""
[[tunnel]]
name = "t"
local_address = "not-an-ip"
remote_address = "10.0.0.2"
local_ccid = 1
host_name = "A"
router_id = "10.0.0.1"
""")


def test_bad_router_id_string_rejected():
    with pytest.raises(ConfigError, match="router_id"):
        load_string("""
[[tunnel]]
name = "t"
local_address = "10.0.0.1"
remote_address = "10.0.0.2"
local_ccid = 1
host_name = "A"
router_id = "not-an-ip"
""")


def test_router_id_int_out_of_range_rejected():
    with pytest.raises(ConfigError, match="router_id int out of range"):
        load_string("""
[[tunnel]]
name = "t"
local_address = "10.0.0.1"
remote_address = "10.0.0.2"
local_ccid = 1
host_name = "A"
router_id = 4294967296
""")


def test_local_ccid_zero_rejected():
    with pytest.raises(ConfigError, match="local_ccid"):
        load_string("""
[[tunnel]]
name = "t"
local_address = "10.0.0.1"
remote_address = "10.0.0.2"
local_ccid = 0
host_name = "A"
router_id = "10.0.0.1"
""")


def test_duplicate_tunnel_name_rejected():
    with pytest.raises(ConfigError, match="duplicate tunnel name"):
        load_string("""
[[tunnel]]
name = "dup"
local_address = "10.0.0.1"
remote_address = "10.0.0.2"
local_ccid = 1
host_name = "A"
router_id = "10.0.0.1"

[[tunnel]]
name = "dup"
local_address = "10.0.0.1"
remote_address = "10.0.0.3"
local_ccid = 2
host_name = "A"
router_id = "10.0.0.1"
""")


def test_hello_interval_out_of_range_rejected():
    with pytest.raises(ConfigError, match="hello_interval"):
        load_string(_MINIMAL_TUNNEL + "\nhello_interval = 5000\n")


def test_bad_log_level_rejected():
    with pytest.raises(ConfigError, match="log_level"):
        load_string("[global]\nlog_level = 'shout'\n" + _MINIMAL_TUNNEL)


def test_bad_listen_port_rejected():
    with pytest.raises(ConfigError, match="listen_port"):
        load_string("[global]\nlisten_port = 70000\n" + _MINIMAL_TUNNEL)


def test_bad_digest_alg_rejected():
    with pytest.raises(ConfigError, match="digest_alg"):
        load_string(_MINIMAL_TUNNEL + '\npassword = "s"\ndigest_alg = "sha256"\n')


# ---------------------------------------------------------------------------
# load_file
# ---------------------------------------------------------------------------

def test_load_file_reads_disk(tmp_path):
    p = tmp_path / "conf.toml"
    p.write_text(_MINIMAL_TUNNEL)
    cfg = load_file(p)
    assert len(cfg.tunnels) == 1


def test_load_file_missing_file_rejected(tmp_path):
    with pytest.raises(ConfigError, match="cannot read"):
        load_file(tmp_path / "does-not-exist.toml")


# ---------------------------------------------------------------------------
# [[tunnel.session]] sub-blocks (0.4.0)
# ---------------------------------------------------------------------------

def test_tunnel_with_no_sessions_has_empty_list():
    cfg = load_string(_MINIMAL_TUNNEL)
    assert cfg.tunnels[0].sessions == []


def test_session_minimal_parses():
    cfg = load_string(_MINIMAL_TUNNEL + """
[[tunnel.session]]
name       = "eth-xc"
local_sid  = 42
""")
    t = cfg.tunnels[0]
    assert len(t.sessions) == 1
    s = t.sessions[0]
    assert isinstance(s, SessionConfigEntry)
    assert s.name == "eth-xc"
    assert s.local_sid == 42
    assert s.pseudowire_type == PseudowireType.ETHERNET   # default
    assert s.initiator is True                              # default


def test_session_ethernet_vlan_pseudowire():
    cfg = load_string(_MINIMAL_TUNNEL + """
[[tunnel.session]]
name              = "vlan-xc"
local_sid         = 10
pseudowire_type   = "ethernet-vlan"
initiator         = false
""")
    s = cfg.tunnels[0].sessions[0]
    assert s.pseudowire_type == PseudowireType.ETHERNET_VLAN
    assert s.initiator is False


def test_session_full_optional_fields():
    cfg = load_string(_MINIMAL_TUNNEL + """
[[tunnel.session]]
name              = "s1"
local_sid         = 42
cookie            = "01020304"
peer_cookie       = "aabbccdd11223344"
l2_specific_sublayer = 0
data_sequencing   = 2
circuit_status    = 3
tx_connect_speed  = 10000000000
rx_connect_speed  = 1000000000
ifname            = "l2tpeth-custom"
""")
    s = cfg.tunnels[0].sessions[0]
    assert s.cookie == b"\x01\x02\x03\x04"
    assert s.peer_cookie == b"\xaa\xbb\xcc\xdd\x11\x22\x33\x44"
    assert s.l2_specific_sublayer == 0
    assert s.data_sequencing == 2
    assert s.circuit_status == 3
    assert s.tx_connect_speed == 10_000_000_000
    assert s.rx_connect_speed == 1_000_000_000
    assert s.ifname == "l2tpeth-custom"


def test_two_sessions_in_one_tunnel():
    cfg = load_string(_MINIMAL_TUNNEL + """
[[tunnel.session]]
name = "s1"
local_sid = 1

[[tunnel.session]]
name = "s2"
local_sid = 2
""")
    sessions = cfg.tunnels[0].sessions
    assert [s.name for s in sessions] == ["s1", "s2"]
    assert [s.local_sid for s in sessions] == [1, 2]


def test_duplicate_session_name_rejected():
    with pytest.raises(ConfigError, match="duplicate session name"):
        load_string(_MINIMAL_TUNNEL + """
[[tunnel.session]]
name = "dup"
local_sid = 1

[[tunnel.session]]
name = "dup"
local_sid = 2
""")


def test_duplicate_session_local_sid_rejected():
    with pytest.raises(ConfigError, match="duplicate local_sid"):
        load_string(_MINIMAL_TUNNEL + """
[[tunnel.session]]
name = "a"
local_sid = 42

[[tunnel.session]]
name = "b"
local_sid = 42
""")


def test_session_missing_name_rejected():
    with pytest.raises(ConfigError, match="missing required 'name'"):
        load_string(_MINIMAL_TUNNEL + """
[[tunnel.session]]
local_sid = 1
""")


def test_session_missing_local_sid_rejected():
    with pytest.raises(ConfigError, match="missing required 'local_sid'"):
        load_string(_MINIMAL_TUNNEL + """
[[tunnel.session]]
name = "s"
""")


def test_session_local_sid_zero_rejected():
    with pytest.raises(ConfigError, match="local_sid"):
        load_string(_MINIMAL_TUNNEL + """
[[tunnel.session]]
name = "s"
local_sid = 0
""")


def test_session_bad_pseudowire_type_rejected():
    with pytest.raises(ConfigError, match="pseudowire_type"):
        load_string(_MINIMAL_TUNNEL + """
[[tunnel.session]]
name = "s"
local_sid = 1
pseudowire_type = "hdlc"
""")


def test_session_bad_l2_sublayer_rejected():
    with pytest.raises(ConfigError, match="l2_specific_sublayer"):
        load_string(_MINIMAL_TUNNEL + """
[[tunnel.session]]
name = "s"
local_sid = 1
l2_specific_sublayer = 5
""")


def test_session_bad_cookie_length_rejected():
    with pytest.raises(ConfigError, match="cookie must"):
        load_string(_MINIMAL_TUNNEL + """
[[tunnel.session]]
name = "s"
local_sid = 1
cookie = "0102030405"
""")   # 5 bytes — not 0/4/8


def test_session_bad_cookie_hex_rejected():
    with pytest.raises(ConfigError, match="valid hex"):
        load_string(_MINIMAL_TUNNEL + """
[[tunnel.session]]
name = "s"
local_sid = 1
cookie = "not-hex"
""")
