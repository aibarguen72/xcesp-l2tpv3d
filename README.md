# xcesp-l2tpv3d

L2TPv3 (RFC 3931) dynamic-mode control-plane daemon for XCESP.

First free-software Linux implementation of the L2TPv3 control channel.
OpenL2TP is L2TPv2-only; go-l2tp's `ql2tpd` is static+keepalive (peer
must also run ql2tpd, not RFC-3931 negotiation); prol2tp is proprietary.
This daemon speaks the RFC-3931 control channel so XCESP can bring up
L2TPv3 pseudowires against Cisco `l2tp-class` peers configured
**with** the L2TPv3 protocol (not the "no protocol" workaround XCESP's
static mode requires today).

## Roadmap

See `~/.claude/plans/i-tell-you-i-fizzy-lemon.md` for the full 0.1.0 →
1.0.0 milestone table.

## Layout

- `xcesp_l2tpv3d/` — importable package
  - `avp.py` — AVP encode/decode (0.1.0)
  - `messages.py`, `transport.py`, `auth.py` — 0.2.0-0.3.0
  - `tunnel_fsm.py`, `session_fsm.py`, `dataplane.py` — 0.3.0-0.4.0
  - `peer.py`, `config.py`, `main.py`, `control.py`, `log.py` — 0.3.0+
- `test/` — pytest suite
- `contrib/systemd/xcesp-l2tpv3d.service` — 0.3.0+
- `contrib/cfg/xcesp-l2tpv3d.toml.example` — 0.3.0+

## Build and test

```
make          # summary of targets
make test     # pytest test/
make install  # pip install -e . into current venv
make clean
```
