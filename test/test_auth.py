"""Unit tests for xcesp_l2tpv3d.auth."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from xcesp_l2tpv3d.auth import (   # noqa: E402
    compute_message_digest,
    digest_size,
    generate_nonce,
    verify_message_digest,
)
from xcesp_l2tpv3d.avp import DigestHash   # noqa: E402


def test_digest_size_md5_is_16():
    assert digest_size(DigestHash.HMAC_MD5) == 16


def test_digest_size_sha1_is_20():
    assert digest_size(DigestHash.HMAC_SHA1) == 20


def test_digest_size_accepts_int():
    assert digest_size(0) == 16
    assert digest_size(1) == 20


def test_digest_size_unknown_hash_rejected():
    with pytest.raises(ValueError):
        digest_size(99)


# ---------------------------------------------------------------------------
# HMAC computation is deterministic for the same inputs
# ---------------------------------------------------------------------------

def test_message_digest_deterministic_md5():
    secret = b"shared-secret"
    msg = b"\xC8\x03\x00\x0C" + b"\x00" * 8   # 12-byte fake header, zeroed
    d1 = compute_message_digest(DigestHash.HMAC_MD5, secret, msg)
    d2 = compute_message_digest(DigestHash.HMAC_MD5, secret, msg)
    assert d1 == d2
    assert len(d1) == 16


def test_message_digest_deterministic_sha1():
    secret = b"shared-secret"
    msg = b"\xC8\x03\x00\x0C" + b"\x00" * 8
    d = compute_message_digest(DigestHash.HMAC_SHA1, secret, msg)
    assert len(d) == 20


def test_message_digest_changes_when_message_changes():
    secret = b"shared-secret"
    m1 = b"\xC8\x03\x00\x0C" + b"\x00" * 8
    m2 = b"\xC8\x03\x00\x0C" + b"\x00" * 7 + b"\x01"
    d1 = compute_message_digest(DigestHash.HMAC_MD5, secret, m1)
    d2 = compute_message_digest(DigestHash.HMAC_MD5, secret, m2)
    assert d1 != d2


def test_message_digest_changes_when_secret_changes():
    m = b"\xC8\x03\x00\x0C" + b"\x00" * 8
    d1 = compute_message_digest(DigestHash.HMAC_MD5, b"secret-A", m)
    d2 = compute_message_digest(DigestHash.HMAC_MD5, b"secret-B", m)
    assert d1 != d2


def test_message_digest_different_hash_types_differ():
    secret = b"shared-secret"
    m = b"\xC8\x03\x00\x0C" + b"\x00" * 8
    d_md5 = compute_message_digest(DigestHash.HMAC_MD5, secret, m)
    d_sha1 = compute_message_digest(DigestHash.HMAC_SHA1, secret, m)
    assert d_md5 != d_sha1[:16]   # even truncated, algorithms differ


def test_empty_secret_rejected():
    with pytest.raises(ValueError, match="shared_secret"):
        compute_message_digest(DigestHash.HMAC_MD5, b"", b"whatever")


# ---------------------------------------------------------------------------
# Verify: round-trip
# ---------------------------------------------------------------------------

def test_verify_message_digest_accepts_valid_md5():
    secret = b"topsecret"
    msg = b"\xC8\x03\x00\x0C" + b"\x00" * 8
    d = compute_message_digest(DigestHash.HMAC_MD5, secret, msg)
    assert verify_message_digest(DigestHash.HMAC_MD5, secret, msg, d)


def test_verify_message_digest_accepts_valid_sha1():
    secret = b"topsecret"
    msg = b"\xC8\x03\x00\x0C" + b"\x00" * 8
    d = compute_message_digest(DigestHash.HMAC_SHA1, secret, msg)
    assert verify_message_digest(DigestHash.HMAC_SHA1, secret, msg, d)


def test_verify_message_digest_rejects_tampered_message():
    secret = b"topsecret"
    m1 = b"\xC8\x03\x00\x0C" + b"\x00" * 8
    m2 = b"\xC8\x03\x00\x0C" + b"\x00" * 7 + b"\xff"
    d = compute_message_digest(DigestHash.HMAC_MD5, secret, m1)
    assert not verify_message_digest(DigestHash.HMAC_MD5, secret, m2, d)


def test_verify_message_digest_rejects_wrong_secret():
    m = b"\xC8\x03\x00\x0C" + b"\x00" * 8
    d = compute_message_digest(DigestHash.HMAC_MD5, b"correct", m)
    assert not verify_message_digest(DigestHash.HMAC_MD5, b"wrong", m, d)


def test_verify_message_digest_rejects_wrong_length():
    m = b"\xC8\x03\x00\x0C" + b"\x00" * 8
    d = compute_message_digest(DigestHash.HMAC_MD5, b"s", m)
    # Truncated digest should never verify (length mismatch guard).
    assert not verify_message_digest(DigestHash.HMAC_MD5, b"s", m, d[:-1])


# ---------------------------------------------------------------------------
# Nonce generation
# ---------------------------------------------------------------------------

def test_nonce_default_length_is_32():
    n = generate_nonce()
    assert len(n) == 32


def test_nonce_respects_custom_length():
    n = generate_nonce(length=64)
    assert len(n) == 64


def test_nonce_is_random_across_calls():
    # Extremely unlikely to collide with 32 random bytes.
    seen = {generate_nonce() for _ in range(64)}
    assert len(seen) == 64


def test_nonce_zero_length_rejected():
    with pytest.raises(ValueError):
        generate_nonce(length=0)


def test_nonce_negative_length_rejected():
    with pytest.raises(ValueError):
        generate_nonce(length=-1)
