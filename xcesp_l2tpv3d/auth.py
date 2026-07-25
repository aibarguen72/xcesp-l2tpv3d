"""Message-Digest AVP authentication (RFC 3931 §5.4.3 + §4.4).

## What this covers

Per-message integrity for L2TPv3 control messages, via the Message
Digest AVP (attribute type 59).  Payload of that AVP is:

    1 byte Hash Type | N-byte Digest

Hash Type 0 → HMAC-MD5 (16-byte digest), Hash Type 1 → HMAC-SHA1
(20-byte digest).

## Computation

The RFC 3931 §4.4 rule for building the digest of an outgoing message
is: prepend a 1-byte Message Integrity Type (=1) to the ENTIRE
control message with the Digest bytes zeroed out, then run HMAC-Hash
over that stream with the pre-shared secret as the HMAC key.

    MIC = HMAC-Hash( shared_secret, 0x01 || message_bytes_digest_zeroed )

On receive: extract the Message Digest AVP, zero its digest bytes in
a copy of the raw message, recompute HMAC over the same input, and
compare against the received digest with a constant-time equality
check.

This module provides the low-level primitives (digest sizing, HMAC,
nonce generation, constant-time compare); the message-building layer
in ``messages.py`` handles the "zero the digest, sign, patch in the
result" mechanics on top of them.
"""

from __future__ import annotations

import enum
import hmac
import os
from hashlib import md5, sha1
from typing import Callable

# Import the enum from avp.py so the numeric-value bytes match on the wire.
from .avp import DigestHash


#: Digest sizes per hash algorithm (bytes).
_DIGEST_SIZE = {
    DigestHash.HMAC_MD5:  16,
    DigestHash.HMAC_SHA1: 20,
}

#: hashlib constructors per hash algorithm.
_HASHLIB_CTOR: dict[DigestHash, Callable] = {
    DigestHash.HMAC_MD5:  md5,
    DigestHash.HMAC_SHA1: sha1,
}

#: RFC 3931 §4.4 "message integrity type" byte prepended before HMAC.
_MIT_MSG_INTEGRITY = b"\x01"


def digest_size(hash_type: DigestHash | int) -> int:
    """Return the byte length of a digest for the given hash algorithm."""
    ht = DigestHash(int(hash_type))
    return _DIGEST_SIZE[ht]


def compute_message_digest(
    hash_type: DigestHash | int,
    shared_secret: bytes,
    message_with_digest_zeroed: bytes,
) -> bytes:
    """Compute the RFC 3931 §4.4 Message Integrity Check.

    ``message_with_digest_zeroed`` is the full control message (12-byte
    L2TPv3 header + all AVPs including the Message Digest AVP) with
    the Digest bytes inside the Message Digest AVP zeroed out (the
    Hash Type byte is NOT zeroed).

    The returned digest has length ``digest_size(hash_type)``.
    """
    ht = DigestHash(int(hash_type))
    ctor = _HASHLIB_CTOR[ht]
    if not shared_secret:
        # HMAC accepts empty keys, but requiring a real secret catches
        # a common misconfiguration (config file omits `password`).
        raise ValueError("shared_secret must not be empty")
    mac = hmac.new(
        shared_secret,
        _MIT_MSG_INTEGRITY + message_with_digest_zeroed,
        ctor,
    )
    return mac.digest()


def verify_message_digest(
    hash_type: DigestHash | int,
    shared_secret: bytes,
    message_with_digest_zeroed: bytes,
    received_digest: bytes,
) -> bool:
    """Constant-time verification of a received Message Digest.

    Returns True iff ``received_digest`` matches the digest computed
    from the arguments.  Uses ``hmac.compare_digest`` for
    timing-safety.
    """
    expected = compute_message_digest(
        hash_type, shared_secret, message_with_digest_zeroed
    )
    if len(received_digest) != len(expected):
        return False
    return hmac.compare_digest(expected, received_digest)


def generate_nonce(length: int = 32) -> bytes:
    """Generate a cryptographically random nonce.

    Used by the Control Message Authentication Nonce AVP (attr 73)
    when peers negotiate authenticated control connections.  Length
    defaults to 32 bytes per common practice; RFC 3931 §4.3 lets
    implementations pick, but requires it be exchanged in SCCRQ/SCCRP.
    """
    if length < 1:
        raise ValueError("nonce length must be positive")
    return os.urandom(length)
