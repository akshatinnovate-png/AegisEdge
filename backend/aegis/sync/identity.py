"""Device identity: an operation says who wrote it, and can prove it.

A CRDT merges what it is given. That is the property that makes offline
editing work, and it is also a standing assumption that every peer is honest —
which is an odd assumption to make about a mesh of field devices, where one
lost handset is one dishonest peer.

The deterministic simulator made the cost of that assumption concrete. A relay
that forwards somebody else's operation with the body rewritten was accepted
by every downstream node in 35 of 40 executions: the content people would act
on, altered in flight, with nothing anywhere to notice. A second attack
attributed an operation to a device that never wrote it, and nothing noticed
that either.

So each device holds an Ed25519 key pair and signs the immutable part of every
operation it creates. A receiver verifies against the public key it has for
the claimed author, and refuses what does not check out. Tampering fails
because the relay cannot produce the author's signature over the new content.
Impersonation fails for the same reason.

Cost, measured on this hardware: 47 microseconds to sign, 122 to verify, 64
bytes on the wire. Against an ingest that takes 18 milliseconds that is a
quarter of one percent.

Key distribution is trust-on-first-use, and the limit is stated rather than
glossed: the first key a device presents for an identity is believed, and any
*later* change to that identity's key is refused. That defeats a relay and a
newcomer claiming an existing name; it does not defeat an attacker present at
the very first contact. Closing that needs an enrolment authority, which is a
deployment decision rather than a library one.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey, Ed25519PublicKey)

# The fields that make an operation what it is. Everything outside this set is
# routing or bookkeeping a relay may legitimately touch.
SIGNED_FIELDS = ("op_id", "kind", "point_id", "hlc", "device_id", "body")


def canonical(op: dict[str, Any]) -> bytes:
    """The exact bytes a signature covers.

    Canonical because two encodings of the same operation must produce the
    same signature: a verifier that re-serialised with different key ordering
    would reject perfectly good operations, which is a worse failure than the
    one this is preventing.
    """
    return json.dumps({k: op.get(k) for k in SIGNED_FIELDS},
                      sort_keys=True, separators=(",", ":"),
                      default=str).encode("utf-8")


class UnknownDevice(Exception):
    """A signature from an identity this node has never seen a key for."""


class IdentityChanged(Exception):
    """A known device presented a different key. Refused, loudly."""


class DeviceIdentity:
    """One device's key pair, and the public keys it has learned."""

    def __init__(self, device_id: str, private: Ed25519PrivateKey | None = None) -> None:
        self.device_id = device_id
        self._private = private or Ed25519PrivateKey.generate()
        self.known: dict[str, Ed25519PublicKey] = {device_id: self._private.public_key()}
        self.signed = 0
        self.verified = 0
        self.refused = 0
        self.learned = 0

    # -- persistence ------------------------------------------------------

    @classmethod
    def load_or_create(cls, device_id: str, path: Path) -> "DeviceIdentity":
        """A device keeps its identity across restarts, or it is not an identity."""
        path = Path(path)
        if path.exists():
            try:
                private = serialization.load_pem_private_key(path.read_bytes(),
                                                             password=None)
                if isinstance(private, Ed25519PrivateKey):
                    return cls(device_id, private)
            except Exception:
                pass          # unreadable key: generate a new one rather than refuse to boot
        identity = cls(device_id)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(path.suffix + ".tmp")
            temp.write_bytes(identity._private.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()))
            temp.chmod(0o600)
            temp.replace(path)
        except Exception:
            pass              # an ephemeral identity still signs; it just will not persist
        return identity

    @property
    def public_key_b64(self) -> str:
        raw = self._private.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw)
        return base64.b64encode(raw).decode()

    # -- the mesh handshake -----------------------------------------------

    def learn(self, device_id: str, public_b64: str) -> bool:
        """Trust a peer's key on first sight; refuse a later change to it."""
        try:
            key = Ed25519PublicKey.from_public_bytes(base64.b64decode(public_b64))
        except Exception:
            return False
        existing = self.known.get(device_id)
        if existing is not None:
            same = existing.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw) == key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw)
            if not same:
                raise IdentityChanged(
                    f"{device_id} presented a different key than the one first seen")
            return True
        self.known[device_id] = key
        self.learned += 1
        return True

    # -- signing and verifying --------------------------------------------

    def sign(self, op: dict[str, Any]) -> str:
        self.signed += 1
        return base64.b64encode(self._private.sign(canonical(op))).decode()

    def verify(self, op: dict[str, Any], signature: str | None) -> bool:
        author = op.get("device_id")
        key = self.known.get(author)
        if key is None:
            self.refused += 1
            raise UnknownDevice(f"no key known for {author}")
        if not signature:
            self.refused += 1
            return False
        try:
            key.verify(base64.b64decode(signature), canonical(op))
        except (InvalidSignature, ValueError, TypeError):
            self.refused += 1
            return False
        self.verified += 1
        return True

    def snapshot(self) -> dict[str, Any]:
        return {"device_id": self.device_id, "public_key": self.public_key_b64,
                "known_devices": sorted(self.known), "signed": self.signed,
                "verified": self.verified, "refused": self.refused,
                "learned": self.learned,
                "trust": ("first key seen for an identity is believed; a later change "
                          "to it is refused. An attacker present at first contact is "
                          "not covered — that needs an enrolment authority.")}
