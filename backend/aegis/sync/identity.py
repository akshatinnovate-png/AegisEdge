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

Key distribution has two modes, and which one a fleet uses is a deployment
decision rather than a library one.

**Trust on first use**, the default, because a mesh has to work before anybody
has provisioned anything. The first key a device presents for an identity is
believed, and any *later* change to that identity's key is refused. That
defeats a relay and a newcomer claiming an existing name. It does not defeat
an attacker present at the very first contact, and that gap is real.

**An enrolment authority**, which closes it. A fleet has one root key pair; a
device is issued a certificate — its id and public key, signed by the root —
before it ships. A node that holds the root *public* key accepts a peer only
on a valid certificate, so an attacker at first contact is refused like any
other, because it cannot produce the root's signature over a name it was never
issued. The root private key is never on a device: it signs certificates
wherever the fleet is provisioned and stays there.

The two are not a spectrum with a safe middle. A node either requires
certificates or it does not, and `require_enrolment` says which — a node that
required them and silently fell back to believing strangers would be worse
than one that never claimed to, because the claim is what an operator plans
around.
"""
from __future__ import annotations

import base64
import json
import os

import numpy as np
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey, Ed25519PublicKey)

# The fields that make an operation what it is. Everything outside this set is
# routing or bookkeeping a relay may legitimately touch.
SIGNED_FIELDS = ("op_id", "kind", "point_id", "hlc", "device_id", "body")


def write_private_key(path: Path, pem: bytes) -> None:
    """Write a private key so it is never briefly world-readable.

    `write_bytes` then `chmod` leaves a window in which the file exists with
    whatever the umask allows — commonly 0644 — and anything on the box can
    read it. The window is short and entirely sufficient. Creating the file
    with the mode already set closes it, and the temp-then-rename keeps the
    write atomic so a crash cannot leave a half-written key behind.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    descriptor = os.open(temp, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(pem)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    temp.replace(path)


def jsonable(value: Any) -> Any:
    """Arrays become lists, wherever they are, before anything serialises them.

    An operation's body holds the *same* float32 array the point holds, so a
    retained operation costs almost nothing beyond the point it describes —
    measured, that is 8,344 bytes per memory of pure duplication removed, on a
    log that is never trimmed. The array cannot go on the wire or under a
    signature, so it is converted here, in the one place both of those paths
    pass through.

    Both must agree exactly. A signature taken over one representation and
    checked against another fails, and it fails as a *forgery*, which is the
    most misleading way for a serialisation bug to present.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def canonical(op: dict[str, Any]) -> bytes:
    """The exact bytes a signature covers.

    Canonical because two encodings of the same operation must produce the
    same signature: a verifier that re-serialised with different key ordering
    would reject perfectly good operations, which is a worse failure than the
    one this is preventing.
    """
    return json.dumps({k: jsonable(op.get(k)) for k in SIGNED_FIELDS},
                      sort_keys=True, separators=(",", ":"),
                      default=str).encode("utf-8")


def _enrolment_bytes(device_id: str, public_b64: str) -> bytes:
    """What a certificate covers: this device id bound to this exact key."""
    return json.dumps({"device_id": device_id, "public_key": public_b64},
                      sort_keys=True, separators=(",", ":")).encode("utf-8")


def _public_from_b64(raw: str) -> Ed25519PublicKey:
    return Ed25519PublicKey.from_public_bytes(base64.b64decode(raw))


class UnknownDevice(Exception):
    """A signature from an identity this node has never seen a key for."""


class NotEnrolled(Exception):
    """A peer offered a key with no valid certificate, on a node that requires one."""


class IdentityChanged(Exception):
    """A known device presented a different key. Refused, loudly."""


class DeviceIdentity:
    """One device's key pair, and the public keys it has learned."""

    def __init__(self, device_id: str, private: Ed25519PrivateKey | None = None,
                 root_public_b64: str | None = None, certificate: str | None = None,
                 require_enrolment: bool = False) -> None:
        self.device_id = device_id
        self._private = private or Ed25519PrivateKey.generate()
        self.known: dict[str, Ed25519PublicKey] = {device_id: self._private.public_key()}
        self.signed = 0
        self.verified = 0
        self.refused = 0
        self.learned = 0
        # Enrolment. `root` is the fleet's public key — never the private one,
        # which stays wherever certificates are issued. `certificate` is this
        # device's own, offered to peers alongside its key.
        self.root: Ed25519PublicKey | None = (
            _public_from_b64(root_public_b64) if root_public_b64 else None)
        self.certificate = certificate
        # Certificates this node has verified, kept so they can be relayed.
        # A certificate is signed by the fleet root and says nothing about who
        # is carrying it, which is exactly what makes it safe to pass on and
        # what makes it better than trust-on-first-use: the recipient checks
        # the root's signature, not the relay's goodwill.
        self.certificates: dict[str, str] = {}
        if certificate:
            self.certificates[device_id] = certificate
        self.require_enrolment = bool(require_enrolment)
        self.enrolled = 0
        if self.require_enrolment and self.root is None:
            raise ValueError(
                "require_enrolment is set but no root public key was given — a node "
                "that requires certificates it cannot check would refuse the whole "
                "fleet, which is not the failure anybody intended")

    # -- persistence ------------------------------------------------------

    @classmethod
    def load_or_create(cls, device_id: str, path: Path, **enrolment: Any) -> "DeviceIdentity":
        """A device keeps its identity across restarts, or it is not an identity."""
        path = Path(path)
        if path.exists():
            try:
                private = serialization.load_pem_private_key(path.read_bytes(),
                                                             password=None)
                if isinstance(private, Ed25519PrivateKey):
                    return cls(device_id, private, **enrolment)
            except Exception:
                pass          # unreadable key: generate a new one rather than refuse to boot
        identity = cls(device_id, **enrolment)
        try:
            write_private_key(path, identity._private.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()))
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

    def learn(self, device_id: str, public_b64: str,
              certificate: str | None = None) -> bool:
        """Accept a peer's key: on a certificate if required, on first sight if not.

        The change-refusal below runs either way. A certificate says who a key
        belongs to; it does not say that an identity may have two keys at once,
        and a fleet that reissues a device's certificate should expect the
        peers that knew the old key to say so rather than swap quietly.
        """
        try:
            key = Ed25519PublicKey.from_public_bytes(base64.b64decode(public_b64))
        except Exception:
            return False
        if self.require_enrolment:
            if not self.check_certificate(device_id, public_b64, certificate):
                self.refused += 1
                raise NotEnrolled(
                    f"{device_id} offered a key with no valid certificate from the "
                    f"fleet root")
            # Verified, so it can be handed on. Without this a fleet converges
            # one hop and no further: every node offered its own certificate
            # and nobody else's, so a third device could never check the key
            # for an author it had not met, and refused every relayed
            # operation as a forgery.
            self.certificates[device_id] = certificate
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

    # -- enrolment --------------------------------------------------------

    @staticmethod
    def issue(root_private: Ed25519PrivateKey, device_id: str, public_b64: str) -> str:
        """Sign a device into the fleet. Runs where the root key lives, not on a device."""
        return base64.b64encode(
            root_private.sign(_enrolment_bytes(device_id, public_b64))).decode()

    def certificate_bundle(self) -> dict[str, str]:
        """Own certificate plus every one verified, for a relay to pass on.

        Own is added here rather than only in `__init__`, because a caller that
        sets `certificate` afterwards — which enrolment tooling and tests both
        do — would otherwise hand out a bundle without itself in it, and the
        fleet would converge exactly one hop.
        """
        bundle = dict(self.certificates)
        if self.certificate:
            bundle[self.device_id] = self.certificate
        return bundle

    def check_certificate(self, device_id: str, public_b64: str,
                          certificate: str | None) -> bool:
        """Does the fleet root vouch for this exact id and key?

        Both are covered. Signing the key alone would let a device present
        somebody else's certificate under its own name; signing the id alone
        would let it present any key it liked under a name it was issued.
        """
        if self.root is None or not certificate:
            return False
        try:
            self.root.verify(base64.b64decode(certificate),
                             _enrolment_bytes(device_id, public_b64))
        except (InvalidSignature, ValueError, TypeError):
            return False
        self.enrolled += 1
        return True

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
        enrolled = self.require_enrolment
        return {"device_id": self.device_id, "public_key": self.public_key_b64,
                "known_devices": sorted(self.known), "signed": self.signed,
                "verified": self.verified, "refused": self.refused,
                "learned": self.learned, "certificates_checked": self.enrolled,
                "mode": "enrolment" if enrolled else "trust-on-first-use",
                "has_certificate": bool(self.certificate),
                "trust": (
                    "a peer is accepted only on a certificate signed by the fleet root, "
                    "so an attacker at first contact is refused like any other. The root "
                    "private key is not on this device."
                    if enrolled else
                    "first key seen for an identity is believed; a later change to it is "
                    "refused. An attacker present at first contact is not covered — set "
                    "AEGIS_FLEET_ROOT and AEGIS_DEVICE_CERT to require enrolment.")}
