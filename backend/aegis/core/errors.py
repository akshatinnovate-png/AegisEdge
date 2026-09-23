"""Typed failures. A subsystem that fails loudly is cheaper than one that lies."""
from __future__ import annotations


class AegisError(Exception):
    """Base for every deliberate failure in the node."""
    code = "aegis_error"


class PolicyDenied(AegisError):
    code = "policy_denied"


class LinkUnavailable(AegisError):
    code = "link_unavailable"


class CircuitOpen(LinkUnavailable):
    code = "circuit_open"


class WalCorrupt(AegisError):
    code = "wal_corrupt"


class ModelUnavailable(AegisError):
    code = "model_unavailable"


class ConflictUnresolved(AegisError):
    code = "conflict_unresolved"
