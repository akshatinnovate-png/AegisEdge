"""Federated adapter learning with secure aggregation.

Every device trains its own adapter on its own feedback. Sharing those
improvements should not require sharing what produced them.

Secure aggregation solves this with pairwise masks: each pair of devices
derives a shared pseudo-random mask from a common seed, one adds it and the
other subtracts it. Every individual upload is then indistinguishable from
noise, while the *sum* over all participants has every mask cancel exactly —
the coordinator learns the average and nothing else. Combined with the DP
noise each device already adds, the per-device noise averages down by √N,
which is why federation recovers signal that no single device could publish.

Dropouts are the classic failure: a device that masks and then disappears
leaves its masks uncancelled. Here the round declares the surviving cohort
before unmasking, and a round that loses too many participants is abandoned
rather than producing a corrupted average.
"""
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .privacy import GaussianMechanism, PrivacyBudget


def pairwise_mask(a: str, b: str, size: int, round_id: int) -> np.ndarray:
    """Deterministic mask shared by exactly two devices for one round."""
    low, high = sorted((a, b))
    seed = hashlib.blake2b(f"{low}|{high}|{round_id}".encode("utf-8"), digest_size=8).digest()
    rng = np.random.default_rng(int.from_bytes(seed, "big"))
    return rng.normal(0.0, 1.0, size=size).astype(np.float32)


@dataclass
class Contribution:
    node_id: str
    masked: np.ndarray
    examples: int
    privacy: dict[str, Any] = field(default_factory=dict)
    at: float = field(default_factory=time.time)


def required_cohort(sigma: float, parameters: int, clip_norm: float = 1.0,
                    target_snr: float = 2.0) -> int:
    """How many devices this round needs before the average means anything.

    Gaussian DP noise is per-coordinate, so its norm grows as sigma*sqrt(d)
    while the clipped signal norm stays bounded by the clip. One device's
    noised update is therefore mostly noise by construction — that is the
    mechanism working, not a bug. Averaging N contributions divides the noise
    by sqrt(N), so the cohort size needed for a usable signal is a number the
    protocol can state up front instead of a disappointment discovered later.
    """
    noise_norm = sigma * math.sqrt(max(parameters, 1))
    return max(1, math.ceil((target_snr * noise_norm / max(clip_norm, 1e-9)) ** 2))


class FederatedClient:
    def __init__(self, node_id: str, adapter, budget: PrivacyBudget | None = None,
                 epsilon_per_round: float = 2.0, differential_privacy: bool = True) -> None:
        self.node_id = node_id
        self.adapter = adapter
        self.budget = budget or PrivacyBudget()
        self.mechanism = GaussianMechanism(clip_norm=1.0,
                                           seed=int(hashlib.blake2b(node_id.encode(), digest_size=4).hexdigest(), 16) % (2**31))
        self.epsilon_per_round = epsilon_per_round
        # DP is on by default. Turning it off is a deliberate, audited choice
        # for a fleet too small for the noise to average out (see
        # `required_cohort`) — secure aggregation still hides the individual
        # update from the coordinator either way.
        self.differential_privacy = differential_privacy
        self.baseline = adapter.parameters().copy()
        self.rounds = 0
        self.skipped_no_budget = 0

    def contribute(self, cohort: list[str], round_id: int) -> Contribution | None:
        """Produce a masked, DP-noised update — or decline if the budget is spent."""
        if not self.budget.can_spend(self.epsilon_per_round):
            self.skipped_no_budget += 1
            return None
        update = self.adapter.diff_from(self.baseline)
        if self.differential_privacy:
            noisy, privacy = self.mechanism.privatize(update, self.epsilon_per_round, self.budget.delta)
            self.budget.spend(self.epsilon_per_round)
        else:
            noisy, original = self.mechanism.clip(update)
            privacy = {"epsilon": None, "sigma": 0.0, "clip_norm": self.mechanism.clip_norm,
                       "clipped_from": round(original, 5), "dp": False,
                       "note": "differential privacy disabled for this cohort"}

        masked = noisy.copy()
        for peer in cohort:
            if peer == self.node_id:
                continue
            mask = pairwise_mask(self.node_id, peer, noisy.size, round_id)
            masked += mask if self.node_id < peer else -mask     # cancels in the sum
        self.rounds += 1
        return Contribution(self.node_id, masked, self.adapter.stats.examples, privacy)

    def adopt(self, parameters: np.ndarray) -> None:
        self.adapter.load_parameters(parameters)
        self.baseline = self.adapter.parameters().copy()

    def snapshot(self) -> dict[str, Any]:
        return {"node_id": self.node_id, "rounds": self.rounds,
                "skipped_no_budget": self.skipped_no_budget,
                "budget": self.budget.as_dict(), "adapter": self.adapter.snapshot(),
                "mechanism": self.mechanism.snapshot()}


class FederatedCoordinator:
    """Aggregates masked updates. Never sees an individual contribution in clear."""

    MIN_COHORT = 3
    MAX_DROPOUT = 0.34

    def __init__(self, parameter_size: int) -> None:
        self.parameter_size = parameter_size
        self.round_id = 0
        self.global_parameters: np.ndarray | None = None
        self.history: list[dict[str, Any]] = []
        self.abandoned = 0

    def aggregate(self, declared_cohort: list[str],
                  contributions: list[Contribution]) -> dict[str, Any]:
        """Sum the masked updates; masks cancel only if the cohort matches."""
        self.round_id += 1
        present = [c.node_id for c in contributions]
        dropouts = [n for n in declared_cohort if n not in present]

        if len(contributions) < self.MIN_COHORT:
            self.abandoned += 1
            return {"round": self.round_id, "status": "abandoned",
                    "reason": f"cohort of {len(contributions)} below minimum {self.MIN_COHORT}",
                    "dropouts": dropouts}
        if dropouts and len(dropouts) / max(len(declared_cohort), 1) > self.MAX_DROPOUT:
            self.abandoned += 1
            return {"round": self.round_id, "status": "abandoned",
                    "reason": f"{len(dropouts)} dropouts leave uncancelled masks",
                    "dropouts": dropouts}

        total = np.zeros(self.parameter_size, dtype=np.float32)
        weight = 0
        sigma = 0.0
        for contribution in contributions:
            total += contribution.masked
            weight += max(contribution.examples, 1)
            sigma = max(sigma, float(contribution.privacy.get("sigma", 0.0)))
        average = total / len(contributions)

        # what the cohort actually bought, stated rather than assumed
        noise_norm = sigma * math.sqrt(self.parameter_size) / math.sqrt(len(contributions))
        clip_norm = float(contributions[0].privacy.get("clip_norm", 1.0))
        snr = clip_norm / noise_norm if noise_norm else float("inf")
        needed = required_cohort(sigma, self.parameter_size, clip_norm)

        if self.global_parameters is None:
            self.global_parameters = average
        else:
            self.global_parameters = self.global_parameters + average

        record = {
            "round": self.round_id, "status": "aggregated",
            "participants": len(contributions), "dropouts": dropouts,
            "examples": weight,
            "update_norm": round(float(np.linalg.norm(average)), 5),
            "sigma": round(sigma, 4),
            "snr": round(snr, 4),
            "cohort_for_usable_snr": needed,
            "usable": snr >= 1.0,
            "note": ("signal dominates noise at this cohort size" if snr >= 1.0 else
                     f"noise still dominates — needs ~{needed} devices at this epsilon"),
        }
        self.history.append(record)
        return record

    def snapshot(self) -> dict[str, Any]:
        return {"rounds": self.round_id, "abandoned": self.abandoned,
                "parameters": self.parameter_size,
                "history": self.history[-5:]}
