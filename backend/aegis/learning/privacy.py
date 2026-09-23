"""Differential privacy for anything that leaves the device.

An adapter trained on local feedback has memorised local behaviour. Shipping
its raw weights up is an information leak wearing a maths costume, so updates
are clipped to a bounded L2 norm and perturbed with Gaussian noise calibrated
to (ε, δ), and the node tracks its cumulative privacy budget. When the budget
is spent, the node stops contributing rather than quietly leaking more.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PrivacyBudget:
    epsilon_total: float = 4.0
    delta: float = 1e-5
    spent: float = 0.0
    releases: list[tuple[float, float]] = field(default_factory=list)

    @property
    def remaining(self) -> float:
        return max(0.0, self.epsilon_total - self.spent)

    def can_spend(self, epsilon: float) -> bool:
        return epsilon <= self.remaining

    def spend(self, epsilon: float) -> bool:
        if not self.can_spend(epsilon):
            return False
        self.spent += epsilon
        self.releases.append((epsilon, self.spent))
        return True

    def as_dict(self) -> dict[str, Any]:
        return {"epsilon_total": self.epsilon_total, "spent": round(self.spent, 4),
                "remaining": round(self.remaining, 4), "delta": self.delta,
                "releases": len(self.releases), "exhausted": self.remaining <= 0}


class GaussianMechanism:
    """Clip to sensitivity S, then add N(0, σ²) with σ = S·√(2 ln(1.25/δ))/ε."""

    def __init__(self, clip_norm: float = 1.0, seed: int = 7) -> None:
        self.clip_norm = clip_norm
        self.rng = np.random.default_rng(seed)
        self.applications = 0
        self.total_noise_norm = 0.0

    def sigma(self, epsilon: float, delta: float) -> float:
        return self.clip_norm * math.sqrt(2.0 * math.log(1.25 / delta)) / max(epsilon, 1e-6)

    def clip(self, update: np.ndarray) -> tuple[np.ndarray, float]:
        norm = float(np.linalg.norm(update))
        if norm <= self.clip_norm or norm == 0.0:
            return update.astype(np.float32), norm
        return (update * (self.clip_norm / norm)).astype(np.float32), norm

    def privatize(self, update: np.ndarray, epsilon: float, delta: float
                  ) -> tuple[np.ndarray, dict[str, Any]]:
        clipped, original_norm = self.clip(np.asarray(update, dtype=np.float32))
        sigma = self.sigma(epsilon, delta)
        noise = self.rng.normal(0.0, sigma, size=clipped.shape).astype(np.float32)
        self.applications += 1
        self.total_noise_norm += float(np.linalg.norm(noise))
        noisy = clipped + noise
        signal = float(np.linalg.norm(clipped))
        return noisy, {
            "epsilon": epsilon, "delta": delta, "sigma": round(sigma, 5),
            "clipped_from": round(original_norm, 5), "clip_norm": self.clip_norm,
            "signal_to_noise": round(signal / (float(np.linalg.norm(noise)) or 1e-9), 4),
        }

    def snapshot(self) -> dict[str, Any]:
        return {"clip_norm": self.clip_norm, "applications": self.applications,
                "avg_noise_norm": round(self.total_noise_norm / self.applications, 4)
                if self.applications else 0.0}
