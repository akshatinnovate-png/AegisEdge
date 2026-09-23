"""NVIDIA Triton escalation client.

The heavy tier: large rerankers and long-context reasoning. It is consulted
only when local confidence is low, the link is good enough, and policy allows
the query text to leave — and every escalation records why, because "it went
to the cloud" must never be a thing that quietly happens.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from ..core.circuit import CircuitBreaker
from ..core.errors import LinkUnavailable
from ..core.metrics import METRICS


@dataclass(slots=True)
class EscalationDecision:
    escalate: bool
    reason: str
    checks: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"escalate": self.escalate, "reason": self.reason, "checks": self.checks}


class TritonClient:
    """gRPC ensemble client (`tokenize → embed → rerank` server-side).

    Without a reachable Triton endpoint the client reports itself unavailable
    and the pipeline stays local — which is the correct behaviour for this
    product, not a degraded one.
    """

    def __init__(self, url: str | None = None, model: str = "aegis_rerank_ensemble",
                 confidence_floor: float = 0.45, rtt_budget_ms: float = 400.0) -> None:
        self.url = url
        self.model = model
        self.confidence_floor = confidence_floor
        self.rtt_budget_ms = rtt_budget_ms
        self.breaker = CircuitBreaker("triton", failure_threshold=3, reset_after_s=10.0)
        self.escalations = 0
        self.declined = 0
        self.failures = 0
        self.last_reason = "not attempted"
        self._client: Any = None

    @property
    def available(self) -> bool:
        return bool(self.url) and self.breaker.allows()

    def should_escalate(self, *, top_score: float, link_state: str, rtt_ms: float,
                        may_egress: bool, complex_query: bool) -> EscalationDecision:
        checks = {
            "top_score": round(top_score, 4),
            "confidence_floor": self.confidence_floor,
            "link_state": link_state,
            "rtt_ms": round(rtt_ms, 1),
            "rtt_budget_ms": self.rtt_budget_ms,
            "may_egress": may_egress,
            "complex_query": complex_query,
            "endpoint_configured": bool(self.url),
        }
        if not self.url:
            self.declined += 1
            return EscalationDecision(False, "no triton endpoint configured — staying local", checks)
        if link_state == "offline":
            self.declined += 1
            return EscalationDecision(False, "link offline — local answer is the only answer", checks)
        if not may_egress:
            self.declined += 1
            return EscalationDecision(False, "policy forbids sending this query off-device", checks)
        if rtt_ms > self.rtt_budget_ms:
            self.declined += 1
            return EscalationDecision(False, f"rtt {rtt_ms:.0f}ms over budget", checks)
        if top_score >= self.confidence_floor and not complex_query:
            self.declined += 1
            return EscalationDecision(False, "local confidence sufficient", checks)
        if not self.breaker.allows():
            self.declined += 1
            return EscalationDecision(False, "triton circuit open", checks)
        return EscalationDecision(True, "low local confidence on a complex query", checks)

    async def rerank(self, query: str, candidates: list[tuple[str, str]],
                     timeout_s: float = 2.0) -> list[tuple[str, float]]:
        """Call the server-side ensemble; one round trip, not three."""
        self.breaker.guard()
        t0 = time.perf_counter()
        try:
            result = await asyncio.wait_for(self._infer(query, candidates), timeout=timeout_s)
            self.breaker.record_success()
            self.escalations += 1
            METRICS.incr("triton.escalations")
            METRICS.observe("triton.rerank_ms", (time.perf_counter() - t0) * 1000)
            return result
        except Exception as exc:
            self.breaker.record_failure()
            self.failures += 1
            METRICS.incr("triton.failures")
            raise LinkUnavailable(f"triton: {exc}") from exc

    async def _infer(self, query: str, candidates: list[tuple[str, str]]) -> list[tuple[str, float]]:
        """Call the ensemble. There is no local stand-in for a remote model.

        An earlier version returned random scores when no server was reachable
        so the code path stayed "measurable". That is a lie with a latency
        histogram attached: it would have reordered real results using noise.
        Without a live endpoint the escalation is declined upstream, and if one
        is configured but unreachable this raises so the breaker can open.
        """
        import tritonclient.grpc.aio as grpcclient  # type: ignore
        import numpy as np

        if self._client is None:
            self._client = grpcclient.InferenceServerClient(url=self.url)

        pairs = [f"{query} [SEP] {text}" for _, text in candidates]
        payload = np.array([[p.encode("utf-8")] for p in pairs], dtype=object)
        inputs = grpcclient.InferInput("TEXT", payload.shape, "BYTES")
        inputs.set_data_from_numpy(payload)
        response = await self._client.infer(
            model_name=self.model, inputs=[inputs],
            outputs=[grpcclient.InferRequestedOutput("SCORES")])
        scores = response.as_numpy("SCORES").reshape(-1)
        ranked = [(pid, float(score)) for (pid, _), score in zip(candidates, scores)]
        ranked.sort(key=lambda row: -row[1])
        return ranked

    def snapshot(self) -> dict[str, Any]:
        hist = METRICS.histograms.get("triton.rerank_ms")
        return {"endpoint": self.url, "model": self.model, "available": self.available,
                "escalations": self.escalations, "declined": self.declined,
                "failures": self.failures, "breaker": self.breaker.snapshot(),
                "latency_ms": hist.snapshot() if hist else {}}
