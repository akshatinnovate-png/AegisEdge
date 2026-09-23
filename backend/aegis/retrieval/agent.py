"""Agentic reasoning loop.

plan → retrieve → (escalate) → verify → answer. Every step is emitted on the
reasoning channel, so the UI shows what the node actually did rather than a
spinner followed by a claim.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from ..core.bus import EventBus
from ..core.metrics import METRICS
from .contradiction import ContradictionDetector
from .pipeline import RetrievalPipeline

PROCEDURAL = re.compile(r"(?i)\b(how do i|how to|steps|procedure|recovery|fix|repair|restore)\b")
TEMPORAL = re.compile(r"(?i)\b(when|last|latest|recent|yesterday|today|history|since)\b")
QUANTITATIVE = re.compile(r"(?i)\b(how many|count|average|total|rate|trend)\b")


@dataclass(slots=True)
class Step:
    name: str
    detail: str
    ms: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"step": self.name, "detail": self.detail, "ms": round(self.ms, 2), **self.data}


@dataclass(slots=True)
class Answer:
    query: str
    answer: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    contradictions: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    escalated: bool = False
    latency_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {"query": self.query, "answer": self.answer, "citations": self.citations,
                "trace": self.trace, "contradictions": self.contradictions,
                "confidence": round(self.confidence, 3), "escalated": self.escalated,
                "latency_ms": round(self.latency_ms, 2)}


class ReasoningAgent:
    def __init__(self, pipeline: RetrievalPipeline, store, bus: EventBus) -> None:
        self.pipeline = pipeline
        self.store = store
        self.bus = bus
        self.detector = ContradictionDetector()
        self.runs = 0

    def _plan(self, query: str) -> dict[str, Any]:
        """Route the query: which collection, how many hits, which mode."""
        if PROCEDURAL.search(query):
            return {"collection": "procedural", "k": 4, "mode": "hybrid", "intent": "procedural"}
        if QUANTITATIVE.search(query):
            return {"collection": "sensor", "k": 8, "mode": "hybrid", "intent": "quantitative"}
        if TEMPORAL.search(query):
            return {"collection": "episodic", "k": 6, "mode": "hybrid", "intent": "temporal"}
        return {"collection": "*", "k": 5, "mode": "hybrid", "intent": "semantic"}

    async def answer(self, query: str) -> Answer:
        t_start = time.perf_counter()
        self.runs += 1
        out = Answer(query=query, answer="")
        steps: list[Step] = []

        t0 = time.perf_counter()
        plan = self._plan(query)
        steps.append(Step("plan", f"intent={plan['intent']} collection={plan['collection']}",
                          (time.perf_counter() - t0) * 1000, plan))
        self._emit(steps[-1])

        t0 = time.perf_counter()
        retrieval = await self.pipeline.search(query, k=plan["k"], collection=plan["collection"],
                                               mode=plan["mode"])
        steps.append(Step("retrieve", f"{len(retrieval.results)} candidates",
                          (time.perf_counter() - t0) * 1000,
                          {"stages": retrieval.stages, "escalated": retrieval.escalated}))
        self._emit(steps[-1])
        out.escalated = retrieval.escalated

        # verify: do the supporting memories disagree with each other?
        t0 = time.perf_counter()
        points = [self.store.points[r["id"]] for r in retrieval.results if r["id"] in self.store.points]
        contradictions: list[dict[str, Any]] = []
        for point in points[:3]:
            for finding in self.detector.check(point, points):
                contradictions.append(finding.as_dict())
        out.contradictions = contradictions
        steps.append(Step("verify", f"{len(contradictions)} contradiction(s)",
                          (time.perf_counter() - t0) * 1000, {"checked": len(points)}))
        self._emit(steps[-1])

        # answer: extractive and cited — an edge node does not get to hallucinate
        t0 = time.perf_counter()
        if not retrieval.results:
            out.answer = "No local memory matches that. The node has nothing to cite."
            out.confidence = 0.0
        else:
            top = retrieval.results[0]
            supporting = [r for r in retrieval.results[1:3] if r["score"] > top["score"] * 0.6]
            out.answer = top["text"]
            if contradictions:
                out.answer += "  ⚠ A newer memory contradicts this; both are retained."
            elif supporting:
                out.answer += f"  (+{len(supporting)} corroborating {'memory' if len(supporting) == 1 else 'memories'})"
            out.confidence = min(0.99, top["score"] * (0.75 if contradictions else 1.0))
            out.citations = [
                {"id": r["id"], "collection": r["collection"], "score": r["score"],
                 "age_s": r["age_s"], "matched_by": r["matched_by"]}
                for r in retrieval.results[:3]
            ]
        steps.append(Step("answer", f"confidence={out.confidence:.2f}",
                          (time.perf_counter() - t0) * 1000, {"citations": len(out.citations)}))
        self._emit(steps[-1])

        out.trace = [s.as_dict() for s in steps]
        out.latency_ms = (time.perf_counter() - t_start) * 1000
        METRICS.observe("agent.answer_ms", out.latency_ms)
        return out

    def _emit(self, step: Step) -> None:
        self.bus.publish("reasoning", step.name, **step.as_dict(),
                         message=f"<b>{step.name}</b> · {step.detail} ({step.ms:.1f} ms)")
