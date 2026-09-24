"""The embedding space: how it is shaped, and what the node has decided about it.

The node can fit a better representation for its own corpus than the one it
shipped with. Because that is a claim, every part of it is inspectable here:
the measured geometry of the space as it stands, the candidates that were
tried, the confidence interval on each result, and the decision the gate
reached — including the ones it refused.
"""
from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from ..core.tenancy import Scope
from ..inference.geometry import AnisotropyProbe
from ..node import EdgeNode
from .deps import get_node
from .security import Principal, requires

router = APIRouter(prefix="/api/v1/space", tags=["space"])


@router.get("")
async def describe(node: EdgeNode = Depends(get_node),
                   _: Principal = Depends(requires(Scope.READ))) -> dict:
    embedder = node.embedder
    return {
        "space_version": embedder.space_version,
        "dim": embedder.dim,
        "pooling": embedder.session.snapshot().get("pooling"),
        "geometry": embedder.geometry.snapshot(),
        "lexicon": embedder.lexicon.snapshot(),
        "gate": embedder.gate.snapshot(),
        "corpus_sample": len(node._corpus_sample),
    }


@router.get("/anisotropy")
async def anisotropy(sample: int = 2048, node: EdgeNode = Depends(get_node),
                     _: Principal = Depends(requires(Scope.READ))) -> dict:
    """Measure the conditioning of the space the stored vectors actually live in.

    A pretrained table is usually far from isotropic — unrelated documents
    land at a cosine well above zero, and most of the variance sits in a
    handful of directions. Both numbers are here rather than asserted, because
    they are what decides whether a corpus-fitted transform is worth the
    migration it costs.
    """
    vectors = [np.asarray(p.dense, dtype=np.float32)
               for p in node.store.points.values() if p.dense]
    if len(vectors) < 4:
        raise HTTPException(status_code=409,
                            detail="need at least 4 embedded points to measure")
    return AnisotropyProbe.measure(np.vstack(vectors), sample=sample)


@router.post("/evaluate")
async def evaluate(queries: int = 300, node: EdgeNode = Depends(get_node),
                   _: Principal = Depends(requires(Scope.ADMIN))) -> dict:
    """Score candidate spaces against the shipped one. Measures; does not arm.

    Ground truth is constructed, not labelled: a probe is a stored document
    with half its words deleted, and the document it came from is the correct
    answer by definition.
    """
    texts = list(node._corpus_sample)
    if len(texts) < node.embedder.gate.min_documents:
        raise HTTPException(
            status_code=409,
            detail=(f"corpus too small to evaluate: {len(texts)} documents, "
                    f"need {node.embedder.gate.min_documents}"))
    node.embedder.observe(texts[-2000:])
    report = node.embedder.self_evaluate(texts, queries=queries, arm=False)
    node.audit.record("space_evaluated", str(report.get("decision", {}).get("candidate")))
    return report


@router.post("/arm")
async def arm(on: bool = True, node: EdgeNode = Depends(get_node),
              _: Principal = Depends(requires(Scope.ADMIN))) -> dict:
    """Switch the fitted transform on or off.

    Refused unless the gate has actually adopted a candidate, and refused
    while the corpus still holds vectors in the old space — a query in one
    space and a vector in another do not produce a worse answer, they produce
    a meaningless one, and no retrieval metric would show it.
    """
    embedder = node.embedder
    if on and embedder.gate.adopted is None:
        raise HTTPException(
            status_code=409,
            detail="no candidate has been adopted; POST /api/v1/space/evaluate first")
    if on and not embedder.geometry.fitted:
        raise HTTPException(status_code=409, detail="no transform has been fitted")

    stale = sum(1 for p in node.store.points.values()
                if p.dense and len(p.dense) != embedder.geometry.out_dim)
    if on and stale:
        raise HTTPException(
            status_code=409,
            detail=(f"{stale:,} stored vectors are still in the {embedder.dim}-dimensional "
                    f"space; run a renewal migration to {embedder.geometry.out_dim} "
                    f"dimensions before arming"))

    result = embedder.arm_geometry(on)
    node.audit.record("space_armed" if on else "space_disarmed", result["space_version"])
    node.bus.publish("inference", "space_armed", **result,
                     message=(f"embedding space now <b>{result['space_version']}</b> "
                              f"at {result['out_dim']} dimensions"))
    return result
