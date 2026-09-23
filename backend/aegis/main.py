"""ASGI entrypoint.

    uvicorn aegis.main:app --host 0.0.0.0 --port 8000

The API layer is deliberately thin: it translates HTTP into calls on the
EdgeNode and nothing else. The node runs whether or not anyone is listening.
"""
from __future__ import annotations

import contextlib
import time
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api import (routes_chaos, routes_graph, routes_health, routes_index,
                  routes_integrity, routes_learning, routes_memory, routes_mesh,
                  routes_renewal, routes_search, routes_slo, routes_sync,
                  routes_tenancy, ws)
from .config import get_settings
from .core.errors import AegisError
from .core.metrics import METRICS
from .node import EdgeNode


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    node = EdgeNode(settings)
    app.state.node = node
    await node.start()
    try:
        yield
    finally:
        await node.stop()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="AegisEdge",
        version="0.1.0",
        summary="Offline-first edge memory and intelligence node",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def observe(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - t0) * 1000
        METRICS.observe("http.request_ms", elapsed)
        METRICS.incr(f"http.status.{response.status_code}")
        response.headers["x-aegis-node"] = settings.node_id
        response.headers["x-aegis-ms"] = f"{elapsed:.2f}"
        return response

    @app.exception_handler(AegisError)
    async def aegis_error(_: Request, exc: AegisError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"error": exc.code, "detail": str(exc)})

    for module in (routes_health, routes_memory, routes_search, routes_sync,
                   routes_renewal, routes_chaos, routes_index, routes_learning,
                   routes_mesh, routes_tenancy, routes_graph, routes_integrity,
                   routes_slo):
        app.include_router(module.router)
    app.include_router(ws.router)

    @app.get("/", tags=["node"])
    async def root() -> dict:
        return {
            "name": "AegisEdge",
            "problem_statement": "PS03 — AI-Powered Edge Memory & Intelligence Platform",
            "node_id": settings.node_id,
            "docs": "/docs",
            "stream": "/api/v1/stream",
        }

    return app


app = create_app()
