"""FastAPI app: JSON endpoints for model structure, runs and per-step traces, plus the static frontend."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .runner import Runner

log = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"


class RunBody(BaseModel):
    expression: str = Field(min_length=1, max_length=400)


class PositionBody(BaseModel):
    run_id: str
    q: int = Field(ge=0)


class AttentionBody(BaseModel):
    run_id: str
    layer: int
    head: int


class ResidualBody(BaseModel):
    run_id: str
    snapshot: int


class MlpBody(BaseModel):
    run_id: str
    layer: int


class WeightsBody(BaseModel):
    kind: str
    layer: int | None = None
    head: int | None = None
    trim: bool = True


def _binary_f32(arr: np.ndarray, meta: dict) -> Response:
    """Exact weights: float32, because gating constants (1e6..1e13) overflow float16."""
    data = np.ascontiguousarray(arr, dtype="<f4").tobytes()
    meta = {**meta, "shape": list(arr.shape), "dtype": "float32"}
    return Response(content=data, media_type="application/octet-stream", headers={"X-Meta": json.dumps(meta)})


def _binary(arr: np.ndarray, meta: dict) -> Response:
    data = np.ascontiguousarray(arr, dtype="<f2").tobytes()
    meta = {**meta, "shape": list(arr.shape), "dtype": "float16"}
    return Response(content=data, media_type="application/octet-stream", headers={"X-Meta": json.dumps(meta)})


def create_app(runner: Runner) -> FastAPI:
    app = FastAPI(title="handformer visualizer")

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        log.error("bad request %s: %s", request.url.path, exc)
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        if request.url.path.startswith("/api/"):
            log.info(
                "%s %s -> %d (%s bytes, %.1fms)",
                request.method, request.url.path, response.status_code,
                response.headers.get("content-length", "?"), (time.perf_counter() - t0) * 1000,
            )
        return response

    @app.get("/api/info")
    def info() -> dict:
        return runner.info()

    @app.post("/api/run")
    def run(body: RunBody) -> dict:
        return runner.run(body.expression)

    @app.post("/api/position")
    def position(body: PositionBody) -> dict:
        return runner.position(body.run_id, body.q)

    @app.post("/api/attention")
    def attention(body: AttentionBody) -> Response:
        return _binary(*runner.attention_map(body.run_id, body.layer, body.head))

    @app.post("/api/residual")
    def residual(body: ResidualBody) -> Response:
        return _binary(*runner.residual_map(body.run_id, body.snapshot))

    @app.post("/api/mlp")
    def mlp(body: MlpBody) -> Response:
        return _binary(*runner.mlp_map(body.run_id, body.layer))

    @app.post("/api/weights")
    def weights(body: WeightsBody) -> Response:
        return _binary_f32(*runner.weights(body.kind, body.layer, body.head, body.trim))

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
