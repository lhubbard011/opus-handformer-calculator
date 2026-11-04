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


class Knockout(BaseModel):
    heads: list[tuple[int, int]] = []
    units: list[tuple[int, int]] = []
    dims: list[str] = []


class SampleBody(BaseModel):
    temperature: float = Field(ge=0.05, le=5.0)
    seed: int = Field(ge=0, le=0xFFFFFFFF)


class RunBody(BaseModel):
    model: str = "calc"
    expression: str = Field(min_length=1, max_length=8000)  # each model checks its own limit
    knockout: Knockout | None = None
    max_new: int | None = Field(default=None, ge=1, le=100_000)  # "run longer"; the model's own limit still applies
    sample: SampleBody | None = None  # language models: seeded sampling instead of greedy


class PositionBody(BaseModel):
    model: str = "calc"
    run_id: str
    q: int = Field(ge=0)


class AttentionBody(BaseModel):
    model: str = "calc"
    run_id: str
    layer: int
    head: int


class ResidualBody(BaseModel):
    model: str = "calc"
    run_id: str
    snapshot: int


class MlpBody(BaseModel):
    model: str = "calc"
    run_id: str
    layer: int


class WeightsBody(BaseModel):
    model: str = "calc"
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


def create_app(runners: dict[str, Runner]) -> FastAPI:
    """One app for the models in ``runners`` (menu order); each request names its model (default: the first)."""
    default = next(iter(runners))

    def pick(model: str) -> Runner:
        if model not in runners:
            raise ValueError(f"unknown model {model!r} (one of {', '.join(runners)})")
        return runners[model]

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

    @app.get("/api/models")
    def models() -> dict:
        """The drop-down: [key, title] per model, in menu order (models.yaml)."""
        return {"models": [[k, r.spec.title] for k, r in runners.items()]}

    @app.get("/api/info")
    def info(model: str | None = None) -> dict:
        return pick(model or default).info()

    @app.post("/api/run")
    def run(body: RunBody) -> dict:
        return pick(body.model).run(body.expression, body.knockout.model_dump() if body.knockout else None, body.max_new,
                                    body.sample.model_dump() if body.sample else None)

    @app.post("/api/position")
    def position(body: PositionBody) -> dict:
        return pick(body.model).position(body.run_id, body.q)

    @app.post("/api/attention")
    def attention(body: AttentionBody) -> Response:
        return _binary(*pick(body.model).attention_map(body.run_id, body.layer, body.head))

    @app.post("/api/residual")
    def residual(body: ResidualBody) -> Response:
        return _binary(*pick(body.model).residual_map(body.run_id, body.snapshot))

    @app.post("/api/mlp")
    def mlp(body: MlpBody) -> Response:
        return _binary(*pick(body.model).mlp_map(body.run_id, body.layer))

    @app.post("/api/weights")
    def weights(body: WeightsBody) -> Response:
        return _binary_f32(*pick(body.model).weights(body.kind, body.layer, body.head, body.trim))

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
