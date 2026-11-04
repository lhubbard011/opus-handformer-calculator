"""Scoring a language model: bits per byte on held-out text.

bits per byte = the mean of -log2 P(the byte that actually came next), with P the softmax of the
model's logits. The text is scored in windows of ``ctx`` bytes, each fed after the sink byte
(circuit.SINK, which conditional heads attend to when their condition fails): every byte of a
window from the third on is predicted from the window so far. The first two are not scored (the
first has no byte before it; the second's byte before it is the sink), which drops 2 bytes in
``ctx``, for every stage alike, so the stages are scored on exactly the same bytes.
"""
from __future__ import annotations

import logging
import time

import numpy as np

from ..model import Transformer
from .circuit import SINK

log = logging.getLogger(__name__)

LN2 = np.log(2.0)


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def byte_costs(model: Transformer, data: bytes, ctx: int = 512) -> np.ndarray:
    """-log2 P(byte) for every scored byte of ``data`` (see the module docstring)."""
    t0 = time.perf_counter()
    x = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
    out = []
    for s in range(0, len(x), ctx):
        win = x[s:s + ctx]
        if len(win) < 3:
            break
        lp = log_softmax(model.logits([SINK, *win.tolist()]))  # position p + 1 holds win[p]
        p = np.arange(1, len(win) - 1)  # predict win[p + 1] from win[..p]
        out.append(-lp[p + 1, win[p + 1]] / LN2)
    costs = np.concatenate(out)
    log.info("scored %d bytes in windows of %d in %.1fs: %.4f bits per byte", len(costs), ctx,
             time.perf_counter() - t0, costs.mean())
    return costs


def bits_per_byte(model: Transformer, data: bytes, ctx: int = 512) -> float:
    return float(byte_costs(model, data, ctx).mean())


def sample(model: Transformer, prompt: bytes, n: int, seed: int = 0, ctx: int = 256) -> bytes:
    """``n`` bytes drawn from the model's distribution (the host samples, the model only scores)."""
    rng = np.random.default_rng(seed)
    out = list(prompt)
    for _ in range(n):
        lp = log_softmax(model.logits([SINK, *out[-ctx:]])[-1])
        out.append(int(rng.choice(len(lp), p=np.exp(lp))))
    return bytes(out)
