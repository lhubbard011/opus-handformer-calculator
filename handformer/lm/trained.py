"""A trained reference model, loaded into the hand-built models' engine for side-by-side study.

``tools/train_reference.py --arch engine`` trains an attention-only transformer under the engine's
constraints (no LayerNorm, no biases, positions only as p and p^2). This module turns its
checkpoint into a ``ModelBuilder``: the same kind of object every hand-built model is, so the
visualizer, the knockouts and the PyTorch export all work on it unchanged. Its dims have no
meanings, only names: ``d0`` ... ``d127`` (learned), plus ``pos`` and ``pos2``.

Two exact rewrites, and nothing else:

* the trainer feeds p/ctx and (p/ctx)^2; the engine feeds p and p^2. Every weight that reads the
  position dims is divided by ctx (or ctx^2), so the scores and values are unchanged;
* ``scaled_dot_product_attention`` divides scores by sqrt(d_head); the engine does not, so the
  query weights are divided by it instead.

``check`` runs both and requires the same logits (float32 training, float64 engine: 1e-4).
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

from ..model import HeadSpec, Layout, ModelBuilder, Transformer

log = logging.getLogger(__name__)


def _load(path: Path):
    """(checkpoint-like dict, weights as float64 arrays): from the .npz the trainer writes next to the
    .pt (numpy only), else from the .pt itself (needs torch)."""
    npz = path.with_suffix(".npz")
    if npz.exists():
        import json

        with np.load(npz) as z:
            args = json.loads(str(z["__args__"]))
            w = {k: z[k].astype(np.float64) for k in z.files if k != "__args__"}
        ck = {"args": args, "test_bpb": args.get("test_bpb")}
    else:
        import torch  # the `torch` extra: only needed to read a .pt

        ck = torch.load(path, map_location="cpu", weights_only=False)
        w = {k: v.double().numpy() for k, v in ck["state"].items()}
    if ck["args"].get("arch") != "engine":
        raise ValueError(f"{path}: trained with --arch {ck['args'].get('arch', 'standard')}; only --arch engine "
                         "models fit the engine (no LayerNorm, no biases)")
    return ck, w


def dims(d: int) -> list[str]:
    return [*(f"d{i}" for i in range(d)), "pos", "pos2"]


def builder(path: Path) -> ModelBuilder:
    ck, w = _load(path)
    a = ck["args"]
    d, H, ctx, L = a["d"], a["heads"], a["ctx"], a["layers"]
    hd = d // H
    names = dims(d)
    lay = Layout()
    lay.add(*names)
    b = ModelBuilder(lay)
    # rows of every reader: x_trained = x_engine * scale (the position dims are divided by ctx, ctx^2)
    scale = np.ones(d + 2)
    scale[d], scale[d + 1] = 1.0 / ctx, 1.0 / ctx ** 2

    def form(col: np.ndarray) -> dict[str, float]:
        return {n: float(c) for n, c in zip(names, col) if c != 0.0}

    for byte in range(256):
        b.embed_byte(byte, form(np.concatenate([w["tok.weight"][byte], [0.0, 0.0]])))
    for li in range(L):
        pre = f"blocks.{li}."
        wq = w[pre + "q.weight"].T * scale[:, None] / math.sqrt(hd)  # (d+2, d): reads x_engine
        wk = w[pre + "k.weight"].T * scale[:, None]
        wv = w[pre + "v.weight"].T * scale[:, None]
        wo = w[pre + "o.weight"].T  # (d, d): writes the learned dims only
        layer = b.at(li + 1)
        for h in range(H):
            sl = slice(h * hd, (h + 1) * hd)
            ov = wv[:, sl] @ wo[sl]  # (d+2, d): what reading each dim writes (the fused OV map)
            layer.heads.append(HeadSpec(
                name=f"L{li + 1} head {h} (trained)",
                query=[form(wq[:, c]) for c in range(sl.start, sl.stop)],
                key=[form(wk[:, c]) for c in range(sl.start, sl.stop)],
                value={names[i]: form(np.concatenate([ov[i], [0.0, 0.0]])) for i in range(d + 2) if np.any(ov[i])},
                where=f"{path.name}: blocks.{li} head {h}",
            ))
    wu = w["out.weight"].T * scale[:, None]  # (d+2, 256)
    for i, n in enumerate(names):
        for byte in range(256):
            if wu[i, byte] != 0.0:
                b.logit(n, byte, float(wu[i, byte]))
    log.info("trained reference %s: %d layer(s), d %d (+pos, pos2), %d heads, test %.4f bits per byte when trained",
             path.name, L, d, H * L, ck.get("test_bpb", float("nan")))
    return b


def check(path: Path, m: Transformer, text: bytes) -> float:
    """Max |logit difference| between the engine and the PyTorch checkpoint on ``text``."""
    import sys

    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
    from train_reference import ARCHS  # noqa: PLC0415  (the trainer's own model class)

    ck, w = _load(path)
    a = ck["args"]
    tm = ARCHS["engine"](a["layers"], a["d"], a["heads"], a["ctx"])
    tm.load_state_dict({k: torch.tensor(v, dtype=torch.float32) for k, v in w.items()})
    tm.eval()
    ids = list(text[: a["ctx"]])
    with torch.no_grad():
        want = tm(torch.tensor([ids])).double().numpy()[0]
    got = m.logits(ids)
    err = float(np.max(np.abs(got - want)))
    log.info("engine vs PyTorch on %d bytes: max logit difference %.2e", len(ids), err)
    return err
