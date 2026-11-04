"""Export the model for the in-browser site: sparse weights + visualizer info.

    python tools/export_web.py OUT_DIR

Writes OUT_DIR/model.json (every nonzero weight, exact: Python float repr
round-trips through JSON into JavaScript float64) and OUT_DIR/info.json (the
same model description the server's /api/info returns: names, meanings,
wiring, head/unit formulas). The Python model stays the source of truth.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from handformer import common as cm  # noqa: E402
from handformer.viz import runner as rn  # noqa: E402
from handformer.viz.runner import Runner  # noqa: E402

log = logging.getLogger("export_web")


def sparse(mat: np.ndarray) -> list[list]:
    """Nonzero entries of a 2-D array as [row, col, value]."""
    r, c = np.nonzero(mat)
    return [[int(i), int(j), float(mat[i, j])] for i, j in zip(r, c)]


def export(out: Path) -> None:
    runner = Runner()
    m = runner.model
    layers = []
    for layer in m.layers:
        heads = [{"dk": h.w_q.shape[1], "dv": h.w_v.shape[1], "q": sparse(h.w_q), "k": sparse(h.w_k),
                  "v": sparse(h.w_v), "o": sparse(h.w_o)} for h in layer.heads]
        layers.append({"name": layer.name, "heads": heads, "units": int(layer.w_in.shape[1]),
                       "win": sparse(layer.w_in), "wout": sparse(layer.w_out)})
    model = {
        "model_id": runner.model_id,
        "d": len(m.layout),
        "pos": m.layout["pos"],
        "pos2": m.layout["pos2"],
        "eos": cm.EOS,
        "names": list(m.layout.names),
        # Host-side constants the server uses, so the page behaves identically.
        "input_chars": "".join(sorted(chr(b) for b in cm.ALPHABET)) + "~",
        "max_tokens": rn.MAX_TOKENS,
        "top_keys": rn.TOP_KEYS,
        "top_logits": rn.TOP_LOGITS,
        "summary_dims": list(rn.SUMMARY_DIMS),
        "byte_labels": [repr(chr(i)) for i in range(256)],
        "embed": sparse(m.embed),
        "unembed": sparse(m.unembed),
        "layers": layers,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.json").write_text(json.dumps(model, separators=(",", ":")))
    (out / "info.json").write_text(json.dumps(runner.info(), separators=(",", ":")))
    nz = len(model["embed"]) + len(model["unembed"]) + sum(
        len(h[k]) for l in layers for h in l["heads"] for k in "qkvo") + sum(len(l["win"]) + len(l["wout"]) for l in layers)
    log.info("wrote %s/model.json (%d bytes, %d nonzero weights) and info.json (%d bytes)", out,
             (out / "model.json").stat().st_size, nz, (out / "info.json").stat().st_size)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    export(args.out)


if __name__ == "__main__":
    main()
