"""Export every hand-written weight tensor: ``python -m handformer.export weights.npz``.

Writes one .npz with dense float64 arrays (embed, unembed, per-layer
L{i}.h{j}.{w_q,w_k,w_v,w_o} (W_OV = W_V @ W_O), L{i}.{w_in,w_out}) and a JSON sidecar naming every
residual dimension, head and MLP unit, so the arrays can be read by name.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from . import arith

log = logging.getLogger("handformer.export")


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m handformer.export")
    ap.add_argument("out", nargs="?", default="weights.npz")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    builder = arith.builder()
    model = builder.build()
    arrays = {"embed": model.embed, "unembed": model.unembed}
    index = {"residual_dims": model.layout.names, "layers": []}
    for li, (layer, spec) in enumerate(zip(model.layers, builder.layers)):
        entry = {"name": spec.name, "heads": [], "units": len(spec.units)}
        for hi, (h, hs) in enumerate(zip(layer.heads, spec.heads)):
            for k in ("w_q", "w_k", "w_v", "w_o"):
                arrays[f"L{li + 1}.h{hi}.{k}"] = getattr(h, k)
            entry["heads"].append(hs.name)
        arrays[f"L{li + 1}.w_in"] = layer.w_in
        arrays[f"L{li + 1}.w_out"] = layer.w_out
        index["layers"].append(entry)

    out = Path(args.out)
    np.savez_compressed(out, **arrays)
    side = out.with_suffix(".json")
    side.write_text(json.dumps(index, indent=1))
    total = sum(a.size for a in arrays.values())
    nonzero = sum(int(np.count_nonzero(a)) for a in arrays.values())
    log.info("wrote %s (%d arrays, %d entries, %d nonzero) and %s", out, len(arrays), total, nonzero, side)


if __name__ == "__main__":
    main()
