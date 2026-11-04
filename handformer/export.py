"""Export every hand-written weight tensor.

    python -m handformer.export weights.npz            numpy archive + weights.json index
    python -m handformer.export weights.safetensors    one safetensors file, index in its metadata

Both hold the same dense float64 tensors (exact, no rounding): ``embed`` (256 x d),
``unembed`` (d x 256), per layer ``L{i}.h{j}.{w_q,w_k,w_v,w_o}`` (W_OV = W_V @ W_O) and
``L{i}.{w_in,w_out}``, plus an index naming every residual dimension, head and unit so the
tensors can be read by name.

The safetensors file is written with numpy alone, following the format's spec (8-byte
little-endian header length, a JSON header, then each tensor's raw little-endian bytes), so
exporting needs no extra package; ``safetensors.numpy.load_file`` / ``safetensors.torch``
read it as usual (tests/test_export.py checks this when the library is installed). Its
``__metadata__`` holds the index as JSON strings, and says how to run it: no LayerNorm, no
biases, unscaled scores, and ``x[:, pos] += p``, ``x[:, pos2] += p*p`` at position p.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from . import arith

log = logging.getLogger("handformer.export")


HOW_TO_RUN = (
    "Decoder-only transformer, float64, no LayerNorm, no biases, unscaled attention scores. "
    "x = embed[tokens]; x[:, pos] += p; x[:, pos2] += p*p (p = position, columns named in residual_dims). "
    "Per layer i: x += sum over heads h of causal_softmax((x W_q)(x W_k)^T) (x W_v) W_o; then "
    "x += relu(x w_in) w_out. logits = x unembed. Prompt bytes: '\\n' + expression + '='; greedy, stop at '\\n'."
)


def tensors() -> tuple[dict[str, np.ndarray], dict]:
    """Every weight tensor by name, and the index (dim names, layer and head names)."""
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
    return arrays, index


_DTYPES = {np.dtype("<f8"): "F64", np.dtype("<f4"): "F32"}


def write_safetensors(path: Path, arrays: dict[str, np.ndarray], metadata: dict[str, str]) -> None:
    """Write a safetensors file with numpy only (format spec: header length u64 LE, JSON header, data)."""
    header: dict = {"__metadata__": metadata}
    offset, blobs = 0, []
    for name, a in arrays.items():
        a = np.ascontiguousarray(a, dtype=np.dtype(a.dtype).newbyteorder("<"))
        raw = a.tobytes()
        header[name] = {"dtype": _DTYPES[a.dtype], "shape": list(a.shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    head = json.dumps(header, separators=(",", ":")).encode("utf-8")
    head += b" " * (-len(head) % 8)  # the spec allows trailing spaces; 8-byte alignment for the data
    with open(path, "wb") as f:
        f.write(len(head).to_bytes(8, "little"))
        f.write(head)
        for raw in blobs:
            f.write(raw)
    log.info("wrote %s: %d tensors, %d data bytes, %d header bytes", path, len(arrays), offset, len(head))


def read_safetensors(path: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Read a safetensors file with numpy only: (tensors by name, metadata)."""
    buf = Path(path).read_bytes()
    n = int.from_bytes(buf[:8], "little")
    header = json.loads(buf[8:8 + n])
    meta = header.pop("__metadata__", {})
    base, inv = 8 + n, {v: k for k, v in _DTYPES.items()}
    out = {}
    for name, h in header.items():
        a, b = h["data_offsets"]
        out[name] = np.frombuffer(buf[base + a: base + b], dtype=inv[h["dtype"]]).reshape(h["shape"])
    return out, meta


def export(out: Path) -> None:
    """Write ``out``: ``.npz`` (+ ``.json`` index) or ``.safetensors`` (index in the metadata)."""
    arrays, index = tensors()
    total = sum(a.size for a in arrays.values())
    nonzero = sum(int(np.count_nonzero(a)) for a in arrays.values())
    if out.suffix == ".safetensors":
        meta = {"format": "handformer", "residual_dims": json.dumps(index["residual_dims"]),
                "layers": json.dumps(index["layers"]), "how_to_run": HOW_TO_RUN,
                "nonzero_weights": str(nonzero)}
        write_safetensors(out, arrays, meta)
    elif out.suffix == ".npz":
        np.savez_compressed(out, **arrays)
        side = out.with_suffix(".json")
        side.write_text(json.dumps(index, indent=1))
        log.info("wrote %s and %s", out, side)
    else:
        raise ValueError(f"unknown export format {out.suffix!r}: use .npz or .safetensors")
    log.info("%s: %d arrays, %d entries, %d nonzero", out, len(arrays), total, nonzero)



def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m handformer.export")
    ap.add_argument("out", nargs="?", default="weights.npz", help="weights.npz or weights.safetensors")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    export(Path(args.out))


if __name__ == "__main__":
    main()
