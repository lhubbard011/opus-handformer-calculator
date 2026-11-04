"""Run the exported weights with a from-scratch forward pass. Imports only numpy.

Usage: python run_weights.py weights.npz "2^10" ...
       python run_weights.py weights.safetensors "2^10" ...
The only facts used beyond the arrays: the model is a decoder-only transformer
(causal softmax attention summed over heads, then a ReLU MLP, residual adds,
no LayerNorm), and two residual columns carry the position p and p^2.
"""
import json
import logging
import sys

import numpy as np

log = logging.getLogger("standalone")


def load(path: str) -> tuple[dict, dict]:
    """Tensors by name and the index (residual dim names, layers), from .npz + .json or .safetensors."""
    if path.endswith(".safetensors"):  # format: u64 LE header length, JSON header, raw tensor bytes
        buf = open(path, "rb").read()
        n = int.from_bytes(buf[:8], "little")
        header = json.loads(buf[8:8 + n])
        meta = header.pop("__metadata__")
        dtypes = {"F64": "<f8", "F32": "<f4"}
        W = {k: np.frombuffer(buf[8 + n + h["data_offsets"][0]: 8 + n + h["data_offsets"][1]],
                              dtype=dtypes[h["dtype"]]).reshape(h["shape"]) for k, h in header.items()}
        return W, {"residual_dims": json.loads(meta["residual_dims"]), "layers": json.loads(meta["layers"])}
    npz = np.load(path)
    return {k: npz[k] for k in npz.files}, json.load(open(path.replace(".npz", ".json")))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    path, exprs = sys.argv[1], sys.argv[2:]
    W, index = load(path)
    dims = index["residual_dims"]
    POS, POS2 = dims.index("pos"), dims.index("pos2")  # the positional feature columns
    layers = [(sorted({k.split(".")[1] for k in W if k.startswith(f"L{i}.h")}, key=lambda h: int(h[1:])),
               W[f"L{i}.w_in"], W[f"L{i}.w_out"]) for i in range(1, len(index["layers"]) + 1)]
    log.info("loaded %d arrays, %d layers, d_model=%d", len(W), len(layers), W["embed"].shape[1])

    def logits(tokens):
        n = len(tokens)
        x = W["embed"][tokens].copy()
        x[:, POS] += np.arange(n)
        x[:, POS2] += np.arange(n) ** 2
        mask = np.triu(np.ones((n, n), bool), 1)
        for i, (heads, w_in, w_out) in enumerate(layers, start=1):
            delta = np.zeros_like(x)
            for h in heads:
                q, k = x @ W[f"L{i}.{h}.w_q"], x @ W[f"L{i}.{h}.w_k"]
                s = q @ k.T
                s[mask] = -np.inf
                a = np.exp(s - s.max(1, keepdims=True))
                a /= a.sum(1, keepdims=True)
                delta += (a @ (x @ W[f"L{i}.{h}.w_v"])) @ W[f"L{i}.{h}.w_o"]
            x = x + delta
            x = x + np.maximum(x @ w_in, 0) @ w_out
        return x[-1] @ W["unembed"]

    for e in exprs:
        tokens = list(f"\n{e}=".encode())
        for _ in range(5000):
            nxt = int(np.argmax(logits(tokens)))
            tokens.append(nxt)
            if nxt == 10:
                break
        out = bytes(tokens).decode()
        log.info("%s  ->  %s", e, out.split("=", 1)[1].rsplit("]", 1)[-1].strip())


if __name__ == "__main__":
    main()
