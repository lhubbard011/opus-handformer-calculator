"""Export the model as an anonymous, vanilla PyTorch package.

    python tools/export_torch.py OUT_DIR [--seed N]

Writes OUT_DIR/model.py (a generic decoder-only transformer, no arithmetic
anywhere in it), OUT_DIR/config.json and OUT_DIR/weights.pt. Nothing that
was named at build time survives: residual dimensions are shuffled with a
random permutation, heads and MLP units are shuffled within each layer, and
parameter keys are generic. All shuffles are exact, so the package computes
the identical function.

``--harden`` adds, all exact: random power-of-two scales on every residual dim, head
column and MLP unit (inverse applied on the reading side); 2*32 decoy dims, half
written with noise but never read, half read with noise but never written (always 0),
so every byte has non-empty embedding and output rows; and no descriptive comments.

Attention is standard packed multi-head attention via
F.scaled_dot_product_attention (causal). Heads are zero-padded to a common
width per layer, and the 1/sqrt(d_k) scaling of that function is folded into
W_Q so the scores are unchanged. float64 throughout: the weights hold
constants up to ~1e13 and squared positions.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from handformer import arith  # noqa: E402

log = logging.getLogger("export_torch")

MODEL_PY = '''"""A small decoder-only transformer (byte vocabulary, float64)."""
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v):
        super().__init__()
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v
        self.q_proj = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.o_proj = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, x):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_heads, self.d_v).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(y.transpose(1, 2).reshape(b, t, self.n_heads * self.d_v))


class MLP(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, d_ff):
        super().__init__()
        self.attn = Attention(d_model, n_heads, d_k, d_v) if n_heads else None
        self.mlp = MLP(d_model, d_ff) if d_ff else None

    def forward(self, x):
        if self.attn is not None:
            x = x + self.attn(x)
        if self.mlp is not None:
            x = x + self.mlp(x)
        return x


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config["d_model"]
        self.embed = nn.Embedding(config["vocab_size"], d)
        self.register_buffer("pos_proj", torch.zeros(2, d, dtype=torch.float64))  # features p and p**2
        self.blocks = nn.ModuleList(Block(d, *spec) for spec in config["blocks"])
        self.lm_head = nn.Linear(d, config["vocab_size"], bias=False)

    def forward(self, ids):
        t = ids.shape[-1]
        p = torch.arange(t, dtype=torch.float64)
        x = self.embed(ids) + p[:, None] * self.pos_proj[0] + (p * p)[:, None] * self.pos_proj[1]
        for block in self.blocks:
            x = block(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, ids, max_new_tokens, stop_id=None):
        ids = list(ids)
        for _ in range(max_new_tokens):
            nxt = int(self(torch.tensor([ids]))[0, -1].argmax())
            ids.append(nxt)
            if nxt == stop_id:
                break
        return ids


def load(directory=os.path.dirname(os.path.abspath(__file__))):
    with open(f"{directory}/config.json") as f:
        config = json.load(f)
    model = Model(config).to(torch.float64)
    model.load_state_dict(torch.load(f"{directory}/weights.pt", weights_only=True))
    return model.eval()
'''

RUN_PY = '''"""Feed raw bytes to the model and print its greedy continuation.

    python run.py "some text"        (escape sequences like \\\\n are interpreted)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import load  # noqa: E402

torch.set_grad_enabled(False)
model = load()
for text in sys.argv[1:]:
    prompt = list(text.encode().decode("unicode_escape").encode("latin-1"))
    out = model.generate(prompt, max_new_tokens=4000, stop_id=10)
    print(repr(bytes(out[len(prompt):]).decode("latin-1")))
'''


def export(out: Path, seed: int, shuffle: bool = True, harden: bool = False, decoys: int = 32) -> None:
    """Write the package. ``harden``: also rescale and add decoy dims (see module docstring)."""
    rng = np.random.default_rng(seed)
    m = arith.build()
    d0 = len(m.layout)
    k = decoys if harden else 0
    d = d0 + 2 * k  # [real dims | written-never-read decoys | read-never-written decoys]
    J1, J2 = np.arange(d0, d0 + k), np.arange(d0 + k, d)
    order_of = (lambda n: rng.permutation(n)) if (shuffle or harden) else np.arange  # noqa: E731
    pow2 = (lambda n: 2.0 ** rng.integers(-6, 7, size=n)) if harden else (lambda n: np.ones(n))  # noqa: E731
    pool = np.concatenate([a[a != 0].ravel() for a in [m.embed, m.unembed] + [w for l in m.layers for w in
                           (l.w_in, l.w_out, *(x for h in l.heads for x in (h.w_q, h.w_k, h.w_v, h.w_o)))]])

    def noise(shape, density):
        z = rng.choice(pool, size=shape) * rng.choice([-1.0, 1.0], size=shape)
        return z * (rng.random(shape) < density)

    perm = order_of(d)  # extended dim -> exported dim
    P = np.zeros((d, d))
    P[np.arange(d), perm] = 1.0
    T = np.diag(pow2(d)) @ P  # x_exported = x_extended @ T (exact: powers of two)
    Tinv = np.linalg.inv(T)

    def ext_rows(w):  # (d0, c) reader/writer row block -> (d, c)
        return np.vstack([w, np.zeros((2 * k, w.shape[1]))])

    def writer(w_cols_d0):  # (r, d0) -> exported (r, d): real columns + noise into J1 decoys
        w = np.hstack([w_cols_d0, np.zeros((w_cols_d0.shape[0], 2 * k))])
        if k:
            w[:, J1] = noise((w.shape[0], k), 0.35)
        return w @ T

    def reader(w_rows_d0, density=0.05):  # (d0, c) -> exported (d, c): real rows + noise from J2 decoys
        w = ext_rows(w_rows_d0)
        if k:
            w[J2] = noise((k, w.shape[1]), density)
        return Tinv @ w

    state: dict[str, torch.Tensor] = {}
    t = lambda a: torch.tensor(a, dtype=torch.float64)  # noqa: E731
    emb = writer(m.embed)
    if k:  # every byte gets a non-empty embedding row
        emb_ext = emb @ Tinv
        emb_ext[:, J1] = noise((256, k), 1.0)
        emb = emb_ext @ T
    state["embed.weight"] = t(emb)
    pos = np.zeros((2, d0))
    pos[0, m.layout["pos"]] = pos[1, m.layout["pos2"]] = 1.0
    state["pos_proj"] = t(np.hstack([pos, np.zeros((2, 2 * k))]) @ T)
    state["lm_head.weight"] = t(reader(m.unembed, density=1.0).T)  # every byte gets a non-empty output row

    blocks = []
    for li, layer in enumerate(m.layers):
        heads = [layer.heads[i] for i in order_of(len(layer.heads))]
        H = len(heads)
        dk = max((h.w_q.shape[1] for h in heads), default=0)
        dv = max((h.w_v.shape[1] for h in heads), default=0)
        if H:
            wq, wk = np.zeros((d0, H * dk)), np.zeros((d0, H * dk))
            wv, wo = np.zeros((d0, H * dv)), np.zeros((H * dv, d0))
            cq, cv = pow2(H * dk), pow2(H * dv)  # score = (q*c).(k/c); value (v*e) -> out (o/e)
            for j, h in enumerate(heads):
                nq, nv = h.w_q.shape[1], h.w_v.shape[1]
                wq[:, j * dk:j * dk + nq] = h.w_q * np.sqrt(dk)  # undo SDPA's 1/sqrt(d_k)
                wk[:, j * dk:j * dk + nq] = h.w_k
                wv[:, j * dv:j * dv + nv] = h.w_v
                wo[j * dv:j * dv + nv, :] = h.w_o
            pre = f"blocks.{li}.attn"
            state[f"{pre}.q_proj.weight"] = t((reader(wq) * cq).T)
            state[f"{pre}.k_proj.weight"] = t((reader(wk) / cq).T)
            state[f"{pre}.v_proj.weight"] = t((reader(wv) * cv).T)
            state[f"{pre}.o_proj.weight"] = t(writer(wo / cv[:, None]).T)
        u = layer.w_in.shape[1]
        if u:
            order, g = order_of(u), pow2(u)  # ReLU(g*z)/g == ReLU(z) for g > 0
            state[f"blocks.{li}.mlp.fc1.weight"] = t((reader(layer.w_in[:, order]) * g).T)
            state[f"blocks.{li}.mlp.fc2.weight"] = t(writer(layer.w_out[order] / g[:, None]).T)
        blocks.append([H, dk, dv, u])
        log.info("block %d: %d heads (d_k=%d, d_v=%d), d_ff=%d", li, H, dk, dv, u)

    out.mkdir(parents=True, exist_ok=True)
    torch.save(state, out / "weights.pt")
    (out / "config.json").write_text(json.dumps({"vocab_size": 256, "d_model": d, "blocks": blocks}, indent=1))
    model_py = MODEL_PY
    if harden:  # drop the two descriptive hints
        model_py = model_py.replace('"""A small decoder-only transformer (byte vocabulary, float64)."""\n', "")
        model_py = model_py.replace("  # features p and p**2", "")
    (out / "model.py").write_text(model_py)
    (out / "run.py").write_text(RUN_PY)
    nz = sum(int(torch.count_nonzero(v)) for v in state.values())
    log.info("wrote %s: d_model=%d (%d decoys), %d tensors, %d entries, %d nonzero",
             out, d, 2 * k, len(state), sum(v.numel() for v in state.values()), nz)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-shuffle", action="store_true", help="keep dims, heads and units in build order")
    ap.add_argument("--harden", action="store_true",
                    help="also: power-of-two rescaling, decoy dims (written-never-read / read-never-written), no hints")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.out.exists():
        shutil.rmtree(args.out)
    export(args.out, args.seed, shuffle=not args.no_shuffle, harden=args.harden)


if __name__ == "__main__":
    main()
