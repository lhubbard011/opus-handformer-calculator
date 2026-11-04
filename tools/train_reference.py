"""Train a small reference transformer on a corpus: the yardstick for the hand-built LMs.

    python tools/train_reference.py OUT.pt --layers 0|1|2 [--arch standard|engine] [--steps N] [--d 128] [--heads 4]

A standard attention-only byte-level transformer of the depth being compared (the shapes the
Transformer Circuits work analyses: 0 layers = a bigram, 1 = skip-trigrams, 2 = induction
heads): learned token and position embeddings, pre-LayerNorm attention blocks, a final
LayerNorm, an unembedding. Trained with AdamW on random windows of the training split; scored
in bits per byte on the held-out split exactly as handformer.lm.evaluate scores the hand-built
models (windows of --ctx bytes, the first byte of each window unscored). Needs torch (the
`torch` extra).

``--arch engine`` trains under the hand-built models' constraints, so the comparison is fair: no
LayerNorm, no biases, and positions only as the two features p and p^2 (fixed, not learned),
exactly as handformer.model adds them. It trains on p/ctx and (p/ctx)^2 for stability;
handformer/lm/trained.py folds that scale into the weights that read them, so the exported model
computes the same function on the engine's raw p and p^2.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from handformer.lm import corpus  # noqa: E402

log = logging.getLogger("train_reference")


class AttnOnly(nn.Module):
    def __init__(self, layers: int, d: int, heads: int, ctx: int) -> None:
        super().__init__()
        self.tok = nn.Embedding(256, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(nn.ModuleDict({"ln": nn.LayerNorm(d), "attn": nn.MultiheadAttention(d, heads, batch_first=True)}))
        self.ln = nn.LayerNorm(d)
        self.out = nn.Linear(d, 256, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        t = ids.shape[1]
        x = self.tok(ids) + self.pos(torch.arange(t))
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)
        for blk in self.blocks:
            h = blk["ln"](x)
            x = x + blk["attn"](h, h, h, attn_mask=mask, need_weights=False)[0]
        return self.out(self.ln(x))


class EngineAttnOnly(nn.Module):
    """Attention-only, no LayerNorm, no biases; the residual stream is d learned dims plus two fixed
    position features (p/ctx, (p/ctx)^2) that every projection may read and nothing writes."""

    def __init__(self, layers: int, d: int, heads: int, ctx: int) -> None:
        super().__init__()
        self.d, self.heads, self.ctx = d, heads, ctx
        self.tok = nn.Embedding(256, d)
        nn.init.normal_(self.tok.weight, std=0.5)
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(nn.ModuleDict({
                "q": nn.Linear(d + 2, d, bias=False), "k": nn.Linear(d + 2, d, bias=False),
                "v": nn.Linear(d + 2, d, bias=False), "o": nn.Linear(d, d, bias=False)}))
        self.out = nn.Linear(d + 2, 256, bias=False)

    def residual(self, ids: torch.Tensor) -> torch.Tensor:
        t = ids.shape[1]
        p = torch.arange(t, dtype=torch.float32) / self.ctx
        pos = torch.stack([p, p * p], -1).expand(ids.shape[0], t, 2)
        x = torch.cat([self.tok(ids), pos], -1)
        b, hd = ids.shape[0], self.d // self.heads
        for blk in self.blocks:
            q, k, v = (blk[n](x).view(b, t, self.heads, hd).transpose(1, 2) for n in "qkv")
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + F.pad(blk["o"](y.transpose(1, 2).reshape(b, t, self.d)), (0, 2))  # nothing writes the position dims
        return x

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.out(self.residual(ids))


ARCHS = {"standard": AttnOnly, "engine": EngineAttnOnly}


def bits_per_byte(model: nn.Module, data: np.ndarray, ctx: int) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for s in range(0, len(data) - 1, ctx):
            win = data[s:s + ctx]
            if len(win) < 2:
                break
            ids = torch.tensor(win[None, :], dtype=torch.long)
            lp = F.log_softmax(model(ids)[0, :-1], dim=-1)
            total += float(-lp[torch.arange(len(win) - 1), ids[0, 1:]].sum())
            n += len(win) - 1
    model.train()
    return total / n / math.log(2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--arch", choices=sorted(ARCHS), default="standard")
    ap.add_argument("--corpus", default="shakespeare")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    train, test = (np.frombuffer(x, dtype=np.uint8).astype(np.int64) for x in corpus.split(corpus.load(args.corpus)))
    model = ARCHS[args.arch](args.layers, args.d, args.heads, args.ctx)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps)
    log.info("training %d-layer attention-only model (%s): d %d, %d heads, %d params, %d steps of %dx%d bytes",
             args.layers, args.arch, args.d, args.heads, sum(p.numel() for p in model.parameters()), args.steps, args.batch, args.ctx)
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        starts = rng.integers(0, len(train) - args.ctx - 1, size=args.batch)
        xb = torch.tensor(np.stack([train[s:s + args.ctx + 1] for s in starts]))
        loss = F.cross_entropy(model(xb[:, :-1]).reshape(-1, 256), xb[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == args.steps:
            log.info("step %d: train loss %.4f bits/byte (%.0fs)", step, float(loss.detach()) / math.log(2), time.perf_counter() - t0)
    bpb = bits_per_byte(model, test, args.ctx)
    log.info("held-out: %.4f bits per byte (windows of %d)", bpb, args.ctx)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "args": vars(args) | {"out": str(args.out)}, "test_bpb": bpb}, args.out)
    # the weights again as plain arrays, so the engine can load them with numpy alone (handformer/lm/trained.py)
    np.savez(args.out.with_suffix(".npz"), __args__=json.dumps(vars(args) | {"out": str(args.out), "test_bpb": bpb}),
             **{k: v.detach().double().numpy() for k, v in model.state_dict().items()})
    args.out.with_suffix(".json").write_text(json.dumps({"layers": args.layers, "test_bpb": bpb, **{k: v for k, v in vars(args).items() if k != "out"}}, indent=1))
    log.info("saved %s", args.out)


if __name__ == "__main__":
    main()
