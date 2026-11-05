"""The recurrent-depth calculator in the visualizer (handformer/viz/models.py: the "loop" entry).

The page shows decoder runs: positions, a stack of layers, a byte read out at each position.
A recurrent-depth run is one: every position, at its last iteration, through the block's
layers. A later position reads an earlier one through that iteration's keys and values (the
KV cache), so tracing the block once over the last iterations' inputs (each position's input
``e`` plus its loop state) gives exactly the residuals the run computed. The readout is the
coda: the hold dims' logits, the next byte, which for a thought is the next cell's byte.

A position that iterated (a product column's predictor, a quotient digit's) can be shown at
any of its iterations: the positions before it at their last iteration, it at that one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .. import ablate, common as cm
from ..model import Layer, LayerSpec, Tracer, Transformer
from . import ROLES, builder as depth_builder, columns, division, parse, parts as P, power, prompt as depth_prompt
from . import walk
from .engine import DepthBuilder, DepthResult, RecurrentDepthLM

log = logging.getLogger(__name__)

MAX_INPUT = 200
MAX_TOKENS = 20_000  # positions after the prompt a page run may take (the page can ask for more)
MAX_ITERS = 2_000  # one position's depth in a page run
SAMPLE = "2*(3+4)-1"
HINT = ("an expression: + - * / ^ ( ), decimals, ~N for division precision; the model thinks in latent "
        "positions and writes only the answer")
CHARS = "".join(sorted(chr(b) for b in cm.ALPHABET)) + "~ "  # the bytes the model has features for
CHARS_HINT = "(use digits . + - * / ^ ( ) and ~N)"
# what a latent position is, from its input (display only)
KINDS = ("prompt", "start", "column", "quotient", "remainder", "copy", "root", "answer")


class ViewBuilder:
    """The model's builder as the visualizer reads it (``embed_where`` also names the lines that
    carry a dim through T or the adapter)."""

    def __init__(self, b: DepthBuilder) -> None:
        self._b = b
        self.layers: list[LayerSpec] = b.layers
        self.embed_where = {**b.embed_where, **b.carry_where}

    def __getattr__(self, name):
        return getattr(self._b, name)

    def build(self) -> Stack:
        return Stack.of(self._b.build())


@dataclass
class Stack:
    """The block as the page's layer stack; ``depth`` is the recurrent-depth model itself."""

    layout: object
    embed: np.ndarray
    layers: list[Layer]
    unembed: np.ndarray
    depth: RecurrentDepthLM
    head_attention = staticmethod(Transformer.head_attention)

    @classmethod
    def of(cls, m: RecurrentDepthLM) -> Stack:
        return cls(m.layout, m.embed, m.block, m.unembed, m)

    def nonzero_params(self) -> int:
        return self.depth.nonzero_params()

    def residual_from(self, x0: np.ndarray, trace: Tracer | None = None) -> np.ndarray:
        """The block over rows ``x0`` (inputs plus loop state; positions are added here), causal:
        one iteration of every row, each reading the rows before it."""
        n = len(x0)
        x = x0 + self.depth.positional(n)
        if trace is not None:
            trace.residual("input", x)
        causal = np.triu(np.ones((n, n), dtype=bool), k=1)
        for li, layer in enumerate(self.layers):
            delta = np.zeros_like(x)
            for hi, h in enumerate(layer.heads):
                attn = Transformer.head_attention(x, h, causal)
                if trace is not None:
                    trace.attention(li, hi, attn)
                delta += (attn @ (x @ h.w_v)) @ h.w_o
            x = x + delta
            if trace is not None:
                trace.residual(f"{layer.name} attn", x)
            if layer.w_in.shape[1]:
                pre = x @ layer.w_in
                if trace is not None:
                    trace.mlp(li, pre)
                x = x + np.maximum(pre, 0.0) @ layer.w_out
            if trace is not None:
                trace.residual(f"{layer.name} mlp", x)
        return x


@dataclass
class Record:
    """A run with every iteration's input kept (without positions), for the page's views."""

    res: DepthResult
    last_in: np.ndarray  # (n, d): each position's last iteration input
    earlier: dict = field(default_factory=dict)  # (position, iteration) -> input, for the others
    kinds: list[str] = field(default_factory=list)
    chars: list[str] = field(default_factory=list)
    cap: int = MAX_TOKENS


def record(stack: Stack, prompt: list[int], max_tokens: int) -> Record:
    m = stack.depth
    d = len(m.layout)
    last: dict[int, np.ndarray] = {}
    earlier: dict[tuple[int, int], np.ndarray] = {}
    pos_cache: dict[int, np.ndarray] = {}

    def on_iter(q, k, x_in, x):
        if q not in pos_cache:
            pos_cache[q] = m.positional(q + 1)[q]
        row = x_in - pos_cache[q]
        if k:
            earlier[(q, k - 1)] = last[q]
        last[q] = row

    res = m.run(prompt, stop=cm.EOS, max_tokens=max_tokens, max_iters=MAX_ITERS, on_iter=on_iter)
    n = len(res.tokens)
    last_in = np.zeros((n, d))
    for q in range(n):
        last_in[q] = last[q]
    kinds, chars = describe(m, res)
    return Record(res, last_in, earlier, kinds, chars, max_tokens)


def describe(m: RecurrentDepthLM, res: DepthResult) -> tuple[list[str], list[str]]:
    """Every position's kind and the character the page shows for it (display only: a thought's
    byte is the nearest embedding row of its byte features; a working position shows its digit)."""
    lay = m.layout
    feat = [j for j in np.flatnonzero(np.abs(m.embed).sum(axis=0)) if j != lay["one"]]
    rows = m.embed[:, feat]
    kinds, chars = [], []
    n_prompt = len(res.prompt)
    for t in res.tokens:
        e = res.inputs[t.index]
        if not t.latent:
            kinds.append("prompt" if t.index < n_prompt else "answer")
            chars.append(chr(t.byte))
            continue
        role = next((r for r in ("k_s", "k_w", "k_q", "k_r") if e[lay[r]] > 0.5), None)
        if role is None:
            ch = chr(int(((e[feat][None] - rows) ** 2).sum(axis=1).argmin()))
            kinds.append("root" if ch == "=" else "copy")
            chars.append(ch)
            continue
        kind, shown = {"k_s": ("start", None), "k_w": ("column", "dA"), "k_q": ("quotient", "qd"),
                       "k_r": ("remainder", "rd")}[role]
        kinds.append(kind)
        chars.append("▸" if shown is None else str(int(round(e[lay[shown]]))))
    return kinds, chars


def builder() -> ViewBuilder:
    return ViewBuilder(depth_builder())


def prompt(text: str) -> list[int]:
    """``\\n<expression>=``. Bytes the model has no features for are refused (such a copy never
    halts). The same checks, in the same order, as the page's engine (viz/static/local-api.js)."""
    if not text:
        raise ValueError("empty expression")
    if len(text) > MAX_INPUT:
        raise ValueError(f"input longer than {MAX_INPUT} characters")
    bad = sorted({ch for ch in text if ch not in CHARS})
    if bad:
        raise ValueError(f"unsupported characters: '{''.join(bad)}' {CHARS_HINT}")
    return depth_prompt(text)


def knock_out(stack: Stack, ko: ablate.Knockout) -> Stack:
    """The same knockout rules as ablate.apply, on the block; a knocked-out dim is also no longer
    carried by the adapter or T."""
    m = stack.depth
    embed, layers = ablate.apply_parts(m.layout, m.embed, m.block, ko)
    adapter, thought = m.adapter.copy(), m.thought.copy()
    for d in ko.dims:
        j = m.layout[d]
        adapter[:, j] = thought[:, j] = 0.0
        adapter[j, :] = thought[j, :] = 0.0
    km = RecurrentDepthLM(m.layout, embed, layers, m.unembed, adapter, thought, m.router, m.think)
    log.info("recurrent-depth knockout %s: %d -> %d nonzero weights", ko.key(), m.nonzero_params(), km.nonzero_params())
    return Stack.of(km)


def category(name: str) -> str:
    if name in cm.INPUT_DIMS or name in ("is_cell",):
        return "input"
    if name in ROLES or name.startswith("n_") or name.startswith("x_"):
        return "latent"
    if name in P.HOLD.values():
        return "output"
    if name.startswith("C_"):
        return "plan"
    for mod, cat in ((parse, "root"), (power, "power"), (walk, "walk"), (columns, "columns"), (division, "division")):
        if name in mod.DIMS or name in getattr(mod, "REGS", ()) or name in getattr(mod, "STATE", ()):
            return cat
    return "block"
