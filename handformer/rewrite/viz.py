"""The looped rewriter in the visualizer (handformer/viz/models.py: the "rewrite" entry, set aside).

The page shows decoder runs: positions, a stack of layers, a byte read out at each position.
A looped run is shown the same way, one view at a time:

* **a pass**: the tape the block read, through the block's layers. A position's "next byte" is
  the byte it holds after this pass (the hold dims, read out by a second unembedding, the
  adapter's), not the byte after it.
* **the answer**: the final tape and the answer bytes, through the coda's layers: an ordinary
  causal decode, the first byte after the root '='.

Both views use one stack, the block's 30 layers then the coda's 3, so layer numbers mean the
same in every view. The layers a view does not run (the coda in a pass; the block for the
answer, whose bytes take depth 0) are empty in that view, so nothing in them reads or writes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .. import ablate, common as cm
from ..model import Layer, LayerSpec, Tracer, Transformer
from . import builder as loop_builder, circuit, division, power, prompt as loop_prompt
from .docs import docs as loop_docs
from .engine import LoopBuilder, LoopedTransformer

log = logging.getLogger(__name__)

MAX_INPUT = 200
MAX_PASSES = 20_000  # a run longer than this is cut off (the page can ask for more)
SAMPLE = "2*(3+4)-1"
HINT = ("an expression: + - * / ^ ( ), decimals, ~N for division precision; the model loops over the "
        "prompt and writes only the answer")
ACTIONS = ("act_sp", "act_br", "rwn", "rwp", "macc", "dwork", "dsub", "grow_any", "opp", "opm", "opmul", "opdiv", "opc",
           "m_dfin", "kk", "qc")


class ViewBuilder:
    """The looped model's builder as the visualizer reads it: ``layers`` is the block's layer
    specs followed by the coda's; every other attribute is the LoopBuilder's."""

    def __init__(self, b: LoopBuilder) -> None:
        self._b = b
        self.layers: list[LayerSpec] = [*b.layers, *b.coda]
        self.embed_where = {**b.embed_where, **b.carry_where}

    def __getattr__(self, name):
        return getattr(self._b, name)

    def build(self) -> Stack:
        return Stack.of(self._b.build(), self._b)


@dataclass
class Stack:
    """One view of the looped model as a layer stack (block layers, then coda layers). ``skip``
    are the layer indices this view does not run: they are empty. ``unembed`` is the coda's
    (answer view) or the hold readout (a pass). ``loop`` is the model itself."""

    layout: object
    embed: np.ndarray
    layers: list[Layer]
    unembed: np.ndarray
    loop: LoopedTransformer
    hold_unembed: np.ndarray
    skip: frozenset = frozenset()

    head_attention = staticmethod(Transformer.head_attention)

    @classmethod
    def of(cls, m: LoopedTransformer, b: LoopBuilder) -> Stack:
        hold = np.zeros_like(m.unembed)
        for dim, byte in b.holds.items():
            hold[m.layout[dim], byte] = 1.0
        return cls(m.layout, m.embed, [*m.core, *m.coda], m.unembed, m, hold)

    def nonzero_params(self) -> int:
        return self.loop.nonzero_params()

    @property
    def n_block(self) -> int:
        return len(self.loop.core)

    def view(self, kind: str) -> Stack:
        """The pass view (block layers, hold readout) or the answer view (coda layers)."""
        d = len(self.layout)
        nb = self.n_block
        skip = set(range(nb, len(self.layers))) if kind == "pass" else set(range(nb))
        layers = [Layer(l.name, [], np.zeros((d, 0)), np.zeros((0, d))) if i in skip else l
                  for i, l in enumerate(self.layers)]
        unembed = self.hold_unembed if kind == "pass" else self.unembed
        return Stack(self.layout, self.embed, layers, unembed, self.loop, self.hold_unembed, frozenset(skip))

    def residual_from(self, x0: np.ndarray, trace: Tracer | None = None) -> np.ndarray:
        """The view's forward pass from the input residual ``x0`` (state, or embeddings; the
        positional features are added here): the block's or the coda's own computation."""
        n = len(x0)
        x = x0 + self.loop.positional(n)
        if trace is not None:
            trace.residual("state" if self.skip and min(self.skip) > 0 else "embed", x)
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


def builder() -> ViewBuilder:
    return ViewBuilder(loop_builder())


CHARS = "".join(sorted(chr(b) for b in cm.ALPHABET)) + "~ "  # the bytes the block has features for
CHARS_HINT = "(use digits . + - * / ^ ( ) and ~N)"


def prompt(text: str) -> list[int]:
    """``\n<expression>=``. Unlike the frame-writing calculator, bytes it has no features for are
    refused: such a tape never reaches a halting pass, so the run would only spend its budget.
    The same checks, in the same order, as the page's engine (viz/static/local-api.js run)."""
    if not text:
        raise ValueError("empty expression")
    if len(text) > MAX_INPUT:
        raise ValueError(f"input longer than {MAX_INPUT} characters")
    bad = sorted({ch for ch in text if ch not in CHARS})
    if bad:
        raise ValueError(f"unsupported characters: '{''.join(bad)}' {CHARS_HINT}")
    return loop_prompt(text)


def knock_out(stack: Stack, ko: ablate.Knockout) -> Stack:
    """The same knockout rules as ablate.apply, on the looped model: a stack index below the
    block's length is a block layer, the rest are the coda's; a knocked-out dim is also no
    longer carried between passes."""
    m = stack.loop
    embed, layers = ablate.apply_parts(m.layout, m.embed, [*m.core, *m.coda], ko)
    adapter = m.adapter.copy()
    for d in ko.dims:
        adapter[:, m.layout[d]] = 0.0
    nb = len(m.core)
    km = LoopedTransformer(m.layout, embed, layers[:nb], layers[nb:], adapter, m.unembed, m.router, m.grow)
    log.info("looped knockout %s: %d -> %d nonzero weights", ko.key(), m.nonzero_params(), km.nonzero_params())
    return Stack(stack.layout, embed, layers, stack.unembed, km, stack.hold_unembed)


def action(v: dict) -> str:
    """What a pass did, from the root's values (display only)."""
    if v["grow_any"] > 0.5:
        return "room"
    if v["act_sp"] > 0.5:
        return "space"
    if v["act_br"] > 0.5:
        return "brackets"
    if v["macc"] > 0.5:
        return f"x digit {round(v['kk'])}"
    if v["dsub"] > 0.5:
        return "/ subtract"
    if v["dwork"] > 0.5:
        return "/ shift"
    if v["rwp"] > 0.5:
        return "^ rewrite"
    if v["rwn"] > 0.5:
        op = "/" if v["opdiv"] > 0.5 else "*" if v["opmul"] > 0.5 else "-" if v["opm"] > 0.5 else "+"
        return f"{op} write"
    return "halt"


def category(name: str) -> str:
    if name in cm.INPUT_DIMS:
        return "input"
    if name in circuit.CARRIED:
        return "carried"
    if name in circuit.HOLD.values():
        return "output"
    if name in circuit.CODA_DIMS:
        return "coda"
    if name in power.DIMS:
        return "power"
    if name in division.DIMS:
        return "division"
    return "block"


DOCS = loop_docs()  # every dim's meaning (handformer/rewrite/docs.py)
