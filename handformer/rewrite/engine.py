"""A looped transformer with hand-written weights: prelude, recurrent block, router, coda.

The layout follows current looped language models (REWRITE.md §2):

* **Prelude** (Huginn, Geiping et al. 2025): the byte embedding. Pass 0's state is the embedding
  of the prompt.
* **Recurrent block** (Universal Transformer, Dehghani et al. 2018): one fixed stack of layers,
  applied again and again to the whole prompt. Each pass reads ``state + positional features``
  (UT re-adds its coordinate embedding at every step) and runs the block.
* **Adapter** (Huginn's linear map between iterations): a fixed linear map from the block's last
  residual to the next state. Here it re-embeds the byte each position chose to hold next and
  carries the dims an operator keeps between passes. Everything else, the block's scratch, is
  dropped.
* **Router** (Mixture-of-Recursions, Bae et al. 2025; exit gates as in Ouro, Zhu et al. 2025):
  a per-position dim the block computes. A position whose router is 1 goes through another pass.
  When no position routes in, the loop ends. Answer bytes are written after the loop and take
  depth 0: they never enter the block.
* **Coda**: a plain causal decoder over the final state that writes the answer, one byte per
  position, the first byte straight after the prompt's '='.

One addition has no counterpart in those models: **room**. Positions are the block's memory, and a
result can be longer than the text it replaces (``2^100``). The block raises the ``grow`` dim at
the root (the prompt's last position) and the runtime inserts one blank position (a space)
before the root, so the tape has room. Like stopping at the end byte, the host acts on one
flag the weights computed. It never reads or changes what the positions hold.

Inside the block, heads may be bidirectional (``HeadSpec.full``). In-place rewriting needs it:
under a causal mask, information never flows from later positions to earlier ones, however many
passes run (REWRITE.md §3).
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..model import Form, Layer, LayerSpec, ModelBuilder, Transformer, Tracer, _DecodeCache

log = logging.getLogger(__name__)

BLANK = ord(" ")  # the byte of an inserted (room) position
_REPO = Path(__file__).resolve().parents[2]


def _caller() -> str:
    """``path:line`` of the circuit line that called a LoopBuilder method (docs only)."""
    f = sys._getframe(2)
    return f"{Path(f.f_code.co_filename).resolve().relative_to(_REPO).as_posix()}:{f.f_lineno}"


class LoopBuilder(ModelBuilder):
    """``ModelBuilder`` for a looped model. ``at(i)`` is layer i of the recurrent block,
    ``coda_at(i)`` layer i of the coda; one residual layout serves both. ``carry(dst, form)`` adds
    to the adapter: the next pass's ``dst`` is ``form`` of this pass's last residual.
    ``hold(dim, byte)`` makes ``dim`` the "hold byte next" choice: its value times that byte's
    embedding row goes into the next state."""

    def __init__(self, layout, router: str, grow: str) -> None:
        super().__init__(layout)
        self.coda: list[LayerSpec] = []
        self.adapter: dict[str, Form] = {}
        self.holds: dict[str, int] = {}
        self.carry_where: dict[str, str] = {}  # carried dim -> the line that carries it (docs only)
        self.router, self.grow = router, grow

    def coda_at(self, index: int) -> LayerSpec:
        while len(self.coda) < index:
            self.coda.append(LayerSpec(f"C{len(self.coda) + 1}"))
        return self.coda[index - 1]

    def carry(self, dst: str, form: Form) -> None:
        if dst in self.adapter:
            raise ValueError(f"adapter already writes {dst!r}")
        self.adapter[dst] = dict(form)
        self.carry_where[dst] = _caller()

    def hold(self, dim: str, byte: int) -> None:
        if dim in self.holds:
            raise ValueError(f"{dim!r} already holds byte {self.holds[dim]}")
        self.holds[dim] = byte

    def build(self) -> LoopedTransformer:
        embed, unembed = self._embed_unembed()
        core, coda = self._layers(self.layers), self._layers(self.coda)
        bad = [f"{l.name}/{h.name}" for l in coda for h in l.heads if h.full]
        if bad:
            raise ValueError(f"the coda decodes causally; bidirectional heads there: {', '.join(bad)}")
        d = len(self.layout)
        adapter = np.zeros((d, d))  # next_state = x_last @ adapter
        for dst, form in self.adapter.items():
            adapter[:, self.layout[dst]] += self._vec(form)
        for dim, byte in self.holds.items():
            adapter[self.layout[dim]] += embed[byte]
        for name in ("pos", "pos2"):  # positions are re-added every pass, never carried
            if np.any(adapter[:, self.layout[name]]):
                raise ValueError(f"the adapter must not write the positional feature {name!r}")
        model = LoopedTransformer(self.layout, embed, core, coda, adapter, unembed,
                                  self.layout[self.router], self.layout[self.grow])
        log.info("built looped model: d_model=%d block layers=%d (heads %d, %d bidirectional; units %d) "
                 "coda layers=%d nonzero_params=%d", d, len(core), sum(len(l.heads) for l in core),
                 sum(h.full for l in core for h in l.heads), sum(l.w_in.shape[1] for l in core), len(coda),
                 model.nonzero_params())
        return model


@dataclass
class Pass:
    """What one pass of the block did (for tests and the visualizer; never fed back)."""

    index: int
    tape: list[int]  # the bytes the positions held going in (argmax of their state's byte features)
    routed: int  # positions whose router asked for another pass
    grew: bool  # the root asked for room: a blank was inserted before it


@dataclass
class LoopResult:
    prompt: list[int]
    answer: list[int]  # the coda's bytes, ending with the stop byte (if reached)
    passes: list[Pass] = field(default_factory=list)
    final_tape: list[int] = field(default_factory=list)
    halted: bool = True  # False: the pass budget ran out with positions still routed in


class LoopedTransformer:
    def __init__(self, layout, embed, core: list[Layer], coda: list[Layer], adapter, unembed,
                 router: int, grow: int) -> None:
        self.layout, self.embed, self.core, self.coda = layout, embed, core, coda
        self.adapter, self.unembed, self.router, self.grow = adapter, unembed, router, grow
        # the coda is an ordinary causal decoder over the same layout and unembedding
        self.decoder = Transformer(layout, embed, coda, unembed)
        # to name what a state holds (display only): only the dims a byte's embedding can set
        self._feat = np.flatnonzero(np.abs(embed).sum(axis=0))
        self._byte_rows = embed[:, self._feat]

    def nonzero_params(self) -> int:
        mats = [self.embed, self.unembed, self.adapter]
        for layer in (*self.core, *self.coda):
            mats += [layer.w_in, layer.w_out]
            for h in layer.heads:
                mats += [h.w_q, h.w_k, h.w_v, h.w_o]
        return int(sum(np.count_nonzero(m) for m in mats))

    def positional(self, n: int) -> np.ndarray:
        return self.decoder.positional(n)

    def block(self, state: np.ndarray, trace: Tracer | None = None) -> np.ndarray:
        """One pass of the recurrent block over every position: the last residual (n, d)."""
        n = len(state)
        x = state + self.positional(n)
        if trace is not None:
            trace.residual("state", x)
        causal = np.triu(np.ones((n, n), dtype=bool), k=1)
        for li, layer in enumerate(self.core):
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

    def tape_of(self, state: np.ndarray) -> list[int]:
        """The byte each position holds: its state's nearest embedding row (display only)."""
        d2 = ((state[:, None, self._feat] - self._byte_rows[None, :, :]) ** 2).sum(axis=2)
        return [int(b) for b in d2.argmin(axis=1)]

    def loop(self, tokens: list[int], max_passes: int, on_pass=None) -> tuple[np.ndarray, list[Pass], bool]:
        """Run the block until no position routes in (or ``max_passes``). Returns the final state,
        the passes, and whether the loop halted by itself. ``on_pass(index, state, x_last)`` sees
        every pass (the visualizer's tracing)."""
        state = self.embed[np.asarray(tokens, dtype=np.int64)].copy()
        passes: list[Pass] = []
        for r in range(max_passes):
            x = self.block(state)
            if on_pass is not None:
                on_pass(r, state, x)
            routed = x[:, self.router] > 0.5
            grew = bool(x[-1, self.grow] > 0.5)
            passes.append(Pass(r, self.tape_of(state), int(routed.sum()), grew))
            if not routed.any():
                log.info("loop halted after %d passes (%d positions)", r + 1, len(state))
                return state, passes, True
            nxt = x @ self.adapter
            state = np.where(routed[:, None], nxt, state)  # a position routed out keeps its state
            if grew:  # room: a blank position just before the root (the last position)
                state = np.concatenate([state[:-1], self.embed[[BLANK]], state[-1:]])
                log.debug("pass %d: inserted a blank before the root (%d positions)", r, len(state))
        log.warning("loop: no halt within %d passes", max_passes)
        return state, passes, False

    def run(self, tokens: list[int], stop: int, max_passes: int = 100_000, max_new: int = 100_000,
            on_pass=None) -> LoopResult:
        """Loop over the prompt, then let the coda write the answer greedily until ``stop``."""
        state, passes, halted = self.loop(tokens, max_passes, on_pass)
        cache = _DecodeCache(self.decoder, len(state) + 64)
        logits = None
        for row in state:  # the final state is the coda's prefix (no byte is re-read)
            logits = cache.step_vec(row)
        answer: list[int] = []
        for _ in range(max_new):
            nxt = int(np.argmax(logits))
            answer.append(nxt)
            if nxt == stop:
                break
            logits = cache.step(nxt)  # answer bytes take depth 0: embedding, then the coda
        else:
            log.warning("coda: no stop byte within %d bytes", max_new)
        log.info("looped run: %d prompt bytes, %d passes, %d answer bytes", len(tokens), len(passes), len(answer))
        return LoopResult(list(tokens), answer, passes, self.tape_of(state), halted)

    def coda_residual(self, state: np.ndarray, answer: list[int], trace: Tracer | None = None) -> np.ndarray:
        """The coda's forward pass over the final state followed by the answer bytes (the same
        computation as ``run``'s cached decoding, all positions at once; for tracing)."""
        x0 = np.concatenate([state, self.embed[np.asarray(answer, dtype=np.int64)]]) if answer else state
        n = len(x0)
        x = x0 + self.positional(n)
        if trace is not None:
            trace.residual("embed", x)
        causal = np.triu(np.ones((n, n), dtype=bool), k=1)
        for li, layer in enumerate(self.coda):
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
