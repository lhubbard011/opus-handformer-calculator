"""A recurrent-depth (latent-recurrent) decoder with hand-written weights (LOOPED.md §2).

Autoregressive like any decoder: one position at a time, causal attention, a KV cache. What is
looped is depth, per token, as in recurrent-depth language models:

* **Prelude**: the input embedding ``e`` of the position (a byte's row, or a latent input below).
* **Recurrent block** (Huginn, Geiping et al. 2025): one fixed stack of layers applied to the
  same position again and again. Iteration k reads ``e + s_k + positional features``: the
  input is injected at every iteration, as in Huginn, and the state starts at ``s_0 = 0``.
* **Adapter**: a fixed linear map from the block's last residual to the next iteration's state
  (``s_{k+1} = x_k @ A``). It keeps the token's loop registers and drops the rest.
* **Router** (Mixture-of-Recursions, Bae et al. 2025; Ouro's exit gate, Zhu et al. 2025): a
  per-token dim the block computes. While it is 1 the token goes round again, so every token
  gets its own depth.
* **KV cache**: a later token attends to an earlier one through the keys and values of the
  earlier token's last iteration (the cache sharing recurrent-depth models use when tokens
  exit at different depths). Within its own iterations a token also attends to itself.
* **Coda**: the unembedding of the last residual: the next byte.
* **Latent positions** (continuous thought, Hao et al. 2024 "Coconut"): when the ``think`` dim
  of the last residual is 1, the next position's input is not a byte but ``x @ T``, a fixed
  linear map of this position's last residual, and nothing is written out. Positions are the
  model's only unbounded memory, so this is where intermediate results live.

The host does what any decoding loop does: feed the prompt, take the argmax, stop at the end
byte. It also follows two bits the weights compute: ``route`` (iterate this position again)
and ``think`` (the next position is latent). It never reads or changes what they hold.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..model import Form, Layer, ModelBuilder, Transformer

log = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[2]
HOLD_LOGIT = 20.0  # a held byte's logit (holds are exact 0/1)


def _caller() -> str:
    """``path:line`` of the circuit line that called a DepthBuilder method (docs only)."""
    f = sys._getframe(2)
    return f"{Path(f.f_code.co_filename).resolve().relative_to(_REPO).as_posix()}:{f.f_lineno}"


class DepthBuilder(ModelBuilder):
    """``ModelBuilder`` for a recurrent-depth decoder. ``at(i)`` is layer i of the block.

    * ``carry(dst, form)``: the adapter. Next iteration's ``dst`` is ``form`` of this one's last
      residual (the token's loop registers).
    * ``think(dst, form)``: the latent map T. A latent next position's input ``dst`` is ``form``
      of this position's last residual.
    * ``hold(dim, byte)``: ``dim`` (an exact 0/1) is "the next byte is ``byte``". Its value
      times the byte's embedding row (without the bias channel ``one``) goes into T, and
      ``HOLD_LOGIT`` times it into that byte's logit (the coda).
    """

    def __init__(self, layout, router: str, think: str) -> None:
        super().__init__(layout)
        self.adapter: dict[str, Form] = {}
        self.thought: dict[str, Form] = {}
        self.holds: dict[str, int] = {}
        self.carry_where: dict[str, str] = {}  # dim -> the line that carries it (docs only)
        self.router, self.think_dim = router, think

    def carry(self, dst: str, form: Form) -> None:
        if dst in self.adapter:
            raise ValueError(f"the adapter already writes {dst!r}")
        self.adapter[dst] = dict(form)
        self.carry_where.setdefault(dst, _caller())

    def think(self, dst: str, form: Form) -> None:
        if dst in self.thought:
            raise ValueError(f"the latent map already writes {dst!r}")
        self.thought[dst] = dict(form)
        self.carry_where.setdefault(dst, _caller())

    def hold(self, dim: str, byte: int) -> None:
        if dim in self.holds:
            raise ValueError(f"{dim!r} already holds byte {self.holds[dim]}")
        self.holds[dim] = byte

    def build(self) -> RecurrentDepthLM:
        embed, unembed = self._embed_unembed()
        block = self._layers(self.layers)
        bad = [f"{l.name}/{h.name}" for l in block for h in l.heads if h.full]
        if bad:
            raise ValueError(f"a decoder's block is causal; bidirectional heads: {', '.join(bad)}")
        d, lay = len(self.layout), self.layout
        adapter, thought = np.zeros((d, d)), np.zeros((d, d))
        for mat, forms in ((adapter, self.adapter), (thought, self.thought)):
            for dst, form in forms.items():
                mat[:, lay[dst]] += self._vec(form)
        one = lay["one"]
        for dim, byte in self.holds.items():
            row = embed[byte].copy()
            row[one] = 0.0  # the latent input's bias channel comes from T's own "one" entry
            thought[lay[dim]] += row
            unembed[lay[dim], byte] += HOLD_LOGIT
        for name in ("pos", "pos2"):  # positions are added from the index, never carried
            if np.any(adapter[:, lay[name]]) or np.any(thought[:, lay[name]]):
                raise ValueError(f"the adapter and T must not write the positional feature {name!r}")
        model = RecurrentDepthLM(lay, embed, block, unembed, adapter, thought, lay[self.router], lay[self.think_dim])
        log.info("built recurrent-depth model: d_model=%d block layers=%d heads=%d mlp_units=%d nonzero_params=%d",
                 d, len(block), sum(len(l.heads) for l in block), sum(l.w_in.shape[1] for l in block),
                 model.nonzero_params())
        return model


@dataclass
class Token:
    """One position of a run (for tests and the visualizer; never fed back)."""

    index: int
    latent: bool  # its input came from T (a thought), not a byte
    byte: int | None  # the byte it is (prompt and written bytes); None for a latent position
    iterations: int  # how many times the block ran on it (its depth)


@dataclass
class DepthResult:
    prompt: list[int]
    answer: list[int]  # the written bytes, ending with the stop byte (if reached)
    tokens: list[Token] = field(default_factory=list)
    inputs: np.ndarray | None = None  # (n, d): every position's input e (prompt rows, thoughts, bytes)
    finals: np.ndarray | None = None  # (n, d): every position's last residual
    finished: bool = True  # False: a budget ran out


class _Stack:
    """A block layer's heads stacked side by side (one product per layer instead of one per head)."""

    def __init__(self, layer: Layer) -> None:
        hs = layer.heads
        self.n = len(hs)
        if not hs:
            return
        self.wq = np.concatenate([h.w_q for h in hs], axis=1)
        self.wk = np.concatenate([h.w_k for h in hs], axis=1)
        self.wv = np.concatenate([h.w_v for h in hs], axis=1)
        self.wo = np.concatenate([h.w_o for h in hs], axis=0)
        dk = [h.w_q.shape[1] for h in hs]
        self.k_starts = np.concatenate([[0], np.cumsum(dk)[:-1]]).astype(np.int64)
        self.v_head = np.repeat(np.arange(len(hs)), [h.w_v.shape[1] for h in hs])


class RecurrentDepthLM:
    def __init__(self, layout, embed, block: list[Layer], unembed, adapter, thought, router: int, think: int) -> None:
        self.layout, self.embed, self.block, self.unembed = layout, embed, block, unembed
        self.adapter, self.thought, self.router, self.think = adapter, thought, router, think
        self.stacks = [_Stack(l) for l in block]
        # the same weights as a plain (one-iteration) decoder, for tracing a position's iteration
        self.decoder = Transformer(layout, embed, block, unembed)

    def nonzero_params(self) -> int:
        mats = [self.embed, self.unembed, self.adapter, self.thought]
        for layer in self.block:
            mats += [layer.w_in, layer.w_out]
            for h in layer.heads:
                mats += [h.w_q, h.w_k, h.w_v, h.w_o]
        return int(sum(np.count_nonzero(m) for m in mats))

    def positional(self, n: int) -> np.ndarray:
        return self.decoder.positional(n)

    def run(self, prompt: list[int], stop: int, max_tokens: int = 1_000_000, max_iters: int = 10_000,
            on_iter=None) -> DepthResult:
        """Decode greedily: the prompt, then thoughts and bytes until ``stop``. ``max_tokens``
        caps the positions after the prompt, ``max_iters`` one position's depth.
        ``on_iter(position, iteration, x_in, x_last)`` sees every iteration (tracing)."""
        cache = _DepthCache(self, max(64, 2 * len(prompt)))
        tokens: list[Token] = []
        answer: list[int] = []
        finished = True
        x = None
        for byte in prompt:
            x, it = cache.step(self.embed[byte], max_iters, on_iter)
            tokens.append(Token(len(tokens), False, byte, it))
        for _ in range(max_tokens):
            if x[self.think] > 0.5:
                e, byte = x @ self.thought, None
            else:
                byte = int(np.argmax(x @ self.unembed))
                answer.append(byte)
                if byte == stop:
                    break
                e = self.embed[byte]
            x, it = cache.step(e, max_iters, on_iter)
            tokens.append(Token(len(tokens), byte is None, byte, it))
        else:
            finished = False
            log.warning("no stop byte within %d positions after the prompt", max_tokens)
        if any(t.iterations >= max_iters for t in tokens):
            finished = False
            log.warning("a position reached the iteration budget (%d)", max_iters)
        n_lat = sum(t.latent for t in tokens)
        log.info("recurrent-depth run: %d prompt bytes, %d latent positions, %d iterations, %d bytes written",
                 len(prompt), n_lat, sum(t.iterations for t in tokens), len(answer))
        return DepthResult(list(prompt), answer, tokens, cache.inputs[: cache.n].copy(),
                           cache.finals[: cache.n].copy(), finished)


class _DepthCache:
    """Keys and values of every committed position's last iteration, per layer (all heads of a
    layer side by side), plus every position's input and last residual."""

    def __init__(self, model: RecurrentDepthLM, capacity: int) -> None:
        self.m, self.n, self.cap = model, 0, capacity
        d = len(model.layout)
        self.keys = [np.zeros((capacity, s.wk.shape[1])) if s.n else None for s in model.stacks]
        self.vals = [np.zeros((capacity, s.wv.shape[1])) if s.n else None for s in model.stacks]
        self.inputs = np.zeros((capacity, d))
        self.finals = np.zeros((capacity, d))

    def _grow(self) -> None:
        g = lambda a: None if a is None else np.concatenate([a, np.zeros_like(a)])  # noqa: E731
        self.keys = [g(k) for k in self.keys]
        self.vals = [g(v) for v in self.vals]
        self.inputs, self.finals = g(self.inputs), g(self.finals)
        self.cap *= 2
        log.debug("depth cache grown to %d positions", self.cap)

    def forward(self, x: np.ndarray, q: int) -> np.ndarray:
        """One iteration of the block at position q (input x, positions included): its last
        residual. Its keys and values are left in slot q (the last iteration's stay)."""
        for li, (layer, st) in enumerate(zip(self.m.block, self.m.stacks)):
            if st.n:
                # this iteration's own key and value sit in slot q until the position commits
                self.keys[li][q] = x @ st.wk
                self.vals[li][q] = x @ st.wv
                keys, vals = self.keys[li][: q + 1], self.vals[li][: q + 1]
                scores = np.add.reduceat(keys * (x @ st.wq), st.k_starts, axis=1)  # (q+1, heads)
                scores -= scores.max(axis=0, keepdims=True)
                attn = np.exp(scores)
                attn /= attn.sum(axis=0, keepdims=True)
                x = x + ((attn[:, st.v_head] * vals).sum(axis=0)) @ st.wo
            if layer.w_in.shape[1]:
                x = x + np.maximum(x @ layer.w_in, 0.0) @ layer.w_out
        return x

    def step(self, e: np.ndarray, max_iters: int, on_iter=None) -> tuple[np.ndarray, int]:
        """Run one new position to its exit and commit it: (its last residual, its depth)."""
        if self.n == self.cap:
            self._grow()
        m, q = self.m, self.n
        pos = m.positional(q + 1)[q]
        s = np.zeros_like(e)
        for k in range(max_iters):
            x_in = e + s + pos
            x = self.forward(x_in, q)
            if on_iter is not None:
                on_iter(q, k, x_in, x)
            if x[m.router] <= 0.5:
                break
            s = x @ m.adapter
        # slot q already holds the last iteration's keys and values (forward wrote them)
        self.inputs[q], self.finals[q] = e, x
        self.n += 1
        return x, k + 1

