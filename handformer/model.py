"""Generic decoder-only transformer whose weights are written by hand.

Nothing in this module knows about arithmetic. A model is described by:

* a residual-stream layout: every dimension has a *name* (``Layout``),
* a byte embedding table (256 rows) and a positional feature function,
* a list of layers, each holding attention heads and an MLP,
* an unembedding into 256 byte logits.

Weights are specified sparsely by dimension name through ``ModelBuilder`` and
materialised into dense float64 matrices by ``ModelBuilder.build``.

Design choices (deliberate, see README):

* No LayerNorm: it rescales the stream and would break exact linear readouts.
* No biases: every token embedding sets a constant ``one`` dimension, so any
  affine map is a linear map that reads ``one``.
* Positional features are a *function* of the integer position (``pos`` and
  ``pos^2``) rather than a learned table, so the model accepts any length.
  float64 keeps ``pos^2`` scores exact enough for sequences into the 1e5 range.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

VOCAB = 256

_PKG = Path(__file__).resolve().parent
# Modules that build specs on a circuit's behalf. The line worth pointing a reader
# at is the circuit line that called them, so they are skipped.
_HELPERS = frozenset({"model.py", "primitives.py", "common.py", "arith.py"})


def authored_at() -> str:
    """``path:line`` (repo-relative) of the circuit code that wrote the current spec.

    Documentation only, like ``UnitSpec.label``: it never reaches the weights.
    """
    f = sys._getframe(1)
    while f is not None:
        p = Path(f.f_code.co_filename)
        # circuit files live in the package or a model subpackage (handformer/life/); the shared
        # helpers and private helpers inside a circuit file (``_lt``, ``_ruler``) are skipped
        in_pkg = p.parent == _PKG or _PKG in p.parents
        helper = p.parent == _PKG and p.name in _HELPERS
        if in_pkg and not helper and "viz" not in p.parts and not f.f_code.co_name.startswith("_"):
            return f"{p.relative_to(_PKG.parent).as_posix()}:{f.f_lineno}"
        f = f.f_back
    return ""


class Layout:
    """Named allocation of residual-stream dimensions."""

    def __init__(self) -> None:
        self._index: dict[str, int] = {}

    def add(self, *names: str) -> None:
        for name in names:
            if name in self._index:
                raise ValueError(f"residual dim {name!r} allocated twice")
            self._index[name] = len(self._index)

    def __getitem__(self, name: str) -> int:
        return self._index[name]

    def __contains__(self, name: str) -> bool:
        return name in self._index

    def __len__(self) -> int:
        return len(self._index)

    @property
    def names(self) -> list[str]:
        return list(self._index)


# A linear form over residual dims: {dim_name: coefficient}.
Form = dict[str, float]


@dataclass
class HeadSpec:
    """One attention head.

    ``query[c]`` and ``key[c]`` are linear forms for head column ``c``; the score
    is ``sum_c query[c](x_q) * key[c](x_k)``. ``value`` maps source dims to
    destination dims: {src_dim: {dst_dim: coefficient}} (the fused OV circuit).

    So W_Q, W_K are (d_model, n_columns) with column c = the form's vector, and
    the head adds, at query q,

        out_q[dst] += sum_k attn_qk * sum_src x_k[src] * value[src][dst]
        attn_q = softmax over k <= q of score(q, k)   (no 1/sqrt(d) scaling)

    Scores are unscaled on purpose: the primitives set the logit margins
    themselves (beta = 30..60). Example, ``primitives.gather_head`` with target
    t: query ``[2*beta*t, {one: -beta}]``, key ``[{pos: 1}, {pos2: 1}]``,
    score = -beta*(k - t)^2 + const, value ``{"val": {dst: 1}}``.
    Head widths are tiny (1-3 columns); ``build`` factors value into W_V
    (one column per source dim) and W_O (where it is written).
    """

    name: str
    query: list[Form]
    key: list[Form]
    value: dict[str, Form]
    where: str = field(default="", compare=False)  # source line that wrote it (docs only)
    # When this head's read is exact (tests/test_claims.py): a 0/1 form over dims. None means
    # the default scope: wherever its calculator is on, or (shared heads) every output position.
    exact_when: Form | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if not self.where:
            self.where = authored_at()


@dataclass
class UnitSpec:
    """One MLP hidden unit: ``out += ReLU(read(x)) * write``.

    ``read`` becomes a column of the layer's W_in and ``write`` a row of W_out,
    so for residual x the unit adds ``max(sum_d read[d]*x[d], 0) * write``.
    There is no bias: a threshold is a coefficient on the always-1 dim
    ``one``. Example: ``read={"pos": 1, "one": -5}``, ``write={"y": 1}`` gives
    y += max(pos - 5, 0). Steps, gates and lookups (primitives.py) are sums of
    such units.

    ``label`` says what the unit computes (set by the primitive that built it)
    and ``where`` is the source line that wrote it; both are documentation only
    and never reach the weights.
    """

    read: Form
    write: Form
    label: str = ""
    where: str = field(default="", compare=False)
    # Checked claims about the pre-activation (tests/test_claims.py), documentation only:
    #   ("outside", lo, hi)        never strictly inside (lo, hi): a robust step's input
    #   ("silent_unless", flag)    <= 0 wherever the 0/1 dim ``flag`` is 0: a gated output
    claims: tuple = field(default=(), compare=False)

    def __post_init__(self) -> None:
        if not self.where:
            self.where = authored_at()


@dataclass
class LayerSpec:
    name: str
    heads: list[HeadSpec] = field(default_factory=list)
    units: list[UnitSpec] = field(default_factory=list)


@dataclass
class Head:
    name: str
    w_q: np.ndarray  # (d_model, d_head)
    w_k: np.ndarray  # (d_model, d_head)
    w_v: np.ndarray  # (d_model, d_value): one column per source dim the head moves
    w_o: np.ndarray  # (d_value, d_model): where each moved value is written

    @property
    def w_ov(self) -> np.ndarray:
        """The fused (d_model, d_model) value-output map, for inspection only."""
        return self.w_v @ self.w_o


@dataclass
class Layer:
    name: str
    heads: list[Head]
    w_in: np.ndarray  # (d_model, n_units)
    w_out: np.ndarray  # (n_units, d_model)


class ModelBuilder:
    """Collects sparse, named weight specifications for a ``Transformer``.

    Every weight is given as a linear form ``{dim_name: coef}``, never as a
    matrix index. ``build`` turns each form into a dense float64 vector over
    the ``Layout`` (``_vec``: v[layout[dim]] += coef) and stacks them:

    * ``embed_byte(byte, form)``: row ``byte`` of the embedding (accumulates);
    * ``at(i)`` / ``layer(name)``: a ``LayerSpec`` whose ``HeadSpec``s become
      W_Q/W_K/W_V/W_O and whose ``UnitSpec``s become W_in/W_out columns/rows;
    * ``logit(dim, byte, coef)``: unembedding entry, logit[byte] += coef * x[dim].

    So a weight is nonzero only where some named formula put it, and every
    entry can be traced back to its spec (``HeadSpec.where``/``UnitSpec.where``).
    A dim name missing from the layout raises KeyError at build time.
    """

    def __init__(self, layout: Layout) -> None:
        self.layout = layout
        self.token_embed: dict[int, Form] = {}
        self.layers: list[LayerSpec] = []
        self.unembed: dict[str, dict[int, float]] = {}
        self.embed_where: dict[str, str] = {}  # dim -> source line that first embeds it (docs only)

    def embed_byte(self, byte: int, form: Form) -> None:
        row = self.token_embed.setdefault(byte, {})
        for dim, coef in form.items():
            row[dim] = row.get(dim, 0.0) + coef
            self.embed_where.setdefault(dim, authored_at())

    def layer(self, name: str) -> LayerSpec:
        spec = LayerSpec(name)
        self.layers.append(spec)
        return spec

    def at(self, index: int) -> LayerSpec:
        """Layer ``index`` (1-based), creating empty layers up to it.

        Lets independent circuits place their sublayers at fixed global depths.
        """
        while len(self.layers) < index:
            self.layers.append(LayerSpec(f"L{len(self.layers) + 1}"))
        return self.layers[index - 1]

    def logit(self, dim: str, byte: int, coef: float) -> None:
        self.unembed.setdefault(dim, {})[byte] = coef

    def _vec(self, form: Form) -> np.ndarray:
        v = np.zeros(len(self.layout))
        for dim, coef in form.items():
            v[self.layout[dim]] += coef
        return v

    def build(self) -> Transformer:
        d = len(self.layout)
        embed = np.zeros((VOCAB, d))
        for byte, form in self.token_embed.items():
            embed[byte] = self._vec(form)

        layers = []
        for spec in self.layers:
            heads = []
            for h in spec.heads:
                if len(h.query) != len(h.key):
                    raise ValueError(f"head {h.name}: query/key width mismatch")
                w_q = np.stack([self._vec(f) for f in h.query], axis=1)
                w_k = np.stack([self._vec(f) for f in h.key], axis=1)
                # Factored value/output (standard per-head d_value): W_V selects each
                # source dim, W_O writes it. W_V @ W_O is the fused map; cost d*r, not d^2.
                srcs = list(h.value)
                w_v = np.zeros((d, len(srcs)))
                w_o = np.zeros((len(srcs), d))
                for c, src in enumerate(srcs):
                    w_v[self.layout[src], c] = 1.0
                    w_o[c] = self._vec(h.value[src])
                heads.append(Head(h.name, w_q, w_k, w_v, w_o))
            w_in = np.zeros((d, len(spec.units)))
            w_out = np.zeros((len(spec.units), d))
            for i, u in enumerate(spec.units):
                w_in[:, i] = self._vec(u.read)
                w_out[i] = self._vec(u.write)
            layers.append(Layer(spec.name, heads, w_in, w_out))

        unembed = np.zeros((d, VOCAB))
        for dim, row in self.unembed.items():
            for byte, coef in row.items():
                unembed[self.layout[dim], byte] = coef

        model = Transformer(self.layout, embed, layers, unembed)
        log.info(
            "built model: d_model=%d layers=%d heads=%d mlp_units=%d nonzero_params=%d",
            d,
            len(layers),
            sum(len(l.heads) for l in layers),
            sum(l.w_in.shape[1] for l in layers),
            model.nonzero_params(),
        )
        return model


class Tracer:
    """Observer interface for ``Transformer.residual`` (all methods optional no-ops)."""

    def residual(self, label: str, x: np.ndarray) -> None: ...

    def attention(self, layer: int, head: int, attn: np.ndarray) -> None: ...

    def mlp(self, layer: int, pre: np.ndarray) -> None: ...


class Transformer:
    def __init__(
        self,
        layout: Layout,
        embed: np.ndarray,
        layers: list[Layer],
        unembed: np.ndarray,
    ) -> None:
        self.layout = layout
        self.embed = embed
        self.layers = layers
        self.unembed = unembed

    def nonzero_params(self) -> int:
        mats = [self.embed, self.unembed]
        for layer in self.layers:
            mats += [layer.w_in, layer.w_out]
            for h in layer.heads:
                mats += [h.w_q, h.w_k, h.w_v, h.w_o]
        return int(sum(np.count_nonzero(m) for m in mats))

    def positional(self, n: int) -> np.ndarray:
        x = np.zeros((n, len(self.layout)))
        p = np.arange(n, dtype=np.float64)
        x[:, self.layout["pos"]] = p
        x[:, self.layout["pos2"]] = p * p
        return x

    @staticmethod
    def head_attention(x: np.ndarray, h: Head, causal: np.ndarray) -> np.ndarray:
        """Causal softmax attention pattern (n, n) of head ``h`` on layer input ``x``."""
        scores = (x @ h.w_q) @ (x @ h.w_k).T
        scores[causal] = -np.inf
        scores -= scores.max(axis=1, keepdims=True)
        attn = np.exp(scores)
        attn /= attn.sum(axis=1, keepdims=True)
        return attn

    def residual(self, tokens: list[int], trace: Tracer | None = None) -> np.ndarray:
        """Run the stack and return the final residual stream (n, d_model).

        Standard multi-head attention: every head in a layer reads the same
        layer input and their outputs are summed into the stream; then the MLP.

            x_0 = embed[token] + [pos, pos^2 features]   (positional)
            x  += sum_h softmax(causal(x W_Q (x W_K)^T)) (x W_V) W_O
            x  += ReLU(x W_in) W_out

        No LayerNorm, no biases, no score scaling, float64 throughout, so each
        named dim keeps the exact value a primitive put there. Because heads
        read the pre-layer x, a dim written by a head of layer i is visible to
        the MLP of layer i but only to heads of layer i+1; a dim written by an
        MLP is visible from layer i+1. Any length works: positions enter only
        as the features ``pos`` and ``pos2``.

        ``trace`` (optional) observes the same computation, it never alters it:
        ``trace.residual(label, x)`` after the embedding and after each
        sublayer, ``trace.attention(layer, head, attn)`` per head, and
        ``trace.mlp(layer, pre)`` with the MLP pre-activations.
        """
        n = len(tokens)
        x = self.embed[np.asarray(tokens, dtype=np.int64)] + self.positional(n)
        if trace is not None:
            trace.residual("embed", x)
        causal = np.triu(np.ones((n, n), dtype=bool), k=1)
        for li, layer in enumerate(self.layers):
            delta = np.zeros_like(x)
            for hi, h in enumerate(layer.heads):
                attn = self.head_attention(x, h, causal)
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
            log.debug("layer %s done: |x|max=%.3g", layer.name, np.abs(x).max())
        return x

    def logits(self, tokens: list[int]) -> np.ndarray:
        return self.residual(tokens) @ self.unembed

    def generate(self, tokens: list[int], stop: int, max_new: int) -> list[int]:
        """Greedy decoding until ``stop`` is emitted or ``max_new`` tokens.

        Incremental (cached) decoding: the same per-position computation as
        ``residual``, but each new position reuses the cached layer inputs
        and keys of earlier positions instead of recomputing them. Causality
        makes this exact; ``tests/test_expr.py`` checks greedy output against
        the reference sequences.
        """
        cache = _DecodeCache(self, min(len(tokens) + max_new, 1024))
        logits = None
        for t in tokens:
            logits = cache.step(t)
        out = list(tokens)
        for step in range(max_new):
            nxt = int(np.argmax(logits))
            out.append(nxt)
            if nxt == stop:
                log.debug("generate: stop byte after %d new tokens", step + 1)
                return out
            logits = cache.step(nxt)
        log.warning("generate: hit max_new=%d without stop byte", max_new)
        return out


class _DecodeCache:
    """Per-layer residual inputs and per-head keys for incremental decoding.

    ``step`` runs one new position q through the same layers as ``residual``:
    it stores the layer input ``x_in[l][q]`` and key ``keys[l][h][q] = x W_K``,
    then attends with q's query over the cached keys 0..q and forms values as
    ``(attn @ x_in[l][:q+1]) @ W_V @ W_O`` (the same product as ``residual``,
    associated differently). Exact, not approximate: under the causal mask an
    earlier position's layer inputs never depend on later tokens, so caching
    them changes nothing. Cost per token is O(q * d) instead of a full rerun.
    Storage starts at ``capacity`` positions and doubles when full (``_grow``).
    """

    def __init__(self, model: Transformer, capacity: int) -> None:
        self.m = model
        d = len(model.layout)
        self.n = 0
        self.x_in = [np.zeros((capacity, d)) for _ in model.layers]
        self.keys = [[np.zeros((capacity, h.w_k.shape[1])) for h in layer.heads] for layer in model.layers]

    def _grow(self) -> None:
        """Double the capacity (amortised O(1) per token)."""
        for li in range(len(self.x_in)):
            self.x_in[li] = np.concatenate([self.x_in[li], np.zeros_like(self.x_in[li])])
            self.keys[li] = [np.concatenate([k, np.zeros_like(k)]) for k in self.keys[li]]
        log.debug("decode cache grown to %d positions", len(self.x_in[0]))

    def step(self, token: int) -> np.ndarray:
        """Append one token; return its next-byte logits."""
        if self.n == len(self.x_in[0]):
            self._grow()
        m, q = self.m, self.n
        x = m.embed[token] + m.positional(q + 1)[q]
        for li, layer in enumerate(m.layers):
            self.x_in[li][q] = x
            xs = self.x_in[li][: q + 1]
            delta = np.zeros_like(x)
            for hi, h in enumerate(layer.heads):
                k = self.keys[li][hi]
                k[q] = x @ h.w_k
                scores = k[: q + 1] @ (x @ h.w_q)
                scores -= scores.max()
                attn = np.exp(scores)
                attn /= attn.sum()
                delta += ((attn @ xs) @ h.w_v) @ h.w_o
            x = x + delta
            if layer.w_in.shape[1]:
                x = x + np.maximum(x @ layer.w_in, 0.0) @ layer.w_out
        self.n += 1
        return x @ m.unembed
