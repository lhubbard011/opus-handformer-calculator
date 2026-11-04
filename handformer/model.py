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
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

VOCAB = 256


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
    """

    name: str
    query: list[Form]
    key: list[Form]
    value: dict[str, Form]


@dataclass
class UnitSpec:
    """One MLP hidden unit: ``out += ReLU(read(x)) * write``."""

    read: Form
    write: Form


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
    w_ov: np.ndarray  # (d_model, d_model)


@dataclass
class Layer:
    name: str
    heads: list[Head]
    w_in: np.ndarray  # (d_model, n_units)
    w_out: np.ndarray  # (n_units, d_model)


class ModelBuilder:
    """Collects sparse, named weight specifications for a ``Transformer``."""

    def __init__(self, layout: Layout) -> None:
        self.layout = layout
        self.token_embed: dict[int, Form] = {}
        self.layers: list[LayerSpec] = []
        self.unembed: dict[str, dict[int, float]] = {}

    def embed_byte(self, byte: int, form: Form) -> None:
        row = self.token_embed.setdefault(byte, {})
        for dim, coef in form.items():
            row[dim] = row.get(dim, 0.0) + coef

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
                w_ov = np.zeros((d, d))
                for src, dst_form in h.value.items():
                    w_ov[self.layout[src]] += self._vec(dst_form)
                heads.append(Head(h.name, w_q, w_k, w_ov))
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
                mats += [h.w_q, h.w_k, h.w_ov]
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
                delta += attn @ (x @ h.w_ov)
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
        cache = _DecodeCache(self, len(tokens) + max_new)
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
    """Per-layer residual inputs and per-head keys for incremental decoding."""

    def __init__(self, model: Transformer, capacity: int) -> None:
        self.m = model
        d = len(model.layout)
        self.n = 0
        self.x_in = [np.zeros((capacity, d)) for _ in model.layers]
        self.keys = [[np.zeros((capacity, h.w_k.shape[1])) for h in layer.heads] for layer in model.layers]

    def step(self, token: int) -> np.ndarray:
        """Append one token; return its next-byte logits."""
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
                delta += (attn @ xs) @ h.w_ov
            x = x + delta
            if layer.w_in.shape[1]:
                x = x + np.maximum(x @ layer.w_in, 0.0) @ layer.w_out
        self.n += 1
        return x @ m.unembed
