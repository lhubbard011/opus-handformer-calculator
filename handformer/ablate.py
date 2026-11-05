"""Knockouts: a copy of the model with chosen parts switched off, in the weights.

Interpretability claims are strongest when breaking the part breaks the
behaviour it is said to cause. Knocking a part out here is an edit of the
weights, so the forward pass, the decoder and the visualizer stay unchanged:

* head (layer, index): its output map W_O is zeroed; it still attends, writes nothing.
* unit (layer, index): its row of W_out is zeroed; it still fires, writes nothing.
* dim name: every write into it is zeroed (embedding column, every head's W_O
  column, every unit's W_out column), so it stays 0 at every position. The
  position features pos and pos² are added by the forward pass, not by a
  weight, so they cannot be knocked out.

The in-browser engine (``viz/static/local-api.js``) applies the same three
rules; ``tests/test_web.py`` checks the two agree.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .model import Head, Layer, Transformer

log = logging.getLogger(__name__)
POSITIONAL = ("pos", "pos2")


@dataclass(frozen=True)
class Knockout:
    heads: frozenset[tuple[int, int]] = field(default_factory=frozenset)
    units: frozenset[tuple[int, int]] = field(default_factory=frozenset)
    dims: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def from_request(cls, req: dict | None) -> "Knockout":
        req = req or {}
        return cls(frozenset((int(l), int(h)) for l, h in req.get("heads", [])),
                   frozenset((int(l), int(u)) for l, u in req.get("units", [])),
                   frozenset(str(d) for d in req.get("dims", [])))

    def __bool__(self) -> bool:
        return bool(self.heads or self.units or self.dims)

    def as_request(self) -> dict:
        return {"heads": sorted(map(list, self.heads)), "units": sorted(map(list, self.units)),
                "dims": sorted(self.dims)}

    def key(self) -> str:
        r = self.as_request()
        return f"h{r['heads']}u{r['units']}d{r['dims']}"


def apply(model: Transformer, ko: Knockout) -> Transformer:
    """A new Transformer with the knocked-out parts writing nothing (``model`` is untouched).

    Only write maps change, so a knockout removes a part's contribution to the
    residual stream and nothing else:

    * head (l, h): W_O = 0, so its term attn @ (x W_V) W_O is 0; W_Q/W_K are
      kept, so its attention pattern is still computed and can be shown;
    * unit (l, u): row u of W_out = 0, so ReLU(x W_in)[u] still fires but adds 0;
    * dim d: column d of the embedding, of every head's W_O and of every W_out
      is 0, so x[d] = 0 at every position and every layer; readers of d
      (W_Q, W_K, W_V, W_in, unembedding) are unchanged and simply read 0.

    Example: knocking out dim ``act_add`` silences every gated output unit of
    the add circuit (``arith._Namespace.unit``). Unknown heads, units or dims
    raise ValueError; ``pos``/``pos2`` cannot be knocked out (see module doc).
    An empty knockout returns ``model`` itself, not a copy. Unchanged matrices
    are shared with ``model``, not copied.
    """
    if not ko:
        return model
    embed, layers = apply_parts(model.layout, model.embed, model.layers, ko)
    out = Transformer(model.layout, embed, layers, model.unembed)
    log.info("knockout %s: %d -> %d nonzero weights", ko.key(), model.nonzero_params(), out.nonzero_params())
    return out


def apply_parts(layout, embed: np.ndarray, model_layers: list[Layer], ko: Knockout) -> tuple[np.ndarray, list[Layer]]:
    """The embedding and layers of ``apply``, for any stack of layers (the looped model's block
    and coda too, handformer/rewrite/viz.py). Same rules, same checks."""
    n_layers = len(model_layers)
    model_layers_ = model_layers
    for li, hi in ko.heads:
        if not (0 <= li < n_layers and 0 <= hi < len(model_layers_[li].heads)):
            raise ValueError(f"no head {hi} in layer {li}")
    for li, ui in ko.units:
        if not (0 <= li < n_layers and 0 <= ui < model_layers_[li].w_in.shape[1]):
            raise ValueError(f"no unit {ui} in layer {li}")
    for d in ko.dims:
        if d not in layout:
            raise ValueError(f"no residual dim named {d!r}")
        if d in POSITIONAL:
            raise ValueError(f"{d} is a position feature added by the forward pass, not a weight; "
                             "it cannot be knocked out")
    cols = [layout[d] for d in ko.dims]

    embed = embed.copy()
    embed[:, cols] = 0.0
    layers = []
    for li, layer in enumerate(model_layers_):
        heads = []
        for hi, h in enumerate(layer.heads):
            w_o = h.w_o.copy()
            if (li, hi) in ko.heads:
                w_o[:] = 0.0
            w_o[:, cols] = 0.0
            heads.append(Head(h.name, h.w_q, h.w_k, h.w_v, w_o, h.full))
        w_out = layer.w_out.copy()
        for lj, ui in ko.units:
            if lj == li:
                w_out[ui, :] = 0.0
        w_out[:, cols] = 0.0
        layers.append(Layer(layer.name, heads, layer.w_in, w_out))
    return embed, layers
