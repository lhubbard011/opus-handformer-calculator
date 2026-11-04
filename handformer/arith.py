"""The arithmetic model: one set of weights for whole expressions.

The host feeds ``\\n<expression>=`` and runs greedy decoding until ``\\n``.
It never parses, routes or post-processes anything. In the weights:

* ``frame`` (layers 1-3): markers, frame-relative flags, phases, and which
  operator the current frame holds (``act_add`` / ``act_sub`` / ``act_mul``).
* ``expr`` (layers 4-12): picks the redex by parentheses and precedence,
  copies it into a frame, and rewrites the line with the result.
* operator circuits (layers 3-12): add / sub / mul / div, each with private
  namespaced dims (``add.A``, ``mul.J``, ...), all computing on every input.
  Only their units that write output dims are gated by their ``act_*``
  flag (ReLU input shifted by ``-BIG*(1-act)``), so an inactive circuit adds
  exactly 0 to the logits.
"""

from __future__ import annotations

import logging

from . import addition, division, expr, frame, multiplication, subtraction
from . import common as cm
from . import primitives as pr
from .model import Form, HeadSpec, Layout, ModelBuilder, Transformer, UnitSpec

log = logging.getLogger(__name__)

CIRCUITS = {"add": addition.builder, "sub": subtraction.builder, "mul": multiplication.builder,
            "div": division.builder}

# Output dims written by operator circuits, and the byte each one emits.
OP_OUTPUTS = {
    **{f"D{d}": (ord("0") + d, cm.DIGIT_LOGIT) for d in range(10)},
    "work_end": (cm.WORK_END, cm.CONTROL_LOGIT),
    "work_end_neg": (cm.WORK_END_NEG, cm.CONTROL_LOGIT),
    "frame_end": (cm.FRAME_CLOSE, cm.CONTROL_LOGIT),
    "emit_minus": (cm.MINUS, cm.CONTROL_LOGIT),
    "emit_comma": (cm.COMMA, cm.CONTROL_LOGIT),
    "emit_bar": (cm.BAR, cm.CONTROL_LOGIT),
}


class _Namespace:
    """Prefix a circuit's private dims; gate its output-writing units."""

    def __init__(self, ns: str, private: set[str]) -> None:
        self.ns, self.private = ns, private

    def dim(self, d: str) -> str:
        return f"{self.ns}.{d}" if d in self.private else d

    def form(self, f: Form) -> Form:
        return {self.dim(d): c for d, c in f.items()}

    def head(self, h: HeadSpec) -> HeadSpec:
        return HeadSpec(
            name=f"{self.ns}:{h.name}",
            query=[self.form(f) for f in h.query],
            key=[self.form(f) for f in h.key],
            value={self.dim(s): self.form(d) for s, d in h.value.items()},
        )

    def unit(self, u: UnitSpec) -> UnitSpec:
        read, write = self.form(u.read), self.form(u.write)
        if any(d in OP_OUTPUTS for d in write):
            read = pr.gated(read, {f"act_{self.ns}": 1.0}, 1.0)
        return UnitSpec(read=read, write=write)


def builder() -> ModelBuilder:
    circuits = {ns: make() for ns, make in CIRCUITS.items()}

    lay = Layout()
    cm.add_input_dims(lay)
    lay.add(*frame.DIMS, *expr.DIMS)
    for ns, cb in circuits.items():
        lay.add(*(f"{ns}.{d}" for d in cb.layout.names))
    lay.add(*OP_OUTPUTS, *expr.OUT_DIMS)

    b = ModelBuilder(lay)
    cm.embed(b)
    frame.build(b)
    expr.build(b)
    for ns, cb in circuits.items():
        nsp = _Namespace(ns, set(cb.layout.names))
        for i, spec in enumerate(cb.layers, start=1):
            if not (spec.heads or spec.units):
                continue
            layer = b.at(i)
            layer.heads += [nsp.head(h) for h in spec.heads]
            layer.units += [nsp.unit(u) for u in spec.units]
        log.info("merged circuit %s: %d private dims, depth %d", ns, len(cb.layout), len(cb.layers))
    for dim, (byte, coef) in OP_OUTPUTS.items():
        b.logit(dim, byte, coef)
    return b


def build() -> Transformer:
    return builder().build()


def prompt(expression: str) -> list[int]:
    return list(f"\n{expression}=".encode("ascii"))


def solve(model: Transformer, expression: str) -> str:
    """Feed ``\\n<expression>=`` and decode; return everything the model wrote."""
    p = prompt(expression)
    budget = 4 * len(expression) ** 2 + 64
    out = model.generate(p, stop=cm.EOS, max_new=budget)
    text = bytes(out[len(p):]).decode("ascii", "replace")
    log.info("solve %s: %r", expression, text)
    return text
