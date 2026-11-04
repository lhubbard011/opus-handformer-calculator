"""Addition circuit, inside a frame ``[ a + b = working ; answer ]``.

* working: the sum LSB first, max(len a, len b) digits (+1 if the last
  carry is set), then ``;``
* answer: the working copied in reverse from its top nonzero digit, then
  ``]`` (the frame close; the rewriter takes over)

Operands are fenced by shared markers: a in (F+sa, P), b in (P+sb, E), where F
is the frame's '[', P its operator, E its '=', and sa/sb skip operand signs.
This circuit is routed + and - whose effective signs agree: it adds
magnitudes and the result takes a's sign (``-`` emitted only if nonzero). Digit i (i=0 units) of an
operand with fences (lo, hi) is at hi-1-i, clamped onto lo (value 0) once
exhausted.

carry_i = [y >= 10] + [y == 9 and previous output digit == '0'],
y = a_{i-1} + b_{i-1}: the carry is recovered, never stored.

Global layers: 3 mlp targets / key features; 4 gather + carry; 5 output.
"""

from __future__ import annotations

from . import common as cm
from . import primitives as pr
from .frame import FA, PB
from .model import Layout, ModelBuilder

PRIVATE = ("tA", "tA1", "tB", "tB1", "exA", "exB", "zkey", "A", "A1", "B", "B1", "Z",
           "carry", "Zo", "sgn", "C")


def builder() -> ModelBuilder:
    lay = Layout()
    lay.add(*PRIVATE)
    b = ModelBuilder(lay)
    working = {"seen_semi": 1.0}  # 0: working phase, 1: answer phase

    i = {"pos": 1.0, "E": -1.0}
    i_prev = pr.add(i, pr.const(-1.0))
    a_lo, a_hi = dict(FA), {"P": 1.0}  # magnitudes: fences sit past the signs
    b_lo, b_hi = dict(PB), {"E": 1.0}
    in_work_nz = {"nz": 1.0, "seen_eq": 1.0, "seen_semi": -1.0}  # == 2 on nonzero working digits

    l3 = b.at(3)
    l3.units += [
        cm.digit_target_unit(a_lo, a_hi, i, "tA"),
        cm.digit_target_unit(a_lo, a_hi, i_prev, "tA1"),
        cm.digit_target_unit(b_lo, b_hi, i, "tB"),
        cm.digit_target_unit(b_lo, b_hi, i_prev, "tB1"),
        *cm.exhausted_units(a_lo, a_hi, i, "exA"),
        *cm.exhausted_units(b_lo, b_hi, i, "exB"),
        pr.pos_if(in_work_nz, 2.0, "zkey"),
        pr.pos_if({"is_eq": 1.0}, 1.0, "zkey"),  # all-zero sum: the frame's '=' -> one '0'
    ]

    l4 = b.at(4)
    l4.heads += [
        pr.gather_head("a_i", pr.add(a_lo, {"tA": 1.0}), "A"),
        pr.gather_head("a_i-1", pr.add(a_lo, {"tA1": 1.0}), "A1"),
        pr.gather_head("b_i", pr.add(b_lo, {"tB": 1.0}), "B"),
        pr.gather_head("b_i-1", pr.add(b_lo, {"tB1": 1.0}), "B1"),
        pr.latest_head("top nonzero", "zkey", "Z"),
    ]
    y = {"A1": 1.0, "B1": 1.0}
    l4.units += pr.clamp01_units(pr.add(y, pr.const(-9.0)), {"carry": 1.0})  # [y >= 10]
    y_if_tok0 = pr.add(y, {"is0": -20.0}, pr.const(20.0))
    l4.units += pr.hat_units(y_if_tok0, 9.0, {"carry": 1.0})  # [y == 9 and token '0']
    l4.units.append(pr.relu_unit(pr.add({"Z": 1.0, "E": -1.0}, pr.const(-1.0)), {"Zo": 1.0}))
    # Routed here only when both effective signs agree, so the result has a's sign,
    # unless it is zero (no nonzero working digit: Z falls back to E).
    l4.units += pr.clamp01_units(pr.gated({"Z": 1.0, "E": -1.0}, {"sa": 1.0}), {"sgn": 1.0})

    l5 = b.at(5)
    j = {"pos": 1.0, "S": -1.0, "sgn": -1.0}  # answer digit index, after an optional '-' 
    l5.heads.append(pr.gather_head("answer digit", pr.add({"E": 1.0, "Zo": 1.0}, pr.scale(j, -1.0),
                                                          pr.const(1.0)), "C"))
    l5.units += cm.ramp_mod10_units({"A": 1.0, "B": 1.0, "carry": 1.0}, 0, 19, working, 0.0)
    l5.units.append(pr.relu_unit(
        pr.gated({"exA": 1.0, "exB": 1.0, "carry": -1.0, "one": -1.0}, working, 0.0), {"work_end": 1.0}))
    l5.units += cm.ramp_mod10_units({"C": 1.0}, 0, 9, working, 1.0)
    l5.units += pr.clamp01_units(pr.gated(pr.add(j, {"Zo": -1.0}), working, 1.0), {"frame_end": 1.0})
    l5.units.append(pr.relu_unit({"sgn": 1.0, "is_semi": 1.0, "one": -1.0}, {"emit_minus": 1.0}))
    return b
