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
    """The addition circuit's private weights, at global layers 3-5 (layers 1-2 are empty).

    arith.builder merges them under the ``add.`` namespace and silences every unit that writes
    an output dim (D0..D9, work_end, frame_end, emit_minus) unless ``act_add`` (frame layer 4).
    Shared inputs: F, P, E, S (frame layer-2 heads), seen_eq, seen_semi (layer-2 MLP), sa, sb
    (layer-3 heads).
    """
    lay = Layout()
    lay.add(*PRIVATE)
    b = ModelBuilder(lay)
    working = {"seen_semi": 1.0}  # 0: working phase, 1: answer phase

    # At position E + i the model predicts working digit i (written at E+1+i), so i = pos - E.
    # i_prev = i - 1 addresses the column below, for the carry.
    i = {"pos": 1.0, "E": -1.0}
    i_prev = pr.add(i, pr.const(-1.0))
    a_lo, a_hi = dict(FA), {"P": 1.0}  # magnitudes: fences sit past the signs
    b_lo, b_hi = dict(PB), {"E": 1.0}
    in_work_nz = {"nz": 1.0, "seen_eq": 1.0, "seen_semi": -1.0}  # == 2 on nonzero working digits

    # Layer 3 MLP: digit addresses and key features. tA = max(hi-1-i - lo, 0) is digit i's offset
    # from the fence lo (0 once exhausted: the layer-4 read lands on the fence, value 0; at i = 0,
    # i_prev = -1 lands on hi, value 0: no carry in). exA = 1 - [a has digit i]. zkey = pos on
    # nonzero working digits of a frame (each key's own flags), for layer 4's "top nonzero" head.
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

    # Layer 4: gather the digits (val at lo + offset) and the top nonzero working digit Z; MLP:
    # the carry, the answer length Zo and the sign. A, B feed layer 5's digit; A1, B1 the carry.
    l4 = b.at(4)
    l4.heads += [
        pr.gather_head("a_i", pr.add(a_lo, {"tA": 1.0}), "A"),
        pr.gather_head("a_i-1", pr.add(a_lo, {"tA1": 1.0}), "A1"),
        pr.gather_head("b_i", pr.add(b_lo, {"tB": 1.0}), "B"),
        pr.gather_head("b_i-1", pr.add(b_lo, {"tB1": 1.0}), "B1"),
        # Z = largest zkey: position of the highest nonzero working digit (E for an all-zero sum).
        pr.latest_head("top nonzero", "zkey", "Z"),
    ]
    # carry into column i = carry out of column i-1, recovered from the text, never stored.
    # y = a_{i-1} + b_{i-1} (0..18). y >= 10 always carries: clamp01 tests y - 9 >= 1.
    y = {"A1": 1.0, "B1": 1.0}
    l4.units += pr.clamp01_units(pr.add(y, pr.const(-9.0)), {"carry": 1.0})  # [y >= 10]
    # y == 9 carries only if column i-1 had a carry in, i.e. it wrote 9+1 -> '0'; that digit is
    # this position's own byte. +20 moves y out of the equality test's range unless the byte is
    # '0' (is0 = 1), so the hat at 9 fires only for y == 9 AND a written '0'. The cases are
    # disjoint, so carry is exactly 0/1; layer 5 reads it (working digit and work_end).
    y_if_tok0 = pr.add(y, {"is0": -20.0}, pr.const(20.0))
    l4.units += pr.hat_units(y_if_tok0, 9.0, {"carry": 1.0})  # [y == 9 and token '0']
    l4.units.append(pr.relu_unit(pr.add({"Z": 1.0, "E": -1.0}, pr.const(-1.0)), {"Zo": 1.0},
                                 "answer length minus 1: Z - E - 1, how far the top nonzero working digit sits past "
                                 "the first one; 0 for an all-zero sum (Z falls back to E)"))
    # Routed here only when both effective signs agree, so the result has a's sign,
    # unless it is zero (no nonzero working digit: Z falls back to E).
    # sgn = [Z - E >= 1 and sa == 1]; gated() adds -BIG unless sa = 1.
    l4.units += pr.clamp01_units(pr.gated({"Z": 1.0, "E": -1.0}, {"sa": 1.0}), {"sgn": 1.0})

    # Layer 5: copy the answer back (head) and write every output byte (MLP, gated by act_add).
    l5 = b.at(5)
    j = {"pos": 1.0, "S": -1.0, "sgn": -1.0}  # answer digit index, after an optional '-' 
    # Answer digit j is working digit Zo - j, at E + 1 + Zo - j: from the top nonzero digit down.
    l5.heads.append(pr.gather_head("answer digit", pr.add({"E": 1.0, "Zo": 1.0}, pr.scale(j, -1.0),
                                                          pr.const(1.0)), "C"))
    # Working phase: one-hot D[(a_i + b_i + carry) mod 10] for sums 0..19 (shared ReLU ramps).
    l5.units += cm.ramp_mod10_units({"A": 1.0, "B": 1.0, "carry": 1.0}, 0, 19, working, 0.0)
    l5.units.append(pr.relu_unit(
        pr.gated({"exA": 1.0, "exB": 1.0, "carry": -1.0, "one": -1.0}, working, 0.0), {"work_end": 1.0},
        "end of working: both operands exhausted (exA, exB) and no carry left, so write ';'"))
    # Answer phase: the copied digit C, then ']' once all Zo+1 digits are out (j > Zo).
    # work_end, frame_end and emit_minus use CONTROL_LOGIT, so they beat a digit voted the same step.
    l5.units += cm.ramp_mod10_units({"C": 1.0}, 0, 9, working, 1.0)
    l5.units += pr.clamp01_units(pr.gated(pr.add(j, {"Zo": -1.0}), working, 1.0), {"frame_end": 1.0})
    l5.units.append(pr.relu_unit({"sgn": 1.0, "is_semi": 1.0, "one": -1.0}, {"emit_minus": 1.0},
                                 "on the ';' byte of a negative result (sgn): write '-' before the answer digits"))
    return b
