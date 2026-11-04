"""Subtraction circuit, inside a frame ``[ a - b = working T [-] answer ]``.

* working: D = (a - b) mod 10^n LSB first, n = max(len a, len b)
* T: ``;`` if the final borrow is 0 (a >= b), ``<`` if it is 1 (a < b);
  the sign is stored in the only memory the model has, the token
* answer: for a >= b, D in reverse from its top nonzero digit. For a < b,
  ``-`` then 10^n - D, complemented *during the copy*: 9-d above the lowest
  nonzero digit z of D, 10-d at z, 0 below. The top digit of 10^n - D is
  max(z, latest digit of D that is not 9).

borrow_i = [y <= -1] + [y == 0 and previous output digit == '9'],
y = a_{i-1} - b_{i-1}.

z is the earliest nonzero working digit *of this frame*: a lexicographic key
feature ``BIG*F_k - pos`` makes keys of the newest frame win, then the earliest.

Shared ``neg`` (frame L3) is the borrow flag of the end-of-working byte: it
says |a| < |b|. This circuit is routed + and - whose effective signs differ,
so it subtracts magnitudes; the answer's sign is sa XOR neg (none for 0).

Global layers: 3 targets/features, 4 gathers + borrow, 5 working digit + top,
6 copy, 7 complement, 8 answer digit.
"""

from __future__ import annotations

from . import common as cm
from . import primitives as pr
from .frame import FA, PB
from .model import Layout, ModelBuilder

PRIVATE = ("tA", "tA1", "tB", "tB1", "exA", "exB", "zkey", "n9key", "zlow",
           "A", "A1", "B", "B1", "Zp", "Z9", "Zlow", "br", "Zo", "topneg", "top",
           "sgn", "Dv", "above", "at", "V")


def builder() -> ModelBuilder:
    """The subtraction circuit's private weights, at global layers 3-8 (layers 1-2 are empty).

    arith.builder merges them under the ``sub.`` namespace and silences every unit that writes
    an output dim unless ``act_sub`` (frame layer 4). Shared inputs: F, P, E, S (frame layer-2
    heads), seen_eq, seen_semi (layer-2 MLP), sa, sb, neg (layer-3 heads).
    """
    lay = Layout()
    lay.add(*PRIVATE)
    b = ModelBuilder(lay)
    working = {"seen_semi": 1.0}  # 0: working phase, 1: answer phase (';' or '<' written)
    work_digit = {"seen_eq": 1.0, "seen_semi": -1.0}  # +1 on a digit: in_work sum == 2

    # At position E + i the model predicts working digit i (written at E+1+i), so i = pos - E.
    i = {"pos": 1.0, "E": -1.0}
    i_prev = pr.add(i, pr.const(-1.0))
    a_lo, a_hi = dict(FA), {"P": 1.0}  # magnitudes: fences sit past the signs
    b_lo, b_hi = dict(PB), {"E": 1.0}

    # Layer 3 MLP: digit addresses (as in addition: offset from the fence, 0 once exhausted) and
    # three key features over the working digits D: zkey = pos if nonzero, n9key = pos if not 9
    # (both fall back to the '='), zlow = BIG·F - pos if nonzero (newest frame, then earliest wins).
    l3 = b.at(3)
    l3.units += [
        cm.digit_target_unit(a_lo, a_hi, i, "tA"),
        cm.digit_target_unit(a_lo, a_hi, i_prev, "tA1"),
        cm.digit_target_unit(b_lo, b_hi, i, "tB"),
        cm.digit_target_unit(b_lo, b_hi, i_prev, "tB1"),
        *cm.exhausted_units(a_lo, a_hi, i, "exA"),
        *cm.exhausted_units(b_lo, b_hi, i, "exB"),
        pr.pos_if(pr.add(work_digit, {"nz": 1.0}), 2.0, "zkey"),
        pr.pos_if({"is_eq": 1.0}, 1.0, "zkey"),
        pr.pos_if(pr.add(work_digit, {"n9": 1.0}), 2.0, "n9key"),
        pr.pos_if({"is_eq": 1.0}, 1.0, "n9key"),
        pr.newest_frame_earliest(pr.add(work_digit, {"nz": 1.0}), 2.0, "F", "zlow"),
    ]

    # Layer 4: gather the digits and three landmarks of D; MLP: borrow, answer length, sign, and
    # where a negative answer starts.
    l4 = b.at(4)
    l4.heads += [
        pr.gather_head("a_i", pr.add(a_lo, {"tA": 1.0}), "A"),
        pr.gather_head("a_i-1", pr.add(a_lo, {"tA1": 1.0}), "A1"),
        pr.gather_head("b_i", pr.add(b_lo, {"tB": 1.0}), "B"),
        pr.gather_head("b_i-1", pr.add(b_lo, {"tB1": 1.0}), "B1"),
        # The landmarks the complement-on-copy needs (read in the answer phase, when D is complete):
        # Zp = top nonzero digit of D: where the answer starts if a >= b.
        # Z9 = top non-9 digit, Zlow = lowest nonzero digit z: 10^n - D has digit 9-d above z,
        # 10-d (nonzero) at z and 0 below, so its top nonzero digit is at max(Zlow, Z9) (topneg).
        # Zp and Z9 read the latest key, which is in this frame; Zlow needs the frame-ranked key.
        pr.latest_head("top nonzero", "zkey", "Zp"),
        pr.latest_head("top non-9", "n9key", "Z9"),
        pr.latest_head("lowest nonzero", "zlow", "Zlow"),
    ]
    # borrow into column i, recovered from the text: y = a_{i-1} - b_{i-1} (-9..9).
    # y <= -1 always borrows: clamp01 tests -y >= 1.
    y = {"A1": 1.0, "B1": -1.0}
    l4.units += pr.clamp01_units(pr.scale(y, -1.0), {"br": 1.0})
    # y == 0 borrows only if column i-1 had a borrow in, i.e. it wrote 0-1 -> '9' (this byte).
    # +20 moves y out of the equality test's range unless the byte is '9' (is9 = 1).
    l4.units += pr.hat_units(pr.add(y, {"is9": -20.0}, pr.const(20.0)), 0.0, {"br": 1.0})
    l4.units.append(pr.relu_unit(pr.add({"Zp": 1.0, "E": -1.0}, pr.const(-1.0)), {"Zo": 1.0},
                                 "Zp - E - 1: how far the top nonzero working digit sits past the first one (answer "
                                 "length minus 1 when a >= b); 0 for a zero result (Zp falls back to E)"))
    # Final sign = sa XOR borrow, suppressed for a zero result:
    #   sa=1, no borrow: negative iff |a|-|b| != 0;  sa=0, borrow: always negative.
    # borrow = neg (frame layer 3): 1 iff the END byte is '<', i.e. |a| < |b|. Case 1 is the step
    # [Zp - E >= 1] (a nonzero digit exists), pushed down by BIG unless sa = 1 and neg = 0.
    # sgn is read in layer 6 (answer index j, emit_minus).
    l4.units += pr.clamp01_units(pr.add({"Zp": 1.0, "E": -1.0}, pr.scale({"sa": 1.0}, pr.BIG),
                                        pr.scale({"neg": 1.0}, -pr.BIG), pr.const(-pr.BIG)), {"sgn": 1.0})
    l4.units.append(pr.relu_unit({"neg": 1.0, "sa": -1.0}, {"sgn": 1.0},
                                 "sign case sa=0 with borrow (|a| < |b|): the result is negative, and never zero"))
    l4.units += [
        pr.relu_unit({"Zlow": 1.0}, {"topneg": 1.0},
                     "topneg = max(Zlow, Z9), part 1: the lowest nonzero working digit's position"),
        pr.relu_unit({"Z9": 1.0, "Zlow": -1.0}, {"topneg": 1.0},
                     "topneg = max(Zlow, Z9), part 2: adds Z9 - Zlow when the top non-9 digit is higher; the top "
                     "digit of 10^n - D sits there"),
    ]

    # Layer 5 MLP: the working digit or END byte, and where the answer copy starts.
    l5 = b.at(5)
    # Working digit: D[(a_i - b_i - borrow) mod 10] for differences -10..9 (shared ReLU ramps).
    l5.units += cm.ramp_mod10_units({"A": 1.0, "B": -1.0, "br": -1.0}, -10, 9, working, 0.0)
    # Once both operands are used up, the borrow still pending is the final one: ';' if 0, '<' if 1.
    done = {"exA": 1.0, "exB": 1.0, "one": -1.0}
    l5.units.append(pr.relu_unit(pr.gated(pr.add(done, {"br": -1.0}), working, 0.0), {"work_end": 1.0},
                                 "end of working with no final borrow (a >= b): both operands exhausted, write ';'"))
    l5.units.append(pr.relu_unit(pr.gated(pr.add(done, {"br": 1.0}, pr.const(-1.0)), working, 0.0),
                                 {"work_end_neg": 1.0},
                                 "end of working with a final borrow (a < b): both operands exhausted, write '<' "
                                 "(the sign lives in this byte)"))
    # top = position of the answer's first digit: E + 1 + Zo if a >= b, topneg if a < b (neg 1).
    top_pos = pr.add({"E": 1.0, "Zo": 1.0}, pr.const(1.0))
    l5.units.append(pr.relu_unit(top_pos, {"top": 1.0},
                                 "answer source for a >= b: E + 1 + Zo, the top nonzero working digit (the select "
                                 "below swaps in topneg when neg)"))
    l5.units += pr.times_flag_units(pr.add({"topneg": 1.0}, pr.scale(top_pos, -1.0)), {"neg": 1.0},
                                    {"top": 1.0})

    # Layer 6: copy working digits from top downward (head); MLP: where the copy is relative to z.
    l6 = b.at(6)
    j = {"pos": 1.0, "S": -1.0, "sgn": -1.0}  # answer index after an optional '-'
    src = pr.add({"top": 1.0}, pr.scale(j, -1.0))
    l6.heads.append(pr.gather_head("answer digit", src, "Dv"))
    # rel = src - Zlow: above = [source above z], at = [source is z]; layer 7 picks 9-d / 10-d.
    # frame_end = [src <= E]: the units digit (E+1) is already out, write ']'.
    rel = pr.add(src, {"Zlow": -1.0})
    l6.units += pr.clamp01_units(rel, {"above": 1.0})
    l6.units += pr.hat_units(rel, 0.0, {"at": 1.0})
    l6.units += pr.clamp01_units(pr.gated(pr.add({"E": 1.0}, pr.const(1.0), pr.scale(src, -1.0)),
                                          working), {"frame_end": 1.0})
    l6.units.append(pr.relu_unit({"sgn": 1.0, "is_semi": 1.0, "one": -1.0}, {"emit_minus": 1.0},
                                 "on the end-of-working byte of a negative result (sgn): write '-' before the answer"))

    # Layer 7 MLP: the answer digit V, complemented during the copy when a < b (0 below z).
    # Each condition of a unit is worth 9 (or 10), so with one missing its input is <= 0 (Dv 0..9).
    l7 = b.at(7)
    l7.units += [
        pr.relu_unit({"Dv": 1.0, "neg": -9.0}, {"V": 1.0},
                     "a >= b (neg 0): the answer digit is the working digit Dv, copied as is"),
        pr.relu_unit({"Dv": -1.0, "one": -9.0, "neg": 9.0, "above": 9.0}, {"V": 1.0},
                     "a < b, above the lowest nonzero digit: complement digit 9 - Dv"),
        pr.relu_unit({"Dv": -1.0, "one": -10.0, "neg": 10.0, "at": 10.0}, {"V": 1.0},
                     "a < b, at the lowest nonzero digit: complement digit 10 - Dv (below it every digit is 0)"),
    ]

    # Layer 8 MLP: the answer digit V as a one-hot vote, answer phase only.
    l8 = b.at(8)
    l8.units += cm.ramp_mod10_units({"V": 1.0}, 0, 9, working, 1.0)
    return b
