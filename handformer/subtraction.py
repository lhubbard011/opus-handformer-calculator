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
    lay = Layout()
    lay.add(*PRIVATE)
    b = ModelBuilder(lay)
    working = {"seen_semi": 1.0}
    work_digit = {"seen_eq": 1.0, "seen_semi": -1.0}  # +1 on a digit: in_work sum == 2

    i = {"pos": 1.0, "E": -1.0}
    i_prev = pr.add(i, pr.const(-1.0))
    a_lo, a_hi = dict(FA), {"P": 1.0}  # magnitudes: fences sit past the signs
    b_lo, b_hi = dict(PB), {"E": 1.0}

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

    l4 = b.at(4)
    l4.heads += [
        pr.gather_head("a_i", pr.add(a_lo, {"tA": 1.0}), "A"),
        pr.gather_head("a_i-1", pr.add(a_lo, {"tA1": 1.0}), "A1"),
        pr.gather_head("b_i", pr.add(b_lo, {"tB": 1.0}), "B"),
        pr.gather_head("b_i-1", pr.add(b_lo, {"tB1": 1.0}), "B1"),
        pr.latest_head("top nonzero", "zkey", "Zp"),
        pr.latest_head("top non-9", "n9key", "Z9"),
        pr.latest_head("lowest nonzero", "zlow", "Zlow"),
    ]
    y = {"A1": 1.0, "B1": -1.0}
    l4.units += pr.clamp01_units(pr.scale(y, -1.0), {"br": 1.0})
    l4.units += pr.hat_units(pr.add(y, {"is9": -20.0}, pr.const(20.0)), 0.0, {"br": 1.0})
    l4.units.append(pr.relu_unit(pr.add({"Zp": 1.0, "E": -1.0}, pr.const(-1.0)), {"Zo": 1.0}))
    # Final sign = sa XOR borrow, suppressed for a zero result:
    #   sa=1, no borrow: negative iff |a|-|b| != 0;  sa=0, borrow: always negative.
    l4.units += pr.clamp01_units(pr.add({"Zp": 1.0, "E": -1.0}, pr.scale({"sa": 1.0}, pr.BIG),
                                        pr.scale({"neg": 1.0}, -pr.BIG), pr.const(-pr.BIG)), {"sgn": 1.0})
    l4.units.append(pr.relu_unit({"neg": 1.0, "sa": -1.0}, {"sgn": 1.0}))
    l4.units += [
        pr.relu_unit({"Zlow": 1.0}, {"topneg": 1.0}),
        pr.relu_unit({"Z9": 1.0, "Zlow": -1.0}, {"topneg": 1.0}),
    ]

    l5 = b.at(5)
    l5.units += cm.ramp_mod10_units({"A": 1.0, "B": -1.0, "br": -1.0}, -10, 9, working, 0.0)
    done = {"exA": 1.0, "exB": 1.0, "one": -1.0}
    l5.units.append(pr.relu_unit(pr.gated(pr.add(done, {"br": -1.0}), working, 0.0), {"work_end": 1.0}))
    l5.units.append(pr.relu_unit(pr.gated(pr.add(done, {"br": 1.0}, pr.const(-1.0)), working, 0.0),
                                 {"work_end_neg": 1.0}))
    top_pos = pr.add({"E": 1.0, "Zo": 1.0}, pr.const(1.0))
    l5.units.append(pr.relu_unit(top_pos, {"top": 1.0}))
    l5.units += pr.times_flag_units(pr.add({"topneg": 1.0}, pr.scale(top_pos, -1.0)), {"neg": 1.0},
                                    {"top": 1.0})

    l6 = b.at(6)
    j = {"pos": 1.0, "S": -1.0, "sgn": -1.0}  # answer index after an optional '-'
    src = pr.add({"top": 1.0}, pr.scale(j, -1.0))
    l6.heads.append(pr.gather_head("answer digit", src, "Dv"))
    rel = pr.add(src, {"Zlow": -1.0})
    l6.units += pr.clamp01_units(rel, {"above": 1.0})
    l6.units += pr.hat_units(rel, 0.0, {"at": 1.0})
    l6.units += pr.clamp01_units(pr.gated(pr.add({"E": 1.0}, pr.const(1.0), pr.scale(src, -1.0)),
                                          working), {"frame_end": 1.0})
    l6.units.append(pr.relu_unit({"sgn": 1.0, "is_semi": 1.0, "one": -1.0}, {"emit_minus": 1.0}))

    l7 = b.at(7)
    l7.units += [
        pr.relu_unit({"Dv": 1.0, "neg": -9.0}, {"V": 1.0}),
        pr.relu_unit({"Dv": -1.0, "one": -9.0, "neg": 9.0, "above": 9.0}, {"V": 1.0}),
        pr.relu_unit({"Dv": -1.0, "one": -10.0, "neg": 10.0, "at": 10.0}, {"V": 1.0}),
    ]

    l8 = b.at(8)
    l8.units += cm.ramp_mod10_units({"V": 1.0}, 0, 9, working, 1.0)
    return b
