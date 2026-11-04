"""Expression reduction in the weights: pick the redex, frame it, rewrite the line.

The source line X_t lies strictly between Ls (the ']' that closed the
previous frame, or BOS) and Le = F-1 (its '='). Every quantity below is
recomputed by attention at each position of step t; no position stores
state for another.

Redex selection (fixed depth, all attention):
  p   innermost group start: the latest '(' in X_t, else Ls       (frame L3)
  c   first ')' after p, else Le                                 next_match
  o   first '*' after p if it is before c, else first '+'/'-'    next_match
      after p  (precedence, left to right)
  a_lo  latest non-digit before o; if it is a unary '-', a is signed and
        its fence is a_lo-1  (AF = a_lo - sa_x)
  b_hi  first non-digit after o, or after o+1 when b is signed  next_match
  drop  the group is exactly "(a op b)": the parentheses go too
  A..B  the bytes replaced by the result: AF+1..b_hi-1 (or p..c on drop)
  cancel  a negative result right after a unary '-' drops both signs
  cont  X_t has another operator: latest op at Le is after o, or the
        latest op before o is inside the line

Writing:
  phase1  at the '=' ending a line: emit '['
  phase2  frame copy: byte AF+1+(q-F), then '=' after b_hi-AF-1 bytes
  (phase3 belongs to the operator circuits; they close the frame with ']')
  phase4  rewrite from Rb: prefix Ls+1..A-1, result S+1..Rb-1 (minus a
          cancelled sign), suffix
          B+1..Le-1, then '=' if cont else '\\n'

Copies move bytes as one-hots over the expression alphabet (digits, + - *
( ) ). Each gated copy unit is an AND of "this byte" and "this phase".

Layers 4..14 (layers 1..4 hold ``frame``, which shares layer 4).
"""

from __future__ import annotations

from . import common as cm
from . import primitives as pr
from .model import HeadSpec, ModelBuilder

NAMES = {byte: dim for byte, dim in cm.ALPHABET.items()}
COPY_LOGIT = cm.DIGIT_LOGIT

DIMS = (
    "PM2_Lrp", "PM2_Lmul", "PM2_Ladd", "PM2_Lnd", "Lt_Lrp", "Lt_Lmul", "Lt_Ladd",
    "c_raw", "fc", "om", "fm", "oa", "fa", "c", "use_mul", "o",
    "a_lo", "sb_x", "LopLe", "LopO", "e3", "cont_raw", "sa_x", "b_hi", "e1", "e2", "cont",
    *(f"Y2_{d}" for d in NAMES.values()), "rneg", "drop", "end2", "prevA_unary", "cancel",
    "in1", "in12", "in123", "end4", "src4",
    *(f"Y4_{d}" for d in NAMES.values()),
)
OUT_DIMS = (*(f"O_{d}" for d in NAMES.values()), "out_lbr", "out_eq", "out_nl")

LE = {"F": 1.0, "one": -1.0}
AF = {"a_lo": 1.0, "sa_x": -1.0}  # byte before the signed operand a
ND: dict[str, float] = {"one": 1.0, "is_digit": -1.0}


def _copy_head(name: str, src: dict[str, float], prefix: str) -> HeadSpec:
    return HeadSpec(
        name=name,
        query=[pr.scale(src, 2 * pr.GATHER_BETA), pr.const(-pr.GATHER_BETA)],
        key=[{"pos": 1.0}, {"pos2": 1.0}],
        value={d: {f"{prefix}_{d}": 1.0} for d in NAMES.values()},
    )


def build(b: ModelBuilder) -> None:
    l4 = b.at(4)
    for c in ("Lrp", "Lmul", "Ladd", "Lnd"):
        l4.heads.append(pr.gather_head(f"pm^2 {c}", {f"PM_{c}": 1.0}, f"PM2_{c}", src="pos2"))
    for c in ("Lrp", "Lmul", "Ladd"):
        l4.heads.append(pr.gather_head(f"latest {c} at p", {"p": 1.0}, f"Lt_{c}", src=c))

    l5 = b.at(5)
    l5.heads += [
        pr.next_match_head("first ) after p", "PM_Lrp", "PM2_Lrp", {"is_rpar": 1.0},
                           {"Lt_Lrp": 1.0}, "c_raw", "fc"),
        pr.next_match_head("first * after p", "PM_Lmul", "PM2_Lmul", {"is_star": 1.0},
                           {"Lt_Lmul": 1.0}, "om", "fm"),
        pr.next_match_head("first +- after p", "PM_Ladd", "PM2_Ladd",
                           {"is_plus": 1.0, "is_binminus": 1.0}, {"Lt_Ladd": 1.0}, "oa", "fa"),
    ]
    l5.units.append(pr.relu_unit(LE, {"c": 1.0}))
    l5.units += pr.times_flag_units(pr.add({"c_raw": 1.0}, pr.scale(LE, -1.0)), {"fc": 1.0}, {"c": 1.0})

    l6 = b.at(6)
    l6.units += pr.clamp01_units(pr.add({"c": 1.0, "om": -1.0}, pr.scale({"fm": 1.0}, pr.BIG),
                                        pr.const(-pr.BIG)), {"use_mul": 1.0})

    l7 = b.at(7)
    l7.units.append(pr.relu_unit({"oa": 1.0}, {"o": 1.0}))
    l7.units += pr.times_flag_units({"om": 1.0, "oa": -1.0}, {"use_mul": 1.0}, {"o": 1.0})

    l8 = b.at(8)
    l8.heads += [
        pr.gather_head("a fence", pr.add({"o": 1.0}, pr.const(-1.0)), "a_lo", src="Lnd"),
        pr.gather_head("sign of b?", pr.add({"o": 1.0}, pr.const(1.0)), "sb_x", src="is_minus"),
        pr.gather_head("latest op at Le", LE, "LopLe", src="P"),
        pr.gather_head("latest op before o", pr.add({"o": 1.0}, pr.const(-1.0)), "LopO", src="P"),
    ]
    l8.units += pr.clamp01_units({"p": 1.0, "Ls": -1.0}, {"e3": 1.0})
    l8.units += pr.clamp01_units({"LopLe": 1.0, "o": -1.0}, {"cont_raw": 1.0})
    l8.units += pr.clamp01_units({"LopO": 1.0, "Ls": -1.0}, {"cont_raw": 1.0})

    # Signed operands: a includes a unary '-' at a_lo; b's digits start past a '-' at o+1.
    l9 = b.at(9)
    l9.heads += [
        pr.gather_head("sign of a?", {"a_lo": 1.0}, "sa_x", src="is_unary"),
        pr.next_match_head("b fence", "PM_Lnd", "PM2_Lnd", ND, {"o": 1.0, "sb_x": 1.0}, "b_hi"),
    ]
    l9.units += pr.hat_units(pr.add(AF, {"p": -1.0}), 0.0, {"e1": 1.0})
    l9.units += pr.hat_units({"b_hi": 1.0, "c": -1.0}, 0.0, {"e2": 1.0})
    l9.units += pr.clamp01_units({"cont_raw": 1.0}, {"cont": 1.0})

    l10 = b.at(10)
    src2 = pr.add(AF, {"pos": 1.0, "F": -1.0}, pr.const(1.0))
    l10.heads += [
        _copy_head("frame copy", src2, "Y2"),
        pr.gather_head("result negative?", pr.add({"S": 1.0}, pr.const(1.0)), "rneg", src="is_minus"),
    ]
    l10.units.append(pr.relu_unit({"e1": 1.0, "e2": 1.0, "e3": 1.0, "one": -2.0}, {"drop": 1.0}))
    l10.units += pr.hat_units(pr.add({"pos": 1.0, "F": -1.0, "b_hi": -1.0}, AF, pr.const(1.0)),
                              0.0, {"end2": 1.0})

    # Rewrite geometry (linear forms over residual dims).
    A = pr.add(AF, {"drop": -1.0}, pr.const(1.0))
    B = pr.add({"b_hi": 1.0, "drop": 1.0}, pr.const(-1.0))

    # Double negation: a negative result replacing bytes right after a unary '-'
    # ("-(1-4)", "2*-(1-4)") drops both signs.
    l11 = b.at(11)
    l11.heads.append(pr.gather_head("unary '-' before A?", pr.add(A, pr.const(-1.0)), "prevA_unary",
                                    src="is_unary"))
    l11.units.append(pr.relu_unit({"prevA_unary": 1.0, "rneg": 1.0, "one": -1.0}, {"cancel": 1.0}))

    n1 = pr.add(A, {"Ls": -1.0, "cancel": -1.0}, pr.const(-1.0))
    n2 = pr.add({"Rb": 1.0, "S": -1.0, "cancel": -1.0}, pr.const(-1.0))
    n3 = pr.add(LE, pr.scale(B, -1.0), pr.const(-1.0))
    r = {"pos": 1.0, "Rb": -1.0}
    neg_r = pr.scale(r, -1.0)

    l12 = b.at(12)
    l12.units += pr.clamp01_units(pr.add(n1, neg_r), {"in1": 1.0})
    l12.units += pr.clamp01_units(pr.add(n1, n2, neg_r), {"in12": 1.0})
    l12.units += pr.clamp01_units(pr.add(n1, n2, n3, neg_r), {"in123": 1.0})
    l12.units += pr.hat_units(pr.add(r, pr.scale(pr.add(n1, n2, n3), -1.0)), 0.0, {"end4": 1.0})

    l13 = b.at(13)
    l13.units += pr.times_flag_units(pr.add({"Ls": 1.0}, pr.const(1.0), r), {"in1": 1.0}, {"src4": 1.0})
    l13.units += pr.times_flag_units(pr.add({"S": 1.0, "cancel": 1.0}, pr.const(1.0), r, pr.scale(n1, -1.0)),
                                     {"in12": 1.0, "in1": -1.0}, {"src4": 1.0})
    l13.units += pr.times_flag_units(pr.add(B, pr.const(1.0), r, pr.scale(n1, -1.0), pr.scale(n2, -1.0)),
                                     {"in123": 1.0, "in12": -1.0}, {"src4": 1.0})

    l14 = b.at(14)
    l14.heads.append(_copy_head("rewrite copy", {"src4": 1.0}, "Y4"))
    for d in NAMES.values():
        l14.units.append(pr.relu_unit({f"Y2_{d}": 1.0, "phase2": 1.0, "end2": -1.0, "one": -1.0},
                                      {f"O_{d}": 1.0}))
        l14.units.append(pr.relu_unit({f"Y4_{d}": 1.0, "phase4": 1.0, "end4": -1.0, "one": -1.0},
                                      {f"O_{d}": 1.0}))
    l14.units += [
        pr.relu_unit({"phase1": 1.0}, {"out_lbr": 1.0}),
        pr.relu_unit({"phase2": 1.0, "end2": 1.0, "one": -1.0}, {"out_eq": 1.0}),
        pr.relu_unit({"phase4": 1.0, "end4": 1.0, "cont": 1.0, "one": -2.0}, {"out_eq": 1.0}),
        pr.relu_unit({"phase4": 1.0, "end4": 1.0, "cont": -1.0, "one": -1.0}, {"out_nl": 1.0}),
    ]

    for byte, d in NAMES.items():
        b.logit(f"O_{d}", byte, COPY_LOGIT)
    b.logit("out_lbr", cm.FRAME_OPEN, cm.CONTROL_LOGIT)
    b.logit("out_eq", ord("="), cm.CONTROL_LOGIT)
    b.logit("out_nl", cm.EOS, cm.CONTROL_LOGIT)
