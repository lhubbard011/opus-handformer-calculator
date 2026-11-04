"""Expression reduction in the weights: pick the redex, frame it, rewrite the line.

The source line X_t lies strictly between Ls (the ']' that closed the
previous frame, or BOS) and Le = F-1 (its '='). Every quantity below is
recomputed by attention at each position of step t; no position stores
state for another. A number is a run of digits and at most one '.'.

Redex selection (fixed depth, all attention):
  p   innermost group start: the latest '(' in X_t, else Ls       (frame L3)
  c   first ')' after p, else Le                                 next_match
  o   first '*' after p if it is before c, else first '+'/'-'    next_match
      after p  (precedence, left to right)
  a_lo  latest non-number byte before o; if it is a unary '-', a is signed
        and its fence is a_lo-1  (AF = a_lo - sa_x)
  b_hi  first non-number byte after o (after o+1 when b is signed)  next_match
  drop  the group is exactly "(a op b)": the parentheses go too
  A..B  the bytes replaced by the result: AF+1..b_hi-1 (or p..c on drop)
  cancel  a negative result right after a unary '-' drops both signs
  cont  X_t has another operator

Decimals (the operator circuits only ever see integers):
  frac_a, frac_b  digits after each operand's '.' (0 without one)
  frame copy      each operand without its '.'; for + and - the shorter
                  fraction is padded with zeros so the two are aligned:
                  1.25+3.5 -> [125+350=...]
  f_res           fractional digits of the result: max(frac_a, frac_b)
                  for + and -, frac_a + frac_b for *
  result rewrite  strip trailing fractional zeros (tz of them), insert '.'
                  f' = f_res - tz digits from the right, left-pad with zeros
                  when the result is shorter: 2 with f'=2 -> 0.02

Writing:
  phase1  at the '=' ending a line: emit '['
  phase2  frame copy (segments: a, zero pad, op, b, zero pad), then '='
  (phase3 belongs to the operator circuits; they close the frame with ']')
  phase4  rewrite from Rb: prefix Ls+1..A-1, formatted result, suffix
          B+1..Le-1, then '=' if cont else '\\n'

Copies move bytes as one-hots over the expression alphabet; fixed bytes
('0' padding, '.') are written directly. Each output unit is an AND of
"this byte or this region" and "this phase".

Layers 4..21 (layers 1..4 hold ``frame``, which shares layer 4).
"""

from __future__ import annotations

from . import common as cm
from . import primitives as pr
from .model import Form, HeadSpec, ModelBuilder

NAMES = {byte: dim for byte, dim in cm.ALPHABET.items()}
COPY_LOGIT = cm.DIGIT_LOGIT

DIMS = (
    "PM2_Lrp", "PM2_Lmul", "PM2_Ladd", "PM2_Lnd", "Lt_Lrp", "Lt_Lmul", "Lt_Ladd",
    "c_raw", "fc", "om", "fm", "oa", "fa", "c", "use_mul", "o",
    "a_lo", "sb_x", "LopLe", "LopO", "op_div", "e3", "cont_raw", "sa_x", "b_hi", "a_dot", "e1", "e2", "cont",
    "has_a", "rneg", "b_dot", "drop", "frac_a", "has_b", "prevA_unary", "Lnzr", "cancel", "frac_b",
    "found_nz", "pad_a_raw", "pad_b_raw", "pad_a", "pad_b", "f_res", "tzx",
    "inA", "inPA", "atOp", "inB", "inPB", "end2", "skipA", "skipB",
    "tz", "src2", *(f"Y2_{d}" for d in NAMES.values()), "fp", "til_line", "Lend", "wd", "dotf",
    "in1", "in12", "in123", "end4", "isSign", "isDot", "afterdot", "padzero", "src4",
    *(f"Y4_{d}" for d in NAMES.values()),
)
OUT_DIMS = (*(f"O_{d}" for d in NAMES.values()), "out_lbr", "out_eq", "out_nl")

LE: Form = {"F": 1.0, "one": -1.0}
AF: Form = {"a_lo": 1.0, "sa_x": -1.0}  # byte before the signed operand a
ND: Form = {"one": 1.0, "is_digit": -1.0, "is_dot": -1.0}  # not part of a number
BSTART: Form = {"o": 1.0, "sb_x": 1.0}  # byte before b's digits (o, or b's sign)


def _copy_head(name: str, src: Form, prefix: str) -> HeadSpec:
    return HeadSpec(
        name=name,
        query=[pr.scale(src, 2 * pr.GATHER_BETA), pr.const(-pr.GATHER_BETA)],
        key=[{"pos": 1.0}, {"pos2": 1.0}],
        value={d: {f"{prefix}_{d}": 1.0} for d in NAMES.values()},
    )


def _lt(x: Form, y: Form, dst: str, sign: float = 1.0) -> list:
    """``dst += sign * [x < y]`` for integers."""
    return pr.clamp01_units(pr.add(y, pr.scale(x, -1.0)), {dst: sign})


def build(b: ModelBuilder) -> None:
    neg = lambda f: pr.scale(f, -1.0)  # noqa: E731

    # ---------------------------------------------------------- redex search
    l4 = b.at(4)
    for c in ("Lrp", "Lmul", "Ladd", "Lnd"):
        l4.heads.append(pr.gather_head(f"pm^2 {c}", {f"PM_{c}": 1.0}, f"PM2_{c}", src="pos2"))
    for c in ("Lrp", "Lmul", "Ladd"):
        l4.heads.append(pr.gather_head(f"latest {c} at p", {"p": 1.0}, f"Lt_{c}", src=c))

    l5 = b.at(5)
    l5.heads += [
        pr.next_match_head("first ) after p", "PM_Lrp", "PM2_Lrp", {"is_rpar": 1.0},
                           {"Lt_Lrp": 1.0}, "c_raw", "fc"),
        pr.next_match_head("first * or / after p", "PM_Lmul", "PM2_Lmul", {"is_star": 1.0, "is_slash": 1.0},
                           {"Lt_Lmul": 1.0}, "om", "fm"),
        pr.next_match_head("first +- after p", "PM_Ladd", "PM2_Ladd",
                           {"is_plus": 1.0, "is_binminus": 1.0}, {"Lt_Ladd": 1.0}, "oa", "fa"),
    ]
    l5.units.append(pr.relu_unit(LE, {"c": 1.0}))
    l5.units += pr.times_flag_units(pr.add({"c_raw": 1.0}, neg(LE)), {"fc": 1.0}, {"c": 1.0})

    l6 = b.at(6)
    l6.units += pr.clamp01_units(pr.add({"c": 1.0, "om": -1.0}, pr.scale({"fm": 1.0}, pr.BIG),
                                        pr.const(-pr.BIG)), {"use_mul": 1.0})

    l7 = b.at(7)
    l7.units.append(pr.relu_unit({"oa": 1.0}, {"o": 1.0}))
    l7.units += pr.times_flag_units({"om": 1.0, "oa": -1.0}, {"use_mul": 1.0}, {"o": 1.0})

    # ---------------------------------------------------------- operands
    l8 = b.at(8)
    l8.heads += [
        pr.gather_head("a fence", pr.add({"o": 1.0}, pr.const(-1.0)), "a_lo", src="Lnd"),
        pr.gather_head("sign of b?", pr.add({"o": 1.0}, pr.const(1.0)), "sb_x", src="is_minus"),
        pr.gather_head("latest op at Le", LE, "LopLe", src="P"),
        pr.gather_head("latest op before o", pr.add({"o": 1.0}, pr.const(-1.0)), "LopO", src="P"),
        pr.gather_head("redex is a division?", {"o": 1.0}, "op_div", src="is_slash"),
    ]
    l8.units += pr.clamp01_units({"p": 1.0, "Ls": -1.0}, {"e3": 1.0})
    l8.units += pr.clamp01_units({"LopLe": 1.0, "o": -1.0}, {"cont_raw": 1.0})
    l8.units += pr.clamp01_units({"LopO": 1.0, "Ls": -1.0}, {"cont_raw": 1.0})

    l9 = b.at(9)
    l9.heads += [
        pr.gather_head("sign of a?", {"a_lo": 1.0}, "sa_x", src="is_unary"),
        pr.next_match_head("b fence", "PM_Lnd", "PM2_Lnd", ND, BSTART, "b_hi"),
        pr.gather_head("a's '.'", pr.add({"o": 1.0}, pr.const(-1.0)), "a_dot", src="Ldot"),
    ]
    l9.units += pr.hat_units(pr.add(AF, {"p": -1.0}), 0.0, {"e1": 1.0})
    l9.units += pr.hat_units({"b_hi": 1.0, "c": -1.0}, 0.0, {"e2": 1.0})
    l9.units += pr.clamp01_units({"cont_raw": 1.0}, {"cont": 1.0})
    l9.units += pr.clamp01_units({"a_dot": 1.0, "a_lo": -1.0}, {"has_a": 1.0})  # '.' inside a

    l10 = b.at(10)
    l10.heads += [
        pr.gather_head("result negative?", pr.add({"S": 1.0}, pr.const(1.0)), "rneg", src="is_minus"),
        pr.gather_head("b's '.'", pr.add({"b_hi": 1.0}, pr.const(-1.0)), "b_dot", src="Ldot"),
    ]
    l10.units.append(pr.relu_unit({"e1": 1.0, "e2": 1.0, "e3": 1.0, "one": -2.0}, {"drop": 1.0}))
    l10.units += pr.times_flag_units(pr.add({"o": 1.0, "a_dot": -1.0}, pr.const(-1.0)), {"has_a": 1.0},
                                     {"frac_a": 1.0})

    A = pr.add(AF, {"drop": -1.0}, pr.const(1.0))
    B = pr.add({"b_hi": 1.0, "drop": 1.0}, pr.const(-1.0))

    # Double negation: a negative result replacing bytes right after a unary '-'
    # ("-(1-4)", "2*-(1-4)") drops both signs.
    l11 = b.at(11)
    l11.heads += [
        pr.gather_head("unary '-' before A?", pr.add(A, pr.const(-1.0)), "prevA_unary", src="is_unary"),
        pr.gather_head("result's last nonzero digit", pr.add({"Rb": 1.0}, pr.const(-1.0)), "Lnzr", src="Lnz"),
    ]
    l11.units.append(pr.relu_unit({"prevA_unary": 1.0, "rneg": 1.0, "one": -1.0}, {"cancel": 1.0}))
    l11.units += pr.clamp01_units(pr.add({"b_dot": 1.0}, neg(BSTART)), {"has_b": 1.0})
    l11.units += pr.clamp01_units({"Lnzr": 1.0, "S": -1.0, "rneg": -1.0}, {"found_nz": 1.0})

    l12 = b.at(12)
    l12.units += pr.times_flag_units(pr.add({"b_hi": 1.0, "b_dot": -1.0}, pr.const(-1.0)), {"has_b": 1.0},
                                     {"frac_b": 1.0})

    # ---------------------------------------------------------- decimal geometry
    l13 = b.at(13)
    l13.units += [
        pr.relu_unit({"frac_b": 1.0, "frac_a": -1.0}, {"pad_a_raw": 1.0}),
        pr.relu_unit({"frac_a": 1.0, "frac_b": -1.0}, {"pad_b_raw": 1.0}),
    ]

    l14 = b.at(14)
    l14.units += [
        pr.relu_unit(pr.gated({"pad_a_raw": 1.0}, {"use_mul": 1.0}, 0.0), {"pad_a": 1.0}),
        pr.relu_unit(pr.gated({"pad_b_raw": 1.0}, {"use_mul": 1.0}, 0.0), {"pad_b": 1.0}),
        pr.relu_unit({"frac_a": 1.0, "pad_a_raw": 1.0}, {"f_res": 1.0}),  # max(frac_a, frac_b)
    ]
    l14.units += pr.times_flag_units({"frac_b": 1.0, "pad_a_raw": -1.0}, {"use_mul": 1.0}, {"f_res": 1.0})
    # a/b: the frame holds a*10^(frac_b+N) / b*10^frac_a, so the integer quotient has N fractional digits
    l14.units += pr.times_flag_units({"frac_b": 1.0, "N": 1.0}, {"op_div": 1.0}, {"pad_a": 1.0})
    l14.units += pr.times_flag_units({"frac_a": 1.0}, {"op_div": 1.0}, {"pad_b": 1.0})
    l14.units += pr.times_flag_units({"N": 1.0, "frac_a": -1.0, "frac_b": -1.0}, {"op_div": 1.0}, {"f_res": 1.0})

    # frame copy segments, t = q - F
    t = {"pos": 1.0, "F": -1.0}
    la_ = pr.add({"o": 1.0, "has_a": -1.0}, neg(AF), pr.const(-1.0))  # signed a without '.'
    lb_ = pr.add({"b_hi": 1.0, "o": -1.0, "has_b": -1.0}, pr.const(-1.0))
    T1 = la_
    T2 = pr.add(T1, {"pad_a": 1.0})
    T3 = pr.add(T2, pr.const(1.0))
    T4 = pr.add(T3, lb_)
    T5 = pr.add(T4, {"pad_b": 1.0})
    u = pr.add(t, neg(T3))  # index inside b's segment

    l15 = b.at(15)
    l15.units.append(pr.relu_unit({"Rb": 1.0, "Lnzr": -1.0, "f_res": -1.0, "one": -1.0},
                                  {"tzx": 1.0}))  # ReLU(trailing zeros - f_res)
    l15.units += _lt(t, T1, "inA")
    l15.units += _lt(t, T2, "inPA") + _lt(t, T1, "inPA", -1.0)
    l15.units += _lt(t, T3, "atOp") + _lt(t, T2, "atOp", -1.0)
    l15.units += _lt(t, T4, "inB") + _lt(t, T3, "inB", -1.0)
    l15.units += _lt(t, T5, "inPB") + _lt(t, T4, "inPB", -1.0)
    l15.units += pr.hat_units(pr.add(t, neg(T5)), 0.0, {"end2": 1.0})
    # skip the operand's '.': from its offset on, read one byte further
    l15.units += pr.clamp01_units(pr.gated(pr.add(t, {"a_dot": -1.0}, AF, pr.const(2.0)), {"has_a": 1.0}),
                                  {"skipA": 1.0})
    l15.units += pr.clamp01_units(pr.gated(pr.add(u, {"b_dot": -1.0, "o": 1.0}, pr.const(2.0)),
                                           {"has_b": 1.0}), {"skipB": 1.0})

    l16 = b.at(16)
    tzraw = pr.add({"Rb": 1.0, "Lnzr": -1.0}, pr.const(-1.0))
    l16.units += pr.times_flag_units(pr.add(tzraw, {"tzx": -1.0}), {"found_nz": 1.0}, {"tz": 1.0})
    src_a = pr.add(AF, t, {"skipA": 1.0}, pr.const(1.0))
    src_b = pr.add({"o": 1.0, "skipB": 1.0}, u, pr.const(1.0))
    l16.units += pr.times_flag_units(src_a, {"inA": 1.0}, {"src2": 1.0})
    l16.units += pr.times_flag_units({"o": 1.0}, {"atOp": 1.0}, {"src2": 1.0})
    l16.units += pr.times_flag_units(src_b, {"inB": 1.0}, {"src2": 1.0})

    # ---------------------------------------------------------- result formatting
    rs = {"rneg": 1.0, "cancel": -1.0}  # sign byte kept in the rewrite
    m = pr.add({"Rb": 1.0, "S": -1.0, "rneg": -1.0}, pr.const(-1.0))  # result digits
    mp = pr.add(m, {"tz": -1.0})  # digits after stripping trailing zeros

    l17 = b.at(17)
    l17.heads.append(_copy_head("frame copy", {"src2": 1.0}, "Y2"))
    l17.units += pr.times_flag_units({"f_res": 1.0, "tz": -1.0}, {"found_nz": 1.0}, {"fp": 1.0})
    # the prompt line carries "~N" after the expression; the rewrite does not copy it
    l17.units += pr.clamp01_units({"Ptil": 1.0, "Ls": -1.0}, {"til_line": 1.0})

    l18 = b.at(18)
    l18.units.append(pr.relu_unit(LE, {"Lend": 1.0}))  # where the line's suffix copy stops
    l18.units += pr.times_flag_units({"Ptil": 1.0, "F": -1.0, "one": 1.0}, {"til_line": 1.0}, {"Lend": 1.0})
    l18.units += [  # width of the zero-padded digit string: max(m', f'+1)
        pr.relu_unit({"fp": 1.0, "one": 1.0}, {"wd": 1.0}),
        pr.relu_unit(pr.add(mp, {"fp": -1.0}, pr.const(-1.0)), {"wd": 1.0}),
    ]
    l18.units += pr.clamp01_units({"fp": 1.0}, {"dotf": 1.0})

    n1 = pr.add(A, {"Ls": -1.0, "cancel": -1.0}, pr.const(-1.0))
    n2 = pr.add(rs, {"wd": 1.0, "dotf": 1.0})
    n3 = pr.add({"Lend": 1.0}, neg(B), pr.const(-1.0))
    r = {"pos": 1.0, "Rb": -1.0}
    ur = pr.add(r, neg(n1))  # index inside the result segment
    v = pr.add(ur, neg(rs))  # index inside the number (after its sign)
    intlen = {"wd": 1.0, "fp": -1.0}

    l19 = b.at(19)
    l19.units += _lt(r, n1, "in1")
    l19.units += _lt(r, pr.add(n1, n2), "in12")
    l19.units += _lt(r, pr.add(n1, n2, n3), "in123")
    l19.units += pr.hat_units(pr.add(r, neg(pr.add(n1, n2, n3))), 0.0, {"end4": 1.0})
    l19.units += pr.hat_units(pr.gated(ur, rs), 0.0, {"isSign": 1.0})
    l19.units += pr.hat_units(pr.gated(pr.add(v, neg(intlen)), {"dotf": 1.0}), 0.0, {"isDot": 1.0})
    l19.units += pr.clamp01_units(pr.gated(pr.add(v, neg(intlen)), {"dotf": 1.0}), {"afterdot": 1.0})

    k = pr.add(v, {"afterdot": -1.0})  # index into the zero-padded digit string
    in_res = {"in12": 1.0, "in1": -1.0}
    l20 = b.at(20)
    off = pr.add(pr.scale(in_res, pr.BIG), {"isSign": -pr.BIG, "isDot": -pr.BIG}, pr.const(-pr.BIG))
    l20.units += pr.clamp01_units(pr.add({"wd": 1.0}, neg(mp), neg(k), off), {"padzero": 1.0})

    l21 = b.at(21)
    resdig = pr.add(in_res, {"isSign": -1.0, "isDot": -1.0, "padzero": -1.0})  # copied result digit
    src_res = pr.add({"S": 1.0, "rneg": 1.0, "wd": -1.0}, k, mp, pr.const(1.0))
    l21.units += pr.times_flag_units(pr.add({"Ls": 1.0}, pr.const(1.0), r), {"in1": 1.0}, {"src4": 1.0})
    l21.units += pr.times_flag_units(src_res, resdig, {"src4": 1.0})
    l21.units += pr.times_flag_units(pr.add(B, pr.const(1.0), r, neg(n1), neg(n2)),
                                     {"in123": 1.0, "in12": -1.0}, {"src4": 1.0})

    # ---------------------------------------------------------- outputs
    l22 = b.at(22)
    l22.heads.append(_copy_head("rewrite copy", {"src4": 1.0}, "Y4"))
    copy2 = {"inA": 1.0, "atOp": 1.0, "inB": 1.0}
    copy4 = pr.add({"in1": 1.0, "in123": 1.0, "in12": -1.0}, resdig)
    for d in NAMES.values():
        l22.units.append(pr.relu_unit(pr.add({f"Y2_{d}": 1.0, "phase2": 1.0}, copy2, pr.const(-2.0)),
                                      {f"O_{d}": 1.0}))
        l22.units.append(pr.relu_unit(pr.add({f"Y4_{d}": 1.0, "phase4": 1.0}, copy4, pr.const(-2.0)),
                                      {f"O_{d}": 1.0}))
    l22.units += [
        pr.relu_unit({"phase1": 1.0}, {"out_lbr": 1.0}),
        pr.relu_unit({"phase2": 1.0, "inPA": 1.0, "inPB": 1.0, "one": -1.0}, {"O_oh0": 1.0}),  # pad
        pr.relu_unit({"phase2": 1.0, "end2": 1.0, "one": -1.0}, {"out_eq": 1.0}),
        pr.relu_unit({"phase4": 1.0, "isSign": 1.0, "in12": 1.0, "in1": -1.0, "one": -2.0}, {"O_is_minus": 1.0}),
        pr.relu_unit({"phase4": 1.0, "isDot": 1.0, "in12": 1.0, "in1": -1.0, "one": -2.0}, {"O_is_dot": 1.0}),
        pr.relu_unit({"phase4": 1.0, "padzero": 1.0, "one": -1.0}, {"O_oh0": 1.0}),
        pr.relu_unit({"phase4": 1.0, "end4": 1.0, "cont": 1.0, "one": -2.0}, {"out_eq": 1.0}),
        pr.relu_unit({"phase4": 1.0, "end4": 1.0, "cont": -1.0, "one": -1.0}, {"out_nl": 1.0}),
    ]

    for byte, d in NAMES.items():
        b.logit(f"O_{d}", byte, COPY_LOGIT)
    b.logit("out_lbr", cm.FRAME_OPEN, cm.CONTROL_LOGIT)
    b.logit("out_eq", ord("="), cm.CONTROL_LOGIT)
    b.logit("out_nl", cm.EOS, cm.CONTROL_LOGIT)
