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
    "c_raw", "fc", "om", "fm", "oa", "fa", "c", "use_mul", "opw", "pw", "o",
    "a_lo", "sb_x", "LopLe", "LopO", "op_div", "op_pow", "op_md", "e3", "cont_raw", "sa_x", "b_hi", "a_dot", "e1", "e2", "cont",
    "has_a", "res_op", "res_wrap", "rneg", "b_dot", "res_expr", "drop", "frac_a", "has_b", "prevA_unary", "Lnzr", "cancel", "cont2", "frac_b",
    "found_nz", "pad_a_raw", "pad_b_raw", "pad_a", "pad_b", "f_res", "tzx",
    "inA", "inPA", "atOp", "inB", "inPB", "end2", "skipA", "skipB",
    "tz", "src2", *(f"Y2_{d}" for d in NAMES.values()), "fp", "wd", "dotf",
    "in1", "in12", "in123", "end4", "isSign", "isDot", "afterdot", "padzero", "src4",
    *(f"Y4_{d}" for d in NAMES.values()), "Y4_is_space",
)
OUT_DIMS = (*(f"O_{d}" for d in NAMES.values()), "out_lbr", "out_eq", "out_nl")

LE: Form = {"F": 1.0, "one": -1.0}
AF: Form = {"a_lo": 1.0, "sa_x": -1.0}  # byte before the signed operand a
ND: Form = {"one": 1.0, "is_digit": -1.0, "is_dot": -1.0}  # not part of a number
BSTART: Form = {"o": 1.0, "sb_x": 1.0}  # byte before b's digits (o, or b's sign)


def _copy_head(name: str, src: Form, prefix: str, space: bool = False) -> HeadSpec:
    """Gather at position ``src``: ``<prefix>_<d>`` = [byte at src is d] for every alphabet byte d
    (and the space, for the rewrite: a space frame copies the line's other spaces, strip.py)."""
    value = {d: {f"{prefix}_{d}": 1.0} for d in NAMES.values()}
    if space:
        value["is_space"] = {f"{prefix}_is_space": 1.0}
    return HeadSpec(
        name=name,
        query=[pr.scale(src, 2 * pr.GATHER_BETA), pr.const(-pr.GATHER_BETA)],
        key=[{"pos": 1.0}, {"pos2": 1.0}],
        value=value,
    )


def _lt(x: Form, y: Form, dst: str, sign: float = 1.0) -> list:
    """``dst += sign * [x < y]`` for integers."""
    return pr.clamp01_units(pr.add(y, pr.scale(x, -1.0)), {dst: sign})


def build(b: ModelBuilder) -> None:
    """Write the parser and rewriter into layers 4-22: choose the redex (p, c, o, a, b), its
    decimal pads and result format, then the frame copy (layer 17) and line rewrite (layer 22)."""
    neg = lambda f: pr.scale(f, -1.0)  # noqa: E731

    # ---------------------------------------------------------- redex search
    # Layer 4 heads (frame's PM_c, p are from layer 3): PM2_c = PM_c^2 (square table: pos2 at
    # PM_c) and Lt_c = latest c-match at or before p, the key and query terms of layer 5.
    l4 = b.at(4)
    for c in ("Lrp", "Lmul", "Ladd", "Lnd"):
        l4.heads.append(pr.gather_head(f"pm^2 {c}", {f"PM_{c}": 1.0}, f"PM2_{c}", src="pos2"))
    for c in ("Lrp", "Lmul", "Ladd"):
        l4.heads.append(pr.gather_head(f"latest {c} at p", {"p": 1.0}, f"Lt_{c}", src=c))

    # Layer 5. Heads: the first ')', '*'/'/', '+'/binary '-' after p (next_match: the one matching
    # key whose PM_c == Lt_c), each with a found flag (fc, fm, fa; ~0 if none). MLP: c.
    l5 = b.at(5)
    l5.heads += [
        pr.next_match_head("first ) after p", "PM_Lrp", "PM2_Lrp", {"is_rpar": 1.0},
                           {"Lt_Lrp": 1.0}, "c_raw", "fc"),
        pr.next_match_head("first * or / after p", "PM_Lmul", "PM2_Lmul", {"is_star": 1.0, "is_slash": 1.0},
                           {"Lt_Lmul": 1.0}, "om", "fm"),
        pr.next_match_head("first +- after p", "PM_Ladd", "PM2_Ladd",
                           {"is_plus": 1.0, "is_binminus": 1.0}, {"Lt_Ladd": 1.0}, "oa", "fa"),
    ]
    # c = min(first ')' after p, Le): a ')' beyond the line (e.g. inside a frame body that
    # wrote "(a*a)^n") must not end the group. fc * ReLU(x) is one ReLU silenced by BIG.
    # c = ReLU(Le) - ReLU(Le - c_raw - BIG(1 - fc)); Le > 0, so the first is just Le.
    # Read by layer 6 (opw at c-1, use_mul) and layer 9 (e2 = [b_hi == c]).
    l5.units.append(pr.relu_unit(LE, {"c": 1.0},
                                 label="'c' base: Le (the line's '=' minus 1); the paired unit subtracts the excess when a ')' before Le is found"))
    l5.units.append(pr.relu_unit(pr.add(LE, {"c_raw": -1.0, "fc": pr.BIG}, pr.const(-pr.BIG)), {"c": -1.0},
                                 label="'c' correction: minus ReLU(Le - c_raw) when a ')' was found (fc), giving c = min(c_raw, Le); silenced by BIG otherwise"))

    # Layer 6. Head: opw = latest '^' before c. MLP: use_mul = [fm AND om < c], i.e.
    # [c - om + BIG(fm - 1) >= 1] (no '*'/'/' found: -BIG keeps it 0); pw = [opw > p].
    l6 = b.at(6)
    l6.heads.append(pr.gather_head("latest ^ before c", pr.add({"c": 1.0}, pr.const(-1.0)), "opw", src="Lpow"))
    l6.units += pr.clamp01_units(pr.add({"c": 1.0, "om": -1.0}, pr.scale({"fm": 1.0}, pr.BIG),
                                        pr.const(-pr.BIG)), {"use_mul": 1.0})
    l6.units += pr.clamp01_units({"opw": 1.0, "p": -1.0}, {"pw": 1.0})  # a '^' inside the group

    # o: latest '^' in the group (right-associative), else first * or /, else first + or -.
    # Positions are >= 0, so each choice is one ReLU silenced by BIG unless selected.
    # Layer 7 MLP: o, the redex operator (exactly one choice passes); read from layer 8 on.
    l7 = b.at(7)
    l7.units += [
        pr.relu_unit({"oa": 1.0, "use_mul": -pr.BIG, "pw": -pr.BIG}, {"o": 1.0},
                     label="'o' choice 3: first '+'/binary '-' after p (oa), used only when no '^' in group and no '*'/'/' before c"),
        pr.relu_unit({"om": 1.0, "use_mul": pr.BIG, "pw": -pr.BIG, "one": -pr.BIG}, {"o": 1.0},
                     label="'o' choice 2: first '*'/'/' after p (om), used when it lies before c (use_mul) and the group has no '^'"),
        pr.relu_unit({"opw": 1.0, "pw": pr.BIG, "one": -pr.BIG}, {"o": 1.0},
                     label="'o' choice 1: latest '^' before c (opw, right-associative), used when it lies inside the group (pw)"),
    ]

    # ---------------------------------------------------------- operands
    # Layer 8. Heads read around o: a's fence, b's sign, the operator type, and the latest binary
    # operators for cont. MLP: e3 and cont_raw.
    l8 = b.at(8)
    l8.heads += [
        # a_lo = latest non-number byte before o: the byte before a, or a's own '-'
        pr.gather_head("a fence", pr.add({"o": 1.0}, pr.const(-1.0)), "a_lo", src="Lnd"),
        # sb_x = [byte after o is '-']: b is signed
        pr.gather_head("sign of b?", pr.add({"o": 1.0}, pr.const(1.0)), "sb_x", src="is_minus"),
        # LopLe, LopO = latest binary operator in the line / before o (cont, below)
        pr.gather_head("latest op at Le", LE, "LopLe", src="P"),
        pr.gather_head("latest op before o", pr.add({"o": 1.0}, pr.const(-1.0)), "LopO", src="P"),
        # op_div, op_pow, op_md = [byte at o is '/'], ['^'], ['*' or '/']
        pr.gather_head("redex is a division?", {"o": 1.0}, "op_div", src="is_slash"),
        pr.gather_head("redex is a power?", {"o": 1.0}, "op_pow", src="is_caret"),
        HeadSpec(name="redex is * or /?",
                 query=[pr.scale({"o": 1.0}, 2 * pr.GATHER_BETA), pr.const(-pr.GATHER_BETA)],
                 key=[{"pos": 1.0}, {"pos2": 1.0}],
                 value={"is_star": {"op_md": 1.0}, "is_slash": {"op_md": 1.0}}),
    ]
    # e3 = [p > Ls]: the group is a real '(' rather than the line start
    l8.units += pr.clamp01_units({"p": 1.0, "Ls": -1.0}, {"e3": 1.0})
    # cont_raw = [LopLe > o] + [LopO > Ls]: another operator after or before o in the line
    # (the sum can be 2; layer 9 clamps it to the 0/1 cont)
    l8.units += pr.clamp01_units({"LopLe": 1.0, "o": -1.0}, {"cont_raw": 1.0})
    l8.units += pr.clamp01_units({"LopO": 1.0, "Ls": -1.0}, {"cont_raw": 1.0})

    # Layer 9. Heads: a's sign, b's end, a's '.', and two facts about the frame's result.
    # MLP: e1, e2 (the redex fills the group), cont, has_a.
    l9 = b.at(9)
    l9.heads += [
        # sa_x = [byte at a_lo is a unary '-']: a is signed, its fence is AF = a_lo - sa_x
        pr.gather_head("sign of a?", {"a_lo": 1.0}, "sa_x", src="is_unary"),
        # b_hi = first non-number byte after BSTART = o + sb_x. BSTART is itself a non-number
        # (o, or b's '-'), so it is the "latest match at or before t" next_match needs.
        pr.next_match_head("b fence", "PM_Lnd", "PM2_Lnd", ND, BSTART, "b_hi"),
        # a_dot = latest '.' before o; it is a's '.' iff a_dot > a_lo
        pr.gather_head("a's '.'", pr.add({"o": 1.0}, pr.const(-1.0)), "a_dot", src="Ldot"),
    ]
    # e1 = [AF == p]: signed a starts right after the group start; e2 = [b_hi == c]: b ends at it
    l9.units += pr.hat_units(pr.add(AF, {"p": -1.0}), 0.0, {"e1": 1.0})
    l9.units += pr.hat_units({"b_hi": 1.0, "c": -1.0}, 0.0, {"e2": 1.0})
    # Read at Rb-1, the frame result's last byte (used by the rewrite after ']'):
    # res_op = latest binary operator up to there (in the result iff > S), res_wrap = [it is ')'].
    l9.heads += [
        pr.gather_head("operator inside the result?", pr.add({"Rb": 1.0}, pr.const(-1.0)), "res_op", src="P"),
        pr.gather_head("result ends in ')'?", pr.add({"Rb": 1.0}, pr.const(-1.0)), "res_wrap", src="is_rpar"),
    ]
    l9.units += pr.clamp01_units({"cont_raw": 1.0}, {"cont": 1.0})
    # has_a = [a_dot - a_lo >= 1]; -BIG*op_pow forces 0 for '^', whose frame copies the base
    # verbatim, '.' included (no decimal normalisation for powers)
    l9.units += pr.clamp01_units({"a_dot": 1.0, "a_lo": -1.0, "op_pow": -pr.BIG}, {"has_a": 1.0})  # '.' in a

    # Layer 10. Heads: rneg = [byte after S is '-'] (the frame's result is negative); b_dot =
    # latest '.' before b_hi, b's '.' iff b_dot > BSTART. MLP: res_expr, frac_a.
    l10 = b.at(10)
    l10.heads += [
        pr.gather_head("result negative?", pr.add({"S": 1.0}, pr.const(1.0)), "rneg", src="is_minus"),
        pr.gather_head("b's '.'", pr.add({"b_hi": 1.0}, pr.const(-1.0)), "b_dot", src="Ldot"),
    ]
    # res_expr = [res_op > S]: the result holds an operator, so it is an expression (power frames)
    l10.units += pr.clamp01_units({"res_op": 1.0, "S": -1.0}, {"res_expr": 1.0})
    # (drop also needs res_expr; both are only read by the rewrite, from layer 11 on)
    # frac_a = has_a * (o - 1 - a_dot): digits after a's '.', 0 without one
    l10.units += pr.times_flag_units(pr.add({"o": 1.0, "a_dot": -1.0}, pr.const(-1.0)), {"has_a": 1.0},
                                     {"frac_a": 1.0})

    # The group's parentheses go when the redex is the whole group and the result is a plain
    # number, or is itself fully parenthesised (ends in ')': power expansions "(a*a^n)", "(1/a^n)").
    # Layer-11 MLP: drop = ReLU(e1+e2+e3 - res_expr + res_wrap - 2). (res_expr, res_wrap) is (0,0)
    # for a number, (1,1) for a ')'-ended expression: 1 iff e1=e2=e3=1. (1,0) keeps it <= 0.
    b.at(11).units.append(pr.relu_unit({"e1": 1.0, "e2": 1.0, "e3": 1.0, "res_expr": -1.0, "res_wrap": 1.0,
                                        "one": -2.0}, {"drop": 1.0},
                                       label="'drop': redex is the whole group (e1,e2,e3) AND result is a plain number or ends in ')', so the parentheses go too"))
    # A..B = the bytes the result replaces: AF+1..b_hi-1, or p..c on drop (then AF = p, b_hi = c)
    A = pr.add(AF, {"drop": -1.0}, pr.const(1.0))
    B = pr.add({"b_hi": 1.0, "drop": 1.0}, pr.const(-1.0))

    # Double negation: a negative result replacing bytes right after a unary '-'
    # ("-(1-4)", "2*-(1-4)") drops both signs.
    # Layer 11 (also holds drop). Head: Lnzr = latest nonzero digit up to Rb-1 (end of result).
    # MLP: has_b = [b_dot > BSTART] (forced 0 for '^', as has_a); found_nz = [Lnzr > S + rneg]:
    # a nonzero digit lies after the sign, so the result is nonzero.
    l11 = b.at(11)
    l11.heads.append(
        pr.gather_head("result's last nonzero digit", pr.add({"Rb": 1.0}, pr.const(-1.0)), "Lnzr", src="Lnz"))
    l11.units += pr.clamp01_units(pr.add({"b_dot": 1.0, "op_pow": -pr.BIG}, neg(BSTART)), {"has_b": 1.0})
    l11.units += pr.clamp01_units({"Lnzr": 1.0, "S": -1.0, "rneg": -1.0}, {"found_nz": 1.0})

    # Layer 12. Head: prevA_unary = [byte at A-1 is a unary '-']. MLP: cancel = prevA_unary AND
    # rneg, cont2, frac_b = has_b * (b_hi - 1 - b_dot).
    l12 = b.at(12)
    l12.heads.append(pr.gather_head("unary '-' before A?", pr.add(A, pr.const(-1.0)), "prevA_unary", src="is_unary"))
    l12.units.append(pr.relu_unit({"prevA_unary": 1.0, "rneg": 1.0, "one": -1.0}, {"cancel": 1.0},
                                  label="'cancel': negative result (rneg) replacing bytes right after a unary '-': AND, both signs are dropped"))
    # keep going if another operator remains, or the frame's result is itself an expression
    l12.units += pr.clamp01_units({"cont": 1.0, "res_expr": 1.0}, {"cont2": 1.0})
    l12.units += pr.times_flag_units(pr.add({"b_hi": 1.0, "b_dot": -1.0}, pr.const(-1.0)), {"has_b": 1.0},
                                     {"frac_b": 1.0})

    # ---------------------------------------------------------- decimal geometry
    # Layer 13 MLP: raw alignment pads, the zeros that give a and b equal fraction lengths.
    l13 = b.at(13)
    l13.units += [
        pr.relu_unit({"frac_b": 1.0, "frac_a": -1.0}, {"pad_a_raw": 1.0},
                     label="'pad_a_raw' = ReLU(frac_b - frac_a): zeros to append to a so its fraction lines up with b's (pair of pad_b_raw)"),
        pr.relu_unit({"frac_a": 1.0, "frac_b": -1.0}, {"pad_b_raw": 1.0},
                     label="'pad_b_raw' = ReLU(frac_a - frac_b): zeros to append to b so its fraction lines up with a's (pair of pad_a_raw)"),
    ]

    # Layer 14 MLP: per-operator pads and f_res (fractional digits of the result).
    #   + -: pads = raw pads, f_res = max(frac_a, frac_b)    *: no pads, f_res = frac_a + frac_b
    #   /: pad_a = frac_b + N, pad_b = frac_a, f_res = N     ^: has_a = has_b = 0, so all are 0
    l14 = b.at(14)
    l14.units += [
        pr.relu_unit(pr.gated({"pad_a_raw": 1.0}, {"op_md": 1.0}, 0.0), {"pad_a": 1.0},
                     label="'pad_a' for + and -: alignment zeros pad_a_raw, silenced when the redex is * or / (op_md)"),
        pr.relu_unit(pr.gated({"pad_b_raw": 1.0}, {"op_md": 1.0}, 0.0), {"pad_b": 1.0},
                     label="'pad_b' for + and -: alignment zeros pad_b_raw, silenced when the redex is * or / (op_md)"),
        pr.relu_unit({"frac_a": 1.0, "pad_a_raw": 1.0}, {"f_res": 1.0},
                     label="'f_res' base: frac_a + ReLU(frac_b - frac_a) = max(frac_a, frac_b), the result's fraction digits for + and -"),  # max(frac_a, frac_b)
    ]
    # * and /: f_res += frac_b - pad_a_raw, turning max(frac_a, frac_b) into frac_a + frac_b
    l14.units += pr.times_flag_units({"frac_b": 1.0, "pad_a_raw": -1.0}, {"op_md": 1.0}, {"f_res": 1.0})
    # a/b: the frame holds a*10^(frac_b+N) / b*10^frac_a, so the integer quotient has N fractional digits
    l14.units += pr.times_flag_units({"frac_b": 1.0, "N": 1.0}, {"op_div": 1.0}, {"pad_a": 1.0})
    l14.units += pr.times_flag_units({"frac_a": 1.0}, {"op_div": 1.0}, {"pad_b": 1.0})
    # /: f_res += N - frac_a - frac_b, so f_res = N
    l14.units += pr.times_flag_units({"N": 1.0, "frac_a": -1.0, "frac_b": -1.0}, {"op_div": 1.0}, {"f_res": 1.0})

    # frame copy segments, t = q - F
    # t = bytes of the frame copied so far (the query at '[' predicts the first). Segments by t:
    # [0,T1) signed a without '.', [T1,T2) pad_a zeros, T2 the operator, [T3,T4) signed b
    # without '.', [T4,T5) pad_b zeros, then '=' at t = T5.
    t = {"pos": 1.0, "F": -1.0}
    la_ = pr.add({"o": 1.0, "has_a": -1.0}, neg(AF), pr.const(-1.0))  # signed a without '.'
    # lb_ = b_hi - o - 1 - has_b: signed b without '.'
    lb_ = pr.add({"b_hi": 1.0, "o": -1.0, "has_b": -1.0}, pr.const(-1.0))
    T1 = la_
    T2 = pr.add(T1, {"pad_a": 1.0})
    T3 = pr.add(T2, pr.const(1.0))
    T4 = pr.add(T3, lb_)
    T5 = pr.add(T4, {"pad_b": 1.0})
    u = pr.add(t, neg(T3))  # index inside b's segment

    # Layer 15 MLP: tzx; segment flags as differences of robust steps [t < T_j]
    # (e.g. inPA = [t < T2] - [t < T1]); end2 = [t == T5]; the '.' skips.
    l15 = b.at(15)
    l15.units.append(pr.relu_unit({"Rb": 1.0, "Lnzr": -1.0, "f_res": -1.0, "one": -1.0},
                                  {"tzx": 1.0},
                                  label="'tzx': trailing zeros of the result beyond f_res; subtracted later so only fractional zeros are stripped (tz)"))  # ReLU(trailing zeros - f_res)
    l15.units += _lt(t, T1, "inA")
    l15.units += _lt(t, T2, "inPA") + _lt(t, T1, "inPA", -1.0)
    l15.units += _lt(t, T3, "atOp") + _lt(t, T2, "atOp", -1.0)
    l15.units += _lt(t, T4, "inB") + _lt(t, T3, "inB", -1.0)
    l15.units += _lt(t, T5, "inPB") + _lt(t, T4, "inPB", -1.0)
    l15.units += pr.hat_units(pr.add(t, neg(T5)), 0.0, {"end2": 1.0})
    # skip the operand's '.': from its offset on, read one byte further
    # skipA = has_a AND [AF+1+t >= a_dot]; skipB = has_b AND [o+1+u >= b_dot]
    # (gated adds -BIG(1 - has_x), so no '.' means no skip)
    l15.units += pr.clamp01_units(pr.gated(pr.add(t, {"a_dot": -1.0}, AF, pr.const(2.0)), {"has_a": 1.0}),
                                  {"skipA": 1.0})
    l15.units += pr.clamp01_units(pr.gated(pr.add(u, {"b_dot": -1.0, "o": 1.0}, pr.const(2.0)),
                                           {"has_b": 1.0}), {"skipB": 1.0})

    # Layer 16 MLP: tz and src2, the frame copy's source position (read by the layer-17 head).
    l16 = b.at(16)
    # tzraw = Rb - 1 - Lnzr: the result's trailing zeros. tz = found_nz * (tzraw - tzx)
    # = min(tzraw, f_res): only fractional zeros are stripped; 0 for a zero result.
    tzraw = pr.add({"Rb": 1.0, "Lnzr": -1.0}, pr.const(-1.0))
    l16.units += pr.times_flag_units(pr.add(tzraw, {"tzx": -1.0}), {"found_nz": 1.0}, {"tz": 1.0})
    # src2 = inA*(AF+1+t+skipA) + atOp*o + inB*(o+1+u+skipB); 0 in the pads, whose zeros are
    # written directly in layer 22
    src_a = pr.add(AF, t, {"skipA": 1.0}, pr.const(1.0))
    src_b = pr.add({"o": 1.0, "skipB": 1.0}, u, pr.const(1.0))
    l16.units += pr.times_flag_units(src_a, {"inA": 1.0}, {"src2": 1.0})
    l16.units += pr.times_flag_units({"o": 1.0}, {"atOp": 1.0}, {"src2": 1.0})
    l16.units += pr.times_flag_units(src_b, {"inB": 1.0}, {"src2": 1.0})

    # ---------------------------------------------------------- result formatting
    # The result S+1..Rb-1 is an optional '-' then m digits; the rewrite writes the sign (rs),
    # wd = max(m', f'+1) digits left-padded with zeros, with '.' before the last f'.
    rs = {"rneg": 1.0, "cancel": -1.0}  # sign byte kept in the rewrite
    m = pr.add({"Rb": 1.0, "S": -1.0, "rneg": -1.0}, pr.const(-1.0))  # result digits
    mp = pr.add(m, {"tz": -1.0})  # digits after stripping trailing zeros

    # Layer 17. Head: frame copy, Y2_d = [byte at src2 is d]. MLP: fp.
    l17 = b.at(17)
    l17.heads.append(_copy_head("frame copy", {"src2": 1.0}, "Y2"))
    # fp = f' = found_nz * (f_res - tz): fraction digits left after stripping; 0 for a zero result
    l17.units += pr.times_flag_units({"f_res": 1.0, "tz": -1.0}, {"found_nz": 1.0}, {"fp": 1.0})

    # Layer 18 MLP: the result width wd and dotf = [f' >= 1] (the result has a '.').
    # (The suffix copy stops at Lend, frame layer 4: Le, or the '~' on the prompt line.)
    l18 = b.at(18)
    l18.units += [  # width of the zero-padded digit string: max(m', f'+1)
        pr.relu_unit({"fp": 1.0, "one": 1.0}, {"wd": 1.0},
                     label="'wd' base: f'+1 (fraction digits plus one leading 0); with the paired unit wd = max(m', f'+1)"),
        pr.relu_unit(pr.add(mp, {"fp": -1.0}, pr.const(-1.0)), {"wd": 1.0},
                     label="'wd' excess: ReLU(m' - f' - 1), the result digits beyond f'+1 after stripping trailing zeros (pair of wd base)"),
    ]
    l18.units += pr.clamp01_units({"fp": 1.0}, {"dotf": 1.0})

    # Rewrite regions, indexed by r = pos - Rb (bytes of the new line written so far):
    # prefix n1 = |Ls+1..A-1| less the cancelled '-' at A-1, result n2 = sign + wd digits + '.',
    # suffix n3 = |B+1..Lend-1|.
    n1 = pr.add(A, {"Ls": -1.0, "cancel": -1.0}, pr.const(-1.0))
    n2 = pr.add(rs, {"wd": 1.0, "dotf": 1.0})
    n3 = pr.add({"Lend": 1.0}, neg(B), pr.const(-1.0))
    r = {"pos": 1.0, "Rb": -1.0}
    ur = pr.add(r, neg(n1))  # index inside the result segment
    v = pr.add(ur, neg(rs))  # index inside the number (after its sign)
    # intlen = wd - f': digits before the '.'
    intlen = {"wd": 1.0, "fp": -1.0}

    # Layer 19 MLP: cumulative region flags in1 = [r < n1], in12 = [r < n1+n2], in123, end4 =
    # [r == n1+n2+n3]; in the result: isSign = [ur == 0] if rs, isDot = [v == intlen] if dotf,
    # afterdot = [v > intlen] if dotf (gated: -BIG unless the flag is 1).
    l19 = b.at(19)
    l19.units += _lt(r, n1, "in1")
    l19.units += _lt(r, pr.add(n1, n2), "in12")
    l19.units += _lt(r, pr.add(n1, n2, n3), "in123")
    l19.units += pr.hat_units(pr.add(r, neg(pr.add(n1, n2, n3))), 0.0, {"end4": 1.0})
    l19.units += pr.hat_units(pr.gated(ur, rs), 0.0, {"isSign": 1.0})
    l19.units += pr.hat_units(pr.gated(pr.add(v, neg(intlen)), {"dotf": 1.0}), 0.0, {"isDot": 1.0})
    l19.units += pr.clamp01_units(pr.gated(pr.add(v, neg(intlen)), {"dotf": 1.0}), {"afterdot": 1.0})

    k = pr.add(v, {"afterdot": -1.0})  # index into the zero-padded digit string
    # in_res = in12 - in1: inside the result segment
    in_res = {"in12": 1.0, "in1": -1.0}
    # Layer 20 MLP: padzero = [k < wd - mp] on a digit slot of the result: a left-pad zero.
    # off = -BIG unless in_res = 1 and isSign = isDot = 0.
    l20 = b.at(20)
    off = pr.add(pr.scale(in_res, pr.BIG), {"isSign": -pr.BIG, "isDot": -pr.BIG}, pr.const(-pr.BIG))
    l20.units += pr.clamp01_units(pr.add({"wd": 1.0}, neg(mp), neg(k), off), {"padzero": 1.0})

    # Layer 21 MLP: src4, the rewrite's source position (read by the layer-22 head):
    # prefix Ls+1+r; result digit S+rneg+1 + (k - (wd - mp)); suffix B+1 + (r - n1 - n2).
    l21 = b.at(21)
    resdig = pr.add(in_res, {"isSign": -1.0, "isDot": -1.0, "padzero": -1.0})  # copied result digit
    src_res = pr.add({"S": 1.0, "rneg": 1.0, "wd": -1.0}, k, mp, pr.const(1.0))
    # In a strip frame (sp or pf = 1) the region flags mean nothing: each select's flag is less
    # sp and pf, so it is off there, and strip.py adds its own source position st_src instead.
    strip = {"sp": -1.0, "pf": -1.0}
    l21.units += pr.times_flag_units(pr.add({"Ls": 1.0}, pr.const(1.0), r), pr.add({"in1": 1.0}, strip), {"src4": 1.0})
    l21.units += pr.times_flag_units(src_res, pr.add(resdig, strip), {"src4": 1.0})
    l21.units += pr.times_flag_units(pr.add(B, pr.const(1.0), r, neg(n1), neg(n2)),
                                     pr.add({"in123": 1.0, "in12": -1.0}, strip), {"src4": 1.0})

    # ---------------------------------------------------------- outputs
    # Layer 22. Head: rewrite copy, Y4_d = [byte at src4 is d]. MLP: the byte votes, each an AND
    # ReLU(byte + phase + region - 2), or a fixed byte ('[', '0', '-', '.', '=', newline) per phase.
    l22 = b.at(22)
    l22.heads.append(_copy_head("rewrite copy", {"src4": 1.0}, "Y4", space=True))
    # copy regions (disjoint flags, so each sum is 0/1): frame copy a, operator, b;
    # rewrite prefix, suffix (in123 - in12), or a copied result digit
    copy2 = {"inA": 1.0, "atOp": 1.0, "inB": 1.0}
    copy4 = pr.add({"in1": 1.0, "in123": 1.0, "in12": -1.0}, resdig)
    for d in NAMES.values():
        l22.units.append(pr.relu_unit(pr.add({f"Y2_{d}": 1.0, "phase2": 1.0}, copy2, pr.const(-2.0)),
                                      {f"O_{d}": 1.0},
                                      label=f"emit byte {d} in phase2 frame copy: AND of copied byte Y2_{d}, phase2, and inside a/op/b segment"))
        l22.units.append(pr.relu_unit(pr.add({f"Y4_{d}": 1.0, "phase4": 1.0}, copy4, pr.const(-2.0)),
                                      {f"O_{d}": 1.0},
                                      label=f"emit byte {d} in phase4 rewrite: AND of copied byte Y4_{d}, phase4, and prefix/suffix/result-digit region"))
    l22.units += [
        pr.relu_unit({"phase1": 1.0}, {"out_lbr": 1.0},
                     label="emit '[' in phase1 (line-ending '=' with no frame open): open a new frame"),
        pr.relu_unit({"phase2": 1.0, "inPA": 1.0, "inPB": 1.0, "one": -1.0}, {"O_oh0": 1.0},
                     label="emit '0' in phase2 inside a zero-pad segment (inPA after a, inPB after b) to align fractions"),  # pad
        pr.relu_unit({"phase2": 1.0, "end2": 1.0, "one": -1.0}, {"out_eq": 1.0},
                     label="emit '=' in phase2 at the end of the frame copy (end2), handing over to the operator circuit"),
        pr.relu_unit({"phase4": 1.0, "isSign": 1.0, "in12": 1.0, "in1": -1.0, "one": -2.0}, {"O_is_minus": 1.0},
                     label="emit '-' in phase4 at the result's sign position (isSign), inside the result segment (in12 - in1)"),
        pr.relu_unit({"phase4": 1.0, "isDot": 1.0, "in12": 1.0, "in1": -1.0, "one": -2.0}, {"O_is_dot": 1.0},
                     label="emit '.' in phase4 at the result's decimal point position (isDot), inside the result segment (in12 - in1)"),
        pr.relu_unit({"phase4": 1.0, "padzero": 1.0, "one": -1.0}, {"O_oh0": 1.0},
                     label="emit '0' in phase4 for a left-pad zero of a result shorter than its fraction (2 with f'=2 -> 0.02)"),
        pr.relu_unit({"phase4": 1.0, "end4": 1.0, "cont2": 1.0, "one": -2.0}, {"out_eq": 1.0},
                     label="emit '=' at the end of the rewritten line (end4) when work remains (cont2): another reduction follows"),
        pr.relu_unit({"phase4": 1.0, "end4": 1.0, "cont2": -1.0, "one": -1.0}, {"out_nl": 1.0},
                     label="emit newline at the end of the rewritten line (end4) when no work remains (not cont2): the answer is final"),
    ]

    # Copied bytes vote with COPY_LOGIT (= DIGIT_LOGIT 10); '[', '=' and newline with
    # CONTROL_LOGIT 20, which wins if a byte vote fires at the same position (common.py).
    for byte, d in NAMES.items():
        b.logit(f"O_{d}", byte, COPY_LOGIT)
    b.logit("out_lbr", cm.FRAME_OPEN, cm.CONTROL_LOGIT)
    b.logit("out_eq", ord("="), cm.CONTROL_LOGIT)
    b.logit("out_nl", cm.EOS, cm.CONTROL_LOGIT)
