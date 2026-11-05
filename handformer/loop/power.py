"""Powers: the copy replaces ``a^b`` by one binary-exponentiation step, as text (LOOPED.md §4.5).

The frame-writing calculator's rewrite (tests/reference.py ``power_expansion``); the products
it writes are reduced by later steps:

  T0  b < 0            (1/a^|b|)
  T1  b = 0            1
  T2  b is "1"         a
  T3  b odd            (a*a^b')       b' = b with its last digit lowered by 1
  T4  b even           (a*a)^h        h = b/2, digit by digit: 5*[previous digit odd] + d//2

Every template is one layout of eleven segments, most of them empty for a given template:

  0 '('  1 '1'  2 '/'  3 a  4 '*'  5 a  6 ')'  7 '^'  8 b (without its sign or last digit)
  9 computed digits (b's last digit - 1, or the halves)  10 ')'

The root works out the template, each segment's start ``C_S<s>`` and where each copied segment
reads (``C_o<s>``: byte k of the replacement copies old position ``C_o<s> + k``). The walk's
replacement byte k (walk.py) then falls in one segment: a fixed byte, a copied byte, or a
computed digit.
"""

from __future__ import annotations

from .. import primitives as pr
from . import parts as P

SEG = 11
LIT = {0: "(", 1: "1", 2: "/", 4: "*", 6: ")", 7: "^", 10: ")"}  # fixed-byte segments

CTX = ("C_T3", "C_T4", *(f"C_S{s}" for s in range(1, SEG)), "C_o3", "C_o5", "C_o8", "C_o9", "C_mp")

DIMS = (
    "blast", "bodd", "bfirst", "lb1", "is1", "b01", "lb2", "T0", "T1", "b1", "skip", "T2", "T3", "T4", "pc",
    *(f"Ln{s}" for s in range(SEG - 1)), "mp",
    *(f"sg{s}" for s in range(1, SEG)), *(f"P9_{d}" for d in range(1, 10)), "P9odd",
    *CTX,
)


def build(b) -> None:
    one = pr.const
    L = b.at
    neg = lambda f: pr.scale(f, -1.0)  # noqa: E731
    la_t = {"o": 1.0, "a_lo": -1.0, "sa_x": 1.0, "one": -1.0}  # a's text, its sign included
    lb_t = {"b_hi": 1.0, "o": -1.0, "one": -1.0}  # b's text, its sign included
    last = pr.add({"b_hi": 1.0}, one(-1.0))

    # ------------------------------------------------------------ at the root: the template
    odd = {f"oh{d}": {"bodd": 1.0} for d in (1, 3, 5, 7, 9)}
    L(12).heads += [P.gather("b's last digit", last, {"val": {"blast": 1.0}, **odd}),
                    P.gather("b's first digit", pr.add({"o": 1.0}, one(1.0)), {"val": {"bfirst": 1.0}})]
    L(12).units += P.hat(pr.add(lb_t, one(-1.0)), "lb1", "'lb1': b is one byte")
    L(12).units += P.hat(pr.add({"blast": 1.0}, one(-1.0)), "is1", "'is1': b's last digit is 1")
    L(12).units += P.flag(pr.add({"bfirst": -1.0}, one(2.0)), "b01", "'b01': b's first digit is 0 or 1")
    L(12).units += P.flag(pr.add(lb_t, one(-1.0)), "lb2", "'lb2': b has two bytes or more")
    L(13).units += P.flag(P.all_of("opc", "sb_x"), "T0", "'T0': a^-n -> (1/a^n)")
    L(13).units += P.flag(pr.add({"opc": 1.0, "sb_x": -1.0, "bz": 1.0}, one(-1.0)), "T1", "'T1': a^0 -> 1")
    L(13).units += P.flag(P.all_of("lb1", "is1"), "b1", "'b1': b is the text '1'")
    L(13).units += P.flag(P.all_of("b01", "lb2"), "skip", "'skip': halving b gives a leading 0, dropped")
    L(14).units += P.flag(P.all_of("opc", "b1"), "T2", "'T2': a^1 -> a")
    L(14).units += P.flag({"opc": 1.0, "sb_x": -1.0, "bz": -1.0, "bodd": 1.0, "b1": -1.0, "one": -1.0}, "T3",
                          "'T3': b odd -> (a*a^(b-1))")
    L(14).units += P.flag({"opc": 1.0, "sb_x": -1.0, "bz": -1.0, "bodd": -1.0}, "T4", "'T4': b even -> (a*a)^(b/2)")

    # segment lengths (a template flag times a text length) and the replacement's length
    lits = {0: ("T0", "T3", "T4"), 1: ("T0", "T1"), 2: ("T0",), 4: ("T3", "T4"), 6: ("T4",), 7: ("T0", "T3", "T4")}
    l15 = L(15)
    for s, ts in lits.items():
        l15.units += [P.relu({t: 1.0}, f"Ln{s}", f"'Ln{s}': segment {s} ('{LIT[s]}') in template {t}") for t in ts]
        l15.units += [P.relu({t: 1.0}, "mp", f"'mp': segment {s} in template {t}") for t in ts]
    l15.units += [P.relu({"T0": 1.0}, "mp", "'mp': the closing ')' of template T0"),
                  P.relu({"T3": 1.0}, "mp", "'mp': the closing ')' of template T3")]
    for s, src, ts in ((3, la_t, ("T0", "T2", "T3", "T4")), (5, la_t, ("T3", "T4")),
                       (8, pr.add(lb_t, one(-1.0)), ("T0", "T3"))):
        l15.units += P.sel(src, pr.add(*({t: 1.0} for t in ts)), f"Ln{s}")
        l15.units += P.sel(src, pr.add(*({t: 1.0} for t in ts)), "mp")
    l15.units.append(P.relu({"T3": 1.0}, "Ln9", "'Ln9': b's last digit, lowered (T3)"))
    l15.units.append(P.relu({"T3": 1.0}, "mp", "'mp': b's last digit, lowered (T3)"))
    l15.units += P.sel(pr.add(lb_t, {"skip": -1.0}), {"T4": 1.0}, "Ln9")
    l15.units += P.sel(pr.add(lb_t, {"skip": -1.0}), {"T4": 1.0}, "mp")
    # pc: a^1 with a negative a right after a sign '-' ("-(-2^1)"): both signs go
    L(17).units += P.flag(P.all_of("T2", "sa_x", "u_lo"), "pc", "'pc': -(-a): the signs cancel")

    starts = {s: {f"Ln{t}": 1.0 for t in range(s)} for s in range(1, SEG)}
    af1 = {"a_lo": 1.0, "sa_x": -1.0, "one": 1.0}  # a's first byte (its sign, if any)
    src9 = pr.add({"o": 1.0, "skip": 1.0}, one(1.0))  # T4: b's first digit (past a leading 0)
    new = {"C_T3": {"T3": 1.0}, "C_T4": {"T4": 1.0}, "C_mp": {"mp": 1.0, "pc": -1.0},
           **{f"C_S{s}": starts[s] for s in range(1, SEG)},
           "C_o3": pr.add(af1, {"pc": 1.0}, neg(starts[3])), "C_o5": pr.add(af1, neg(starts[5])),
           "C_o8": pr.add({"o": 1.0, "T0": 1.0}, one(1.0), neg(starts[8])), "C_o9": pr.add(src9, neg(starts[9]))}
    lc = L(18)
    for dim, form in new.items():  # written for a reduce step only (parse.PLAN_GATE)
        lc.units += P.sel(neg({dim: 1.0}), {"is_eq": 1.0}, dim)
        lc.units += P.sel(form, P.all_of("is_eq", "act_red"), dim)
    # T3's one computed digit is b's last one (b_hi - 1), not the first after o
    lc.units += P.sel(pr.add({"b_hi": 1.0, "o": -1.0, "skip": -1.0}, one(-2.0)), P.all_of("T3", "is_eq", "act_red"), "C_o9")

    # ------------------------------------------------------------ in the walk: the replacement's byte
    k = {"mkN": 1.0}
    on = P.all_of("to_mid", "C_pow")
    for s in range(1, SEG):
        L(14).units += P.flag(pr.add(k, {f"C_S{s}": -1.0}, one(1.0)), f"sg{s}", f"'sg{s}': at or past segment {s}")

    def inseg(s: int) -> dict:
        lo_f = {"one": 1.0} if s == 0 else {f"sg{s}": 1.0}
        hi_f = {} if s == SEG - 1 else {f"sg{s + 1}": 1.0}
        return pr.add(lo_f, neg(hi_f))  # 1 inside segment s, 0 elsewhere

    for s, ch in LIT.items():
        L(15).units += P.flag(pr.add(inseg(s), on, one(-1.0)), P.HOLD[ord(ch)], f"power: segment {s} is '{ch}'")
    for s in (3, 5, 8):
        g = pr.add(inseg(s), on, one(-1.0))
        L(15).units += P.sel(pr.add({f"C_o{s}": 1.0}, k), g, "cp_t")
        L(15).units += P.flag(g, "cp_on", f"'cp_on': power segment {s} copies a byte")
    L(14).heads += [
        P.gather("digit of b (segment 9)", pr.add({"C_o9": 1.0}, k), {f"oh{d}": {f"P9_{d}": 1.0} for d in range(1, 10)}),
        P.gather("previous digit of b odd?", pr.add({"C_o9": 1.0}, k, one(-1.0)),
                 {f"oh{d}": {"P9odd": 1.0} for d in (1, 3, 5, 7, 9)}),
    ]
    g9 = pr.add(inseg(9), on, one(-1.0))
    L(15).units += P.flag(g9, "dg_on", "'dg_on': a computed digit of the power's exponent")
    for d in range(1, 10):
        if d >= 1:  # T3: b's last digit lowered by one
            L(15).units += P.gflag({f"P9_{d}": 1.0}, pr.add(g9, {"C_T3": 1.0}, one(-1.0)), "dg_v",
                                   f"power: {d} - 1", float(d - 1))
        if d // 2:  # T4: half of b, digit by digit
            L(15).units += P.gflag({f"P9_{d}": 1.0}, pr.add(g9, {"C_T4": 1.0}, one(-1.0)), "dg_v",
                                   f"power: half of {d}", float(d // 2))
    L(15).units += P.gflag({"P9odd": 1.0}, pr.add(g9, {"C_T4": 1.0}, one(-1.0)), "dg_v",
                           "power: 5 after an odd digit", 5.0)
