"""Powers in the looped calculator: one pass rewrites ``a^b`` as text (REWRITE.md §4.4).

The same binary-exponentiation step as the frame-writing calculator (tests/reference.py
``power_expansion``); the products it writes are evaluated by later passes:

  T0  b < 0            (1/a^|b|)
  T1  b = 0            1
  T2  b is "1"         a
  T3  b odd            (a*a^b')       b' = b with its last digit lowered by 1
  T4  b even           (a*a)^h        h = b/2, digit by digit: 5*[previous digit odd] + d//2

Every template is one layout of eleven segments, most of them empty for a given template:

  0 '('  1 '1'  2 '/'  3 a  4 '*'  5 a  6 ')'  7 '^'  8 b (without its sign or last digit)
  9 computed digits (b's last digit - 1, or the halves)  10 ')'

The root works out each segment's length from the template, every cell learns from broadcast
which segment it falls in, and each writes a fixed byte, a copied byte, or a computed digit.
A rewrite longer than the cells it replaces needs blanks after the expression. When there are
not enough, the root raises ``grow`` and keeps the tape: the runtime inserts a blank before
the root, and the next pass tries again.
"""

from __future__ import annotations

from .. import primitives as pr
from . import circuit as cc

SEG = 11
LIT = {0: "(", 1: "1", 2: "/", 4: "*", 6: ")", 7: "^", 10: ")"}  # fixed-byte segments

DIMS = (
    "bnzp", "blast", "bodd", "bfirst", "bnz", "lb1", "is1", "b01", "lb2",
    "T0", "T1", "b1", "skip", "T2", "T3", "T4", "pc",
    *(f"L{s}" for s in range(SEG - 1)), "mp", "off3", "off5", "off8", "off9", "growp",
    "pw_B", "t3_B", "t4_B", *(f"S{s}_B" for s in range(1, SEG)), "o3_B", "o5_B", "o8_B", "o9_B",
    *(f"sg{s}" for s in range(1, SEG)),
    *(f"P3_{d}" for d in cc.CONTENT.values()), *(f"P5_{d}" for d in cc.CONTENT.values()),
    *(f"P8_{d}" for d in cc.CONTENT.values()), *(f"P9_{d}" for d in range(10)), "P9odd",
)


def build(b) -> None:
    one = pr.const
    pos = {"pos": 1.0}
    la_t = {"o": 1.0, "a_lo": -1.0, "sa_x": 1.0, "one": -1.0}  # a's text, its sign included
    lb_t = {"b_hi": 1.0, "o": -1.0, "one": -1.0}  # b's text, its sign included
    last = pr.add({"b_hi": 1.0}, one(-1.0))

    l10 = b.at(10)
    odd = {f"oh{d}": {"bodd": 1.0} for d in (1, 3, 5, 7, 9)}
    l10.heads += [
        cc._gather("b nonzero?", last, {"Lnz": {"bnzp": 1.0}}),
        cc._gather("b's last digit", last, {"val": {"blast": 1.0}, **odd}),
        cc._gather("b's first digit", pr.add({"o": 1.0}, one(1.0)), {"val": {"bfirst": 1.0}}),
    ]
    l10.units += cc._flag({"bnzp": 1.0, "o": -1.0}, "bnz", "'bnz': b has a nonzero digit")
    l10.units += cc._hat(pr.add(lb_t, one(-1.0)), "lb1")  # b is one byte
    l10.units += cc._hat(pr.add({"blast": 1.0}, one(-1.0)), "is1")  # b's last digit is 1
    l10.units += cc._flag(pr.add({"bfirst": -1.0}, one(2.0)), "b01", "'b01': b's first digit is 0 or 1")
    l10.units += cc._flag(pr.add(lb_t, one(-1.0)), "lb2", "'lb2': b has two bytes or more")

    l11 = b.at(11)
    l11.units += cc._flag(cc._all("opc", "sb_x"), "T0", "'T0': a^-n -> (1/a^n)")
    l11.units += cc._flag({"opc": 1.0, "sb_x": -1.0, "bnz": -1.0}, "T1", "'T1': a^0 -> 1")
    l11.units += cc._flag(cc._all("lb1", "is1"), "b1", "'b1': b is the text '1'")
    l11.units += cc._flag(cc._all("b01", "lb2"), "skip", "'skip': halving b gives a leading 0, dropped")

    l12 = b.at(12)
    l12.units += cc._flag(pr.add({"opc": 1.0, "b1": 1.0}, one(-1.0)), "T2", "'T2': a^1 -> a")
    l12.units += cc._flag({"opc": 1.0, "sb_x": -1.0, "bnz": 1.0, "bodd": 1.0, "b1": -1.0, "one": -2.0}, "T3",
                          "'T3': b odd -> (a*a^(b-1))")
    l12.units += cc._flag({"opc": 1.0, "sb_x": -1.0, "bnz": 1.0, "bodd": -1.0, "one": -1.0}, "T4",
                          "'T4': b even -> (a*a)^(b/2)")

    # Segment lengths (each a product of a template flag and a text length) and their sum.
    l13 = b.at(13)
    lits = {0: ("T0", "T3", "T4"), 1: ("T0", "T1"), 2: ("T0",), 4: ("T3", "T4"), 6: ("T4",),
            7: ("T0", "T3", "T4")}  # segment 10 is last: nothing starts after it, so its length is not needed
    for s, ts in lits.items():
        l13.units += [cc._relu({t: 1.0}, f"L{s}", f"'L{s}': segment {s} ('{LIT[s]}') in template {t}") for t in ts]
    l13.units += cc._sel(la_t, {"T0": 1.0, "T2": 1.0, "T3": 1.0, "T4": 1.0}, "L3")
    l13.units += cc._sel(la_t, {"T3": 1.0, "T4": 1.0}, "L5")
    l13.units += cc._sel(pr.add(lb_t, one(-1.0)), {"T0": 1.0, "T3": 1.0}, "L8")
    l13.units.append(cc._relu({"T3": 1.0}, "L9", "'L9': b's last digit, lowered (T3)"))
    l13.units += cc._sel(pr.add(lb_t, {"skip": -1.0}), {"T4": 1.0}, "L9")
    l13.units += cc._sel(la_t, {"T0": 1.0, "T2": 1.0, "T3": 1.0, "T4": 1.0}, "mp")
    l13.units += cc._sel(la_t, {"T3": 1.0, "T4": 1.0}, "mp")
    l13.units += cc._sel(pr.add(lb_t, one(-1.0)), {"T0": 1.0, "T3": 1.0}, "mp")
    l13.units += cc._sel(pr.add(lb_t, {"skip": -1.0}), {"T4": 1.0}, "mp")
    l13.units += [cc._relu({t: float(n)}, "mp", f"'mp': the fixed bytes of template {t}")
                  for t, n in (("T0", 5.0), ("T1", 1.0), ("T3", 5.0), ("T4", 4.0))]

    # pc: a^1 with a negative a right after a sign '-' ("-(-2^1)"): both signs go
    l15 = b.at(15)
    l15.units += cc._flag(cc._all("T2", "sa_x", "u_lo"), "pc", "'pc': -(-a): the signs cancel")
    lo0 = cc.LO0
    span = pr.add(cc.RED_HI, pr.scale(lo0, -1.0), one(1.0))  # cells the redex covers
    blanks = pr.add(pos, {"SP": -1.0})  # at the root: the blanks after the expression
    l16 = b.at(16)
    need = pr.add({"mp": 1.0, "pc": -1.0}, pr.scale(span, -1.0), pr.scale(blanks, -1.0))
    l16.units += cc._flag(pr.add(need, pr.scale({"opc": 1.0, "act_red": 1.0}, pr.BIG), one(-2 * pr.BIG)), "growp",
                          "'growp': the rewrite is longer than its cells plus the blanks: ask for room")
    # where each copied segment starts reading, relative to the cell (lo = lo0 - pc)
    lo = pr.add(lo0, {"pc": -1.0})
    starts = {s: {f"L{t}": 1.0 for t in range(s)} for s in range(1, SEG)}
    for seg, src in ((3, pr.add(cc.AF, one(1.0), {"pc": 1.0})), (5, pr.add(cc.AF, one(1.0))),
                     (8, {"o": 1.0, "T0": 1.0, "one": 1.0}),
                     (9, pr.add({"o": 1.0, "skip": 1.0, "one": 1.0}, ))):
        l16.units += cc._sel(pr.add(src, pr.scale(lo, -1.0), pr.scale(starts[seg], -1.0)), {"one": 1.0}, f"off{seg}")
    # T3's digit is b's last one (one cell left of b_hi), not the first after o
    l16.units += cc._sel(pr.add({"b_hi": 1.0, "o": -1.0, "skip": -1.0}, one(-2.0)), {"T3": 1.0}, "off9")

    l29 = b.at(29)
    value = {"T3": {"t3_B": 1.0}, "T4": {"t4_B": 1.0}, "off3": {"o3_B": 1.0}, "off5": {"o5_B": 1.0},
             "off8": {"o8_B": 1.0}, "off9": {"o9_B": 1.0}, "rwp": {"pw_B": 1.0}}
    for s in range(SEG - 1):  # S_s_B = L0 + ... + L(s-1): where segment s starts
        value[f"L{s}"] = {f"S{t}_B": 1.0 for t in range(s + 1, SEG)}
    l29.heads.append(cc._broadcast("power template to every cell", value))
    k = pr.add(pos, {"lo_B": -1.0})  # this cell's index in the rewrite
    for s in range(1, SEG):
        l29.units += cc._step(pr.add(k, {f"S{s}_B": -1.0}, one(1.0)), f"sg{s}")  # k >= start of segment s

    l30 = b.at(30)
    chars = cc.CONTENT.values()
    l30.heads += [
        cc._gather("copy of a (segment 3)", pr.add(pos, {"o3_B": 1.0}), {d: {f"P3_{d}": 1.0} for d in chars}, full=True),
        cc._gather("copy of a (segment 5)", pr.add(pos, {"o5_B": 1.0}), {d: {f"P5_{d}": 1.0} for d in chars}, full=True),
        cc._gather("copy of b (segment 8)", pr.add(pos, {"o8_B": 1.0}), {d: {f"P8_{d}": 1.0} for d in chars}, full=True),
        cc._gather("digit of b (segment 9)", pr.add(pos, {"o9_B": 1.0}),
                   {f"oh{d}": {f"P9_{d}": 1.0} for d in range(10)}, full=True),
        cc._gather("previous digit of b odd?", pr.add(pos, {"o9_B": 1.0}, one(-1.0)),
                   {f"oh{d}": {"P9odd": 1.0} for d in (1, 3, 5, 7, 9)}, full=True),
    ]

    def inseg(s: int) -> dict:
        lo_f = {"gelo": 1.0} if s == 0 else {f"sg{s}": 1.0}
        hi_f = {"one": 1.0, "ltend": -1.0} if s == SEG - 1 else {f"sg{s + 1}": 1.0}  # 1 past the segment
        return pr.add(lo_f, pr.scale(hi_f, -1.0))  # 1 inside segment s, 0 elsewhere

    on = {"pw_B": 1.0, "ltend": 1.0}  # a power rewrite, and inside the rewritten bytes
    for s, ch in LIT.items():
        l30.units += cc._flag(pr.add(inseg(s), on, one(-2.0)), cc.HOLD[ord(ch)], f"power: segment {s} is '{ch}'")
    for s in (3, 5, 8):
        for byte, dim in cc.CONTENT.items():
            l30.units += cc._flag(pr.add({f"P{s}_{dim}": 1.0}, inseg(s), on, one(-3.0)), cc.HOLD[byte],
                                  f"power: segment {s} copies '{chr(byte)}'")
    for dgt in range(1, 10):  # T3: b's last digit lowered by one
        l30.units += cc._flag(pr.add({f"P9_{dgt}": 1.0, "t3_B": 1.0}, inseg(9), on, one(-4.0)),
                              cc.HOLD[ord("0") + dgt - 1], f"power: {dgt} - 1")
    for dgt in range(10):  # T4: half of b, digit by digit
        for p in (0, 1):
            out = 5 * p + dgt // 2
            par = {"P9odd": 1.0} if p else {"P9odd": -1.0, "one": 1.0}
            l30.units += cc._flag(pr.add({f"P9_{dgt}": 1.0, "t4_B": 1.0}, par, inseg(9), on, one(-5.0)),
                                  cc.HOLD[ord("0") + out], f"power: half of {dgt} after an {'odd' if p else 'even'} digit")
