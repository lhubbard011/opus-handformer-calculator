"""Multiplication circuit, inside a frame
``[ a * b = L_0 , L_1 , ... L_{lb-1} | C ; answer ]``.

Lines: L_j is T_j LSB first with exactly la+1 digits,
``T_j = floor(T_{j-1}/10) + a*b_j`` (T_{-1} = 0), so

    line j digit m = T_{j-1}[m+1] + a_m * b_j + carry_m   (mod 10)

T_j < 10^(la+1), so every line has stride s = la+2 = P-F+1 and line j starts
at E+1+j*s. Product digit k is T_k[0] for k < lb-1; the top la+1 product
digits are the last line. C gathers the full product LSB first (la+lb
digits); the answer is C reversed from its top nonzero digit.

Neural tricks:

* pos^2 is a table of squares, so x*y = ((x+y)^2 - x^2 - y^2)/2 costs three
  gathers: s^2, s*x, j*s, k*s.
* floor(x/s) by attention over a ruler of positions: the line index j.
* b_j moves as a one-hot, so a_m*b_j is 10 gated ReLUs.
* carry = floor(y/10) + [prev_digit < y mod 10] with y = T_{j-1}[m] +
  a_{m-1}*b_j <= 90: the carry (<= 9) is recovered, never stored.

Positions: F '[', P '*', E '=', R '|', S ';' (shared). Operands are magnitudes:
a in (F+sa, P), b in (P+sb, E); the answer's sign is sa XOR sb (none for 0). Lines phase:
x = q - E is the next byte's offset, j = floor(x/s), m = x - j*s.
Collect phase: k = q - R.

Global layers 3..10.
"""

from __future__ import annotations

from . import common as cm
from . import primitives as pr
from .model import HeadSpec, Layout, ModelBuilder

PRIVATE = ("tPrev1", "tPrev0", "zkey", "SX2", "X2", "T1", "T0", "KS2", "K2", "Z",
           "low", "Zo", "J", "Av", "csrc", "JS2", "J2", *(f"Boh{d}" for d in range(10)), "Cv",
           "tAm", "tAm1", "sep_slot", "lastline", "Am", "Am1", "Yp", "Ym", "Fl", "carry", "sgn")

LA = {"P": 1.0, "F": -1.0, "sa": -1.0, "one": -1.0}  # len(a), a in (F+sa, P)
S_ = {"P": 1.0, "F": -1.0, "sa": -1.0, "one": 1.0}  # s = la + 2
S2 = {"PF2": 1.0, "P": 2.0, "F": -2.0, "sa": -2.0, "one": 1.0}  # s^2 = (P-Fa)^2 + 2(P-Fa) + 1
LB = {"E": 1.0, "P": -1.0, "sb": -1.0, "one": -1.0}  # len(b), b in (P+sb, E)
X = {"pos": 1.0, "E": -1.0}
K = {"pos": 1.0, "R": -1.0}
SX = pr.product_form("SX2", "X2", S2)
SXC = pr.add(SX, pr.scale(S2, -0.5), pr.scale(S_, 0.5))  # s*(x - (s-1)/2)
JS = pr.product_form("JS2", "J2", S2)
M = pr.add(X, pr.scale(JS, -1.0))
KS = pr.product_form("KS2", "K2", S2)


def builder() -> ModelBuilder:
    lay = Layout()
    lay.add(*PRIVATE)
    b = ModelBuilder(lay)
    lines = {"seen_bar": 1.0}  # on_when=0
    collect = {"seen_bar": 1.0, "seen_semi": -1.0}  # on_when=1
    answer = {"seen_semi": 1.0}  # on_when=1

    l3 = b.at(3)
    l3.units += [
        pr.relu_unit(pr.add(X, pr.scale(S_, -1.0), pr.const(2.0)), {"tPrev1": 1.0}),
        pr.relu_unit(pr.add(X, pr.scale(S_, -1.0), pr.const(1.0)), {"tPrev0": 1.0}),
        pr.pos_if({"nz": 1.0, "seen_bar": 1.0, "seen_semi": -1.0}, 2.0, "zkey"),
        pr.pos_if({"is_bar": 1.0}, 1.0, "zkey"),
    ]

    l4 = b.at(4)
    l4.heads += [
        pr.gather_head("(s+x)^2", pr.add(S_, X), "SX2", src="pos2"),
        pr.gather_head("x^2", X, "X2", src="pos2"),
        pr.gather_head("T[m+1]", {"E": 1.0, "tPrev1": 1.0}, "T1"),
        pr.gather_head("T[m]", {"E": 1.0, "tPrev0": 1.0}, "T0"),
        pr.gather_head("(k+s)^2", pr.add(K, S_), "KS2", src="pos2"),
        pr.gather_head("k^2", K, "K2", src="pos2"),
        pr.latest_head("top nonzero of C", "zkey", "Z"),
    ]
    l4.units += pr.clamp01_units(pr.add(LB, pr.const(-1.0), pr.scale(K, -1.0)), {"low": 1.0})
    l4.units.append(pr.relu_unit(pr.add({"Z": 1.0, "R": -1.0}, pr.const(-1.0)), {"Zo": 1.0}))
    # sign = sa XOR sb, suppressed for a zero product (Z falls back to R)
    nz = {"Z": 1.0, "R": -1.0}
    l4.units += pr.clamp01_units(pr.add(nz, pr.scale({"sa": 1.0}, pr.BIG), pr.scale({"sb": 1.0}, -pr.BIG),
                                        pr.const(-pr.BIG)), {"sgn": 1.0})
    l4.units += pr.clamp01_units(pr.add(nz, pr.scale({"sb": 1.0}, pr.BIG), pr.scale({"sa": 1.0}, -pr.BIG),
                                        pr.const(-pr.BIG)), {"sgn": 1.0})

    l5 = b.at(5)
    l5.heads += [
        pr.floor_div_head("j = floor(x/s)", S2, SXC, "J"),
        pr.gather_head("answer digit", pr.add({"R": 1.0, "Zo": 1.0, "pos": -1.0, "S": 1.0, "sgn": 1.0},
                                              pr.const(1.0)), "Av"),
    ]
    low_src = pr.add({"E": 1.0}, pr.const(1.0), KS)  # T_k[0] at E+1+k*s
    high_src = pr.add({"R": 1.0, "F": 1.0, "sa": 1.0, "sb": 1.0, "E": -1.0}, K, pr.const(2.0))  # last line
    l5.units.append(pr.relu_unit(high_src, {"csrc": 1.0}))
    l5.units += pr.times_flag_units(pr.add(low_src, pr.scale(high_src, -1.0)), {"low": 1.0}, {"csrc": 1.0})
    l5.units += pr.clamp01_units(pr.gated({"pos": 1.0, "S": -1.0, "sgn": -1.0, "Zo": -1.0}, answer),
                                 {"frame_end": 1.0})
    l5.units.append(pr.relu_unit({"sgn": 1.0, "is_semi": 1.0, "one": -1.0}, {"emit_minus": 1.0}))
    l5.units += pr.hat_units(pr.gated(pr.add(K, pr.scale(LA, -1.0), pr.scale(LB, -1.0)), collect), 0.0,
                             {"work_end": 1.0})

    l6 = b.at(6)
    l6.heads += [
        pr.gather_head("(j+s)^2", pr.add({"J": 1.0}, S_), "JS2", src="pos2"),
        pr.gather_head("j^2", {"J": 1.0}, "J2", src="pos2"),
        HeadSpec(
            name="b_j one-hot",
            query=[pr.scale(pr.add({"E": 1.0, "J": -1.0}, pr.const(-1.0)), 2 * pr.GATHER_BETA),
                   pr.const(-pr.GATHER_BETA)],
            key=[{"pos": 1.0}, {"pos2": 1.0}],
            value={f"oh{d}": {f"Boh{d}": 1.0} for d in range(10)},
        ),
        pr.gather_head("collect digit", {"csrc": 1.0}, "Cv"),
    ]
    # This MLP reads JS2/J2 written by this layer's heads (heads run first).
    l6.units += [
        pr.relu_unit(pr.add({"P": 1.0, "F": -1.0, "sa": -1.0}, pr.const(-1.0), pr.scale(M, -1.0)), {"tAm": 1.0}),
        pr.relu_unit(pr.add({"P": 1.0, "F": -1.0, "sa": -1.0}, pr.scale(M, -1.0)), {"tAm1": 1.0}),
    ]
    l6.units += pr.hat_units(pr.add(M, pr.scale(S_, -1.0), pr.const(1.0)), 0.0, {"sep_slot": 1.0})
    l6.units += pr.hat_units(pr.add({"J": 1.0}, pr.scale(LB, -1.0), pr.const(1.0)), 0.0, {"lastline": 1.0})

    l7 = b.at(7)
    l7.heads += [
        pr.gather_head("a_m", {"F": 1.0, "sa": 1.0, "tAm": 1.0}, "Am"),
        pr.gather_head("a_m-1", {"F": 1.0, "sa": 1.0, "tAm1": 1.0}, "Am1"),
    ]
    for v in range(10):  # exactly one Boh_v is 1
        off = pr.add(pr.scale({f"Boh{v}": 1.0}, 99.0), pr.const(-99.0))
        l7.units.append(pr.relu_unit(pr.add({"T0": 1.0, "Am1": float(v)}, off), {"Yp": 1.0}))
        l7.units.append(pr.relu_unit(pr.add({"T1": 1.0, "Am": float(v)}, off), {"Ym": 1.0}))
    l7.units.append(pr.relu_unit(pr.gated({"sep_slot": 1.0, "lastline": -1.0}, lines, 0.0),
                                 {"emit_comma": 1.0}))
    l7.units.append(pr.relu_unit(pr.gated({"sep_slot": 1.0, "lastline": 1.0, "one": -1.0}, lines, 0.0),
                                 {"emit_bar": 1.0}))

    l8 = b.at(8)
    for t in range(1, 10):
        l8.units += pr.clamp01_units(pr.add({"Yp": 1.0}, pr.const(-10.0 * t + 1.0)), {"Fl": 1.0})

    l9 = b.at(9)
    not_digit = {"one": 1.0, "is_digit": -1.0}  # first column of a line: no carry in
    l9.units.append(pr.relu_unit(pr.add({"Fl": 1.0}, pr.scale(not_digit, -9.0)), {"carry": 1.0}))
    l9.units += pr.clamp01_units(pr.add({"Yp": 1.0, "Fl": -10.0, "val": -1.0},
                                        pr.scale(not_digit, -pr.BIG)), {"carry": 1.0})

    l10 = b.at(10)
    l10.units += cm.ramp_mod10_units({"Ym": 1.0, "carry": 1.0}, 0, 99, lines, 0.0)
    l10.units += cm.ramp_mod10_units({"Cv": 1.0}, 0, 9, collect, 1.0)
    l10.units += cm.ramp_mod10_units({"Av": 1.0}, 0, 9, answer, 1.0)
    return b
