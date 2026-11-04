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
# Lines phase: x = pos - E, so the next byte (pos+1) is byte x of the lines block, which starts
# at E+1. Collect phase: k = pos - R, the index of the next C digit (written at R+1+k).
X = {"pos": 1.0, "E": -1.0}
K = {"pos": 1.0, "R": -1.0}
# Products of two varying integers, x*y = ((x+y)^2 - x^2 - y^2)/2 with the squares gathered from
# the pos2 table: s*x (layer-4 gathers SX2, X2) for the ruler that finds j = floor(x/s);
# j*s (layer-6 gathers JS2, J2) for the column m = x - j*s; k*s (layer-4 gathers KS2, K2) for
# the address E+1+k*s of a low product digit.
SX = pr.product_form("SX2", "X2", S2)
SXC = pr.add(SX, pr.scale(S2, -0.5), pr.scale(S_, 0.5))  # s*(x - (s-1)/2)
JS = pr.product_form("JS2", "J2", S2)
M = pr.add(X, pr.scale(JS, -1.0))
KS = pr.product_form("KS2", "K2", S2)


def builder() -> ModelBuilder:
    """The multiplication circuit's private weights, at global layers 3-10 (layers 1-2 empty).

    arith.builder merges them under the ``mul.`` namespace and silences every unit that writes
    an output dim unless ``act_mul`` (frame layer 4). Shared inputs: F, P, E, R, S (frame
    layer-2 heads), seen_bar, seen_semi (layer-2 MLP), sa, sb (layer-3 heads), PF2 (layer-4 head).
    """
    lay = Layout()
    lay.add(*PRIVATE)
    b = ModelBuilder(lay)
    lines = {"seen_bar": 1.0}  # on_when=0
    collect = {"seen_bar": 1.0, "seen_semi": -1.0}  # on_when=1
    answer = {"seen_semi": 1.0}  # on_when=1

    # Layer 3 MLP: addresses into the previous line (one line back = s bytes back), and zkey =
    # pos on nonzero C digits (between '|' and ';'; falls back to the '|') for C's top digit.
    l3 = b.at(3)
    l3.units += [
        pr.relu_unit(pr.add(X, pr.scale(S_, -1.0), pr.const(2.0)), {"tPrev1": 1.0},
                     "address of T_(j-1)[m+1] (previous line, one column up) as offset from E: x - s + 2; on "
                     "line 0 it clamps to 0, so the gather reads '=' (value 0, T_(-1) = 0)"),
        pr.relu_unit(pr.add(X, pr.scale(S_, -1.0), pr.const(1.0)), {"tPrev0": 1.0},
                     "address of T_(j-1)[m] (previous line, same column) as offset from E: x - s + 1, for the "
                     "previous column's sum; clamps to 0 ('=', value 0) on line 0"),
        pr.pos_if({"nz": 1.0, "seen_bar": 1.0, "seen_semi": -1.0}, 2.0, "zkey"),
        pr.pos_if({"is_bar": 1.0}, 1.0, "zkey"),
    ]

    # Layer 4: squares for s*x and k*s, the previous line's digits, C's top nonzero digit;
    # MLP: low/high collect switch, answer length, sign.
    l4 = b.at(4)
    l4.heads += [
        # Squares from the pos2 table: gathering pos2 at position t reads t^2. With s^2 (S2) they
        # give s*x = ((s+x)^2 - x^2 - s^2)/2 (SX), which layer 5's ruler head needs to find the
        # line index j = floor(x/s), and from it the column m being written.
        pr.gather_head("(s+x)^2", pr.add(S_, X), "SX2", src="pos2"),
        pr.gather_head("x^2", X, "X2", src="pos2"),
        pr.gather_head("T[m+1]", {"E": 1.0, "tPrev1": 1.0}, "T1"),
        pr.gather_head("T[m]", {"E": 1.0, "tPrev0": 1.0}, "T0"),
        # Likewise k*s (KS) for collect: low product digit k is T_k[0], the first digit of line k.
        pr.gather_head("(k+s)^2", pr.add(K, S_), "KS2", src="pos2"),
        pr.gather_head("k^2", K, "K2", src="pos2"),
        pr.latest_head("top nonzero of C", "zkey", "Z"),
    ]
    # low = [k < lb-1]: product digit k is T_k[0] (line k's units digit); higher digits come
    # from the last line. Selects layer 5's collect address csrc.
    l4.units += pr.clamp01_units(pr.add(LB, pr.const(-1.0), pr.scale(K, -1.0)), {"low": 1.0})
    l4.units.append(pr.relu_unit(pr.add({"Z": 1.0, "R": -1.0}, pr.const(-1.0)), {"Zo": 1.0},
                                 "answer length minus 1: Z - R - 1, how far the top nonzero digit of C sits past its "
                                 "first digit; 0 for a zero product (Z falls back to R)"))
    # sign = sa XOR sb, suppressed for a zero product (Z falls back to R)
    # Two steps [Z - R >= 1], one gated on (sa=1, sb=0), the other on (sb=1, sa=0) via +-BIG.
    nz = {"Z": 1.0, "R": -1.0}
    l4.units += pr.clamp01_units(pr.add(nz, pr.scale({"sa": 1.0}, pr.BIG), pr.scale({"sb": 1.0}, -pr.BIG),
                                        pr.const(-pr.BIG)), {"sgn": 1.0})
    l4.units += pr.clamp01_units(pr.add(nz, pr.scale({"sb": 1.0}, pr.BIG), pr.scale({"sa": 1.0}, -pr.BIG),
                                        pr.const(-pr.BIG)), {"sgn": 1.0})

    # Layer 5: line index j and the answer copy (heads); MLP: collect address, ']', '-', ';'.
    l5 = b.at(5)
    l5.heads += [
        # j = floor(x/s) by attention over a ruler of positions k: score -30*(s*k - x + (s-1)/2)^2
        # peaks at k = j. Expanded, the query needs only s^2 (S2) and s*(x - (s-1)/2) (SXC, from
        # the layer-4 squares). J is read by layer 6 (j*s squares, b_j one-hot, lastline).
        pr.floor_div_head("j = floor(x/s)", S2, SXC, "J"),
        # answer digit j' = pos - S - sgn is C digit Zo - j', at R + 1 + Zo - j' (as in addition).
        pr.gather_head("answer digit", pr.add({"R": 1.0, "Zo": 1.0, "pos": -1.0, "S": 1.0, "sgn": 1.0},
                                              pr.const(1.0)), "Av"),
    ]
    low_src = pr.add({"E": 1.0}, pr.const(1.0), KS)  # T_k[0] at E+1+k*s
    high_src = pr.add({"R": 1.0, "F": 1.0, "sa": 1.0, "sb": 1.0, "E": -1.0}, K, pr.const(2.0))  # last line
    l5.units.append(pr.relu_unit(high_src, {"csrc": 1.0},
                                 "collect address for a high product digit k: last line at R+F+sa+sb-E+k+2 (the "
                                 "select below swaps in E+1+k*s while 'low')"))
    l5.units += pr.times_flag_units(pr.add(low_src, pr.scale(high_src, -1.0)), {"low": 1.0}, {"csrc": 1.0})
    # Answer phase: ']' once all Zo+1 digits are out (j' > Zo).
    l5.units += pr.clamp01_units(pr.gated({"pos": 1.0, "S": -1.0, "sgn": -1.0, "Zo": -1.0}, answer),
                                 {"frame_end": 1.0})
    l5.units.append(pr.relu_unit({"sgn": 1.0, "is_semi": 1.0, "one": -1.0}, {"emit_minus": 1.0},
                                 "on the ';' byte of a negative product (sgn): write '-' before the answer digits"))
    # Collect phase: ';' when k == la + lb, i.e. all la+lb product digits are in C.
    l5.units += pr.hat_units(pr.gated(pr.add(K, pr.scale(LA, -1.0), pr.scale(LB, -1.0)), collect), 0.0,
                             {"work_end": 1.0})

    # Layer 6: squares for j*s, digit b_j as a one-hot, the collected digit; MLP: column m
    # = x - j*s, a's digit addresses, and where in the line block the next byte falls.
    l6 = b.at(6)
    l6.heads += [
        pr.gather_head("(j+s)^2", pr.add({"J": 1.0}, S_), "JS2", src="pos2"),
        pr.gather_head("j^2", {"J": 1.0}, "J2", src="pos2"),
        # A gather at E-1-j (b's digit j) that copies oh0..oh9 instead of val, so b_j arrives as
        # a one-hot Boh0..Boh9 and a_m*b_j becomes a table lookup (layer 7).
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
        pr.relu_unit(pr.add({"P": 1.0, "F": -1.0, "sa": -1.0}, pr.const(-1.0), pr.scale(M, -1.0)), {"tAm": 1.0},
                     "address of a_m as offset from a's fence F+sa: la - m; 0 at m = la (the line's extra top "
                     "digit), so the gather reads the fence, value 0"),
        pr.relu_unit(pr.add({"P": 1.0, "F": -1.0, "sa": -1.0}, pr.scale(M, -1.0)), {"tAm1": 1.0},
                     "address of a_(m-1) as offset from F+sa: la - m + 1; at m = 0 it lands on the operator P "
                     "(value 0: no column below the units)"),
    ]
    # sep_slot = [m == s-1]: the byte after a line's la+1 digits (',' or '|');
    # lastline = [j == lb-1]. Layer 7 picks ',' or '|' from them.
    l6.units += pr.hat_units(pr.add(M, pr.scale(S_, -1.0), pr.const(1.0)), 0.0, {"sep_slot": 1.0})
    l6.units += pr.hat_units(pr.add({"J": 1.0}, pr.scale(LB, -1.0), pr.const(1.0)), 0.0, {"lastline": 1.0})

    # Layer 7: gather a_m and a_(m-1); MLP: the two column sums through the one-hot b_j, and
    # the line separators.
    l7 = b.at(7)
    l7.heads += [
        pr.gather_head("a_m", {"F": 1.0, "sa": 1.0, "tAm": 1.0}, "Am"),
        pr.gather_head("a_m-1", {"F": 1.0, "sa": 1.0, "tAm1": 1.0}, "Am1"),
    ]
    # Ym = T_(j-1)[m+1] + a_m*b_j (this column), Yp = T_(j-1)[m] + a_(m-1)*b_j (the column
    # below, for the carry), as sums over v of ReLU(T + v*a - 99*(1 - Boh_v)).
    for v in range(10):  # exactly one Boh_v is 1
        off = pr.add(pr.scale({f"Boh{v}": 1.0}, 99.0), pr.const(-99.0))
        l7.units.append(pr.relu_unit(pr.add({"T0": 1.0, "Am1": float(v)}, off), {"Yp": 1.0},
                                     f"previous column's sum if b_j = {v}: T[m] + {v}*a_(m-1); silenced by -99 "
                                     f"when Boh{v} is 0 (the sum is at most 90)"))
        l7.units.append(pr.relu_unit(pr.add({"T1": 1.0, "Am": float(v)}, off), {"Ym": 1.0},
                                     f"this column's sum before carry if b_j = {v}: T[m+1] + {v}*a_m; silenced by -99 "
                                     f"when Boh{v} is 0"))
    l7.units.append(pr.relu_unit(pr.gated({"sep_slot": 1.0, "lastline": -1.0}, lines, 0.0),
                                 {"emit_comma": 1.0},
                                 "lines phase, end of a line that is not the last (sep_slot, not lastline): write ','"))
    l7.units.append(pr.relu_unit(pr.gated({"sep_slot": 1.0, "lastline": 1.0, "one": -1.0}, lines, 0.0),
                                 {"emit_bar": 1.0},
                                 "lines phase, end of the last line (sep_slot and lastline): write '|' to start collecting"))

    # Layer 8 MLP: Fl = floor(Yp/10) = sum over t = 1..9 of [Yp >= 10t] (Yp <= 90).
    l8 = b.at(8)
    for t in range(1, 10):
        l8.units += pr.clamp01_units(pr.add({"Yp": 1.0}, pr.const(-10.0 * t + 1.0)), {"Fl": 1.0})

    # Layer 9 MLP: the carry into column m, recovered from the text (never stored). With
    # y = Yp (column m-1 before its carry-in c <= 9), column m-1 wrote d = (y + c) mod 10, which is
    # this byte (val). y + c passes a multiple of 10 at most once, and does iff d < y mod 10, so
    # carry = floor(y/10) + [y mod 10 > d] = Fl + [Yp - 10*Fl - val >= 1].
    # A non-digit byte (',' or '=') means the next byte opens a line: forced to 0 (-9 >= Fl, -BIG).
    l9 = b.at(9)
    not_digit = {"one": 1.0, "is_digit": -1.0}  # first column of a line: no carry in
    l9.units.append(pr.relu_unit(pr.add({"Fl": 1.0}, pr.scale(not_digit, -9.0)), {"carry": 1.0},
                                 "carry part 1: floor(Yp/10) from Fl; zeroed when this byte is not a digit (the next "
                                 "byte starts a line, no carry in)"))
    l9.units += pr.clamp01_units(pr.add({"Yp": 1.0, "Fl": -10.0, "val": -1.0},
                                        pr.scale(not_digit, -pr.BIG)), {"carry": 1.0})

    # Layer 10 MLP: the digit to write, one ramp set per phase: line digit (Ym + carry) mod 10
    # for 0..99 (lines), collected digit Cv (collect), answer digit Av (answer).
    l10 = b.at(10)
    l10.units += cm.ramp_mod10_units({"Ym": 1.0, "carry": 1.0}, 0, 99, lines, 0.0)
    l10.units += cm.ramp_mod10_units({"Cv": 1.0}, 0, 9, collect, 1.0)
    l10.units += cm.ramp_mod10_units({"Av": 1.0}, 0, 9, answer, 1.0)
    return b
