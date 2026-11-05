"""Division in the looped calculator: long division over passes (REWRITE.md §4.5).

``a/b~N`` means floor(A / B) with N fractional digits, A = |a|'s digits followed by fb + N
zeros and B = |b|'s digits followed by fa zeros: the same integers as the frame-writing
calculator (tests/reference.py ``step``). With L = len(A), the root keeps the remainder r in
the column accumulator K (columns 0..len(B) of the redex's own cells), the next digit index t
and the current quotient digit's count qc in the carried dims kk and qc, and the passes go:

  room    the trailing blanks hold the quotient: fewer than L + 3 of them, ask for one more
  shift   r < B (or t = 0): store qc as quotient digit t-1 in blank cell SP + t - 1, bring down
          the next digit of A (r = 10 r + A_t: every column takes the one below), qc = 0, t + 1
  sub     r >= B: r = r - B (the borrow chain the subtraction uses), qc + 1
  write   t = L + 1: the stored digits are the quotient's columns; the usual writer formats
          them (sign sa XOR sb, F = N fractional digits) in place of the redex

Every pass but the write keeps the tape's bytes; only the carried dims change. r < B is the
borrow out of r - B, read at the top column as for a subtraction.
"""

from __future__ import annotations

from .. import common as cm
from .. import primitives as pr
from . import circuit as cc

N_DIGITS = 4  # precision ~N: N <= 9999

DIMS = (
    "has_til", "f_Ndg", "Ndg", *(f"qN{k}" for k in range(1, N_DIGITS)), *(f"Nd{k}" for k in range(N_DIGITS)), "N",
    "inA", "geia", "t1", "At_raw", "At", "dfinal", "growd", "dwork", "m_dfin",
    "B_div", "B_dfin", "B_hdb", "d_lo", "d_hi", "skpb", "bvd", "dK", "dsub",
    "dsub_B", "dw_B", "At_B", "qc_B", "QS_B", "t_B", "t1_B", "Kn", "shK", "st_at", "nQ", "nqc",
)


def build(b) -> None:
    one = pr.const
    pos = {"pos": 1.0}
    neg = lambda f: pr.scale(f, -1.0)  # noqa: E731

    # Precision N: the digits after the '~' (spaces are gone by then), right to left from the
    # latest one; Nd_k is N's 10^k digit (0 where N has none, and N = 0 without a '~').
    b.at(2).units += cc._flag({"Ptil": 1.0}, "has_til", "'has_til': a '~' is at or before this cell")
    b.at(3).units += pr.latest_feature({"is_digit": 1.0, "has_til": 1.0, "one": -1.0}, "f_Ndg")
    b.at(4).heads.append(pr.latest_head("latest digit of N", "f_Ndg", "Ndg"))
    at = {0: {"Ndg": 1.0}, **{k: {f"qN{k}": 1.0} for k in range(1, N_DIGITS)}}
    for k in range(N_DIGITS):
        lk = b.at(5 + k)
        lk.heads.append(cc._gather(f"precision digit {k}", at[k], {"val": {f"Nd{k}": 1.0}}))
        if k + 1 < N_DIGITS:
            lk.heads.append(cc._gather(f"precision digit {k + 1} position", pr.add(at[k], one(-1.0)),
                                       {"Ndg": {f"qN{k + 1}": 1.0}}))
    b.at(5 + N_DIGITS).units += pr.times_flag_units({f"Nd{k}": 10.0 ** k for k in range(N_DIGITS)},
                                                    {"has_til": 1.0}, {"N": 1.0})

    ia = {"o": 1.0, "fa": -1.0, "hd_a": -1.0, "a_lo": -1.0, "one": -1.0}
    la = pr.add(ia, {"fa": 1.0})
    lb = {"b_hi": 1.0, "hd_b": -1.0, "o": -1.0, "sb_x": -1.0, "one": -1.0}  # all of b's digits
    big_l = pr.add(la, {"fb": 1.0, "N": 1.0})  # L = len(A)
    blanks = pr.add(pos, {"SP": -1.0})

    l11 = b.at(11)
    l11.units += cc._flag(pr.add(la, {"kk": -1.0}), "inA", "'inA': A's digit t is one of a's (not a padding 0)")
    l11.units += cc._step(pr.add({"kk": 1.0}, pr.scale(ia, -1.0), one(1.0)), "geia", gate={"hd_a": 1.0})  # past a's '.'
    l11.units += cc._flag({"kk": 1.0}, "t1", "'t1': a digit of A has been brought down")

    l12 = b.at(12)
    l12.heads.append(cc._gather("A's digit t", {"a_lo": 1.0, "kk": 1.0, "geia": 1.0, "one": 1.0}, {"val": {"At_raw": 1.0}}))
    l12.units += pr.times_flag_units({"At_raw": 1.0}, {"inA": 1.0}, {"At": 1.0})
    l12.units += cc._flag(pr.add({"kk": 1.0}, neg(big_l)), "dfinal", "'dfinal': every quotient digit is stored")
    l12.units += cc._step(pr.add(big_l, one(3.0), neg(blanks)), "growd",
                          gate={"act_red": 1.0, "opdiv": 1.0, "one": -1.0})  # the blanks cannot hold the quotient

    l13 = b.at(13)
    l13.units += cc._flag({"act_red": 1.0, "opdiv": 1.0, "dfinal": -1.0, "growd": -1.0, "one": -1.0}, "dwork",
                          "'dwork': a long-division pass (shift or subtract; the tape keeps its bytes)")
    l13.units += cc._flag({"act_red": 1.0, "opdiv": 1.0, "dfinal": 1.0, "growd": -1.0, "one": -2.0}, "m_dfin",
                          "'m_dfin': the pass that writes the quotient")
    mdf = {"act_red": 1.0, "opdiv": 1.0, "dfinal": 1.0, "growd": -1.0, "one": -2.0}  # = m_dfin, usable this layer
    dw = {"act_red": 1.0, "opdiv": 1.0, "dfinal": -1.0, "growd": -1.0, "one": -1.0}  # = dwork
    # the quotient's column 0 is the last stored digit, at SP + L - 1; its F is N
    l13.units += cc._sel(pr.add({"SP": 1.0}, big_l, {"b_hi": -1.0}), mdf, "CB")
    l13.units += cc._sel(pr.add(lb, {"fa": 1.0}), dw, "Wm")  # r's columns: 0..len(B)
    l13.units += cc._sel(pr.add(big_l, one(-1.0)), mdf, "Wm")
    l13.units += cc._sel({"N": 1.0}, mdf, "F")
    l13.units.append(cc._relu(pr.add({"growd": 1.0}), "grow_any", "'grow_any': the quotient needs room"))

    # Columns. In a division pass a_j is the remainder's digit K_j and b_j is B's: b's digit
    # i = j - fa (fa zeros first), one cell further left past b's '.'.
    l14 = b.at(14)
    next(h for h in l14.heads if h.name == "redex to every cell").value.update({"dwork": {"B_div": 1.0}, "m_dfin": {"B_dfin": 1.0}, "hd_b": {"B_ib": -1.0, "B_hdb": 1.0}})
    jj = pr.add({"B_bhi": 1.0}, one(-1.0), neg(pos))
    ib_ = pr.add(jj, {"B_fa": -1.0})  # b's digit index i
    l14.units += cc._step(pr.add(ib_, one(1.0)), "d_lo")  # i >= 0
    l14.units += cc._step(pr.add({"B_ib": 1.0, "B_fb": 1.0}, neg(ib_)), "d_hi")  # i <= lb - 1
    l14.units += cc._step(pr.add(ib_, {"B_fb": -1.0}, one(1.0)), "skpb", gate={"B_hdb": 1.0})  # i >= fb
    l15 = b.at(15)
    l15.heads.append(cc._gather("B's digit for this column", pr.add(pos, {"B_fa": 1.0, "skpb": -1.0}),
                                {"val": {"bvd": 1.0}}, full=True))
    l15.units += pr.times_flag_units({"K": 1.0}, cc._all("c1", "c2", "B_div"), {"aj": 1.0})
    l15.units += pr.times_flag_units({"bvd": 1.0}, cc._all("d_lo", "d_hi", "c1", "c2", "B_div"), {"bj": 1.0})
    # the write: the stored digits become the result's columns
    b.at(16).units += cm.ramp_mod10_units({"Q": 1.0}, 0, 9, {"in_col": 1.0, "B_dfin": 1.0, "one": -1.0}, 1.0, prefix="RA")
    # r - B, exactly (the next K when subtracting)
    b.at(23).units += cc._robust_mod10({"aj": 1.0, "bj": -1.0, "bw1": -1.0, "one": 10.0}, 19, "dK",
                                       {"in_col": 1.0, "B_div": 1.0, "one": -1.0})

    # At the root: subtract when r >= B (no borrow out of the top column) and a digit is down
    l25 = b.at(25)
    l25.units += cc._flag({"dwork": 1.0, "neg1": -1.0, "t1": 1.0, "one": -1.0}, "dsub", "'dsub': r >= B: subtract B")
    shift = {"dwork": 1.0, "dsub": -1.0}
    l26 = b.at(26)
    l26.units += cc._sel(pr.add({"kk": 1.0}, one(1.0)), shift, "nkk")
    l26.units += cc._sel({"kk": 1.0}, {"dsub": 1.0}, "nkk")
    l26.units += cc._sel(pr.add({"qc": 1.0}, one(1.0)), {"dsub": 1.0}, "nqc")

    l29 = b.at(29)
    l29.heads.append(cc._broadcast("division step to every cell", {
        "dsub": {"dsub_B": 1.0}, "dwork": {"dw_B": 1.0}, "At": {"At_B": 1.0}, "qc": {"qc_B": 1.0},
        "SP": {"QS_B": 1.0}, "kk": {"t_B": 1.0}, "t1": {"t1_B": 1.0}}))
    l29.heads.append(cc._gather("remainder digit one column down", pr.add(pos, one(1.0)), {"K": {"Kn": 1.0}}, full=True))
    for t in range(1, 10):  # the shifted digit, rebuilt exactly
        l29.units += cc._step(pr.add({"Kn": 1.0}, one(-(t - 1.0))), "shK")
    dsh = {"dw_B": 1.0, "dsub_B": -1.0}  # a shift pass (0/1)
    l29.units += cc._hat(pr.add(pos, {"QS_B": -1.0, "t_B": -1.0}, one(1.0)), "st_at",
                         gate=pr.add(dsh, {"t1_B": 1.0}, one(-1.0)))  # store qc here

    l30 = b.at(30)
    l30.units += pr.times_flag_units({"dK": 1.0}, {"in_col": 1.0, "dsub_B": 1.0, "one": -1.0}, {"nK": 1.0})
    l30.units += pr.times_flag_units({"shK": 1.0}, pr.add({"in_col": 1.0, "fenceB": -1.0}, dsh, one(-1.0)), {"nK": 1.0})
    l30.units += pr.times_flag_units({"At_B": 1.0}, pr.add({"fenceB": 1.0}, dsh, one(-1.0)), {"nK": 1.0})
    l30.units += pr.times_flag_units({"Q": 1.0}, {"dw_B": 1.0}, {"nQ": 1.0})
    l30.units += pr.times_flag_units({"qc_B": 1.0}, {"st_at": 1.0}, {"nQ": 1.0})
    b.carry("Q", {"nQ": 1.0})
    b.carry("qc", {"nqc": 1.0})
