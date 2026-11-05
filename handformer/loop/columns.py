"""The working of + - and *: one position per column of the result (LOOPED.md §4.3).

A column position holds its digits and what the next column needs (registers). It is written
by the position before it (the step's start writes column 0), which computes column
``c = nc`` (``nc``: the column after its own; 0 at the start):

* ``+ -``: the digits of |a| and |b| in column c (column j is worth 10^(j-F); a's digit for
  column j sits at ``C_PA - j``, one further right of a '.') and the carry or borrows in:
  dA = a + b + carry (or a - b - borrow) mod 10 and dB = b - a - borrow mod 10, side by side.
  One iteration.
* ``*``: column c of |a| * |b| (digits as integers) is ``cm + sum_i A_i * B_(c-i)``, cm the
  carry in. The position iterates once per term: iteration k adds term i0 + k into a two-part
  accumulator (``slo`` a digit, ``shi`` its tens), and the router sends it round again while
  terms are left. Its last iteration writes the digit and the carry out,
  ``floor(cm / 10) + shi + [cm mod 10 + slo >= 10]`` (floor(cm / 10) by a ruler head: the
  carry is at most 9 per term, so it stays below the position).
* the quotient row (``/``): column c is the digit of the quotient position ``qa``, the
  positions before it holding the quotient digits (division.py) ``len(B) + 1`` apart.

Every column also carries the result's shape: ``tA`` (top nonzero column + 1, 0 if none) and
``lA`` (lowest nonzero column below F, + 1, 0 if none), and the same for the B digits. The last
column (``nc = C_W``) starts the copy (walk.py).
"""

from __future__ import annotations

from .. import primitives as pr
from . import parts as P

REGS = ("nc", "dA", "dB", "cA", "cB", "cm", "tA", "tB", "lA", "lB", "qa")
STATE = ("kk", "slo", "shi")  # a product column's loop registers (the adapter carries them)

DIMS = (
    "PWa", "PWm", "PWq", "lastR", "ltF", "ina1", "ina2", "inb1", "inb2", "av_raw", "bv_raw", "av", "bv",
    "nzA", "nzB", "noA", "noB",
    # products
    "i0", "ic", "skA", "skB", "valid", *(f"mA{v}" for v in range(1, 10)), *(f"qA{v}" for v in range(1, 10)),
    "mB_raw", "mB", "prod", "xs", "fl", "cmh", "yge",
    # the quotient row
    "qaE", "qv_raw",
    *(f"n_{r}" for r in REGS), "n_kw", *(f"x_{s}" for s in STATE),
)


def build(b) -> None:
    pos = {"pos": 1.0}
    one = pr.const
    L = b.at
    G_a, G_m, G_q = {"PWa": 1.0}, {"PWm": 1.0}, {"PWq": 1.0}

    # ------------------------------------------------------------ who predicts a column
    nxt = pr.add({"k_s": 1.0, "k_w": 1.0, "row_end": -1.0})  # the start, or a column that is not the last
    L(2).units += P.flag(pr.add(nxt, {"C_add": 1.0, "C_sub": 1.0}, one(-1.0)), "PWa", "'PWa': predicts a +- column")
    L(2).units += P.flag(pr.add(nxt, {"C_mul": 1.0}, one(-1.0)), "PWm", "'PWm': predicts a product column")
    L(2).units += P.flag(pr.add({"k_w": 1.0, "row_end": -1.0, "C_div": 1.0}, one(-1.0)), "PWq",
                         "'PWq': predicts a quotient-row column")
    L(3).units += P.flag({"lastR": 1.0}, "PWq", "'PWq': the division is done: its first quotient-row column")
    for g in ("PWa", "PWm", "PWq"):
        L(4).units += P.flag({g: 1.0}, "n_kw", f"'n_kw': the next position is a column ({g})")
        L(4).units += P.sel(pr.add({"nc": 1.0}, one(1.0)), {g: 1.0}, "n_nc")

    # ------------------------------------------------------------ + -
    nc = {"nc": 1.0}
    L(1).units += P.flag(pr.add({"C_F": 1.0}, pr.scale(nc, -1.0)), "ltF", "'ltF': column c is right of the '.'")
    for side in ("a", "b"):
        lo, hi = f"C_{side}lo", f"C_{side}hi"
        L(1).units += P.flag(pr.add(nc, {lo: -1.0}, one(1.0)), f"in{side}1", f"'in{side}1': c >= the lowest column of {side}")
        L(1).units += P.flag(pr.add({hi: 1.0}, pr.scale(nc, -1.0), one(1.0)), f"in{side}2", f"'in{side}2': c <= its top column")
    L(2).heads += [
        P.gather("a's digit in column c", pr.add({"C_PA": 1.0, "ltF": 1.0}, pr.scale(nc, -1.0)), {"val": {"av_raw": 1.0}}),
        P.gather("b's digit in column c", pr.add({"C_PB": 1.0, "ltF": 1.0}, pr.scale(nc, -1.0)), {"val": {"bv_raw": 1.0}}),
    ]
    L(3).units += P.count_up({"av_raw": 1.0}, 9, "av", "a's digit", gate=P.all_of("ina1", "ina2", "PWa"))
    L(3).units += P.count_up({"bv_raw": 1.0}, 9, "bv", "b's digit", gate=P.all_of("inb1", "inb2", "PWa"))
    s = {"av": 1.0, "bv": 1.0, "cA": 1.0}  # + : the column's sum
    u = {"av": 1.0, "bv": -1.0, "cA": -1.0}  # - : a - b - borrow
    v = {"bv": 1.0, "av": -1.0, "cB": -1.0}  # - : b - a - borrow
    add, sub = P.all_of("PWa", "C_add"), P.all_of("PWa", "C_sub")
    L(4).units += P.sel(s, add, "n_dA")
    L(4).units += P.gflag(pr.add(s, one(-9.0)), add, "n_dA", "+: the sum's tens leave the column", -10.0)
    L(4).units += P.gflag(pr.add(s, one(-9.0)), add, "n_cA", "+: carry out")
    for x, d_dst, c_dst in ((u, "n_dA", "n_cA"), (v, "n_dB", "n_cB")):
        L(4).units += P.sel(x, sub, d_dst)
        L(4).units += P.gflag(pr.scale(x, -1.0), sub, d_dst, "-: borrow ten", 10.0)
        L(4).units += P.gflag(pr.scale(x, -1.0), sub, c_dst, "-: borrow out")
    _shape(b, 5, "A", G_a)
    _shape(b, 5, "B", G_a)

    # ------------------------------------------------------------ *
    # terms i0 .. i1 with i0 = max(0, c - lb + 1), i1 = min(c, la - 1); iteration kk: i = i0 + kk
    L(1).units.append(P.relu(pr.add(nc, {"C_lb": -1.0}, one(1.0)), "i0", "'i0': the column's first term"))
    L(1).units.append(P.relu(pr.add(nc, {"C_la": -1.0}, one(1.0)), "ic", "'ic': c - i1 (terms past a's top digit)"))
    i = {"i0": 1.0, "kk": 1.0}
    j = pr.add(nc, {"i0": -1.0, "kk": -1.0})
    nterms = pr.add(nc, {"ic": -1.0, "i0": -1.0}, one(1.0))  # i1 - i0 + 1
    L(2).units += P.flag(pr.add(nterms, {"kk": -1.0}), "valid", "'valid': this iteration has a term")
    L(2).units += P.gflag(pr.add(i, {"C_fa": -1.0}, one(1.0)), {"C_hda": 1.0}, "skA", "'skA': A_i is left of a's '.'")
    L(2).units += P.gflag(pr.add(j, {"C_fb": -1.0}, one(1.0)), {"C_hdb": 1.0}, "skB", "'skB': B_j is left of b's '.'")
    L(3).units += P.gflag(pr.add(nterms, {"kk": -1.0}, one(-1.0)), G_m, "route", "route: another term is left")
    L(3).heads += [
        P.gather("A_i", pr.add({"C_MA": 1.0, "skA": -1.0}, pr.scale(i, -1.0)), {f"oh{v}": {f"mA{v}": 1.0} for v in range(1, 10)}),
        P.gather("B_(c-i)", pr.add({"C_MB": 1.0, "skB": -1.0}, pr.scale(j, -1.0)), {"val": {"mB_raw": 1.0}}),
    ]
    for d in range(1, 10):
        L(3).units += P.gflag({f"mA{d}": 1.0}, P.all_of("valid", "PWm"), f"qA{d}", f"A_i is {d}")
    L(3).units += P.count_up({"mB_raw": 1.0}, 9, "mB", "B_(c-i)", gate=P.all_of("valid", "PWm"))
    L(4).units += P.onehot_times("qA", {"mB": 1.0}, G_m, "prod", "'prod' = A_i * B_(c-i)")
    xs = {"slo": 1.0, "prod": 1.0}
    L(5).units += P.sel(xs, {"one": 1.0}, "xs")
    L(5).units += P.tens(xs, 90, "fl", "the accumulator's tens", gate=G_m)
    # next iteration: kk + 1, slo = xs - 10 fl, shi + fl
    L(6).units += P.sel(pr.add({"kk": 1.0}, one(1.0)), G_m, "x_kk")
    L(6).units += P.sel({"xs": 1.0, "fl": -10.0}, G_m, "x_slo")
    L(6).units += P.sel({"shi": 1.0, "fl": 1.0}, G_m, "x_shi")
    # the last iteration: cm + 10 x_shi + x_slo -> digit, carry
    L(1).heads.append(pr.floor_div_head("floor(carry in / 10)", pr.const(100.0), {"cm": 10.0, "one": -45.0}, "cmh"))
    y = {"cm": 1.0, "cmh": -10.0, "x_slo": 1.0}  # cm mod 10 + slo, 0..18
    L(7).units += P.gflag(pr.add(y, one(-9.0)), G_m, "yge", "*: the column's units carry")
    L(8).units += P.sel(pr.add(y, {"yge": -10.0}), G_m, "n_dA")
    L(8).units += P.sel({"cmh": 1.0, "x_shi": 1.0, "yge": 1.0}, G_m, "n_cm")
    _shape(b, 9, "A", G_m)

    # ------------------------------------------------------------ the quotient row (/)
    # qaE: the quotient position for column c (the first: the last remainder row's quotient)
    L(3).units += P.sel({"qa": 1.0}, {"k_w": 1.0}, "qaE")
    L(3).units += P.sel(pr.add(pos, {"C_lB": -1.0}), {"lastR": 1.0}, "qaE")
    L(4).heads.append(P.gather("quotient digit for column c", {"qaE": 1.0}, {"qd": {"qv_raw": 1.0}}))
    L(5).units += P.count_up({"qv_raw": 1.0}, 9, "n_dA", "the quotient digit", gate=G_q)
    L(5).units += P.sel(pr.add({"qaE": 1.0, "C_lB": -1.0}, one(-1.0)), G_q, "n_qa")
    _shape(b, 6, "A", G_q)


def _shape(b, layer: int, k: str, gate: dict) -> None:
    """The next column's shape registers from its digit ``n_d<k>`` (two layers from ``layer``):
    t<k> = c + 1 if the digit is nonzero, else carried; l<k> = c + 1 the first time a nonzero
    digit lands right of the '.', else carried."""
    one = pr.const
    L = b.at
    nz, no = f"nz{k}", f"no{k}"
    L(layer).units += P.gflag({f"n_d{k}": 1.0}, gate, nz, f"'{nz}': the column's digit is not 0")
    L(layer).units += P.hat({f"l{k}": 1.0}, no, f"'{no}': no nonzero digit right of the '.' yet", gate=gate)
    L(layer + 1).units += P.sel({f"t{k}": 1.0}, pr.add(gate, {nz: -1.0}), f"n_t{k}")
    L(layer + 1).units += P.sel(pr.add({"nc": 1.0}, one(1.0)), P.all_of(nz, gate), f"n_t{k}")
    L(layer + 1).units += P.sel({f"l{k}": 1.0}, gate, f"n_l{k}")
    L(layer + 1).units += P.sel(pr.add({"nc": 1.0}, one(1.0)), P.all_of(nz, no, "ltF", gate), f"n_l{k}")
