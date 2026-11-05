"""Long division in latent positions (LOOPED.md §4.4).

``a/b~N`` is floor(A / B) with N fractional digits, A = |a|'s digits followed by fb + N zeros
and B = |b|'s digits followed by fa zeros (the frame-writing calculator's integers). With
L = len(A) and nB = len(B), for each digit t of A (left to right) the model writes

    q_t        the quotient digit floor(x / B), x = 10 r + A_t (r: the remainder so far)
    r_t        nB positions: the new remainder x - q_t B, one digit each, least significant first

and then the quotient row (columns.py). Each position is written by the one before it:

* ``q_t`` by the start (t = 0) or by r_(t-1)'s last digit. It iterates once per column
  j = 0 .. nB of x, comparing x with d * B for every d = 1..9 at once: per d a carry (the
  digits of d * B) and a borrow (x - d * B), kept in the loop registers ``c<d>``, ``br<d>``.
  After column nB, q = the number of d without a borrow out. The router's depth: nB + 1.
* a remainder digit j by q_t (j = 0) or by the digit before it: x_j - (q B)_j - borrow, with
  the product's carry ``mq`` and the borrow ``br`` passed along (x_0 = A_t, x_j = r_(t-1)'s
  digit j - 1, nB + 1 positions back).
"""

from __future__ import annotations

from .. import primitives as pr
from . import parts as P

REGS = ("qd", "qt", "rd", "rn", "mq", "br")
STATE = ("kk", *(f"c{d}" for d in range(1, 10)), *(f"b{d}" for d in range(1, 10)))

DIMS = (
    "PQ", "PR", "rlast", "tlast", "lastR", "tq", "first",
    "jx", "Aok", "skAt", "At_raw", "At", "Bk_lo", "Bk_hi", "skBj", "Bj_raw", "Bj", "jge1", "xr_raw", "x1",
    *(f"m{d}" for d in range(1, 10)), *(f"f{d}" for d in range(1, 10)), *(f"x_c{d}" for d in range(1, 10)),
    *(f"x_b{d}" for d in range(1, 10)), "nbr",
    *(f"qh{v}" for v in range(1, 10)), "qprod", "mR", "flR",
    *(f"n_{r}" for r in REGS), "n_kq", "n_kr",
)


def build(b) -> None:
    pos = {"pos": 1.0}
    one = pr.const
    L = b.at
    neg = lambda f: pr.scale(f, -1.0)  # noqa: E731

    # ------------------------------------------------------------ roles
    L(1).units += P.hat(pr.add({"rn": 1.0, "C_lB": -1.0}), "rlast", "'rlast': the remainder's last digit",
                        gate={"k_r": 1.0})
    L(1).units += P.hat(pr.add({"qt": 1.0, "C_W": -1.0}, one(1.0)), "tlast", "'tlast': the last quotient digit",
                        gate={"k_r": 1.0})
    L(2).units += P.flag(P.all_of("rlast", "tlast"), "lastR", "'lastR': the division is done")
    L(2).units += P.flag(P.all_of("k_s", "C_div"), "PQ", "'PQ': predicts the first quotient digit")
    L(2).units += P.flag(pr.add({"rlast": 1.0, "tlast": -1.0}), "PQ", "'PQ': predicts the next quotient digit")
    L(2).units += P.flag(pr.add({"k_q": 1.0, "k_r": 1.0, "rlast": -1.0}), "PR", "'PR': predicts a remainder digit")
    # first: no remainder yet (the start, or a remainder digit of q_0)
    L(2).units += P.flag({"k_s": 1.0}, "first", "'first': the start: no remainder yet")
    L(3).units += P.hat({"qt": 1.0}, "first", "'first': a digit of r_0: no remainder before it", gate={"PR": 1.0})
    gate = {"PQ": 1.0, "PR": 1.0}  # one of the two

    # tq: the digit of A this position works on (PQ: the next one; PR: its own); j: x's column
    L(3).units += P.sel(pr.add({"qt": 1.0}, one(1.0)), pr.add({"PQ": 1.0, "k_s": -1.0}), "tq")
    L(3).units += P.sel({"qt": 1.0}, {"PR": 1.0}, "tq")
    L(3).units += P.sel({"kk": 1.0}, {"PQ": 1.0}, "jx")
    L(3).units += P.sel({"rn": 1.0}, {"PR": 1.0}, "jx")
    L(3).units += P.gflag({"C_lB": 1.0, "kk": -1.0}, {"PQ": 1.0}, "route", "route: another column of x")

    # A_tq: a's digit tq (left to right, past its '.'), 0 for the zeros after it; only x_0
    L(4).units += P.flag({"jx": 1.0}, "jge1", "'jge1': column j >= 1 (x_j is a remainder digit)")
    L(4).units += P.flag(pr.add({"C_la": 1.0, "tq": -1.0}), "Aok", "'Aok': A's digit tq is one of a's")
    L(4).units += P.gflag(pr.add({"tq": 1.0, "C_ia": -1.0}, one(1.0)), {"C_hda": 1.0}, "skAt", "'skAt': past a's '.'")
    L(5).heads.append(P.gather("A's digit", pr.add({"C_AF": 1.0, "tq": 1.0, "skAt": 1.0}), {"val": {"At_raw": 1.0}}))
    L(6).units += P.count_up({"At_raw": 1.0}, 9, "At", "x_0: A's digit",
                             gate=pr.add(P.all_of("Aok", gate), {"jge1": -1.0}))
    # B_j: 0 for j < fa; else |b|'s digit k = j - fa from the right (past its '.')
    kb = pr.add({"jx": 1.0}, {"C_fa": -1.0})
    L(4).units += P.flag(pr.add(kb, one(1.0)), "Bk_lo", "'Bk_lo': B's digit j is one of b's (j >= fa)")
    L(4).units += P.flag(pr.add({"C_lb": 1.0}, neg(kb)), "Bk_hi", "'Bk_hi': j < len(B)")
    L(4).units += P.gflag(pr.add(kb, {"C_fb": -1.0}, one(1.0)), {"C_hdb": 1.0}, "skBj", "'skBj': left of b's '.'")
    L(5).heads.append(P.gather("B's digit j", pr.add({"C_MB": 1.0, "skBj": -1.0}, neg(kb)), {"val": {"Bj_raw": 1.0}}))
    L(6).units += P.count_up({"Bj_raw": 1.0}, 9, "Bj", "B's digit j", gate=P.all_of("Bk_lo", "Bk_hi", gate))
    # x_j for j >= 1: the remainder so far, digit j - 1 (PQ: at pos - nB + j; PR: pos - nB - 1)
    xr_at = pr.add(pos, {"C_lB": -1.0, "kk": 1.0, "PR": -1.0})
    L(5).heads.append(P.gather("the remainder so far, digit j - 1", xr_at, {"rd": {"xr_raw": 1.0}}))
    L(6).units += P.count_up({"xr_raw": 1.0}, 9, "x1", "x_j (j >= 1)",
                             gate=pr.add(P.all_of("jge1", gate), {"first": -1.0}))
    xj = {"x1": 1.0, "At": 1.0}

    # ------------------------------------------------------------ PQ: compare x with d B, d = 1..9
    for d in range(1, 10):
        m = pr.add({"Bj": float(d)}, {f"c{d}": 1.0})  # (d B)_j and the carry in, 0..89
        L(7).units += P.sel(m, {"PQ": 1.0}, f"m{d}")
        L(7).units += P.tens(m, 89, f"f{d}", f"(d B) carry, d = {d}", gate={"PQ": 1.0})
        diff = pr.add(xj, {f"m{d}": -1.0, f"f{d}": 10.0, f"b{d}": -1.0})  # x_j - (d B)_j - borrow
        L(8).units += P.sel({f"f{d}": 1.0}, {"PQ": 1.0}, f"x_c{d}")
        L(8).units += P.gflag(neg(diff), {"PQ": 1.0}, f"x_b{d}", f"x < d B so far, d = {d}")
        L(9).units += P.gflag({f"x_b{d}": 1.0}, {"PQ": 1.0}, "nbr", f"q < {d}")
    L(8).units += P.sel(pr.add({"kk": 1.0}, one(1.0)), {"PQ": 1.0}, "x_kk")
    # its last iteration: q = 9 - (the d with a borrow out)
    L(10).units += P.sel(pr.add(one(9.0), {"nbr": -1.0}), {"PQ": 1.0}, "n_qd")
    L(10).units += P.sel({"tq": 1.0}, {"PQ": 1.0}, "n_qt")
    L(10).units += P.flag({"PQ": 1.0}, "n_kq", "'n_kq': the next position is a quotient digit")

    # ------------------------------------------------------------ PR: a remainder digit
    L(3).units += P.onehot({"qd": 1.0}, "qh", "the quotient digit", gate={"PR": 1.0}, values=range(1, 10))
    L(7).units += P.onehot_times("qh", {"Bj": 1.0}, {"PR": 1.0}, "qprod", "'qprod' = q * B_j")
    L(8).units += P.sel({"qprod": 1.0, "mq": 1.0}, {"PR": 1.0}, "mR")
    L(8).units += P.tens({"qprod": 1.0, "mq": 1.0}, 89, "flR", "(q B) carry", gate={"PR": 1.0})
    dR = pr.add(xj, {"mR": -1.0, "flR": 10.0, "br": -1.0})  # x_j - (q B)_j - borrow
    L(9).units += P.sel(dR, {"PR": 1.0}, "n_rd")
    L(9).units += P.gflag(neg(dR), {"PR": 1.0}, "n_rd", "remainder: borrow ten", 10.0)
    L(9).units += P.gflag(neg(dR), {"PR": 1.0}, "n_br", "remainder: borrow out")
    L(9).units += P.sel({"flR": 1.0}, {"PR": 1.0}, "n_mq")
    L(9).units += P.sel(pr.add({"rn": 1.0}, one(1.0)), {"PR": 1.0}, "n_rn")
    L(9).units += P.sel({"qd": 1.0}, {"PR": 1.0}, "n_qd")
    L(9).units += P.sel({"qt": 1.0}, {"PR": 1.0}, "n_qt")
    L(9).units += P.flag({"PR": 1.0}, "n_kr", "'n_kr': the next position is a remainder digit")
