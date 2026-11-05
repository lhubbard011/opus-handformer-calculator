"""Writing the next copy, one position at a time, and the answer (LOOPED.md §4.2).

Every position predicts the next one: its byte (the hold dims) and, for a thought, the
registers it starts with (``n_<reg>``, which T moves to ``<reg>``). A copy is written by a walk
over the old copy, carried from position to position in three registers:

    zone   z_pre (before the replaced bytes), z_mid (their replacement); neither: after them
    ptr    the old position the cell copied (pre and suf)
    mk     the cell's index in the replacement (mid)

The walk starts at the position before the copy (the step's start, or the last column of a
working) with ``ptr = C_ofz``, the old copy's start, in the pre zone. Each position:

    ptr1 = the first non-space old position after ptr (ptr + 1 when there are no spaces)
    pre:  ptr1 = C_lo                 -> mid, mk 0
    mid:  mk + 1 = the replacement's length -> suf, ptr C_hi + 1; else mk + 1
    pre, suf: the old copy ends (ptr1 = C_X) -> the next root '='

and copies the old byte at the new ptr, or writes the replacement's byte mk (``_mid``):

    brackets  the group's number, copied (old position C_mlo + mk)
    power     the binary-exponentiation template (power.py)
    number    the working's result: sign, integer digits, '.', fractional digits

A number's shape comes from the working's last column (the copy's start, ``fz``): the top
nonzero column and the lowest nonzero one below F, for each digit set, and the borrow out.

The answer: a written byte k after the root copies the copy's cell k (the root itself ends it
with the end byte). The root that halts writes the first one.
"""

from __future__ import annotations

from .. import common as cm
from .. import primitives as pr
from ..model import Form
from . import parts as P

ZONES = ("z_pre", "z_mid")  # after the replacement: neither
REGS = (*ZONES, "ptr", "mk")  # a copy cell's registers
SHAPE = ("tA", "tB", "lA", "lB", "cA")  # a working column's shape registers (columns.py)

DIMS = (
    "wk", "ws", "wt", "ans", "row_end", "ptrE", "preE", "afz",
    *(f"s_{r}" for r in SHAPE), "useB", "top1", "low1", "nzr", "hlow", "hiC", "loC", "negr", "cancel", "sgn",
    "dot", "nint", "mlen", "lo_eff", "ptr1", "e_mid", "in_mid", "x_mid", "endp", "to_root", "to_mid", "to_suf",
    "ptrN", "mkN", "mk_dot", "mk_frac", "dg_addr",
    "cp_t", "cp_on", *(f"cpb_{d}" for d in (*P.CONTENT.values(), "is_eq")),
    "dg_v", "dg_on", "dgA", "dgB",
    *(f"n_{r}" for r in REGS),
)


def build(b) -> None:
    pos = {"pos": 1.0}
    one = pr.const
    L = b.at

    # ------------------------------------------------------------ roles
    # wk: a cell of a copy being written (a latent cell); ws: the position before a copy
    L(1).units += P.flag(P.all_of("lat", "is_cell"), "wk", "'wk': a cell of the copy being written")
    L(1).units += P.flag(pr.add({"k_s": 1.0, "C_sp": 1.0, "C_br": 1.0, "C_pow": 1.0}, one(-1.0)), "ws",
                         "'ws': the step's start begins a copy (spaces, brackets, power)")
    L(1).units += P.hat(pr.add({"nc": 1.0, "C_W": -1.0}), "row_end", "'row_end': the working's last column",
                        gate={"k_w": 1.0})
    L(2).units += P.flag({"row_end": 1.0}, "ws", "'ws': the working's last column begins the copy")
    L(3).units += P.flag({"wk": 1.0, "ws": 1.0}, "wt", "'wt': this position walks (it predicts a copy cell)")
    # ptrE, preE: the walk's state going in (a walk's start: the old copy's start, pre zone)
    L(3).units += P.sel({"ptr": 1.0}, {"wk": 1.0}, "ptrE")
    L(3).units += P.sel({"C_ofz": 1.0}, {"ws": 1.0}, "ptrE")
    L(3).units += P.flag({"z_pre": 1.0, "ws": 1.0}, "preE", "'preE': in the pre zone (or starting)")

    # ------------------------------------------------------------ the number's shape (from fz)
    # the position before this copy is the latest non-cell at or before here (Lnc)
    L(3).heads.append(P.gather("the copy's start: its shape", {"Lnc": 1.0}, {r: {f"s_{r}": 1.0} for r in SHAPE}))
    L(4).units += P.flag(P.all_of("C_sub", "s_cA"), "useB", "'useB': |a| < |b|: the result is b - a")
    for dst, (a_src, b_src) in (("top1", ("s_tA", "s_tB")), ("low1", ("s_lA", "s_lB"))):
        L(5).units += P.sel({a_src: 1.0}, pr.add({"useB": -1.0}, one(1.0)), dst)
        L(5).units += P.sel({b_src: 1.0}, {"useB": 1.0}, dst)
    # negr: the result's sign (+: a's; -: a's XOR the borrow; * /: a's XOR b's)
    L(5).units += P.flag(P.all_of("C_add", "C_sa"), "negr", "'negr' for +: a's sign")
    L(5).units += P.hat(pr.add({"C_sa": 1.0, "useB": 1.0}, one(-1.0)), "negr", "'negr' for -: a's sign XOR |a| < |b|",
                        gate={"C_sub": 1.0})
    L(5).units += P.flag(pr.add({"C_mul": 1.0, "C_div": 1.0, "C_sx": 1.0}, one(-1.0)), "negr",
                         "'negr' for * /: exactly one operand negative")
    L(6).units += P.flag({"top1": 1.0}, "nzr", "'nzr': the result is not zero")
    L(6).units += P.flag({"low1": 1.0}, "hlow", "'hlow': a nonzero column below F")
    # hiC = max(top, F) with top = top1 - 1; loC = the lowest nonzero column below F, else F
    L(6).units.append(P.relu({"C_F": 1.0}, "hiC", "'hiC' base: F (the units column)"))
    L(6).units.append(P.relu(pr.add({"top1": 1.0, "C_F": -1.0}, one(-1.0)), "hiC", "'hiC': the top nonzero column"))
    L(7).units.append(P.relu({"C_F": 1.0}, "loC", "'loC' base: F"))
    L(7).units += P.sel(pr.add({"C_F": 1.0, "low1": -1.0}, one(1.0)), {"hlow": 1.0}, "loC", -1.0)
    L(7).units += P.sel(pr.add({"hiC": 1.0, "C_F": -1.0}, one(1.0)), {"one": 1.0}, "nint")
    L(8).units += P.flag(P.all_of("negr", "nzr", "C_ulo"), "cancel", "'cancel': -(-v): the sign before goes too")
    L(8).units += P.flag(pr.add(P.all_of("negr", "nzr"), {"C_ulo": -1.0}), "sgn", "'sgn': the number starts with '-'")
    L(8).units += P.flag({"C_F": 1.0, "loC": -1.0}, "dot", "'dot': the number has a '.'")
    # the replacement's length: brackets its number; power C_mp; a number sgn + nint + dot + frac
    L(9).units += P.sel(pr.add({"C_mhi": 1.0, "C_mlo": -1.0}, one(1.0)), {"C_br": 1.0}, "mlen")
    L(9).units += P.sel({"C_mp": 1.0}, {"C_pow": 1.0}, "mlen")
    L(9).units += P.sel({"sgn": 1.0, "nint": 1.0, "dot": 1.0, "C_F": 1.0, "loC": -1.0}, {"C_num": 1.0}, "mlen")
    L(9).units += P.sel({"C_lo": 1.0, "cancel": -1.0}, {"one": 1.0}, "lo_eff")

    # ------------------------------------------------------------ the walk
    # ptr1: the first non-space after ptrE (lt = ptrE: itself a non-space)
    # (the start byte is not a match: with ptrE = 0 it would tie with position 1)
    L(5).heads.append(pr.next_match_head("next old byte", "PM_Lns", "PM2_Lns",
                                         {"one": 1.0, "is_space": -1.0, "is_bos": -1.0}, {"ptrE": 1.0}, "ptr1"))
    L(10).units += P.hat(pr.add({"ptr1": 1.0, "lo_eff": -1.0}), "e_mid",
                         "'e_mid': the next old byte is the first replaced one", gate={"preE": 1.0})
    L(10).units += P.hat(pr.add({"mk": 1.0, "mlen": -1.0}, one(1.0)), "x_mid", "'x_mid': the replacement is done",
                         gate={"z_mid": 1.0})
    L(11).units += P.flag({"z_mid": 1.0, "x_mid": -1.0}, "in_mid", "'in_mid': the next cell is still in the replacement")
    # ptrN: the next old byte (pre and suf: ptr1; leaving the replacement: C_hi + 1)
    L(11).units += P.sel({"ptr1": 1.0}, pr.add({"wt": 1.0, "z_mid": -1.0}), "ptrN")
    L(11).units += P.sel(pr.add({"C_hi": 1.0}, one(1.0)), {"x_mid": 1.0}, "ptrN")
    L(12).units += P.hat(pr.add({"ptrN": 1.0, "C_X": -1.0}), "endp", "'endp': the old copy ends there",
                         gate=pr.add({"wt": 1.0, "in_mid": -1.0, "e_mid": -1.0}))
    L(13).units += P.flag({"endp": 1.0}, "to_root", "'to_root': the next position is the copy's root")
    L(13).units += P.flag(pr.add({"in_mid": 1.0, "e_mid": 1.0}), "to_mid", "'to_mid': the next cell is in the replacement")
    L(13).units += P.flag(pr.add({"wt": 1.0, "endp": -1.0, "in_mid": -1.0, "e_mid": -1.0}), "to_suf",
                          "'to_suf': the next cell copies an old byte")
    L(13).units += P.sel(pr.add({"mk": 1.0}, one(1.0)), {"in_mid": 1.0}, "mkN")  # entering: mk 0
    # the next cell's registers
    L(14).units += P.flag(pr.add({"to_suf": 1.0, "preE": 1.0}, one(-1.0)), "n_z_pre",
                          "'n_z_pre': the next cell is still before the replacement")
    L(14).units += P.flag({"to_mid": 1.0}, "n_z_mid", "'n_z_mid': the next cell is in the replacement")
    L(14).units += P.sel({"ptrN": 1.0}, {"to_suf": 1.0}, "n_ptr")
    L(14).units += P.sel({"mkN": 1.0}, {"to_mid": 1.0}, "n_mk")
    # the next byte: an old byte (copied from ptrN), the root, or the replacement's byte mkN
    L(14).units += P.sel({"ptrN": 1.0}, {"to_suf": 1.0}, "cp_t")
    L(14).units += P.flag({"to_suf": 1.0}, "cp_on", "'cp_on': copy the byte at cp_t")
    L(14).units += P.flag({"to_root": 1.0}, P.HOLD[ord("=")], "the next position is the root '='")
    _mid(b)

    # ------------------------------------------------------------ the answer
    # afz: the latest root's copy start; the k-th written byte copies cell afz + 1 + k
    L(3).heads.append(P.gather("the root's copy start", pr.add({"E": 1.0}, one(-1.0)), {"Lnc": {"afz": 1.0}}))
    L(14).units += P.flag(P.all_of("is_eq", "halt"), "ans", "'ans': the root halts: write the first byte")
    L(14).units += P.flag(pr.add({"lat": -1.0, "is_eq": -1.0}, one(1.0)), "ans", "'ans': a written byte: write the next")
    L(15).units += P.sel(pr.add({"afz": 1.0, "E": -1.0}, pos, one(1.0)), {"ans": 1.0}, "cp_t")
    L(15).units += P.flag({"ans": 1.0}, "cp_on", "'cp_on': copy the answer's next byte")
    L(15).units += P.flag(pr.add({"ans": -1.0}, one(1.0)), "think", "'think': the next position is a thought")

    # ------------------------------------------------------------ the shared writer
    L(17).heads.append(P.gather("the byte to copy", {"cp_t": 1.0},
                                {d: {f"cpb_{d}": 1.0} for d in (*P.CONTENT.values(), "is_eq")}))
    for byte, d in P.CONTENT.items():
        L(17).units += P.flag(P.all_of(f"cpb_{d}", "cp_on"), P.HOLD[byte], f"copy '{chr(byte)}'")
    L(17).units += P.flag(P.all_of("cpb_is_eq", "cp_on", "ans"), P.HOLD[cm.EOS], "the answer ends at the root")
    L(17).units += P.hold_digit({"dg_v": 1.0}, {"dg_on": 1.0}, "write the digit")


def _mid(b) -> None:
    """The replacement's byte mkN (``to_mid``): a bracket's number is copied, a result's digits
    come from the working (power.py writes a power's)."""
    one = pr.const
    L = b.at
    L(14).units += P.sel(pr.add({"C_mlo": 1.0, "mkN": 1.0}), P.all_of("to_mid", "C_br"), "cp_t")
    L(14).units += P.flag(P.all_of("to_mid", "C_br"), "cp_on", "'cp_on': copy the group's number")
    # a number: mkN < sgn: '-'; then the integer digits; '.' at sgn + nint; then the fraction
    L(14).units += P.hat({"mkN": 1.0}, P.HOLD[ord("-")], "the number's '-'", gate=P.all_of("to_mid", "C_num", "sgn"))
    L(14).units += P.hat(pr.add({"mkN": 1.0, "sgn": -1.0, "nint": -1.0}), "mk_dot", "'mk_dot': the '.'",
                         gate=P.all_of("to_mid", "C_num", "dot"))
    L(14).units += P.flag(pr.add({"mkN": 1.0, "sgn": -1.0, "nint": -1.0}), "mk_frac", "'mk_frac': past the '.'")
    L(15).units += P.flag({"mk_dot": 1.0}, P.HOLD[ord(".")], "the number's '.'")
    num = P.all_of("to_mid", "C_num")
    L(15).units += P.flag(pr.add(num, {"mk_dot": -1.0, "sgn": -1.0}), "dg_on", "'dg_on': a digit (no sign)")
    L(15).units += P.gflag({"mkN": 1.0, "mk_dot": -pr.BIG}, P.all_of("to_mid", "C_num", "sgn"),
                           "dg_on", "'dg_on': a digit (after the sign)")
    # column c = hiC + sgn - mkN (+1 past the '.'); its cell: fz - W + 1 + c
    L(15).units += P.sel(pr.add({"Lnc": 1.0, "C_W": -1.0, "hiC": 1.0, "sgn": 1.0, "mkN": -1.0, "mk_frac": 1.0},
                                one(1.0)), {"one": 1.0}, "dg_addr")
    L(16).heads.append(P.gather("the result's digit", {"dg_addr": 1.0}, {"dA": {"dgA": 1.0}, "dB": {"dgB": 1.0}}))
    L(16).units += P.count_up({"dgA": 1.0}, 9, "dg_v", "the digit (set A)", gate=pr.add(num, {"useB": -1.0}))
    L(16).units += P.count_up({"dgB": 1.0}, 9, "dg_v", "the digit (set B)", gate=P.all_of(num, "useB"))
