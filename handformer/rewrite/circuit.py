"""The recurrent block: one pass rewrites the whole tape in place (REWRITE.md §4).

Tape: position 0 is the start byte, the last position is the root (the prompt's '='), and the
cells between hold the expression, then the prompt's "~N" if any, then blanks (spaces). One
pass does one of these, decided at the root and broadcast to every cell:

  act_sp   a space has content after it: remove the first one (everything after moves left)
  act_br   the innermost group holds one number: drop the brackets ("-(-v)": the signs too)
  act_red  the redex (innermost group, precedence, left to right): + and - in one pass; * one
           pass per digit of b (an accumulator the adapter carries), then one that writes
  halt     none of these (the expression is a number): the router sends no cell round again

Every write is "replace cells lo..hi by m bytes": cells before lo keep their byte, the m result
cells take the result, and the cells after take the byte g = hi - lo + 1 - m cells to their
right (a blank once that would be the root).

Numbers are columns. Column j of a result is worth 10^(j-F) (F fractional digits: max(fa, fb)
for + and -, fa + fb for *) and lives at cell b_hi-1-j, the redex's own cells. Every column cell
reads its digits of a and b, then its carry or borrow in one next-match lookup: the first lower
column whose sum is not 9 (difference not 0) decides it. a+b, a-b and b-a are computed side by
side; for *, a pass adds |a| * b_k * 10^k into the column accumulators K (two levels: the
digit products' tens go one column up, then one binary carry chain).

Layers (heads read the layer input; the MLP also sees this layer's heads):
  1-13   parse, at the root (causal reads: the root sees the whole tape)
  14     broadcast 1 (bidirectional): the redex's fences, offsets and mode reach every cell
  14-23  columns: digits, products, the carry/borrow chain, result digits, the next K
  24-28  at the root: the result's shape (top and bottom nonzero column, sign, cancel, '.')
  29     broadcast 2: every cell learns its zone and the router bit
  30     each cell gathers its next byte and writes it (hold dims, re-embedded by the adapter)
"""

from __future__ import annotations

from .. import common as cm
from .. import primitives as pr
from ..model import Form, HeadSpec
from .engine import BLANK, LoopBuilder

# Bytes a tape cell can hold, and the input dim that flags each (the state's byte features).
CONTENT = {**cm.ALPHABET, ord("~"): "is_tilde", BLANK: "is_space"}
FIXED = {cm.BOS: "is_bos", ord("="): "is_eq"}  # the start byte and the root never change
HOLD = {byte: f"H_{dim}" for byte, dim in {**CONTENT, **FIXED}.items()}  # "hold this byte next"
# What the adapter carries besides the byte: the column accumulator (a product's columns, or a
# division's remainder), the root's digit counter, a division's stored quotient digits and the
# count of the digit it is working on.
CARRIED = ("K", "kk", "Q", "qc")

BINMINUS: Form = {"is_minus": 1.0, "prev_num": 1.0, "prev_rpar": 1.0, "one": -1.0}
MARKERS: dict[str, list[Form]] = {
    "Lg": [{"is_lpar": 1.0}],
    "Lrp": [{"is_rpar": 1.0}],
    "Lmul": [{"is_star": 1.0}, {"is_slash": 1.0}],
    "Lpow": [{"is_caret": 1.0}],
    "Ladd": [{"is_plus": 1.0}, BINMINUS],
    "Lnd": [{"one": 1.0, "is_digit": -1.0, "is_dot": -1.0}],  # not part of a number
    "Ldot": [{"is_dot": 1.0}],
    "Lnz": [{"nz": 1.0}],  # a nonzero digit
    "Ptil": [{"is_tilde": 1.0}],  # the precision marker
    "P": [{"is_plus": 1.0}, {"is_star": 1.0}, {"is_slash": 1.0}, {"is_caret": 1.0}, BINMINUS],
    "Lc": [{"one": 1.0, "is_space": -1.0, "is_eq": -1.0, "is_bos": -1.0}],  # a cell with content
}
NEXT = ("Lrp", "Lmul", "Ladd", "Lnd")  # classes searched forward (next_match)
CHAINS = ("A", "S")  # carry chain (a+b, and the products of *), borrow chain (a-b and b-a)
R_SETS = ("A", "B")  # result digits: RA (a+b, a-b, a*b), RB (b-a)

DIMS = (
    "prev_num", "prev_rpar", "is_unary", "is_binminus", "route", "grow",
    *(f"f_{m}" for m in MARKERS), *MARKERS, "f_X", "X", "f_SP", "SP", "act_sp",
    *(f"PM_{c}" for c in NEXT), "p", *(f"PM2_{c}" for c in NEXT), "Lt_Lrp", "Lt_Lmul", "Lt_Ladd",
    "c_raw", "fc", "om", "fm", "oa", "c", "opw", "Pg", "use_mul", "pw", "bin_in", "gp",
    "o", "act_red", "act_br",
    # operands
    "a_lo", "sb_x", "g_plus", "g_minus", "g_star", "g_caret", "g_slash", "opc", "opdiv", "CB", "B_pmr", "rwn", "rwp", "grow_any", "u_p", "sneg_p", "a_dot", "opp", "opm", "opmul", "hd_a",
    "sa_x", "b_hi", "e1", "e2", "sub", "fa", "cb", "b_dot", "drop", "hd_b", "m_add", "m_sub", "m_mul",
    "fb", "F", "Wm", "final", "sk_b", "bk", "macc", "offA", "offB", "offM", "la", "u_lo", "nkk",
    # columns
    "B_bhi", "B_W", "B_F", "B_fa", "B_ia", "B_fb", "B_ib", "B_la", "B_hda", "B_kk", "B_bk", "B_offA", "B_offB",
    "B_offM", "B_red", "B_add", "B_sub", "B_mul", "B_macc",
    "c1", "c2", "fence", "fenceB", "ge0", "ha_lo", "ha_hi", "hb_lo", "hb_hi",
    "v1_lo", "v1_hi", "v2_lo", "v2_hi", "skp1", "skp2",
    "av", "bv", *(f"A1_{d}" for d in range(1, 10)), *(f"A2_{d}" for d in range(1, 10)),
    "in_col", "aj", "bj", "P1", "P2", *(f"L1_{d}" for d in range(1, 10)), *(f"L2_{d}" for d in range(1, 10)),
    *(f"DU{d}" for d in range(1, 10)), "eprev",
    "ge10", "le8", "gt", "lt", *(f"f_m{k}" for k in CHAINS), *(f"Lm{k}" for k in CHAINS),
    *(f"PMm{k}" for k in CHAINS), *(f"PM2m{k}" for k in CHAINS), "cy", "bw1", "bw2",
    *(f"R{k}{d}" for k in R_SETS for d in range(10)), "nK",
    # the result's shape
    *(f"f_t{k}" for k in R_SETS), *(f"f_b{k}" for k in R_SETS), *(f"top{k}" for k in R_SETS),
    *(f"bot{k}" for k in R_SETS), *(f"z{k}" for k in R_SETS), "neg1", "ltm",
    "topS", "botS", "zero", "sgnw", "cancel", "ni", "Fp", "hasfrac",
    "LO", "M", "G", "DS", "CS", "NI", "act_go",
    # the write
    "go", "lo_B", "m_B", "g_B", "ds_B", "cs_B", "sg_B", "ni_B", "hf_B", "ltm_B", "red_B", "br_B", "R_B",
    "gelo", "gelo2", "ltend", "blank", "atdot", "afterdot",
    *(f"S_{d}" for d in CONTENT.values()), *(f"C_{d}" for d in CONTENT.values()),
    *(f"G{k}{d}" for k in R_SETS for d in range(10)),
)


AF: Form = {"a_lo": 1.0, "sa_x": -1.0}  # the byte before signed a
LO0: Form = {"a_lo": 1.0, "sa_x": -1.0, "drop": -1.0, "one": 1.0}  # first replaced byte (before a cancel)
RED_HI: Form = {"b_hi": 1.0, "drop": 1.0, "one": -1.0}  # last replaced byte


def _gather(name: str, target: Form, value: dict[str, Form], full: bool = False) -> HeadSpec:
    """``gather_head`` with any value map (one or many dims moved from the byte at ``target``)."""
    return HeadSpec(name=name, query=[pr.scale(target, 2 * pr.GATHER_BETA), pr.const(-pr.GATHER_BETA)],
                    key=[{"pos": 1.0}, {"pos2": 1.0}], value=value, full=full)


# The root's margin in a broadcast. Every other cell holds its own copy of the parse, computed
# from what it can see, and those values are unbounded garbage (a gated position can sit near
# BIG^2). A 60-logit margin leaks e^-60 of them, which is visible; at 700 the leak underflows
# to exactly 0 in float64 (e^-700 * 1e13 is still below 1e-290).
BROADCAST_BETA = 700.0


def _broadcast(name: str, value: dict[str, Form]) -> HeadSpec:
    """Every cell reads the root (the one cell holding '='): bidirectional, the root wins by 700."""
    return HeadSpec(name=name, query=[{"one": BROADCAST_BETA}], key=[{"is_eq": 1.0}], value=value, full=True)


def _step(x: Form, dst: str, sign: float = 1.0, gate: Form | None = None) -> list:
    """``dst += sign * [x >= 1]`` (robust step), silenced unless ``gate`` (0/1 form) is 1."""
    if gate is not None:
        x = pr.gated(x, gate)
    return pr.clamp01_units(x, {dst: sign})


def _hat(x: Form, dst: str, gate: Form | None = None) -> list:
    """``dst += [x == 0]``, silenced unless ``gate`` is 1."""
    return [*_step(pr.add(x, pr.const(1.0)), dst, 1.0, gate), *_step(x, dst, -1.0, gate)]


def _all(*flags: str) -> Form:
    """A 0/1 form: 1 when every flag is 1, <= 0 otherwise (gate or AND input)."""
    return pr.add(*({f: 1.0} for f in flags), pr.const(1.0 - len(flags)))


def _relu(x: Form, dst: str, label: str, k: float = 1.0):
    return pr.relu_unit(x, {dst: k}, label)


def _flag(x: Form, dst: str, label: str) -> list:
    """``dst += [x >= 1]`` as a robust step (exact 0/1 for x within 1/4 of an integer), labelled.
    Every flag here is one: a gather leaks ~1e-13, and a flag that gates by BIG must not."""
    units = pr.clamp01_units(x, {dst: 1.0})
    for u, part in zip(units, ("rises at 1/2", "caps at 1")):
        u.label = f"{label} ({part})"
    return units


def _parity3(x: Form, dst: str, label: str, gate: Form) -> list:
    """``dst = x mod 2`` for x in {0, 1, 2, 3}: the XOR of three flags, as robust steps
    ([x == 1] + [x == 3]; a ReLU parity would pass a gather's leak through)."""
    units = [*_hat(pr.add(x, pr.const(-1.0)), dst, gate), *_hat(pr.add(x, pr.const(-3.0)), dst, gate)]
    for u in units:
        u.label = label
    return units


def _sel(x: Form, flag: Form, dst: str) -> list:
    """``dst += x * flag`` (signed x, 0/1 flag form)."""
    return pr.times_flag_units(x, flag, {dst: 1.0})


def _robust_mod10(x: Form, hi: int, dst: str, gate: Form) -> list:
    """``dst += x mod 10`` for integer x in 0..hi (< 20), exactly: a sum of robust steps,
    sum_t [x >= t] - 10 [x >= 10]. Feeds the adapter, so it must not carry a ramp's leak."""
    units = []
    for t in range(1, hi + 1):
        units += _step(pr.add(x, pr.const(-(t - 1.0))), dst, 1.0, gate)
    units += _step(pr.add(x, pr.const(-9.0)), dst, -10.0, gate)
    return units


def _onehot_times(oh: str, scalar: Form, gate: Form, dst: str, label: str) -> list:
    """``dst = v * scalar`` where the one-hot ``oh<v>`` is set (0 <= v*scalar <= 81): one unit per
    v, silenced by 100 (more than any product) unless that v is the one and ``gate`` holds."""
    return [_relu(pr.add(pr.scale(scalar, float(v)), pr.scale({f"{oh}{v}": 1.0}, 100.0),
                         pr.scale(gate, 100.0), pr.const(-200.0)), dst, label) for v in range(1, 10)]


def build(b: LoopBuilder) -> None:
    pos = {"pos": 1.0}
    prev = pr.add(pos, pr.const(-1.0))
    neg = lambda f: pr.scale(f, -1.0)  # noqa: E731
    one = pr.const

    # ------------------------------------------------------------ parse (every cell computes it;
    # the root's values are the ones read: under the causal mask only it sees the whole tape)
    l1 = b.at(1)
    l1.heads += [
        _gather("prev byte in a number?", prev, {"is_digit": {"prev_num": 1.0}, "is_dot": {"prev_num": 1.0}}),
        _gather("prev byte ')'?", prev, {"is_rpar": {"prev_rpar": 1.0}}),
    ]
    l1.units.append(_relu({"is_minus": 1.0, "prev_num": -1.0, "prev_rpar": -1.0}, "is_unary",
                          "'is_unary': a '-' not after a number or ')' is a sign"))
    l1.units.append(_relu(BINMINUS, "is_binminus", "'is_binminus': a '-' after a number or ')' subtracts"))
    for m, matches in MARKERS.items():
        for k, match in enumerate(matches):
            l1.units += pr.latest_feature(match, f"f_{m}", bos=(k == 0))
    # X: where the expression ends (the first '~', blank, or the root); SP: the first blank or the root
    l1.units.append(pr.earliest_if({"is_tilde": 1.0, "is_space": 1.0, "is_eq": 1.0}, 1.0, "f_X"))
    l1.units.append(pr.earliest_if({"is_space": 1.0, "is_eq": 1.0}, 1.0, "f_SP"))

    l2 = b.at(2)
    l2.heads += [pr.latest_head(f"latest {m}", f"f_{m}", m) for m in MARKERS]
    l2.heads += [pr.latest_head("expression end", "f_X", "X"), pr.latest_head("first blank", "f_SP", "SP")]
    # act_sp = [Lc > SP]: content after the first blank (at the root: anywhere on the tape)
    l2.units += _flag({"Lc": 1.0, "SP": -1.0}, "act_sp", "'act_sp': a space has content after it")

    l3 = b.at(3)
    l3.heads += [_gather(f"prev {c}", prev, {c: {f"PM_{c}": 1.0}}) for c in NEXT]
    # p = latest '(' before X: the innermost group's '(' (0, the start byte, if none)
    l3.heads.append(_gather("innermost group", pr.add({"X": 1.0}, one(-1.0)), {"Lg": {"p": 1.0}}))

    l4 = b.at(4)
    l4.heads += [_gather(f"pm^2 {c}", {f"PM_{c}": 1.0}, {"pos2": {f"PM2_{c}": 1.0}}) for c in NEXT]
    l4.heads += [_gather(f"latest {c} at p", {"p": 1.0}, {c: {f"Lt_{c}": 1.0}}) for c in ("Lrp", "Lmul", "Ladd")]

    l5 = b.at(5)
    l5.heads += [
        pr.next_match_head("first ) after p", "PM_Lrp", "PM2_Lrp", {"is_rpar": 1.0}, {"Lt_Lrp": 1.0}, "c_raw", "fc"),
        pr.next_match_head("first * or / after p", "PM_Lmul", "PM2_Lmul", {"is_star": 1.0, "is_slash": 1.0},
                           {"Lt_Lmul": 1.0}, "om", "fm"),
        pr.next_match_head("first +- after p", "PM_Ladd", "PM2_Ladd", {"is_plus": 1.0, "is_binminus": 1.0},
                           {"Lt_Ladd": 1.0}, "oa"),
    ]
    # c = min(first ')' after p, X)
    l5.units.append(_relu({"X": 1.0}, "c", "'c' base: X, where the expression ends"))
    l5.units.append(_relu(pr.add({"X": 1.0, "c_raw": -1.0, "fc": pr.BIG}, one(-pr.BIG)), "c",
                          "'c' correction: min with the first ')' after p, when there is one", -1.0))

    l6 = b.at(6)
    before_c = pr.add({"c": 1.0}, one(-1.0))
    l6.heads += [_gather("latest ^ before c", before_c, {"Lpow": {"opw": 1.0}}),
                 _gather("latest operator before c", before_c, {"P": {"Pg": 1.0}})]
    l6.units += _step(pr.add({"c": 1.0, "om": -1.0, "fm": pr.BIG}, one(-pr.BIG)), "use_mul")
    l6.units += _step({"opw": 1.0, "p": -1.0}, "pw")
    l6.units += _step({"Pg": 1.0, "p": -1.0}, "bin_in")  # a binary operator inside the group
    l6.units += _step({"p": 1.0}, "gp")  # the group is a real '('

    l7 = b.at(7)
    l7.units += [
        _relu({"oa": 1.0, "use_mul": -pr.BIG, "pw": -pr.BIG}, "o", "'o': first + or - (no ^, no * or / in the group)"),
        _relu({"om": 1.0, "use_mul": pr.BIG, "pw": -pr.BIG, "one": -pr.BIG}, "o", "'o': first * or /"),
        _relu({"opw": 1.0, "pw": pr.BIG, "one": -pr.BIG}, "o", "'o': latest ^ (right-associative)"),
    ]
    l7.units += _flag({"bin_in": 1.0, "act_sp": -1.0}, "act_red", "'act_red': the group has an operator and no space is left")
    l7.units += _flag({"gp": 1.0, "bin_in": -1.0, "act_sp": -1.0}, "act_br",
                      "'act_br': the innermost group holds one number: drop its brackets")

    # ------------------------------------------------------------ operands: a_lo | sign? a | o | sign? b | b_hi
    l8 = b.at(8)
    before_o = pr.add({"o": 1.0}, one(-1.0))
    l8.heads += [
        _gather("a fence", before_o, {"Lnd": {"a_lo": 1.0}}),
        _gather("a's '.'", before_o, {"Ldot": {"a_dot": 1.0}}),
        _gather("sign of b?", pr.add({"o": 1.0}, one(1.0)), {"is_minus": {"sb_x": 1.0}}),
        _gather("operator", {"o": 1.0}, {"is_plus": {"g_plus": 1.0}, "is_minus": {"g_minus": 1.0},
                                         "is_star": {"g_star": 1.0}, "is_caret": {"g_caret": 1.0},
                                         "is_slash": {"g_slash": 1.0}}),
        _gather("sign before the group?", pr.add({"p": 1.0}, one(-1.0)), {"is_unary": {"u_p": 1.0}}),
        _gather("group's number negative?", pr.add({"p": 1.0}, one(1.0)), {"is_minus": {"sneg_p": 1.0}}),
    ]
    l8.units += _flag({"g_plus": 1.0}, "opp", "'opp': the redex adds")
    l8.units += _flag({"g_minus": 1.0}, "opm", "'opm': the redex subtracts")
    l8.units += _flag({"g_star": 1.0}, "opmul", "'opmul': the redex multiplies")
    l8.units += _flag({"g_slash": 1.0}, "opdiv", "'opdiv': the redex divides (division.py)")
    l8.units += _flag({"g_caret": 1.0}, "opc", "'opc': the redex is a power (a text rewrite, power.py)")
    l8.units += _flag({"a_dot": 1.0, "a_lo": -1.0}, "hd_a", "'hd_a': a has a '.'")

    l9 = b.at(9)
    l9.heads += [
        _gather("sign of a?", {"a_lo": 1.0}, {"is_unary": {"sa_x": 1.0}}),
        pr.next_match_head("b fence", "PM_Lnd", "PM2_Lnd", {"one": 1.0, "is_digit": -1.0, "is_dot": -1.0},
                           {"o": 1.0, "sb_x": 1.0}, "b_hi"),
    ]
    af = {"a_lo": 1.0, "sa_x": -1.0}  # the byte before signed a
    l9.units += _hat(pr.add(af, {"p": -1.0}), "e1")  # a starts right after the '('
    l9.units += _hat({"b_hi": 1.0, "c": -1.0}, "e2")  # b ends at the ')'
    # sub: + or -, and the signs of a and of b as added (b's sign XOR the '-') differ
    l9.units += _parity3({"sa_x": 1.0, "sb_x": 1.0, "opm": 1.0}, "sub", "'sub': the magnitudes are subtracted",
                         gate={"opp": 1.0, "opm": 1.0})
    l9.units += _sel({"o": 1.0, "a_dot": -1.0, "one": -1.0}, {"hd_a": 1.0}, "fa")  # a's fractional digits
    l9.units += _flag(_all("act_br", "u_p", "sneg_p"), "cb", "'cb': -(-v): both signs go with the brackets")

    l10 = b.at(10)
    l10.heads.append(_gather("b's '.'", pr.add({"b_hi": 1.0}, one(-1.0)), {"Ldot": {"b_dot": 1.0}}))
    l10.units += _flag({"b_dot": 1.0, "o": -1.0}, "hd_b", "'hd_b': b has a '.'")
    l10.units += _flag(pr.add({"opp": 1.0, "opm": 1.0, "sub": -1.0, "act_red": 1.0}, one(-1.0)), "m_add",
                       "'m_add': add magnitudes this pass")
    l10.units += _flag(_all("sub", "act_red"), "m_sub", "'m_sub': subtract magnitudes this pass")
    l10.units += _flag(_all("opmul", "act_red"), "m_mul", "'m_mul': a multiplication pass")
    lo0 = LO0

    l11 = b.at(11)
    l11.units += _sel({"b_hi": 1.0, "b_dot": -1.0, "one": -1.0}, {"hd_b": 1.0}, "fb")  # b's fractional digits

    # integer digits: ia = (a's '.' or o) - a_lo - 1, ib likewise; la, lb: all digits of a, b
    ia = {"o": 1.0, "fa": -1.0, "hd_a": -1.0, "a_lo": -1.0, "one": -1.0}
    ib = {"b_hi": 1.0, "fb": -1.0, "hd_b": -1.0, "o": -1.0, "sb_x": -1.0, "one": -1.0}
    la = pr.add(ia, {"fa": 1.0})
    lb = pr.add(ib, {"fb": 1.0})
    pm = {"opp": 1.0, "opm": 1.0}
    l12 = b.at(12)
    # F: fractional digits of the result's columns; Wm: the top column (one above the result's
    # largest possible digit). +-: max(fa, fb) and max(ia, ib) + F; *: fa + fb and la + lb.
    l12.units += [*_sel({"fa": 1.0}, pr.add(pm, {"opmul": 1.0}), "F"),
                  _relu(pr.gated(pr.add({"fb": 1.0, "fa": -1.0}), pm), "F", "'F' for +-: max(fa, fb)"),
                  *_sel({"fb": 1.0}, {"opmul": 1.0}, "F")]
    l12.units += [*_sel(pr.add(ia, {"fa": 1.0}), pm, "Wm"),
                  _relu(pr.gated(pr.add(ib, neg(ia)), pm), "Wm", "'Wm' for +-: max(ia, ib)"),
                  _relu(pr.gated(pr.add({"fb": 1.0, "fa": -1.0}), pm), "Wm", "'Wm' for +-: + max(fa, fb)"),
                  *_sel(pr.add(la, lb), {"opmul": 1.0}, "Wm")]
    l12.units += _flag(pr.add({"kk": 1.0}, neg(lb), one(1.0)), "final", "'final': every digit of b is in K")
    l12.units += _step(pr.add({"kk": 1.0, "fb": -1.0}, one(1.0)), "sk_b", gate={"hd_b": 1.0})  # b's digit kk: left of its '.'?
    l12.units.append(_relu(la, "la", "'la': digits of |a|"))

    l13 = b.at(13)
    # CB: the cell of the result's column 0 (b's last digit; a quotient's is set in division.py)
    l13.units.append(_relu({"b_hi": 1.0, "one": -1.0}, "CB", "'CB': column 0's cell"))
    # drop: the redex is the whole group, so its brackets go too, unless the result is an
    # expression that does not end in ')' (a power's (a*a)^h: tests/reference.py step)
    l13.units += _flag(pr.add({"e1": 1.0, "e2": 1.0, "gp": 1.0, "T4": -1.0}, one(-2.0)), "drop",
                       "'drop': the redex is the whole group: its brackets go")
    # b's digit kk (0 once kk = lb: the cell is b's sign or the operator, value 0)
    l13.heads.append(_gather("b's digit kk", pr.add({"b_hi": 1.0, "kk": -1.0, "sk_b": -1.0}, one(-1.0)),
                             {"val": {"bk": 1.0}}))
    l13.units += _flag(pr.add({"opmul": 1.0, "act_red": 1.0, "final": -1.0}, one(-1.0)), "macc",
                       "'macc': this pass adds |a| * b_kk into K and rewrites nothing")
    # a's digit for column j sits at q + offA (- 1 left of the '.'), b's at q + offB; for *, digit i
    # of |a| sits at q + offM (- 1 left of the '.') with i = j - kk
    l13.units += _sel({"o": 1.0, "fa": -1.0, "hd_a": -1.0, "b_hi": -1.0, "F": 1.0, "one": 1.0}, {"one": 1.0}, "offA")
    l13.units += _sel({"fb": -1.0, "hd_b": -1.0, "F": 1.0, "one": 1.0}, {"one": 1.0}, "offB")
    l13.units += _sel({"o": 1.0, "b_hi": -1.0, "kk": 1.0}, {"one": 1.0}, "offM")
    l13.units += _sel(pr.add({"kk": 1.0}, one(1.0)), {"opmul": 1.0, "act_red": 1.0, "final": -1.0, "one": -1.0}, "nkk")

    # ------------------------------------------------------------ columns
    l14 = b.at(14)
    l14.heads.append(_gather("sign before the redex?", pr.add(lo0, one(-1.0)), {"is_unary": {"u_lo": 1.0}}))
    # B_ia = o - fa - hd_a - a_lo - 1 and B_ib = b_hi - fb - hd_b - o - sb_x - 1 (integer digits)
    # are sums of moved values, so the head writes them directly; B_bhi = CB + 1
    l14.heads.append(_broadcast("redex to every cell", {
        "CB": {"B_bhi": 1.0}, "b_hi": {"B_ib": 1.0}, "Wm": {"B_W": 1.0}, "F": {"B_F": 1.0},
        "fa": {"B_fa": 1.0, "B_ia": -1.0}, "fb": {"B_fb": 1.0, "B_ib": -1.0}, "la": {"B_la": 1.0},
        "hd_a": {"B_hda": 1.0, "B_ia": -1.0}, "hd_b": {"B_ib": -1.0}, "o": {"B_ia": 1.0, "B_ib": -1.0},
        "a_lo": {"B_ia": -1.0}, "sb_x": {"B_ib": -1.0}, "one": {"B_ia": -1.0, "B_ib": -1.0, "B_bhi": 1.0},
        "kk": {"B_kk": 1.0}, "bk": {"B_bk": 1.0}, "offA": {"B_offA": 1.0}, "offB": {"B_offB": 1.0},
        "offM": {"B_offM": 1.0}, "act_red": {"B_red": 1.0}, "m_add": {"B_add": 1.0, "B_pmr": 1.0}, "m_sub": {"B_sub": 1.0, "B_pmr": 1.0},
        "m_mul": {"B_mul": 1.0}, "macc": {"B_macc": 1.0}}))
    colB = pr.add({"B_bhi": 1.0}, one(-1.0))  # column 0's cell
    jj = pr.add(colB, neg(pos))  # this cell's column j
    s_ = pr.add(jj, {"B_F": -1.0})  # its power of ten, s = j - F
    l14.units += _step(pr.add(neg(jj), {"B_W": 1.0}, one(1.0)), "c1")  # j <= W
    l14.units += _step(pr.add(jj, one(1.0)), "c2")  # j >= 0
    l14.units += _hat(pr.add(pos, {"B_bhi": -1.0}), "fence", gate={"B_red": 1.0})  # b's end: stops a chain
    l14.units += _hat(jj, "fenceB", gate={"B_red": 1.0})  # column 0
    l14.units += _step(pr.add(s_, one(1.0)), "ge0")  # s >= 0: left of the '.'
    l14.units += _step(pr.add(s_, {"B_fa": 1.0}, one(1.0)), "ha_lo")  # s >= -fa
    l14.units += _step(pr.add({"B_ia": 1.0}, neg(s_)), "ha_hi")  # s <= ia - 1
    l14.units += _step(pr.add(s_, {"B_fb": 1.0}, one(1.0)), "hb_lo")
    l14.units += _step(pr.add({"B_ib": 1.0}, neg(s_)), "hb_hi")
    ii = pr.add(jj, {"B_kk": -1.0})  # *: digit i = j - kk of |a| meets b_kk in column j
    l14.units += _step(pr.add(ii, one(1.0)), "v1_lo")  # i >= 0
    l14.units += _step(pr.add({"B_la": 1.0}, neg(ii)), "v1_hi")  # i <= la - 1
    l14.units += _step(ii, "v2_lo")  # i - 1 >= 0
    l14.units += _step(pr.add({"B_la": 1.0}, neg(ii), one(1.0)), "v2_hi")  # i - 1 <= la - 1
    l14.units += _step(pr.add(ii, {"B_fa": -1.0}, one(1.0)), "skp1", gate={"B_hda": 1.0})  # i >= fa: skip a's '.'
    l14.units += _step(pr.add(ii, {"B_fa": -1.0}), "skp2", gate={"B_hda": 1.0})  # i - 1 >= fa

    l15 = b.at(15)
    oh = {f"oh{d}": 1.0 for d in range(10)}
    l15.heads += [
        _gather("a's digit for this column", pr.add(pos, {"B_offA": 1.0, "ge0": -1.0}), {"val": {"av": 1.0}}, full=True),
        _gather("b's digit for this column", pr.add(pos, {"B_offB": 1.0, "ge0": -1.0}), {"val": {"bv": 1.0}}, full=True),
        _gather("a's digit i (times b_kk)", pr.add(pos, {"B_offM": 1.0, "skp1": -1.0}),
                {f"oh{d}": {f"A1_{d}": 1.0} for d in range(1, 10)}, full=True),  # a 0 multiplies to 0
        _gather("a's digit i-1 (times b_kk, its tens)", pr.add(pos, {"B_offM": 1.0, "skp2": -1.0}, one(1.0)),
                {f"oh{d}": {f"A2_{d}": 1.0} for d in range(1, 10)}, full=True),
    ]
    del oh
    l15.units += _flag(_all("c1", "c2", "B_red"), "in_col", "'in_col': this cell is a result column")
    l15.units += pr.times_flag_units({"av": 1.0}, _all("ha_lo", "ha_hi", "c1", "c2", "B_pmr"), {"aj": 1.0})
    l15.units += pr.times_flag_units({"bv": 1.0}, _all("hb_lo", "hb_hi", "c1", "c2", "B_pmr"), {"bj": 1.0})
    l15.units += _onehot_times("A1_", {"B_bk": 1.0}, _all("v1_lo", "v1_hi", "c1", "c2", "B_mul"), "P1",
                               "'P1' = a_i * b_kk, the product landing in this column")
    l15.units += _onehot_times("A2_", {"B_bk": 1.0}, _all("v2_lo", "v2_hi", "c1", "c2", "B_mul"), "P2",
                               "'P2' = a_(i-1) * b_kk, whose tens land in this column")

    l16 = b.at(16)
    l16.units += cm.ramp_mod10_units({"P1": 1.0}, 0, 81, None, 1.0, prefix="L1_", skip_zero=True)
    l16.units += cm.ramp_mod10_units({"P2": 1.0}, 0, 81, None, 1.0, prefix="L2_", skip_zero=True)
    # u = K + (P1 mod 10) + (P2 div 10), 0..26
    u = pr.add({"K": 1.0}, {f"L1_{d}": float(d) for d in range(1, 10)},
               {"P2": 0.1}, {f"L2_{d}": -0.1 * d for d in range(1, 10)})
    l17 = b.at(17)
    l17.units += cm.ramp_mod10_units(u, 0, 26, None, 1.0, prefix="DU", skip_zero=True)
    du = {f"DU{d}": float(d) for d in range(1, 10)}
    tens = pr.scale(pr.add(u, neg(du)), 0.1)  # u div 10, 0..2: goes one column up
    l18 = b.at(18)
    l18.heads.append(_gather("tens from the column below", pr.add(pos, one(1.0)),
                             {d_: {"eprev": c} for d_, c in tens.items() if d_ != "one"}, full=True))
    v = pr.add(du, {"eprev": 1.0})  # *: this column's digit sum before the binary carry, 0..11
    sab = {"aj": 1.0, "bj": 1.0}
    dab = {"aj": 1.0, "bj": -1.0}
    add_on = pr.add({"in_col": 1.0, "B_add": 1.0}, one(-1.0))
    sub_on = pr.add({"in_col": 1.0, "B_sub": 1.0, "B_div": 1.0}, one(-1.0))  # a division's r - B too
    mul_on = pr.add({"in_col": 1.0, "B_mul": 1.0}, one(-1.0))
    for z, on in ((sab, add_on), (v, mul_on)):
        l18.units += _step(pr.add(z, one(-9.0)), "ge10", gate=on)
        l18.units += _step(pr.add(neg(z), one(9.0)), "le8", gate=on)
    l18.units += _step(dab, "gt", gate=sub_on)
    l18.units += _step(neg(dab), "lt", gate=sub_on)

    l19 = b.at(19)
    match = {"A": {"ge10": 1.0, "le8": 1.0, "fence": 1.0}, "S": {"gt": 1.0, "lt": 1.0, "fence": 1.0}}
    for k in CHAINS:
        l19.units += pr.latest_feature(match[k], f"f_m{k}")
    b.at(20).heads += [pr.latest_head(f"latest stop {k}", f"f_m{k}", f"Lm{k}") for k in CHAINS]
    b.at(21).heads += [_gather(f"prev stop {k}", prev, {f"Lm{k}": {f"PMm{k}": 1.0}}) for k in CHAINS]
    b.at(22).heads += [_gather(f"pm^2 stop {k}", {f"PMm{k}": 1.0}, {"pos2": {f"PM2m{k}": 1.0}}) for k in CHAINS]

    l23 = b.at(23)
    # carry into column j: the first lower column (cell to the right) whose sum is not 9, or b's end
    hA = pr.next_match_head("carry in", "PMmA", "PM2mA", match["A"], {"LmA": 1.0}, "_")
    hA.value = {"ge10": {"cy": 1.0}}
    hS = pr.next_match_head("borrow in", "PMmS", "PM2mS", match["S"], {"LmS": 1.0}, "_")
    hS.value = {"lt": {"bw1": 1.0}, "gt": {"bw2": 1.0}}  # a-b borrows where a_i < b_i; b-a where a_i > b_i
    hA.full = hS.full = True
    l23.heads += [hA, hS]
    l23.units += cm.ramp_mod10_units(pr.add(sab, {"cy": 1.0}), 0, 19, add_on, 1.0, prefix="RA")
    l23.units += cm.ramp_mod10_units(pr.add(dab, {"bw1": -1.0}, one(10.0)), 0, 19, sub_on, 1.0, prefix="RA")
    l23.units += cm.ramp_mod10_units(pr.add(neg(dab), {"bw2": -1.0}, one(10.0)), 0, 19, sub_on, 1.0, prefix="RB")
    l23.units += cm.ramp_mod10_units(pr.add(v, {"cy": 1.0}), 0, 12, mul_on, 1.0, prefix="RA")
    # the next accumulator digit (exact; only while accumulating: the writing pass clears K)
    l23.units += _robust_mod10(pr.add(v, {"cy": 1.0}), 12, "nK", pr.add(mul_on, {"B_macc": 1.0}, one(-1.0)))

    # ------------------------------------------------------------ the result's shape, at the root
    l24 = b.at(24)
    for k in R_SETS:  # top nonzero column (the earliest such cell, else column 0), bottom one (latest)
        l24.units.append(pr.earliest_if({"in_col": 1.0, f"R{k}0": -1.0}, 1.0, f"f_t{k}"))
        l24.units.append(pr.earliest_if({"fenceB": 1.0, f"R{k}0": 1.0}, 2.0, f"f_t{k}"))
        l24.units += pr.latest_feature({"in_col": 1.0, f"R{k}0": -1.0}, f"f_b{k}")
    l25 = b.at(25)
    for k in R_SETS:
        h = pr.latest_head(f"top digit {k}", f"f_t{k}", f"top{k}")
        h.value = {"pos": {f"top{k}": 1.0}, f"R{k}0": {f"z{k}": 1.0}}  # z: that digit is 0, the result is 0
        l25.heads += [h, pr.latest_head(f"bottom digit {k}", f"f_b{k}", f"bot{k}")]
    # a-b went below zero: its top column (both digits 0) borrowed, so it reads 9
    l25.heads.append(_gather("a-b negative?", {"CB": 1.0, "Wm": -1.0}, {"RA9": {"neg1": 1.0}}))
    l25.units += _flag(_all("m_sub", "neg1"), "ltm", "'ltm': subtracting and |a| < |b|: the result is b-a")

    l26 = b.at(26)
    l26.units += _sel({"topA": 1.0}, {"one": 1.0}, "topS")
    l26.units += _sel({"topB": 1.0, "topA": -1.0}, {"ltm": 1.0}, "topS")
    l26.units += _sel({"botA": 1.0}, {"one": 1.0}, "botS")
    l26.units += _sel({"botB": 1.0, "botA": -1.0}, {"ltm": 1.0}, "botS")
    l26.units += _flag({"zA": 1.0, "ltm": -1.0}, "zero", "'zero': the result is 0")
    l26.units += _flag({"zB": 1.0, "ltm": 1.0, "one": -1.0}, "zero", "'zero': the result (b-a) is 0")
    # The sign: + and -: a's, or b's when b-a; *: sa XOR sb. Written as '-' (sgnw), or cancelled
    # with a sign '-' right before the redex (cancel); a zero result has none.
    pmr = {"m_add": 1.0, "m_sub": 1.0}
    cases = (({"sa_x": 1.0, "ltm": -1.0, "zA": -1.0}, pmr, "a's sign"),
             ({"sa_x": -1.0, "ltm": 1.0, "zB": -1.0}, {"m_sub": 1.0}, "b's sign (b-a)"),
             ({"sa_x": 1.0, "sb_x": -1.0, "zA": -1.0}, {"m_mul": 1.0, "m_dfin": 1.0}, "a negative, b positive"),
             ({"sa_x": -1.0, "sb_x": 1.0, "zA": -1.0}, {"m_mul": 1.0, "m_dfin": 1.0}, "a positive, b negative"))
    for form, gate, what in cases:
        l26.units += _flag(pr.add(form, gate, {"u_lo": -1.0}, one(-1.0)), "sgnw", f"'sgnw': a negative result ({what})")
        l26.units += _flag(pr.add(form, gate, {"u_lo": 1.0}, one(-2.0)), "cancel",
                           f"'cancel': a negative result ({what}) right after a sign '-': both go")

    # which write this pass makes: a number, a power's text, or none (accumulating, or room)
    b.at(17).units.append(_relu({"growp": 1.0}, "grow_any", "'grow_any': this pass asks for room"))
    b.at(18).units.append(_relu({"grow_any": 1.0}, "grow", "'grow': the router's room request (read at the root)"))
    l26.units += _flag({"act_red": 1.0, "macc": -1.0, "grow_any": -1.0, "opc": -1.0, "dwork": -1.0}, "rwn",
                       "'rwn': this pass writes a number")
    l26.units += _flag({"act_red": 1.0, "opc": 1.0, "grow_any": -1.0, "one": -1.0}, "rwp",
                       "'rwp': this pass writes a power's rewrite")

    # T = top column, Bc = bottom nonzero column; ni = integer digits written (at least "0");
    # Fp = fractional digits written (trailing zeros stripped: F - Bc, 0 if negative or zero)
    l27 = b.at(27)
    top_col = {"CB": 1.0, "topS": -1.0}
    l27.units += [_relu(one(1.0), "ni", "'ni' base: one integer digit at least"),
                  _relu(pr.add(top_col, {"F": -1.0}), "ni", "'ni': the integer digits above the units")]
    frac = {"F": 1.0, "CB": -1.0, "botS": 1.0, "zero": -pr.BIG}
    l27.units.append(_relu(frac, "Fp", "'Fp': fractional digits written (F minus the trailing zeros)"))
    l27.units += _flag(frac, "hasfrac", "'hasfrac': the result has a '.'")

    # Replace lo..hi by m bytes; DS: integer digit k is at column cell q + DS (frac digits: one
    # less); CS: the copied number of a bracket group is at q + CS. One product per action.
    l28 = b.at(28)
    rw = {"rwn": 1.0}  # a reduction that writes a number this pass
    red_lo = pr.add(lo0, {"cancel": -1.0})
    red_hi = RED_HI
    red_m = {"sgnw": 1.0, "ni": 1.0, "hasfrac": 1.0, "Fp": 1.0}
    br_lo = {"p": 1.0, "cb": -1.0}
    br_m = {"c": 1.0, "p": -1.0, "cb": -1.0, "one": -1.0}
    sp_lo = {"SP": 1.0}
    pw_lo = pr.add(lo0, {"pc": -1.0})
    acts = ((rw, red_lo, red_hi, red_m), ({"rwp": 1.0}, pw_lo, red_hi, {"mp": 1.0, "pc": -1.0}),
            ({"act_br": 1.0}, br_lo, {"c": 1.0}, br_m), ({"act_sp": 1.0}, sp_lo, sp_lo, {}))
    for act, lo, hi, m in acts:
        l28.units += _sel(lo, act, "LO")
        if m:
            l28.units += _sel(m, act, "M")
        l28.units += _sel(pr.add(hi, neg(lo), neg(m), one(1.0)), act, "G")
    l28.units += _sel(pos, {"macc": 1.0, "grow_any": 1.0, "dwork": 1.0}, "LO")  # accumulating or growing: every cell keeps its byte
    l28.units += _sel(pr.add({"CB": 1.0, "F": -1.0, "ni": -1.0, "sgnw": -1.0}, one(1.0), neg(red_lo)), rw, "DS")
    l28.units += _sel({"cb": 2.0, "one": 1.0}, {"act_br": 1.0}, "CS")
    l28.units += _sel({"ni": 1.0}, rw, "NI")
    l28.units += [_relu({k: 1.0}, "act_go", f"'act_go': this pass does something ({k})")
                  for k in ("act_red", "act_br", "act_sp")]

    # ------------------------------------------------------------ every cell writes its next byte
    l29 = b.at(29)
    l29.heads.append(_broadcast("action to every cell", {
        "act_go": {"go": 1.0, "route": 1.0}, "LO": {"lo_B": 1.0}, "M": {"m_B": 1.0}, "G": {"g_B": 1.0},
        "DS": {"ds_B": 1.0}, "CS": {"cs_B": 1.0}, "sgnw": {"sg_B": 1.0}, "NI": {"ni_B": 1.0},
        "hasfrac": {"hf_B": 1.0}, "ltm": {"ltm_B": 1.0}, "rwn": {"red_B": 1.0},
        "act_br": {"br_B": 1.0}, "pos": {"R_B": 1.0}}))
    l29.units += _step(pr.add(pos, {"lo_B": -1.0}, one(1.0)), "gelo")  # q >= LO
    l29.units += _step(pr.add(pos, {"lo_B": -1.0, "sg_B": -1.0}, one(1.0)), "gelo2")  # q >= LO + sign
    l29.units += _step(pr.add({"lo_B": 1.0, "m_B": 1.0}, neg(pos)), "ltend")  # q < LO + M
    l29.units += _step(pr.add(pos, {"g_B": 1.0, "R_B": -1.0}, one(1.0)), "blank")  # q + G >= root
    dot_at = pr.add(pos, {"lo_B": -1.0, "sg_B": -1.0, "ni_B": -1.0})  # 0 at the '.'
    l29.units += _hat(dot_at, "atdot", gate={"hf_B": 1.0})
    l29.units += _step(dot_at, "afterdot")

    l30 = b.at(30)
    l30.heads += [
        _gather("byte from the right (after the result)", pr.add(pos, {"g_B": 1.0}),
                {dim: {f"S_{dim}": 1.0} for dim in CONTENT.values()}, full=True),
        _gather("byte of the bracketed number", pr.add(pos, {"cs_B": 1.0}),
                {dim: {f"C_{dim}": 1.0} for dim in CONTENT.values()}, full=True),
        _gather("result digit", pr.add(pos, {"ds_B": 1.0, "afterdot": -1.0}),
                {f"R{k}{d}": {f"G{k}{d}": 1.0} for k in R_SETS for d in range(10)}, full=True),
    ]
    for byte, dim in {**CONTENT, **FIXED}.items():
        h = HOLD[byte]
        own = {dim: 1.0}
        l30.units += _flag(pr.add(own, {"go": -1.0}), h, f"halt: keep '{chr(byte)!s}'")
        if dim in FIXED.values():
            l30.units += _flag(pr.add(own, {"go": 1.0}, one(-1.0)), h, f"keep the fixed byte {byte}")
            continue
        l30.units += _flag(pr.add(own, {"go": 1.0, "gelo": -1.0}, one(-1.0)), h,
                           f"before the replaced bytes: keep '{chr(byte)}'")
        l30.units += _flag(pr.add({f"S_{dim}": 1.0, "go": 1.0, "gelo": 1.0, "ltend": -1.0, "blank": -1.0,
                                   "is_eq": -1.0}, one(-2.0)), h, f"after the result: '{chr(byte)}' from G cells right")
        l30.units += _flag(pr.add({f"C_{dim}": 1.0, "gelo": 1.0, "ltend": 1.0, "br_B": 1.0}, one(-3.0)), h,
                           f"bracket group: copy '{chr(byte)}' of its number")
    l30.units += _flag(pr.add({"go": 1.0, "gelo": 1.0, "ltend": -1.0, "blank": 1.0, "is_eq": -1.0}, one(-2.0)),
                       HOLD[BLANK], "after the result, past the tape's end: a blank")
    l30.units += _flag(pr.add({"gelo": 1.0, "gelo2": -1.0, "red_B": 1.0}, one(-1.0)), HOLD[cm.MINUS],
                       "the result's sign")
    l30.units += _flag(pr.add({"atdot": 1.0, "ltend": 1.0, "red_B": 1.0}, one(-2.0)), HOLD[ord(".")],
                       "the result's '.'")
    zone = {"gelo2": 1.0, "ltend": 1.0, "red_B": 1.0, "atdot": -1.0}
    for dg in range(10):
        hd = HOLD[ord("0") + dg]
        l30.units += _flag(pr.add({f"GA{dg}": 1.0, "ltm_B": -1.0}, zone, one(-3.0)), hd, f"result digit {dg} (a+b, a-b, a*b)")
        l30.units += _flag(pr.add({f"GB{dg}": 1.0, "ltm_B": 1.0}, zone, one(-4.0)), hd, f"result digit {dg} (b-a)")

    # the adapter: the byte each cell holds next, the accumulator and the root's digit counter
    for byte, dim in HOLD.items():
        b.hold(dim, byte)
    b.carry("K", {"nK": 1.0})
    b.carry("kk", {"nkk": 1.0})


def build_coda(b: LoopBuilder) -> None:
    """The coda: a causal decoder that writes the expression on the final tape, then '\\n'.
    At the root (position R) and every answer byte after it, position q writes the tape's byte
    q - R + 1 while that is before the expression's end X."""
    c1 = b.coda_at(1)
    c1.units.append(pr.earliest_if({"is_tilde": 1.0, "is_space": 1.0, "is_eq": 1.0}, 1.0, "cf_X"))
    c1.units.append(pr.earliest_if({"is_eq": 1.0}, 1.0, "cf_R"))
    c2 = b.coda_at(2)
    c2.heads += [pr.latest_head("expression end", "cf_X", "cX"), pr.latest_head("root", "cf_R", "cR")]
    c2.units += _step({"cX": 1.0, "cR": 1.0, "pos": -1.0, "one": -1.0}, "inx")  # q - R + 1 < X
    c3 = b.coda_at(3)
    src = pr.add({"pos": 1.0, "cR": -1.0}, pr.const(1.0))
    c3.heads.append(_gather("answer byte", src, {dim: {f"A_{dim}": 1.0} for dim in ANSWER.values()}))
    for byte, dim in ANSWER.items():
        c3.units.append(_relu({f"A_{dim}": 1.0, "inx": 1.0, "one": -1.0}, f"O_{dim}", f"write '{chr(byte)}'"))
        b.logit(f"O_{dim}", byte, cm.DIGIT_LOGIT)
    c3.units.append(_relu({"one": 1.0, "inx": -1.0}, "O_nl", "past the expression: end the answer"))
    b.logit("O_nl", cm.EOS, cm.DIGIT_LOGIT)


ANSWER = {**{ord("0") + d: f"oh{d}" for d in range(10)}, cm.MINUS: "is_minus", ord("."): "is_dot"}
CODA_DIMS = ("cf_X", "cf_R", "cX", "cR", "inx", *(f"A_{d}" for d in ANSWER.values()),
             *(f"O_{d}" for d in ANSWER.values()), "O_nl")
