"""The root: parse the latest copy of the expression and plan the step (LOOPED.md §4.1).

A root is a position holding '=': the prompt's, or the one each step writes after its copy.
Under the causal mask it sees its whole copy. The copy is the run of cells (expression bytes)
right before it, after ``fz``, the latest position that is not a cell (the start byte, or the
position that wrote the copy's first byte). Every "latest X" a parse looks up may find an X in
an older copy, so each is clamped to ``fz``.

The parse is the frame-writing calculator's: the innermost group ``p``..``c``, the operator
``o`` (latest '^', else first '*' '/', else first '+' '-'), the operands with their signs and
fractional digits. Then the root picks the step (spaces, brackets, reduce, or halt) and writes
its plan into the ``C_`` dims. The latent map T carries those unchanged into every position of
the step (``C_ -> C_``), so the root writes "new value minus the value it was handed".

Layers (heads read the layer input; the MLP also sees this layer's heads):
  1      previous-byte flags; landmark tags for every marker
  2      latest heads (each marker's latest position)
  3      fz (the copy's start), forward-search pointers, X (the copy's end)
  4-8    the innermost group, the operator, the step's kind
  10-13  operands: fences, signs, '.', fractional digits; precision N (4-9); halt
  14-17  the plan; 18: the plan is written into C_
"""

from __future__ import annotations

from .. import primitives as pr
from ..model import Form
from . import parts as P

MARKERS: dict[str, list[Form]] = {
    "Lg": [{"is_lpar": 1.0}],
    "Lcx": [{"is_rpar": 1.0}, {"is_tilde": 1.0}, {"is_eq": 1.0}],  # ')' or where a copy ends
    "Lmul": [{"is_star": 1.0}, {"is_slash": 1.0}],
    "Lpow": [{"is_caret": 1.0}],
    "Ladd": [{"is_plus": 1.0}, P.BINMINUS],
    "Lnd": [{"one": 1.0, "is_digit": -1.0, "is_dot": -1.0}],  # not part of a number
    "Ldot": [{"is_dot": 1.0}],
    "Lnz": [{"nz": 1.0}],  # a nonzero digit
    "Ptil": [{"is_tilde": 1.0}],  # the precision marker (the prompt's only)
    "P": [{"is_plus": 1.0}, {"is_star": 1.0}, {"is_slash": 1.0}, {"is_caret": 1.0}, P.BINMINUS],
    "Lsp": [{"is_space": 1.0}],
    "Lnc": [{"one": 1.0, "is_cell": -1.0}],  # not a cell: where a copy starts
    "E": [{"is_eq": 1.0}],  # a root
    "Lns": [{"one": 1.0, "is_space": -1.0}],  # not a space (the walker's next byte)
}
NEXT = ("Lcx", "Lmul", "Ladd", "Lnd", "Lns")  # classes searched forward (next_match)
N_DIGITS = 4  # precision ~N: N <= 9999

# The plan, carried by T into every position of the step (see the module docstring).
CTX = (
    "C_sp", "C_br", "C_red", "C_add", "C_sub", "C_mul", "C_div", "C_pow", "C_num",
    "C_ofz", "C_X", "C_lo", "C_hi", "C_mlo", "C_mhi", "C_ulo",
    "C_F", "C_W", "C_sa", "C_sx",
    "C_PA", "C_alo", "C_ahi", "C_PB", "C_blo", "C_bhi",  # + -: column j's digit of a at C_PA - j (+1 right of '.')
    "C_MA", "C_MB", "C_fa", "C_fb", "C_hda", "C_hdb", "C_la", "C_lb",  # * and /: digits as integers
    "C_AF", "C_ia", "C_lB",  # /: A's digits left to right, B's length
)

DIMS = (
    "prev_num", "prev_rpar", "is_unary", "is_binminus",
    *(f"f_{m}" for m in MARKERS), *MARKERS,
    "fz", *(f"PM_{c}" for c in NEXT), *(f"PM2_{c}" for c in NEXT), "tl", "X",
    "p_raw", "sp_raw", "p", "gp", "act_sp", *(f"Lt_{c}" for c in ("Lcx", "Lmul", "Ladd")),
    "om", "fm", "oa", "c", "opw", "Pg", "use_mul", "pw", "bin_in", "o", "act_red", "act_br",
    "a_lo", "a_dot", "sb_x", "g_plus", "g_minus", "g_star", "g_caret", "g_slash", "u_p", "sneg_p",
    "opp", "opm", "opmul", "opdiv", "opc", "hd_a", "sa_x", "b_hi", "e1", "e2", "sub", "fa", "cb",
    "b_dot", "bnzp", "hd_b", "bz", "fb", "drop", "halt", "m_add", "m_sub",
    "has_til", "seen_eq", "f_Ndg", "Ndg", *(f"qN{k}" for k in range(1, N_DIGITS)),
    *(f"Nd{k}" for k in range(N_DIGITS)), "N",
    "lo0", "u_lo", "F", "W", "sx", "ia", "ib", *(f"pl_{k}" for k in ("add", "sub", "mul", "div", "pow", "num")),
    *CTX,
)

ROOT: Form = {"is_eq": 1.0}
# which step a plan dim is for: the root writes it only then (a spaces step's parse reads
# garbage, which must not reach the thoughts); the rest are for every step
PLAN_GATE = {**{d: ("act_br",) for d in ("C_mlo", "C_mhi")},
             **{d: ("act_red",) for d in ("C_ulo", "C_F", "C_W", "C_sa", "C_sx", "C_PA", "C_alo", "C_ahi", "C_PB",
                                          "C_blo", "C_bhi", "C_MA", "C_MB", "C_fa", "C_fb", "C_hda", "C_hdb", "C_la",
                                          "C_lb", "C_AF", "C_ia", "C_lB")}}
CTX_LAYER = 18  # the root writes the plan here (power.py adds its own C_ dims at this layer)


def ia_form() -> Form:
    """a's integer digits: a spans a_lo+1 .. o-1 (its sign is a_lo itself)."""
    return {"o": 1.0, "fa": -1.0, "hd_a": -1.0, "a_lo": -1.0, "one": -1.0}


def ib_form() -> Form:
    return {"b_hi": 1.0, "fb": -1.0, "hd_b": -1.0, "o": -1.0, "sb_x": -1.0, "one": -1.0}


def build(b) -> None:
    pos = {"pos": 1.0}
    prev = pr.add(pos, pr.const(-1.0))
    one = pr.const
    neg = lambda f: pr.scale(f, -1.0)  # noqa: E731

    l1 = b.at(1)
    l1.heads += [
        P.gather("prev byte in a number?", prev, {"is_digit": {"prev_num": 1.0}, "is_dot": {"prev_num": 1.0}}),
        P.gather("prev byte ')'?", prev, {"is_rpar": {"prev_rpar": 1.0}}),
    ]
    l1.units.append(P.relu({"is_minus": 1.0, "prev_num": -1.0, "prev_rpar": -1.0}, "is_unary",
                           "'is_unary': a '-' not after a number or ')' is a sign"))
    l1.units.append(P.relu(P.BINMINUS, "is_binminus", "'is_binminus': a '-' after a number or ')' subtracts"))
    for m, matches in MARKERS.items():
        for k, match in enumerate(matches):
            l1.units += pr.latest_feature(match, f"f_{m}", bos=(k == 0))

    l2 = b.at(2)
    l2.heads += [pr.latest_head(f"latest {m}", f"f_{m}", m) for m in MARKERS]
    l2.units += P.flag({"Ptil": 1.0}, "has_til", "'has_til': the prompt's '~' is at or before here")
    l2.units += P.flag({"E": 1.0}, "seen_eq", "'seen_eq': a root is at or before here")

    l3 = b.at(3)
    l3.heads.append(P.gather("copy start (latest non-cell before here)", prev, {"Lnc": {"fz": 1.0}}))
    l3.heads += [P.gather(f"prev {c}", prev, {c: {f"PM_{c}": 1.0}}) for c in NEXT]
    # the digits of N: after the prompt's '~', before its '=', in the prompt (not a thought)
    l3.units += pr.latest_feature({"is_digit": 1.0, "has_til": 1.0, "seen_eq": -1.0, "lat": -1.0, "one": -1.0},
                                  "f_Ndg")
    # X, where the copy ends: the prompt's '~' if it is in this copy, else the root
    l3.units += P.flag({"Ptil": 1.0, "fz": -1.0}, "tl", "'tl': the copy holds the prompt's '~'")
    l3.units.append(P.relu(pos, "X", "'X' base: this position (the root)"))

    l4 = b.at(4)
    l4.units += P.sel(pr.add(pos, {"Ptil": -1.0}), {"tl": 1.0}, "X", -1.0)  # X = Ptil when tl
    l4.heads += [P.gather(f"pm^2 {c}", {f"PM_{c}": 1.0}, {"pos2": {f"PM2_{c}": 1.0}}) for c in NEXT]
    l4.heads.append(pr.latest_head("latest digit of N", "f_Ndg", "Ndg"))

    l5 = b.at(5)
    before_x = pr.add({"X": 1.0}, one(-1.0))
    l5.heads += [P.gather("innermost '(' (before X)", before_x, {"Lg": {"p_raw": 1.0}}),
                 P.gather("latest space (before X)", before_x, {"Lsp": {"sp_raw": 1.0}})]
    # p = max(latest '(', fz): a '(' of an older copy does not count
    l5.units.append(P.relu({"fz": 1.0}, "p", "'p' base: the copy's start"))
    l5.units.append(P.relu({"p_raw": 1.0, "fz": -1.0}, "p", "'p': the innermost '(' when it is in this copy"))
    l5.units += P.flag({"p_raw": 1.0, "fz": -1.0}, "gp", "'gp': the innermost group is a real '('")
    l5.units += P.flag({"sp_raw": 1.0, "fz": -1.0}, "act_sp", "'act_sp': the copy has a space: remove them all")

    # precision N: digits right to left from the latest one (0 at the start byte once they run out)
    at = {0: {"Ndg": 1.0}, **{k: {f"qN{k}": 1.0} for k in range(1, N_DIGITS)}}
    for k in range(N_DIGITS):
        lk = b.at(5 + k)
        lk.heads.append(P.gather(f"precision digit {k}", at[k], {"val": {f"Nd{k}": 1.0}}))
        if k + 1 < N_DIGITS:
            lk.heads.append(P.gather(f"precision digit {k + 1} position", pr.add(at[k], one(-1.0)),
                                     {"Ndg": {f"qN{k + 1}": 1.0}}))
    for k in range(N_DIGITS):  # N = sum of 10^k Nd_k, each digit re-quantized (a gather leaks)
        b.at(5 + N_DIGITS).units += P.count_up({f"Nd{k}": 1.0}, 9, "N", f"'N': precision digit {k}", k=10.0 ** k)

    l6 = b.at(6)
    l6.heads += [P.gather(f"latest {c} at p", {"p": 1.0}, {c: {f"Lt_{c}": 1.0}}) for c in ("Lcx", "Lmul", "Ladd")]

    l7 = b.at(7)
    l7.heads += [
        # c: the first ')' after p, or the copy's end X (its '~' or root) when there is none
        pr.next_match_head("first ) after p", "PM_Lcx", "PM2_Lcx", {"is_rpar": 1.0, "is_tilde": 1.0, "is_eq": 1.0},
                           {"Lt_Lcx": 1.0}, "c"),
        pr.next_match_head("first * or / after p", "PM_Lmul", "PM2_Lmul", {"is_star": 1.0, "is_slash": 1.0},
                           {"Lt_Lmul": 1.0}, "om", "fm"),
        pr.next_match_head("first +- after p", "PM_Ladd", "PM2_Ladd", {"is_plus": 1.0, "is_binminus": 1.0},
                           {"Lt_Ladd": 1.0}, "oa"),
    ]

    l8 = b.at(8)
    before_c = pr.add({"c": 1.0}, one(-1.0))
    l8.heads += [P.gather("latest ^ before c", before_c, {"Lpow": {"opw": 1.0}}),
                 P.gather("latest operator before c", before_c, {"P": {"Pg": 1.0}})]
    l8.units += P.flag(pr.add({"c": 1.0, "om": -1.0, "fm": pr.BIG}, one(-pr.BIG)), "use_mul",
                       "'use_mul': a * or / inside the group")
    l8.units += P.flag({"opw": 1.0, "p": -1.0}, "pw", "'pw': a ^ inside the group")
    l8.units += P.flag({"Pg": 1.0, "p": -1.0}, "bin_in", "'bin_in': a binary operator inside the group")

    l9 = b.at(9)
    l9.units += [
        P.relu({"oa": 1.0, "use_mul": -pr.BIG, "pw": -pr.BIG}, "o", "'o': first + or - (no ^, no * or / in the group)"),
        P.relu({"om": 1.0, "use_mul": pr.BIG, "pw": -pr.BIG, "one": -pr.BIG}, "o", "'o': first * or /"),
        P.relu({"opw": 1.0, "pw": pr.BIG, "one": -pr.BIG}, "o", "'o': latest ^ (right-associative)"),
    ]
    l9.units += P.flag({"bin_in": 1.0, "act_sp": -1.0}, "act_red", "'act_red': reduce the redex")
    l9.units += P.flag({"gp": 1.0, "bin_in": -1.0, "act_sp": -1.0}, "act_br",
                       "'act_br': the innermost group holds one number: drop its brackets")

    l10 = b.at(10)
    before_o = pr.add({"o": 1.0}, one(-1.0))
    l10.heads += [
        P.gather("a fence", before_o, {"Lnd": {"a_lo": 1.0}}),
        P.gather("a's '.'", before_o, {"Ldot": {"a_dot": 1.0}}),
        P.gather("sign of b?", pr.add({"o": 1.0}, one(1.0)), {"is_minus": {"sb_x": 1.0}}),
        P.gather("operator", {"o": 1.0}, {"is_plus": {"g_plus": 1.0}, "is_minus": {"g_minus": 1.0},
                                          "is_star": {"g_star": 1.0}, "is_caret": {"g_caret": 1.0},
                                          "is_slash": {"g_slash": 1.0}}),
        P.gather("sign before the group?", pr.add({"p": 1.0}, one(-1.0)), {"is_unary": {"u_p": 1.0}}),
        P.gather("group's number negative?", pr.add({"p": 1.0}, one(1.0)), {"is_minus": {"sneg_p": 1.0}}),
    ]
    l10.units += P.flag({"g_plus": 1.0}, "opp", "'opp': the redex adds")
    l10.units += P.flag({"g_minus": 1.0}, "opm", "'opm': the redex subtracts")
    l10.units += P.flag({"g_star": 1.0}, "opmul", "'opmul': the redex multiplies")
    l10.units += P.flag({"g_slash": 1.0}, "opdiv", "'opdiv': the redex divides")
    l10.units += P.flag({"g_caret": 1.0}, "opc", "'opc': the redex is a power (a text rewrite)")
    l10.units += P.flag({"a_dot": 1.0, "a_lo": -1.0}, "hd_a", "'hd_a': a has a '.'")

    l11 = b.at(11)
    l11.heads += [
        P.gather("sign of a?", {"a_lo": 1.0}, {"is_unary": {"sa_x": 1.0}}),
        pr.next_match_head("b fence", "PM_Lnd", "PM2_Lnd", {"one": 1.0, "is_digit": -1.0, "is_dot": -1.0},
                           {"o": 1.0, "sb_x": 1.0}, "b_hi"),
    ]
    af = {"a_lo": 1.0, "sa_x": -1.0}  # the byte before signed a
    l11.units += P.hat(pr.add(af, {"p": -1.0}), "e1", "'e1': a starts right after the '('")
    l11.units += P.hat({"b_hi": 1.0, "c": -1.0}, "e2", "'e2': b ends at the ')'")
    # sub: + or -, and the signs of a and of b as added (b's sign XOR the '-') differ
    sx3 = {"sa_x": 1.0, "sb_x": 1.0, "opm": 1.0}
    gate = {"opp": 1.0, "opm": 1.0}
    for k in (1, 3):
        l11.units += P.hat(pr.add(sx3, one(-float(k))), "sub", "'sub': the magnitudes are subtracted", gate)
    l11.units += P.sel({"o": 1.0, "a_dot": -1.0, "one": -1.0}, {"hd_a": 1.0}, "fa")
    l11.units += P.flag(P.all_of("act_br", "u_p", "sneg_p"), "cb", "'cb': -(-v): both signs go with the brackets")

    l12 = b.at(12)
    l12.heads += [P.gather("b's '.'", pr.add({"b_hi": 1.0}, one(-1.0)), {"Ldot": {"b_dot": 1.0}}),
                  P.gather("b's latest nonzero digit", pr.add({"b_hi": 1.0}, one(-1.0)), {"Lnz": {"bnzp": 1.0}})]
    l12.units += P.flag({"b_dot": 1.0, "o": -1.0}, "hd_b", "'hd_b': b has a '.'")
    l12.units += P.flag(pr.add({"o": 1.0, "bnzp": -1.0}, one(1.0)), "bz",
                        "'bz': a division by zero (b has no nonzero digit)")
    l12.units += P.flag(pr.add({"opp": 1.0, "opm": 1.0, "sub": -1.0}), "m_add", "'m_add': add the magnitudes")
    l12.units += P.flag(P.all_of("sub"), "m_sub", "'m_sub': subtract the magnitudes")
    # sx: the product's or quotient's sign, sa XOR sb
    l12.units += P.hat(pr.add({"sa_x": 1.0, "sb_x": 1.0}, one(-1.0)), "sx", "'sx': exactly one operand is negative")

    l13 = b.at(13)
    l13.units += P.sel({"b_hi": 1.0, "b_dot": -1.0, "one": -1.0}, {"hd_b": 1.0}, "fb")
    # halt: nothing to do (the copy is a number), or a division by zero is next
    l13.units += P.flag(pr.add({"act_sp": -1.0, "act_br": -1.0, "act_red": -1.0}, one(1.0)), "halt",
                        "'halt': the copy is a number: it is the answer")
    l13.units += P.flag(P.all_of("act_red", "opdiv", "bz"), "halt", "'halt': a division by zero is next")

    # ------------------------------------------------------------ the plan
    ia, ib = ia_form(), ib_form()
    pm = {"opp": 1.0, "opm": 1.0}
    l14 = b.at(14)
    l14.units += P.sel(ia, {"one": 1.0}, "ia")
    l14.units += P.sel(ib, {"one": 1.0}, "ib")
    # F: the result's fractional digits. + -: max(fa, fb); *: fa + fb; /: N
    l14.units += [*P.sel({"fa": 1.0}, pr.add(pm, {"opmul": 1.0}), "F"),
                  P.relu(pr.gated(pr.add({"fb": 1.0, "fa": -1.0}), pm), "F", "'F' for +-: max(fa, fb)"),
                  *P.sel({"fb": 1.0}, {"opmul": 1.0}, "F"), *P.sel({"N": 1.0}, {"opdiv": 1.0}, "F")]

    l15 = b.at(15)
    # drop: the redex is the whole group, so its brackets go too, unless the result is an
    # expression that does not end in ')' (a power's (a*a)^h: tests/reference.py step)
    l15.units += P.flag(pr.add({"e1": 1.0, "e2": 1.0, "gp": 1.0, "T4": -1.0}, one(-2.0)), "drop",
                        "'drop': the redex is the whole group: its brackets go")
    # W: the result's columns. + -: max(ia, ib) + F + 1; *: la + lb; /: len(A) = la + fb + N
    la, lb = {"ia": 1.0, "fa": 1.0}, {"ib": 1.0, "fb": 1.0}
    l15.units += [*P.sel(pr.add({"ia": 1.0, "F": 1.0}, one(1.0)), pm, "W"),
                  P.relu(pr.gated(pr.add({"ib": 1.0, "ia": -1.0}), pm), "W", "'W' for +-: max(ia, ib) + F + 1"),
                  *P.sel(pr.add(la, lb), {"opmul": 1.0}, "W"),
                  *P.sel(pr.add(la, {"fb": 1.0, "N": 1.0}), {"opdiv": 1.0}, "W")]

    l16 = b.at(16)
    # lo0: the first replaced position (reduce: a's sign, or the '(' when the brackets go)
    l16.units += P.sel(pr.add(af, one(1.0), {"drop": -1.0}), {"one": 1.0}, "lo0")

    l17 = b.at(17)
    # the reduce step's kind, as exact flags (an AND form is <= 0, not 0, when it fails)
    for k, f in (("add", ("m_add", "act_red")), ("sub", ("m_sub", "act_red")), ("mul", ("opmul", "act_red")),
                 ("div", ("opdiv", "act_red")), ("pow", ("opc", "act_red"))):
        l17.units += P.flag(P.all_of(*f), f"pl_{k}", f"'pl_{k}': the step reduces with {f[0]}")
    l17.units += P.flag(pr.add({"act_red": 1.0, "opc": -1.0}), "pl_num", "'pl_num': the result is a number")
    l17.heads.append(P.gather("sign before the redex?", pr.add({"lo0": 1.0}, one(-1.0)), {"is_unary": {"u_lo": 1.0}}))

    # every C_ dim: new value - the value T handed the root (written at the root only)
    new: dict[str, Form] = {
        "C_sp": {"act_sp": 1.0}, "C_br": {"act_br": 1.0}, "C_red": {"act_red": 1.0},
        **{f"C_{k}": {f"pl_{k}": 1.0} for k in ("add", "sub", "mul", "div", "pow", "num")},
        "C_ofz": {"fz": 1.0}, "C_X": {"X": 1.0},
        "C_mlo": pr.add({"p": 1.0, "cb": 1.0}, one(1.0)), "C_mhi": pr.add({"c": 1.0}, one(-1.0)),
        "C_ulo": {"u_lo": 1.0}, "C_F": {"F": 1.0}, "C_W": {"W": 1.0}, "C_sa": {"sa_x": 1.0}, "C_sx": {"sx": 1.0},
        "C_PA": pr.add({"o": 1.0, "fa": -1.0, "hd_a": -1.0, "F": 1.0}, one(-1.0)),
        "C_alo": {"F": 1.0, "fa": -1.0}, "C_ahi": pr.add({"F": 1.0, "ia": 1.0}, one(-1.0)),
        "C_PB": pr.add({"b_hi": 1.0, "fb": -1.0, "hd_b": -1.0, "F": 1.0}, one(-1.0)),
        "C_blo": {"F": 1.0, "fb": -1.0}, "C_bhi": pr.add({"F": 1.0, "ib": 1.0}, one(-1.0)),
        "C_MA": pr.add({"o": 1.0}, one(-1.0)), "C_MB": pr.add({"b_hi": 1.0}, one(-1.0)),
        "C_fa": {"fa": 1.0}, "C_fb": {"fb": 1.0}, "C_hda": {"hd_a": 1.0}, "C_hdb": {"hd_b": 1.0},
        "C_la": la, "C_lb": lb,
        "C_AF": pr.add({"a_lo": 1.0}, one(1.0)), "C_ia": {"ia": 1.0}, "C_lB": {"ib": 1.0, "fb": 1.0, "fa": 1.0},
    }
    lc = b.at(CTX_LAYER)
    # C_lo / C_hi: spaces -1 (never reached), brackets p - cb .. c, reduce lo0 .. b_hi - 1 + drop
    # (a power also drops a sign that cancels, power.py ``pc``)
    lc.units += P.sel(one(-1.0), P.all_of("act_sp", "is_eq"), "C_lo")
    lc.units += P.sel({"p": 1.0, "cb": -1.0}, P.all_of("act_br", "is_eq"), "C_lo")
    lc.units += P.sel(pr.add({"lo0": 1.0}, neg({"pc": 1.0})), P.all_of("act_red", "is_eq"), "C_lo")
    lc.units += P.sel({"c": 1.0}, P.all_of("act_br", "is_eq"), "C_hi")
    lc.units += P.sel(pr.add({"b_hi": 1.0, "drop": 1.0}, one(-1.0)), P.all_of("act_red", "is_eq"), "C_hi")
    for dim in ("C_lo", "C_hi"):
        lc.units += P.sel(neg({dim: 1.0}), ROOT, dim)
    for dim, form in new.items():
        lc.units += P.sel(neg({dim: 1.0}), ROOT, dim)
        lc.units += P.sel(form, P.all_of("is_eq", *PLAN_GATE.get(dim, ())), dim)
