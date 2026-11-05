"""What every dim of the recurrent-depth calculator means (the visualizer's Residual stream notes).

Display only. One sentence per dim, or per family of dims (one per marker class, digit, byte,
template segment). "At the root" marks the parse values only a root's are read: every
position computes them, from what it can see; a root sees its whole copy.
"""

from __future__ import annotations

from . import parts as P, power

ROOT = "At a root (every position computes it; a root's is read): "
WALK = "Walking (a position that writes a copy cell next): "

MARKER_NAMES = {
    "Lg": "'('", "Lcx": "')' or where a copy ends ('~' or a root)", "Lmul": "'*' or '/'", "Lpow": "'^'",
    "Ladd": "'+' or a binary '-'", "Lnd": "a position that is not part of a number", "Ldot": "'.'",
    "Lnz": "a nonzero digit", "Ptil": "'~' (the prompt's precision marker)", "P": "a binary operator",
    "Lsp": "a space", "Lnc": "a position that is not a cell (a copy starts after the latest one)",
    "E": "a root '='", "Lns": "a position that is not a space",
}

FIXED = {
    # roles and the loop
    "is_cell": "1 on an expression byte (a cell of a copy): digits, operators, brackets, '.', '~', space.",
    "lat": "1 on a latent position (a thought: its input came from T, not a byte).",
    "k_s": "1 on a step's start (the thought after a root that does not halt).",
    "k_w": "1 on a working column (+ - *, or the quotient row of /).",
    "k_q": "1 on a quotient digit of a long division.",
    "k_r": "1 on a remainder digit of a long division.",
    "think": "1: the next position is a thought (T of this one), not a written byte. The host follows it.",
    "route": "1: this position goes through the block again (the router). The host follows it.",
    "n_ks": "The next position is a step's start (a root that does not halt writes it).",
    "z_pre": "This copy cell is before the replaced bytes.",
    "z_mid": "This copy cell is in the replacement (neither flag: after it).",
    "ptr": "The old copy's position this cell copied (pre and suf zones).",
    "mk": "This cell's index in the replacement (mid zone).",
    "nc": "This column's index + 1: the column the position after it computes.",
    "dA": "This column's digit: |a|+|b|, |a|-|b|, |a|*|b| or the quotient.",
    "dB": "This column's digit of |b|-|a| (used when |a| < |b|).",
    "cA": "The carry (+) or borrow (|a|-|b|) out of this column.", "cB": "The borrow out of this column of |b|-|a|.",
    "cm": "The carry out of this product column (any size: up to 9 per term).",
    "tA": "The top nonzero column so far + 1 (0: none), digit set A.", "tB": "The same for digit set B.",
    "lA": "The lowest nonzero column right of the '.' + 1 (0: none so far), digit set A.", "lB": "The same for set B.",
    "qa": "Quotient row: the quotient digit's position for the next column.",
    "qd": "A quotient digit (on q_t and passed along its remainder digits).",
    "qt": "Which digit of A the quotient digit is (t).", "rd": "A remainder digit.",
    "rn": "The remainder digit's index + 1.", "mq": "The carry of q * B out of this remainder column.",
    "br": "The borrow out of this remainder column.",
    "kk": "Loop register: the iteration (a product's term, or the column of x a quotient digit compares).",
    "slo": "Loop register: a product column's accumulator, units digit.", "shi": "Loop register: its tens (any size).",
    # the parse
    "prev_num": "1 if the byte before this position is part of a number (a digit or '.').",
    "prev_rpar": "1 if the byte before this position is ')'.",
    "is_unary": "1 on a '-' that is a sign: the byte before it is not part of a number and not ')'.",
    "is_binminus": "1 on a '-' that subtracts (after a number or ')').",
    "fz": ROOT + "where its copy starts: the latest position before it that is not a cell.",
    "tl": ROOT + "1 if the copy holds the prompt's '~' (it is the prompt's copy).",
    "X": ROOT + "where the copy ends: its '~', else the root itself.",
    "p_raw": ROOT + "the latest '(' before X (perhaps in an older copy).",
    "sp_raw": ROOT + "the latest space before X.",
    "p": ROOT + "the innermost group's '(' in this copy (fz when there is none).",
    "gp": ROOT + "1 if the group is a real '('.", "act_sp": ROOT + "1 if the copy has a space: this step removes them all.",
    "c": ROOT + "where the innermost group ends: the first ')' after p, or X.",
    "om": ROOT + "the first '*' or '/' after p.", "fm": ROOT + "1 if there is a '*' or '/' after p.",
    "oa": ROOT + "the first '+' or binary '-' after p.",
    "opw": ROOT + "the latest '^' before c.", "Pg": ROOT + "the latest binary operator before c.",
    "use_mul": ROOT + "1 if a '*' or '/' lies inside the group.", "pw": ROOT + "1 if a '^' lies inside the group.",
    "bin_in": ROOT + "1 if the group holds a binary operator (else it holds one number).",
    "o": ROOT + "the redex's operator: the latest '^', else the first '*' '/', else the first '+' '-'.",
    "act_red": ROOT + "1: this step reduces the redex.", "act_br": ROOT + "1: this step drops the group's brackets.",
    "a_lo": ROOT + "the position before a's digits (a's sign, or what precedes a).",
    "a_dot": ROOT + "the latest '.' before o (a's, when after a_lo).", "sb_x": ROOT + "1 if b has a sign.",
    "u_p": ROOT + "1 if a sign '-' stands before the group's '('.",
    "sneg_p": ROOT + "1 if the group's number is negative.", "hd_a": ROOT + "1 if a has a '.'.",
    "sa_x": ROOT + "1 if a has a sign.", "b_hi": ROOT + "the position after b's digits.",
    "e1": ROOT + "1 if a starts right after the group's '('.", "e2": ROOT + "1 if b ends at the group's ')'.",
    "sub": ROOT + "1 if the signs as added differ: the magnitudes are subtracted.",
    "fa": ROOT + "a's fractional digits.", "fb": ROOT + "b's fractional digits.",
    "cb": ROOT + "-(-v): both signs go with the brackets.", "b_dot": ROOT + "the latest '.' before b_hi.",
    "bnzp": ROOT + "the latest nonzero digit before b_hi.", "hd_b": ROOT + "1 if b has a '.'.",
    "bz": ROOT + "1 if b has no nonzero digit.",
    "drop": ROOT + "1 if the redex is the whole group: its brackets go too.",
    "halt": ROOT + "1: the copy is a number (or a division by zero is next): it is the answer.",
    "m_add": ROOT + "1: add the magnitudes.", "m_sub": ROOT + "1: subtract the magnitudes.",
    "has_til": "1 if the prompt's '~' is at or before this position.", "seen_eq": "1 if a root is at or before here.",
    "f_Ndg": "Tag for the precision's digits: 2·(position+1) on a digit after the prompt's '~', in the prompt.",
    "Ndg": ROOT + "the precision's last digit.", "N": ROOT + "the precision N (fractional digits of a quotient).",
    "lo0": ROOT + "the first replaced position of a reduce step.",
    "u_lo": ROOT + "1 if a sign '-' stands just before the replaced bytes (it cancels a negative result).",
    "F": ROOT + "the result's fractional digits: max(fa, fb), fa + fb, or N.",
    "W": ROOT + "the result's columns.", "sx": ROOT + "1 if exactly one operand is negative.",
    "ia": ROOT + "a's integer digits.", "ib": ROOT + "b's integer digits.",
    # the plan
    "C_sp": "Plan: a spaces step.", "C_br": "Plan: a brackets step.", "C_red": "Plan: a reduce step.",
    "C_add": "Plan: add the magnitudes.", "C_sub": "Plan: subtract the magnitudes (both differences).",
    "C_mul": "Plan: multiply.", "C_div": "Plan: divide (long division).", "C_pow": "Plan: a power's rewrite.",
    "C_num": "Plan: the replacement is a number from the working.",
    "C_ofz": "Plan: where the old copy starts (the walk begins there).", "C_X": "Plan: where the old copy ends.",
    "C_lo": "Plan: the first replaced old position (-1 for a spaces step).", "C_hi": "Plan: the last replaced one.",
    "C_mlo": "Plan (brackets): the group's number starts here.", "C_mhi": "Plan (brackets): and ends here.",
    "C_ulo": "Plan: a sign '-' stands before the replaced bytes.", "C_F": "Plan: the result's fractional digits.",
    "C_W": "Plan: the working's columns.", "C_sa": "Plan: a is negative.", "C_sx": "Plan: exactly one operand is negative.",
    "C_PA": "Plan (+-): a's digit for column j is at C_PA - j (+1 right of the '.').",
    "C_alo": "Plan (+-): a's lowest column.", "C_ahi": "Plan (+-): a's top column.",
    "C_PB": "Plan (+-): b's digit for column j is at C_PB - j (+1 right of the '.').",
    "C_blo": "Plan (+-): b's lowest column.", "C_bhi": "Plan (+-): b's top column.",
    "C_MA": "Plan (* /): a's last digit.", "C_MB": "Plan (* /): b's last digit.",
    "C_fa": "Plan: a's fractional digits.", "C_fb": "Plan: b's fractional digits.",
    "C_hda": "Plan: a has a '.'.", "C_hdb": "Plan: b has a '.'.", "C_la": "Plan: a's digits.", "C_lb": "Plan: b's digits.",
    "C_AF": "Plan (/): a's first digit.", "C_ia": "Plan (/): a's integer digits.", "C_lB": "Plan (/): B's digits (b's and fa zeros).",
    # the walk and the answer
    "wk": "1 on a latent copy cell.", "ws": "1 on the position before a copy (a step's start, or the last column).",
    "wt": "1 if this position writes a copy cell next.", "ans": "1 if this position writes a byte of the answer next.",
    "row_end": "1 on the working's last column.",
    "ptrE": WALK + "the old position the walk is at.", "preE": WALK + "1 in the pre zone (or starting).",
    "afz": "The latest root's copy start (the answer copies that copy).",
    "useB": WALK + "1 if |a| < |b| in a subtraction: the result is b - a.",
    "top1": WALK + "the result's top nonzero column + 1 (0: the result is 0).",
    "low1": WALK + "the lowest nonzero column right of the '.' + 1 (0: none).",
    "nzr": WALK + "1 if the result is not 0.", "hlow": WALK + "1 if a nonzero digit is right of the '.'.",
    "hiC": WALK + "the number's first column: max(top, F).", "loC": WALK + "its last column.",
    "negr": WALK + "1 if the result is negative.", "cancel": WALK + "1: the sign before the replaced bytes and the result's cancel.",
    "sgn": WALK + "1 if the number is written with a '-'.", "dot": WALK + "1 if the number has a '.'.",
    "nint": WALK + "the number's integer digits.", "mlen": WALK + "the replacement's length.",
    "lo_eff": WALK + "the first replaced old position (one earlier when signs cancel).",
    "ptr1": WALK + "the first non-space old position after ptr.",
    "e_mid": WALK + "1 if the next cell starts the replacement.", "in_mid": WALK + "1 if the next cell continues it.",
    "x_mid": WALK + "1 if this cell is the replacement's last.", "endp": WALK + "1 if the old copy ends at the next position.",
    "to_root": WALK + "1: the next position is the new copy's root.", "to_mid": WALK + "1: the next cell is in the replacement.",
    "to_suf": WALK + "1: the next cell copies an old byte.", "ptrN": WALK + "the old position the next cell copies.",
    "mkN": WALK + "the next cell's index in the replacement.", "mk_dot": WALK + "1 if that is the number's '.'.",
    "mk_frac": WALK + "1 if it is past the '.'.", "dg_addr": WALK + "the working column the next digit comes from.",
    "cp_t": "The position whose byte the next position copies.", "cp_on": "1: the next byte is copied from cp_t.",
    "dg_v": "The digit the next position holds.", "dg_on": "1: the next byte is the digit dg_v.",
    "dgA": WALK + "the working column's digit (set A).", "dgB": WALK + "the working column's digit (set B).",
    # columns
    "PWa": "1 if this position computes the next +- column.", "PWm": "1 if it computes the next product column.",
    "PWq": "1 if it computes the next quotient-row column.", "lastR": "1 on a division's last remainder digit.",
    "ltF": "1 if column nc is right of the '.'.", "ina1": "a has column nc (from below).", "ina2": "a has column nc (from above).",
    "inb1": "b has column nc (from below).", "inb2": "b has column nc (from above).",
    "av_raw": "a's digit for column nc, as gathered.", "bv_raw": "b's, as gathered.",
    "av": "a's digit for column nc (exact).", "bv": "b's digit for column nc (exact).",
    "nzA": "The next column's digit (set A) is not 0.", "nzB": "The same for set B.",
    "noA": "No nonzero digit right of the '.' yet (set A).", "noB": "The same for set B.",
    "i0": "Product: the column's first term i.", "ic": "Product: terms past a's top digit.",
    "skA": "Product: A_i is left of a's '.' (skip it).", "skB": "Product: B_j is left of b's '.'.",
    "valid": "Product: this iteration has a term.", "mB_raw": "Product: B_(c-i) as gathered.", "mB": "Product: B_(c-i).",
    "prod": "Product: A_i * B_(c-i).", "xs": "Product: the accumulator's units plus this term.",
    "fl": "Product: the tens leaving the accumulator.", "cmh": "Product: floor(carry in / 10) (a ruler head).",
    "yge": "Product: the column's units carry.", "qaE": "Quotient row: where this column's quotient digit is.",
    "qv_raw": "Quotient row: the digit as gathered.",
    # division
    "PQ": "1 if this position finds the next quotient digit (it iterates over the columns of x).",
    "PR": "1 if it computes the next remainder digit.", "rlast": "1 on a remainder's last digit.",
    "tlast": "1 if the quotient digit is the last.", "tq": "The digit of A being worked on.",
    "first": "1 if there is no remainder yet (t = 0).", "jx": "The column j of x.",
    "Aok": "A's digit tq is one of a's (not an appended zero).", "skAt": "A's digit tq is past a's '.'.",
    "At_raw": "A's digit, as gathered.", "At": "x_0 = A's digit.", "Bk_lo": "B's digit j is one of b's (j >= fa).",
    "Bk_hi": "j < len(B).", "skBj": "B's digit j is left of b's '.'.", "Bj_raw": "B's digit j, as gathered.",
    "Bj": "B's digit j.", "jge1": "Column j >= 1: x_j is a remainder digit.", "xr_raw": "x_j, as gathered.",
    "x1": "x_j for j >= 1.", "nbr": "How many d have a borrow out: q = 9 - this.",
    "qprod": "Remainder: q * B_j.", "mR": "Remainder: q * B_j plus the carry in.", "flR": "Remainder: that sum's tens.",
    # power
    "blast": ROOT + "b's last digit.", "bodd": ROOT + "1 if b's last digit is odd.", "bfirst": ROOT + "b's first digit.",
    "lb1": ROOT + "b is one byte.", "is1": ROOT + "b's last digit is 1.", "b01": ROOT + "b's first digit is 0 or 1.",
    "lb2": ROOT + "b has two bytes or more.", "T0": ROOT + "template a^-n -> (1/a^n).", "T1": ROOT + "template a^0 -> 1.",
    "b1": ROOT + "b is the text '1'.", "skip": ROOT + "halving b drops a leading 0.", "T2": ROOT + "template a^1 -> a.",
    "T3": ROOT + "template: b odd -> (a*a^(b-1)).", "T4": ROOT + "template: b even -> (a*a)^(b/2).",
    "pc": ROOT + "-(-a^1): the signs cancel.", "mp": ROOT + "the rewrite's length.",
    "P9odd": WALK + "the previous digit of b is odd (halving).",
    "C_T3": "Plan (^): template T3.", "C_T4": "Plan (^): template T4.", "C_mp": "Plan (^): the rewrite's length.",
}


def docs() -> dict[str, str]:
    out = dict(FIXED)
    for m, what in MARKER_NAMES.items():
        out[m] = f"The latest {what} up to here (0 if none)."
        out[f"f_{m}"] = f"Tag for {m}: 2·(position+1) on {what}, so a latest head finds the most recent one."
    for c in ("Lcx", "Lmul", "Ladd", "Lnd", "Lns"):
        out[f"PM_{c}"] = f"The latest {MARKER_NAMES[c]} strictly before here (a next-match key)."
        out[f"PM2_{c}"] = f"PM_{c} squared (a next-match key)."
    for c in ("Lcx", "Lmul", "Ladd"):
        out[f"Lt_{c}"] = ROOT + f"the latest {MARKER_NAMES[c]} at or before p."
    for op, sym in (("plus", "+"), ("minus", "-"), ("star", "*"), ("slash", "/"), ("caret", "^")):
        out[f"g_{op}"] = ROOT + f"the byte at o is '{sym}' (as gathered)."
    for op in ("opp", "opm", "opmul", "opdiv", "opc"):
        out.setdefault(op, ROOT + "the redex's operator (exact).")
    for k, what in (("add", "+ (add magnitudes)"), ("sub", "- (subtract magnitudes)"), ("mul", "*"), ("div", "/"),
                    ("pow", "^"), ("num", "a number result")):
        out[f"pl_{k}"] = ROOT + f"1: the step's kind is {what}."
    for k in range(1, 4):
        out[f"qN{k}"] = ROOT + f"the position of the precision's digit {k} from the right."
    for k in range(4):
        out[f"Nd{k}"] = ROOT + f"the precision's digit {k} from the right."
    for r in ("z_pre", "z_mid", "ptr", "mk", "nc", "dA", "dB", "cA", "cB", "cm", "tA", "tB", "lA", "lB",
              "qa", "qd", "qt", "rd", "rn", "mq", "br"):
        out[f"n_{r}"] = f"The next position's {r} (T moves it there): {FIXED[r][0].lower()}{FIXED[r][1:]}"
    for role, what in (("kw", "a working column"), ("kq", "a quotient digit"), ("kr", "a remainder digit")):
        out[f"n_{role}"] = f"The next position is {what} (T sets its role)."
    for r in ("tA", "tB", "lA", "lB", "cA"):
        out[f"s_{r}"] = WALK + f"the working's last column's {r} (the number's shape)."
    for s in ("kk", "slo", "shi"):
        out[f"x_{s}"] = f"Next iteration's {s} (the adapter moves it there)."
    for d in range(1, 10):
        out[f"c{d}"] = f"Loop register (quotient digit): the carry of {d} * B so far."
        out[f"b{d}"] = f"Loop register (quotient digit): 1 if x < {d} * B so far (a borrow)."
        out[f"x_c{d}"] = f"Next iteration's c{d}."
        out[f"x_b{d}"] = f"Next iteration's b{d}."
        out[f"m{d}"] = f"Quotient digit: column j of {d} * B before its carry out."
        out[f"f{d}"] = f"Quotient digit: the carry out of that column ({d} * B)."
        out[f"mA{d}"] = f"Product: A_i is {d} (as gathered)."
        out[f"qA{d}"] = f"Product: A_i is {d} (exact)."
    for v in range(1, 10):
        out[f"qh{v}"] = f"Remainder: the quotient digit is {v}."
    for byte, dim in P.HOLD.items():
        name = {"\n": "the end byte", " ": "a space"}.get(chr(byte), f"'{chr(byte)}'")
        out[dim] = f"The next byte is {name} (exact 0/1: T embeds it, the coda's logit)."
    for d in (*P.CONTENT.values(), "is_eq"):
        out[f"cpb_{d}"] = f"The copied position's {d} (as gathered)."
    for s in range(power.SEG - 1):
        out[f"Ln{s}"] = ROOT + f"the power template's segment {s} length."
    for s in range(1, power.SEG):
        out[f"C_S{s}"] = f"Plan (^): segment {s} starts at this index of the rewrite."
        out[f"sg{s}"] = WALK + f"the next cell is at or past the power template's segment {s}."
    for s in (3, 5, 8, 9):
        out[f"C_o{s}"] = f"Plan (^): byte k of segment {s} copies old position C_o{s} + k."
    for d in range(1, 10):
        out[f"P9_{d}"] = WALK + f"the exponent's digit for segment 9 is {d}."
    return out
