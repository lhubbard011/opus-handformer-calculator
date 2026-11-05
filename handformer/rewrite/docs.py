"""What every dim of the looped calculator means (the visualizer's Residual stream notes).

Display only. One sentence per dim, or per family of dims (one per marker class, digit,
template segment, broadcast copy). "At the root" marks the values only the root's are read:
every cell computes them, from what it can see.
"""

from __future__ import annotations

from .. import common as cm
from . import circuit, division, power

ROOT = "At the root (every cell computes it; only the root, which sees the whole tape, is read): "

MARKER_NAMES = {
    "Lg": "'('", "Lrp": "')'", "Lmul": "'*' or '/'", "Lpow": "'^'", "Ladd": "'+' or a binary '-'",
    "Lnd": "a byte that is not part of a number", "Ldot": "'.'", "Lnz": "a nonzero digit", "Ptil": "'~'",
    "P": "a binary operator", "Lc": "a cell with content (not a blank, the start byte or the root)",
}

FIXED = {
    # the parse
    "prev_num": "1 if the byte before this cell is part of a number (a digit or '.').",
    "prev_rpar": "1 if the byte before this cell is ')'.",
    "is_unary": "1 on a '-' that is a sign: the byte before it is not part of a number and not ')'.",
    "is_binminus": "1 on a '-' that subtracts (after a number or ')').",
    "X": ROOT + "where the expression ends: the first '~', blank, or the root.",
    "f_X": "Tag for X: BIG - position on a '~', a blank or the root, so the latest head finds the earliest.",
    "SP": ROOT + "the first blank (or the root): where the trailing blanks start once no space is left inside.",
    "f_SP": "Tag for SP: BIG - position on a blank or the root.",
    "act_sp": ROOT + "1 when a space has content after it: this pass removes the first one.",
    "p": ROOT + "the innermost group's '(' (0, the start byte, when there is none).",
    "c_raw": ROOT + "the first ')' after p.", "fc": ROOT + "1 if there is a ')' after p.",
    "c": ROOT + "where the innermost group ends: the first ')' after p, or X.",
    "om": ROOT + "the first '*' or '/' after p.", "fm": ROOT + "1 if there is a '*' or '/' after p.",
    "oa": ROOT + "the first '+' or binary '-' after p.",
    "opw": ROOT + "the latest '^' before c.", "Pg": ROOT + "the latest binary operator before c.",
    "use_mul": ROOT + "1 if a '*' or '/' lies inside the group.", "pw": ROOT + "1 if a '^' lies inside the group.",
    "bin_in": ROOT + "1 if the group holds a binary operator (else it holds one number).",
    "gp": ROOT + "1 if the group is a real '(' (not the line start).",
    "o": ROOT + "the redex's operator: the latest '^' in the group, else the first '*' '/', else the first '+' '-'.",
    "act_red": ROOT + "1 when this pass works on the redex a o b.",
    "act_br": ROOT + "1 when the innermost group holds one number: this pass drops its brackets.",
    "a_lo": ROOT + "the byte before a's digits (a's sign, or the byte before a).",
    "sb_x": ROOT + "1 if b has a sign '-'.", "sa_x": ROOT + "1 if a has a sign '-'.",
    "g_plus": ROOT + "the operator byte is '+' (as read).", "g_minus": ROOT + "the operator byte is '-' (as read).",
    "g_star": ROOT + "the operator byte is '*' (as read).", "g_caret": ROOT + "the operator byte is '^' (as read).",
    "g_slash": ROOT + "the operator byte is '/' (as read).",
    "opp": ROOT + "1: the redex adds.", "opm": ROOT + "1: the redex subtracts.", "opmul": ROOT + "1: the redex multiplies.",
    "opdiv": ROOT + "1: the redex divides (long division over passes).",
    "opc": ROOT + "1: the redex is a power (a text rewrite).",
    "u_p": ROOT + "1 if a sign '-' stands right before the group's '('.",
    "sneg_p": ROOT + "1 if the group's number starts with '-'.",
    "a_dot": ROOT + "the latest '.' before o (a's '.' when it is after a_lo).", "hd_a": ROOT + "1 if a has a '.'.",
    "b_dot": ROOT + "the latest '.' before b_hi (b's '.' when it is after o).", "hd_b": ROOT + "1 if b has a '.'.",
    "b_hi": ROOT + "the byte after b's digits.",
    "fa": ROOT + "a's fractional digits.", "fb": ROOT + "b's fractional digits.", "la": ROOT + "all of a's digits.",
    "e1": ROOT + "1 if a starts right after the group's '('.", "e2": ROOT + "1 if b ends at the group's ')'.",
    "sub": ROOT + "1 when + or - subtracts magnitudes: a's sign and b's sign as added (b's XOR the '-') differ.",
    "cb": ROOT + "1 for -(-v): the brackets and both signs go.",
    "drop": ROOT + "1 when the redex is the whole group: its brackets go too.",
    "m_add": ROOT + "1: this pass adds magnitudes.", "m_sub": ROOT + "1: this pass subtracts magnitudes.",
    "m_mul": ROOT + "1: a multiplication pass (accumulating, or writing).",
    "F": ROOT + "fractional digits of the result's columns: max(fa, fb) for + -, fa + fb for *, N for /.",
    "Wm": ROOT + "the top column (one above the largest the result can need).",
    "CB": ROOT + "the cell of the result's column 0 (b's last digit; a quotient's last stored digit).",
    "final": ROOT + "1 once every digit of b has been added into K: the multiplication writes this pass.",
    "sk_b": ROOT + "1 if b's digit kk is left of b's '.' (one more cell to skip).",
    "bk": ROOT + "b's digit kk, the one this multiplication pass adds |a| times.",
    "macc": ROOT + "1: this pass adds |a| * b_kk * 10^kk into K and rewrites nothing.",
    "offA": ROOT + "where a column cell finds its digit of a: q + offA (one less left of the '.').",
    "offB": ROOT + "where a column cell finds its digit of b: q + offB (one less left of the '.').",
    "offM": ROOT + "where a column cell finds digit i of |a| for a product: q + offM.",
    "u_lo": ROOT + "1 if a sign '-' stands right before the replaced bytes (a negative result cancels it).",
    "nkk": ROOT + "the digit counter's next value (the adapter carries it into kk).",
    "rwn": ROOT + "1: this pass writes a number.", "rwp": ROOT + "1: this pass writes a power's rewrite.",
    "grow_any": ROOT + "1: this pass asks for room and keeps every byte.",
    # columns
    "c1": "1 if this cell's column is at most the top column.", "c2": "1 if this cell's column is at least 0.",
    "fence": "1 on the cell after column 0: every carry or borrow chain stops there.",
    "fenceB": "1 on column 0's cell.", "ge0": "1 if this column is worth 10^0 or more (left of the '.').",
    "ha_lo": "1 if a has a digit this low.", "ha_hi": "1 if a has a digit this high.",
    "hb_lo": "1 if b has a digit this low.", "hb_hi": "1 if b has a digit this high.",
    "v1_lo": "Product: digit i = j - kk of |a| exists (i >= 0).", "v1_hi": "Product: digit i of |a| exists (i < la).",
    "v2_lo": "Product: digit i - 1 of |a| exists.", "v2_hi": "Product: digit i - 1 of |a| exists (below la).",
    "skp1": "Product: digit i is left of a's '.', one more cell to skip.", "skp2": "Product: digit i - 1 is left of a's '.'.",
    "av": "a's digit for this column, as gathered.", "bv": "b's digit for this column, as gathered.",
    "in_col": "1 if this cell is one of the result's columns this pass.",
    "aj": "This column's digit of a (of the remainder, in a division).", "bj": "This column's digit of b (of B, in a division).",
    "P1": "Product: a_i * b_kk, whose units land in this column.", "P2": "Product: a_(i-1) * b_kk, whose tens land in this column.",
    "eprev": "Product: the tens (0..2) of the column below's sum, carried up one column.",
    "ge10": "This column's sum is 10 or more: a carry stops here and carries.",
    "le8": "This column's sum is 8 or less: a carry stops here and does not carry.",
    "gt": "a's digit is larger here: b - a borrows from this column.", "lt": "a's digit is smaller here: a - b borrows from this column.",
    "cy": "The carry into this column: the first lower column whose sum is not 9 decides it.",
    "bw1": "The borrow into this column for a - b.", "bw2": "The borrow into this column for b - a.",
    "nK": "The next value of K at this cell (the adapter carries it).",
    # the result's shape
    "neg1": ROOT + "a - b's top column reads 9: it borrowed past the top, so |a| < |b|.",
    "ltm": ROOT + "1: subtracting and |a| < |b|, so the result is b - a.",
    "topS": ROOT + "the result's top nonzero column's cell.", "botS": ROOT + "the result's bottom nonzero column's cell.",
    "zero": ROOT + "1 if the result is 0.", "sgnw": ROOT + "1 if the result is written with a '-'.",
    "cancel": ROOT + "1 if a negative result cancels the sign '-' right before it (-(-v) = v).",
    "ni": ROOT + "integer digits written (at least one: '0').", "Fp": ROOT + "fractional digits written (trailing zeros dropped).",
    "hasfrac": ROOT + "1 if the result has a '.'.",
    "LO": ROOT + "the first cell this pass replaces.", "M": ROOT + "how many bytes the replacement has.",
    "G": ROOT + "how far to the right the cells after it read their byte (negative: to the left).",
    "DS": ROOT + "where result digit k is found: at cell q + DS (one less after the '.').",
    "CS": ROOT + "where a bracketed number's bytes are found: at q + CS.", "NI": ROOT + "ni, for the broadcast.",
    "act_go": ROOT + "1 if this pass does anything (the router's decision).",
    # the write
    "go": "The root's decision, broadcast: 1 if this pass changes anything.",
    "lo_B": "LO, broadcast.", "m_B": "M, broadcast.", "g_B": "G, broadcast.", "ds_B": "DS, broadcast.", "cs_B": "CS, broadcast.",
    "sg_B": "sgnw, broadcast.", "ni_B": "ni, broadcast.", "hf_B": "hasfrac, broadcast.", "ltm_B": "ltm, broadcast.",
    "red_B": "1 if this pass writes a number (broadcast).", "br_B": "1 if this pass drops brackets (broadcast).",
    "R_B": "The root's position, broadcast.",
    "gelo": "1 if this cell is at or after LO.", "gelo2": "1 if this cell is past the result's sign.",
    "ltend": "1 if this cell is before LO + M (inside the replacement).",
    "blank": "1 if the byte G cells away would be the root or beyond: this cell becomes a blank.",
    "atdot": "1 on the result's '.'.", "afterdot": "1 after the result's '.' (its digits are one column further).",
    # carried, router
    "K": "Carried between passes: a column of a product being accumulated, or of a division's remainder.",
    "kk": "Carried at the root: the digit of b a multiplication is on, or the digit of A a division has brought down.",
    "Q": "Carried: a quotient digit, kept in a blank cell until the division writes its result.",
    "qc": "Carried at the root: how many times b has been subtracted for the current quotient digit.",
    "route": "The router: 1 sends this position through the block again. Every tape cell takes the root's decision.",
    "grow": "The root asks for room: the runtime inserts one blank before the root.",
    # power
    "bnzp": ROOT + "the latest nonzero digit up to b's end.", "blast": ROOT + "b's last digit.",
    "bodd": ROOT + "1 if b's last digit is odd.", "bfirst": ROOT + "b's first digit.",
    "bnz": ROOT + "1 if b has a nonzero digit.", "lb1": ROOT + "1 if b is one byte.", "is1": ROOT + "1 if b's last digit is 1.",
    "b01": ROOT + "1 if b's first digit is 0 or 1.", "lb2": ROOT + "1 if b has two bytes or more.",
    "T0": ROOT + "1: a^-n becomes (1/a^n).", "T1": ROOT + "1: a^0 becomes 1.", "T2": ROOT + "1: a^1 becomes a.",
    "T3": ROOT + "1: b odd, a^b becomes (a*a^(b-1)).", "T4": ROOT + "1: b even, a^b becomes (a*a)^(b/2).",
    "b1": ROOT + "1 if b is the text '1'.", "skip": ROOT + "1 if halving b leaves a leading 0, which is dropped.",
    "pc": ROOT + "1 for -(-a^1): the signs cancel.", "mp": ROOT + "the rewrite's length.",
    "off3": ROOT + "where segment 3 (a) reads its bytes.", "off5": ROOT + "where segment 5 (a) reads its bytes.",
    "off8": ROOT + "where segment 8 (b) reads its bytes.", "off9": ROOT + "where segment 9 reads b's digits.",
    "growp": ROOT + "1: the power's rewrite does not fit: ask for room.",
    "pw_B": "1 if this pass writes a power's rewrite (broadcast).", "t3_B": "T3, broadcast.", "t4_B": "T4, broadcast.",
    "o3_B": "off3, broadcast.", "o5_B": "off5, broadcast.", "o8_B": "off8, broadcast.", "o9_B": "off9, broadcast.",
    "P9odd": "The digit of b before this one is odd (the halving's 5).",
    # division
    "has_til": "1 if a '~' is at or before this cell.", "f_Ndg": "Tag for the latest digit of N.",
    "Ndg": "The latest digit of N up to this cell.", "N": ROOT + "the precision N (0 without a '~').",
    "inA": ROOT + "1 if A's digit t is one of a's (not a padding 0).", "geia": ROOT + "1 if A's digit t is past a's '.'.",
    "t1": ROOT + "1 once a digit of A has been brought down.", "At_raw": ROOT + "A's digit t, as gathered.",
    "At": ROOT + "A's digit t (0 for the padding zeros).", "dfinal": ROOT + "1 once every quotient digit is stored.",
    "growd": ROOT + "1: the blanks cannot hold the quotient: ask for room.",
    "dwork": ROOT + "1: a long-division pass (shift or subtract; the tape keeps its bytes).",
    "m_dfin": ROOT + "1: the pass that writes the quotient.", "B_div": "dwork, broadcast.", "B_dfin": "m_dfin, broadcast.",
    "B_hdb": "hd_b, broadcast.", "d_lo": "B has a digit this low (fa zeros first).", "d_hi": "B has a digit this high.",
    "skpb": "B's digit is left of b's '.'.", "bvd": "B's digit for this column.", "dK": "r - B at this column, exactly.",
    "dsub": ROOT + "1: r >= B, subtract B this pass.", "dsub_B": "dsub, broadcast.", "dw_B": "dwork, broadcast.",
    "At_B": "A's digit t, broadcast.", "qc_B": "qc, broadcast.", "QS_B": "Where the quotient's digits are stored (SP), broadcast.",
    "t_B": "t (kk), broadcast.", "t1_B": "t1, broadcast.", "Kn": "The remainder digit one column down.",
    "shK": "That digit, rebuilt exactly: the next K when shifting.", "st_at": "1 on the cell that stores this quotient digit.",
    "nQ": "The next value of Q at this cell.", "nqc": ROOT + "qc's next value.",
    # coda
    "cf_X": "Coda tag: BIG - position on a '~', blank or '='.", "cf_R": "Coda tag: BIG - position on a '='.",
    "cX": "Coda: where the final tape's expression ends.", "cR": "Coda: the root's position.",
    "inx": "Coda: 1 while the tape byte this position copies is part of the expression.",
    "O_nl": "Coda: past the expression, end the answer with '\\n'.",
}

BROADCAST = {"B_bhi": "CB + 1", "B_W": "Wm", "B_F": "F", "B_fa": "fa", "B_ia": "a's integer digits", "B_fb": "fb",
             "B_ib": "b's integer digits", "B_la": "la", "B_hda": "hd_a", "B_kk": "kk", "B_bk": "bk",
             "B_offA": "offA", "B_offB": "offB", "B_offM": "offM", "B_red": "act_red", "B_add": "m_add",
             "B_sub": "m_sub", "B_mul": "m_mul", "B_macc": "macc", "B_pmr": "m_add + m_sub"}


def docs() -> dict[str, str]:
    out = dict(FIXED)
    for m, what in MARKER_NAMES.items():
        out[m] = f"The latest {what} up to this cell (0 if none)."
        out[f"f_{m}"] = f"Tag for {m}: 2·(position+1) on {what}, so a latest head can find the most recent one."
    for c in circuit.NEXT:
        out[f"PM_{c}"] = f"The latest {MARKER_NAMES[c]} strictly before this cell (a next-match key)."
        out[f"PM2_{c}"] = f"PM_{c} squared (a next-match key)."
    for c in ("Lrp", "Lmul", "Ladd"):
        out[f"Lt_{c}"] = ROOT + f"the latest {MARKER_NAMES[c]} at or before p."
    for k, what in (("A", "a column that stops a carry chain"), ("S", "a column that stops a borrow chain")):
        out[f"f_m{k}"] = f"Tag for {what} (or the fence)."
        out[f"Lm{k}"] = f"The latest {what} up to this cell."
        out[f"PMm{k}"] = f"The latest {what} strictly before this cell (a next-match key)."
        out[f"PM2m{k}"] = f"PMm{k} squared (a next-match key)."
    for k, what in (("A", "a+b, a-b or a*b (or a quotient)"), ("B", "b-a")):
        out[f"f_t{k}"] = f"Tag for the top nonzero column of {what}."
        out[f"f_b{k}"] = f"Tag for the bottom nonzero column of {what}."
        out[f"top{k}"] = ROOT + f"the top nonzero column's cell of {what}."
        out[f"bot{k}"] = ROOT + f"the bottom nonzero column's cell of {what}."
        out[f"z{k}"] = ROOT + f"1 if {what} is 0."
        for d in range(10):
            out[f"R{k}{d}"] = f"This column's result digit of {what} is {d}."
            out[f"G{k}{d}"] = f"The result digit this cell writes is {d} ({what})."
    for d in range(10):
        if d:  # a 0 contributes nothing to the sums these feed
            out[f"A1_{d}"] = f"Product: digit i of |a| is {d}."
            out[f"A2_{d}"] = f"Product: digit i - 1 of |a| is {d}."
            out[f"L1_{d}"] = f"Product: a_i * b_kk ends in {d}."
            out[f"L2_{d}"] = f"Product: a_(i-1) * b_kk ends in {d}."
            out[f"DU{d}"] = f"Product: this column's sum ends in {d}."
        out[f"P9_{d}"] = f"Power: the digit of b this cell reads is {d}."
    for byte, dim in circuit.CONTENT.items():
        ch = chr(byte) if byte != circuit.BLANK else "blank"
        out[f"S_{dim}"] = f"The byte G cells away is '{ch}'."
        out[f"C_{dim}"] = f"The bracketed number's byte for this cell is '{ch}'."
        for s in (3, 5, 8):
            out[f"P{s}_{dim}"] = f"Power: segment {s}'s copied byte is '{ch}'."
    for byte, dim in circuit.HOLD.items():
        ch = chr(byte) if byte not in (cm.BOS, circuit.BLANK) else ("start byte" if byte == cm.BOS else "blank")
        out[dim] = f"Hold '{ch}' next: the adapter re-embeds this byte as the cell's next state."
    for s in range(power.SEG):
        if s < power.SEG - 1:
            out[f"L{s}"] = ROOT + f"the length of the power template's segment {s}."
        if s:
            out[f"S{s}_B"] = f"Where the power template's segment {s} starts, broadcast."
            out[f"sg{s}"] = f"1 if this cell is at or past segment {s}'s start."
    for k in range(division.N_DIGITS):
        out[f"Nd{k}"] = ROOT + f"N's 10^{k} digit."
        if k:
            out[f"qN{k}"] = ROOT + f"the position of N's 10^{k} digit."
    for name, what in BROADCAST.items():
        out[name] = f"{what}, broadcast from the root to every cell."
    for byte, dim in circuit.ANSWER.items():
        out[f"A_{dim}"] = f"Coda: the tape byte this position copies is '{chr(byte)}'."
        out[f"O_{dim}"] = f"Coda: write '{chr(byte)}'."
    return out
