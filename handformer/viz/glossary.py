"""Plain-language meanings for residual dims and heads, for the visualizer's hover text.

Display only: nothing here reaches the weights. Exact names first, then
patterns for dim families (oh0..oh9, act_*, ...). Private operator dims are
looked up without their circuit prefix (``mul.J`` -> ``J`` in MUL).
"""
from __future__ import annotations

import re

SHARED = {
    # ---- input (set by the byte embedding)
    "one": "Always 1. Acts as the constant term (the model has no biases).",
    "pos": "This byte's position in the sequence (0, 1, 2, ...).",
    "pos2": "Position squared. Lets attention aim at one exact position, and doubles as a table of squares.",
    "val": "Digit value 0-9 of this byte (0 for anything that isn't a digit).",
    "is_digit": "1 if this byte is a digit.",
    "nz": "1 if this byte is a nonzero digit (1-9).",
    "n9": "1 if this byte is a digit other than 9.",
    "is0": "1 if this byte is the digit 0.", "is9": "1 if this byte is the digit 9.",
    "is_bos": "1 for the newline at position 0 (start of the sequence).",
    "is_eq": "1 if this byte is '='.", "is_comma": "1 if this byte is ',' (between multiplication lines).",
    "is_bar": "1 if this byte is '|' (ends the multiplication lines).",
    "is_semi": "1 if this byte ends an operation's working: ';' or '<'.",
    "is_lt": "1 if this byte is '<': a subtraction found |a| < |b|.",
    "is_lbr": "1 if this byte is '[' (opens a working step).", "is_rbr": "1 if this byte is ']' (closes a step).",
    "is_lpar": "1 if this byte is '('.", "is_rpar": "1 if this byte is ')'.", "is_dot": "1 if this byte is '.'.",
    "is_plus": "1 if '+'.", "is_minus": "1 if '-' (sign or subtraction).", "is_star": "1 if '*'.",
    "is_slash": "1 if '/'.", "is_caret": "1 if '^'.", "is_opany": "1 for any operator byte.",
    "is_tilde": "1 if '~' (precision marker in the prompt).",
    # ---- frame locator
    "prev_digit": "1 if the previous byte is part of a number (a digit or '.').",
    "prev_rpar": "1 if the previous byte is ')'.",
    "is_unary": "1 if this '-' is a negative sign (previous byte isn't part of a number or ')').",
    "is_binminus": "1 if this '-' means subtract.",
    "F": "Position of the most recent '[': where the current working step started.",
    "Rb": "Position of the most recent ']': where the last step closed.",
    "E": "Position of the most recent '='.", "S": "Position of the most recent ';' or '<' (end of working).",
    "R": "Position of the most recent '|' (multiplication).", "P": "Position of the most recent operator.",
    "Lg": "Position of the most recent '(' or ']' (start of the current group).",
    "Lrp": "Position of the most recent ')'.", "Lmul": "Position of the most recent '*' or '/'.",
    "Lpow": "Position of the most recent '^'.", "Ladd": "Position of the most recent '+' or subtraction '-'.",
    "Lnd": "Position of the most recent byte that isn't part of a number.",
    "Ldot": "Position of the most recent '.'.", "Lnz": "Position of the most recent nonzero digit.",
    "Ptil": "Position of the prompt's '~' (0 if there is none).",
    "open": "1 while a working step [ ... ] is open.",
    "seen_eq": "1 once the current step's '=' is written.", "seen_semi": "1 once the current step's answer has started.",
    "seen_bar": "1 once the multiplication lines are finished ('|' written).",
    "Ls": "Start of the line being reduced (the ']' before it, or position 0).",
    "p": "Start of the innermost group in that line: its latest '(' (or the line start).",
    "Pf": "Position of this step's operator. Read at the step's '=', so it never moves.",
    "neg": "1 if the step's working ended with '<' (|a| < |b| in a subtraction).",
    "sa": "1 if this step's first number is negative.", "sb": "1 if this step's second number is negative.",
    "PF2": "Square of the first number's length (multiplication uses it for line spacing).",
    "f_E0": "Tag used to find the prompt's '=' (the earliest one).", "E0": "Position of the prompt's '='.",
    "has_til": "1 if the prompt gives a precision with '~'.", "N": "Division precision: digits after the point (from '~N').",
    "phase1": "Job: write '[' to start a new step.", "phase2": "Job: copy the chosen piece into the step.",
    "phase4": "Job: write the next, simpler line (or still reading the prompt).",
    # ---- parser / rewriter
    "c": "End of the innermost group: its ')' (or the line's end).", "c_raw": "First ')' after the group start.",
    "fc": "1 if a ')' was found.", "om": "First '*' or '/' after the group start.", "fm": "1 if one was found.",
    "oa": "First '+' or '-' after the group start.", "fa": "1 if one was found.",
    "use_mul": "1 if a '*' or '/' lies inside the group (it goes before + and -).",
    "opw": "Latest '^' before the group's end.", "pw": "1 if that '^' is inside the group (powers go first).",
    "o": "Position of the operator chosen for this step.",
    "a_lo": "Last byte before the first number (or its sign).", "sa_x": "1 if the first number has a '-' sign.",
    "sb_x": "1 if the second number has a '-' sign.", "b_hi": "First byte after the second number.",
    "op_div": "1 if the chosen operator is '/'.", "op_pow": "1 if the chosen operator is '^'.",
    "op_md": "1 if the chosen operator is '*' or '/'.",
    "e1": "1 if the first number starts right after the group's '('.", "e2": "1 if the second number ends at the group's ')'.",
    "e3": "1 if the group really is in brackets.", "drop": "1 if the brackets disappear after this step.",
    "cont": "1 if another operator remains, so there is another step.",
    "cont2": "1 if there is another step (another operator, or the result is itself an expression).",
    "a_dot": "Latest '.' before the operator.", "b_dot": "Latest '.' in the second number's span.",
    "has_a": "1 if the first number has a decimal point.", "has_b": "1 if the second number has a decimal point.",
    "frac_a": "Digits after the first number's '.'.", "frac_b": "Digits after the second number's '.'.",
    "pad_a": "Zeros added after the first number in the copy (aligning decimals, or division precision).",
    "pad_b": "Zeros added after the second number in the copy.",
    "f_res": "How many digits of the result belong after the decimal point.",
    "tz": "Trailing zeros removed from the result's fraction.", "fp": "Digits after the point once trailing zeros are gone.",
    "wd": "Width of the result's digit string after zero-padding.", "dotf": "1 if the result has a decimal point.",
    "rneg": "1 if the step's result is negative.", "cancel": "1 if two minus signs cancel: -(-v) becomes v.",
    "res_op": "Latest operator at the end of the result (detects a result that is an expression).",
    "res_expr": "1 if the result is an expression (a power step writes one).",
    "res_wrap": "1 if the result is fully wrapped in brackets.",
    "src2": "Which earlier byte to copy next while copying the chosen piece.",
    "src4": "Which earlier byte to copy next while writing the simpler line.",
    "in1": "1 while writing the part of the line before the solved piece.",
    "in12": "1 while writing the line up to the end of the result.", "in123": "1 while writing anything in the line.",
    "end2": "1 when the copied piece is complete: write '='.", "end4": "1 when the new line is complete.",
    "isSign": "1 when writing the result's minus sign.", "isDot": "1 when writing the result's decimal point.",
    "padzero": "1 when writing a padding zero of the result (e.g. the 0s in 0.02).",
    "Lend": "Where copying the old line stops (before the prompt's '~N').",
    "LopLe": "Latest operator anywhere in the line (to see if another one remains).",
    "LopO": "Latest operator before the chosen one (to see if another one remains).",
    "cont_raw": "Counts the reasons another step is needed (before rounding to yes/no).",
    "prevA_unary": "1 if the byte just before the solved piece is a minus sign.",
    "Lnzr": "Position of the result's last nonzero digit (to count trailing zeros).",
    "found_nz": "1 if the result has any nonzero digit (it isn't 0).",
    "pad_a_raw": "How much shorter the first number's fraction is (before choosing padding).",
    "pad_b_raw": "How much shorter the second number's fraction is.",
    "tzx": "Trailing zeros beyond the fraction length (removed from the count).",
    "inA": "1 while copying the first number into the step.", "inPA": "1 while writing its padding zeros.",
    "atOp": "1 when copying the operator.", "inB": "1 while copying the second number.",
    "inPB": "1 while writing the second number's padding zeros.",
    "skipA": "1 once the copy of the first number has passed its '.' (skip one byte).",
    "skipB": "1 once the copy of the second number has passed its '.'.",
    "til_line": "1 if this line still carries the prompt's '~N'.",
    "afterdot": "1 for result digits after the decimal point.",
    # ---- outputs
    "work_end": "Votes for ';': the working is finished.", "work_end_neg": "Votes for '<': finished, and |a| < |b|.",
    "frame_end": "Votes for ']': the step is finished.", "emit_minus": "Votes for '-' at the start of an answer.",
    "emit_comma": "Votes for ',' between multiplication lines.", "emit_bar": "Votes for '|' after the last line.",
    "out_lbr": "Votes for '['.", "out_eq": "Votes for '='.", "out_nl": "Votes for the final newline: done.",
}

PATTERNS = [
    (r"^oh(\d)$", "One-hot digit: 1 if this byte is the digit {0}."),
    (r"^D(\d)$", "Vote for writing the digit {0} next (an operator circuit's output)."),
    (r"^O_oh(\d)$", "Vote for writing the digit {0} next (copied or padding)."),
    (r"^O_(is_\w+)$", "Vote for writing '{0}' next (copied)."),
    (r"^f_(\w+)$", "Landmark tag for {0}: nonzero only on bytes that count as a {0} landmark "
                  "(it grows with position), so the 'latest {0}' head can pick out the most recent one."),
    (r"^PM_(\w+)$", "The most recent {0} landmark strictly before this byte (used to search forward)."),
    (r"^PM2_(\w+)$", "Square of PM_{0} (for the forward-search score)."),
    (r"^Lt_(\w+)$", "Most recent {0} landmark at the group start (start point of a forward search)."),
    (r"^typ_(\w+)$", "1 if this step's operator is the {0} operator."),
    (r"^act_(\w+)$", "1 while the {0} calculator is switched on: only its outputs can reach the logits."),
    (r"^tN(\d)$", "Where precision digit {0} (from the right) sits in the prompt."),
    (r"^Nd(\d)$", "Precision digit {0} (from the right), read from the prompt."),
    (r"^Y2_((?:is_|oh)\w+)$", "The byte being copied into the step is '{0}' (frame copy)."),
    (r"^Y4_((?:is_|oh)\w+)$", "The byte being copied into the new line is '{0}' (rewrite)."),
    (r"^Boh(\d)$", "Multiplication: the current digit of b is {0} (one-hot)."),
    (r"^Bo(\d)$", "Division: the divisor's last digit is {0} (one-hot)."),
    (r"^(gt|lt)(\d)$", "Division: this remainder digit is {0}-greater/less than multiple {1}'s digit."),
    (r"^(GT|LT)(\d)$", "Division: the remainder is greater/less than {1}×divisor at its top differing digit."),
    (r"^c0(\d)$", "Division: 1 if the next digit is at least multiple {0}'s last digit (tie-break)."),
    (r"^dkey(\d)$", "Division: tag marking remainder digits that differ from multiple {0}."),
    (r"^m(\d)$", "Division: digit of multiple {0}×divisor aligned with this remainder digit."),
]

OPS = {
    "add": {"tA": "Where digit i of a is.", "tB": "Where digit i of b is.", "tA1": "Where digit i-1 of a is.",
            "tB1": "Where digit i-1 of b is.", "exA": "1 once a has no digit i.", "exB": "1 once b has no digit i.",
            "A": "Digit i of a.", "B": "Digit i of b.", "A1": "Digit i-1 of a.", "B1": "Digit i-1 of b.",
            "carry": "Carry into this digit, recovered from the previous output digit.",
            "Z": "Position of the sum's top nonzero digit.", "Zo": "How many digits the answer has, minus 1.",
            "C": "The working digit being copied into the answer.", "sgn": "1 if the answer gets a '-'.",
            "zkey": "Tag on nonzero working digits."},
    "sub": {"br": "Borrow into this digit, recovered from the previous output digit.",
            "Zp": "Top nonzero digit of the working.", "Z9": "Top working digit that isn't 9.",
            "Zlow": "Lowest nonzero working digit.", "top": "Where the answer's first digit comes from.",
            "topneg": "Top digit when the answer is negative.", "Dv": "The working digit being read.",
            "above": "1 above the lowest nonzero digit (complement as 9-d).", "at": "1 at the lowest nonzero digit (10-d).",
            "V": "The answer digit to write.", "sgn": "1 if the answer gets a '-'."},
    "mul": {"J": "Which multiplication line the next digit belongs to.", "Am": "Digit m of a.",
            "Ym": "Digit sum before the carry: T[m+1] + a_m·b_j.", "Yp": "Previous column's sum, for the carry.",
            "Fl": "floor(previous sum / 10).", "carry": "Carry into this digit (recovered).",
            "sep_slot": "1 at the end of a line: write ','.", "lastline": "1 on the last line.",
            "low": "1 while gathering the product's low digits.", "Cv": "Product digit being gathered.",
            "Av": "Answer digit being copied.", "sgn": "1 if the answer gets a '-'."},
    "div": {"row": "Row of the long-division grid for the next byte.", "rowS": "This byte's own row.",
            "qd": "The quotient digit to write.", "qv": "This row's quotient digit.", "ai": "Next digit of the dividend.",
            "br": "Borrow while subtracting the multiple.", "carry": "Carry while building the multiples.",
            "found_q": "1 if the quotient has a nonzero digit.", "start": "Where the answer starts in the grid.",
            "Av": "Answer digit being copied.", "sgn": "1 if the answer gets a '-'."},
    "pow": {"negexp": "Case: negative exponent → (1/a^n).", "zero": "Case: exponent 0 → 1.", "unit": "Case: exponent 1 → a.",
            "odd": "Case: odd exponent → (a*a^(n-1)).", "even": "Case: even exponent → (a*a)^(n/2).",
            "ld": "Exponent's last digit.", "src": "Which byte to copy into the expansion.", "lead": "1 if halving drops a leading 0."},
}


CHAR = {"is_plus": "+", "is_minus": "-", "is_star": "*", "is_slash": "/", "is_caret": "^",
        "is_lpar": "(", "is_rpar": ")", "is_dot": ".", **{f"oh{d}": str(d) for d in range(10)}}


def _fmt(text: str, groups: tuple) -> str:
    out = text.format(*(CHAR.get(g, g) for g in groups))
    # A name built from a landmark (f_R, PM_Lg, Lt_Lmul, ...): say what that landmark is.
    for g in groups:
        if g in SHARED and not g.isdigit():
            out += f" ({g} = {SHARED[g][0].lower()}{SHARED[g][1:].rstrip('.')}.)"
    return out


def describe_dim(name: str) -> str:
    if "." in name:
        ns, local = name.split(".", 1)
        doc = OPS.get(ns, {}).get(local)
        if doc:
            return f"{ns} calculator: {doc}"
        for pat, text in PATTERNS:
            m = re.match(pat, local)
            if m:
                return f"{ns} calculator: " + _fmt(text, m.groups())
        return f"{ns} calculator: private working value '{local}'."
    if name in SHARED:
        return SHARED[name]
    for pat, text in PATTERNS:
        m = re.match(pat, name)
        if m:
            return _fmt(text, m.groups())
    return ""


def describe_head(spec) -> str:
    """Say what a head does, from the exact shape of its keys (the primitives are fixed shapes)."""
    keys = [tuple(sorted(k)) for k in spec.key]
    reads = ", ".join(spec.value)
    writes = ", ".join(dict.fromkeys(d for dst in spec.value.values() for d in dst))
    if keys == [("pos",), ("pos2",)]:
        kind = ("Exact lookup: its score −β(k−t)² peaks at one position t that this byte computed, "
                "so it reads exactly that byte.")
    elif keys == [("pos2",), ("pos",)]:
        kind = "Integer division by attention: every position bids, and the score peaks exactly at floor(x/s)."
    elif len(keys) == 3:
        kind = ("Forward search: finds the first byte of a class after a starting point. Each byte stores "
                "where the previous match was; the target is the only match whose stored pointer equals "
                "the latest match at the start.")
    elif len(keys) == 1:
        kind = ("Latest match: picks the most recent byte carrying the tag it looks for "
                "(tags grow with position, so the newest wins).")
    else:
        kind = "Attention head."
    return f"{kind} It reads {reads} there and writes {writes}."
