"""Independent reference for the looped calculator (handformer/rewrite): the tape after every pass.

Pure Python string processing, written separately from the weights. The tape is what the
positions between the start byte and the root (the prompt's '=') hold: the expression, the
prompt's "~N" if any, then blanks (spaces) that rewrites leave at the end. One pass does
exactly one of these, chosen in this order (REWRITE.md §4):

1. a space with something after it: remove the first one (the rest moves left one cell);
2. the innermost group holds one number: drop its brackets; "-(-v)" drops both signs too;
3. a redex (innermost group, precedence, left to right): replace it by its value, the same
   rewrite as tests/reference.py ``step`` (+ and - in one pass; * after one pass per digit of b);
4. none of these: halt. The answer is the expression.

A rewrite longer than the bytes it replaces (a power's "(a*a)^h") uses blanks after the
expression; while there are too few, a pass only asks for room and the runtime adds one blank.

The final answer must also equal the frame-writing calculator's (tests/reference.py).
"""

from __future__ import annotations

from tests import reference as ref

END = "~ "  # what ends the expression on the tape (besides the tape's end)


def expression_end(s: str) -> int:
    return min([i for i, ch in enumerate(s) if ch in END], default=len(s))


def _mul_passes(x: str) -> int:
    """Passes a multiplication spends accumulating before the one that writes: one per digit
    of |b| (its '.' and sign are not digits)."""
    _, _, b = _operands(x)
    return sum(ch.isdigit() for ch in b)


def _operands(x: str) -> tuple[str, str, str]:
    """(a, op, b) of the innermost redex, the same choice as tests/reference.py ``step``."""
    p = x.rfind("(")
    c = x.find(")", p + 1) if p >= 0 else len(x)
    span = range(p + 1, c)
    o = next((i for i in reversed(span) if x[i] == "^"), None)
    if o is None:
        o = next((i for i in span if x[i] in "*/"), None)
    if o is None:
        o = next(i for i in span if x[i] in "+-" and ref.binary_op(x, i))
    a_lo = max(i for i in range(-1, o) if i == -1 or not ref.is_num(x[i]))
    afence = a_lo - 1 if a_lo >= 0 and ref.unary(x, a_lo) else a_lo
    sb = o + 1 < len(x) and x[o + 1] == "-"
    b_hi = next(i for i in range(o + 1 + sb, len(x) + 1) if i == len(x) or not ref.is_num(x[i]))
    return x[afence + 1:o], x[o], x[o + 1:b_hi]


def step(s: str) -> str | None:
    """The tape after one pass, or None when the pass halts (the tape is final)."""
    content = [i for i, ch in enumerate(s) if ch != " "]
    sp = s.find(" ")
    if sp >= 0 and content and sp < content[-1]:
        return s[:sp] + s[sp + 1:] + " "
    end = expression_end(s)
    x, rest = s[:end], s[end:]
    if ref.bare_group(x):
        nxt = ref.bracket_step(x)[1]
    elif any(ref.binary_op(x, i) for i in range(len(x))):
        op = _operands(x)[1]
        if op not in "+-*/^":
            raise NotImplementedError(f"operator {op!r} is not in the looped model yet")
        nxt = ref.step(x)[1]  # the same reduction as the frame-writing calculator's
    else:
        return None
    blanks = len(s) - len(s.rstrip(" "))
    if len(nxt) - len(x) > blanks:  # no room: the root asks for a blank, the tape waits a pass
        return s + " "
    return (nxt + rest + " " * (len(x) - len(nxt)))[:len(s)]  # a longer rewrite uses up blanks


def _precision(s: str) -> int:
    end = expression_end(s)
    digits = "".join(ch for ch in s[end:] if ch.isdigit())
    return int(digits) if digits else 0


def _division(x: str, precision: int) -> tuple[int, int]:
    """(L, passes): the quotient's digit count, and the long division's passes before the one
    that writes: a shift for every digit of A and one more, and a subtraction per unit of
    every quotient digit (handformer/rewrite/division.py)."""
    a, _, b = _operands(x)
    ia, fa = ref.split_decimal(a.lstrip("-"))
    ib, fb = ref.split_decimal(b.lstrip("-"))
    big_a = ia + "0" * (fb + precision)
    big_b = int(ib + "0" * fa)
    r, passes = 0, 1
    for dg in big_a:
        r = 10 * r + int(dg)
        q, r = divmod(r, big_b)
        passes += q + 1
    return len(big_a), passes


def tapes(expression: str) -> list[str]:
    """The tape going into every pass; the last one is the tape the pass that halts reads.
    A multiplication reads the same tape once per digit of b (accumulating), then rewrites; a
    division first asks for room for its quotient, then works for many passes, then rewrites."""
    out = [expression]
    while True:
        s = out[-1]
        end = expression_end(s)
        x = s[:end]
        content = [i for i, ch in enumerate(s) if ch != " "]
        reducing = (" " not in s[:content[-1] if content else 0] and not ref.bare_group(x)
                    and any(ref.binary_op(x, i) for i in range(len(x))))
        ref.PRECISION = _precision(s)
        if reducing and _operands(x)[1] == "*":
            out += [s] * _mul_passes(x)
        elif reducing and _operands(x)[1] == "/":
            digits, passes = _division(x, ref.PRECISION)
            blanks = len(s) - len(s.rstrip(" "))
            for _ in range(max(0, digits + 3 - blanks)):  # room for the quotient, a blank a pass
                s += " "
                out.append(s)
            out += [s] * passes
        nxt = step(s)
        if nxt is None:
            return out
        out.append(nxt)


def answer(expression: str) -> str:
    final = tapes(expression)[-1]
    return final[:expression_end(final)]
