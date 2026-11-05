"""Independent reference for the recurrent-depth calculator (handformer/loop, LOOPED.md).

Pure Python, written separately from the weights. After the prompt ``\\n<expr>[~N]=`` the model
thinks in latent positions and then writes the answer. The latent positions come in steps; a
step reads the latest copy of the expression (the prompt's, at first) and writes the next:

    root  '='   the end of a copy: it parses the copy and plans the step (or halts)
    start       holds the plan
    working     reduce steps only: the result's digits, one position per column
    copy        the next expression, one byte per position, then the next root

The step's kind, in this order (the same rules as the frame-writing calculator, except that
one step removes every space):

1. spaces: the copy without its spaces (and without the prompt's "~N");
2. brackets: the innermost group holds one number: drop its brackets ("-(-v)": the signs too);
3. reduce: the redex (innermost group, precedence, left to right) becomes its value, the
   rewrite of tests/reference.py ``step``;
4. halt: the copy is a number (or a division by zero is next): it is the answer.

Working, per operator (column j is worth 10^(j-F); F fractional digits):

* ``+ -``: columns 0..W-1, W = max(ia, ib) + F + 1, of |a|+|b| when the signs as added agree;
  otherwise two digits each, |a|-|b| and |b|-|a| mod 10^W (the second is used when |a| < |b|).
* ``*``: columns 0..la+lb-1 of |a|*|b| (digits as integers, F = fa + fb). The position before
  column j iterates once per term a_i * b_(j-i) (at least once): its depth is the router's.
* ``/``: A = |a|'s digits and fb + N zeros, B = |b|'s digits and fa zeros, L = len(A). For each
  digit t of A: a quotient position q_t = floor((10 r + A_t) / B), found by the position before
  it in len(B) + 1 iterations (a column each), then len(B) positions of the remainder. Then a
  row of the quotient's L columns (F = N), as for the other operators.
* ``^``: no working; the copy holds the binary-exponentiation step (tests/reference.py
  ``power_expansion``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tests import reference as ref


@dataclass
class Step:
    kind: str  # "spaces" | "brackets" | "reduce" | "halt"
    copy: str  # the expression after the step (for "halt": the answer)
    op: str = ""
    columns: list[tuple[int, int]] = field(default_factory=list)  # + - * and the quotient row: (digit, other digit)
    quotient: list[int] = field(default_factory=list)  # / : the quotient digits q_t
    remainders: list[list[int]] = field(default_factory=list)  # / : remainder digits after each q_t, LSB first
    depths: list[int] = field(default_factory=list)  # iterations of the positions that predict each working position


def _operands(x: str):
    """(p, c, o, afence, b_hi) of the innermost redex: the same choice as tests/reference.py ``step``."""
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
    return p, c, o, afence, b_hi


def _digits(n: int, width: int) -> list[int]:
    """n's digits, least significant first, ``width`` of them (n >= 0)."""
    return [(n // 10 ** j) % 10 for j in range(width)]


def _working(x: str, precision: int) -> Step:
    _, _, o, afence, b_hi = _operands(x)
    a, op, b = x[afence + 1:o], x[o], x[o + 1:b_hi]
    ref.PRECISION = precision
    st = Step("reduce", ref.step(x)[1], op)
    if op == "^":
        return st
    ia_s, fa = ref.split_decimal(a.lstrip("-"))
    ib_s, fb = ref.split_decimal(b.lstrip("-"))
    if op in "+-":
        f = max(fa, fb)
        ma, mb = int(ia_s) * 10 ** (f - fa), int(ib_s) * 10 ** (f - fb)
        ia, ib = len(ia_s) - fa, len(ib_s) - fb
        w = max(ia, ib) + f + 1
        sub = a.startswith("-") != ((op == "-") != b.startswith("-"))  # the signs as added differ
        if sub:
            st.columns = list(zip(_digits((ma - mb) % 10 ** w, w), _digits((mb - ma) % 10 ** w, w)))
        else:
            st.columns = [(d, 0) for d in _digits(ma + mb, w)]
        st.depths = [1] * w
    elif op == "*":
        ma, mb = int(ia_s), int(ib_s)
        la, lb = len(ia_s), len(ib_s)
        w = la + lb
        st.columns = [(d, 0) for d in _digits(ma * mb, w)]
        st.depths = [max(1, min(j, la - 1) - max(0, j - lb + 1) + 1) for j in range(w)]
    else:  # "/"
        big_a = ia_s + "0" * (fb + precision)
        big_b = ib_s + "0" * fa
        nb, ib_v = len(big_b), int(big_b)
        r = 0
        for dg in big_a:
            x_t = 10 * r + int(dg)
            q, r = divmod(x_t, ib_v)
            st.quotient.append(q)
            st.remainders.append(_digits(r, nb))
            st.depths += [nb + 1] + [1] * nb  # q_t's predictor scans the columns; each remainder digit once
        w = len(big_a)
        st.columns = [(d, 0) for d in _digits(int("".join(map(str, st.quotient))), w)]
        st.depths += [1] * w
    return st


def _zero_divisor(x: str) -> bool:
    _, _, o, _, b_hi = _operands(x)
    return x[o] == "/" and not any(ch in "123456789" for ch in x[o + 1:b_hi])


def steps(expression: str) -> list[Step]:
    """Every step after the prompt, the last one a halt whose ``copy`` is the answer."""
    precision = int(expression.split("~")[1].replace(" ", "")) if "~" in expression else 0
    x, out = expression, []
    while True:
        line = x.split("~")[0]
        if " " in line:
            x = line.replace(" ", "")
            out.append(Step("spaces", x))
            continue
        line = x.split("~")[0]
        if ref.bare_group(line):
            x = ref.bracket_step(line)[1]
            out.append(Step("brackets", x))
        elif any(ref.binary_op(line, i) for i in range(len(line))) and not _zero_divisor(line):
            st = _working(line, precision)
            out.append(st)
            x = st.copy
        else:
            out.append(Step("halt", line))
            return out


def answer(expression: str) -> str:
    return steps(expression)[-1].copy
