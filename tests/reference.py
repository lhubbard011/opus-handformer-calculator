"""Independent reference for the exact byte sequence the model must write.

Pure Python string processing, written separately from the weights. Every
test compares the model's output against this, one byte at a time.
"""



def unary(x: str, i: int) -> bool:
    """A '-' is a sign when the byte before it is not part of a number and not ')'."""
    return x[i] == "-" and (i == 0 or not (x[i - 1].isdigit() or x[i - 1] in ".)"))


def binary_op(x: str, i: int) -> bool:
    return x[i] in "+*/" or (x[i] == "-" and not unary(x, i))


def magnitude_body(a: str, op: str, b: str) -> tuple[str, str, int]:
    """An op circuit's working on magnitudes: (working, end byte, signed value)."""
    ia, ib = int(a), int(b)
    if op == "+":
        s = ia + ib
        return str(s).zfill(max(len(a), len(b)))[::-1], ";", s
    if op == "-":
        n = max(len(a), len(b))
        d = ia - ib
        return str(d % 10**n).zfill(n)[::-1], (";" if d >= 0 else "<"), d
    la, lb = len(a), len(b)
    t, lines = 0, []
    for j in range(lb):
        t = t // 10 + ia * int(b[lb - 1 - j])
        lines.append(str(t).zfill(la + 1)[::-1])
    prod = ia * ib
    return ",".join(lines) + "|" + str(prod).zfill(la + lb)[::-1], ";", prod


def division_body(a: str, b: str) -> tuple[str, int]:
    """Long-division grid for magnitudes a / b (see handformer/division.py)."""
    ia, ib, L = int(a), int(b), len(b)
    rows = [str(k * ib).zfill(L + 1)[::-1] + "," for k in range(1, 10)]
    rows.append("0" * (L + 2))
    r, q = 0, 0
    for digit in a:
        r = 10 * r + int(digit)
        qi = r // ib
        r -= qi * ib
        q = 10 * q + qi
        rows.append(str(qi) + str(r).zfill(L + 1)[::-1])
    return "".join(rows), q


def working(a: str, op: str, b: str) -> str:
    """Frame body after '=' for signed operands: working, end byte, result.

    '*' multiplies magnitudes, sign = sa XOR sb. '+'/'-' compare a's sign with
    b's effective sign (flipped by '-'): equal -> add magnitudes with a's sign;
    different -> subtract magnitudes, sign = sa XOR borrow. Never "-0".
    """
    sa, sb = a.startswith("-"), b.startswith("-")
    ma, mb = a.lstrip("-"), b.lstrip("-")
    if op == "/":
        work, v = division_body(ma, mb)
        return work + ";" + ("-" if sa != sb and v else "") + str(v)
    if op == "*":
        work, end, v = magnitude_body(ma, "*", mb)
        neg = sa != sb
    elif sa == (sb != (op == "-")):
        work, end, v = magnitude_body(ma, "+", mb)
        neg = sa
    else:
        work, end, v = magnitude_body(ma, "-", mb)
        neg = sa != (v < 0)
    v = abs(v)
    return work + end + ("-" if neg and v else "") + str(v)


def is_num(ch: str) -> bool:
    return ch.isdigit() or ch == "."


def split_decimal(operand: str) -> tuple[str, int]:
    """Signed decimal text -> (signed integer digits without '.', fraction length)."""
    sign = "-" if operand.startswith("-") else ""
    body = operand.lstrip("-")
    if "." not in body:
        return sign + body, 0
    whole, frac = body.split(".")
    return sign + whole + frac, len(frac)


def format_result(result: str, f: int) -> str:
    """Integer result text with f implied fractional digits -> canonical decimal.

    Trailing fractional zeros are stripped, the '.' is inserted, and a
    result shorter than its fraction is left-padded with zeros. Zero stays "0".
    """
    sign = "-" if result.startswith("-") else ""
    digits = result.lstrip("-")
    if int(digits) == 0:
        return "0"
    tz = min(len(digits) - len(digits.rstrip("0")), f)
    f -= tz
    digits = digits[: len(digits) - tz]
    if f == 0:
        return sign + digits
    w = max(len(digits), f + 1)
    padded = digits.zfill(w)
    return sign + padded[: w - f] + "." + padded[w - f:]


def step(x: str) -> tuple[str, str, bool]:
    """One reduction of line x: (frame, next line, more steps needed)."""
    p = x.rfind("(")
    c = x.find(")", p + 1) if p >= 0 else len(x)
    span = range(p + 1, c)
    o = next((i for i in span if x[i] in "*/"), None)
    if o is None:
        o = next(i for i in span if x[i] in "+-" and binary_op(x, i))
    a_lo = max(i for i in range(-1, o) if i == -1 or not is_num(x[i]))
    afence = a_lo - 1 if a_lo >= 0 and unary(x, a_lo) else a_lo
    sb = o + 1 < len(x) and x[o + 1] == "-"
    b_hi = next(i for i in range(o + 1 + sb, len(x) + 1) if i == len(x) or not is_num(x[i]))
    a, op, b = x[afence + 1:o], x[o], x[o + 1:b_hi]

    ia, fa = split_decimal(a)
    ib, fb = split_decimal(b)
    if op == "/":  # a*10^(fb+N) / b*10^fa: the integer quotient has N fractional digits
        f = PRECISION
        ia, ib = ia + "0" * (fb + PRECISION), ib + "0" * fa
    elif op == "*":
        f = fa + fb
    else:  # align the fractions: pad the shorter one with zeros
        f = max(fa, fb)
        ia, ib = ia + "0" * (f - fa), ib + "0" * (f - fb)
    body = working(ia, op, ib)
    raw = body[max(body.rfind(";"), body.rfind("<")) + 1:]
    result = format_result(raw, f)

    drop = p >= 0 and afence == p and b_hi == c
    lo, hi = (p, c) if drop else (afence + 1, b_hi - 1)
    if result.startswith("-") and lo >= 1 and unary(x, lo - 1):
        lo, result = lo - 1, result[1:]  # -(-v) == v
    more = sum(binary_op(x, i) for i in range(len(x))) >= 2
    tail = x[hi + 1:].split("~")[0]  # the prompt's "~N" is not copied forward
    return f"[{ia}{op}{ib}={body}]", x[:lo] + result + tail, more


PRECISION = 0  # set per call by output()


def output(expression: str) -> str:
    """Everything the model writes after the prompt ``\\n<expression>=``.

    ``expression`` may end with ``~N``: N fractional digits for every '/'.
    """
    global PRECISION
    PRECISION = int(expression.split("~")[1]) if "~" in expression else 0
    x, out = expression, []
    while True:
        frame, x, more = step(x)
        out.append(frame + x + ("=" if more else "\n"))
        if not more:
            return "".join(out)
