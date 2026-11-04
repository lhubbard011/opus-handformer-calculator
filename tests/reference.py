"""Independent reference for the exact byte sequence the model must write.

Pure Python string processing, written separately from the weights. Every
test compares the model's output against this, one byte at a time.
"""



def unary(x: str, i: int) -> bool:
    """A '-' is a sign when the byte before it is not a digit and not ')'."""
    return x[i] == "-" and (i == 0 or not (x[i - 1].isdigit() or x[i - 1] == ")"))


def binary_op(x: str, i: int) -> bool:
    return x[i] in "+*" or (x[i] == "-" and not unary(x, i))


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


def working(a: str, op: str, b: str) -> str:
    """Frame body after '=' for signed operands: working, end byte, result.

    '*' multiplies magnitudes, sign = sa XOR sb. '+'/'-' compare a's sign with
    b's effective sign (flipped by '-'): equal -> add magnitudes with a's sign;
    different -> subtract magnitudes, sign = sa XOR borrow. Never "-0".
    """
    sa, sb = a.startswith("-"), b.startswith("-")
    ma, mb = a.lstrip("-"), b.lstrip("-")
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


def step(x: str) -> tuple[str, str, bool]:
    """One reduction of line x: (frame, next line, more steps needed)."""
    p = x.rfind("(")
    c = x.find(")", p + 1) if p >= 0 else len(x)
    span = range(p + 1, c)
    o = next((i for i in span if x[i] == "*"), None)
    if o is None:
        o = next(i for i in span if x[i] in "+-" and binary_op(x, i))
    a_lo = max(i for i in range(-1, o) if i == -1 or not x[i].isdigit())
    afence = a_lo - 1 if a_lo >= 0 and unary(x, a_lo) else a_lo
    sb = o + 1 < len(x) and x[o + 1] == "-"
    b_hi = next(i for i in range(o + 1 + sb, len(x) + 1) if i == len(x) or not x[i].isdigit())
    a, op, b = x[afence + 1:o], x[o], x[o + 1:b_hi]
    body = working(a, op, b)
    result = body[max(body.rfind(";"), body.rfind("<")) + 1:]
    drop = p >= 0 and afence == p and b_hi == c
    lo, hi = (p, c) if drop else (afence + 1, b_hi - 1)
    if result.startswith("-") and lo >= 1 and unary(x, lo - 1):
        lo, result = lo - 1, result[1:]  # -(-v) == v
    more = sum(binary_op(x, i) for i in range(len(x))) >= 2
    return f"[{a}{op}{b}={body}]", x[:lo] + result + x[hi + 1:], more


def output(expression: str) -> str:
    """Everything the model writes after the prompt ``\\n<expression>=``."""
    x, out = expression, []
    while True:
        frame, x, more = step(x)
        out.append(frame + x + ("=" if more else "\n"))
        if not more:
            return "".join(out)
