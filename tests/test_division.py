"""Division: long division grid, precision ~N, signs, decimals, expressions.

Every case is checked byte for byte against the reference, and the
reference's answer against exact rational arithmetic: each '/' truncates
toward zero to N fractional digits.
"""

import logging
import random
import unittest
from decimal import Decimal
from fractions import Fraction

from tests import reference
from tests.helpers import check, model, rand_digits

log = logging.getLogger(__name__)
PREC = {"+": 1, "-": 1, "*": 2, "/": 2}


def exact(text: str) -> Fraction:
    return Fraction(Decimal(text))


def answer(expression: str) -> Fraction:
    return exact(reference.output(expression).rsplit("]", 1)[-1].strip())


def divide(x: Fraction, y: Fraction, n: int) -> Fraction:
    q = x / y * 10**n
    q = q.numerator // q.denominator if q >= 0 else -((-q.numerator) // q.denominator)
    return Fraction(q, 10**n)


def apply(op: str, x: Fraction, y: Fraction, n: int) -> Fraction:
    return x + y if op == "+" else x - y if op == "-" else x * y if op == "*" else divide(x, y, n)


def gen(rng: random.Random, depth: int, n: int, root: bool = True):
    """Random tree with '/' (nonzero divisors) as (text, value, precedence)."""
    if depth == 0 or (not root and rng.random() < 0.3):
        whole = rng.randint(0, 10 ** rng.randint(0, 2))
        frac = rand_digits(rng, rng.randint(0, 2))
        text = f"{whole}.{frac}" if frac else str(whole)
        if rng.random() < 0.2:
            text = "-" + text
        return text, exact(text), 3
    op = rng.choice("+-*//")
    lt, lv, lp = gen(rng, depth - 1, n, False)
    rt, rv, rp = gen(rng, depth - 1, n, False)
    if op == "/" and rv == 0:
        rt, rv, rp = "7", Fraction(7), 3
    if lp < PREC[op]:
        lt = f"({lt})"
    if rp <= PREC[op]:
        rt = f"({rt})"
    return f"{lt}{op}{rt}", apply(op, lv, rv, n), PREC[op]


class TestDivision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model()

    def run_case(self, e: str, want: Fraction) -> None:
        self.assertEqual(answer(e), want, f"reference disagrees with exact value on {e}")
        check(e)

    def test_small_integers(self):
        for n in (0, 3):
            for a in range(0, 31):
                for b in range(1, 13):
                    self.run_case(f"{a}/{b}~{n}" if n else f"{a}/{b}", divide(Fraction(a), Fraction(b), n))
            log.info("all a/b, a < 31, b < 13, N=%d ok", n)

    def test_signs_and_decimals(self):
        for a in ["-7", "7", "0.5", "-2.25", "100.01", "0"]:
            for b in ["-3", "3", "0.4", "-0.125", "12.5"]:
                for n in (0, 2, 5):
                    self.run_case(f"{a}/{b}~{n}", divide(exact(a), exact(b), n))
        log.info("signed / decimal operands ok")

    def test_long(self):
        rng = random.Random(7)
        for la, lb, n in ((30, 3, 0), (20, 6, 4), (8, 2, 12)):
            for _ in range(3):
                a = rand_digits(rng, la).lstrip("0") or "1"
                b = rand_digits(rng, lb).lstrip("0") or "1"
                self.run_case(f"{a}/{b}~{n}", divide(Fraction(int(a)), Fraction(int(b)), n))
            log.info("long %d/%d digits, N=%d ok", la, lb, n)

    def test_expressions(self):
        rng = random.Random(8)
        for e, want in [("2+6/3*2", 6), ("12/4/3", 1), ("(1+2)/(3-4)~2", -3)]:
            self.run_case(e, Fraction(want))
        for depth, n in ((2, 2), (3, 3)):
            for _ in range(10):
                text, value, _ = gen(rng, depth, n)
                self.run_case(f"{text}~{n}", value)
            log.info("random trees with '/' depth<=%d ~%d ok", depth, n)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    unittest.main()
