"""Integer powers: binary-exponentiation rewrite frames.

Every case is checked byte for byte against the reference, and the
reference's answer against exact arithmetic (a^-n = 1/a^n, truncated to ~N).
"""

import logging
import random
import unittest
from decimal import Decimal
from fractions import Fraction

from tests import reference
from tests.helpers import check, model

log = logging.getLogger(__name__)


def exact(text: str) -> Fraction:
    return Fraction(Decimal(text))


def answer(expression: str) -> Fraction:
    return exact(reference.output(expression).rsplit("]", 1)[-1].strip())


def trunc(x: Fraction, n: int) -> Fraction:
    q = x * 10**n
    q = q.numerator // q.denominator if q >= 0 else -((-q.numerator) // q.denominator)
    return Fraction(q, 10**n)


def power(a: Fraction, e: int, n: int = 0) -> Fraction:
    return a**e if e >= 0 else trunc(1 / a ** (-e), n)


class TestPower(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model()

    def run_case(self, e: str, want: Fraction) -> None:
        self.assertEqual(answer(e), want, f"reference disagrees with exact value on {e}")
        check(e)

    def test_small(self):
        for a in range(-5, 6):
            for k in range(0, 7):
                self.run_case(f"{a}^{k}", Fraction(a) ** k)
        log.info("all a^k, a in -5..5, k in 0..6 ok")

    def test_decimal_and_negative_exponents(self):
        for a, k, n in [("1.5", 2, 0), ("0.1", 3, 0), ("-0.5", 3, 0), ("2.5", 4, 0),
                        ("2", -3, 4), ("3", -2, 5), ("-2", -3, 3), ("0.5", -2, 0), ("10", -1, 2),
                        ("4", -3, 6), ("1.5", -1, 3)]:
            e = f"{a}^{k}~{n}" if n else f"{a}^{k}"
            self.run_case(e, power(exact(a), k, n))
        log.info("decimal bases and negative exponents ok")

    def test_precedence(self):
        cases = {"2^3^2": 512, "(2^3)^2": 64, "2*3^2+1": 19, "-2^2": 4, "2^(1+2)": 8,
                 "(1+1)^(2+1)": 8, "3^2*2^3": 72, "2^2^2^2": 65536, "1+2^0": 2,
                 "1/2^3~4": Fraction(1, 8), "6/2^-1~2": 12, "2^3/2^2": 2, "-2^3*-1": 8}
        for e, want in cases.items():
            self.run_case(e, Fraction(want))
        log.info("precedence and right-associativity ok")

    def test_large(self):
        for a, k in [(2, 64), (3, 40), (7, 20), (12, 12), (2, 100)]:
            self.run_case(f"{a}^{k}", Fraction(a) ** k)
        log.info("large powers ok")

    def test_random_mixed(self):
        rng = random.Random(9)
        for _ in range(15):
            a, b = rng.randint(-9, 9), rng.randint(0, 5)
            c, d = rng.randint(1, 30), rng.randint(0, 4)
            op = rng.choice("+-*")
            e = f"{a}^{b}{op}{c}^{d}"
            x, y = Fraction(a) ** b, Fraction(c) ** d
            self.run_case(e, x + y if op == "+" else x - y if op == "-" else x * y)
        log.info("random mixed expressions ok")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    unittest.main()
