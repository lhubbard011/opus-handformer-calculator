"""Decimal operands: alignment, point placement, trailing-zero stripping.

Every case is checked byte for byte against the reference, and the
reference's answer is checked against exact rational arithmetic.
"""

import logging
import random
import unittest
from decimal import Decimal
from fractions import Fraction

from tests import reference
from tests.helpers import check, model, rand_digits

log = logging.getLogger(__name__)

LITERALS = ["0", "0.5", "1.25", "-2.5", "10.01", "0.001", "3", "-0.75", "100.1", "-7", "9.99", ".5"]
PREC = {"+": 1, "-": 1, "*": 2}


def exact(text: str) -> Fraction:
    return Fraction(Decimal(text if not text.lstrip("-").startswith(".") else text.replace(".", "0.", 1)))


def answer(expression: str) -> Fraction:
    return exact(reference.output(expression).rsplit("]", 1)[-1].strip())


def apply(op: str, x: Fraction, y: Fraction) -> Fraction:
    return x + y if op == "+" else x - y if op == "-" else x * y


def gen(rng: random.Random, depth: int, root: bool = True):
    """Random tree with decimal leaves as (text, exact value, precedence)."""
    if depth == 0 or (not root and rng.random() < 0.3):
        whole = rng.randint(0, 10 ** rng.randint(0, 3))
        frac = rand_digits(rng, rng.randint(0, 3))
        text = f"{whole}.{frac}" if frac else str(whole)
        if rng.random() < 0.25:
            text = "-" + text
        return text, exact(text), 3
    op = rng.choice("+-*")
    lt, lv, lp = gen(rng, depth - 1, False)
    rt, rv, rp = gen(rng, depth - 1, False)
    if lp < PREC[op]:
        lt = f"({lt})"
    if rp <= PREC[op]:
        rt = f"({rt})"
    return f"{lt}{op}{rt}", apply(op, lv, rv), PREC[op]


class TestDecimals(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model()

    def test_literal_pairs(self):
        for op in "+-*":
            for a in LITERALS:
                for b in LITERALS:
                    e = f"{a}{op}{b}"
                    self.assertEqual(answer(e), apply(op, exact(a), exact(b)), e)
                    check(e)
            log.info("%s: all %d literal pairs ok", op, len(LITERALS) ** 2)

    def test_formatting(self):
        cases = {"1.5+1.5": "3", "0.1+0.2": "0.3", "0.1*0.2": "0.02", "2.5*4": "10",
                 "0.5-1.25": "-0.75", "1.10-0.1": "1", "0.0*5.5": "0", "-0.5+0.5": "0",
                 "0.001*0.001": "0.000001", "99.99+0.01": "100"}
        for e, want in cases.items():
            self.assertEqual(reference.output(e).rsplit("]", 1)[-1].strip(), want, e)
            check(e)

    def test_long_fractions(self):
        rng = random.Random(5)
        for op, n in (("+", 60), ("-", 60), ("*", 6)):
            for _ in range(4):
                a = f"{rand_digits(rng, rng.randint(1, n))}.{rand_digits(rng, rng.randint(1, n))}"
                b = f"{rand_digits(rng, rng.randint(1, n))}.{rand_digits(rng, rng.randint(1, n))}"
                e = f"{a}{op}{b}"
                self.assertEqual(answer(e), apply(op, exact(a), exact(b)), e)
                check(e)
            log.info("%s: long fractions ok", op)

    def test_random_trees(self):
        rng = random.Random(6)
        for depth in (2, 3):
            for _ in range(15):
                text, value, _ = gen(rng, depth)
                self.assertEqual(answer(text), value, f"reference disagrees with exact value on {text}")
                check(text)
            log.info("random decimal trees depth<=%d ok", depth)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    unittest.main()
