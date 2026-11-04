"""Whole expressions: parentheses, precedence, left-to-right order."""

import logging
import random
import unittest

from tests import reference
from tests.helpers import check, model

log = logging.getLogger(__name__)

PREC = {"+": 1, "-": 1, "*": 2}


def gen(rng: random.Random, depth: int, root: bool = True):
    """Random tree as (text, value, precedence), with signs anywhere.

    Leaves may be negative ("-12"); groups may be negated ("-(a+b)"); '-' nodes
    may go negative. A unary minus binds tightest, so "-3*2" means (-3)*2.
    The root is always an operator: a prompt needs at least one.
    """
    if depth == 0 or (not root and rng.random() < 0.3):
        n = rng.randint(0, 10 ** rng.randint(1, 3))
        if rng.random() < 0.25:
            return f"-{n}", -n, 3
        return str(n), n, 3
    op = rng.choice("+-*")
    lt, lv, lp = gen(rng, depth - 1, False)
    rt, rv, rp = gen(rng, depth - 1, False)
    if lp < PREC[op]:
        lt = f"({lt})"
    if rp <= PREC[op]:  # right operand of equal precedence keeps left-to-right meaning
        rt = f"({rt})"
    val = lv + rv if op == "+" else lv - rv if op == "-" else lv * rv
    text = f"{lt}{op}{rt}"
    if not root and rng.random() < 0.15:
        return f"-({text})", -val, 3
    return text, val, PREC[op]


class TestExpressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model()

    def test_hand_picked(self):
        for e in ["2+3*4", "2*3+4", "(2+3)*4", "2*(3+4)", "1+2+3+4", "10-3-2", "10-(3-2)",
                  "2*3*4*5", "(1+2)*(3+4)", "((1+2)*3+4)*5", "100-99+1", "2+3*4-5*2",
                  "7-3", "1*(2*(3*(4*(5+1))))", "12-3*4", "5-5*1",
                  "(3-7)*2", "-3+5", "2*-3", "-(1-4)", "2*-(1-4)", "1+(3-7)*2", "-0*5", "-2-3",
                  "3--2", "-(2+3)*-4", "-1-(-2*-3)", "-9*-9", "0-5-5", "-5+5", "2-(3-(4-(5-6)))"]:
            check(e)

    def test_greedy_generation(self):
        """Cached greedy decoding must reproduce the reference byte for byte."""
        from handformer import arith
        from tests import reference
        for e in ["2+3*(4-1)", "((1+2)*3+4)*5", "10-3-2", "3-7", "99*99+1", "1+(3-7)*2", "-(2+3)*-4"]:
            self.assertEqual(arith.solve(model(), e), reference.output(e))

    def test_random_trees(self):
        rng = random.Random(4)
        for depth in (2, 3, 4):
            for _ in range(15):
                text, value, _ = gen(rng, depth)
                answer = reference.output(text).rsplit("]", 1)[-1].strip()
                self.assertEqual(int(answer), value, f"reference disagrees with Python on {text}")
                check(text)
            log.info("random trees depth<=%d ok", depth)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    unittest.main()
