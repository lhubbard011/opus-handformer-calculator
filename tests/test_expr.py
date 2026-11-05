"""Whole expressions: parentheses, precedence, left-to-right order."""

import logging
import random
import unittest

from tests import reference
from tests.helpers import check, model

log = logging.getLogger(__name__)

PREC = {"+": 1, "-": 1, "*": 2}


def gen(rng: random.Random, depth: int, root: bool = True, wrap: float = 0.0):
    """Random tree as (text, value, precedence), with signs anywhere.

    Leaves may be negative ("-12"); groups may be negated ("-(a+b)"); '-' nodes
    may go negative. A unary minus binds tightest, so "-3*2" means (-3)*2.
    The root is always an operator: a prompt needs at least one. ``wrap``: the chance that a
    leaf is put in brackets of its own, "(12)" or "(-12)" (once or twice).
    """
    if depth == 0 or (not root and rng.random() < 0.3):
        n = rng.randint(0, 10 ** rng.randint(1, 3))
        text, val = (f"-{n}", -n) if rng.random() < 0.25 else (str(n), n)
        while wrap and rng.random() < wrap:
            text = f"({text})" if rng.random() < 0.7 else f"-({text})"
            val = val if text[0] == "(" else -val
        return text, val, 3
    op = rng.choice("+-*")
    lt, lv, lp = gen(rng, depth - 1, False, wrap)
    rt, rv, rp = gen(rng, depth - 1, False, wrap)
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

    def test_spaces(self):
        """Spaces anywhere are removed in the weights, one per frame; the answer is unchanged."""
        for e in ["1 + 2", " 2+3", "2+3 ", "(1 + 2) * 3", "1 2+3", "- 3 + 1", "3 - -2", "2 *-3",
                  "-( 1 - 4 )", "  7  *  8  ", "2 ^ 3", "1.5 + 2.25", "5 ", "1 / 3 ~ 1 2 ", "1/3~ 4"]:
            check(e)
        self.assertEqual(arith_solve("( 2 + 3 ) * 4"), reference.output("( 2 + 3 ) * 4"))
        rng = random.Random(7)
        for _ in range(20):
            text, value, _ = gen(rng, 3)
            spaced = "".join(ch + " " * rng.choice((0, 0, 1, 2)) for ch in text)
            answer = reference.output(spaced).rsplit("]", 1)[-1].strip()
            self.assertEqual(int(answer), value, f"reference disagrees with Python on {spaced!r}")
            check(spaced)
        log.info("spaced expressions ok")


    def test_bracketed_numbers(self):
        """A group holding one number loses its brackets in a frame of its own, [()]; -(-v) is v."""
        cases = {"(5)+1": 6, "2*(3)": 6, "((5))": 5, "(5)": 5, "((2+3))*2": 10, "(-5)*2": -10, "3-(-5)": 8,
                 "(1.5)+1": 2.5, "-(5)+1": -4, "2^(3)": 8, "(2*3)+(5)": 11, "((1))+((2))": 3, "-(-5)+1": 6,
                 "2*-(-5)": 10, "-(-(-3))*2": -6, "1+-(-2)": 3, "( 5 )": 5, "- ( - 5 ) + 1": 6, "10/(4)~2": 2.5}
        for e, want in cases.items():
            check(e)
            got = reference.output(e).rsplit("]", 1)[-1].strip()
            self.assertEqual(float(got), want, e)
        rng = random.Random(11)
        for _ in range(20):
            text, value, _ = gen(rng, 3, wrap=0.35)
            answer = reference.output(text).rsplit("]", 1)[-1].strip()
            self.assertEqual(int(answer), value, f"reference disagrees with Python on {text}")
            check(text)
        log.info("bracketed numbers ok")


def arith_solve(e: str) -> str:
    from handformer import arith
    return arith.solve(model(), e)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    unittest.main()
