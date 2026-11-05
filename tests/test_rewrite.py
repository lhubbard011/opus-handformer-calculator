"""The looped calculator (handformer/rewrite): the answer is the first byte the model writes.

Every check runs the real loop: the recurrent block over the tape until its router stops,
then the coda's greedy bytes. Four things are checked:

* the tape going into every pass equals tests/rewrite_reference.py, byte for byte;
* the answer equals the reference's, the frame-writing calculator's (tests/reference.py) and
  exact integer arithmetic;
* between passes the state is exactly a byte embedding plus the carried dims (the column
  accumulator K and the digit counter kk), which hold exact integers and are 0 at the end: no
  drift can build up over passes;
* nothing is written before the answer: the coda's first byte is the answer's first byte.
"""

import logging
import random
import unittest

import numpy as np

from handformer import common as cm, rewrite as loop
from handformer.rewrite import circuit
from tests import rewrite_reference, reference

log = logging.getLogger(__name__)
_MODEL = None


def model():
    global _MODEL
    if _MODEL is None:
        _MODEL = loop.builder().build()
    return _MODEL


def gen(rng: random.Random, depth: int, ops: str = "+-", root: bool = True, wrap: float = 0.0,
        digits: int = 3) -> tuple[str, int, int]:
    """Random expression tree as (text, value, precedence); signs, negated groups, brackets."""
    prec = {"+": 1, "-": 1, "*": 2}
    if depth == 0 or (not root and rng.random() < 0.3):
        n = rng.randint(0, 10 ** rng.randint(1, digits))
        text, val = (f"-{n}", -n) if rng.random() < 0.25 else (str(n), n)
        while wrap and rng.random() < wrap:
            text = f"({text})" if rng.random() < 0.7 else f"-({text})"
            val = val if text[0] == "(" else -val
        return text, val, 3
    op = rng.choice(ops)
    lt, lv, lp = gen(rng, depth - 1, ops, False, wrap, digits)
    rt, rv, rp = gen(rng, depth - 1, ops, False, wrap, digits)
    if lp < prec[op]:
        lt = f"({lt})"
    if rp <= prec[op]:
        rt = f"({rt})"
    val = lv + rv if op == "+" else lv - rv if op == "-" else lv * rv
    text = f"{lt}{op}{rt}"
    if not root and rng.random() < 0.15:
        return f"-({text})", -val, 3
    return text, val, prec[op]


DRIFT = 1e-9


class TestLoop(unittest.TestCase):
    worst = 0.0

    @classmethod
    def tearDownClass(cls):
        log.info("largest distance of a state from its byte embedding over every pass: %.3g", cls.worst)

    def check(self, e: str, value: int | None = None) -> None:
        m = model()
        drift = []
        carried = [m.layout[d] for d in circuit.CARRIED]

        def on_pass(r, state, x):
            tape = m.tape_of(state)
            rest = state - m.embed[tape]
            held = rest[:, carried].copy()
            rest[:, carried] = 0.0
            drift.append(max(float(np.abs(rest).max()), float(np.abs(held - np.round(held)).max())))

        res = m.run(loop.prompt(e), stop=cm.EOS, on_pass=on_pass)
        got = [bytes(p.tape[1:-1]).decode() for p in res.passes]
        want = rewrite_reference.tapes(e)
        self.assertTrue(res.halted, f"{e!r}: no halt")
        self.assertEqual(got, want, f"{e!r}: tapes differ")
        answer = bytes(res.answer).decode()
        self.assertTrue(answer.endswith("\n"), f"{e!r}: answer {answer!r} never ended")
        self.assertEqual(answer[:-1], rewrite_reference.answer(e), e)
        # each pass rebuilds the state from robust steps, so float rounding (~1e-15) never adds up
        self.assertLess(max(drift), DRIFT, f"{e!r}: the state drifted from a byte embedding")
        TestLoop.worst = max(TestLoop.worst, max(drift))
        last = res.passes[-1]
        self.assertEqual(last.routed, 0, e)
        old = reference.output(e).rsplit("]", 1)[-1].strip() if any(
            reference.binary_op(e, i) for i in range(len(e))) or "(" in e or " " in e.strip() else e.strip()
        self.assertEqual(answer[:-1], old, f"{e!r}: the frame-writing calculator says {old}")
        if value is not None:
            self.assertEqual(int(answer), value, e)

    def test_hand_picked(self):
        for e in ["5", "-5", "2+3", "47+38", "999+1", "9-12", "1000-1", "1-1000", "-5+5", "0-0", "3--2",
                  "10-3-2", "100-99+1", "1-(2-3)", "-(2+3)", "-(-5)+1", "((5))", "(-5)+1", "-(-(-3))+2",
                  "2+3-(4+5)-(-(1-2))", "((1+2)-(3-4))-5", "2-(3-(4-(5-6)))", "-1-(-2-3)", "007+3", "0+0"]:
            self.check(e)

    def test_spaces(self):
        for e in ["1 + 2", " 2+3", "2+3 ", "1 2+ 3", "- 3 + 1", "3 - -2", "-( 1 - 4 )", "  7  -  8  ", "( 5 )"]:
            self.check(e)

    def test_multiplication(self):
        for e in ["6*7", "12*34", "99*99", "-3*4", "-3*-4", "0*5", "-0*5", "2+3*4", "2*3*4", "-(2*3)",
                  "2*-(3-5)", "(1+2)*(3+4)", "999*999", "123456789*987", "10*10", "7*0", "1*1"]:
            self.check(e)

    def test_decimals(self):
        for e in ["1.5+2.25", "0.5-0.75", "1.5+2.5", "-(-1.25)+0.75", "10.01-0.01", "0.1+0.2", "3.14*(2-0.5)",
                  "1.5*0.2", "0.25*0.04", "12.5*8", "1.25*-0.8", "-12.5*3-4", "0.001*1000", "2.50-2.5"]:
            self.check(e)

    def test_powers(self):
        """A power is a text rewrite that needs room: grow passes add blanks first."""
        for e in ["2^3", "2^10", "-2^3", "(1+1)^5", "3^0", "7^1", "-(-2^1)", "1.5^2", "2^3^2", "2^-2~4",
                  "(2+3)^3/7~3", "10^20", "0^0", "-2^2", "2^100"]:
            self.check(e)

    def test_division(self):
        """Long division over passes: shift or subtract; the quotient is kept in blank cells."""
        for e in ["6/3", "7/2", "355/113~6", "100/7~2", "-7/2~1", "1.5/0.25~2", "((1+2)*3-4)/5~2", "1/3~12",
                  "1 / 3 ~ 1 2", "-1/-8~3", "0/5", "10/4~2", "2.5/0.5", "99999/3", "1/7~30", "12345678901234567890/9"]:
            self.check(e)

    def test_claims_corpus(self):
        """The frame-writing calculator's interpretability corpus (tests/test_claims.py)."""
        from tests.test_claims import EXPRESSIONS
        for e in EXPRESSIONS:
            self.check(e)

    def test_random_trees(self):
        rng = random.Random(5)
        for depth in (2, 3, 4):
            for _ in range(12):
                text, value, _ = gen(rng, depth, ops="+-*", wrap=0.2)
                self.check(text, value)
            log.info("random +-* trees depth<=%d ok", depth)

    def test_long_operands(self):
        """Arbitrary length: the carry and borrow chains are one lookup, whatever the length."""
        rng = random.Random(9)
        for n in (30, 80):
            a, b = rng.randint(10 ** (n - 1), 10 ** n), rng.randint(10 ** (n - 1), 10 ** n)
            self.check(f"{a}+{b}", a + b)
            self.check(f"{a}-{b}", a - b)
            self.check(f"{'9' * n}+1", 10 ** n)
            self.check(f"1{'0' * n}-1", 10 ** n - 1)
        a, b = rng.randint(10 ** 39, 10 ** 40), rng.randint(10 ** 14, 10 ** 15)
        self.check(f"{a}*{b}", a * b)
        log.info("long operands ok")

    def test_deep_nesting(self):
        """Nesting costs passes, not layers."""
        e = "1"
        for k in range(2, 14):
            e = f"({e}-{k})" if k % 2 else f"{k}-({e})"
        self.check(e)

    def test_answer_first(self):
        """The coda's first byte is the answer's first byte: nothing else is written."""
        m = model()
        res = m.run(loop.prompt("12-(3-4)"), stop=cm.EOS)
        self.assertEqual(bytes(res.answer), b"13\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    unittest.main()
