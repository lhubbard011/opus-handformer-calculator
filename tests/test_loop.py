"""The recurrent-depth calculator (handformer/loop): an autoregressive decoder whose block loops
per token, writing the answer as its first byte.

Every check decodes the real run, position by position, and compares it with
tests/loop_reference.py:

* the steps: their kinds, every copy of the expression (one latent cell per byte), the working
  (column digits, quotient digits, remainders);
* the router: how many iterations each position took (a product column's predictor one per
  term, a quotient digit's one per column of the remainder, everything else one);
* the answer: the reference's, the frame-writing calculator's (tests/reference.py) and exact
  arithmetic; and nothing is written before it;
* exactness: every latent position's input (a byte's embedding, the plan, the registers) is
  integers to within 1e-9, so nothing drifts however long the run.
"""

import logging
import random
import unittest
from fractions import Fraction

import numpy as np

from handformer import loop
from tests import loop_reference, reference
from tests.test_rewrite import gen

log = logging.getLogger(__name__)
_MODEL = None
DRIFT = 1e-9


def model():
    global _MODEL
    if _MODEL is None:
        _MODEL = loop.build()
    return _MODEL


class Decoded:
    """A run, read back position by position."""

    def __init__(self, m, res) -> None:
        lay = m.layout
        self.res = res
        feat = [j for j in np.flatnonzero(np.abs(m.embed).sum(axis=0)) if j != lay["one"]]
        rows = m.embed[:, feat]
        role = {r: lay[r] for r in ("k_s", "k_w", "k_q", "k_r")}
        self.kinds, self.cells, self.regs = [], [], []
        for t in res.tokens:
            e = res.inputs[t.index]
            if not t.latent:
                self.kinds.append("byte")
                self.cells.append(chr(t.byte))
            else:
                k = [r for r, j in role.items() if e[j] > 0.5]
                if k:
                    self.kinds.append(k[0])
                    self.cells.append(None)
                else:
                    d = ((e[feat][None] - rows) ** 2).sum(axis=1)
                    self.kinds.append("cell")
                    self.cells.append(chr(int(d.argmin())))
            self.regs.append({r: int(round(e[lay[r]])) for r in ("dA", "dB", "qd", "rd")})
        lat = res.inputs[[t.index for t in res.tokens if t.latent]]
        self.drift = float(np.abs(lat - np.round(lat)).max()) if len(lat) else 0.0

    def steps(self, n_prompt: int):
        """(kind, copy text, working positions, their predictors' depths) per step after the prompt."""
        toks, out = self.res.tokens, []
        i = n_prompt - 1  # the prompt's root
        while True:
            if i + 1 >= len(toks) or not toks[i + 1].latent:
                return out  # this root halted
            j = i + 2  # after the step's start
            work = []
            while self.kinds[j] in ("k_w", "k_q", "k_r"):
                work.append(j)
                j += 1
            copy = []
            while self.cells[j] != "=":
                copy.append(self.cells[j])
                j += 1
            depths = [toks[w - 1].iterations for w in work]
            out.append(("".join(copy), work, depths))
            i = j


class TestLoop(unittest.TestCase):
    worst = 0.0

    @classmethod
    def tearDownClass(cls):
        log.info("largest distance of a latent input from integers over every run: %.3g", cls.worst)

    def check(self, e: str, value=None) -> None:
        m = model()
        res = loop.run(m, e)
        self.assertTrue(res.finished, f"{e!r}: ran out of budget")
        dec = Decoded(m, res)
        want = loop_reference.steps(e)
        got = dec.steps(len(res.prompt))
        self.assertEqual(len(got), len(want) - 1, f"{e!r}: {len(got)} steps, the reference has {len(want) - 1}")
        for k, ((copy, work, depths), st) in enumerate(zip(got, want)):
            where = f"{e!r} step {k} ({st.kind} {st.op})"
            self.assertEqual(copy, st.copy, f"{where}: the copy")
            kinds = [dec.kinds[w] for w in work]
            if st.op in "+-*" and st.kind == "reduce" and st.op:
                self.assertEqual(kinds, ["k_w"] * len(st.columns), where)
                cols = [(dec.regs[w]["dA"], dec.regs[w]["dB"]) for w in work]
                if st.op == "*":
                    cols = [(a, 0) for a, _ in cols]
                self.assertEqual(cols, st.columns, f"{where}: the columns")
            elif st.op == "/":
                nb = len(st.remainders[0])
                q = [dec.regs[w]["qd"] for w in work if dec.kinds[w] == "k_q"]
                rem = [dec.regs[w]["rd"] for w in work if dec.kinds[w] == "k_r"]
                row = [dec.regs[w]["dA"] for w in work if dec.kinds[w] == "k_w"]
                self.assertEqual(q, st.quotient, f"{where}: the quotient digits")
                self.assertEqual([rem[i:i + nb] for i in range(0, len(rem), nb)], st.remainders, f"{where}: remainders")
                self.assertEqual(row, [a for a, _ in st.columns], f"{where}: the quotient row")
            else:
                self.assertEqual(work, [], where)
            self.assertEqual(depths, st.depths, f"{where}: the router's depths")
        # the answer: nothing before it, and the same as every other account of the value
        answer = bytes(res.answer).decode()
        self.assertEqual(answer, want[-1].copy + "\n", e)
        written = [t for t in res.tokens[len(res.prompt):] if not t.latent]
        self.assertEqual(len(written), len(answer) - 1, f"{e!r}: a byte was written before the answer")
        final = want[-1].copy
        if not any(reference.binary_op(final, i) for i in range(len(final))) and (
                any(reference.binary_op(e, i) for i in range(len(e))) or "(" in e or " " in e.strip()):
            old = reference.output(e).rsplit("]", 1)[-1].strip()
            self.assertEqual(answer[:-1], old, f"{e!r}: the frame-writing calculator says {old}")
        if value is not None:
            self.assertEqual(Fraction(answer[:-1]), Fraction(value), e)
        self.assertLess(dec.drift, DRIFT, f"{e!r}: a latent input drifted from integers")
        TestLoop.worst = max(TestLoop.worst, dec.drift)
        routed = [t.iterations for t in res.tokens[len(res.prompt):] if not t.latent]
        self.assertEqual(set(routed), {1}, f"{e!r}: written bytes take one iteration")

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
        for e in ["2^3", "2^10", "-2^3", "(1+1)^5", "3^0", "7^1", "-(-2^1)", "1.5^2", "2^3^2", "2^-2~4",
                  "(2+3)^3/7~3", "10^20", "0^0", "-2^2", "2^64"]:
            self.check(e)

    def test_division(self):
        for e in ["6/3", "7/2", "355/113~6", "100/7~2", "-7/2~1", "1.5/0.25~2", "((1+2)*3-4)/5~2", "1/3~12",
                  "1 / 3 ~ 1 2", "-1/-8~3", "0/5", "10/4~2", "2.5/0.5", "99999/3", "1/7~30", "12345678901234567890/9"]:
            self.check(e)

    def test_division_by_zero(self):
        """Not evaluated: the root halts and the answer is the expression left."""
        for e in ["1/0", "2+3/0", "(1+1)/(2-2)"]:
            self.check(e)

    def test_claims_corpus(self):
        """The frame-writing calculator's interpretability corpus (tests/test_claims.py)."""
        from tests.test_claims import EXPRESSIONS
        for e in EXPRESSIONS:
            self.check(e)

    def test_random_trees(self):
        rng = random.Random(5)
        for depth in (2, 3, 4):
            for _ in range(10):
                text, value, _ = gen(rng, depth, ops="+-*", wrap=0.2)
                self.check(text, value)
            log.info("random +-* trees depth<=%d ok", depth)

    def test_long_operands(self):
        """Any length: a column is one position, a product column one iteration per term."""
        rng = random.Random(9)
        for n in (30, 80):
            a, b = rng.randint(10 ** (n - 1), 10 ** n), rng.randint(10 ** (n - 1), 10 ** n)
            self.check(f"{a}+{b}", a + b)
            self.check(f"{a}-{b}", a - b)
            self.check(f"{'9' * n}+1", 10 ** n)
            self.check(f"1{'0' * n}-1", 10 ** n - 1)
        a, b = rng.randint(10 ** 29, 10 ** 30), rng.randint(10 ** 14, 10 ** 15)
        self.check(f"{a}*{b}", a * b)
        log.info("long operands ok")

    def test_deep_nesting(self):
        e = "1"
        for k in range(2, 14):
            e = f"({e}-{k})" if k % 2 else f"{k}-({e})"
        self.check(e)

    def test_causal(self):
        """A decoder: no head of the block reads a later position (the builder refuses one)."""
        m = model()
        self.assertFalse(any(h.full for layer in m.block for h in layer.heads))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    unittest.main()
