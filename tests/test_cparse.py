"""The parser (handformer/cparse) writes exactly the records of the LL(1) table, and the table
parses exactly like an independent recursive-descent parser (tests/cparse_reference.py)."""

import logging
import random
import unittest

import numpy as np

from handformer.cparse import circuit as cp, grammar as gm, toy
from handformer.model import Tracer
from tests import cparse_reference as ref

log = logging.getLogger(__name__)

PROGRAM = "i (x) { x + 1; w (1) ; } e { (x + (1)); ; }\nx;"
EDGES = ["x;", ";", "{}", "{{}{}}", "i(x);", "i(x);e;", "i(x)i(1);e;", "w(x)w(1){}", "((((x))));", "x+1+x+(1+x);",
         "  x  ;  ", "\tx\n;\n", "{" * 12 + "}" * 12,
         # errors: at the first token that cannot continue a program
         "x", "}", "{", "i(x)", "x+;", "e", "(x;", "x;}", "z;", "{x;", "i x", "x;|O 001", "$", "x;$", "{$}"]


def program(rng: random.Random, d: int = 0) -> str:
    r = rng.random()
    if d > 4 or r < 0.3:
        return rng.choice(["x;", "1+x;", ";", "(x+(1));"])
    if r < 0.5:
        return "{" + "".join(program(rng, d + 1) for _ in range(rng.randint(0, 3))) + "}"
    if r < 0.7:
        return "i(x)" + program(rng, d + 1) + ("e" + program(rng, d + 1) if rng.random() < 0.5 else "")
    return "w(1+x)" + program(rng, d + 1)


def spaced(rng: random.Random, s: str) -> str:
    return "".join(ch + (rng.choice(["", " ", "\n", "\t "]) if rng.random() < 0.3 else "") for ch in s)


def mutated(rng: random.Random, s: str) -> str:
    cs = list(s)
    k = rng.randrange(len(cs) + 1)
    if rng.random() < 0.5 and len(cs) > 1:
        del cs[min(k, len(cs) - 1)]
    else:
        cs.insert(k, rng.choice(toy.TOKENS + "z"))
    return "".join(cs)


class TestTable(unittest.TestCase):
    """The compiled table (simulated with a stack) against the recursive-descent reference."""

    def test_against_reference(self):
        t = toy.G.compile()
        rng = random.Random(3)
        cases = [PROGRAM, *EDGES, *(spaced(rng, program(rng)) for _ in range(300)),
                 *(mutated(rng, program(rng)) for _ in range(300)),
                 *("".join(rng.choice(toy.TOKENS + "  ") for _ in range(rng.randint(1, 25))) for _ in range(3000))]
        for src in cases:
            if not src.strip():
                continue
            out, want = gm.simulate(t, src), ref.parse(src)
            last = gm.parse_records(out)[-1]
            if isinstance(want, int):
                self.assertEqual((last[0], last[3]), ("!", want), src)
            else:
                self.assertEqual(gm.tree(out), want, src)


class TestParser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = toy.G.compile()
        cls.m = cp.build(cls.t)

    def check(self, src: str) -> None:
        """Teacher-forced: the argmax at every output position is the next byte of the records."""
        p = cp.prompt(src, self.t)
        want = gm.simulate(self.t, src)
        seq = p + list(want.encode())
        pred = np.argmax(self.m.logits(seq[:-1]), axis=1)[len(p) - 1:]
        got = bytes(int(v) for v in pred).decode("ascii", "replace")
        if got != want:
            k = next(i for i, (g, w) in enumerate(zip(got, want)) if g != w)
            r = k - k % cp.W
            self.fail(f"{src!r}: first diff at record {k // cp.W} offset {k % cp.W}: want {want[k]!r} got {got[k]!r}\n"
                      f" want {want[r:r + cp.W]!r}\n  got {got[r:r + cp.W]!r}")

    def test_program(self):
        self.check(PROGRAM)

    def test_edges(self):
        for src in EDGES:
            with self.subTest(src=src):
                self.check(src)

    def test_fuzz(self):
        rng = random.Random(11)
        for _ in range(25):
            self.check(spaced(rng, program(rng)))
        for _ in range(25):
            self.check(mutated(rng, program(rng)))

    def test_greedy_decoding(self):
        src = "i(x){x+1; w(x) ;} e {(x);}"
        out = cp.solve(self.m, self.t, src)
        self.assertEqual(out, gm.simulate(self.t, src))
        self.assertEqual(gm.tree(out), ref.parse(src))


class TestParserClaims(unittest.TestCase):
    """At every byte the model writes: exactly one rule dim is 1 (the rest 0); every head reads one
    exact value (in its declared scope); every yes/no step stays off its ramp."""

    def test_exactly_one_rule_and_exact_reads(self):
        t = toy.G.compile()
        b = cp.builder(t)
        m = b.build()
        rule_dims = [m.layout[f"r_{r.name}"] for r in cp.rules(t)]

        class T(Tracer):
            def __init__(self):
                self.snaps, self.att, self.pre = [], {}, {}

            def residual(self, label, x):  # noqa: ARG002
                self.snaps.append(x)

            def attention(self, layer, head, attn):
                self.att[(layer, head)] = attn

            def mlp(self, layer, pre):
                self.pre[layer] = pre

        rng = random.Random(23)
        bad, checked = [], 0
        for src in [PROGRAM, *EDGES[:14], *EDGES[14:], *(mutated(rng, program(rng)) for _ in range(10))]:
            p = cp.prompt(src, t)
            toks = p + list(gm.simulate(t, src).encode())
            tr = T()
            m.residual(toks, trace=tr)
            scope = np.arange(len(toks)) >= len(p) - 1
            scope[-1] = False
            final = tr.snaps[-1]
            on = final[:, rule_dims]
            for q in np.flatnonzero(scope):
                if not (np.isclose(on[q].sum(), 1.0, atol=1e-6) and np.all((np.abs(on[q]) < 1e-6) | (np.abs(on[q] - 1) < 1e-6))):
                    bad.append(f"{src[:20]!r} q={q}: rule dims {on[q][np.abs(on[q]) > 1e-6]}")
            checked += int(scope.sum())
            for (li, hi), a in tr.att.items():
                spec = b.layers[li].heads[hi]
                where = scope.copy()
                if spec.exact_when is not None:
                    w = sum(c * final[:, m.layout[d]] for d, c in spec.exact_when.items())
                    where &= np.abs(w - 1) < 1e-6
                v = tr.snaps[2 * li] @ m.layers[li].heads[hi].w_v
                top = v[a.argmax(axis=1)]
                err = np.abs(a @ v - top).max(axis=1) / np.maximum(1.0, np.abs(top).max(axis=1))
                bad += [f"{src[:20]!r} head {spec.name} q={q}" for q in np.flatnonzero(where & (err > 1e-6))]
            for li, pre in tr.pre.items():
                for ui, spec in enumerate(b.layers[li].units):
                    for claim in spec.claims:
                        z = pre[:, ui]
                        bad += [f"{src[:20]!r} unit L{li + 1}#{ui} q={q}"
                                for q in np.flatnonzero(scope & (z > claim[1]) & (z < claim[2]))]
        log.info("parser claims: %d output bytes, %d violations", checked, len(bad))
        self.assertEqual(bad, [], "\n".join(bad[:20]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
