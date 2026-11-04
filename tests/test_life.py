"""The Game of Life model (handformer/life) writes exactly the reference generations."""

import itertools
import logging
import random
import unittest

import numpy as np

from handformer import life
from tests import life_reference as ref

log = logging.getLogger(__name__)


def grid(rng: random.Random, h: int, w: int, p: float = 0.4) -> str:
    return "/".join("".join("#" if rng.random() < p else "." for _ in range(w)) for _ in range(h))


class TestLife(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = life.build()

    def check(self, text: str) -> None:
        """Teacher-forced: the argmax at every output position is the next expected byte."""
        p = life.prompt(text)
        want = ref.output(text)
        seq = p + list(want.encode())
        pred = np.argmax(self.m.logits(seq[:-1]), axis=1)[len(p) - 1:]
        got = bytes(int(t) for t in pred).decode("ascii", "replace")
        if got != want:
            k = next(i for i, (g, w) in enumerate(zip(got, want)) if g != w)
            self.fail(f"{text}: first diff at {k}\n want {want!r}\n  got {got!r}")

    def test_every_3x3_grid(self):
        for cells in itertools.product(".#", repeat=9):
            s = "".join(cells)
            self.check("1:" + "/".join((s[0:3], s[3:6], s[6:9])))
        log.info("all 512 3x3 grids exact")

    def test_classic_patterns_over_generations(self):
        glider = "......./..#..../...#.../.###.../......./......."
        for text in ("4:.#./.#./.#.", "3:..../.##./.##./....", "9:" + glider,
                     "6:....../..##../.#..#./..##../......"):  # blinker, block, glider, beehive
            self.check(text)

    def test_shapes_and_edges(self):
        for text in ("1:#", "1:.", "2:###", "2:#/#/#", "1:#####", "3:#/#", "0:.#./.#./.#.", "1:##/##", "1:#.#.#/.#.#."):
            self.check(text)

    def test_random_grids(self):
        rng = random.Random(11)
        for _ in range(60):
            h, w = rng.randint(1, 8), rng.randint(1, 8)
            self.check(f"{rng.randint(1, 3)}:{grid(rng, h, w)}")

    def test_large_grid(self):
        rng = random.Random(5)
        self.check("1:" + grid(rng, 24, 30, 0.35))

    def test_greedy_decoding(self):
        self.assertEqual(life.solve(self.m, "2:.#./.#./.#."), ".../###/...=".join(["1:", ""]) + "0:.#./.#./.#.=")

    def test_bad_input_is_refused(self):
        for bad in ("abc", ".#./###", "x:###"):
            with self.assertRaises(ValueError):
                life.prompt(bad)



class TestLifeClaims(unittest.TestCase):
    """The checked claims of tests/test_claims.py, for the Life model: every head reads one exact
    value and every yes/no step's input stays off its ramp, at every position that writes output.

    Prompt positions before the prompt's '=' are out of scope: there "one generation back" does
    not exist, so the neighbourhood lookups land on the start byte. Nothing uses those values:
    prompt bytes are given, not predicted, and every head in this model reads only byte features
    (alive, row ends, digits) or the layer-1 tags, never another position's computed values.
    """

    def test_claims_hold(self):
        from handformer.model import Tracer

        b = life.builder()
        m = b.build()

        class T(Tracer):
            def __init__(self):
                self.snaps, self.att, self.pre = [], {}, {}

            def residual(self, label, x):
                self.snaps.append(x)

            def attention(self, layer, head, attn):
                self.att[(layer, head)] = attn

            def mlp(self, layer, pre):
                self.pre[layer] = pre

        rng = random.Random(3)
        texts = ["4:.#./.#./.#.", "1:#", "2:###", "1:#/#/#", "3:" + grid(rng, 6, 7)]
        bad, checked = [], 0
        for text in texts:
            p = life.prompt(text)
            toks = m.generate(p, stop=10, max_new=2000)
            t = T()
            m.residual(toks, trace=t)
            scope = np.arange(len(toks)) >= len(p) - 1
            scope[-1] = False
            for (li, hi), a in t.att.items():
                v = t.snaps[2 * li] @ m.layers[li].heads[hi].w_v
                top = v[a.argmax(axis=1)]
                err = np.abs(a @ v - top).max(axis=1) / np.maximum(1.0, np.abs(top).max(axis=1))
                checked += int(scope.sum())
                bad += [f"{text} head {b.layers[li].heads[hi].name} q={q}" for q in np.flatnonzero(scope & (err > 1e-6))]
            for li, pre in t.pre.items():
                for ui, spec in enumerate(b.layers[li].units):
                    for claim in spec.claims:
                        if claim[0] == "outside":
                            z = pre[:, ui]
                            bad += [f"{text} unit L{li + 1}#{ui} q={q} z={z[q]:.3g}"
                                    for q in np.flatnonzero(scope & (z > claim[1]) & (z < claim[2]))]
                            checked += int(scope.sum())
        log.info("life claims: %d checks (head reads and steps), %d violations", checked, len(bad))
        self.assertEqual(bad, [], "\n".join(bad[:20]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
