"""The C syntax highlighter (handformer/csyntax) writes exactly the reference lexer's states."""

import logging
import random
import unittest

import numpy as np

from handformer import csyntax
from tests import csyntax_reference as ref

log = logging.getLogger(__name__)

PROGRAM = r'''#include <stdio.h>
#define MAX(a, b) ((a) > (b) ? \
                   (a) : (b))
/* a block comment with * stars ** and / slashes */
static const char *names[] = {"tab\t", "quote\"", "back\\", ""};
// a line comment with "quotes" and 'q' \
   continued by a backslash
int main(int argc, char **argv) {
    unsigned long x = 0x1Fu, y = 10UL;
    double d = 1.5e-3 + .25 - 3.E+2;
    char c = '\'', e = '\\', n = '\n';
    for (int i = 0; i < argc; i++) { x += i * 2 / 1; }
    _Static_assert(sizeof(int) >= 2, "int");
    if (x != y && !(d <= 0.0)) return x % 3;  /* trailing */
    integer = interrupt + do_it + double2;   # not a directive
    return 0;
}
'''

EDGES = ["int x;", "/**/x", "/*/ still comment */y", "//", "a//b\nc", '"unterminated\nx', "'\\''", '"\\\\"x',
         "  #  define A 1\n#x", "1.5e+3e-2", "0x.1p-3f", ".5", "a.b", "...", "x+++++y", "_Bool b;", "_Bool_x",
         "returnx return", "if(1)else", "a/=b/c", "*/x", "#", "\n\n#if 1\n#endif", "'a", "L\"wide\"",
         "do{}while(0);", "\t\v\f\r", "int\\\nx"]


def fuzz(rng: random.Random, n: int) -> str:
    alphabet = "ab_09ex /*'\"\\#\n.+-E\tp(){};<=!"
    return "".join(rng.choice(alphabet) for _ in range(n))


class TestCSyntax(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = csyntax.build()

    def check(self, src: str) -> None:
        """Teacher-forced: the argmax at every output position is the reference's next letter."""
        p = csyntax.prompt(src)
        want = ref.lex(src) + "\n"
        seq = p + list(want.encode())
        pred = np.argmax(self.m.logits(seq[:-1]), axis=1)[len(p) - 1:]
        got = bytes(int(t) for t in pred).decode("ascii", "replace")
        if got != want:
            k = next(i for i, (g, w) in enumerate(zip(got, want)) if g != w)
            self.fail(f"{src!r}: first diff at byte {k} ({src[k]!r}): want {want[k]!r} got {got[k]!r}\n"
                      f" want {want!r}\n  got {got!r}")

    def test_program(self):
        self.check(PROGRAM)

    def test_edges(self):
        for src in EDGES:
            with self.subTest(src=src):
                self.check(src)

    def test_fuzz(self):
        rng = random.Random(17)
        for _ in range(150):
            self.check(fuzz(rng, rng.randint(1, 60)))

    def test_all_keywords(self):
        self.check(" ".join(csyntax.KEYWORDS) + " " + " ".join(k + "x" for k in csyntax.KEYWORDS))

    def test_greedy_decoding(self):
        self.assertEqual(csyntax.solve(self.m, "int x = 1; // c"), ref.lex("int x = 1; // c"))



class TestCSyntaxClaims(unittest.TestCase):
    """The ruleset is describable because it is exact: at every byte the model writes, exactly one
    rule dim is 1 and every other is 0; every head reads one exact value; every yes/no step stays
    off its ramp. Prompt positions are out of scope: nothing reads their computed values (every
    head reads byte features, or the layer-2 landmarks that are exact everywhere)."""

    def test_exactly_one_rule_and_exact_reads(self):
        from handformer.model import Tracer

        b = csyntax.builder()
        m = b.build()
        rule_dims = [m.layout[f"r_{r.name}"] for r in csyntax.RULES] + [m.layout["r_end"]]

        class T(Tracer):
            def __init__(self):
                self.snaps, self.att, self.pre = [], {}, {}

            def residual(self, label, x):
                self.snaps.append(x)

            def attention(self, layer, head, attn):
                self.att[(layer, head)] = attn

            def mlp(self, layer, pre):
                self.pre[layer] = pre

        rng = random.Random(23)
        bad, checked = [], 0
        for src in [PROGRAM, *EDGES, *(fuzz(rng, 50) for _ in range(30))]:
            p = csyntax.prompt(src)
            toks = p + list((ref.lex(src) + "\n").encode())
            t = T()
            m.residual(toks, trace=t)
            scope = np.arange(len(toks)) >= len(p) - 1
            scope[-1] = False
            final = t.snaps[-1][:, rule_dims]
            for q in np.flatnonzero(scope):
                on = final[q]
                if not (np.isclose(on.sum(), 1.0, atol=1e-6) and np.all((np.abs(on) < 1e-6) | (np.abs(on - 1) < 1e-6))):
                    bad.append(f"{src[:20]!r} q={q}: rule dims {on[np.abs(on) > 1e-6]}")
            checked += int(scope.sum())
            for (li, hi), a in t.att.items():
                v = t.snaps[2 * li] @ m.layers[li].heads[hi].w_v
                top = v[a.argmax(axis=1)]
                err = np.abs(a @ v - top).max(axis=1) / np.maximum(1.0, np.abs(top).max(axis=1))
                bad += [f"{src[:20]!r} head {b.layers[li].heads[hi].name} q={q}" for q in np.flatnonzero(scope & (err > 1e-6))]
            for li, pre in t.pre.items():
                for ui, spec in enumerate(b.layers[li].units):
                    for claim in spec.claims:
                        z = pre[:, ui]
                        bad += [f"{src[:20]!r} unit L{li + 1}#{ui} q={q}" for q in np.flatnonzero(scope & (z > claim[1]) & (z < claim[2]))]
        log.info("C highlighter claims: %d output bytes, %d violations", checked, len(bad))
        self.assertEqual(bad, [], "\n".join(bad[:20]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
