"""The C parser (handformer/cparse) lexes, then writes exactly the records of the LL(1) table; and
the table parses exactly like an independent recursive-descent parser (tests/cparse_reference.py).

The expected output of a source is built without the weights: the letters from the separately
written lexer (tests/csyntax_reference.py), the tokens read off those letters (tokens.py), and
the records from the table run with a real stack (grammar.simulate)."""

import logging
import random
import unittest

import numpy as np

from handformer import cparse
from handformer.cparse import circuit as cp, grammar as gm, tokens as tk
from handformer.model import Tracer
from tests import cparse_reference as ref
from tests import csyntax_reference as lexref

log = logging.getLogger(__name__)
TABLE, TOKENS = cparse.TABLE, cparse.TOKENS

PROGRAM = """/* the toy grammar, in C */
if (count) {
    count + 1;
    while (0x1F) ;
} else { (total + (2.5e-3)); ; }
#define IGNORED 1
next; // done
"""
EDGES = ["x;", ";", "{}", "{{}{}}", "if(x);", "if(x);else;", "if (a) if (b) ; else ;", "while(x)while(1){}",
         "((((x))));", "a+1+b+(1+c);", "  x  ;  ", "\tx\n;\n", "{" * 12 + "}" * 12,
         "ifx; whiley; elsewhere;", "_t1 + 0x10u + .5;", "a/**/+/**/b;", "x; // c\ny;", "#if 1\nx;\n#endif\n",
         "a\\\nb;", "if/**/(x)x;",
         # errors: at the first token that cannot continue a program
         "x", "}", "{", "if(x)", "x+;", "else", "(x;", "x;}", "int x;", "{x;", "if x", "a += 1;", "a ++;",
         "x;|O 001", "$;", "x; $", '"s";', "'c';", "a->b;", "x * y;", "if (a) else ;", "while (a) ) ;",
         "@;", "a | 1;"]

WORDS = ["a", "count", "x1", "_t", "ifx", "whiled", "elsewhere"]
NUMBERS = ["1", "0", "0x1F", "42u", "3.5e-2", ".5", "1e+3"]
ALIEN = ["int", "+=", "*", '"s"', "'c'", "++", "...", "->", "return", "@", "|", "=", "-", "[", "for"]
SEPS = [" ", " ", "  ", "\n", "\t", "/* c */", "// c\n", "\n#x\n"]


def program(rng: random.Random, d: int = 0) -> list[str]:
    """A random program of the toy grammar, as C tokens."""
    r = rng.random()

    def expr() -> list[str]:
        out = []
        for k in range(rng.randint(1, 3)):
            if k:
                out.append("+")
            out += (["(", *expr(), ")"] if d < 3 and rng.random() < 0.2
                    else [rng.choice(WORDS + NUMBERS)])
        return out

    if d > 4 or r < 0.3:
        return rng.choice([[*expr(), ";"], [";"]])
    if r < 0.5:
        return ["{", *(t for _ in range(rng.randint(0, 3)) for t in program(rng, d + 1)), "}"]
    if r < 0.7:
        out = ["if", "(", *expr(), ")", *program(rng, d + 1)]
        return out + (["else", *program(rng, d + 1)] if rng.random() < 0.5 else [])
    return ["while", "(", *expr(), ")", *program(rng, d + 1)]


def join(rng: random.Random, toks: list[str]) -> str:
    """The tokens as source: whitespace, comments and directives between them, or nothing where
    the two cannot merge (one of them is a bracket or ';')."""
    out = []
    for i, t in enumerate(toks):
        if i:
            safe = t in "{}();" or toks[i - 1] in "{}();"
            out.append("" if safe and rng.random() < 0.4 else rng.choice(SEPS))
        out.append(t)
    return "".join(out)


def mutated(rng: random.Random, toks: list[str]) -> list[str]:
    toks = list(toks)
    k = rng.randrange(len(toks) + 1)
    if rng.random() < 0.4 and len(toks) > 1:
        del toks[min(k, len(toks) - 1)]
    else:
        toks.insert(k, rng.choice(ALIEN + ["if", "else", "while", "(", ")", "{", "}", ";", "+"]))
    return toks


def expected(src: str) -> tuple[str, str, list]:
    """(letters, records, tokens) the model must write for ``src``, built without the weights."""
    letters = lexref.lex(src)
    toks = tk.tokens(src, letters, TOKENS)
    return letters, gm.simulate(TABLE, toks), toks


class TestTokens(unittest.TestCase):
    """Tokens read off the letters: where each starts and which terminal it is."""

    def test_known(self):
        cases = {
            "if (a) x1;": [(1, "i"), (4, "("), (5, "x"), (6, ")"), (8, "x"), (10, ";"), (11, "$")],
            "a+=1;": [(1, "x"), (2, None), (4, "1"), (5, ";"), (6, "$")],
            "ifx/*if*/else": [(1, "x"), (10, "e"), (14, "$")],
            '"a b"+L\'c\'': [(1, None), (6, "+"), (7, None), (11, "$")],
            "a+++b": [(1, "x"), (2, None), (4, "+"), (5, "x"), (6, "$")],
            "#define A\nint": [(11, None), (14, "$")],
        }
        for src, want in cases.items():
            with self.subTest(src=src):
                self.assertEqual(tk.tokens(src, lexref.lex(src), TOKENS), want)


class TestTable(unittest.TestCase):
    """The compiled table (simulated with a stack) against the recursive-descent reference."""

    def test_against_reference(self):
        rng = random.Random(3)
        cases = [PROGRAM, *EDGES, *(join(rng, program(rng)) for _ in range(300)),
                 *(join(rng, mutated(rng, program(rng))) for _ in range(500)),
                 *(join(rng, [rng.choice(WORDS[:2] + ["if", "else", "while", "(", ")", "{", "}", ";", "+", "1"])
                              for _ in range(rng.randint(1, 20))]) for _ in range(2000))]
        for src in cases:
            if not src.strip():
                continue
            _letters, out, toks = expected(src)
            want = ref.parse(toks)
            last = gm.parse_records(out)[-1]
            if isinstance(want, int):
                self.assertEqual((last[0], last[3]), ("!", want), src)
            else:
                self.assertEqual(gm.tree(out), want, src)


class TestParser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = cparse.build()

    def check(self, src: str) -> None:
        """Teacher-forced: the argmax at every output position is the next byte (letters, the
        letters' end, records)."""
        p = cp.prompt(src)
        letters, records, _ = expected(src)
        want = letters + chr(cp.LEND) + records
        seq = p + list(want.encode())
        pred = np.argmax(self.m.logits(seq[:-1]), axis=1)[len(p) - 1:]
        got = bytes(int(v) for v in pred).decode("ascii", "replace")
        if got != want:
            k = next(i for i, (g, w) in enumerate(zip(got, want)) if g != w)
            if k <= len(letters):
                self.fail(f"{src!r}: letters differ at byte {k}: want {want[k]!r} got {got[k]!r}")
            k2 = k - len(letters) - 1
            r = k - k2 % cp.W
            self.fail(f"{src!r}: first diff at record {k2 // cp.W} offset {k2 % cp.W}: want {want[k]!r} "
                      f"got {got[k]!r}\n want {want[r:r + cp.W]!r}\n  got {got[r:r + cp.W]!r}")

    def test_program(self):
        self.check(PROGRAM)

    def test_edges(self):
        for src in EDGES:
            with self.subTest(src=src):
                self.check(src)

    def test_fuzz(self):
        rng = random.Random(11)
        for k in range(50):
            src = ""
            while not src or len(src) > cp.MAX_SOURCE:  # the model's input limit
                toks = program(rng)
                src = join(rng, mutated(rng, toks) if k % 2 else toks)
            self.check(src)

    def test_greedy_decoding(self):
        src = "if (a) { b + 1; while (a) ; } else { (c); }"
        letters, records = cp.split(cparse.solve(self.m, src))
        want_letters, want_records, toks = expected(src)
        self.assertEqual(letters, want_letters)
        self.assertEqual(records, want_records)
        self.assertEqual(gm.tree(records), ref.parse(toks))


class TestParserClaims(unittest.TestCase):
    """At every byte the model writes: exactly one rule of the phase's side is 1 (the rest 0) and
    every rule of the other side is 0, with the other side's output dims at exactly 0 (the two
    models in one never write over each other); every head reads one exact value (in its
    declared scope); every yes/no step stays off its ramp."""

    def test_exactly_one_rule_and_exact_reads(self):
        b = cparse.builder()
        m = b.build()
        lex_rules = [m.layout[f"r_{r.name}"] for r in cp.lexer.RULES] + [m.layout["r_end"]]
        parse_rules = [m.layout[f"r_{r.name}"] for r in cparse.rules()]
        quiet_in_lex = [m.layout[d] for d in ("out", "after_M")]

        class T(Tracer):
            def __init__(self):
                self.snaps, self.att, self.pre = [], {}, {}

            def residual(self, label, x):  # noqa: ARG002
                self.snaps.append(x)

            def attention(self, layer, head, attn):
                self.att[(layer, head)] = attn

            def mlp(self, layer, pre):
                self.pre[layer] = pre

        def one_hot(row: np.ndarray) -> bool:
            return bool(np.isclose(row.sum(), 1.0, atol=1e-6) and np.all((np.abs(row) < 1e-6) | (np.abs(row - 1) < 1e-6)))

        rng = random.Random(23)
        bad, checked = [], 0
        for src in [PROGRAM, *EDGES, *(join(rng, mutated(rng, program(rng))) for _ in range(10))]:
            p = cp.prompt(src)
            letters, records, _ = expected(src)
            seq = p + list((letters + chr(cp.LEND) + records).encode())
            tr = T()
            m.residual(seq, trace=tr)
            q = np.arange(len(seq))
            in_lex = (q >= len(p) - 1) & (q < len(p) + len(letters))  # predicting a letter or the letters' end
            in_rec = (q >= len(p) + len(letters)) & (q < len(seq) - 1)  # predicting a record byte
            final = tr.snaps[-1]
            for k in np.flatnonzero(in_lex):
                if not one_hot(final[k, lex_rules]) or np.abs(final[k, parse_rules]).max() > 1e-9 \
                        or np.abs(final[k, quiet_in_lex]).max() > 1e-9:
                    bad.append(f"{src[:20]!r} q={k} (letters): lexer rules {final[k, lex_rules].round(3).max()}, "
                               f"parser rules max {np.abs(final[k, parse_rules]).max():.3g}")
            for k in np.flatnonzero(in_rec):
                if not one_hot(final[k, parse_rules]) or np.abs(final[k, lex_rules]).max() > 1e-9:
                    bad.append(f"{src[:20]!r} q={k} (records): parser rules "
                               f"{final[k, parse_rules][np.abs(final[k, parse_rules]) > 1e-6]}, "
                               f"lexer rules max {np.abs(final[k, lex_rules]).max():.3g}")
            checked += int(in_lex.sum() + in_rec.sum())
            for (li, hi), a in tr.att.items():
                spec = b.layers[li].heads[hi]
                # the lexer's layers serve the letters; the parser's layers serve the records
                where = (in_lex | in_rec) if spec.name == "letters' end" else (in_lex if li < cp.LEX else in_rec)
                if spec.exact_when is not None:
                    w = sum(c * final[:, m.layout[d]] for d, c in spec.exact_when.items())
                    where = where & (np.abs(w - 1) < 1e-6)
                v = tr.snaps[2 * li] @ m.layers[li].heads[hi].w_v
                top = v[a.argmax(axis=1)]
                err = np.abs(a @ v - top).max(axis=1) / np.maximum(1.0, np.abs(top).max(axis=1))
                bad += [f"{src[:20]!r} head L{li + 1} {spec.name} q={k}" for k in np.flatnonzero(where & (err > 1e-6))]
            for li, pre in tr.pre.items():
                for ui, spec in enumerate(b.layers[li].units):
                    for claim in spec.claims:
                        z = pre[:, ui]
                        bad += [f"{src[:20]!r} unit L{li + 1}#{ui} q={k}"
                                for k in np.flatnonzero((in_lex | in_rec) & (z > claim[1]) & (z < claim[2]))]
        log.info("parser claims: %d output bytes, %d violations", checked, len(bad))
        self.assertEqual(bad, [], "\n".join(bad[:20]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
