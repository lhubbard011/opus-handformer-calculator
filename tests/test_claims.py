"""Machine-checked interpretability claims.

The explanations of this model are claims about what each part computes. This
test runs real traces (every operator, signs, decimals, precision, powers,
parentheses) and checks three kinds of claim at every position:

1. **Exact reads.** A head moves the value of one key: the attention-weighted
   mix of the values it moves equals the value at its top key (ties between
   keys holding the same value are exact too). Scope: ``HeadSpec.exact_when``
   (a 0/1 form) when the head declares one; otherwise wherever its calculator
   is switched on (``act_<op>``), or, for shared heads, every position that
   writes output.
2. **Robust steps.** A unit claiming ``("outside", lo, hi[, when])`` never has
   its pre-activation strictly inside (lo, hi): a yes/no test's input is within
   1/4 of an integer, so noise cannot leak into the flag.
3. **Silence.** A unit claiming ``("silent_unless", flag)`` has pre-activation
   <= 0 wherever ``flag`` is 0: a calculator's outputs cannot fire while it is off.

The last position (the final newline) is never read, so it is not checked.

Known exceptions live in ``tests/claims_known.json``, each with its diagnosis
(why the value is never used where the claim fails). The test fails on any
violation not listed there, and on a listed one that no longer occurs, so the
list can only shrink deliberately. DESIGN.md §28 summarises it.
"""

import json
import logging
import unittest
from pathlib import Path

import numpy as np

from handformer import arith, common as cm
from handformer.model import Form, Tracer

log = logging.getLogger(__name__)

EXPRESSIONS = [
    "47+38", "99999+1", "0.5+0.25", "1-999", "-(1-4)", "12*34-400", "2*-3", "-12.5*3-4", "3.14*(2-0.5)",
    "1+(3-7)*2", "(1+2)*(3+4)", "355/113~6", "100/7~2", "-7/2~1", "1.5/0.25~2", "(2+3)^3/7~3", "2^-2~4",
    "-2^3", "2^10", "0*5", "-0*5", "10-10", "123456789*987", "((1+2)*3-4)/5~2",
    # strip frames (the parser also runs on those lines, its writing switched off by sp / pf)
    " -3 * ( 2 + 4 ) ", "1 / 3 ~ 1 2", "(5)+1", "-(-5)*2", "((2+3))*(4)",
]
EXACT = 1e-6
KNOWN = json.loads((Path(__file__).with_name("claims_known.json")).read_text())


class _Trace(Tracer):
    def __init__(self) -> None:
        self.snaps: list[np.ndarray] = []
        self.att: dict[tuple[int, int], np.ndarray] = {}
        self.pre: dict[int, np.ndarray] = {}

    def residual(self, label: str, x: np.ndarray) -> None:  # noqa: ARG002
        self.snaps.append(x)

    def attention(self, layer: int, head: int, attn: np.ndarray) -> None:
        self.att[(layer, head)] = attn

    def mlp(self, layer: int, pre: np.ndarray) -> None:
        self.pre[layer] = pre


def _parser_off(spec, layer: int, final: np.ndarray, layout) -> np.ndarray:
    """Positions where a parser spec's claims are not owed, because nothing reads what it computes
    there (each position recomputes the parser; none reads another's, expr.py):
    - a strip frame (sp or pf, strip.py): the parser has no redex and its writing is off
      (phases 2 and 4, and every src4 select, whose flag is at most 1 less the strip flag);
    - a line's '=' (phase1): the byte written is '[' whatever the parser holds.
    The strip frame reads only the parser's p and c, found by layer 5, so only the parser's
    specs from layer 6 on (operator, operands, regions) are excused; all it reads is checked."""
    if not spec.where.startswith("handformer/expr.py") or layer < 6:
        return np.zeros(len(final), bool)
    return final[:, layout["sp"]] + final[:, layout["pf"]] + final[:, layout["phase1"]] > 0.5


def _form(x: np.ndarray, form: Form, layout) -> np.ndarray:
    out = np.zeros(len(x))
    for d, c in form.items():
        out += c * x[:, layout[d]]
    return out


class TestClaims(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builder = arith.builder()
        cls.model = cls.builder.build()
        cls.traces = []
        for e in EXPRESSIONS:
            toks = cls.model.generate(arith.prompt(e), stop=cm.EOS, max_new=4000)
            tr = _Trace()
            cls.model.residual(toks, trace=tr)
            cls.traces.append((e, toks, len(arith.prompt(e)), tr))
        log.info("traced %d expressions, %d positions", len(cls.traces), sum(len(t[1]) for t in cls.traces))

    def _report(self, kind: str, bad: dict, checked: int) -> None:
        known = KNOWN[kind]
        found = {f"{where.split(':')[0]} {name}" for where, name in bad}
        new = {k: v for k, v in bad.items() if f"{k[0].split(':')[0]} {k[1]}" not in known}
        gone = sorted(set(known) - found)
        log.info("%s: %d checks, %d parts violate (%d known, %d new), %d known no longer occur",
                 kind, checked, len(bad), len(bad) - len(new), len(new), len(gone))
        msgs = [f"{where} {name}: {v[0]} (+{len(v) - 1} more)" for (where, name), v in sorted(new.items())]
        if gone:
            msgs.append("fixed, remove from tests/claims_known.json: " + "; ".join(gone))
        if msgs:
            self.fail(f"{kind}:\n" + "\n".join(msgs[:60]))

    def test_heads_read_one_exact_value(self):
        lay, bad, checked = self.model.layout, {}, 0
        for e, toks, plen, tr in self.traces:
            n, final = len(toks), tr.snaps[-1]
            for (li, hi), a in tr.att.items():
                spec, head = self.builder.layers[li].heads[hi], self.model.layers[li].heads[hi]
                if spec.exact_when is not None:
                    scope = _form(final, spec.exact_when, lay) > 0.5
                elif ":" in spec.name:
                    scope = final[:, lay[f"act_{spec.name.split(':')[0]}"]] > 0.5
                else:
                    scope = np.arange(n) >= plen - 1
                scope &= ~_parser_off(spec, li + 1, final, lay)
                scope[-1] = False
                v = tr.snaps[2 * li] @ head.w_v
                top = v[a.argmax(axis=1)]
                err = np.abs(a @ v - top).max(axis=1) / np.maximum(1.0, np.abs(top).max(axis=1))
                checked += int(scope.sum())
                for q in np.flatnonzero(scope & (err > EXACT)):
                    bad.setdefault((spec.where, f"L{li + 1} '{spec.name}'"), []).append(
                        f"{e!r} q={q} {chr(toks[q])!r} mixes values (err {err[q]:.3g})")
        self._report("exact reads", bad, checked)

    def test_unit_claims(self):
        lay, bad, checked = self.model.layout, {}, 0
        for e, toks, _plen, tr in self.traces:
            final = tr.snaps[-1][:-1]
            for li, pre in tr.pre.items():
                x = tr.snaps[2 * li + 1]  # the MLP's input: after this layer's attention
                for ui, spec in enumerate(self.builder.layers[li].units):
                    owed = ~_parser_off(spec, li + 1, final, lay)
                    for claim in spec.claims:
                        p = pre[:-1, ui]
                        if claim[0] == "outside":
                            lo, hi = claim[1], claim[2]
                            scope = (_form(x[:-1], claim[3], lay) > 0.5) if len(claim) > 3 else np.ones(len(p), bool)
                            viol = np.flatnonzero(owed & scope & (p > lo) & (p < hi))
                            what = f"input {{:.3g}} inside ({lo:g}, {hi:g})"
                        elif claim[0] == "silent_unless":
                            viol = np.flatnonzero((x[:-1, lay[claim[1]]] < 0.5) & (p > 1e-9))
                            what = f"fires ({{:.3g}}) while {claim[1]} is off"
                        else:
                            raise AssertionError(f"unknown claim {claim!r} at {spec.where}")
                        checked += len(p)
                        for q in viol:
                            bad.setdefault((spec.where, f"L{li + 1} unit #{ui}"), []).append(
                                f"{e!r} q={q} {chr(toks[q])!r} " + what.format(p[q]))
        self._report("unit claims", bad, checked)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
