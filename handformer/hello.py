"""Hello world: a hand-written transformer that only adds two numbers.

    python -m handformer.hello 47+38       ->  47+38=58;85

The smallest complete example of every idea in the full calculator, in one
file and five layers. Prompt ``\\n<a>+<b>=``; the model writes the sum's digits
least-significant first (the order the carry flows), then ``;``, then the
answer most-significant first, then ``\\n``. The host only feeds bytes, takes
the argmax, and stops at ``\\n``.

    layer 1  MLP   tag the latest '+', '=' and ';' (landmark features)
    layer 2  attn  read where they are: P ('+'), E ('='), S (';')
             MLP   which phase we are in; where digit i of a and of b sits
    layer 3  attn  read digits a_i, a_{i-1}, b_i, b_{i-1}
             MLP   the carry, recovered from a_{i-1} + b_{i-1} and the previous
                   output digit (never stored); tag nonzero working digits
    layer 4  attn  find the top nonzero working digit
             MLP   write working digit (a_i + b_i + carry) mod 10, or ';'
    layer 5  attn  read the working digit to copy into the answer
             MLP   write it, or '\\n' when the answer is complete

Every weight comes from a named formula below; nothing is trained.
"""
from __future__ import annotations

import argparse
import logging

from . import common as cm
from . import primitives as pr
from .model import Layout, ModelBuilder, Transformer

log = logging.getLogger(__name__)

DIMS = ("f_P", "f_E", "f_S", "P", "E", "S", "seen_eq", "working_done", "tA", "tA1", "tB", "tB1", "exA", "exB",
        "A", "A1", "B", "B1", "carry", "zkey", "Z", "Zo", "C", *(f"D{d}" for d in range(10)), "work_end", "done")


def builder() -> ModelBuilder:
    lay = Layout()
    cm.add_input_dims(lay)
    lay.add(*DIMS)
    b = ModelBuilder(lay)
    cm.embed(b)  # the shared byte embedding: digit value and one-hot, '+', '=', ';', start byte

    # Layer 1: landmark tags. f_X = 2·(pos+1) on an X byte, 1 on the start byte (no match yet).
    l1 = b.at(1)
    l1.units += pr.latest_feature({"is_plus": 1.0}, "f_P")
    l1.units += pr.latest_feature({"is_eq": 1.0}, "f_E")
    l1.units += pr.latest_feature({"is_semi": 1.0}, "f_S")

    # Layer 2: where are '+', '=' and ';'? Then: phase flags and digit addresses.
    l2 = b.at(2)
    l2.heads += [pr.latest_head("latest '+'", "f_P", "P"), pr.latest_head("latest '='", "f_E", "E"),
                 pr.latest_head("latest ';'", "f_S", "S")]
    l2.units += pr.clamp01_units({"E": 1.0}, {"seen_eq": 1.0})                   # [E >= 1]: past the prompt's '='
    l2.units += pr.clamp01_units({"S": 1.0, "E": -1.0}, {"working_done": 1.0})  # [S > E]: ';' written, answer phase
    i = {"pos": 1.0, "E": -1.0}                    # the next working digit is digit i (i = 0 at '=')
    i_prev = pr.add(i, pr.const(-1.0))
    a_lo, a_hi = {}, {"P": 1.0}                    # a sits strictly between the start byte (0) and '+'
    b_lo, b_hi = {"P": 1.0}, {"E": 1.0}            # b between '+' and '='
    l2.units += [cm.digit_target_unit(a_lo, a_hi, i, "tA"), cm.digit_target_unit(a_lo, a_hi, i_prev, "tA1"),
                 cm.digit_target_unit(b_lo, b_hi, i, "tB"), cm.digit_target_unit(b_lo, b_hi, i_prev, "tB1"),
                 *cm.exhausted_units(a_lo, a_hi, i, "exA"), *cm.exhausted_units(b_lo, b_hi, i, "exB")]

    # Layer 3: read the digits (an exhausted operand lands on its fence, value 0). Recover the carry.
    l3 = b.at(3)
    l3.heads += [pr.gather_head("a_i", pr.add(a_lo, {"tA": 1.0}), "A"),
                 pr.gather_head("a_i-1", pr.add(a_lo, {"tA1": 1.0}), "A1"),
                 pr.gather_head("b_i", pr.add(b_lo, {"tB": 1.0}), "B"),
                 pr.gather_head("b_i-1", pr.add(b_lo, {"tB1": 1.0}), "B1")]
    y = {"A1": 1.0, "B1": 1.0}                     # the previous column's digit sum, 0..18
    l3.units += pr.clamp01_units(pr.add(y, pr.const(-9.0)), {"carry": 1.0})     # [y >= 10]
    l3.units += pr.hat_units(pr.add(y, {"is0": -20.0}, pr.const(20.0)), 9.0, {"carry": 1.0})  # [y == 9, wrote '0']
    in_work_nz = {"nz": 1.0, "seen_eq": 1.0, "working_done": -1.0}  # == 2 on nonzero working digits
    l3.units.append(pr.pos_if(in_work_nz, 2.0, "zkey"))
    l3.units.append(pr.pos_if({"is_eq": 1.0}, 1.0, "zkey"))  # all-zero sum: the '=' stands in, answer '0'

    # Layer 4: the top nonzero working digit; write the working digit or ';'.
    l4 = b.at(4)
    l4.heads.append(pr.latest_head("top nonzero working digit", "zkey", "Z"))
    l4.units.append(pr.relu_unit(pr.add({"Z": 1.0, "E": -1.0}, pr.const(-1.0)), {"Zo": 1.0},
                                 "answer length - 1: how far the top nonzero working digit is past '=' + 1"))
    working = {"working_done": 1.0}
    l4.units += cm.ramp_mod10_units({"A": 1.0, "B": 1.0, "carry": 1.0}, 0, 19, working, 0.0)
    l4.units.append(pr.relu_unit(pr.gated({"exA": 1.0, "exB": 1.0, "carry": -1.0, "one": -1.0}, working, 0.0),
                                 {"work_end": 1.0}, "write ';' once both operands are used up and no carry is left"))

    # Layer 5: copy the working digits back, top first, then stop.
    l5 = b.at(5)
    j = {"pos": 1.0, "S": -1.0}                    # answer digit index (0 at ';')
    l5.heads.append(pr.gather_head("answer digit", pr.add({"E": 1.0, "Zo": 1.0}, pr.scale(j, -1.0), pr.const(1.0)),
                                   "C"))
    l5.units += cm.ramp_mod10_units({"C": 1.0}, 0, 9, working, 1.0)
    l5.units += pr.clamp01_units(pr.gated(pr.add(j, {"Zo": -1.0}), working, 1.0), {"done": 1.0})  # [j > Zo]

    for d in range(10):
        b.logit(f"D{d}", ord("0") + d, cm.DIGIT_LOGIT)
    b.logit("work_end", cm.WORK_END, cm.CONTROL_LOGIT)  # a control byte beats a digit firing at the same step
    b.logit("done", cm.EOS, cm.CONTROL_LOGIT)
    return b


def build() -> Transformer:
    m = builder().build()
    log.info("hello model: %d layers, d_model %d, %d heads, %d units, %d nonzero weights", len(m.layers),
             len(m.layout), sum(len(l.heads) for l in m.layers), sum(l.w_in.shape[1] for l in m.layers),
             m.nonzero_params())
    return m


def prompt(expression: str) -> list[int]:
    return list(f"\n{expression}=".encode("ascii"))


def solve(model: Transformer, expression: str) -> str:
    """Everything the model writes after the prompt (the host only stops at '\\n')."""
    p = prompt(expression)
    out = model.generate(p, stop=cm.EOS, max_new=4 * len(p) + 8)
    return bytes(out[len(p):]).decode("ascii").rstrip("\n")


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m handformer.hello", description="Add two numbers, a+b.")
    ap.add_argument("sums", nargs="+", help="e.g. 47+38")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    m = build()
    for s in args.sums:
        log.info("%s=%s", s, solve(m, s))


if __name__ == "__main__":
    main()
