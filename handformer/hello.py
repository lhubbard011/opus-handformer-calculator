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
    """All five layers of the adder, as named forms; ``.build()`` turns them into dense weights.

    Sequence: ``\\n a + b = w_0 w_1 ... ; answer \\n``. Position 0 is the start byte, P the
    '+', E the '=', S the ';'. Working digit i (LSB first) is written at position E+1+i.
    """
    lay = Layout()
    cm.add_input_dims(lay)
    lay.add(*DIMS)
    b = ModelBuilder(lay)
    cm.embed(b)  # the shared byte embedding: digit value and one-hot, '+', '=', ';', start byte

    # Layer 1: landmark tags. f_X = 2·(pos+1) on an X byte, 1 on the start byte (no match yet).
    # Each is a key feature for a layer-2 "latest" head: it grows with position, so the most
    # recent matching byte has the largest key. Other bytes get 0 (the unit's -BIG offset).
    l1 = b.at(1)
    l1.units += pr.latest_feature({"is_plus": 1.0}, "f_P")
    l1.units += pr.latest_feature({"is_eq": 1.0}, "f_E")
    l1.units += pr.latest_feature({"is_semi": 1.0}, "f_S")

    # Layer 2: where are '+', '=' and ';'? Then: phase flags and digit addresses.
    l2 = b.at(2)
    # A latest head: query 40·one, key f_X, so score = 40·f_X. Softmax puts ~all weight on the
    # largest f_X among positions <= this one (the most recent X, or the start byte if none)
    # and copies that key's pos: P, E, S = where the latest '+', '=', ';' sit (0 if not yet).
    l2.heads += [pr.latest_head("latest '+'", "f_P", "P"), pr.latest_head("latest '='", "f_E", "E"),
                 pr.latest_head("latest ';'", "f_S", "S")]
    # Phase flags, robust 0/1 steps over the positions just read (this MLP sees this layer's heads).
    # seen_eq feeds layer 3's zkey; working_done gates every output: 0 = write working digits
    # (layer 4), 1 = copy the answer (layer 5). Before ';' exists S = 0, so S - E < 1.
    l2.units += pr.clamp01_units({"E": 1.0}, {"seen_eq": 1.0})                   # [E >= 1]: past the prompt's '='
    l2.units += pr.clamp01_units({"S": 1.0, "E": -1.0}, {"working_done": 1.0})  # [S > E]: ';' written, answer phase
    # Digit addresses. At position E + i the model predicts working digit i (written at E+1+i),
    # so i = pos - E. An operand fenced by (lo, hi) keeps digit i (i = 0 units) at hi-1-i;
    # i_prev = i - 1 addresses the column below, for the carry.
    i = {"pos": 1.0, "E": -1.0}                    # the next working digit is digit i (i = 0 at '=')
    i_prev = pr.add(i, pr.const(-1.0))
    a_lo, a_hi = {}, {"P": 1.0}                    # a sits strictly between the start byte (0) and '+'
    b_lo, b_hi = {"P": 1.0}, {"E": 1.0}            # b between '+' and '='
    # tA = max(hi-1-i - lo, 0): offset of digit i from the fence lo; layer 3 gathers at lo + tA.
    # Once the operand has no digit i the offset clamps to 0 and the read lands on the fence
    # (start byte or '+', value 0). At i = 0, i_prev = -1 lands on hi ('+' or '=', value 0): no
    # carry into the units. exA = 1 - [a has digit i] tells layer 4 when the working may end.
    l2.units += [cm.digit_target_unit(a_lo, a_hi, i, "tA"), cm.digit_target_unit(a_lo, a_hi, i_prev, "tA1"),
                 cm.digit_target_unit(b_lo, b_hi, i, "tB"), cm.digit_target_unit(b_lo, b_hi, i_prev, "tB1"),
                 *cm.exhausted_units(a_lo, a_hi, i, "exA"), *cm.exhausted_units(b_lo, b_hi, i, "exB")]

    # Layer 3: read the digits (an exhausted operand lands on its fence, value 0). Recover the carry.
    l3 = b.at(3)
    # A gather head reads val at position t = lo + offset: query [60t, -30] against keys
    # [pos, pos²] scores -30·(k - t)² + const, a peak at k = t that beats each neighbour by 30.
    # A, B (digit i) feed layer 4's digit; A1, B1 (digit i-1) feed the carry just below.
    l3.heads += [pr.gather_head("a_i", pr.add(a_lo, {"tA": 1.0}), "A"),
                 pr.gather_head("a_i-1", pr.add(a_lo, {"tA1": 1.0}), "A1"),
                 pr.gather_head("b_i", pr.add(b_lo, {"tB": 1.0}), "B"),
                 pr.gather_head("b_i-1", pr.add(b_lo, {"tB1": 1.0}), "B1")]
    y = {"A1": 1.0, "B1": 1.0}                     # the previous column's digit sum, 0..18
    # carry into column i = carry out of column i-1, recovered from the text (never stored).
    # Case 1: y >= 10 always carries. clamp01 tests y - 9 >= 1, exactly 0 or 1.
    l3.units += pr.clamp01_units(pr.add(y, pr.const(-9.0)), {"carry": 1.0})     # [y >= 10]
    # Case 2: y == 9 carries only if a carry came into column i-1, which then wrote 9+1 -> '0'.
    # That digit is this position's own byte (is0 = 1 on '0'). Adding 20 - 20·is0 moves y to
    # >= 20 unless the byte is '0', so the hat at 9 fires only for y == 9 AND a written '0'.
    # The two cases never overlap, so carry is exactly 0 or 1; layer 4 reads it.
    l3.units += pr.hat_units(pr.add(y, {"is0": -20.0}, pr.const(20.0)), 9.0, {"carry": 1.0})  # [y == 9, wrote '0']
    in_work_nz = {"nz": 1.0, "seen_eq": 1.0, "working_done": -1.0}  # == 2 on nonzero working digits
    # zkey: key feature for layer 4's latest head, = pos on nonzero working digits, else 0.
    # Each key uses its own flags, so prompt digits (seen_eq 0) and answer digits
    # (working_done 1) never count.
    l3.units.append(pr.pos_if(in_work_nz, 2.0, "zkey"))
    l3.units.append(pr.pos_if({"is_eq": 1.0}, 1.0, "zkey"))  # all-zero sum: the '=' stands in, answer '0'

    # Layer 4: the top nonzero working digit; write the working digit or ';'.
    l4 = b.at(4)
    # Z = the largest zkey seen: position of the highest nonzero working digit (E when the sum is
    # 0). The answer starts there, so Zo = Z - E - 1 is its index; layer 5 reads both.
    l4.heads.append(pr.latest_head("top nonzero working digit", "zkey", "Z"))
    l4.units.append(pr.relu_unit(pr.add({"Z": 1.0, "E": -1.0}, pr.const(-1.0)), {"Zo": 1.0},
                                 "answer length - 1: how far the top nonzero working digit is past '=' + 1"))
    working = {"working_done": 1.0}
    # Working digit: one-hot vote D[(A + B + carry) mod 10] for sums 0..19, from shared ReLU
    # ramps (three per value combine into a bump); gated to the working phase (working_done 0).
    l4.units += cm.ramp_mod10_units({"A": 1.0, "B": 1.0, "carry": 1.0}, 0, 19, working, 0.0)
    # work_end = ReLU(exA + exB - carry - 1): 1 only when both operands are used up and no carry
    # remains. The ramps then also vote D0 (0+0+0), but ';' has the larger logit (see below).
    l4.units.append(pr.relu_unit(pr.gated({"exA": 1.0, "exB": 1.0, "carry": -1.0, "one": -1.0}, working, 0.0),
                                 {"work_end": 1.0}, "write ';' once both operands are used up and no carry is left"))

    # Layer 5: copy the working digits back, top first, then stop.
    l5 = b.at(5)
    j = {"pos": 1.0, "S": -1.0}                    # answer digit index (0 at ';')
    # Answer digit j is working digit Zo - j, which sits at E + 1 + Zo - j: a gather walking
    # down from the top nonzero digit (j = 0) to the units digit at E+1 (j = Zo).
    l5.heads.append(pr.gather_head("answer digit", pr.add({"E": 1.0, "Zo": 1.0}, pr.scale(j, -1.0), pr.const(1.0)),
                                   "C"))
    # The copied digit C (0..9) as a one-hot vote, answer phase only (working_done 1). Once all
    # Zo+1 digits are out (j > Zo), done = 1 writes '\n' and the host stops.
    l5.units += cm.ramp_mod10_units({"C": 1.0}, 0, 9, working, 1.0)
    l5.units += pr.clamp01_units(pr.gated(pr.add(j, {"Zo": -1.0}), working, 1.0), {"done": 1.0})  # [j > Zo]

    # Unembedding: each output dim adds its weight to one byte's logit; the host takes the argmax.
    # D0..D9 -> '0'..'9' at 10, work_end -> ';' and done -> '\n' at 20.
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
