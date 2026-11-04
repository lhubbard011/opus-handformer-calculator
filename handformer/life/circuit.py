"""Conway's Game of Life as a hand-written transformer: every weight below is a named formula.

Sequence (the host only feeds bytes, takes the argmax, stops at '\\n'):

    \\n3:.#./.#./.#.=2:.../###/...=1:.#./.#./.#.=0:.../###/...=\\n
       ^ prompt: count, ':', grid ('.' dead, '#' alive, '/' ends a row, '=' ends the grid)

After the prompt the model writes the next generation as a block "k:<grid>=", counting k down
from count-1 to 0, then '\\n'. The count lives in the text: each block's first byte is the
previous block's count minus one. Any grid size works; the width is read from the first '/'.

Why it is easy for attention: every block has the same length L, the position of the prompt's
'=' (BOS is 0, the prompt block is 1..L). So the cell written at position p sits at p - L in the
previous block, and its 3x3 neighbourhood there is at fixed offsets: +-1 along the row, +-W1
between rows (W1 = row length including its '/'). Nine exact lookups read it.

Edges cost nothing on the left and right: a row's neighbours there are '/' ':' or '=' bytes,
which read as dead (``alive`` is 0 on them). Above the top row and below the bottom row the
lookup would land in other rows, so two exact yes/no steps on the cell's offset switch those
neighbours off. Row ends need no arithmetic either: where the previous block has '/' or '=',
the model copies it.

    layer 1  MLP   landmark tags (latest ':', earliest '=', earliest row end)
             attn  is the previous byte '=' or the start byte?
    layer 2  attn  read C (latest ':'), L (prompt '='), F (first row end)
             MLP   which byte comes next (count digit, ':' or a cell); top / bottom row tests
    layer 3  attn  read the 3x3 neighbourhood and the byte at the same offset in the previous
                   block; read this block's count digit
             MLP   n = number of live neighbours (0..8)
    layer 4  MLP   the rule: next = [n == 3] + [n == 2 and alive]; "writing a cell" flag
    layer 5  MLP   votes for the next byte: '#', '.', '/', '=', ':', the count digit, '\\n'
"""
from __future__ import annotations

import logging
import re

from .. import common as cm
from .. import primitives as pr
from ..model import Layout, ModelBuilder, Transformer

log = logging.getLogger(__name__)

ALIVE, DEAD, ROW_END, GRID_END, COLON = ord("#"), ord("."), ord("/"), ord("="), ord(":")

# Life's own dims (the byte features come from common.INPUT_DIMS, plus 'alive' and 'is_colon').
NEIGHBOURS = {  # name: (row offset, column offset)
    "ul": (-1, -1), "u": (-1, 0), "ur": (-1, 1),
    "l": (0, -1), "r": (0, 1),
    "dl": (1, -1), "d": (1, 0), "dr": (1, 1),
}
DIMS = (
    "alive", "is_colon",
    "f_C", "f_L", "f_F", "prev_eq", "prev_bos",
    "C", "L", "F", "is_count", "up_ok", "down_ok",
    "self", *(f"a_{k}" for k in NEIGHBOURS), "pT_slash", "pT_eq", *(f"K{d}" for d in range(10)),
    "n", "nxt", "cellp",
    "out_hash", "out_dot", "out_slash", "out_eq", "out_colon", "out_nl",
    *(f"D{d}" for d in range(9)),  # a new block's count is the previous one minus 1: 0..8, never 9
)
OUTPUTS = {"out_hash": ALIVE, "out_dot": DEAD, "out_slash": ROW_END, "out_eq": GRID_END,
           "out_colon": COLON, "out_nl": cm.EOS, **{f"D{d}": ord("0") + d for d in range(9)}}


def builder() -> ModelBuilder:
    """Every weight of the Life model, at global layers 1-5 (see the module docstring)."""
    lay = Layout()
    cm.add_input_dims(lay)
    lay.add(*DIMS)
    b = ModelBuilder(lay)
    cm.embed(b)  # digits, '=', '/', '.', the start byte
    b.embed_byte(ALIVE, {"alive": 1.0})
    b.embed_byte(COLON, {"is_colon": 1.0})

    # ---- Layer 1. MLP: landmark tags for the layer-2 heads. Head: the previous byte's class.
    l1 = b.at(1)
    # latest ':' (this block's count separator): 2(pos+1) on ':', the start byte scores 1
    l1.units += pr.latest_feature({"is_colon": 1.0}, "f_C")
    # earliest '=' (the prompt's, at position L) and earliest row end ('/' or '='): BIG - pos
    l1.units.append(pr.earliest_if({"is_eq": 1.0}, 1.0, "f_L"))
    l1.units.append(pr.earliest_if({"is_slash": 1.0, "is_eq": 1.0}, 1.0, "f_F"))
    # Is the previous byte '=' (a block just ended) or the start byte? Exact lookups at pos - 1.
    prev = {"pos": 1.0, "one": -1.0}
    l1.heads += [pr.gather_head("prev byte '='?", prev, "prev_eq", src="is_eq"),
                 pr.gather_head("prev byte is the start?", prev, "prev_bos", src="is_bos")]

    # ---- Layer 2. Heads: the three landmarks. MLP: which byte comes next; top / bottom row tests.
    l2 = b.at(2)
    l2.heads += [pr.latest_head("latest ':'", "f_C", "C"),
                 pr.latest_head("prompt '='", "f_L", "L"),
                 pr.latest_head("first row end", "f_F", "F")]
    # The count digit starts each block: a digit right after '=' (or, in the prompt, the start byte).
    l2.units.append(pr.relu_unit({"is_digit": 1.0, "prev_eq": 1.0, "prev_bos": 1.0, "one": -1.0}, {"is_count": 1.0},
                                 "this byte is a block's count digit: a digit right after '=' or the start byte "
                                 "(AND of 0/1 flags), so the next byte is ':'"))
    # The cell written next has offset o = pos - C in its block (0 = first cell); a row holds
    # W1 = F - 2 bytes (W cells + its '/'), the grid B = L - 2 bytes. It has a row above iff
    # o >= W1, and a row below iff o + W1 <= B - 1. Both are exact steps on integers.
    o = {"pos": 1.0, "C": -1.0}
    w1 = {"F": 1.0, "one": -2.0}
    l2.units += pr.clamp01_units(pr.add(o, pr.scale(w1, -1.0), pr.const(1.0)), {"up_ok": 1.0})  # [o - W1 >= 0]
    l2.units += pr.clamp01_units(pr.add({"L": 1.0, "one": -2.0}, pr.scale(w1, -1.0), pr.scale(o, -1.0)),
                                 {"down_ok": 1.0})  # [B - W1 - o >= 1]

    # ---- Layer 3. Heads: read the previous block around this cell. MLP: count the live neighbours.
    l3 = b.at(3)
    # Same offset one block back: (pos + 1) - L. Neighbour (dr, dc) is dr*W1 + dc further.
    same = {"pos": 1.0, "one": 1.0, "L": -1.0}
    l3.heads.append(pr.gather_head("this cell, one generation back", same, "self", src="alive"))
    for name, (dr, dc) in NEIGHBOURS.items():
        l3.heads.append(pr.gather_head(f"neighbour {name}, one generation back",
                                       pr.add(same, pr.scale(w1, float(dr)), pr.const(float(dc))), f"a_{name}",
                                       src="alive"))
    # Is the byte at the same offset one block back a row end? Then this byte is one too: copy it.
    term = pr.gather_head("row end at this offset, one generation back", same, "pT_slash", src="is_slash")
    term.value = {"is_slash": {"pT_slash": 1.0}, "is_eq": {"pT_eq": 1.0}}
    l3.heads.append(term)
    # This block's count digit sits just before its ':' (read when the block ends, at '=').
    count = pr.gather_head("this block's count digit", {"C": 1.0, "one": -1.0}, "K0", src="oh0")
    count.value = {f"oh{d}": {f"K{d}": 1.0} for d in range(10)}
    l3.heads.append(count)
    # n = live neighbours. Left/right always count (row ends and ':' read as dead); the rows above
    # and below count only when they exist: AND of the 0/1 neighbour and the 0/1 row test.
    for name, (dr, _dc) in NEIGHBOURS.items():
        gate = {-1: "up_ok", 0: None, 1: "down_ok"}[dr]
        read = {f"a_{name}": 1.0} if gate is None else {f"a_{name}": 1.0, gate: 1.0, "one": -1.0}
        l3.units.append(pr.relu_unit(read, {"n": 1.0},
                                     f"count neighbour {name}" + ("" if gate is None else f" if {gate} (it exists)")))

    # ---- Layer 4. MLP: the rule, and whether the next byte is a cell at all.
    l4 = b.at(4)
    l4.units += pr.hat_units({"n": 1.0}, 3.0, {"nxt": 1.0})  # born or survives: exactly 3 neighbours
    # survives with exactly 2: +10 moves n out of the test's range unless the cell is alive (self = 1)
    l4.units += pr.hat_units({"n": 1.0, "self": 10.0, "one": -10.0}, 2.0, {"nxt": 1.0})
    l4.units.append(pr.relu_unit({"one": 1.0, "is_eq": -1.0, "is_count": -1.0, "pT_slash": -1.0, "pT_eq": -1.0},
                                 {"cellp": 1.0}, "the next byte is a cell: not after '=', not after the count "
                                 "digit, and not a row end"))

    # ---- Layer 5. MLP: one vote for the next byte (exactly one fires).
    l5 = b.at(5)
    l5.units += [
        pr.relu_unit({"nxt": 1.0, "cellp": 1.0, "one": -1.0}, {"out_hash": 1.0}, "write '#': a cell that is alive next"),
        pr.relu_unit({"cellp": 1.0, "nxt": -1.0}, {"out_dot": 1.0}, "write '.': a cell that is dead next"),
        pr.relu_unit({"pT_slash": 1.0, "is_eq": -1.0, "is_count": -1.0}, {"out_slash": 1.0},
                     "write '/': the previous generation has a row end here"),
        pr.relu_unit({"pT_eq": 1.0, "is_eq": -1.0, "is_count": -1.0}, {"out_eq": 1.0},
                     "write '=': the previous generation's grid ends here"),
        pr.relu_unit({"is_count": 1.0}, {"out_colon": 1.0}, "write ':' after the count digit"),
        pr.relu_unit({"K0": 1.0, "is_eq": 1.0, "one": -1.0}, {"out_nl": 1.0},
                     "write '\\n': the block that just ended was count 0, so the run is done"),
    ]
    for v in range(1, 10):  # the next block's count: this block's minus one
        l5.units.append(pr.relu_unit({f"K{v}": 1.0, "is_eq": 1.0, "one": -1.0}, {f"D{v - 1}": 1.0},
                                     f"start the next block with count {v - 1} (this block's {v} minus one)"))
    for dim, byte in OUTPUTS.items():
        b.logit(dim, byte, cm.CONTROL_LOGIT)  # exactly one vote fires at any byte
    return b


def build() -> Transformer:
    m = builder().build()
    log.info("life model: %d layers, d_model %d, %d heads, %d units, %d nonzero weights", len(m.layers),
             len(m.layout), sum(len(l.heads) for l in m.layers), sum(l.w_in.shape[1] for l in m.layers),
             m.nonzero_params())
    return m


ALLOWED = set("0123456789:.#/")
CHARS_HINT = "(use N: then rows of . and # separated by /)"
PATTERN = r"^[0-9]:[.#/]+$"
PATTERN_ERROR = "start with the number of generations (one digit), e.g. 3:.#./.#./.#."


def check(text: str) -> str:
    """The prompt text without its trailing '=', or ValueError (input characters and framing only;
    the page's in-browser engine applies the same two checks with the same messages)."""
    text = text.replace(" ", "").removesuffix("=")
    bad = sorted(set(text) - ALLOWED)
    if bad:
        raise ValueError(f"unsupported characters: {''.join(bad)!r} {CHARS_HINT}")
    if not re.match(PATTERN, text):
        raise ValueError(PATTERN_ERROR)
    return text


def prompt(text: str) -> list[int]:
    return list(f"\n{check(text)}=".encode("ascii"))


def solve(model: Transformer, text: str) -> str:
    """Everything the model writes after the prompt."""
    p = prompt(text)
    out = model.generate(p, stop=cm.EOS, max_new=12 * len(p) + 16)
    return bytes(out[len(p):]).decode("ascii").rstrip("\n")


# What each Life dim means (the visualizer shows these on hover; byte features come from the glossary).
DOCS = {
    "alive": "1 if this byte is '#' (a live cell).", "is_colon": "1 if this byte is ':' (after a block's count).",
    "f_C": "Landmark tag for ':', so the 'latest :' head finds this block's separator.",
    "f_L": "Tag (BIG - position) on '=', so a latest-match head finds the earliest '=' (the prompt's).",
    "f_F": "Tag (BIG - position) on '/' and '=', so a head finds the first row end.",
    "prev_eq": "1 if the previous byte is '=' (a block just ended).", "prev_bos": "1 if the previous byte is the start byte.",
    "C": "Position of the latest ':': this block's cells start right after it.",
    "L": "Position of the prompt's '=': every block is this long, so one generation back is L bytes back.",
    "F": "Position of the first row end: a row (with its '/') is F - 2 bytes.",
    "is_count": "1 on a block's count digit: the next byte is ':'.",
    "up_ok": "1 if the cell being written has a row above it (offset >= row length).",
    "down_ok": "1 if the cell being written has a row below it.",
    "self": "1 if this cell was alive one generation back.",
    **{f"a_{k}": f"1 if the {k} neighbour was alive one generation back ('/' ':' '=' read as dead)."
       for k in NEIGHBOURS},
    "pT_slash": "1 if one generation back has '/' at this offset: this byte is a row end too.",
    "pT_eq": "1 if one generation back has '=' at this offset: the grid ends here.",
    **{f"K{d}": f"1 if this block's count digit is {d}." for d in range(10)},
    "n": "Number of live neighbours of the cell being written (0..8).",
    "nxt": "1 if the cell is alive in the next generation: n == 3, or n == 2 and it was alive.",
    "cellp": "1 if the next byte is a cell (not ':', a count digit, or a row end).",
    "out_hash": "Vote for writing '#'.", "out_dot": "Vote for writing '.'.", "out_slash": "Vote for writing '/'.",
    "out_eq": "Vote for writing '='.", "out_colon": "Vote for writing ':'.", "out_nl": "Vote for writing the final newline.",
    **{f"D{d}": f"Vote for writing the count digit {d}." for d in range(9)},
}
