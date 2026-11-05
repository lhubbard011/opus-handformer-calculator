"""The calculator as a recurrent-depth decoder: the answer is the first byte written (LOOPED.md).

An autoregressive model whose block loops per token (engine.py). After the prompt
``\\n<expression>[~N]=`` it thinks in latent positions, one copy of the expression per step,
and then writes the answer and the end byte:

    python -m handformer.loop "2*(3+4)-1"
"""

from __future__ import annotations

import logging

from .. import common as cm
from ..model import Layout
from . import columns, division, parse, parts as P, power, walk
from .engine import DepthBuilder, DepthResult, RecurrentDepthLM

log = logging.getLogger(__name__)

ROLES = ("is_cell", "lat", "k_s", "k_w", "k_q", "k_r", "think", "route", "n_ks")
# registers a position starts with (T moves n_<r> to <r>)
REGS = (*walk.REGS, *columns.REGS, *division.REGS)
STATE = tuple(dict.fromkeys((*columns.STATE, *division.STATE)))  # loop registers (the adapter)
MAX_ITERS = 10_000  # one position's depth
MAX_TOKENS = 2_000_000  # positions after the prompt


def builder() -> DepthBuilder:
    lay = Layout()
    cm.add_input_dims(lay)
    names = [*ROLES, *REGS, *STATE, *P.HOLD.values(), *parse.DIMS, *walk.DIMS, *columns.DIMS,
             *division.DIMS, *power.DIMS]
    seen = set(lay.names)
    lay.add(*[n for n in dict.fromkeys(names) if n not in seen])
    b = DepthBuilder(lay, router="route", think="think")
    cm.embed(b)
    for byte in P.CONTENT:
        b.embed_byte(byte, {"is_cell": 1.0})
    for byte, dim in P.HOLD.items():
        b.hold(dim, byte)
    # T: a thought starts with the bias channel, the "latent" flag, the plan, and its registers
    b.think("one", {"one": 1.0})
    b.think("lat", {"one": 1.0})
    for dim in (*parse.CTX, *power.CTX):
        b.think(dim, {dim: 1.0})
    for r in REGS:
        b.think(r, {f"n_{r}": 1.0})
    for role, src in (("k_s", "n_ks"), ("k_w", "n_kw"), ("k_q", "n_kq"), ("k_r", "n_kr")):
        b.think(role, {src: 1.0})
    for s in STATE:
        b.carry(s, {f"x_{s}": 1.0})
    parse.build(b)
    power.build(b)
    walk.build(b)
    columns.build(b)
    division.build(b)
    # the root that does not halt: the next position is the step's start (the plan is in C_)
    b.at(parse.CTX_LAYER).units += P.flag(P.all_of("is_eq", {"halt": -1.0, "one": 1.0}), "n_ks",
                                          "'n_ks': the next position is the step's start")
    return b


def build() -> RecurrentDepthLM:
    return builder().build()


def prompt(expression: str) -> list[int]:
    return [cm.BOS, *expression.encode("ascii"), ord("=")]


def run(model: RecurrentDepthLM, expression: str, **kw) -> DepthResult:
    return model.run(prompt(expression), stop=cm.EOS, max_tokens=kw.get("max_tokens", MAX_TOKENS),
                     max_iters=kw.get("max_iters", MAX_ITERS), on_iter=kw.get("on_iter"))


def solve(model: RecurrentDepthLM, expression: str) -> str:
    """The answer the model writes (without the end byte)."""
    res = run(model, expression)
    return bytes(res.answer).decode("ascii", "replace").rstrip("\n")
