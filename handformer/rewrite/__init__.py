"""The looped calculator: the answer is the first byte written (REWRITE.md).

The same expressions as the frame-writing calculator (handformer/arith.py), evaluated by one
recurrent block that rewrites the prompt in place, pass after pass, until its router stops it.
The coda then writes the answer and nothing else::

    \\n2*(3+4)-1=13\\n

    python -m handformer.rewrite "2*(3+4)-1"
"""

from __future__ import annotations

import logging

from .. import common as cm
from ..model import Layout
from . import circuit, division, power
from .engine import LoopBuilder, LoopedTransformer, LoopResult

log = logging.getLogger(__name__)

MAX_PASSES = 100_000


def builder() -> LoopBuilder:
    lay = Layout()
    cm.add_input_dims(lay)
    lay.add(*circuit.CARRIED, *circuit.DIMS, *circuit.HOLD.values(), *power.DIMS, *division.DIMS, *circuit.CODA_DIMS)
    b = LoopBuilder(lay, router="route", grow="grow")
    cm.embed(b)
    circuit.build(b)
    power.build(b)
    division.build(b)
    circuit.build_coda(b)
    return b


def prompt(expression: str) -> list[int]:
    return [cm.BOS, *expression.encode("utf-8"), ord("=")]


def run(model: LoopedTransformer, expression: str, max_passes: int = MAX_PASSES) -> LoopResult:
    res = model.run(prompt(expression), stop=cm.EOS, max_passes=max_passes)
    log.info("looped %r: %d passes, answer %r", expression, len(res.passes), bytes(res.answer).decode(errors="replace"))
    return res


def solve(model: LoopedTransformer, expression: str) -> str:
    """The answer the coda writes (without its final '\\n')."""
    return bytes(run(model, expression).answer).decode(errors="replace").rstrip("\n")
