"""An LL(1) parser written by hand into transformer weights (see circuit.py, docs/PARSER_DESIGN.md).

``circuit`` turns any compiled grammar into weights; the names below are bound to the stage-1
toy grammar (``toy.G``), like the other model packages' entry points."""
from __future__ import annotations

from ..model import ModelBuilder, Transformer
from . import circuit, grammar, toy

TABLE = toy.G.compile()


def builder() -> ModelBuilder:
    return circuit.builder(TABLE)


def build() -> Transformer:
    return circuit.build(TABLE)


def prompt(text: str) -> list[int]:
    return circuit.prompt(text, TABLE)


def solve(model: Transformer, text: str) -> str:
    return circuit.solve(model, TABLE, text)


__all__ = ["TABLE", "build", "builder", "circuit", "grammar", "prompt", "solve", "toy"]
