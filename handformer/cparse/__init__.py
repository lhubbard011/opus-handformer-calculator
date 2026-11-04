"""A C parser written by hand into transformer weights (see circuit.py, docs/PARSER_DESIGN.md).

The model lexes the C source (the highlighter's letters) and then parses its tokens with an
LL(1) pushdown automaton. ``circuit`` turns any compiled grammar and token map into weights; the
names below are bound to the toy grammar (``toy.G``, ``toy.TOKENS``), like the other model
packages' entry points."""
from __future__ import annotations

from ..model import ModelBuilder, Transformer
from . import circuit, grammar, tokens, toy

TABLE = toy.G.compile()
TOKENS = toy.TOKENS


def builder() -> ModelBuilder:
    return circuit.builder(TABLE, TOKENS)


def build() -> Transformer:
    return circuit.build(TABLE, TOKENS)


def prompt(text: str) -> list[int]:
    return circuit.prompt(text)


def solve(model: Transformer, text: str) -> str:
    return circuit.solve(model, text)


def rules() -> list[circuit.Rule]:
    return circuit.rules(TABLE, TOKENS)


def docs() -> dict[str, str]:
    return circuit.docs(TABLE, TOKENS)


__all__ = ["TABLE", "TOKENS", "build", "builder", "circuit", "docs", "grammar", "prompt", "rules", "solve",
           "tokens", "toy"]
