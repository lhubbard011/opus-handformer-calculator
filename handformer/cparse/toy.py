"""The toy grammar: a C-shaped language small enough to read, to prove the pushdown engine.

Its input is real C source (stage 2): the model lexes it first (the C highlighter's letters) and
parses the tokens. The language has what makes C's statements context-free: nested blocks,
statements inside statements, parenthesised expressions, and the dangling ``else`` (it binds to
the nearest ``if``: the table's token entry beats closing). Expressions are identifiers and
numbers joined by ``+``; anything else C has (``int``, ``*``, strings, ...) is a syntax error.

The grammar's terminals are one byte each, so every C token has a code (``TOKENS``):
``i`` = ``if``, ``e`` = ``else``, ``w`` = ``while``, ``x`` = any identifier, ``1`` = any number.
"""
from .grammar import END, Grammar, alt, opt, seq, star
from .tokens import Tok

NAMES = {"P": "program", "S": "statement", "B": "block", "E": "expression", "F": "factor"}
TOKENS = {"i": Tok("keyword", "if"), "e": Tok("keyword", "else"), "w": Tok("keyword", "while"),
          "x": Tok("identifier"), "1": Tok("number"),
          **{c: Tok("punct", c) for c in "{}();+"}}

G = Grammar("P", NAMES)
G.rule("P", seq(star("S"), END), "a program: statements, then the end of the input")
G.rule("S", alt("B",
                seq("i", "(", "E", ")", "S", opt("e", "S")),
                seq("w", "(", "E", ")", "S"),
                seq("E", ";"),
                ";"),
       "a statement: a block, if (with an optional else), while, an expression and ';', or ';' alone")
G.rule("B", seq("{", star("S"), "}"), "a block: statements between braces")
G.rule("E", seq("F", star("+", "F")), "an expression: factors joined by '+'")
G.rule("F", alt("x", "1", seq("(", "E", ")")), "a factor: an identifier, a number, or a parenthesised expression")
