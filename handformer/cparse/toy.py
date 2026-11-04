"""Stage 1 grammar: a C-shaped toy language with one-byte tokens, to prove the pushdown engine.

Tokens: ``{ } ( ) ; + x 1`` and three one-letter keywords, ``i`` (if), ``e`` (else), ``w``
(while). Whitespace separates tokens and is skipped. It has what makes C's statements
context-free: nested blocks, statements inside statements, parenthesised expressions, and the
dangling ``else`` (it binds to the nearest ``if``: the table's token entry beats closing).
"""
from .grammar import END, Grammar, alt, opt, seq, star

NAMES = {"P": "program", "S": "statement", "B": "block", "E": "expression", "F": "factor"}
TOKENS = "{}();+x1iew"

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
G.rule("F", alt("x", "1", seq("(", "E", ")")), "a factor: x, 1, or a parenthesised expression")
