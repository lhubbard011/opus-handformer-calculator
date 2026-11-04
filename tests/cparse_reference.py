"""Plain recursive-descent parser for the toy grammar (handformer/cparse/toy.py), written
separately from the grammar compiler. Takes the token list (prompt position, terminal code or
None, the last one the end '$') and returns the parse tree as an S-expression in the format of
``grammar.tree``, or for a syntax error the prompt position of the offending token.

    P -> S* $        S -> B | i ( E ) S [e S] | w ( E ) S | E ; | ;
    B -> { S* }      E -> F (+ F)*        F -> x | 1 | ( E )
"""
STATEMENT_START = "{iwx1(;"


class SyntaxErr(Exception):
    def __init__(self, pos: int) -> None:
        super().__init__(pos)
        self.pos = pos


class Parser:
    def __init__(self, toks: list[tuple[int, str | None]]) -> None:
        self.toks = toks
        self.i = 0  # index of the next unread token

    def peek(self) -> tuple[str, int]:
        """The next token's code ('?' for a token the grammar does not know) and position."""
        pos, t = self.toks[self.i]
        return ("?" if t is None else t), pos

    def take(self, want: str) -> str:
        t, pos = self.peek()
        if t != want:
            raise SyntaxErr(pos)
        self.i += 1
        return "'" + t + "'" if t in "()'" else t

    def program(self) -> str:
        parts = []
        while self.peek()[0] in STATEMENT_START:
            parts.append(self.statement())
        parts.append(self.take("$"))
        return "(P " + " ".join(parts) + ")"

    def statement(self) -> str:
        t, pos = self.peek()
        if t == "{":
            body = [self.take("{")]
            while self.peek()[0] in STATEMENT_START:
                body.append(self.statement())
            body.append(self.take("}"))
            return "(S (B " + " ".join(body) + "))"
        if t in ("i", "w"):
            parts = [self.take(t), self.take("("), self.expression(), self.take(")"), self.statement()]
            if t == "i" and self.peek()[0] == "e":  # the nearest if takes the else
                parts += [self.take("e"), self.statement()]
            return "(S " + " ".join(parts) + ")"
        if t == ";":
            return "(S " + self.take(";") + ")"
        if t in ("x", "1", "("):
            return "(S " + self.expression() + " " + self.take(";") + ")"
        raise SyntaxErr(pos)

    def expression(self) -> str:
        parts = [self.factor()]
        while self.peek()[0] == "+":
            parts += [self.take("+"), self.factor()]
        return "(E " + " ".join(parts) + ")"

    def factor(self) -> str:
        t, pos = self.peek()
        if t in ("x", "1"):
            return "(F " + self.take(t) + ")"
        if t == "(":
            return "(F " + self.take("(") + " " + self.expression() + " " + self.take(")") + ")"
        raise SyntaxErr(pos)


def parse(toks: list[tuple[int, str | None]]) -> str | int:
    """The tree, or the prompt position of the first token that cannot continue a program."""
    try:
        return Parser(toks).program()
    except SyntaxErr as e:
        return e.pos
