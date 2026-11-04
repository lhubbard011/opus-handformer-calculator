"""Plain recursive-descent parser for the stage-1 toy grammar (handformer/cparse/toy.py),
written separately from the grammar compiler. Returns the parse tree as an S-expression in the
format of ``grammar.tree``, or for a syntax error the prompt position of the offending token.

    P -> S* $        S -> B | i ( E ) S [e S] | w ( E ) S | E ; | ;
    B -> { S* }      E -> F (+ F)*        F -> x | 1 | ( E )
"""

SPACE = " \t\n\r"


class SyntaxErr(Exception):
    def __init__(self, pos: int) -> None:
        super().__init__(pos)
        self.pos = pos


class Parser:
    def __init__(self, src: str) -> None:
        self.src = src
        self.i = 0  # index into src of the next unread byte

    def peek(self) -> tuple[str, int]:
        """The next token and its prompt position (source index + 1); '$' at the end."""
        while self.i < len(self.src) and self.src[self.i] in SPACE:
            self.i += 1
        if self.i == len(self.src):
            return "$", self.i + 1
        t = self.src[self.i]
        return ("?" if t == "$" else t), self.i + 1  # a '$' byte is no token (only the end is '$')

    def take(self, want: str) -> str:
        t, pos = self.peek()
        if t != want:
            raise SyntaxErr(pos)
        self.i += 1
        return "'" + t + "'" if t in "()'" else t

    def program(self) -> str:
        parts = []
        while self.peek()[0] in "{iwx1(;":
            parts.append(self.statement())
        parts.append(self.take("$"))
        return "(P " + " ".join(parts) + ")"

    def statement(self) -> str:
        t, pos = self.peek()
        if t == "{":
            body = ["{"]
            self.i += 1
            while self.peek()[0] in "{iwx1(;":
                body.append(self.statement())
            body.append(self.take("}"))
            return "(S (B " + " ".join(body) + "))"
        if t in "iw":
            parts = [self.take(t), self.take("("), self.expression(), self.take(")"), self.statement()]
            if t == "i" and self.peek()[0] == "e":  # the nearest if takes the else
                parts += [self.take("e"), self.statement()]
            return "(S " + " ".join(parts) + ")"
        if t == ";":
            return "(S " + self.take(";") + ")"
        if t in "x1(":
            return "(S " + self.expression() + " " + self.take(";") + ")"
        raise SyntaxErr(pos)

    def expression(self) -> str:
        parts = [self.factor()]
        while self.peek()[0] == "+":
            parts += [self.take("+"), self.factor()]
        return "(E " + " ".join(parts) + ")"

    def factor(self) -> str:
        t, pos = self.peek()
        if t in "x1":
            return "(F " + self.take(t) + ")"
        if t == "(":
            return "(F " + self.take("(") + " " + self.expression() + " " + self.take(")") + ")"
        raise SyntaxErr(pos)


def parse(src: str) -> str | int:
    """The tree, or the prompt position of the first token that cannot continue a program."""
    try:
        return Parser(src).program()
    except SyntaxErr as e:
        return e.pos
