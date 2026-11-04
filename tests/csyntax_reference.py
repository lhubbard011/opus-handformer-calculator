"""Plain-Python C lexer (C11 §6.4, §6.10 line rule), written separately from the weights.
Returns one state letter per source byte (see handformer/csyntax/circuit.py STATES)."""

KEYWORDS = {
    "auto", "break", "case", "char", "const", "continue", "default", "do", "double", "else", "enum", "extern",
    "float", "for", "goto", "if", "inline", "int", "long", "register", "restrict", "return", "short", "signed",
    "sizeof", "static", "struct", "switch", "typedef", "union", "unsigned", "void", "volatile", "while",
    "_Alignas", "_Alignof", "_Atomic", "_Bool", "_Complex", "_Generic", "_Imaginary", "_Noreturn",
    "_Static_assert", "_Thread_local",
}
START = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_")
IDENT = START | set("0123456789")
SPACE = set(" \t\n\r\v\f")


def lex(src: str) -> str:
    out = []
    state = None  # the previous letter; None = start
    for i, ch in enumerate(src):
        prev = src[i - 1] if i else "\n"
        nxt = src[i + 1] if i + 1 < len(src) else ""
        # inside a token: continue it, or fall through to "outside" below
        if state in ("L", "c"):
            letter = "_" if ch == "\n" and prev != "\\" else "c"
        elif state == "B":
            letter = "D"
        elif state in ("D", "C"):
            letter = "E" if state == "C" and ch == "/" and prev == "*" else "C"
        elif state in ("s", "t"):
            letter = {"\\": "S", '"': "z", "\n": "_"}.get(ch, "s")
        elif state == "S":
            letter = "t"
        elif state in ("a", "b"):
            letter = {"\\": "A", "'": "y", "\n": "_"}.get(ch, "a")
        elif state == "A":
            letter = "b"
        elif state == "p":
            letter = "_" if ch == "\n" and prev != "\\" else "p"
        elif state == "n" and (ch in IDENT or ch == "." or (ch in "+-" and prev in "eEpP")):
            letter = "n"
        elif state in ("i", "k") and ch in IDENT:
            letter = state
        else:
            letter = None
        if letter is None:  # outside any token: what starts here
            line = src[src.rfind("\n", 0, i) + 1:i]
            if ch == "/" and nxt == "*":
                letter = "B"
            elif ch == "/" and nxt == "/":
                letter = "L"
            elif ch == '"':
                letter = "s"
            elif ch == "'":
                letter = "a"
            elif ch == "#" and all(c in SPACE for c in line):
                letter = "p"
            elif ch.isdigit() or (ch == "." and nxt.isdigit()):
                letter = "n"
            elif ch in START:
                j = i
                while j < len(src) and src[j] in IDENT:
                    j += 1
                letter = "k" if src[i:j] in KEYWORDS else "i"
            elif ch in SPACE:
                letter = "_"
            else:
                letter = "o"
        out.append(letter)
        state = letter
    return "".join(out)
