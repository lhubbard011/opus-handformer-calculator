"""Plain-Python C lexer (C11 §6.4, §6.10 line rule), written separately from the weights.
Returns one state letter per source byte (see handformer/csyntax/circuit.py STATES).

Covered: keywords, identifiers (with universal character names), pp-numbers, strings and
character constants with escapes and prefixes (L u U u8), block and line comments,
directives ('#', '%:' or '??=' first on a line), backslash-newline splices in line comments,
directives, strings, identifiers and numbers, the trigraph '??/' acting as a backslash, and
punctuators by the longest match (the first byte's letter gives the length, the rest count down)."""

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
PUNCTUATORS = {  # C11 §6.4.6, digraphs included
    "[", "]", "(", ")", "{", "}", ".", "->", "++", "--", "&", "*", "+", "-", "~", "!", "/", "%", "<<", ">>",
    "<", ">", "<=", ">=", "==", "!=", "^", "|", "&&", "||", "?", ":", ";", "...", "=", "*=", "/=", "%=", "+=",
    "-=", "<<=", ">>=", "&=", "^=", "|=", ",", "#", "##", "<:", ":>", "<%", "%>", "%:", "%:%:",
}
# a punctuator's first byte says its length; the later bytes count down what is left
FIRST = {1: "o", 2: "q", 3: "r", 4: "v"}
NEXT = {"q": "j", "r": "l", "v": "m", "m": "l", "l": "j"}


def lex(src: str) -> str:
    out = []
    state = None  # the previous letter; None = start
    for i, ch in enumerate(src):
        prev = src[i - 1] if i else "\n"
        nxt = src[i + 1] if i + 1 < len(src) else ""
        nxt2 = src[i + 2] if i + 2 < len(src) else ""
        # the byte before acts as a backslash: '\' itself, or the '/' ending the trigraph '??/'
        prev_bs = prev == "\\" or (i >= 3 and src[i - 3:i] == "??/")
        trigraph_bs = src[i:i + 3] == "??/"  # this byte starts a '??/' (a backslash)
        splice = (ch == "\\" and nxt == "\n") or (ch == "\n" and prev_bs)  # backslash-newline: deleted (phase 2)
        ucn = ch == "\\" and nxt in "uU" and nxt != ""
        # inside a token: continue it, or fall through to "outside" below
        if state in ("L", "c"):
            letter = "_" if ch == "\n" and not prev_bs else "c"
        elif state == "B":
            letter = "D"
        elif state in ("D", "C"):
            letter = "E" if state == "C" and ch == "/" and prev == "*" else "C"
        elif state in ("s", "t"):
            letter = "G" if trigraph_bs else {"\\": "S", '"': "z", "\n": "_"}.get(ch, "s")
        elif state == "G":
            letter = "H"
        elif state == "H":
            letter = "S"
        elif state == "S":
            letter = "t"
        elif state in ("a", "b"):
            letter = "g" if trigraph_bs else {"\\": "A", "'": "y", "\n": "_"}.get(ch, "a")
        elif state == "g":
            letter = "h"
        elif state == "h":
            letter = "A"
        elif state == "A":
            letter = "b"
        elif state == "P":
            letter = {'"': "s", "'": "a"}.get(ch, "P")
        elif state == "p":
            letter = "_" if ch == "\n" and not prev_bs else "p"
        elif state == "n" and (ch in IDENT or ch == "." or (ch in "+-" and prev in "eEpP") or splice):
            letter = "n"
        elif state in ("i", "k") and (ch in IDENT or splice or ucn):
            letter = state
        elif state in NEXT:
            letter = NEXT[state]
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
            elif all(c in SPACE for c in line) and (ch == "#" or src[i:i + 2] == "%:" or src[i:i + 3] == "??="):
                letter = "p"
            elif (ch in "LUu" and nxt in "\"'" and nxt != "") or (ch == "u" and nxt == "8" and nxt2 == '"'):
                letter = "P"  # a string or character prefix
            elif ucn:
                letter = "i"  # an identifier starting with a universal character name
            elif ch.isdigit() or (ch == "." and nxt.isdigit()):
                letter = "n"
            elif ch in START:
                j = i
                while j < len(src) and src[j] in IDENT:
                    j += 1
                letter = "k" if src[i:j] in KEYWORDS else "i"
            elif ch in SPACE:
                letter = "_"
            else:  # a punctuator: the longest one that starts here (any other byte counts as one byte)
                n = max((len(p) for p in PUNCTUATORS if src.startswith(p, i)), default=1)
                letter = FIRST[n]
        out.append(letter)
        state = letter
    return "".join(out)
