"""The parser's tokens: what each grammar terminal is in C, and the token list of a source.

The grammar's terminals are one byte each (grammar.py), so a C token needs a code: ``i`` stands
for the keyword ``if``, ``x`` for any identifier, ``1`` for any number, ``+`` for the punctuator
``+``. A ``TokenMap`` says which C token each terminal is.

The model finds its tokens in the letters its lexer writes (handformer/csyntax): a token starts
where the lexer's rule there starts one. ``tokens`` reads the same thing off the letters with
plain Python, from the letter pairs alone, so the simulator and the tests share no code with the
weights: a byte starts a token when its letter can start one and does not continue the token
before it (same letter for a word or a number; inside a literal; a punctuator's countdown).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from ..csyntax import circuit as lexer
from . import grammar as gm

log = logging.getLogger(__name__)

KINDS = ("keyword", "identifier", "number", "punct")


@dataclass(frozen=True)
class Tok:
    """One kind of C token: ``kind`` with ``text`` (the keyword or punctuator; empty for any
    identifier or any number)."""
    kind: str
    text: str = ""

    def describe(self) -> str:
        return {"identifier": "an identifier", "number": "a number"}.get(self.kind, repr(self.text))


TokenMap = dict[str, Tok]  # grammar terminal -> the C token it stands for

PUNCT_FIRST = {v: n for n, v in lexer.PUNCT_START.items()}  # first letter -> punctuator length
PUNCT_REST = "mlj"  # a punctuator's later bytes (handformer/csyntax: they count down)
WORD = "ikn"  # identifier, keyword, number: the token goes on while the letter stays the same
STRING_IN, CHAR_IN = "sStGH", "aAbgh"  # inside a string / a character constant (not its end)
IDENT = lexer.CLASSES["c_ident"]


def check_map(table: gm.Table, cmap: TokenMap) -> None:
    """Every terminal but the end is one C token, and two terminals are never the same token."""
    terms = [t for t in table.terminals if t != gm.END]
    missing = [t for t in terms if t not in cmap]
    if missing:
        raise gm.GrammarError(f"terminals with no C token: {missing}")
    seen: dict[Tok, str] = {}
    for t in terms:
        tok = cmap[t]
        if tok.kind not in KINDS:
            raise gm.GrammarError(f"terminal {t!r}: unknown token kind {tok.kind!r}")
        if tok.kind == "keyword" and tok.text not in lexer.KEYWORDS:
            raise gm.GrammarError(f"terminal {t!r}: {tok.text!r} is not a C keyword")
        if tok.kind == "punct" and tok.text not in lexer.PUNCTUATORS:
            raise gm.GrammarError(f"terminal {t!r}: {tok.text!r} is not a C punctuator")
        if tok in seen:
            raise gm.GrammarError(f"terminals {seen[tok]!r} and {t!r} are the same token {tok}")
        seen[tok] = t


def starts(letters: str) -> list[bool]:
    """For each source byte: does a token start here (from the letters alone)?"""
    out = []
    for i, cur in enumerate(letters):
        prev = letters[i - 1] if i else "_"
        if cur in PUNCT_FIRST:
            out.append(True)
        elif cur in WORD:
            out.append(prev != cur)
        elif cur == "P":  # a literal's prefix (u8 is two bytes)
            out.append(prev != "P")
        elif cur == "s":  # a string's opening quote, unless it follows its prefix or is its body
            out.append(prev not in STRING_IN + "P")
        elif cur == "a":
            out.append(prev not in CHAR_IN + "P")
        else:
            out.append(False)
    return out


def tokens(source: str, letters: str, cmap: TokenMap) -> list[tuple[int, str | None]]:
    """(prompt position, terminal or None) per token, then the end of the input (END).

    Prompt position = source index + 1 (byte 0 is the start byte). A token that is no terminal
    of the grammar (a string, ``int``, ``+=``) is kept as None: the parser sees it, matches no
    table entry, and reports it."""
    if len(letters) != len(source):
        raise ValueError(f"{len(letters)} letters for {len(source)} source bytes")
    by_tok = {tok: t for t, tok in cmap.items()}
    out: list[tuple[int, str | None]] = []
    st = starts(letters)
    for i, is_start in enumerate(st):
        if not is_start:
            continue
        cur = letters[i]
        if cur in PUNCT_FIRST:
            tok = Tok("punct", source[i:i + PUNCT_FIRST[cur]])
        elif cur == "k":  # the word's identifier bytes (a splice inside it: the lexer's documented limit)
            j = i
            while j < len(source) and source[j] in IDENT:
                j += 1
            tok = Tok("keyword", source[i:j])
        elif cur == "i":
            tok = Tok("identifier")
        elif cur == "n":
            tok = Tok("number")
        else:  # a string or character literal (no terminal in the grammars so far)
            tok = None
        out.append((i + 1, by_tok.get(tok) if tok is not None else None))
    out.append((len(source) + 1, gm.END))
    log.debug("%d tokens in %d source bytes", len(out), len(source))
    return out


def texts(source: str, letters: str) -> dict[int, str]:
    """Prompt position -> the token's source text, for every token (for showing a parse tree)."""
    st = starts(letters)
    out = {}
    for i, is_start in enumerate(st):
        if not is_start:
            continue
        j = i + 1
        while j < len(source) and not st[j] and letters[j] not in "_BLp":
            j += 1
        out[i + 1] = source[i:j]
    return out
