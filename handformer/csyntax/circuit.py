"""A C syntax highlighter written by hand into transformer weights: the lexer is a table of rules.

Sequence (the host only feeds bytes, takes the argmax, stops at '\\n'):

    \\n<C source>\\x03<one state letter per source byte>\\n

For every source byte the model writes one letter: the lexer state after that byte. The page
colours the source with them. The lexical grammar follows C11 §6.4 (tokens, comments) and §6.10
(a line starting with '#' is a preprocessing directive).

The whole idea: the state is the model's own previous letter, which at every output step is
the current byte, so the text is the memory. Each step reads that letter, the source byte being
classified and the bytes around it (exact lookups into the prompt), and applies one rule from
``RULES`` below. Each rule is one AND unit with its own named dim; a fallback rule ("anything
else in a string") is its group minus its siblings, so exactly one rule fires per byte. The
unembedding reads the rule dims directly: a rule's dim votes for its letter.

Keywords are decided at a word's first byte: one AND unit per C11 keyword checks the next
characters (the source is all in the prompt, so the model can look ahead) and that the word
ends there. The later bytes of the word keep the state ('k' or 'i').

    layer 1  MLP   landmark tags: the end marker, the latest newline, the latest non-space byte
    layer 2  attn  read them: E (end marker), Lnl, Lns at every position
    layer 3  attn  read the source byte being classified (s = pos - E + 1), the bytes before and
                   after it, and the next 14 bytes' letters (for keywords)
             MLP   keyword matches; "first non-space on its line"; "+/- after an exponent letter"
    layer 4  MLP   the rules inside comments, strings, characters, directives, numbers, words;
                   "outside any token" (norm)
    layer 5  MLP   the fallbacks of those groups; the rules outside any token
    layer 6  MLP   the last fallback: any other byte outside a token is a punctuator

Limits (documented, not hidden): trigraphs and digraphs are not translated; a backslash-newline
splices only line comments, directives and strings, not other tokens; string prefixes (L"x",
u8"x") colour the prefix as an identifier; macros are not expanded (this is a lexer).
"""
from __future__ import annotations

import logging
import re
import string
from dataclasses import dataclass, field

from .. import primitives as pr
from ..model import Form, Layout, ModelBuilder, Transformer, UnitSpec, authored_at

log = logging.getLogger(__name__)

END = 3  # '\x03' ends the source in the prompt
BOS = EOS = ord("\n")

# The state letters the model writes (one per source byte) and what each means.
STATES = {
    "_": "whitespace outside any token", "o": "a punctuator or operator", "i": "an identifier", "k": "a keyword",
    "n": "a number (pp-number)",
    "s": "a string literal: its opening quote or body", "S": "a backslash starting an escape in a string",
    "t": "the escaped character in a string", "z": "the closing quote of a string",
    "a": "a character constant: its opening quote or body", "A": "a backslash in a character constant",
    "b": "the escaped character in a character constant", "y": "the closing quote of a character constant",
    "B": "the '/' that opens a block comment", "D": "the '*' that opens a block comment",
    "C": "inside a block comment", "E": "the '/' that closes a block comment",
    "L": "the first '/' of a line comment", "c": "inside a line comment", "p": "a preprocessing directive",
}
# How the page colours them.
DISPLAY = {"comment": "BDCELc", "string": "sStz", "char": "aAby", "preproc": "p", "number": "n",
           "keyword": "k", "ident": "i", "punct": "o", "space": "_"}

KEYWORDS = (  # C11 §6.4.1
    "auto", "break", "case", "char", "const", "continue", "default", "do", "double", "else", "enum", "extern",
    "float", "for", "goto", "if", "inline", "int", "long", "register", "restrict", "return", "short", "signed",
    "sizeof", "static", "struct", "switch", "typedef", "union", "unsigned", "void", "volatile", "while",
    "_Alignas", "_Alignof", "_Atomic", "_Bool", "_Complex", "_Generic", "_Imaginary", "_Noreturn",
    "_Static_assert", "_Thread_local",
)
MAX_KW = max(map(len, KEYWORDS))
LETTERS_AT = [sorted({k[j] for k in KEYWORDS if len(k) > j}) for j in range(MAX_KW)]  # letters a keyword has at offset j

# Byte classes of the source (each a 0/1 feature of the byte embedding).
CLASSES = {
    "c_slash": "/", "c_star": "*", "c_dq": '"', "c_sq": "'", "c_hash": "#", "c_bs": "\\", "c_nl": "\n", "c_dot": ".",
    "c_pm": "+-", "c_digit": string.digits, "c_start": string.ascii_letters + "_",
    "c_ident": string.ascii_letters + string.digits + "_", "c_space": " \t\n\r\v\f", "c_exp": "eEpP", "c_mark": chr(END),
}
XFEATS = ("c_slash", "c_star", "c_dq", "c_sq", "c_hash", "c_bs", "c_nl", "c_dot", "c_pm", "c_digit", "c_start",
          "c_ident", "c_space", "c_mark")
x_ = lambda c: "x_" + c[2:]  # noqa: E731  the classes of the byte being classified
st = lambda letter: "st__" if letter == "_" else f"st_{letter}"  # noqa: E731  "the current byte is this letter"


@dataclass
class Rule:
    """One lexer rule: in states ``src``, when every dim in ``when`` is 1 and none in ``unless``,
    the byte is ``out``. ``fallback``: the rest of the group (group minus the listed siblings).
    ``src`` may also be "norm": outside any token."""
    name: str
    src: str
    out: str
    doc: str
    when: tuple[str, ...] = ()
    unless: tuple[str, ...] = ()
    fallback: tuple[str, ...] | None = None
    layer: int = 4
    where: str = field(default_factory=authored_at, compare=False)


# ---------------------------------------------------------------- the lexer, as rules
# Read top to bottom: each group is one lexer state; its rules are mutually exclusive by the
# byte they look at, and its fallback catches the rest.
RULES = [
    # line comment: '//' to the end of the line; a backslash-newline continues it (C11 §5.1.1.2 phase 2)
    Rule("line_end", "Lc", "_", "a newline ends a line comment (unless a backslash splices the lines)",
         when=("x_nl",), unless=("p_bs",)),
    Rule("line", "Lc", "c", "anything else continues the line comment", fallback=("r_line_end",), layer=5),
    # block comment
    Rule("block_star", "B", "D", "the byte after an opening '/' is its '*'"),
    Rule("block_end", "C", "E", "'*/' closes the block comment (a '*' that opened it does not count)",
         when=("x_slash", "p_star")),
    Rule("block", "DC", "C", "anything else stays in the block comment", fallback=("r_block_end",), layer=5),
    # string literal
    Rule("str_escape", "st", "S", "in a string, a backslash starts an escape", when=("x_bs",)),
    Rule("str_close", "st", "z", "in a string, '\"' closes it", when=("x_dq",)),
    Rule("str_newline", "st", "_", "an unterminated string ends at the newline", when=("x_nl",)),
    Rule("str", "st", "s", "anything else is string body", fallback=("r_str_escape", "r_str_close", "r_str_newline"),
         layer=5),
    Rule("str_escaped", "S", "t", "the byte after an escaping backslash is part of the escape"),
    # character constant
    Rule("chr_escape", "ab", "A", "in a character constant, a backslash starts an escape", when=("x_bs",)),
    Rule("chr_close", "ab", "y", "in a character constant, \"'\" closes it", when=("x_sq",)),
    Rule("chr_newline", "ab", "_", "an unterminated character constant ends at the newline", when=("x_nl",)),
    Rule("chr", "ab", "a", "anything else is character-constant body",
         fallback=("r_chr_escape", "r_chr_close", "r_chr_newline"), layer=5),
    Rule("chr_escaped", "A", "b", "the byte after an escaping backslash is part of the escape"),
    # preprocessing directive: to the end of the line, backslash-newline continues it
    Rule("pp_end", "p", "_", "a newline ends the directive (unless a backslash splices the lines)",
         when=("x_nl",), unless=("p_bs",)),
    Rule("pp", "p", "p", "anything else continues the directive", fallback=("r_pp_end",), layer=5),
    # number (pp-number, C11 §6.4.8): digits, letters, '_', '.', and '+'/'-' right after e E p P
    Rule("num", "n", "n", "a pp-number continues with a digit, letter, '_', '.' or an exponent sign",
         when=("numcont",)),
    # identifier or keyword: the word keeps the state its first byte got
    Rule("word_k", "k", "k", "a keyword continues with identifier characters", when=("x_ident",)),
    Rule("word_i", "i", "i", "an identifier continues with identifier characters", when=("x_ident",)),
    # outside any token (norm): what the next token is
    Rule("open_block", "norm", "B", "'/' followed by '*' opens a block comment", when=("x_slash", "n_star"), layer=5),
    Rule("open_line", "norm", "L", "'/' followed by '/' opens a line comment", when=("x_slash", "n_slash"), layer=5),
    Rule("open_str", "norm", "s", "'\"' opens a string literal", when=("x_dq",), layer=5),
    Rule("open_chr", "norm", "a", "\"'\" opens a character constant", when=("x_sq",), layer=5),
    Rule("pp_start", "norm", "p", "'#' first on its line starts a preprocessing directive",
         when=("x_hash", "linestart"), layer=5),
    Rule("num_start", "norm", "n", "a digit starts a number", when=("x_digit",), layer=5),
    Rule("num_dot", "norm", "n", "'.' followed by a digit starts a number", when=("x_dot", "n_digit"), layer=5),
    Rule("keyword", "norm", "k", "a word that is a C11 keyword", when=("x_start", "kw"), layer=5),
    Rule("ident", "norm", "i", "any other word is an identifier", when=("x_start",), unless=("kw",), layer=5),
    Rule("space", "norm", "_", "whitespace outside a token", when=("x_space",), layer=5),
    Rule("punct", "norm", "o", "any other byte outside a token is a punctuator", layer=6,
         fallback=("r_open_block", "r_open_line", "r_open_str", "r_open_chr", "r_pp_start", "r_num_start",
                   "r_num_dot", "r_keyword", "r_ident", "r_space")),
]
NORMAL_STATES = "_ozyE"  # after these the next byte starts fresh (plus the end marker: the first byte)

DIMS = (
    *CLASSES, *(f"ch_{c}" for c in sorted(set("".join(KEYWORDS)))), *(st(s) for s in STATES),
    "f_E", "f_nl", "f_ns", "E", "Lnl", "Lns",
    *(x_(c) for c in XFEATS), "p_star", "p_bs", "p_exp", "pLnl", "pLns", "n_star", "n_slash", "n_digit",
    *(f"k{j}_{c}" for j in range(MAX_KW) for c in LETTERS_AT[j]), *(f"id{j}" for j in range(MAX_KW + 1)),
    "kw", "linestart", "pexp_pm", "numcont", "norm", "r_end",
    *(f"r_{r.name}" for r in RULES),
)


def _group(src: str) -> Form:
    """0/1: the current byte (the previous output letter) is one of the states in ``src``."""
    if src == "norm":
        return {"norm": 1.0}
    return {st(s): 1.0 for s in src}


def _rule_unit(r: Rule) -> UnitSpec:
    """AND of the rule's group and conditions (all exact 0/1), or its group minus the siblings."""
    read = dict(_group(r.src))
    if r.fallback is not None:
        for sib in r.fallback:
            read[sib] = read.get(sib, 0.0) - 1.0
    else:
        for d in r.when:
            read[d] = read.get(d, 0.0) + 1.0
        for d in r.unless:
            read[d] = read.get(d, 0.0) - 1.0
        read["one"] = read.get("one", 0.0) - float(len(r.when))
    read["x_mark"] = -1.0  # past the last source byte only r_end fires (exactly one rule per byte)
    states = "outside any token" if r.src == "norm" else "in state " + "/".join(repr(s) for s in r.src)
    return UnitSpec(read=read, write={f"r_{r.name}": 1.0}, where=r.where,
                    label=f"rule {r.name}: {states}, {r.doc} -> '{r.out}'")


def builder() -> ModelBuilder:
    """Every weight of the C highlighter, at global layers 1-6 (see the module docstring)."""
    lay = Layout()
    lay.add("one", "pos", "pos2", *DIMS)
    b = ModelBuilder(lay)
    # Byte embedding: the constant, the source byte's classes and keyword letter, and (for the
    # letters the model writes) which lexer state the byte names.
    for byte in range(256):
        ch = chr(byte)
        form = {"one": 1.0, **{c: 1.0 for c, chars in CLASSES.items() if ch in chars}}
        if f"ch_{ch}" in lay:
            form[f"ch_{ch}"] = 1.0
        if ch in STATES:
            form[st(ch)] = 1.0
        b.embed_byte(byte, form)

    # ---- Layer 1. MLP: landmark tags (2(pos+1) on a match; the start byte scores 1).
    l1 = b.at(1)
    # The start byte is '\n' like every newline of the source, so "the start" is a position test:
    # ReLU(1 - pos) is 1 at position 0 only. It is each tag's "no match yet" fallback.
    for dst, match in (("f_E", {"c_mark": 1.0}), ("f_nl", {"c_nl": 1.0}), ("f_ns", {"one": 1.0, "c_space": -1.0})):
        l1.units += pr.latest_feature(match, dst, bos=False)
        l1.units.append(pr.relu_unit({"one": 1.0, "pos": -1.0}, {dst: 1.0},
                                     f"fallback for '{dst}': position 0 scores 1, so 'no match yet' reads position 0"))

    # ---- Layer 2. Heads: the end marker E, and at every position its latest newline / non-space.
    l2 = b.at(2)
    l2.heads += [pr.latest_head("end marker", "f_E", "E"), pr.latest_head("latest newline", "f_nl", "Lnl"),
                 pr.latest_head("latest non-space byte", "f_ns", "Lns")]

    # ---- Layer 3. Heads: the source byte s = pos - E + 1 being classified, its neighbours, and the
    # letters ahead for keywords. MLP: keyword matches and two small facts.
    l3 = b.at(3)
    s = {"pos": 1.0, "E": -1.0, "one": 1.0}
    x = pr.gather_head("the source byte being classified", s, "x_slash", src="c_slash")
    x.value = {c: {x_(c): 1.0} for c in XFEATS}
    prev = pr.gather_head("the byte before it", pr.add(s, pr.const(-1.0)), "p_star", src="c_star")
    prev.value = {"c_star": {"p_star": 1.0}, "c_bs": {"p_bs": 1.0}, "c_exp": {"p_exp": 1.0},
                  "Lnl": {"pLnl": 1.0}, "Lns": {"pLns": 1.0}}
    nxt = pr.gather_head("the byte after it", pr.add(s, pr.const(1.0)), "n_star", src="c_star")
    nxt.value = {"c_star": {"n_star": 1.0}, "c_slash": {"n_slash": 1.0}, "c_digit": {"n_digit": 1.0}}
    l3.heads += [x, prev, nxt]
    for j in range(MAX_KW + 1):  # offset j: the letters a keyword can have there, and "is it an identifier byte"
        h = pr.gather_head(f"byte {j} of the word (for keywords)", pr.add(s, pr.const(float(j))), f"id{j}", src="c_ident")
        h.value = {"c_ident": {f"id{j}": 1.0},
                   **({f"ch_{c}": {f"k{j}_{c}": 1.0} for c in LETTERS_AT[j]} if j < MAX_KW else {})}
        l3.heads.append(h)
    for kw in KEYWORDS:  # all of the keyword's letters, then a byte that ends the word: AND of 0/1 terms
        read: Form = {f"k{j}_{c}": 1.0 for j, c in enumerate(kw)}
        read[f"id{len(kw)}"] = -1.0
        read["one"] = -float(len(kw) - 1)
        l3.units.append(UnitSpec(read=read, write={"kw": 1.0},
                                 label=f"keyword '{kw}': the word here spells {kw} and ends after {len(kw)} bytes"))
    # '#' is first on its line when no non-space byte comes after the latest newline before it
    l3.units += pr.clamp01_units({"pLnl": 1.0, "pLns": -1.0, "one": 1.0}, {"linestart": 1.0})
    l3.units.append(pr.relu_unit({"x_pm": 1.0, "p_exp": 1.0, "one": -1.0}, {"pexp_pm": 1.0},
                                 "'+' or '-' right after e E p P (an exponent sign inside a number)"))

    # ---- Layer 4. MLP: the rules inside a token; and "outside any token" (norm).
    l4 = b.at(4)
    l4.units.append(pr.relu_unit({"x_ident": 1.0, "x_dot": 1.0, "pexp_pm": 1.0}, {"numcont": 1.0},
                                 "a pp-number continues: an identifier byte, '.', or an exponent sign"))
    # norm: the previous letter ended a token, or a number / word just ended at this byte
    l4.units += [
        pr.relu_unit({**{st(c): 1.0 for c in NORMAL_STATES}, "c_mark": 1.0}, {"norm": 1.0},
                     "outside any token: the previous letter ended one (or this is the first source byte)"),
        pr.relu_unit({st("n"): 1.0, "x_ident": -1.0, "x_dot": -1.0, "pexp_pm": -1.0}, {"norm": 1.0},
                     "outside any token: a number ends here"),
        pr.relu_unit({st("i"): 1.0, st("k"): 1.0, "x_ident": -1.0}, {"norm": 1.0}, "outside any token: a word ends here"),
        pr.relu_unit({"x_mark": 1.0}, {"r_end": 1.0}, "the source is used up: write the final newline"),
    ]
    for r in RULES:
        if r.name == "num":  # needs numcont from this layer: placed in layer 5
            r.layer = 5
        b.at(r.layer).units.append(_rule_unit(r))

    # ---- Unembedding: each rule's dim votes for its letter; the end beats any letter.
    for r in RULES:
        b.logit(f"r_{r.name}", ord(r.out), 10.0)
    b.logit("r_end", EOS, 20.0)
    return b


def build() -> Transformer:
    m = builder().build()
    log.info("C highlighter: %d layers, d_model %d, %d heads, %d units, %d nonzero weights", len(m.layers),
             len(m.layout), sum(len(l.heads) for l in m.layers), sum(l.w_in.shape[1] for l in m.layers),
             m.nonzero_params())
    return m


ALLOWED = set(string.printable) - {"\x0b", "\x0c"} | {"\v", "\f"}
MAX_SOURCE = 6000
CHARS_HINT = "(printable ASCII, tabs and newlines)"
PATTERN = r"^[\s\S]+$"
PATTERN_ERROR = "empty source"


def check(text: str) -> str:
    """The source unchanged, or ValueError (characters and size only)."""
    bad = sorted(set(text) - ALLOWED)
    if bad:
        raise ValueError(f"unsupported characters: {''.join(bad)!r} {CHARS_HINT}")
    if not re.match(PATTERN, text):
        raise ValueError(PATTERN_ERROR)
    if len(text) > MAX_SOURCE:
        raise ValueError(f"source longer than {MAX_SOURCE} characters")
    return text


def prompt(text: str) -> list[int]:
    return [BOS, *check(text).encode("ascii"), END]


def solve(model: Transformer, text: str) -> str:
    """The state letters the model writes for ``text`` (one per byte)."""
    p = prompt(text)
    out = model.generate(p, stop=EOS, max_new=len(p) + 2)
    return bytes(out[len(p):]).decode("ascii").rstrip("\n")


def _docs() -> dict[str, str]:
    """What each dim means (the visualizer shows these on hover)."""
    d = {c: f"1 if this byte is {' or '.join(repr(ch) for ch in chars) if len(chars) <= 4 else repr(chars[:6]) + '…'}."
         for c, chars in CLASSES.items()}
    d.update({
        "c_start": "1 if this byte can start an identifier (a letter or '_').",
        "c_ident": "1 if this byte can continue an identifier (a letter, digit or '_').",
        "c_space": "1 if this byte is whitespace.", "c_digit": "1 if this byte is a digit.",
        "c_mark": "1 on the end marker (\\x03) that ends the source in the prompt.",
        "f_E": "Tag for the end marker, so a head can find it.", "f_nl": "Tag for newlines.",
        "f_ns": "Tag for non-space bytes.", "E": "Position of the end marker: source byte s is classified at position E + s - 1.",
        "Lnl": "Position of the latest newline at or before this byte.", "Lns": "Position of the latest non-space byte.",
        "p_star": "1 if the byte before the one being classified is '*'.",
        "p_bs": "1 if the byte before it is a backslash (it splices a newline).",
        "p_exp": "1 if the byte before it is e E p or P (an exponent letter).",
        "pLnl": "The latest newline before the byte being classified.", "pLns": "The latest non-space byte before it.",
        "n_star": "1 if the byte after the one being classified is '*'.", "n_slash": "1 if the byte after it is '/'.",
        "n_digit": "1 if the byte after it is a digit.",
        "kw": "1 if the word starting at the byte being classified is a C11 keyword.",
        "linestart": "1 if nothing but whitespace comes before this byte on its line (for '#').",
        "pexp_pm": "1 if this '+' or '-' follows an exponent letter (it stays inside the number).",
        "numcont": "1 if this byte continues a pp-number (identifier byte, '.', or exponent sign).",
        "norm": "1 outside any token: the previous letter ended one, so a new token starts here.",
        "r_end": "Rule end: the source is used up, so write the final newline.",
    })
    for c in sorted(set("".join(KEYWORDS))):
        d[f"ch_{c}"] = f"1 if this byte is the letter {c!r} (used to spell keywords)."
    for letter, what in STATES.items():
        d[st(letter)] = f"1 if this byte is the state letter {letter!r}: {what}."
    for c in XFEATS:
        d[x_(c)] = f"The byte being classified: {d[c][2:]}"
    for j in range(MAX_KW):
        for c in LETTERS_AT[j]:
            d[f"k{j}_{c}"] = f"1 if byte {j} of the word is {c!r} (for matching keywords)."
    for j in range(MAX_KW + 1):
        d[f"id{j}"] = f"1 if byte {j} of the word is an identifier byte (the word goes on)."
    for r in RULES:
        d[f"r_{r.name}"] = f"Rule {r.name}: {r.doc} -> writes {r.out!r} ({STATES[r.out]})."
    return d


DOCS = _docs()


def category(name: str) -> str:
    """How the visualizer groups the dims: byte features, landmarks, lookups, keywords, rules."""
    if name in ("one", "pos", "pos2") or name.startswith(("c_", "ch_", "st_")):
        return "input"
    if name in ("f_E", "f_nl", "f_ns", "E", "Lnl", "Lns"):
        return "landmarks"
    if name.startswith(("x_", "p_", "n_", "pL", "id")) or re.match(r"^k\d+_", name):
        return "lookups"
    if name == "kw":
        return "keywords"
    return "rules"
