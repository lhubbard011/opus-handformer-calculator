"""A C parser written by hand into transformer weights: the lexer, then an LL(1) pushdown
automaton whose stack is text.

Sequence (the host only feeds bytes, takes the argmax, stops at '\\n'):

    \\n<C source>\\x03<one lexer letter per source byte>\\x04<records>\\n

**First it lexes.** The model writes the C highlighter's letters, one per source byte, with the
highlighter's own weights (handformer/csyntax, layers 1-6), then ``LEND`` ('\\x04'). Lexing is
sequential (each letter depends on the one before), so it lives in the text like every running
value here. A token starts at a source byte exactly when the lexer rule that fired there starts
one (an identifier, a keyword, a number, a literal, a punctuator); the rule's dims at that
letter also say which token it is.

**Then it parses.** Each record is one parser action, ``WIDTH`` = 20 bytes at fixed offsets:

    offset  0     action: O open a node, T consume a token, C close a node, ! syntax error
            1     symbol: the node's code (O, C) or the token's code (T); '?' for an error
            3-5   depth after the action: the number of open nodes (3 decimal digits)
            7-10  pointer: the prompt position of the last token consumed (the error's token on '!')
            12-14 item after the action: the parser state (grammar.Item), 3 digits
            16-18 return item (O only): where the parent continues when this node closes; else '...'
            19    '|' ends the record
            2 6 11 15 are spaces

Why records: the parser needs the depth and the input position at every step, and attention
cannot count, so (as in every model here) the state is in the text, at fixed offsets from the
latest '|'. Each step reads the previous record and the next input token and applies one rule:

* one rule per table entry (item, lookahead) -> consume or open; one default per item (close if
  the node may end there, else error); the start, and the halts after the root closes or an error;
* a close needs the parent's item, which is not in the previous record. It is in the record that
  opened the node: the **latest open record whose depth equals the current depth** (deeper opens
  come later but have larger depths; the node's siblings were opened and closed before it). An
  attention head finds it by an exact equality search and reads its return item. That lookup of
  a value computed at another position is the stack.

Every byte of the new record is the same function of those reads, so the record is written byte
by byte: the dim ``out`` holds the ASCII code of the byte to write, and the unembedding writes the
byte nearest to it (logit(b) = K*(2*b*out - b^2)).

Two models in one, so two phases: ``in_lex`` (the letters are being written) gates every lexer
rule, ``after_M`` (the records are) gates every parser rule, every byte the parser proposes and
its unembedding's constant term. Each side writes exactly nothing outside its phase.

    layer 1-6  the lexer (handformer/csyntax/circuit.py), gated by in_lex
    layer 1    MLP   tag for the letters' end M
    layer 2    attn  M          MLP   in_lex, after_M
    layer 3    attn  the classified byte's punctuator character
               MLP   s (the source position a letter classifies); the grammar's keywords
    layer 7    MLP   tags: token starts (the lexer rule that fired), record ends; token classes
    layer 8    attn  Lsep (latest '|' or M), Lts (latest token start, as a source position)
    layer 9    attn  the previous record's fields (13 fixed offsets from Lsep); pm (previous token
                     start, for the forward search)
               MLP   which offset is being written; "first record"
    layer 10   attn  pm^2, depth^2, the digits of depth+1 and depth-1
               MLP   "this '|' ends an open record" (a key for the stack search)
    layer 11   attn  the lookahead: the first token start after the pointer (forward search)
               MLP   the rules (exactly one fires); the gated key features of open records
    layer 12   attn  the stack: the open record matching the depth; the lookahead position's digits
               MLP   the byte to write: the field at this offset, chosen by the rule that fired
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from .. import primitives as pr
from ..csyntax import circuit as lexer
from ..model import Form, HeadSpec, Layout, ModelBuilder, Transformer, UnitSpec, authored_at
from . import grammar as gm
from . import tokens as tk

log = logging.getLogger(__name__)

END = lexer.END  # '\x03' ends the source in the prompt
LEND = 4  # '\x04' ends the letters: the records start after it
BOS = EOS = ord("\n")
LEX = 6  # the lexer's layers; the parser's layers come after them
W = gm.WIDTH
DIGIT_OFFSETS = {"depth": (3, 4, 5), "ptr": (7, 8, 9, 10), "item": (12, 13, 14), "ret": (16, 17, 18)}
FIXED = {2: " ", 6: " ", 11: " ", 15: " ", 19: gm.SEP}
ACTS = {"O": "open", "C": "close", "T": "token", "!": "error"}
OUT_K = 30.0  # unembedding sharpness: the nearest byte code wins by >= 30 logits
# The stack search. A sequence is at most MAX_POS bytes; the depth term (A per unit of squared
# depth difference) must beat any position difference (C per byte), and C*W (the least gap
# between two open records) is the margin among records of equal depth.
MAX_POS = 100_000
STACK_C = 2.0
STACK_A = 2.0 * STACK_C * MAX_POS
TOKNAMES = {"{": "lbrace", "}": "rbrace", "(": "lparen", ")": "rparen", ";": "semi", "+": "plus", gm.END: "end",
            "[": "lbrack", "]": "rbrack", ",": "comma", "*": "star", "-": "minus", "=": "eq"}
# The lexer rules whose letter starts a token (not whitespace, a comment or a directive).
START_RULES = tuple(f"r_{r.name}" for r in lexer.RULES if r.src == "norm" and r.out not in "_BLp")
SRC_POS: Form = {"pos": 1.0, "E": -1.0, "one": 1.0}  # the source byte a letter classifies (lexer's s)


def tokname(t: str) -> str:
    return TOKNAMES.get(t, t)


def term_text(cmap: tk.TokenMap) -> dict[str, str]:
    """How the docs show each terminal: its C text, or what any identifier / number is."""
    return {t: tok.describe() for t, tok in cmap.items()} | {gm.END: "the end of the input"}


@dataclass
class Rule:
    """One parser rule: fires when every term of ``read`` (0/1 dims) holds, i.e. ReLU(read) = 1.
    ``consts``: byte codes it proposes per record offset; ``flags``: which dynamic fields apply."""
    name: str
    doc: str
    read: Form
    consts: dict[int, int] = field(default_factory=dict)
    flags: tuple[str, ...] = ()
    where: str = field(default_factory=authored_at, compare=False)  # the grammar line, or the line here


def _item_form(num: int) -> Form:
    """0/1 terms for 'the previous record's item is num' (3 digits, each one-hot)."""
    return {f"it{k}_{d}": 1.0 for k, d in enumerate(f"{num:03d}")}


def _digits(off: tuple[int, ...], value: int) -> dict[int, int]:
    return {o: ord(c) for o, c in zip(off, f"{value:0{len(off)}d}")}


def rules(table: gm.Table, cmap: tk.TokenMap) -> list[Rule]:
    """The rule table, from the grammar's LL(1) table. Read top to bottom."""
    names, items = table.grammar.names, table.items
    prods = table.grammar.productions
    terms = term_text(cmap)
    out = [Rule("start", f"the first record opens the root ({names[table.grammar.root]})", {"first": 1.0},
                {0: ord("O"), 1: ord(table.grammar.root), **_digits(DIGIT_OFFSETS["depth"], 1),
                 **_digits(DIGIT_OFFSETS["ptr"], 0), **_digits(DIGIT_OFFSETS["item"], table.starts[table.grammar.root]),
                 **_digits(DIGIT_OFFSETS["ret"], gm.DONE)}, where=prods[table.grammar.root].where),
           Rule("halt", "the root has closed (item 000): write the final newline",
                {**_item_form(gm.DONE), "one": -2.0, "first": -1.0, "pa_err": -1.0}, {0: EOS}),
           Rule("halt_error", "after a syntax error: write the final newline",
                {"pa_err": 1.0, "first": -1.0}, {0: EOS})]
    dots = {o: ord(".") for o in DIGIT_OFFSETS["ret"]}
    for it in items[1:]:
        where = prods[it.nt].where
        base = {**_item_form(it.num), "first": -1.0, "pa_err": -1.0}
        here = it.describe(names, terms)
        for e in table.by_item(it.num):
            read = {**base, f"la_{tokname(e.token)}": 1.0, "one": -3.0}
            if e.act == "T":
                out.append(Rule(f"i{it.num}_{tokname(e.token)}", f"in {here}, the token {terms[e.token]} is consumed",
                                read, {0: ord("T"), 1: ord(e.sym), **_digits(DIGIT_OFFSETS["item"], e.target), **dots},
                                ("keepd", "adv"), where))
            else:
                out.append(Rule(f"i{it.num}_{tokname(e.token)}",
                                f"in {here}, {terms[e.token]} starts a {names[e.sym]}: open it (then continue at item {e.ret})",
                                read, {0: ord("O"), 1: ord(e.sym), **_digits(DIGIT_OFFSETS["item"], e.target),
                                       **_digits(DIGIT_OFFSETS["ret"], e.ret or 0)}, ("up", "keepp"), where))
        # the default: no entry matches the lookahead (a 0/1 class, at most one is 1)
        read = {**base, "one": -2.0}
        for e in table.by_item(it.num):
            read[f"la_{tokname(e.token)}"] = read.get(f"la_{tokname(e.token)}", 0.0) - 1.0
        if it.accepting:
            out.append(Rule(f"i{it.num}_close", f"{here}: the {names[it.nt]} is complete, close it "
                            "(the parent's item comes from the record that opened it)",
                            read, {0: ord("C"), 1: ord(it.nt), **dots}, ("down", "keepp", "isclose"), where))
        else:
            expect = ", ".join(terms[e.token] for e in table.by_item(it.num))
            out.append(Rule(f"i{it.num}_error", f"in {here}, anything but {expect} is a syntax error",
                            read, {0: ord("!"), 1: ord("?"), **_digits(DIGIT_OFFSETS["item"], it.num), **dots},
                            ("keepd", "adv"), where))
    return out


def _class(t: str, tok: tk.Tok) -> Form:
    """0/1 at a letter: the token starting at the byte it classifies is terminal ``t``. Read off the
    lexer rule that fired there (and, for a keyword or punctuator, which one it spells)."""
    if tok.kind == "identifier":
        return {"r_ident": 1.0, "r_ucn_start": 1.0}
    if tok.kind == "number":
        return {"r_num_start": 1.0, "r_num_dot": 1.0}
    if tok.kind == "keyword":
        return {"r_keyword": 1.0, f"kw_{tok.text}": 1.0, "one": -1.0}
    n = len(tok.text)
    if n == 1:
        return {"r_punct": 1.0, f"xb_{lexer.PUNCT_NAMES[tok.text]}": 1.0, "one": -1.0}
    return {f"r_punct{n}": 1.0, **{lexer.pq(j, c): 1.0 for j, c in enumerate(tok.text)}, "one": -float(n)}


def _keywords(cmap: tk.TokenMap) -> list[str]:
    return sorted(tok.text for tok in cmap.values() if tok.kind == "keyword")


def _one_byte_puncts(cmap: tk.TokenMap) -> list[str]:
    return sorted(tok.text for tok in cmap.values() if tok.kind == "punct" and len(tok.text) == 1)


def glue_dims(table: gm.Table, cmap: tk.TokenMap) -> list[str]:
    """Between the lexer and the parser: the phases, and the tokens read off the letters."""
    terms = [t for t in table.terminals if t != gm.END]
    return ["c_lend", "f_M", "M", "in_lex", "after_M", "sp", *(f"kw_{w}" for w in _keywords(cmap)),
            *(f"xb_{lexer.PUNCT_NAMES[c]}" for c in _one_byte_puncts(cmap)), "f_ts", "Lts",
            *(f"t_{tokname(t)}" for t in terms)]


def dims(table: gm.Table, rs: list[Rule]) -> list[str]:
    """The parser's own dims (records, lookahead, stack, rules, output)."""
    return [
        "c_sep", "val", *(f"g{d}" for d in range(10)), *(f"b_{a}" for a in "OCTX"),
        "f_sep", "Lsep",
        *(f"pa_{a}" for a in ("open", "close", "tok", "err")), "pd2", "pd1", "pd0", "pp3", "pp2", "pp1", "pp0",
        *(f"it{k}_{d}" for k in range(3) for d in range(10)), "pr2", "pr1", "pr0", "pm",
        *(f"of{j}" for j in range(W)), "first",
        "pm2", "dsq", "dp_q2", "dp_q1", "dm_q2", "dm_q1", "okey",
        "la_pos", *(f"la_{tokname(t)}" for t in table.terminals), "kd", "kd2", "kp",
        *(f"r_{r.name}" for r in rs), *(f"w{j}" for j in range(W) if j not in FIXED), "keepd", "up", "down", "keepp", "adv", "isclose",
        "mr2", "mr1", "mr0", "mpos", "la_q3", "la_q2", "la_q1", "out",
    ]


def _floor_div(name: str, x: Form, s: int, dst: str) -> HeadSpec:
    """dst = floor(x / s) (primitives.floor_div_head with a constant divisor: s^2 and s*xc are linear)."""
    xc = pr.add(x, pr.const(-(s - 1) / 2.0))
    return pr.floor_div_head(name, pr.const(float(s * s)), pr.scale(xc, float(s)), dst)


def _gate(value: Form, gates: tuple[str, ...], write: Form, label: str) -> UnitSpec:
    """``write * value`` when every 0/1 dim in ``gates`` is 1, else 0 (value >= 0, < BIG)."""
    read = dict(value)
    for g in gates:
        read[g] = read.get(g, 0.0) + pr.BIG
    read["one"] = read.get("one", 0.0) - pr.BIG * len(gates)
    return UnitSpec(read=read, write=write, label=label)


CATEGORIES = ("input", "lexer", "tokens", "landmarks", "record", "lookahead", "stack", "rules", "output")


def builder(table: gm.Table, cmap: tk.TokenMap) -> ModelBuilder:
    tk.check_map(table, cmap)
    rs = rules(table, cmap)
    lay = Layout()
    # grouped by circuit (category below), so each group is one block of rows in the visualizer
    names = ["one", "pos", "pos2", *lexer.DIMS, *glue_dims(table, cmap), *dims(table, rs)]
    lay.add(*sorted(names, key=lambda n: CATEGORIES.index(category(n))))
    b = ModelBuilder(lay)
    terminals = [t for t in table.terminals if t != gm.END]

    # ---- The lexer: layers 1-6, the highlighter's own weights, writing only while in_lex.
    lexer.add_lexer(b, gate="in_lex", end=LEND)

    # ---- Byte embedding (added to the lexer's): the letters' end; '|'; a digit's value and
    # one-hot; the action letters (for reading records).
    for byte in range(256):
        ch = chr(byte)
        form: Form = {}
        if byte == LEND:
            form["c_lend"] = 1.0
        if ch == gm.SEP:
            form["c_sep"] = 1.0
        if ch in "0123456789":
            form["val"] = float(int(ch))
            form[f"g{ch}"] = 1.0
        if ch in "OCT!":
            form["b_" + ("X" if ch == "!" else ch)] = 1.0
        if form:
            b.embed_byte(byte, form)

    # ---- Layers 1-3, beside the lexer: the phases and what the tokens need from the lexer's lookups.
    # in_lex: the end marker E is behind (E >= 1) but the letters' end M is not; after_M: it is.
    l1 = b.at(1)
    l1.units += pr.latest_feature({"c_lend": 1.0}, "f_M", bos=False)
    l1.units.append(pr.relu_unit({"one": 1.0, "pos": -1.0}, {"f_M": 1.0},
                                 "fallback for 'f_M': position 0 scores 1, so 'no match yet' reads position 0"))
    l2 = b.at(2)
    l2.heads.append(pr.latest_head("letters' end", "f_M", "M"))
    l2.units += pr.clamp01_units({"E": 1.0}, {"in_lex": 1.0})
    l2.units += pr.clamp01_units({"M": 1.0}, {"in_lex": -1.0, "after_M": 1.0})
    l3 = b.at(3)
    puncts = _one_byte_puncts(cmap)
    if puncts:
        h = pr.gather_head("the classified byte: which of the grammar's punctuators", SRC_POS,
                           f"xb_{lexer.PUNCT_NAMES[puncts[0]]}", src=lexer.pc(puncts[0]))
        h.value = {lexer.pc(c): {f"xb_{lexer.PUNCT_NAMES[c]}": 1.0} for c in puncts}
        l3.heads.append(h)
    l3.units.append(_gate(SRC_POS, ("in_lex",), {"sp": 1.0},
                          "s: the source position this letter classifies (0 outside the letters)"))
    for w in _keywords(cmap):  # as the lexer's keyword units, one per keyword the grammar uses
        read: Form = {f"k{j}_{c}": 1.0 for j, c in enumerate(w)}
        read[f"id{len(w)}"] = -1.0
        read["one"] = -float(len(w) - 1)
        l3.units.append(UnitSpec(read=read, write={f"kw_{w}": 1.0},
                                 label=f"keyword '{w}': the word here spells {w} and ends after {len(w)} bytes"))

    # ---- Layer 7. MLP: tags for record ends and token starts; each token's class.
    # The letters' end counts as a record end, so before the first record Lsep is M: "the
    # previous record" then ends at M, and "first" (Lsep == M) starts the parse. A token starts at
    # a letter whose lexer rule starts one (the end rule: the end of the input, $).
    l7 = b.at(LEX + 1)
    tstart: Form = {d: 1.0 for d in (*START_RULES, "r_end")}
    for dst, match in (("f_sep", {"c_sep": 1.0, "c_lend": 1.0}), ("f_ts", tstart)):
        l7.units += pr.latest_feature(match, dst, bos=False)
        l7.units.append(pr.relu_unit({"one": 1.0, "pos": -1.0}, {dst: 1.0},
                                     f"fallback for '{dst}': position 0 scores 1, so 'no match yet' reads position 0"))
    for t in terminals:
        l7.units.append(pr.relu_unit(_class(t, cmap[t]), {f"t_{tokname(t)}": 1.0},
                                     f"token class: the token starting here is {cmap[t].describe()} (code {t!r})"))

    # ---- Layer 8. Heads: the latest record end; the latest token start, as a source position.
    l8 = b.at(LEX + 2)
    lts = pr.latest_head("latest token start", "f_ts", "Lts")
    lts.value = {"sp": {"Lts": 1.0}}  # its source position (0: none yet)
    l8.heads += [pr.latest_head("latest record end", "f_sep", "Lsep"), lts]

    # ---- Layer 9. Heads: the previous record's fields, at fixed offsets from its start Lsep-W+1.
    l9 = b.at(LEX + 3)
    start = {"Lsep": 1.0, "one": 1.0 - W}

    def _field(name: str, off: int, value: dict[str, Form]) -> HeadSpec:
        h = pr.gather_head(name, pr.add(start, pr.const(float(off))), "pd2", src="val")
        h.value = value
        return h

    l9.heads.append(_field("previous record: action", 0, {"b_O": {"pa_open": 1.0}, "b_C": {"pa_close": 1.0},
                                                              "b_T": {"pa_tok": 1.0}, "b_X": {"pa_err": 1.0}}))
    for k, off in zip((2, 1, 0), DIGIT_OFFSETS["depth"]):
        l9.heads.append(_field(f"previous record: depth digit {2 - k}", off, {"val": {f"pd{k}": 1.0}}))
    for k, off in zip((3, 2, 1, 0), DIGIT_OFFSETS["ptr"]):
        l9.heads.append(_field(f"previous record: pointer digit {3 - k}", off, {"val": {f"pp{k}": 1.0}}))
    for k, off in enumerate(DIGIT_OFFSETS["item"]):  # one-hot: rules match items digit by digit
        l9.heads.append(_field(f"previous record: item digit {k}", off,
                               {f"g{d}": {f"it{k}_{d}": 1.0} for d in range(10)}))
    for k, off in zip((2, 1, 0), DIGIT_OFFSETS["ret"]):
        l9.heads.append(_field(f"previous record: return digit {2 - k}", off, {"val": {f"pr{k}": 1.0}}))
    # pm: the latest token start strictly before this position (Lts one byte back). With pm at
    # every key, "the first token start after p" is the token start whose pm is p.
    l9.heads.append(pr.gather_head("previous token start", {"pos": 1.0, "one": -1.0}, "pm", src="Lts"))
    # MLP: which offset of the new record is being written (o = pos - Lsep: 0 right after a '|').
    o = {"pos": 1.0, "Lsep": -1.0}
    for j in range(W):
        l9.units += pr.hat_units(o, float(j), {f"of{j}": 1.0})
    l9.units += pr.hat_units({"Lsep": 1.0, "M": -1.0}, 0.0, {"first": 1.0})

    depth: Form = {"pd2": 100.0, "pd1": 10.0, "pd0": 1.0}
    ptr: Form = {"pp3": 1000.0, "pp2": 100.0, "pp1": 10.0, "pp0": 1.0}

    # ---- Layer 10. Heads: squares (the pos^2 table, read at pm and at the depth) and the digits of
    # depth +- 1 (floor division by 10 and 100).
    l10 = b.at(LEX + 4)
    l10.heads += [pr.gather_head("pm squared", {"pm": 1.0}, "pm2", src="pos2"),
                  pr.gather_head("depth squared", depth, "dsq", src="pos2"),
                  _floor_div("(depth+1) // 100", pr.add(depth, pr.const(1.0)), 100, "dp_q2"),
                  _floor_div("(depth+1) // 10", pr.add(depth, pr.const(1.0)), 10, "dp_q1"),
                  _floor_div("(depth-1) // 100", pr.add(depth, pr.const(-1.0)), 100, "dm_q2"),
                  _floor_div("(depth-1) // 10", pr.add(depth, pr.const(-1.0)), 10, "dm_q1")]
    for h in l10.heads[4:]:
        h.exact_when = {"down": 1.0}  # depth - 1 is read only on a close (depth >= 1 there)
    # MLP: a '|' in the records that ends an open record is a key of the stack search.
    l10.units.append(pr.relu_unit({"c_sep": 1.0, "pa_open": 1.0, "after_M": 1.0, "one": -2.0}, {"okey": 1.0},
                                  "stack key: this '|' ends an open record (its depth and return item are known here)"))

    # ---- Layer 11. Head: the lookahead, the first token start after the pointer (next_match_head:
    # the token start k whose previous token start pm_k is the pointer). Copies its source
    # position and class.
    l11 = b.at(LEX + 5)
    la = pr.next_match_head("lookahead token", "pm", "pm2", tstart, ptr, "la_pos")
    la.value = {"sp": {"la_pos": 1.0}, **{f"t_{tokname(t)}": {f"la_{tokname(t)}": 1.0} for t in terminals},
                "r_end": {f"la_{tokname(gm.END)}": 1.0}}
    l11.heads.append(la)
    # MLP: the rules. Each is an AND of exact 0/1 terms (and after_M); it writes its own dim, the
    # byte codes it proposes for constant fields (w<offset>), and flags for the dynamic fields.
    for r in rs:
        write: Form = {f"r_{r.name}": 1.0, **{f"w{o}": float(c) for o, c in r.consts.items()},
                       **{f: 1.0 for f in r.flags}}
        l11.units.append(UnitSpec(read=pr.gated(r.read, {"after_M": 1.0}, 1.0), write=write, where=r.where,
                                  label=f"rule {r.name}: {r.doc}"))
    # The stack keys: depth, depth^2 and position of each open record, 0 elsewhere.
    l11.units += [_gate(depth, ("okey",), {"kd": 1.0}, "stack key: the open record's depth (0 if not a key)"),
                  _gate({"dsq": 1.0}, ("okey",), {"kd2": 1.0}, "stack key: its depth squared"),
                  _gate({"pos": 1.0}, ("okey",), {"kp": 1.0}, "stack key: its position (the latest wins a tie)")]

    # ---- Layer 12. Heads: the stack search and the lookahead position's digits.
    #   score(k) = A*(2*d*kd - kd2) + C*kp = A*(d^2 - (kd - d)^2) + C*kp on open records, 0 elsewhere:
    # the open records at depth d lead every other open by >= A - C*MAX_POS, every non-open key by
    # >= A (d >= 1), and among them the latest leads by >= C*W.
    l12 = b.at(LEX + 6)
    stack = HeadSpec(name="stack: the open record at this depth",
                     query=[pr.scale(depth, 2 * STACK_A), pr.const(-STACK_A), pr.const(STACK_C)],
                     key=[{"kd": 1.0}, {"kd2": 1.0}, {"kp": 1.0}],
                     # its return item, and where it is (shown by the visualizer: the record the close pops)
                     value={"pr2": {"mr2": 1.0}, "pr1": {"mr1": 1.0}, "pr0": {"mr0": 1.0}, "pos": {"mpos": 1.0}},
                     exact_when={"isclose": 1.0})  # read only on a close (an open node at this depth exists)
    l12.heads += [stack, _floor_div("lookahead // 1000", {"la_pos": 1.0}, 1000, "la_q3"),
                  _floor_div("lookahead // 100", {"la_pos": 1.0}, 100, "la_q2"),
                  _floor_div("lookahead // 10", {"la_pos": 1.0}, 10, "la_q1")]
    # MLP: the byte at this offset. Constant fields come from the rule (w<j>), the rest are digits
    # chosen by the rule's flags: '0' + digit, gated by "this offset", the flag and after_M.
    zero = float(ord("0"))
    for j in range(W):
        if j in FIXED:
            l12.units.append(_gate({f"of{j}": 1.0}, ("after_M",), {"out": float(ord(FIXED[j]))},
                                   f"offset {j} is always {FIXED[j]!r}"))
        else:
            l12.units.append(_gate({f"w{j}": 1.0}, (f"of{j}", "after_M"), {"out": 1.0},
                                   f"offset {j}: the byte the rule proposes (if any)"))
    dplus = [{"dp_q2": 1.0}, {"dp_q1": 1.0, "dp_q2": -10.0}, pr.add(depth, {"one": 1.0, "dp_q1": -10.0})]
    dminus = [{"dm_q2": 1.0}, {"dm_q1": 1.0, "dm_q2": -10.0}, pr.add(depth, {"one": -1.0, "dm_q1": -10.0})]
    for j, same, up, down in zip(DIGIT_OFFSETS["depth"], ({"pd2": 1.0}, {"pd1": 1.0}, {"pd0": 1.0}), dplus, dminus):
        for flag, digit, what in (("keepd", same, "unchanged"), ("up", up, "plus one"), ("down", down, "minus one")):
            l12.units.append(_gate(pr.add(digit, pr.const(zero)), (f"of{j}", flag, "after_M"), {"out": 1.0},
                                   f"offset {j}: depth digit, {what}"))
    lad = [{"la_q3": 1.0}, {"la_q2": 1.0, "la_q3": -10.0}, {"la_q1": 1.0, "la_q2": -10.0},
           {"la_pos": 1.0, "la_q1": -10.0}]
    for j, same, nxt in zip(DIGIT_OFFSETS["ptr"], ({"pp3": 1.0}, {"pp2": 1.0}, {"pp1": 1.0}, {"pp0": 1.0}), lad):
        l12.units += [_gate(pr.add(same, pr.const(zero)), (f"of{j}", "keepp", "after_M"), {"out": 1.0},
                            f"offset {j}: pointer digit, unchanged"),
                      _gate(pr.add(nxt, pr.const(zero)), (f"of{j}", "adv", "after_M"), {"out": 1.0},
                            f"offset {j}: pointer digit of the lookahead token (consumed, or the error's token)")]
    for j, k in zip(DIGIT_OFFSETS["item"], (2, 1, 0)):
        l12.units.append(_gate({f"mr{k}": 1.0, "one": zero}, (f"of{j}", "isclose", "after_M"), {"out": 1.0},
                               f"offset {j}: on a close, the item is the return item of the matching open record"))

    # ---- Unembedding: logit(b) = K*(2*b*out - b^2), the byte whose code is nearest to out. The
    # constant term is on after_M, not on 'one': while the letters are written it would tilt the
    # lexer's votes toward small byte codes.
    for byte in sorted(OUT_BYTES(table)):
        b.logit("out", byte, 2.0 * OUT_K * byte)
        b.logit("after_M", byte, -OUT_K * byte * byte)
    return b


def OUT_BYTES(table: gm.Table) -> set[int]:  # noqa: N802  the bytes a record can hold
    syms = set(table.grammar.productions) | {t for t in table.terminals if t != gm.END} | {"$", "?"}
    return {ord(c) for c in "0123456789 .|\nOCT!"} | {ord(c) for c in syms}


def build(table: gm.Table, cmap: tk.TokenMap) -> Transformer:
    m = builder(table, cmap).build()
    log.info("C parser (%s): %d layers, d_model %d, %d heads, %d units, %d nonzero weights",
             table.grammar.root, len(m.layers), len(m.layout), sum(len(l.heads) for l in m.layers),
             sum(l.w_in.shape[1] for l in m.layers), m.nonzero_params())
    return m


MAX_SOURCE = 300  # about 7 records (140 bytes) per token, plus one letter per byte


def check(text: str) -> str:
    """The source unchanged, or ValueError (characters and size only: syntax is the model's job)."""
    bad = sorted(set(text) - lexer.ALLOWED)
    if bad:
        raise ValueError(f"unsupported characters: {''.join(bad)!r} {lexer.CHARS_HINT}")
    if not re.search(r"\S", text):
        raise ValueError("empty source")
    if len(text) > MAX_SOURCE:
        raise ValueError(f"source longer than {MAX_SOURCE} characters")
    return text


def prompt(text: str) -> list[int]:
    return [BOS, *check(text).encode("ascii"), END]


def solve(model: Transformer, text: str) -> str:
    """Everything the model writes for ``text``: the letters, '\\x04', the records, the final newline."""
    p = prompt(text)
    out = model.generate(p, stop=EOS, max_new=MAX_POS - len(p))
    log.debug("parsed %d source bytes: %d bytes written", len(text), len(out) - len(p))
    return bytes(out[len(p):]).decode("ascii")


def split(written: str) -> tuple[str, str]:
    """(letters, records) of what ``solve`` returns."""
    letters, _, records = written.partition(chr(LEND))
    return letters, records


def docs(table: gm.Table, cmap: tk.TokenMap) -> dict[str, str]:
    """What each dim means (the visualizer shows these on hover)."""
    d = dict(lexer.DOCS)
    d.update({
        "c_lend": "1 on '\\x04', the byte the model writes after the last letter: the records start after it.",
        "f_M": "Tag for the letters' end, so a head can find it.",
        "M": "Position of the letters' end ('\\x04'): the records start after it (0: not written yet).",
        "in_lex": "1 while the letters are written (after the end marker, before the letters' end): the lexer's gate.",
        "after_M": "1 while the records are written: the parser's gate.",
        "sp": "At a letter: the source position of the byte it classifies (0 outside the letters).",
        "f_ts": "Tag for token starts: the letters whose lexer rule starts a token (and the end of the input).",
        "Lts": "The source position of the latest token start at or before this byte (0: none).",
        "c_sep": "1 on '|', the byte that ends a record.",
        "val": "The value of a digit byte (records hold depth, pointer and items in decimal).",
        "f_sep": "Tag for record ends ('|', and the letters' end).",
        "Lsep": "Position of the latest record end: the previous record is the 20 bytes ending there.",
        "pm": "The latest token start strictly before this byte, as a source position (for the forward search).",
        "pm2": "pm squared (the forward search's score is quadratic in pm).",
        "pd2": "Previous record: depth, hundreds digit.", "pd1": "Previous record: depth, tens digit.",
        "pd0": "Previous record: depth, units digit.",
        "pr2": "Previous record: return item, hundreds digit.", "pr1": "Previous record: return item, tens digit.",
        "pr0": "Previous record: return item, units digit.",
        "first": "1 while writing the first record (no record yet: the previous 'record end' is the letters' end).",
        "dsq": "The previous record's depth, squared (for the stack search's key).",
        "okey": "1 on a '|' that ends an open record: a key of the stack search.",
        "kd": "Stack key: the depth of the open record ending here (0 elsewhere).",
        "kd2": "Stack key: its depth squared.",
        "kp": "Stack key: its position (the latest of equal depth wins).",
        "mpos": "Where the matching open record ends: the record a close pops (the stack search's answer).",
        "la_pos": "The lookahead: source position of the first token start after the pointer.",
        "keepd": "The depth stays (a token or an error).", "up": "The depth goes up by one (an open).",
        "down": "The depth goes down by one (a close).", "keepp": "The pointer stays (an open or a close).",
        "adv": "The pointer moves to the lookahead (a token is consumed, or an error points at it).",
        "isclose": "A close: the new item is the return item of the matching open record.",
        "out": "The ASCII code of the byte to write; the unembedding writes the nearest byte.",
    })
    for w in _keywords(cmap):
        d[f"kw_{w}"] = f"1 if the word starting at the classified byte is the keyword {w!r}."
    for c in _one_byte_puncts(cmap):
        d[f"xb_{lexer.PUNCT_NAMES[c]}"] = f"1 if the classified byte is {c!r}."
    for a, what in (("open", "an open (O)"), ("close", "a close (C)"), ("tok", "a token (T)"), ("err", "an error (!)")):
        d[f"pa_{a}"] = f"1 if the previous record is {what}."
    for a in "OCTX":
        d[f"b_{a}"] = f"1 if this byte is the action letter {'!' if a == 'X' else a}."
    for k in range(10):
        d[f"g{k}"] = f"1 if this byte is the digit {k} (items are matched digit by digit)."
    for k, place in enumerate(("hundreds", "tens", "units")):
        for v in range(10):
            d[f"it{k}_{v}"] = f"1 if the previous record's item has {v} in the {place} place."
    for k, place in ((3, "thousands"), (2, "hundreds"), (1, "tens"), (0, "units")):
        d[f"pp{k}"] = f"Previous record: pointer, {place} digit."
    for q, s in (("dp_q2", "(depth+1) // 100"), ("dp_q1", "(depth+1) // 10"), ("dm_q2", "(depth-1) // 100"),
                 ("dm_q1", "(depth-1) // 10"), ("la_q3", "lookahead // 1000"), ("la_q2", "lookahead // 100"),
                 ("la_q1", "lookahead // 10")):
        d[q] = f"{s}, for writing its digits (floor division by attention)."
    for k in range(3):
        d[f"mr{2 - k}"] = f"The matching open record's return item, digit {k} (the item after a close)."
    terms = term_text(cmap)
    for t in table.terminals:
        if t != gm.END:
            d[f"t_{tokname(t)}"] = f"1 at a letter if the token starting at its byte is {terms[t]} (code {t!r})."
        d[f"la_{tokname(t)}"] = f"1 if the lookahead token is {terms[t]} (code {t!r})."
    for j in range(W):
        d[f"of{j}"] = f"1 when the byte being written is at offset {j} of its record."
        if j not in FIXED:
            d[f"w{j}"] = f"The byte code the rule that fired proposes for offset {j} (0: not a constant field)."
    for r in rules(table, cmap):
        d[f"r_{r.name}"] = f"Rule {r.name}: {r.doc}."
    return d


LEXER_DIMS = frozenset(("one", "pos", "pos2", *lexer.DIMS))
GLUE = frozenset(("c_lend", "f_M", "M", "in_lex", "after_M", "sp", "f_ts", "Lts"))


def category(name: str) -> str:
    """How the visualizer groups the dims: input, the lexer, tokens, landmarks, record fields,
    lookahead, stack, rules, output."""
    if name in LEXER_DIMS:
        return "input" if lexer.category(name) == "input" else "lexer"
    if name in ("c_lend", "c_sep", "val") or re.match(r"^(g\d|b_)", name):
        return "input"
    if name in GLUE or name.startswith(("kw_", "xb_", "t_")):
        return "tokens"
    if name in ("f_sep", "Lsep", "first") or name.startswith("of"):
        return "landmarks"
    if re.match(r"^(pa_|pd\d|pp\d|it\d_|pr\d)", name):
        return "record"
    if name in ("pm", "pm2", "la_pos") or name.startswith("la_"):
        return "lookahead"
    if name in ("dsq", "okey", "kd", "kd2", "kp", "mpos") or name.startswith("mr"):
        return "stack"
    if name.startswith(("r_", "w")) or name in ("keepd", "up", "down", "keepp", "adv", "isclose"):
        return "rules"
    return "output"
