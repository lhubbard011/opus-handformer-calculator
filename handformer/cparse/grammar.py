"""Grammars as data, compiled to the LL(1) table the parser's weights implement.

A grammar is a set of nonterminals, each one byte (its node code in the parse records), with a
body written in EBNF: ``seq``, ``alt``, ``star`` (zero or more), ``opt`` (zero or one). A string
that names a nonterminal is that nonterminal; any other one-byte string is a terminal (an input
token), and ``END`` is the end of the input.

``Grammar.compile`` turns every body into a small automaton (a DFA over terminals and
nonterminals). Its states are the parser's **items**: "in an S, having read ``i ( E )``". From
each item the LL(1) table says, for each lookahead token, one action:

* ``T``: consume the token (an edge on that terminal);
* ``O``: open a child node (an edge on a nonterminal whose FIRST set holds the token); the
  record carries the **return item** the parent continues from when the child closes;
* the item's default when no entry matches: ``C`` (close the node) if the item may end it,
  else ``!`` (syntax error at that token).

Closing by default ("default reductions") delays an error to an ancestor, but never past the
offending token: closes do not consume, so the error is still reported at the same token.
A conflict (two entries for one item and token) is a grammar error, except that a token edge is
never in conflict with closing: the entry wins over the default, which is how C's dangling
``else`` binds to the nearest ``if``.

``simulate`` runs the table with an explicit stack and writes the records the model must write
(see circuit.py for the format). The model has no stack: on a close it finds the matching open
record by its depth.
"""
from __future__ import annotations

import logging
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

END = "$"  # the end marker as a terminal (the byte '\x03' in the prompt)
DONE = 0  # the item a finished parse returns to: the root's return item
SPACE = " \t\n\r"


class GrammarError(ValueError):
    pass


# ---------------------------------------------------------------- EBNF bodies
# A body is a terminal/nonterminal string or a tuple (op, *parts).

def seq(*parts):
    return ("seq", *parts)


def alt(*parts):
    return ("alt", *parts)


def star(*parts):
    return ("star", seq(*parts))


def opt(*parts):
    return ("opt", seq(*parts))


def _source_line() -> str:
    """``path:line`` of the grammar line that defined a production (docs only, like ``where``)."""
    f = sys._getframe(2)
    p = Path(f.f_code.co_filename).resolve()
    root = Path(__file__).resolve().parents[2]
    return f"{p.relative_to(root).as_posix()}:{f.f_lineno}" if root in p.parents else ""


@dataclass
class Production:
    nt: str
    name: str
    body: object
    doc: str
    where: str


@dataclass
class Item:
    """One parser state: a state of one nonterminal's automaton."""
    num: int
    nt: str
    path: tuple[str, ...]  # a shortest sequence of symbols read from the node's start to here
    accepting: bool  # the node may close here
    edges: dict[str, int] = field(default_factory=dict)  # symbol -> item

    def describe(self, names: dict[str, str]) -> str:
        read = " ".join(names.get(s, repr(s)) if len(s) == 1 and s in names else repr(s) for s in self.path)
        return f"{names[self.nt]}, after {read}" if self.path else f"{names[self.nt]}, at its start"


@dataclass
class Entry:
    """One LL(1) table entry: at ``item``, lookahead ``token`` -> action."""
    item: int
    token: str
    act: str  # 'T' consume, 'O' open
    sym: str  # the token (T) or the child nonterminal (O)
    target: int  # the item after the action (T), or the child's start item (O)
    ret: int | None  # O only: the parent's item after the child


@dataclass
class Table:
    grammar: "Grammar"
    items: list[Item]  # items[0] is DONE
    entries: list[Entry]
    starts: dict[str, int]
    first: dict[str, frozenset[str]]

    @property
    def terminals(self) -> list[str]:
        return sorted({e.token for e in self.entries} | {END})

    def by_item(self, num: int) -> list[Entry]:
        return [e for e in self.entries if e.item == num]


class Grammar:
    def __init__(self, root: str, names: dict[str, str]) -> None:
        """``names``: nonterminal byte -> readable name (one byte each, e.g. 'S' -> 'statement')."""
        self.root = root
        self.names = dict(names)
        self.productions: dict[str, Production] = {}

    def rule(self, nt: str, body, doc: str) -> None:
        if nt in self.productions:
            raise GrammarError(f"nonterminal {nt!r} defined twice")
        if nt not in self.names:
            raise GrammarError(f"nonterminal {nt!r} has no name")
        self.productions[nt] = Production(nt, self.names[nt], body, doc, _source_line())

    def is_nt(self, s: str) -> bool:
        return s in self.productions

    # ------------------------------------------------------------ automata
    def _nfa(self, body, edges: list, new) -> tuple[int, int]:
        """Thompson construction: (start, final) of a fragment; edges (src, symbol or None, dst)."""
        if isinstance(body, str):
            if len(body) != 1:
                raise GrammarError(f"symbol {body!r}: terminals and nonterminals are one byte")
            a, b = new(), new()
            edges.append((a, body, b))
            return a, b
        op, *parts = body
        if op == "seq":
            a = b = new()
            for p in parts:
                s, f = self._nfa(p, edges, new)
                edges.append((b, None, s))
                b = f
            return a, b
        if op == "alt":
            a, b = new(), new()
            for p in parts:
                s, f = self._nfa(p, edges, new)
                edges += [(a, None, s), (f, None, b)]
            return a, b
        if op in ("star", "opt"):
            a, b = new(), new()
            s, f = self._nfa(parts[0], edges, new)
            edges += [(a, None, s), (f, None, b), (a, None, b)]
            if op == "star":
                edges.append((f, None, s))
            return a, b
        raise GrammarError(f"unknown operator {op!r}")

    def _dfa(self, nt: str) -> tuple[list[frozenset[int]], dict[tuple[int, str], int], set[int]]:
        """Subset construction: states (sets of NFA states), transitions, accepting states."""
        edges: list = []
        counter = iter(range(10**9))
        start, final = self._nfa(self.productions[nt].body, edges, lambda: next(counter))
        eps: dict[int, list[int]] = {}
        moves: dict[int, list[tuple[str, int]]] = {}
        for a, s, b in edges:
            (eps.setdefault(a, []).append(b) if s is None else moves.setdefault(a, []).append((s, b)))

        def closure(states):
            out, todo = set(states), list(states)
            while todo:
                for b in eps.get(todo.pop(), []):
                    if b not in out:
                        out.add(b)
                        todo.append(b)
            return frozenset(out)

        states = [closure({start})]
        index = {states[0]: 0}
        trans: dict[tuple[int, str], int] = {}
        todo = deque([0])
        while todo:
            i = todo.popleft()
            by_sym: dict[str, set[int]] = {}
            for a in states[i]:
                for s, b in moves.get(a, []):
                    by_sym.setdefault(s, set()).add(b)
            for s in sorted(by_sym):  # sorted: the numbering is deterministic
                c = closure(by_sym[s])
                if c not in index:
                    index[c] = len(states)
                    states.append(c)
                    todo.append(index[c])
                trans[(i, s)] = index[c]
        accepting = {i for i, st in enumerate(states) if final in st}
        return self._minimal(len(states), trans, accepting)

    @staticmethod
    def _minimal(n: int, trans: dict[tuple[int, str], int], accepting: set[int]):
        """Merge equivalent states (Moore's refinement): fewer items, one per distinct situation.

        Example: in ``S* $`` the start and "after a statement" behave the same, so they are one item."""
        block = [int(i in accepting) for i in range(n)]
        syms = sorted({s for _, s in trans})
        while True:
            sig = [(block[i], *(block[trans[(i, s)]] if (i, s) in trans else -1 for s in syms)) for i in range(n)]
            ids: dict[tuple, int] = {}
            for i in range(n):  # numbered in first-appearance order, so state 0 stays 0
                ids.setdefault(sig[i], len(ids))
            new = [ids[sig[i]] for i in range(n)]
            if len(ids) == len(set(block)):
                break
            block = new
        block = new
        m = max(block) + 1
        states = [frozenset(i for i in range(n) if block[i] == b) for b in range(m)]
        mtrans = {(block[a], s): block[b] for (a, s), b in trans.items()}
        return states, mtrans, {block[i] for i in accepting}

    def compile(self) -> Table:
        if self.root not in self.productions:
            raise GrammarError(f"root {self.root!r} has no production")
        items = [Item(DONE, self.root, (), False)]
        starts: dict[str, int] = {}
        order = [self.root] + [nt for nt in self.productions if nt != self.root]
        for nt in order:
            states, trans, accepting = self._dfa(nt)
            base = len(items)
            starts[nt] = base
            # a shortest path to each state, for its description
            paths: dict[int, tuple[str, ...]] = {0: ()}
            todo = deque([0])
            while todo:
                i = todo.popleft()
                for (a, s), b in sorted(trans.items()):
                    if a == i and b not in paths:
                        paths[b] = paths[i] + (s,)
                        todo.append(b)
            for i in range(len(states)):
                items.append(Item(base + i, nt, paths[i], i in accepting))
            for (a, s), b in trans.items():
                items[base + a].edges[s] = base + b
            if 0 in accepting:
                raise GrammarError(f"{nt!r} can be empty: nullable nonterminals are not supported")
        for it in items[1:]:
            for s in it.edges:
                if len(s) == 1 and s.isupper() and s not in self.productions:
                    log.debug("symbol %r in %r treated as a terminal", s, it.nt)

        first: dict[str, frozenset[str]] = {}

        def first_of(nt: str, seen: tuple[str, ...] = ()) -> frozenset[str]:
            if nt in first:
                return first[nt]
            if nt in seen:
                raise GrammarError(f"left recursion through {' -> '.join(seen + (nt,))}")
            out: set[str] = set()
            for s in items[starts[nt]].edges:
                out |= first_of(s, seen + (nt,)) if self.is_nt(s) else {s}
            first[nt] = frozenset(out)
            return first[nt]

        for nt in self.productions:
            first_of(nt)

        entries: list[Entry] = []
        seen: dict[tuple[int, str], Entry] = {}
        for it in items[1:]:
            for s, b in sorted(it.edges.items()):
                if self.is_nt(s):
                    new = [Entry(it.num, t, "O", s, starts[s], b) for t in sorted(first[s])]
                else:
                    new = [Entry(it.num, s, "T", s, b, None)]
                for e in new:
                    if (e.item, e.token) in seen:
                        o = seen[(e.item, e.token)]
                        raise GrammarError(f"LL(1) conflict in {it.describe(self.names)} on {e.token!r}: "
                                           f"{o.act} {o.sym!r} vs {e.act} {e.sym!r}")
                    seen[(e.item, e.token)] = e
                    entries.append(e)
        table = Table(self, items, entries, starts, first)
        log.info("grammar %r: %d nonterminals, %d items, %d table entries, %d terminals",
                 self.root, len(self.productions), len(items), len(entries), len(table.terminals))
        return table


# ---------------------------------------------------------------- the records
WIDTH = 20  # bytes per record, separator included
SEP = "|"


def record(act: str, sym: str, depth: int, ptr: int, item: int, ret: int | None) -> str:
    """``A S ddd pppp iii rrr|``: action, symbol, depth, pointer, item, return item (opens only)."""
    r = f"{ret:03d}" if ret is not None else "..."
    s = f"{act}{sym} {depth:03d} {ptr:04d} {item:03d} {r}{SEP}"
    assert len(s) == WIDTH, s
    return s


def lookahead(source: str, ptr: int) -> tuple[int, str | None]:
    """The first token after prompt position ``ptr``: (its position, its terminal or None)."""
    prompt = "\n" + source + "\x03"
    k = ptr + 1
    while k < len(prompt) - 1 and prompt[k] in SPACE:
        k += 1
    if k >= len(prompt):  # past the end marker: the model sees an output byte, which is no token
        return k, None
    if k == len(prompt) - 1:
        return k, END
    # a source byte spelled like the end terminal is not the end of the input (no token at all)
    return k, (None if prompt[k] == END else prompt[k])


def simulate(table: Table, source: str) -> str:
    """The records the model writes for ``source``, then the final newline. An explicit stack."""
    out = []
    root = table.grammar.root
    item, depth, ptr = table.starts[root], 1, 0
    out.append(record("O", root, depth, ptr, item, DONE))
    stack = [DONE]
    entries = {(e.item, e.token): e for e in table.entries}
    terminals = set(table.terminals)
    while item != DONE:
        k, t = lookahead(source, ptr)
        e = entries.get((item, t)) if t in terminals else None
        if e is not None and e.act == "T":
            ptr, item = k, e.target
            out.append(record("T", e.sym, depth, ptr, item, None))
        elif e is not None:
            depth += 1
            stack.append(e.ret)
            item = e.target
            out.append(record("O", e.sym, depth, ptr, item, e.ret))
        elif table.items[item].accepting:
            nt = table.items[item].nt
            item = stack.pop()
            depth -= 1
            out.append(record("C", nt, depth, ptr, item, None))
        else:
            out.append(record("!", "?", depth, k, item, None))
            break
    return "".join(out) + "\n"


def parse_records(text: str) -> list[tuple]:
    """(act, sym, depth, ptr, item, ret) per record of model output."""
    recs = []
    for chunk in text.rstrip("\n").split(SEP):
        if not chunk:
            continue
        act, sym = chunk[0], chunk[1]
        d, p, i, r = chunk[3:6], chunk[7:11], chunk[12:15], chunk[16:19]
        recs.append((act, sym, int(d), int(p), int(i), None if r == "..." else int(r)))
    return recs


def tree(text: str) -> str:
    """The parse tree in the records as an S-expression, e.g. ``(P (S (E (F x)) ;) $)``.

    Nodes are ``(<node code> children...)``; tokens are bare, except ``(`` ``)`` and ``'`` which
    are quoted (``'('``) so the nesting stays unambiguous. An error record becomes
    ``(error at N)``, N the prompt position of the offending token (1 = first source byte)."""
    out: list[str] = []
    for act, sym, _depth, ptr, _item, _ret in parse_records(text):
        if act == "O":
            out.append(("" if not out or out[-1].endswith("(") else " ") + "(" + sym)
        elif act == "C":
            out.append(")")
        elif act == "T":
            out.append(" " + (f"'{sym}'" if sym in "()'" else sym))
        else:
            out.append(f" (error at {ptr})")
    return "".join(out).lstrip()
