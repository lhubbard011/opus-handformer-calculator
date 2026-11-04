# A C89 parser in hand-written weights: design

Goal: a model that parses ANSI C89 (X3.159-1989, plus K&R function definitions) and writes the
full parse tree, with a real typedef symbol table, everything computed in the weights, every
rule named. Built in stages; this note fixes the architecture. Stage 1 (the engine, on a toy
grammar) is in `handformer/cparse/`.

## Why records, not free-form tree text

A parser needs, at every step, the next input token and the nesting depth. Both are counts over
what has been written so far, and attention cannot count exactly; a running value also cannot be
carried from one position to the next inside the network (each position's values are visible to
later positions only through attention, one layer later). So, as in every model here, the state
lives in the text: the parser writes **fixed-width records**, one per action, and each record
carries the state the next step needs at fixed offsets.

    OS 002 0004 003 014|
    ||  |   |    |   |  '|' ends the record (20 bytes)
    ||  |   |    |   return item (opens only, else '...'): where the parent continues
    ||  |   |    item after this action (the parser state), 3 decimal digits
    ||  |   pointer: prompt position of the last token consumed (on '!', the bad token)
    ||  depth after this action (number of open nodes)
    |symbol: the node's code (O, C) or the token (T)
    action: O open, T token, C close, ! syntax error

Decimal, not a denser base: the records are meant to be read, and the digit arithmetic is done
by attention (floor division by 10, 100, 1000), which costs the same in any base.

## One step

At every byte of a new record the model reads the previous record's fields (exact gathers at
fixed offsets from the latest '|') and the lookahead token, and one rule fires:

- **open / token**: one rule per LL(1) table entry (item, lookahead), an AND unit, exactly as
  the C highlighter's lexer rules;
- **default**: one rule per item for every other lookahead: close the node if the item may end
  it, else a syntax error. Closing by default delays an error to an ancestor but never past the
  offending token (closes consume nothing). A token entry always beats closing, which is how
  the dangling `else` binds to the nearest `if`;
- **close** needs the parent's item, which is not in the previous record. It is in the record
  that opened the node: the latest open record whose depth equals the current depth. An
  attention head finds it by an exact equality search (score A·(d² − (d_k − d)²) + C·pos over
  open records) and reads its return item. That lookup of values computed at another position
  is the stack.

The **lookahead** is a forward search: every position knows the token start before it (pm);
the first token start after the pointer p is the token start whose pm is p
(`primitives.next_match_head`). Whitespace is skipped by construction.

**Output.** The dim `out` holds the ASCII code of the byte to write; the unembedding scores
byte b as K·(2·b·out − b²), so the nearest byte code wins by ≥ K. Each offset's byte is a gated
sum: the rule's constant (action, symbol, item, return), or a digit chosen by the rule's flags
(depth ±1 or kept, pointer kept or advanced to the lookahead, item from the stack on a close).

## Grammar to weights

A grammar is data (`grammar.py`): one-byte nonterminals with EBNF bodies (`seq`, `alt`, `star`,
`opt`). The compiler builds each body's automaton (Thompson construction, subset construction,
Moore minimisation); its states are the items. It computes FIRST sets, builds the table and
rejects LL(1) conflicts, left recursion and nullable nonterminals. `circuit.py` turns any
compiled table into weights: the rules plus the shared machinery (record addressing, digit
arithmetic, the stack search, the forward search). Each rule's `where` is the grammar line of
its production, so the visualizer links a step to the grammar rule that drove it.

Tests (`tests/test_cparse.py`): the table, simulated with a real stack, against an independent
recursive-descent parser (trees and error positions); the weights against the simulator, byte
for byte; the claims (exactly one rule per output byte, exact reads, robust steps).

## Stages

1. The engine on a toy grammar (blocks, if/else, while, expressions): done.
2. Tokens from the C lexer: the model first writes the highlighter's state letters, then parses
   reading token boundaries and classes from them (the text is the memory, again).
3. C89 expressions (15 precedence levels, casts, sizeof, ?:, comma). The tree must stay flat
   (a node per operator, not per precedence level) or the output grows ~30 records per operand.
4. Statements.
5. Declarations and declarators.
6. The typedef symbol table (exact identifier hashes, scope by depth).
7. Function definitions (ANSI and K&R), the translation unit, error messages.

Limits by design: no macro expansion (directives are skipped, a syntax check after
preprocessing); depth ≤ 999, sequence ≤ 100,000 bytes (the stack search's margins assume it);
the output is about 7 records (140 bytes) per token, so inputs are capped at a few hundred
bytes for now.
