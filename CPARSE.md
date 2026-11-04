# A parser, written by hand into a transformer

`handformer/cparse/` is a fourth model built with the same engine and primitives as the others:
6 layers, 29 attention heads, 223 MLP units, 2,799 nonzero weights. It is stage 1 of a C89 syntax
checker (the plan: [docs/PARSER_DESIGN.md](docs/PARSER_DESIGN.md)). It is a **pushdown
automaton**: an LL(1) parser that reads a program and writes its full parse tree. Its grammar is a
C-shaped toy with one-byte tokens, chosen to have what makes C's statements context-free: nested
blocks, statements inside statements, parenthesised expressions, and the dangling `else`.

    P -> S* $        S -> B | i ( E ) S [e S] | w ( E ) S | E ; | ;     (i = if, e = else, w = while)
    B -> { S* }      E -> F (+ F)*        F -> x | 1 | ( E )

The highlighter showed that a regular language (a lexer) is a table of rules. A context-free
language needs one more thing: a **stack**. Here the stack is text too, and popping it is one
attention head doing an exact equality search over computed values.

```
$ python -m handformer.cparse --records "i(x) x; e ;"
  OP 001 0000 001 000        open the program (depth 1) at item 1; it returns to item 000 (done)
  OS 002 0000 003 001        'i' starts a statement: open it, and return to item 1 afterwards
  Ti 002 0001 006 ...        consume 'i' (pointer: source byte 1)
  T( 002 0002 008 ...
  OE 003 0002 018 010        '(' consumed; 'x' starts an expression (return to item 10)
  ...
  CS 002 0007 014 ...        close the inner statement: item 14 comes from the record that opened it
  Te 002 0009 013 ...        item 14 sees 'e': the else binds to the nearest if
  ...
  CP 000 0012 000 ...        close the program: back to item 000, so the model stops
i(x) x; e ;
  (P (S i '(' (E (F x)) ')' (S (E (F x)) ;) e (S ;)) $)
```

**Try it:** pick "parser (stage 1)" in the visualizer's model menu, or
[open it directly](https://lhubbard011.github.io/opus-handformer-calculator/#m=parse). The
**Tree** panel grows the tree as the records are written and outlines the lookahead token; the
step panel names the field being written and the rule that fired, and on a close where the stack
head found the open record (tag `pop` on the sequence).

## How it works

The prompt is the source, then an end marker (`\x03`). The model then writes one **record** per
parser action, 20 bytes at fixed offsets:

    OS 002 0004 003 014|
    action  O open, T token, C close, ! error
    symbol  the node (O, C) or the token (T)
    depth   number of open nodes after the action
    pointer prompt position of the last token consumed (on '!': the offending token)
    item    the parser state after the action (which node, and how far into it)
    return  (opens) the parent's item to continue from when this node closes

Why records: the parser needs the depth and its position in the input at every step. Attention
cannot count, and a value cannot be carried from one position to the next inside the network, so
the state is written down, at offsets the next step can find from the latest `|`
([circuit.py:218](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/circuit.py#L218 "src:dim:parse:Lsep")). Every byte of a record is computed the same way:

1. **Read the previous record**: 13 exact gathers at fixed offsets ([circuit.py:233](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/circuit.py#L233 "src:dim:parse:pd0")). The
   item comes as one-hot digits, so a rule can match it with an AND.
2. **Find the lookahead**: the first token start after the pointer. Every position knows the
   token start before it (`pm`), so the target is the token start whose `pm` is the pointer: a
   forward search by a causal head ([circuit.py:272](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/circuit.py#L272 "src:dim:parse:la_pos"), `primitives.next_match_head`).
   Whitespace is skipped by construction.
3. **Apply one rule** ([toy.py:15](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L15 "src:dim:parse:r_i3_x")). There is one rule per LL(1) table entry
   (item, lookahead) -> consume or open, and one default per item for every other lookahead:
   close the node if the item may end it, else a syntax error. Exactly one fires, checked at every
   output byte. A rule writes its own dim, the byte codes of the record's constant fields, and
   flags for the others (depth +1 / -1 / same, pointer kept / advanced).
4. **Pop the stack on a close.** The parent's item is not in the previous record; it is in the
   record that opened this node, which is **the latest open record whose depth equals the current
   depth** (deeper opens come later but are deeper; siblings at this depth were opened before it).
   One head finds it ([circuit.py:292](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/circuit.py#L292 "src:dim:parse:mr0")): every `|` that ends an open record is a key
   carrying its depth d_k, d_k² and position, and the score A·(d² − (d_k − d)²) + C·pos is exact by
   a margin of at least C·20 = 40 logits.
5. **Write the byte.** The dim `out` holds the ASCII code of the byte to write: the field at this
   offset, picked by gated units ([circuit.py:310](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/circuit.py#L310 "src:dim:parse:out")). Digits of depth ± 1 and of the
   lookahead's position come from floor division by attention (`primitives.floor_div_head`). The
   unembedding scores byte b as 30·(2·b·out − b²), so the nearest code wins by at least 30.

The grammar is data ([handformer/cparse/toy.py](handformer/cparse/toy.py)). The compiler
([handformer/cparse/grammar.py](handformer/cparse/grammar.py)) builds each body's automaton
(Thompson, subset construction, minimisation), whose states are the items; it computes FIRST sets,
builds the table and rejects LL(1) conflicts, left recursion and empty nonterminals.
`circuit.py` turns any compiled table into weights, so stage 2 onwards changes the grammar, not
the machinery. Each rule's source link below goes to the grammar production that produced it.

## The rules

87 rules: 3 for the start and the stops, one per table entry, one default per item.

| item | where the parser is | rule | lookahead | action | source |
|---|---|---|---|---|---|
| — | the first record opens the root (program) | `start` | any | open the root | [toy.py:14](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L14 "src:dim:parse:r_start") |
| — | the root has closed (item 000): write the final newline | `halt` | any | write the final newline | [circuit.py:109](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/circuit.py#L109 "src:dim:parse:r_halt") |
| — | after a syntax error: write the final newline | `halt_error` | any | write the final newline | [circuit.py:111](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/circuit.py#L111 "src:dim:parse:r_halt_error") |
| 1 | program, at its start | `i1_end` | `$` | consume, go to 2 | [toy.py:14](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L14 "src:dim:parse:r_i1_end") |
| 1 | program, at its start | `i1_lparen` | `(` | open S (statement), return to 1 | [toy.py:14](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L14 "src:dim:parse:r_i1_lparen") |
| 1 | program, at its start | `i1_1` | `1` | open S (statement), return to 1 | [toy.py:14](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L14 "src:dim:parse:r_i1_1") |
| 1 | program, at its start | `i1_semi` | `;` | open S (statement), return to 1 | [toy.py:14](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L14 "src:dim:parse:r_i1_semi") |
| 1 | program, at its start | `i1_i` | `i` | open S (statement), return to 1 | [toy.py:14](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L14 "src:dim:parse:r_i1_i") |
| 1 | program, at its start | `i1_w` | `w` | open S (statement), return to 1 | [toy.py:14](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L14 "src:dim:parse:r_i1_w") |
| 1 | program, at its start | `i1_x` | `x` | open S (statement), return to 1 | [toy.py:14](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L14 "src:dim:parse:r_i1_x") |
| 1 | program, at its start | `i1_lbrace` | `{` | open S (statement), return to 1 | [toy.py:14](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L14 "src:dim:parse:r_i1_lbrace") |
| 1 | program, at its start | `i1_error` | any other | syntax error | [toy.py:14](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L14 "src:dim:parse:r_i1_error") |
| 2 | program, after '$' | `i2_close` | any other | close the node | [toy.py:14](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L14 "src:dim:parse:r_i2_close") |
| 3 | statement, at its start | `i3_semi` | `;` | consume, go to 4 | [toy.py:15](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L15 "src:dim:parse:r_i3_semi") |
| 3 | statement, at its start | `i3_lbrace` | `{` | open B (block), return to 4 | [toy.py:15](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L15 "src:dim:parse:r_i3_lbrace") |
| 3 | statement, at its start | `i3_lparen` | `(` | open E (expression), return to 5 | [toy.py:15](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L15 "src:dim:parse:r_i3_lparen") |
| 3 | statement, at its start | `i3_1` | `1` | open E (expression), return to 5 | [toy.py:15](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L15 "src:dim:parse:r_i3_1") |
| 3 | statement, at its start | `i3_x` | `x` | open E (expression), return to 5 | [toy.py:15](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L15 "src:dim:parse:r_i3_x") |
| 3 | statement, at its start | `i3_i` | `i` | consume, go to 6 | [toy.py:15](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L15 "src:dim:parse:r_i3_i") |
| 3 | statement, at its start | `i3_w` | `w` | consume, go to 7 | [toy.py:15](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L15 "src:dim:parse:r_i3_w") |
| 3 | statement, at its start | `i3_error` | any other | syntax error | [toy.py:15](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L15 "src:dim:parse:r_i3_error") |
| 4 | statement, after ';' | `i4_close` | any other | close the node | [toy.py:15](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L15 "src:dim:parse:r_i4_close") |
| 5 | statement, after expression | `i5_semi` | `;` | consume, go to 4 | [toy.py:15](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L15 "src:dim:parse:r_i5_semi") |
| 5 | statement, after expression | `i5_error` | any other | syntax error | [toy.py:15](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L15 "src:dim:parse:r_i5_error") |
| 6 | statement, after 'i' | `i6_lparen` | `(` | consume, go to 8 | [toy.py:15](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L15 "src:dim:parse:r_i6_lparen") |
| 6 | statement, after 'i' | `i6_error` | any other | syntax error | [toy.py:15](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L15 "src:dim:parse:r_i6_error") |
| 7 | statement, after 'w' | `i7_lparen` | `(` | consume, go to 9 | [toy.py:15](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L15 "src:dim:parse:r_i7_lparen") |
| 7 | statement, after 'w' | `i7_error` | any other | syntax error | [toy.py:15](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L15 "src:dim:parse:r_i7_error") |
| 8 | statement, after 'i' '(' | `i8_lparen` | `(` | open E (expression), return to 10 | [toy.py:15](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L15 "src:dim:parse:r_i8_lparen") |
| 8 | statement, after 'i' '(' | `i8_1` | `1` | open E (expression), return to 10 | [toy.py:15](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L15 "src:dim:parse:r_i8_1") |
| 8 | statement, after 'i' '(' | `i8_x` | `x` | open E (expression), return to 10 | [toy.py:15](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L15 "src:dim:parse:r_i8_x") |
| 8 | statement, after 'i' '(' | `i8_error` | any other | syntax error | [toy.py:15](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L15 "src:dim:parse:r_i8_error") |
| 9 | statement, after 'w' '(' | `i9_lparen` | `(` | open E (expression), return to 11 | [toy.py:15](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L15 "src:dim:parse:r_i9_lparen") |
| 9 | statement, after 'w' '(' | `i9_1` | `1` | open E (expression), return to 11 | [toy.py:15](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L15 "src:dim:parse:r_i9_1") |
| 9 | statement, after 'w' '(' | `i9_x` | `x` | open E (expression), return to 11 | [toy.py:15](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L15 "src:dim:parse:r_i9_x") |
| 9 | statement, after 'w' '(' | `i9_error` | any other | syntax error | [toy.py:15](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L15 "src:dim:parse:r_i9_error") |
| 10 | statement, after 'i' '(' expression | `i10_rparen` | `)` | consume, go to 12 | [toy.py:15](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L15 "src:dim:parse:r_i10_rparen") |
| 10 | statement, after 'i' '(' expression | `i10_error` | any other | syntax error | [toy.py:15](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L15 "src:dim:parse:r_i10_error") |
| 11 | statement, after 'w' '(' expression | `i11_rparen` | `)` | consume, go to 13 | [toy.py:15](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L15 "src:dim:parse:r_i11_rparen") |
| 11 | statement, after 'w' '(' expression | `i11_error` | any other | syntax error | [toy.py:15](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L15 "src:dim:parse:r_i11_error") |
| 12 | statement, after 'i' '(' expression ')' | `i12_lparen` | `(` | open S (statement), return to 14 | [toy.py:15](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L15 "src:dim:parse:r_i12_lparen") |
| 12 | statement, after 'i' '(' expression ')' | `i12_1` | `1` | open S (statement), return to 14 | [toy.py:15](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L15 "src:dim:parse:r_i12_1") |
| 12 | statement, after 'i' '(' expression ')' | `i12_semi` | `;` | open S (statement), return to 14 | [toy.py:15](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L15 "src:dim:parse:r_i12_semi") |
| 12 | statement, after 'i' '(' expression ')' | `i12_i` | `i` | open S (statement), return to 14 | [toy.py:15](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L15 "src:dim:parse:r_i12_i") |
| 12 | statement, after 'i' '(' expression ')' | `i12_w` | `w` | open S (statement), return to 14 | [toy.py:15](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L15 "src:dim:parse:r_i12_w") |
| 12 | statement, after 'i' '(' expression ')' | `i12_x` | `x` | open S (statement), return to 14 | [toy.py:15](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L15 "src:dim:parse:r_i12_x") |
| 12 | statement, after 'i' '(' expression ')' | `i12_lbrace` | `{` | open S (statement), return to 14 | [toy.py:15](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L15 "src:dim:parse:r_i12_lbrace") |
| 12 | statement, after 'i' '(' expression ')' | `i12_error` | any other | syntax error | [toy.py:15](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L15 "src:dim:parse:r_i12_error") |
| 13 | statement, after 'w' '(' expression ')' | `i13_lparen` | `(` | open S (statement), return to 4 | [toy.py:15](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L15 "src:dim:parse:r_i13_lparen") |
| 13 | statement, after 'w' '(' expression ')' | `i13_1` | `1` | open S (statement), return to 4 | [toy.py:15](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L15 "src:dim:parse:r_i13_1") |
| 13 | statement, after 'w' '(' expression ')' | `i13_semi` | `;` | open S (statement), return to 4 | [toy.py:15](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L15 "src:dim:parse:r_i13_semi") |
| 13 | statement, after 'w' '(' expression ')' | `i13_i` | `i` | open S (statement), return to 4 | [toy.py:15](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L15 "src:dim:parse:r_i13_i") |
| 13 | statement, after 'w' '(' expression ')' | `i13_w` | `w` | open S (statement), return to 4 | [toy.py:15](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L15 "src:dim:parse:r_i13_w") |
| 13 | statement, after 'w' '(' expression ')' | `i13_x` | `x` | open S (statement), return to 4 | [toy.py:15](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L15 "src:dim:parse:r_i13_x") |
| 13 | statement, after 'w' '(' expression ')' | `i13_lbrace` | `{` | open S (statement), return to 4 | [toy.py:15](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L15 "src:dim:parse:r_i13_lbrace") |
| 13 | statement, after 'w' '(' expression ')' | `i13_error` | any other | syntax error | [toy.py:15](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L15 "src:dim:parse:r_i13_error") |
| 14 | statement, after 'i' '(' expression ')' statement | `i14_e` | `e` | consume, go to 13 | [toy.py:15](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L15 "src:dim:parse:r_i14_e") |
| 14 | statement, after 'i' '(' expression ')' statement | `i14_close` | any other | close the node | [toy.py:15](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L15 "src:dim:parse:r_i14_close") |
| 15 | block, at its start | `i15_lbrace` | `{` | consume, go to 16 | [toy.py:21](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L21 "src:dim:parse:r_i15_lbrace") |
| 15 | block, at its start | `i15_error` | any other | syntax error | [toy.py:21](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L21 "src:dim:parse:r_i15_error") |
| 16 | block, after '{' | `i16_lparen` | `(` | open S (statement), return to 16 | [toy.py:21](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L21 "src:dim:parse:r_i16_lparen") |
| 16 | block, after '{' | `i16_1` | `1` | open S (statement), return to 16 | [toy.py:21](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L21 "src:dim:parse:r_i16_1") |
| 16 | block, after '{' | `i16_semi` | `;` | open S (statement), return to 16 | [toy.py:21](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L21 "src:dim:parse:r_i16_semi") |
| 16 | block, after '{' | `i16_i` | `i` | open S (statement), return to 16 | [toy.py:21](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L21 "src:dim:parse:r_i16_i") |
| 16 | block, after '{' | `i16_w` | `w` | open S (statement), return to 16 | [toy.py:21](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L21 "src:dim:parse:r_i16_w") |
| 16 | block, after '{' | `i16_x` | `x` | open S (statement), return to 16 | [toy.py:21](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L21 "src:dim:parse:r_i16_x") |
| 16 | block, after '{' | `i16_lbrace` | `{` | open S (statement), return to 16 | [toy.py:21](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L21 "src:dim:parse:r_i16_lbrace") |
| 16 | block, after '{' | `i16_rbrace` | `}` | consume, go to 17 | [toy.py:21](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L21 "src:dim:parse:r_i16_rbrace") |
| 16 | block, after '{' | `i16_error` | any other | syntax error | [toy.py:21](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L21 "src:dim:parse:r_i16_error") |
| 17 | block, after '{' '}' | `i17_close` | any other | close the node | [toy.py:21](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L21 "src:dim:parse:r_i17_close") |
| 18 | expression, at its start | `i18_lparen` | `(` | open F (factor), return to 19 | [toy.py:22](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L22 "src:dim:parse:r_i18_lparen") |
| 18 | expression, at its start | `i18_1` | `1` | open F (factor), return to 19 | [toy.py:22](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L22 "src:dim:parse:r_i18_1") |
| 18 | expression, at its start | `i18_x` | `x` | open F (factor), return to 19 | [toy.py:22](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L22 "src:dim:parse:r_i18_x") |
| 18 | expression, at its start | `i18_error` | any other | syntax error | [toy.py:22](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L22 "src:dim:parse:r_i18_error") |
| 19 | expression, after factor | `i19_plus` | `+` | consume, go to 18 | [toy.py:22](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L22 "src:dim:parse:r_i19_plus") |
| 19 | expression, after factor | `i19_close` | any other | close the node | [toy.py:22](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L22 "src:dim:parse:r_i19_close") |
| 20 | factor, at its start | `i20_lparen` | `(` | consume, go to 21 | [toy.py:23](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L23 "src:dim:parse:r_i20_lparen") |
| 20 | factor, at its start | `i20_1` | `1` | consume, go to 22 | [toy.py:23](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L23 "src:dim:parse:r_i20_1") |
| 20 | factor, at its start | `i20_x` | `x` | consume, go to 22 | [toy.py:23](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L23 "src:dim:parse:r_i20_x") |
| 20 | factor, at its start | `i20_error` | any other | syntax error | [toy.py:23](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L23 "src:dim:parse:r_i20_error") |
| 21 | factor, after '(' | `i21_lparen` | `(` | open E (expression), return to 23 | [toy.py:23](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L23 "src:dim:parse:r_i21_lparen") |
| 21 | factor, after '(' | `i21_1` | `1` | open E (expression), return to 23 | [toy.py:23](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L23 "src:dim:parse:r_i21_1") |
| 21 | factor, after '(' | `i21_x` | `x` | open E (expression), return to 23 | [toy.py:23](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L23 "src:dim:parse:r_i21_x") |
| 21 | factor, after '(' | `i21_error` | any other | syntax error | [toy.py:23](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L23 "src:dim:parse:r_i21_error") |
| 22 | factor, after '1' | `i22_close` | any other | close the node | [toy.py:23](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L23 "src:dim:parse:r_i22_close") |
| 23 | factor, after '(' expression | `i23_rparen` | `)` | consume, go to 22 | [toy.py:23](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L23 "src:dim:parse:r_i23_rparen") |
| 23 | factor, after '(' expression | `i23_error` | any other | syntax error | [toy.py:23](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/cparse/toy.py#L23 "src:dim:parse:r_i23_error") |

## Check it by breaking it

Knock parts out in the visualizer (or `handformer.ablate`) and the failure is the one the
explanation predicts:

- **The stack head** (layer 6, head 0). Every close now reads return item 000, so on
  `i(x)i(1);e;` the first close (`CF 003 0003 000`) "returns" to the done item and the model
  stops after 8 records, believing the parse finished. Without the stack, a pushdown automaton is
  a finite one.
- **Rule `i14_e`**, the unit that consumes `else` after `if (E) S`. On `i(x)i(1);e;` nothing
  covers (item 14, lookahead `e`): the item's default is "close unless the lookahead is `e`", so
  no rule fires, the record's constant fields get no byte code, and the model writes blank
  records (`                   |`) forever. The rules leave no slack: each case is one unit.

## Tests and claims

`python -m tests.test_cparse`:

- the compiled table, simulated with a real stack, against a separately written
  recursive-descent parser (`tests/cparse_reference.py`): the same trees, and syntax errors at
  the same token, on 3,600 programs, mutated programs and random strings;
- the weights against the simulator, byte for byte, teacher-forced and greedy: edge cases
  (deep nesting, whitespace, errors at every kind of token, input bytes that look like records),
  50 random and mutated programs;
- the claims: at every byte the model writes, exactly one rule dim is 1 and every other is 0;
  every head reads one exact value in its declared scope (the stack head on closes, the depth − 1
  digits on closes); every yes/no step stays off its ramp. 0 violations.

## Limits

- Stage 1: one-byte tokens and a toy grammar. C's tokens come from the lexer in stage 2.
- Depth ≤ 999 and sequences ≤ 100,000 bytes: the stack search's margins are set for that bound.
- Records are verbose (about 7 records, 140 bytes, per token), so the visualizer takes up to
  80 bytes of source.
- Errors are reported at the offending token, but the tree before it may already have closed
  nodes (default closes): the error is in the right place, its message names no expected token yet.
