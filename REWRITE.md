# The looped rewriter (set aside): the answer is the first byte it writes

Status: set aside (2026-11-05). This model's block has heads that read later positions
(bidirectional), so it is not a decoder. `LOOPED.md` describes its replacement, a causal
recurrent-depth decoder. This one is kept under the name `rewrite`: `python -m handformer.rewrite`,
and `python -m handformer.viz --models rewrite`.

`handformer/rewrite` is a second calculator. It takes the same expressions as the frame-writing one
(`+ - * / ^`, unary minus, brackets, precedence, signed decimals, `~N` precision, spaces, any
length). It writes no working. After the prompt `\n<expr>=`, its first byte is the answer's first
byte:

```
\n2*(3+4)-1=13\n
\n355/113~6=3.141592\n
\n2^64=18446744073709551616\n
```

`ANSWER_FIRST.md` explains why a fixed-depth decoder cannot do this for arbitrary nesting or length,
and names the way out, the looped transformer of its §6. This is that design, built: every weight
is still written by hand from a named formula, and every step is still exact.

    python -m handformer.rewrite "2*(3+4)-1"            # 13
    python -m handformer.rewrite --passes "12*34-400"   # the tape after every pass
    python -m handformer.viz                          # the visualizer: pick it in the menu

## 1. What runs, and what the host does

The model has four parts. Only the second one loops.

| part | what it is | here |
|---|---|---|
| prelude | the byte embedding | pass 0's state is the embedding of the prompt |
| recurrent block | one fixed stack of layers, applied again and again to every position of the prompt | 30 layers: one pass = one rewrite of the whole tape (§4) |
| router | a dim the block computes at each position: 1 sends it through the block again | `route`; the loop ends when no position routes in |
| coda | a causal decoder over the final state | 3 layers: writes the tape's expression, then `\n` |

Between passes, a fixed linear map, the **adapter**, turns the block's last residual into the
next state. Each position writes one of its `H_<byte>` dims ("hold this byte next"), and the
adapter maps it to that byte's embedding row. It also carries four integer dims (§4.3, §4.5) and
drops all of the block's scratch work. Positions are re-added each pass, as features
(`pos`, `pos²`).

The host:

1. frames the bytes;
2. runs the block while any position's router is 1;
3. inserts one blank position before the last one when the root raises `grow` (§3);
4. then decodes the coda greedily (argmax) until `\n`.

It never reads, routes or edits what a position holds. Items 2 and 3 act on single flags the
weights computed, like stopping at `\n`.

## 2. How this lines up with current looped transformers

- **Prelude, recurrent block, coda**: the layout of recurrent-depth language models (Geiping et
  al. 2025, "Scaling up Test-Time Compute with Latent Reasoning"). Their adapter is a learned
  linear map between iterations; ours is written: re-embed the chosen byte, keep the carried
  dims.
- **Re-adding position features every step**: as the Universal Transformer does with its
  coordinate embedding (Dehghani et al. 2018).
- **A router that decides, token by token, whether to recurse again**: Mixture-of-Recursions
  (Bae et al. 2025) routes each token to its own depth. Exit gates in looped LMs such as Ouro
  (Zhu et al. 2025) stop a token's recursion when a learned signal fires. Here:
  - every position computes `route`;
  - the answer bytes take depth 0: they go straight from the embedding to the coda and never
    enter the block;
  - the expression's cells all take the root's decision, because one in-place rewrite can move
    any cell (a shorter result shifts everything after it).
  - The runtime is written for per-position routing. A position routed out keeps its state.
  - That the expression's cells share one decision is a property of this algorithm, not of the
    runtime.
- **Bidirectional attention inside the loop**: the Universal Transformer's encoder and
  looped transformers built by hand as programs (Giannou et al. 2023, "Looped Transformers as
  Programmable Computers") attend in both directions. We need it (§3). The block mixes causal
  heads (prefix scans, read at the root) with bidirectional ones (`HeadSpec.full`: broadcasts
  from the root, lookups to the right). The coda is causal, so answer bytes are generated like
  any decoder's.
- **What differs**: nothing is trained. The block is an exact program, and its halting is a
  decision, not a threshold on a learned probability.

## 3. Why the block attends both ways, and why it asks for room

**Direction.** Under a causal mask, information flows only from earlier positions to later
ones, at every layer and in every pass. Looping a causal block gives each position unbounded
time but leaves it blind to everything after it. In-place rewriting needs the cell that holds
`1` in `1+2*3` to learn that it becomes `7`, which depends on later cells. So the block keeps
causal heads where a prefix view is the point (the parser runs at the root, the last position,
which sees all), and bidirectional heads to move the root's decisions back to every cell.

**Room.** Positions are the model's memory. `2^100` has 5 bytes and a 31-digit answer, and a
power's text rewrite (`2^10 → (2*2)^5`) is longer than what it replaces.

- When a rewrite would not fit in the blanks after the expression, the root raises `grow` and
  the pass keeps every byte.
- The runtime then inserts one blank before the root, and the next pass tries again.
- A division asks for room for its quotient before it starts (§4.5).

This is the scratchpad of the hand-built looped computers, grown on request instead of fixed in
advance.

## 4. One pass

The tape:

- position 0 is the start byte, and the last position is the root (the prompt's `=`);
- between them sit the expression, the prompt's `~N` if any, then blanks.

Each pass the root decides one action, then every cell writes its next byte.

| action | when | write |
|---|---|---|
| space | a space has content after it | the first such space goes; every later cell takes its right neighbour |
| brackets | the innermost group holds one number | the brackets go (`-(-v)`: both signs too) |
| reduce | the innermost group has an operator | the redex (innermost group, then `^`, `* /`, `+ -`, left to right) becomes its value |
| room | a rewrite needs more blanks | nothing: a blank is inserted before the root |
| halt | none of these: the expression is a number | nothing, and no position routes in |

Every write is **replace cells `lo..hi` by `m` bytes**:

- cells before `lo` keep their byte;
- the `m` result cells take the result;
- every later cell takes the byte `g = hi - lo + 1 - m` cells to its right, or a blank past the
  tape's end. When `g < 0` the result is longer, and later cells move right into the blanks.

The redex rules are exactly those of the frame-writing calculator (`tests/reference.py` `step`),
so both models agree on every expression.

The block's 30 layers:

| layers | what |
|---|---|
| 1-13 | the parse, at the root: marker classes and their latest positions (prefix scans), the innermost group `p..c`, the operator `o`, both operands, signs, decimal points, the action |
| 14 | broadcast 1 (bidirectional): the redex's fences, offsets and mode reach every cell |
| 14-23 | columns (§4.1): each column cell reads its digits, then its carry or borrow |
| 24-28 | the result's shape, at the root: top and bottom nonzero column, sign, cancel, `.` position, `lo`, `m`, `g` |
| 29 | broadcast 2: every cell learns its zone and the router bit |
| 30 | each cell gathers its next byte and writes its `H_<byte>` dim |

### 4.1 Numbers are columns

- Column `j` of a result is worth 10^(j-F) and lives in one of the redex's own cells,
  `CB - j`, where CB is b's last digit.
- `F` is the number of fractional digits: max(fa, fb) for + and -, fa + fb for *, N for /.
- Each column cell gathers its digit of `a` and of `b`, aligned at the decimal point.

**Carries.** The carry into column `j` is decided by the first lower column whose digit sum is
not 9: a carry if that sum is 10 or more, none if it is 8 or less. That column is the first
match to the cell's right, which is one `next_match` lookup (`primitives.py`). Borrows work the
same way, using the first lower column whose digit difference is not 0. So + and - take one
pass whatever the length.

**Both orders.** a-b and b-a are computed side by side. The borrow out of a-b's top column (it
reads 9) says |a| < |b|, and the root picks b-a and its sign.

**Writing the result.**

- The top nonzero column sets the length.
- The bottom nonzero column sets how many trailing fractional zeros are dropped.
- The result is written as: sign, integer digits (at least `0`), `.`, fractional digits.

### 4.2 + and -

One pass each, any length. `999…9+1` and `100…0-1` cost the same single lookup per column.

### 4.3 ×: one pass per digit of b

The adapter carries a column accumulator `K` per cell and a counter `kk` at the root. Each pass
adds |a| × b_kk × 10^kk into `K`, and the tape keeps its bytes. The addition happens in two
levels:

1. Each column sums K, the units of a_i·b_kk, and the tens of a_(i-1)·b_kk. That is at most 26.
   The sum's own tens (0 to 2) go one column up.
2. One binary carry chain (§4.1) settles the rest.

After the last digit of b, one more pass writes K as the result and clears it. So `12*34` takes
3 passes.

### 4.4 ^: a text rewrite

`a^b` is rewritten the way the frame-writing calculator does it:

| case | rewrite |
|---|---|
| b < 0 | `(1/a^n)` |
| b = 0 | `1` |
| b is `1` | `a` |
| b odd | `(a*a^(b-1))` |
| b even | `(a*a)^h`, with h = b/2 computed digit by digit: 5·[previous digit odd] + d//2 |

Every case is one layout of eleven segments: fixed bytes, copies of `a` and `b`, and computed
digits. The cells learn their segment by broadcast. Later passes evaluate the products. Room
comes from `grow`.

### 4.5 /: long division over passes

`a/b~N` is floor(A/B) with N fractional digits, where:

- A = |a|'s digits followed by fb + N zeros;
- B = |b|'s digits followed by fa zeros;
- L is the number of digits in A.

The state is:

- the remainder `r`, in `K` (columns of the redex's cells);
- the next digit index `t`, in `kk`;
- the count `qc` of the current quotient digit, carried at the root;
- the stored quotient digits `Q`, kept in blank cells after the expression.

The passes:

| pass | when | what changes |
|---|---|---|
| room | fewer than L + 3 blanks | a blank is added (before anything starts) |
| shift | r < B, or t = 0 | store qc as quotient digit t-1; r = 10r + A_t (every column takes the one below); qc = 0; t + 1 |
| subtract | r ≥ B | r = r - B (the subtraction's own borrow chain; r < B is its borrow out); qc + 1 |
| write | t = L + 1 | the stored digits become the result's columns, and the usual writer formats them |

The tape keeps its bytes until the write. `355/113~6` takes 49 passes (12 of them room), and
`1/7~30` takes 203.

## 5. Exactness

| guarantee | how |
|---|---|
| exact flags | every flag is a robust step (input within 1/4 of an integer), never a bare ReLU of gathered values. A gather leaks about 1e-13, and a flag that gates by BIG = 1e6 must not pass that on. A bare-ReLU XOR was the first bug here |
| exact broadcasts | every cell holds its own copy of the parse, and away from the root those values are unbounded garbage. Broadcasts read the root with a 700-logit margin, so the leak underflows to exactly 0 |
| exact state between passes | every byte written is a robust step, and the carried integers are rebuilt from robust steps. The state is a byte embedding plus exact integers at every pass. Over the whole test suite it stays within 7e-15 of that, and the error never builds up, because each pass rebuilds it. The carried dims are 0 when the loop halts |

## 6. Size and cost

| | d_model | layers | heads | units | nonzero weights |
|---|---|---|---|---|---|
| recurrent block | 602 | 30 | 87 (21 bidirectional) | 1,227 | |
| coda | (same stream) | 3 | 3 | 17 | |
| whole model, with the adapter | | | | | 9,993 |

Passes taken:

| expression | answer | passes | room passes | positions at the end |
|---|---|---|---|---|
| `2+3` | 5 | 2 | 0 | 5 |
| `123456789*987` | 121851850743 | 5 | 0 | 15 |
| `2*(3+4)-1` | 13 | 5 | 0 | 11 |
| `2^10` | 1024 | 26 | 9 | 15 |
| `2^64` | 18446744073709551616 | 57 | 21 | 27 |
| `355/113~6` | 3.141592 | 49 | 12 | 23 |
| `1/7~30` | 0.1428…57 (30 digits) | 203 | 34 | 42 |

Each count includes the final pass that halts.

**Depth.** The frame-writing calculator spends its extra depth as written bytes; this one
spends it as passes. The ceilings of `ANSWER_FIRST.md` §6 are gone:

- nesting costs passes, not layers;
- the length of a number costs nothing for + and -;
- for × it costs one pass per digit of b.

## 7. Tests and the visualizer

`tests/test_loop.py` runs the real loop:

- **Tapes**: the tape going into every pass equals `tests/rewrite_reference.py`, byte for byte. That
  reference is a pure-Python model of the passes, built on `tests/reference.py`'s own reduction
  step.
- **Answers**: each answer equals the frame-writing calculator's and exact arithmetic.
- **Drift**: the state stays a byte embedding, within 1e-9 of exact.
- **Coverage**:
  - hand-picked cases and random + - × trees with signs and brackets;
  - 80-digit operands and deep nesting;
  - powers and division;
  - the interpretability corpus of `tests/test_claims.py`.

In the visualizer (the menu's "calculator, looped"; the site runs it in the page), the
**Passes** bar lists every pass by what the root decided, then the answer.

- **A pass** shows the tape through the block's layers. A position's "next byte" is the byte
  it holds after this pass (the hold dims, read out by their own unembedding).
- **The answer** shows the final tape and the answer through the coda.
- **Layers**: both views use one stack of 33 layers (block, then coda). The layers a view does
  not run are empty in it.
- Heads, units, heatmaps, knockouts and links work as for the other models.

## 8. Limits

- Division by zero never satisfies r < B, so it runs until the pass budget (the runtime's
  `max_passes`).
- Precision `~N` takes up to four digits (N ≤ 9999), as in the frame-writing calculator.
- Brackets must balance.
- Positions stay below the global bound of about 10^6 (`DESIGN.md` §12).
- Every pass runs the whole block over the whole tape, and the root's decision is shared, so a
  long division costs passes × positions × 30 layers.
