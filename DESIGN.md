# handformer: design

A decoder-only transformer for arithmetic. Every weight is written by hand and
nothing is trained. The tokenizer is raw bytes (vocab 256). One set of weights
evaluates whole expressions (`+ - * / ^`, unary minus, parentheses,
precedence) on signed decimal numbers of any length, with division to a
precision given in the prompt (`~N`).

```
\n2+3*(4-1)=[4-1=3;3]2+3*3=[3*3=90|90;9]2+9=[2+9=11;11]11\n
```

Everything after the first `=` is written by the model.

---

## 1. Rules

1. **All computation happens inside the model.** The host feeds bytes, runs
   greedy decoding, and stops at `\n`. It never parses, routes, reverses,
   pads or post-processes anything.
2. **Arbitrary length.** No context size is baked into the weights. Position
   enters only as functions of the integer index (`pos`, `pos²`).
3. **Every weight comes from an explicit, documented formula** that names
   the residual dimensions it reads and writes. Builder loops are allowed
   (for example, one ReLU ramp per value 0..99). Learning and optimisation are not.
4. **The residual stream is laid out deliberately.** Each dimension has one
   meaning and a name, and is written by known sublayers.
5. **Circuits follow the codebase's patterns.** New operations reuse the shared
   primitives and the frame/phase conventions instead of adding parallel
   mechanisms.

## 2. Decisions log

| date | decision | consequence |
|---|---|---|
| 2026-11-03 | Byte tokenizer, hand-written weights, one task: arithmetic | |
| 2026-11-03 | Arbitrary-length operands | Positional features are functions, not a table; the model writes its working |
| 2026-11-03 | No host-side reversal or fix-ups ("no hacks like flipping at the end") | The answer comes out most-significant digit first, copied by attention |
| 2026-11-03 | Precision goes in the prompt: `<expr>~N=` | Built (§21): the model reads N (up to 4 digits) from the prompt |
| 2026-11-03 | Full expressions with precedence and parentheses | Term rewriting in frames; operands addressed by bounding bytes |
| 2026-11-03 | Powers: integer exponents; roots: digit-by-digit square root | Integer powers built (§23); roots not yet |
| 2026-11-03 | Operator selection must be in the weights, not separate models | One merged model; circuits are gated by flags computed in the weights |
| 2026-11-03 | Use the residual stream for as much work as possible | Position arithmetic (products, division) is done with attention instead of extra tokens |
| 2026-11-03 | Unary minus binds tighter than `^`: `-2^2` = 4 | Simpler parser; differs from the usual maths convention (§23) |
| 2026-11-03 | Factored per-head value/output maps (W_V d×r, W_O r×d) | 40× faster runs than dense d×d; same nonzero weights (§7) |
| 2026-11-03 | Every dim, head and unit carries a plain-language explanation | Generated from names and unit labels (§25); labels never change a weight |
| 2026-11-03 | Step explanations are computed by the page from raw values | Display only: nothing the page computes is fed back to the model (§26) |
| 2026-11-03 | Public demo is a static site; the model runs in the browser | No server to host; a parity test holds the JS engine to the Python model (§24) |
| 2026-11-03 | Mobile: same features, touch-aware | Tap replaces hover; the heatmap scrolls inside its panel (§18) |

## 3. Why the model writes its working

A transformer does a fixed amount of work per token. With unbounded operands:

- the first digit of `a+b` depends on the entire carry chain;
- each product digit depends on O(n) partial products;
- reducing an expression is a sequential process.

So the model writes intermediate bytes and reads them back. That token stream
is its only memory across steps, and every byte of it is produced by the
weights. Within one step, as much as possible is done in the residual stream
(see §4–§5 and the primitives in §11): position products, integer division and pointer
chasing are all computed by attention instead of being written out.

## 4. How attention is used

### 4.1 Heads are addressing, not similarity

No head here measures how similar two tokens are. Each head answers exactly one
question: *which position do I read from?* The query encodes an address,
computed in the residual stream; the keys encode position features; and the
score is a function whose maximum falls exactly on the intended key. The value
side is almost always a copy of one to ten dimensions. Of the 65 heads (before signed operands; there are now 73):

| kind | count | score shape | what it reads |
|---|---|---|---|
| gather / copy | 40 | `−β(k − t)²` | the value at computed position t: a digit, a one-hot byte, a pointer, a square |
| latest-match | 21 | `λ·feature_k`, feature = position on matches | the most recent byte of a class, or an ordered (frame, −pos) variant |
| next-match | 4 | `−β(pm_k − lt)² + bonus·match_k` | the first byte of a class after a point |
| floor division | 1 | `−β(s·k − x + (s−1)/2)²` | ⌊x/s⌋, with positions acting as a number line |

Softmax is used as an exact argmax. The scores are built so that the winner
leads every other key by a fixed margin (30–60 in logit units) at any
sequence length, so its weight is 1 − O(n·e⁻³⁰).

### 4.2 Why a quadratic score in `pos`, `pos²`

A dot product can only produce scores linear in the key features. Using the
features `pos` and `pos²` makes the score quadratic in key position, with
coefficients chosen by the query:

```
score(k) = q₁·pos²_k + q₂·pos_k = −β(k − t)² + const   when q₁ = −β, q₂ = 2βt
```

- **Sharpness doesn't decay.** The gap between the target and its neighbours
  is β at every length. Sinusoidal or rotary positions get blurrier with
  distance.
- **The address can be anything the residual stream computes:** a reflection
  (`2S − 1 − q`), an offset from a marker, a clamped pointer, a product.
- **The same features make a lookup table of squares.** Gathering `pos²` at
  position x returns x², which gives exact products of positions and the
  floor-division head (§11). Positions double as a number line for arithmetic.

### 4.3 Causality, and how the model searches forward anyway

Under causal attention, "latest match at or before me" is natural; "first match
after point t" is not, because nothing in a dot product can express k > t. It
is solved by pointer chasing in two hops:

1. Every position stores `pm` = the latest match strictly before it (a
   latest-match value gathered at pos − 1), plus `pm²` from the square table.
2. Let `lt` = the latest match at or before t (gathered from position t). The
   first match after t is the only matching key with `pm = lt`. That equality
   is a quadratic score, and a match bonus separates the target from the
   non-matching keys that share its pointer.

This needs no counting, no depth tracking and no fixed window, and it is what
the parser uses to find `)`, the next operator and the end of each operand.

### 4.4 Attention as arithmetic

| need | head-level construction |
|---|---|
| x·y for positions | `((x+y)² − x² − y²)/2` from three gathers of `pos²` |
| ⌊x/s⌋ | ruler head: every position k bids with `−β(sk − x′)²`; the query needs only s² and s·x′, both products |
| "which line and column am I on" in × | j = ⌊(q−E)/s⌋, m = (q−E) − j·s: no line counters are written to the token stream |
| reverse a scratch line | gather at `top − (q − S)`: a reflection |
| skip leading zeros | latest-match over "nonzero working digit" gives where the copy starts |

## 5. How the residual stream is used

### 5.1 Allocation

All 576 dimensions are named. They are allocated by category:

| category | dims | contents |
|---|---|---|
| input | 37 | `one`, `pos`, `pos²`, digit value, digit one-hot, byte-class flags |
| shared frame locator | 74 | marker positions, frame flags, phases, `act_*`, previous-match pointers, signs, precision N |
| parser / rewriter | 114 | p, c, o, operand bounds, decimal pads, region flags, source positions, copy one-hots |
| addition (private) | 16 | digit targets, gathered digits, carry, answer pointer |
| subtraction (private) | 25 | same, plus top/lowest-digit pointers and complement value |
| multiplication (private) | 39 | squares, line index j, column m, b_j one-hot, products, carry |
| division (private) | 132 | grid geometry, multiples 0–9 of the divisor, digit comparisons, quotient digit |
| power (private) | 102 | exponent parity and halving, expansion segments, copied base |
| output | 37 | one dimension per emitted byte class, read linearly by the unembedding |

The operator circuits never run at the same position, so their private
dimensions could share space (planned compaction, about 576 → 300).

### 5.2 Conventions

- **Every dimension holds one exact quantity:** an integer position, a digit,
  a 0/1 flag, or a one-hot entry. Nothing is a learned mixture, so any
  dimension can be read off the residual stream to check a step (that is how
  both bugs in §12 were found).
- **Each scalar is written once and read many times.** Later sublayers read it
  through linear forms. Many derived quantities never get their own dimension
  at all: `Le = F − 1`, `s = P − F + 1`, `m = x − j·s` and the rewrite lengths
  n1/n2/n3 are linear forms that attention queries and MLP units read directly.
- **Exactly 0/1 flags.** Every flag comes from a robust step, so it can safely
  gate products and selects that scale by BIG.
- **Gating, not routing.** Circuits run everywhere; a flag decides whether a
  unit's ReLU can fire (`−BIG·(1 − flag)` on its input). Phases, operator
  selection and selects all use this one mechanism.
- **Private namespaces.** Each operator circuit's dimensions are prefixed
  (`mul.J`), so circuits can't collide. Shared dimensions are written only by
  the frame locator.
- **One-hot transport.** Digits and expression bytes move between positions as
  one-hots (10 or 15 dims) in a single head. Digit products and byte copies
  then become ANDs of one-hots, with no decoding step.

### 5.3 The residual stream as scratch, the token stream as memory

There are two kinds of scratch, and the design keeps them apart:

- **Within a position (residual stream):** any fixed-depth computation, such
  as products of positions, division, pointer chasing, carry recovery, or
  selecting a source. These are free in tokens and limited only by depth.
- **Across positions (tokens):** anything whose size grows with the input,
  such as carry chains, partial products, or a reduced expression. These must
  be written, because a position's residual stream holds only a fixed amount.

Two rules decide what gets written:

- **Never write what can be recomputed.** Carries and borrows are recovered
  from the previous output digit. Line and column indices in × are recomputed
  by division. The sub-expression's bounds are recomputed by every position
  in a step.
- **Write a fact once, when it is first known, and read it back by position.**
  The result's sign is put in the end-of-working byte (`;` or `<`) at the
  moment the final borrow is known.

## 6. Order of operations

### 6.1 No stack

A stack needs memory that grows without bound and a pointer that moves, and a
transformer has neither in the residual stream. Rewriting reduces an
expression while keeping only one fact, the current line, and the token
stream already holds that line. So evaluation is repeated rewriting: choose
one reducible sub-expression, compute it in a frame, write the line with that
sub-expression replaced, and repeat.

### 6.2 Choosing the sub-expression with fixed-depth lookups

| rule | why it is correct | construction |
|---|---|---|
| innermost parentheses first | the latest `(` has no `(` after it, so its group contains no other group | latest-match over `(` / `]` at Le |
| its end is the first `)` after it | the nearest closing parenthesis after an innermost `(` is its partner | next-match over `)` from p |
| `*` before `+` `-` | precedence | next-match over `*` from p; used only if it lies before c |
| leftmost first within a level | left-to-right: `10−3−2 = (10−3)−2` | next-match returns the *first* operator after p |
| operands are maximal digit runs | there are no unary operators yet | latest non-digit before o, first non-digit after o |
| drop parentheses that now hold one number | `(a op b)` → result | equality of the bounds: `a_lo = p` and `b_hi = c` |
| stop when one number is left | the line had only this operator | latest operator at Le is o, and latest operator before o is outside the line |

Each rule is a fixed number of attention lookups, whatever the expression
length or nesting depth. Nesting depth never appears in the circuit: choosing
innermost-first means a depth counter is never needed.

### 6.3 Rejected alternatives

- **Depth counting** (prefix sums of `(` minus `)`): attention gives averages,
  and turning an average back into an exact count needs a product of two
  unbounded values.
- **Rightmost-first with sign flipping:** this avoids forward search, but
  breaks left-to-right order for `−` (`a−b+c` would become `a−(b+c)` unless
  operators are flipped), and it makes negative intermediate values more likely.
- **Storing parse results in a "register" at the `[` position:** it works, but
  next-match makes every position able to recompute them, which is simpler and
  needs no extra dependency between positions.
- **A fixed lookahead window:** breaks arbitrary length.

## 7. Engine (`handformer/model.py`)

- **Pre-residual, decoder-only.** Each layer adds its attention heads to the
  residual stream, then adds its MLP output (`ReLU`). Causal mask.
- **No LayerNorm.** It rescales the stream and would break exact linear readouts.
- **No biases.** Every byte's embedding sets a constant `one` dimension, so
  every affine map is a linear map that reads `one`.
- **Positional features:** `pos` and `pos²`, computed from the integer index.
  Positions have no learned table, so the model accepts any length.
- **float64 throughout.**
- **Weights are written sparsely by name.** `ModelBuilder` collects linear forms
  `{dim: coefficient}` for queries, keys, value/output maps and MLP units, and
  turns them into dense matrices. `ModelBuilder.at(i)` places a sublayer at a
  fixed global depth, so independent circuits can share layers.
- Each head stores a factored value/output map: `W_V` (d × r) selects the r
  dimensions the head moves and `W_O` (r × d) writes them. Their product is the
  fused map shown by the visualizer. Before this, a dense d × d map made every
  head cost d² and the visualizer took 25–36 s per run; now it takes 0.4 s.

## 8. Sequence grammar

```
\n X_0 = [ a op b = <working> T <result> ] X_1 = [ ... ] ... X_n \n
```

| byte | meaning | written by |
|---|---|---|
| `\n` (pos 0) | BOS: zero-valued bounding byte and attention sink | prompt |
| `=` after a line | the line has more work; the next byte is `[` | rewriter |
| `[` | opens a frame (one reduction step) | rewriter (phase 1) |
| `a op b =` | copy of the chosen sub-expression | rewriter (phase 2) |
| working | operator-specific scratch, see §10 | operator circuit (phase 3) |
| `;` / `<` | end of working; `<` means the result is negative | operator circuit |
| result | the answer, most-significant digit first | operator circuit |
| `]` | closes the frame | operator circuit |
| X_{t+1} | the line with the sub-expression replaced by its result | rewriter (phase 4) |
| final `\n` | no operator left: X_n is the answer | rewriter |

**Phases**, decided per query position in layer 3:

| phase | condition | action |
|---|---|---|
| 1 | current byte is `=` and no frame is open | emit `[` |
| 2 | frame open, frame's `=` not yet written | copy the sub-expression, then `=` |
| 3 | frame open, `=` written | the active operator circuit writes working, result, `]` |
| 4 | frame closed, current byte is not `=` | write the reduced line, then `=` or `\n` |

## 9. Global layer map

Every position runs all 22 layers. Which circuits matter depends on the
position's phase (§8): a byte being copied into a frame uses the parser's
copy path (L17), a byte of working uses its operator's layers (3–12), and a
byte of the rewritten line uses the rewrite path (L22). The circuits share
layers because they never need to be active at the same position.

| layer | heads | heads by circuit | MLP units | units by circuit |
|---|---|---|---|---|
| 1 | 2 | frame 2 | 40 | frame 40 |
| 2 | 16 | frame 16 | 14 | frame 14 |
| 3 | 14 | frame 14 | 36 | frame 5, add 12, sub 15, mul 4 |
| 4 | 39 | frame 2, parser 7, add 5, sub 7, mul 7, div 8, pow 3 | 49 | frame 11, add 9, sub 12, mul 7, div 2, pow 8 |
| 5 | 9 | parser 3, add 1, mul 2, div 2, pow 1 | 89 | parser 6, add 38, sub 27, mul 6, pow 12 |
| 6 | 10 | parser 1, sub 1, mul 4, div 4 | 78 | parser 4, sub 9, mul 10, div 44, pow 11 |
| 7 | 21 | mul 2, div 17, pow 2 | 170 | parser 3, sub 3, mul 22, div 48, pow 94 |
| 8 | 9 | parser 7, div 2 | 97 | parser 6, sub 12, mul 18, div 32, pow 29 |
| 9 | 17 | parser 5, div 12 | 57 | parser 12, mul 3, div 26, pow 16 |
| 10 | 3 | parser 2, pow 1 | 257 | parser 4, mul 126, div 66, pow 61 |
| 11 | 3 | parser 1, div 2 | 9 | parser 5, div 4 |
| 12 | 2 | parser 1, div 1 | 17 | parser 5, div 12 |
| 13 | 0 | — | 2 | parser: raw decimal pads |
| 14 | 0 | — | 11 | parser: pads `pad_a`, `pad_b`, result places `f_res` |
| 15 | 0 | — | 27 | parser: frame-copy segment flags (in a, pad, op, in b, pad), '.' skips |
| 16 | 0 | — | 8 | parser: frame-copy source position `src2`, trailing zeros |
| 17 | 1 | parser: frame copy | 4 | parser |
| 18 | 0 | — | 7 | parser: end of the reduced line `Lend`, result width, '.' position |
| 19 | 0 | — | 20 | parser: rewrite region flags (prefix, result, suffix) |
| 20 | 0 | — | 2 | parser: left padding of the result |
| 21 | 0 | — | 6 | parser: rewrite source position `src4` |
| 22 | 1 | parser: rewrite copy | 44 | parser: output votes for every byte |

Totals: 147 heads (frame 34, parser 29, add 6, sub 8, mul 15, div 48, pow 7)
and 1,044 units (frame 70, parser 176, add 59, sub 78, mul 196, div 234, pow 231).

**Why layers 13–16 and 18–21 have no heads.** Those layers do arithmetic on
values the position already holds, so there is nothing to look up. The two
copies at the end each need a source position that takes a chain of ReLU
steps to compute, and an MLP of layer i is only visible to the heads of layer
i+1 (§7). The frame copy chain is raw pads (L13) → pads (L14) → segment flags
(L15) → source position (L16) → copy head (L17). The rewrite chain is line end
(L18) → region flags (L19) → padding (L20) → source position (L21) → copy head
(L22). Depth here is the length of a dependency chain, not a count of
lookups.

### 9.1 Shared locator (`frame.py`, layers 1–3)

Every marker is the **latest** occurrence of its byte class, so each frame sees
its own markers. Flags are relative to the current frame:

```
F  = latest '['    Rb = latest ']'    E = latest '='    S = latest ';' or '<'
R  = latest '|'    P  = latest operator byte
open      = [F > Rb]        seen_eq   = [E > F]
seen_semi = [S > F]         seen_bar  = [R > F]
act_<op>  = open AND seen_eq AND (byte at P is <op>)
neg       = byte at S is '<'
```

### 9.2 Choosing and rewriting the sub-expression (`expr.py`, layers 4–22)

The source line X_t lies strictly between `Ls` (the `]` that closed the
previous frame, or BOS) and `Le = F − 1` (its `=`).

```
p      = latest '(' or ']' at Le        innermost group start (or line start)
c      = first ')' after p, else Le     next-match
o      = first '*' after p if < c, else first '+'/'-' after p
a_lo   = latest non-digit before o      b_hi = first non-digit after o
drop   = [a_lo == p] AND [b_hi == c] AND [p > Ls]    group is exactly (a op b)
A..B   = a_lo+1 .. b_hi-1, or p .. c when drop
cont   = [latest op at Le > o] OR [latest op before o > Ls]
```

- **Frame copy:** byte `a_lo + 1 + (q − F)`, then `=` after `b_hi − a_lo − 1` bytes.
- **Rewrite**, with `r = q − Rb`:
  - prefix: `Ls+1 .. A−1`
  - result: `S+1 .. Rb−1`
  - suffix: `B+1 .. Le−1`
  - then `=` if `cont`, else `\n`

  Region flags come from robust steps. The source position is assembled with
  flag-gated selects.
- **Copies** move bytes as one-hots over the alphabet `0-9 + - * ( )`. Each
  output unit is an AND of "this byte" and "this phase".

## 10. Operator circuits

Each circuit has private, namespaced residual dimensions (`add.A`, `mul.J`, …).
It reads the shared dimensions and runs on every input. Only the units that
write output logits are gated by `act_<op>`, so an inactive circuit adds
exactly 0. Operands are addressed by their bounding bytes: `a` lies strictly
between `F` (the `[`) and `P`, and `b` between `P` and `E`. Digit i (i = 0 is
the units digit) sits at `hi−1−i`. Once an operand runs out, the gather lands
on the bounding byte, whose value is 0.

### 10.1 Addition (layers 3–5)

```
[95+7=201;102]
```
- Working: the sum, LSB first, with max(len a, len b) digits (+1 if the last carry is set).
- **Carry recovery:** `carry_i = [y ≥ 10] + [y = 9 ∧ previous output digit = 0]`
  with `y = a_{i−1} + b_{i−1}`. The carry is never stored.
- Answer: the working copied in reverse, starting from its highest nonzero
  digit (found by a latest-match head; falls back to `=` so the all-zero case
  gives one `0`).

### 10.2 Subtraction (layers 3–8)

```
[3-7=6<-4]      [100-1=990;99]
```
- Working: `D = (a − b) mod 10ⁿ`, LSB first.
- **Borrow recovery:** `borrow_i = [y ≤ −1] + [y = 0 ∧ previous output digit = 9]`,
  with `y = a_{i−1} − b_{i−1}`.
- **The sign goes in the end byte:** `;` or `<` is decided when the final borrow
  is known, and read back later through `neg`.
- **Complement during the copy:** for a negative result, emit `-`, then
  `10ⁿ − D` digit by digit:
  - 9−d above the lowest nonzero digit z,
  - 10−d at z,
  - 0 below z.

  The top digit of the result is max(z, latest digit ≠ 9). z is found with
  the newest-frame-earliest feature (§11).

### 10.3 Multiplication (layers 3–10)

```
[12*34=840,040|8040;408]
```
- **Lines:** `T_j = ⌊T_{j−1}/10⌋ + a·b_j`. Each line is the previous line
  shifted down one digit plus a × one digit of b, written LSB first with
  exactly la+1 digits:
  `line j digit m = T_{j−1}[m+1] + a_m·b_j + carry_m (mod 10)`.
- Since `T_j < 10^{la+1}`, every line has stride `s = la + 2 = P − F + 1`.
  Line j starts at `E + 1 + j·s`.
- **Where the next byte goes:** `x = q − E`, `j = ⌊x/s⌋` (floor-division head),
  `m = x − j·s` (position product).
- **Digit products:** b_j arrives as a one-hot, so `a_m·b_j` is 10 gated ReLUs.
- **Carry:** `carry = ⌊y/10⌋ + [previous digit < y mod 10]` with
  `y = T_{j−1}[m] + a_{m−1}·b_j ≤ 90`. This works because the carry is ≤ 9.
- **Collect line C:** the product LSB first, la+lb digits. Digit k is
  `T_k[0]` (at `E + 1 + k·s`) for k < lb−1, and comes from the last line otherwise.
- Answer: C copied in reverse from its highest nonzero digit.

## 11. Primitive library (`primitives.py`, `common.py`)

| primitive | computes | construction |
|---|---|---|
| `gather_head` | value at a computed position t | score `−β(k−t)² = 2βtk − βk²`: query `[2βt, −β]`, keys `[pos, pos²]`. Score gap to neighbours is β at any length |
| `latest_head` + `latest_feature` | latest position in a byte class | key feature `2(pos+1)` on matches, `+1` on BOS (fallback); take the max |
| `pos_if` | "position if predicate" key feature | `ReLU(pos + BIG·(Σpred − n))` |
| `newest_frame_earliest` | earliest match in the current frame | key feature `BIG·F_k − pos`: orders keys by (frame, −pos) |
| `next_match_head` | first match after t | every position stores `pm` = latest match at pos−1. The first match after t is the only matching key with `pm = latest match at t`. Score `−β(pm−lt)² + bonus·match` needs `pm²` from the square table |
| square table | x² for a computed x | gather `pos2` at position x |
| `product_form` | x·y exactly | `((x+y)² − x² − y²)/2`, three square-table gathers |
| `floor_div_head` | ⌊x/s⌋ | attention over the positions 0..q: score `−β(s·k − x + (s−1)/2)²`, needing only s² and s·x |
| `clamp01_units` | `[x ≥ 1]`, exact 0/1 | `clamp(4(x − ½), 0, 1)`: breakpoints at ½ and ¾ |
| `hat_units` | `[x = c]`, exact 0/1 | `step(x ≥ c) − step(x ≥ c+1)` |
| `gated` | silence a unit unless flag = v | adds `−BIG·|flag − v|` to the unit's input |
| `times_flag_units` | `x·flag`, signed x | two gated ReLUs |
| `ramp_mod10_units` | one-hot of `x mod 10` for integer x | shared ramps `ReLU(x−c)`; each hat is `R(k−1) − 2R(k) + R(k+1)`, grouped by residue |
| one-hot transport | move a digit or byte as one-hot dims | embedding carries `oh0..9` and byte-class flags |

## 12. Numerics

| constant | value | role |
|---|---|---|
| `GATHER_BETA` | 30 | sharpness of the quadratic lookup (leak ≈ e⁻³⁰) |
| `LATEST_BETA` | 40 | latest-match lookup (feature gap ≥ 2) |
| `NEXT_BETA`, `NEXT_BONUS` | 60, 30 | next-match equality and match bonus |
| `BIG` = `LEN_MAX` | 10⁶ | gating constant; longest supported sequence |
| `BIG_SQ` | 10¹³ | gate for features that scale like BIG·pos |
| `STEP_SLOPE` | 4 | robust steps reject noise up to ±¼ |
| `DIGIT_LOGIT`, `CONTROL_LOGIT` | 10, 20 | a control byte always beats a digit firing at the same position |

- The length bound comes from float64 resolution of `β·pos²` and from `BIG`.
  It is documented, not hidden. Everything stays exact up to roughly 10⁵–10⁶
  positions.
- **Lesson 1: noise amplification.** Selects multiply flags by BIG, so 10⁻¹²
  of attention noise became 10⁻⁶. Integer tests with breakpoints on the
  integers passed that through, and a copy landed one byte off. All integer
  tests are now robust steps with half-integer breakpoints, so every flag is
  exactly 0 or 1.
- **Lesson 2: ties.** With a `pos+1` feature, a matching BOS (1 + 1 fallback)
  tied with a match at position 1, and attention averaged the two. The
  feature is now `2(pos+1)`.

## 13. Operator routing and merging (`arith.py`)

- **One embedding.** Each operator byte sets `is_plus`, `is_minus` or
  `is_star`, plus `is_opany`.
- **The frame locator works out which operator this frame holds** (`act_*`).
  The host never sees the operator.
- **Circuits are merged by global layer index.** Private dimensions are
  prefixed with the circuit name, while the input, shared-frame and output
  dimensions are common.
- **Output-writing units are gated** by their circuit's `act_*`.

## 14. Size (merged model)

| | |
|---|---|
| layers | 22 |
| heads | 147 |
| MLP units | 1,044 |
| d_model | 576 |
| nonzero parameters | 8,478 |

Stored as dense matrices that is 2.08M cells, of which 0.41% are nonzero.
Heads are factored (§7): query/key width 1–3, and each head moves at most
18 values. A dense d×d value/output map per head would be 48.8M cells.

## 15. Verification

- `tests/reference.py` generates the exact expected byte sequence. It is plain
  Python written separately from the weights, using the same reduction rule
  and working formats.
- **Teacher forcing.** `tests/helpers.check` runs one forward pass over the
  prompt plus the expected output, and requires the argmax at every output
  position to equal the next byte. That is equivalent to greedy decoding
  producing the sequence exactly.
- `tests/test_ops.py`:
  - every `a op b` with a, b < 40, for each operator;
  - random operands: + and − up to 300 digits, × up to 25 digits;
  - edge cases: zeros, leading zeros, long carry and borrow chains.
- `tests/test_expr.py`: hand-picked expressions, plus random expression trees
  up to depth 4 with parentheses, precedence and left-to-right order.
- Run all: `python -m unittest discover -s tests -t .`
- The test harness recomputes full n² attention with no caching, which limits
  how long the tested × cases can be. The weights themselves are not limited
  this way.

## 16. Limitations

- **A prompt must contain at least one operator.**
- **Parentheses around a bare number** (`(5)+1`) are not handled.
- **Each frame rewrites the whole line,** so an expression with k operators
  takes about k × its length in bytes, on top of each operation's working.
- **About 2 extra layers' worth of heads are spent recomputing pointers**
  (pm, pm²) for every next-match class. Sharing them across classes is possible.

## 17. Roadmap

1. ~~Signed operands~~: done (§19).
2. ~~Decimals~~: done (§20).
3. ~~Division~~: done (§21).
4. ~~Integer powers~~: done (§23).
5. **Square root**, digit by digit, by the subtraction-only (Jarvis) method.
6. **Test tooling:** teacher-forced checks still run the full n² pass; long × / ÷ cases
   should decode with the cached path.

See `ANSWER_FIRST.md` for the analysis of emitting the answer with no working
(fixed-depth limits, parallel evaluation, looped transformers). Deferred.

## 18. Visualizer (`handformer/viz`)

`python -m handformer.viz` (port 8768) serves a step-through view, and
`tools/build_site.py` builds the same page as a static site (§24). One run is
the model's own greedy decode plus a trace of every position. **Step q is the
computation at position q that predicts byte q+1.**

- **Sequence strip:** bytes coloured by the phase or calculator that wrote
  them, with the step's pointers (p, c, o, a_lo, b_hi, F, S, Rb, …) as tags.
- **Step panel:** a plain-language reading of the step, the top logits, and
  the pointer values.
- **Attention heads:** every head's read position, what it added to the
  stream, and (on click) its hand-written score and value formulas.
- **Residual stream:** every named dimension, with the sublayer that wrote it
  and its value after every layer.
- **Active MLP units:** each unit that fired, with its formula.
- **Heatmaps:**
  - residual stream (dims × positions, after any sublayer);
  - this step (dims × sublayers);
  - attention of one head (rows read columns);
  - MLP activations (units × positions);
  - raw weights (every matrix, with named rows and columns).
- **Explanations everywhere:** hovering (or tapping) any dim, head, unit,
  byte, control or heatmap row name shows what it means (§25). Each heatmap
  has a "how to use this map" box.

Choices that are not obvious from the page:

- **"Focus active circuit" hides, it does not remove.** Hidden heads and
  units belong to calculators that are switched off at this step. They still
  run, but nothing they write reaches the logits unless their `act_*` flag is
  on. The heads panel says how many are hidden.
- **Each heatmap row is scaled by its own maximum.** Positions, flags and
  BIG-gated features differ by orders of magnitude, so a shared scale would
  show only the largest rows. Compare along a row, not between rows. Rows
  whose maximum is float noise (below 10⁻⁶) count as empty.
- **Positions after the current step are dimmed.** They had not been
  computed when this byte was written (causality).
- **What a head "added"** is computed from its full attention row, not just
  its top keys. The top keys are listed with ties in position order.
- **The attention map is sparse on purpose.** Most heads are pointers (§4.1),
  so each row has one bright cell. The blue line traces each row's peak:
  a vertical run is a pointer that holds, and a sideways step is a jump to a
  newer landmark. Row names read `33 4 → 9 [` ("character 33, a '4', reads
  position 9, a '['"). Reading position 0 (the start byte) means "no match
  yet". Gamma below 1 brightens weak attention.
- **Transport precision:** server heatmaps are sent as float16 (display
  only); weights as float32, because gating constants up to 10¹³ overflow
  float16. The exact float64 values are in the export (§22) and in
  `model.json` (§24).
- **Touch screens:** tap replaces hover. On a heatmap the first tap reads a
  cell and a second tap on the same cell does its click action. Cells are
  drawn at least 8 px wide and the map scrolls sideways inside its panel. On
  narrow screens the step controls are a fixed bottom bar. Canvases are capped
  at 16M backing pixels, because iOS Safari shows larger ones blank.

## 19. Signed operands (unary minus)

```
1+(3-7)*2=[3-7=6<-4]1+-4*2=[-4*2=80|80;-8]1+-8=[1+-8=3<-7]-7
-(1-4)=[1-4=7<-3]3            2*-3=[2*-3=60|60;-6]-6           -0*5=[-0*5=00|00;0]0
```

- **Unary vs binary `-`.** Layer 1 attention reads the previous byte's
  `is_digit` and `is_rpar`. A `-` is unary iff neither is set (line start,
  after `(`, after an operator, the answer sign after `;`/`<`). `is_binminus`
  is the exact 0/1 class `is_minus + prev_digit + prev_rpar − 1`. Every search
  for operators (`P`, `+-`, the continue check) uses binary operators only, so
  the answer's `-` no longer looks like an operator.
- **Signs are part of the operands.** In the parser: if the byte at `a_lo` is
  unary, a is signed and its fence is `a_lo − 1`. If the byte after the
  operator is `-`, b is signed and its end is the first non-digit after `o+1`.
  The frame copies the signed sub-expression.
- **Circuits work on magnitudes.** In the frame, `sa` and `sb` are the bytes
  after `[` and after the operator. The operand fences are `F+sa` and `P+sb`.
- **Routing.** `eff_sb = sb ⊕ [op is −]`. `act_add` = additive and
  `sa = eff_sb` (add magnitudes, sign sa). `act_sub` = additive and signs
  differ (subtract |a|−|b|, sign sa ⊕ borrow). `act_mul`: sign sa ⊕ sb. Each
  is one or two ReLUs over 0/1 flags.
- **Answer sign.** Each circuit's `sgn` is its final sign AND "result nonzero"
  (from its top-nonzero head's fallback), so `-0` never appears. It emits `-`
  at the end-of-working position, and digit indices shift by `sgn`.
- **Double negation.** If a negative result replaces bytes right after a
  unary `-`, the rewrite drops both signs (`cancel`).
- **Bug found while building it:** the "first `+`/`-` after p" key class was
  still `is_plus + is_minus`, so a leading unary `-` and the real operator
  both matched and attention averaged them. Every key class now uses
  `is_binminus`.

Tests: every signed `a op b` for a, b ∈ [−12, 12]; random trees with negative
leaves, negated groups and negative intermediate values; the reference's
answer is also checked against Python's own arithmetic.

## 20. Decimals

```
1.25+3.5=[125+350=574;475]4.75        0.1*0.2=[01*02=200,000|2000;2]0.02
0.5-1.25=[050-125=529<-75]-0.75       1.5+1.5=[15+15=03;30]3
3.14*(2-0.5)=[20-05=51;15]3.14*1.5=[314*15=0751,1740|01740;4710]4.71
```

The operator circuits are unchanged: they only ever see integers. All
decimal handling lives in the parser and rewriter (`expr.py`, layers 9–22):

- **Numbers.** A number is a run of digits and `.`. The non-number class
  `Lnd` excludes both, so operand bounds include the point. A `-` after a
  digit, `.` or `)` is binary.
- **Fraction lengths.** `a_dot` = latest `.` before o, inside a iff
  `a_dot > a_lo`, giving `frac_a = o − 1 − a_dot`; likewise for b from `b_hi − 1`.
- **Frame copy.** The frame is written in five segments: signed a without
  its `.`, zero padding, the operator, signed b without its `.`, zero
  padding. Each segment is a region flag. Skipping the `.` is one robust step
  ("from the dot's offset on, read one byte further"). For + and −, the pads
  make the fractions equal (`max − own`); × has no padding.
- **Result scale.** `f_res = max(frac_a, frac_b)` for + and −, and
  `frac_a + frac_b` for ×, picked with the `use_mul` select.
- **Formatting in the rewrite:**
  - trailing zeros: `tz = min(Rb − 1 − latest nonzero digit, f_res)`;
  - `f' = f_res − tz`, forced to 0 for a zero result;
  - `m' = digits − tz`;
  - width `w = max(m', f' + 1)` (zero-pad on the left);
  - emit the sign, the first `w − f'` digits, `.` if `f' > 0`, then the rest.

  The padding zeros and `.` are written directly; result digits are copied.
- **Integers are unchanged:** with no `.` every pad and `f_res` is 0, so the
  rewrite reduces exactly to the integer rule (checked: the integer suites
  pass byte for byte).

Tests (`tests/test_decimal.py`):
- all 432 pairs of 12 decimal literals under + − ×;
- formatting edge cases;
- fractions 60 digits long (+ −) and 6 digits (×);
- random decimal trees.

Every answer is also checked against exact `Fraction` arithmetic.

## 21. Division

```
1/3~5=[100000/3=30,60,90,21,51,81,12,42,72,000010310310310310310;33333]0.33333
22/7~10 → 3.1428571428      -7/2~1 → -3.5      12/4/3 → 1      2/3*3~4 → 1.9998
```

- **Syntax.** `expr~N=` gives every `/` N fractional digits, truncated toward
  zero. Without `~N`, N = 0 (integer quotient). N is read from the prompt (up to
  4 digits) by gathers in the frame locator, and the rewrite never copies
  `~N` forward. `/` has the same precedence as `*`, left to right.
- **Scaling in the frame copy.** a/b becomes `a·10^(frac_b+N) / b·10^frac_a`,
  and the result scale is N. The existing padding and point-placement logic
  handles it.
- **Circuit (`division.py`).** A uniform grid with stride s = len(B)+2:
  - rows 0–8 hold the multiples 1·B … 9·B (each the previous row plus B, with
    addition's recovered carry);
  - row 9 holds q₀ = 0 and the zero remainder;
  - each row after that holds qᵢ, then Rᵢ = 10·R_{i−1} + aᵢ − qᵢ·B (with
    subtraction's recovered borrow).

  Row and column come from the floor-division head.
- **qᵢ in one pass.** Each remainder digit, as it is written at column c,
  compares itself with M_k[c] for k = 1..9, storing gt/lt flags (remainder
  cells only). At the quotient-digit position, 9 latest-match heads find, for
  each k, the most significant differing digit. When all upper digits agree
  they fall back to the row start; then aᵢ vs M_k[0] = (k·b₀) mod 10 decides.
  q = Σₖ [R′ ≥ M_k].
- **Answer.** The q column is read from its first nonzero digit (newest-frame
  earliest; a dummy at row 9 falls back to the last digit for q = 0), in
  stride-s jumps using square-table products.
- **Bug found while building it:** the row-start fallback cell compared its own
  quotient digit and carried gt/lt flags, so "all equal" read as "greater".
  The flags are now gated to remainder cells.
- **Not supported:** division by zero (undefined output).

Tests (`tests/test_division.py`): every a/b for a < 31 and b < 13 at N = 0 and
3; signed and decimal operands at N = 0, 2, 5; long quotients (30÷3 digits,
20÷6 with N = 4, 8÷2 with N = 12); random trees with `/`. All are checked
against exact truncated rational arithmetic.

## 22. Raw weights and export

- **Visualizer:** Heatmaps → Raw weights shows any matrix with named rows and
  columns, as float32 (the gating constants overflow float16), coloured by
  sign·log(1+|w|).
- **Export:** `python -m handformer.export weights.npz` writes every tensor plus
  a JSON index of dimension, head and unit names.
- **Safetensors:** `python -m handformer.export weights.safetensors` writes the same
  tensors (dense float64, exact) as one file, with the index (dimension, layer and
  head names) and a how-to-run note in its `__metadata__`. It is written with
  numpy alone by following the format's spec, so exporting adds no dependency.
  `tests/test_export.py` checks a bit-exact round trip, that the official
  `safetensors` reader agrees when it is installed, and that the numpy-only runner
  computes correct answers from the file.
- **Density:** 8,478 nonzero of 2.08M stored cells (0.41%). The median residual
  dimension has one writer and one reader, and the median MLP unit reads 5
  dimensions. The sparsity is the price of one meaning per dimension. Planned
  compaction: time-share the private dimensions of the mutually exclusive
  operator circuits (about 576 → 300), and merge repeated building blocks.
- **Standalone runners:** `tools/run_weights.py` runs the `.npz` with numpy
  alone and never imports `handformer`. `tools/export_torch.py` writes a
  self-contained PyTorch package. With `--no-shuffle` it keeps the dimension
  order; by default it shuffles dimensions and drops names. `--harden` also
  rescales by powers of two and adds 64 inert decoy dimensions. Because the
  program is sparse and exact, an outside reader could still recover its
  circuits: they removed the decoys by dataflow and recovered the dimension
  names from the byte embedding and the output map. The next step would be
  decoys that are live but cancel exactly.

## 23. Integer powers

```
2^5=[2^5=;(2*2^4)](2*2^4)=[2^4=;(2*2)^2](2*(2*2)^2)=…=32
2^100 → 1267650600228229401496703205376 (1,350 bytes, all written by the model)
```

- **Grammar.** `^` is the highest precedence and right-associative: the
  parser chooses the latest `^` in the innermost group. Unary minus binds
  tighter, so `-2^2` = (−2)² = 4 (this differs from the usual maths convention).
- **A rewrite frame, not a computing frame (`power.py`).** The frame body is
  `;` plus one step of binary exponentiation written as text, which the rewrite
  splices into the line:

  | case | written |
  |---|---|
  | b < 0 | `(1/a^\|b\|)` |
  | b = 0 | `1` |
  | b = 1 | `a` |
  | b odd | `(a*a^(b−1))` (only the last digit changes) |
  | b even | `(a*a)^(b/2)` (halving is digit-local) |

  The base is copied verbatim (sign, digits, `.`). Each case is a list of
  segments (fixed bytes, copied spans, computed digits) laid out by region
  flags.
- **Parser changes.**
  - Power frames get no decimal normalisation.
  - The group's parentheses drop only for a plain-number result, or a result
    that is itself fully parenthesised (ends in `)`).
  - `cont` also fires when the result is an expression.
  - The group end is c = min(first `)` after p, Le), because a frame body can
    now contain `)`.
- **Stable frame operator.** A power frame's body writes `*` and `^`, which
  moved P ("latest binary operator"). The frame's operator is now Pf = P read
  at the frame's `=`.
- **Three bugs found while building it:**
  1. **Shadowed dims.** Private dims named `one` and `neg` shadowed the shared
     constant and borrow dims. `arith` now refuses any private name equal to a
     shared dim.
  2. **Unwrapped expansions.** `1/2^3` → `1/2*2^2` = 2 (wrong). The exact-value
     check in the tests caught it, though the model matched the reference.
  3. **A fixed generation cap.** The decode cache now grows as needed.

Tests (`tests/test_power.py`):
- a^k for a ∈ [−5, 5], k ∈ [0, 6];
- decimal bases and negative exponents with `~N`;
- precedence and right-associativity;
- 2^64, 3^40, 7^20, 12^12, 2^100;
- random mixed expressions.

All answers are checked against exact arithmetic.

## 24. Running in the browser (`viz/static/engine.js`, `local-api.js`)

The public demo is a static site: the page runs the model itself, so it needs
no server and costs nothing to host. GitHub Actions builds and deploys it
(`.github/workflows/pages.yml`).

- **`tools/export_web.py`** writes `model.json`, a list of every nonzero
  weight as `[row, col, value]` (8,478 entries, 130 KB). Python's float repr
  round-trips exactly through JSON into a JavaScript float64. It also writes
  `info.json`, the same model description the server returns: names,
  explanations, formulas and wiring.
- **`engine.js`** is a ~270-line forward pass that knows nothing about
  arithmetic. It applies the same rules as `model.py`:
  - byte embedding plus the `pos` and `pos²` features;
  - every head reads the same layer input, with unscaled q·k and a causal
    softmax, and the heads' outputs are summed into the stream;
  - then the ReLU MLP;
  - float64 throughout; no LayerNorm and no biases.
- **Values are cached per head** (v = x·W_V) as well as keys, so a new token
  costs O(n · head width), not O(n · d). Mathematically (A·X)·W_V = A·(X·W_V);
  the order of float sums differs, so values match within float rounding
  (the parity test requires 10⁻⁶) and the argmax bytes are identical.
- **One decode, no second pass.** While decoding, the engine records a compact
  trace for every position: the residual entries each sublayer changed (their
  exact new values), queries, keys and values, the top attention keys, and the
  positive MLP pre-activations. That answers every visualizer request,
  including the full attention row of any head at any step. The server
  instead re-runs one traced forward pass.
- **`local-api.js`** answers the same `/api/*` requests as the Python server,
  so `app.js` is one page for both. `window.HANDFORMER_STATIC` (set by
  `build_site.py`) picks the in-page model.
- **Host-side code is framing only:** the input character check, the prompt
  `\n<expr>=`, argmax, and stopping at `\n`. These match the server.
- **Parity test (`tests/test_web.py`)** runs `local-api.js` under Node and the
  Python runner on the same requests. The generated bytes must be identical,
  and every traced number (step values, residuals, head writes, logits,
  attention, MLP activations, weight views) must agree within 10⁻⁶. Exact ties
  between attention keys, and rows that are pure float noise, are compared by
  value rather than by index. CI runs it on every push, and the Pages deploy
  runs it before building.
- **Speed:** a 300-byte product decodes in about 0.3 s, and a 925-byte power
  expression in about 1.2 s.

## 25. Reading the names

Every dimension's explanation comes from `viz/glossary.py`: a table for the
shared dimensions, per-calculator tables, and these name patterns. Names
built from a landmark also spell out what the landmark is.

| name | meaning |
|---|---|
| `F`, `Rb`, `E`, `S`, `R`, `P` | position of the latest `[`, `]`, `=`, `;`/`<`, `\|`, operator |
| `Lg`, `Lrp`, `Lmul`, `Lpow`, `Ladd`, `Lnd`, `Ldot`, `Lnz`, `Ptil` | latest `(`/`]`, `)`, `*`/`/`, `^`, `+`/binary `-`, non-number byte, `.`, nonzero digit, `~` |
| `f_X` | landmark tag for X: nonzero only on X's bytes and growing with position, so the "latest X" head finds the newest |
| `PM_X`, `PM2_X` | latest X strictly before this byte, and its square (forward search, §4.3) |
| `Lt_X` | latest X at the group start (where a forward search begins) |
| `ohD`, `is_*` | this byte is digit D / this byte class |
| `D0`–`D9`, `O_*` | votes for the next byte, read by the unembedding |
| `act_op`, `typ_op` | the op calculator is switched on / this frame's operator is op |
| `Y2_*`, `Y4_*` | the byte being copied into the frame / into the new line |
| `ns.name` | a private value of calculator `ns` (`mul.J`, `div.m3`, …) |

**Source lines.** Every `HeadSpec` and `UnitSpec` records the line of circuit
code that created it (`model.authored_at`, skipping the shared helpers and
private `_helper` functions). A dimension's source is the line of each head or
unit that writes it, so `inB` links to `expr.py`, where the frame-copy flags
are built. The visualizer shows these as "view source" links, and the docs use
the same data through `tools/doclinks.py` (§27). Like labels, they are
documentation only.

Unit explanations come from labels attached where each unit is built (for
example "fallback for 'f_R': the start byte scores 1, so 'no match yet'
reads position 0"). Labels are documentation only and never change a weight.
Every dimension has a specific doc, every unit a hand-written label, and every
head a description and a source line. `tests/test_explanations.py` enforces
this in CI, so a new part without an explanation fails the build. The
calculators' private dims are documented in `viz/op_dims.json`.

## 26. Things that look wrong but aren't

A review list for anyone reading the model or the visualizer. §27 links each mechanism to its source lines.

1. **Some layers have no attention heads** (13–16, 18–21): see §9.
2. **Reading position 0 is often deliberate.** The start byte is the
   "nothing found yet" answer of every latest-match head, so early in a line
   many heads read it (about 11% of all head reads in a typical run).
3. **Attention is nearly one-hot.** Scores are unscaled (no 1/√d) and use
   β = 30–60, so softmax acts as an exact lookup. A fractional pointer means
   two keys tied and were averaged, which the features are designed to
   prevent (for example `2(pos+1)` rather than `pos+1`, §12).
4. **Inactive circuits still compute.** Routing is gating (§5.2): every
   circuit runs at every position, and `act_*` decides whether its outputs can
   fire. That is why one merged model needs no dispatch.
5. **Huge constants (10⁶, 10¹³) are gates, not values.** `−BIG·(1 − flag)`
   switches a unit off. Flags are exactly 0 or 1 (robust steps), so BIG never
   amplifies noise into an answer.
6. **Working digits are least-significant first, the answer is
   most-significant first.** The carry chain runs from the low digit, and the
   answer is copied out in reading order by attention. Nothing is reversed by
   the host.
7. **Every frame rewrites the whole line,** so output grows roughly with
   (number of operators) × (line length), plus each operation's working.
8. **`~N` truncates, it does not round.** `2/3~3` → `0.666`.
9. **`-2^2` is 4:** unary minus binds tighter than `^` (§23).
10. **The page writes the explanations, not the model.** They are computed in
    JavaScript from the model's raw values and only describe them; the model
    never sees them.
11. **The length limit is float64, not the weights:** exactness holds to about
    10⁵–10⁶ positions (§12).
12. **The browser and Python agree to 10⁻⁶, not bit for bit,** because the
    float sums run in a different order. The bytes are identical (§24).
13. **Some reads are not exact where nothing reads them** (37 known cases,
    §28). The claims test pins them down so that no new one can appear unnoticed.

## 27. Where each circuit is written

Each circuit is a short list of named heads and units, placed at fixed layers with `b.at(i)`. The links are generated: `docs/links.toml` names each location by a unique line of text, and `python tools/doclinks.py` rewrites every `[](src:KEY)` (or `[](src:dim:NAME)` for any residual dimension) to its current line. CI fails if a link goes stale.

| what | where |
|---|---|
| **Engine** | |
| weights from named linear forms | [model.py:185](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/model.py#L185 "src:engine.builder") `ModelBuilder`: `{dim: coef}` forms become matrices |
| a head = query, key, value forms | [model.py:91](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/model.py#L91 "src:engine.head_spec") `HeadSpec` |
| an MLP unit = ReLU(read) → write | [model.py:127](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/model.py#L127 "src:engine.unit_spec") `UnitSpec` |
| the forward pass | [model.py:333](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/model.py#L333 "src:engine.forward") `Transformer.residual` |
| cached greedy decoding | [model.py:408](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/model.py#L408 "src:engine.decode_cache") `_DecodeCache` |
| **Primitives** | |
| exact lookup: score −β(k−t)² | [primitives.py:66](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/primitives.py#L66 "src:prim.gather") `gather_head` |
| latest match (tags grow with position) | [primitives.py:142](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/primitives.py#L142 "src:prim.latest_feature") `latest_feature` |
|  | [primitives.py:220](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/primitives.py#L220 "src:prim.latest_head") `latest_head` |
| first match after a point (forward search) | [primitives.py:183](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/primitives.py#L183 "src:prim.next_match") `next_match_head` |
| x·y from a table of squares | [primitives.py:123](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/primitives.py#L123 "src:prim.product") `product_form` |
| floor(x/s) by attention over a ruler | [primitives.py:97](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/primitives.py#L97 "src:prim.floor_div") `floor_div_head` |
| exact 0/1 steps that reject noise | [primitives.py:291](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/primitives.py#L291 "src:prim.clamp01") `clamp01_units` |
| gating with −BIG·(1−flag) | [primitives.py:330](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/primitives.py#L330 "src:prim.gated") `gated` |
| digit of x mod 10 by ReLU ramps | [common.py:119](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/common.py#L119 "src:prim.ramp_mod10") `ramp_mod10_units` |
| byte embedding | [common.py:56](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/common.py#L56 "src:prim.embed") `embed` |
| **Frame locator (layers 1–4)** | |
| unary vs binary '-' | [frame.py:39](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/frame.py#L39 "src:frame.unary_minus") frame.py |
| latest-marker heads F, Rb, E, S, P, … | [frame.py:124](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/frame.py#L124 "src:frame.latest_heads") layer 2 |
| precision N read from the prompt | [frame.py:127](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/frame.py#L127 "src:frame.precision") layer 2 |
| routing: which calculator runs (sign XOR) | [frame.py:193](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/frame.py#L193 "src:frame.routing") layer 4 `act_*` |
| **Parser and rewriter (layers 4–22)** | |
| find the innermost group | [expr.py:94](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/expr.py#L94 "src:parser.redex") layers 4–5 |
| group end c = min(first ')', line end) | [expr.py:114](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/expr.py#L114 "src:parser.group_end") layer 5 |
| precedence: choose the operator o | [expr.py:131](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/expr.py#L131 "src:parser.precedence") layer 7 |
| operand bounds a_lo, b_hi | [expr.py:144](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/expr.py#L144 "src:parser.operands") layers 8–9 |
| drop the parentheses | [expr.py:211](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/expr.py#L211 "src:parser.drop_parens") layer 11 |
| cancel −(−v) | [expr.py:222](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/expr.py#L222 "src:parser.cancel") layer 11 |
| decimal alignment and result places | [expr.py:244](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/expr.py#L244 "src:parser.decimals") layers 13–14 |
| frame copy segments | [expr.py:274](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/expr.py#L274 "src:parser.frame_copy") layers 14–16, copied by the L17 head |
| result formatting and the rewrite | [expr.py:323](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/expr.py#L323 "src:parser.rewrite") layers 17–21 |
| output votes | [expr.py:396](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/expr.py#L396 "src:parser.outputs") layer 22 |
| **Calculators** | |
| addition: whole circuit (~60 lines) | [addition.py:32](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/addition.py#L32 "src:add.circuit") layers 3–5 |
| addition: carry recovered, never stored | [addition.py:83](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/addition.py#L83 "src:add.carry") layer 4 |
| subtraction: borrow and sign | [subtraction.py:101](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/subtraction.py#L101 "src:sub.sign") layer 4 |
| subtraction: top digit and complement on copy | [subtraction.py:87](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/subtraction.py#L87 "src:sub.complement") layers 4–7 |
| multiplication: squares table gathers | [multiplication.py:95](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/multiplication.py#L95 "src:mul.squares") layer 4 |
| multiplication: line index j = floor(x/s) | [multiplication.py:124](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/multiplication.py#L124 "src:mul.line_index") layer 5 |
| multiplication: carry | [multiplication.py:210](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/multiplication.py#L210 "src:mul.carry") layer 9 |
| division: digit vs all nine multiples | [division.py:172](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/division.py#L172 "src:div.compare") layer 7 |
| division: nine heads pick the quotient digit | [division.py:237](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/division.py#L237 "src:div.quotient") layer 9 |
| power: the five rewrite cases | [power.py:50](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/power.py#L50 "src:pow.cases") `_segments` |
| power: exponent parity | [power.py:107](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/power.py#L107 "src:pow.parity") layer 5 |
| **Merging** | |
| outputs gated by act_* (no dispatch) | [arith.py:72](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/arith.py#L72 "src:merge.gating") `_Namespace.unit` |
| private dims may not shadow shared ones | [arith.py:110](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/arith.py#L110 "src:merge.no_shadow") `builder` |
| **Example dims** | |
| `inB`: this byte copies operand b into the frame | [expr.py:298](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/expr.py#L298 "src:dim:inB") |
| `f_R`: landmark tag for the latest `\|` | [frame.py:115](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/frame.py#L115 "src:dim:f_R") |
| `mul.carry`: the multiplication carry | [multiplication.py:211](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/multiplication.py#L211 "src:dim:mul.carry") |

## 28. Checked claims (`tests/test_claims.py`)

Each explanation is a claim about what a part computes. The test runs 24
traces (every operator, signs, decimals, `~N`, powers, nested parentheses)
and checks three kinds of claim at every position:

| claim | where it comes from | checks |
|---|---|---|
| a head reads one exact value | every head (scope: `HeadSpec.exact_when`, else where its calculator is on, else every output position) | about 81,000 head reads |
| a yes/no step never sees an input on its ramp (within ¼ of an integer) | `clamp01_units` / `hat_units`, `("outside", -1, 1)` | about 237,000 |
| a calculator's outputs are silent while it is off | `arith` gating, `("silent_unless", "act_op")` | about 418,000 |

**Result.** Silence holds everywhere. The other two hold everywhere except 37
known cases (23 heads, 14 steps), listed with their diagnosis in
`tests/claims_known.json`. The test fails on any new violation, and on a
listed one that no longer occurs. No known case changes an answer. Each is a
value that nothing reads at that position:

- **Searches with no match yet.** "Top nonzero digit" (add, sub, mul) and the
  division remainder keys use position tags with no start-byte fallback, so
  before the first match the head averages every key. Their readers run only
  in the answer phase, gated by `act_op`.
- **Forward searches that find nothing.** With no `)`, `*`/`/` or `+`/`−` after
  p, the search averages and its found-flag is about 0. `c` and choice 2 of
  `o` are gated by that flag. Choice 3 is not, so `o` is fractional at the
  prompt's `=`, where the only output is `[`. The seven gathers at o ± 1
  inherit this.
- **Prompt bytes.** `phase4 = 1 − open − is_eq` is also 1 before the prompt's
  `=`, so rewrite steps run on undefined operand bounds there. Prompt
  positions are teacher-forced and never sampled.
- **Position 0 and out-of-frame lookups.** The multiplication `sep_slot`
  lookup targets a position after the start byte, and the division borrow
  outside a division frame looks up a negative position. Both readers are
  silenced.

The fixes are known (a start-byte fallback on those position tags, an exact
found-flag gating choice 3, `phase4` gated by `seen_eq`, `ND` excluding the
start byte) but are not applied: the model is correct as it stands, and the
list keeps the exceptions explicit.

## 29. Knockouts (`handformer/ablate.py`)

A knockout zeroes a part's writes in the weights: a head's W_O, a unit's row of
W_out, or every write into a dim. The forward pass is unchanged, so the
visualizer, the browser engine (`local-api.js`, the same three rules) and the
tests all work on the knocked-out model. `tests/test_web.py` checks the two
engines agree. In the page, "knock out" buttons sit on the selected head, an
opened dim and every active unit. The bar shows the answer with and without
the knockout. Example: removing the six addition carry units turns `47+38`
into 75, and leaves `12+34` (no carry) unchanged.

## 30. Circuit graph and the hello-world model

- **Circuit graph** (`viz/static/graph.js`): for one circuit, every dim it
  reads or writes, placed in the column of the first layer that writes it,
  with an edge for each head (blue) or unit (orange) that reads one and writes
  the other. It is generated from the specs, so it cannot drift from the
  weights. `one`, `pos` and `pos²` are left out because they feed almost
  everything.
- **All circuits together** (the default view): one box per circuit, joined
  by the residual values one circuit reads from another (arrow width = how
  many; hover for the list), plus a 22-layer strip of each circuit's heads and
  units. It shows a design fact that is easy to miss: **the parser and the
  calculators share no residual values.** They cooperate only through the text
  the model writes. The parser copies the sub-expression into the frame as
  bytes, which the calculator reads, and the calculator's result bytes are
  copied back by the rewrite (dashed arrows). The frame locator is the only
  circuit every calculator reads.
- **Hello world** (`handformer/hello.py`, `HELLO.md`): addition alone in 5
  layers (9 heads, 66 units, 761 weights), built from the same primitives,
  with a layer-by-layer walkthrough and a knockout that breaks the carry chain
  exactly as the explanation predicts. `tests/test_hello.py` checks every pair
  below 100 and long random operands.

## 31. A second model: Game of Life (`handformer/life/`, `LIFE.md`)

The same engine and primitives build a Game of Life model (5 layers, 16 heads, 41 units, 641
nonzero weights). The sequence is `\nN:<grid>=` followed by blocks `k:<grid>=` counting down to
0, then `\n`; the count is kept in the text. Because every block has the length L of the prompt
block, the cell written at p reads its 3×3 neighbourhood at fixed offsets p − L + dr·W1 + dc with
nine exact lookups; left and right edges read the dead `/` `:` `=` bytes, top and bottom edges
are two exact steps on the cell's offset, and row ends are copied from the previous block, so no
division or modulo is needed. The rule is two equality tests, [n = 3] + [n = 2 and alive].

The visualizer shows both models (`viz/models.py`: each model brings its builder, input check,
circuit grouping, pointers and docs; everything else is shared). Life adds a Board panel and
its own step explanations, all derived from the written text and the model's own values.
Unlike the calculator, Life has no known claim exceptions (`tests/test_life.py`).

## 32. A third model: a C syntax highlighter (`handformer/csyntax/`, `CSYNTAX.md`)

The lexer is a table of 47 rules (`RULES`), covering C11's lexical grammar (prefixes, splices, UCNs, the `%:` and `??=` directive spellings, the `??/` backslash). The prompt is the source and an end marker; for each
source byte the model writes a state letter. The state is the model's own previous letter (the
current byte at each output step), so each rule is an AND of 0/1 conditions on that letter and
on exact lookups of the source byte and its neighbours, one unit with its own dim; fallback rules
are their group minus their siblings, so exactly one rule dim is 1 at every byte (checked in
`tests/test_csyntax.py`). Rule dims vote for their letters directly in the unembedding. Keywords:
one AND unit per C11 keyword over gathered letters at the word's first byte. Because the start
byte is a newline like those in the source, the "no match yet" fallback of its landmark tags is a
position test (ReLU(1 − pos)) instead of `is_bos`.

## 33. A fourth model: a parser (`handformer/cparse/`, `CPARSE.md`, `docs/PARSER_DESIGN.md`)

An LL(1) parser for a C-shaped toy grammar, stage 1 of a C89 syntax checker. A grammar is data
(EBNF bodies); `grammar.py` compiles it to minimal automata whose states are the items, and to
an LL(1) table; `circuit.py` turns any table into weights. The output is one fixed-width record
per action (action, symbol, depth, pointer, item, return item), because depth and input position
are counts and attention cannot count: each step gathers the previous record's fields at fixed
offsets from the latest `|`. The lookahead is a forward search (`next_match_head`: the token
start whose previous token start is the pointer). One rule per table entry and one default per
item (close or error) fire as AND units, exactly one per byte. A close pops the stack with one
head: the latest open record whose depth equals the current depth, found by the score
A·(d² − (d_k − d)²) + C·pos over keys that are the `|` of open records (gated depth, depth² and
position), exact for depth ≤ 999 and sequences ≤ 10⁵ bytes. The byte to write is a code in the dim
`out` (each field a gated sum: the rule's constant or a digit chosen by its flags, digits of
depth ± 1 and of the lookahead position by `floor_div_head`); the unembedding is the quadratic
K·(2·b·out − b²), so the nearest byte code wins. Decoding caches each head's values (x·W_V) as the
JS engine does, which the parser's long outputs need.
