# handformer: design

A decoder-only transformer for arithmetic. Every weight is written by hand and
nothing is trained. The tokenizer is raw bytes (vocab 256). One set of weights
evaluates whole expressions (`+ - *`, parentheses, precedence) on non-negative
integers of any length.

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
| 2026-11-03 | Precision goes in the prompt: `<expr>~N=` | Not built yet; the model will read N |
| 2026-11-03 | Full expressions with precedence and parentheses | Term rewriting in frames; operands addressed by bounding bytes |
| 2026-11-03 | Powers: integer exponents; roots: digit-by-digit square root | Not built yet |
| 2026-11-03 | Operator selection must be in the weights, not separate models | One merged model; circuits are gated by flags computed in the weights |
| 2026-11-03 | Use the residual stream for as much work as possible | Position arithmetic (products, division) is done with attention instead of extra tokens |

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

All dimensions are named (249 before signed operands; 263 now). They are allocated by category:

| category | dims | contents |
|---|---|---|
| input | 33 | `one`, `pos`, `pos²`, digit value, digit one-hot, byte-class flags |
| shared frame locator | 43 | marker positions, frame flags, phases, `act_*`, previous-match pointers |
| parser / rewriter | 62 | p, c, o, operand bounds, region flags, source positions, copy one-hots |
| addition (private) | 15 | digit targets, gathered digits, carry, answer pointer |
| subtraction (private) | 24 | same, plus top/lowest-digit pointers and complement value |
| multiplication (private) | 38 | squares, line index j, column m, b_j one-hot, products, carry |
| output | 34 | one dimension per emitted byte class, read linearly by the unembedding |

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
- Each head stores a fused value/output matrix (`d × d`). That inflates the
  dense size (see §14) but not the number of real parameters.

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

| layer | attention | MLP |
|---|---|---|
| 1 | — | latest-match key features for every marker class |
| 2 | latest heads: `[` `]` `=` `;` `\|` op, `(`/`]`, `)`, `*`, `+-`, non-digit | frame flags: `open`, `seen_eq`, `seen_semi`, `seen_bar` |
| 3 | line start `Ls`, innermost group `p`, previous-match pointers, op type at P, sign at S, (P−F)² | `act_add/sub/mul`, phases; operator targets and key features |
| 4 | pm² ×4, latest-at-p ×3; operand-digit gathers; × squares and previous line | carry / borrow; × collect flags |
| 5 | next `)` / `*` / `+-` after p; × floor-division head; answer gathers | select `c`; + digits and `]`; − digit and top; × collect source |
| 6 | × (j+s)², j², b_j one-hot, collect digit; − answer digit | `use_mul`; × a-digit targets, separator flags |
| 7 | × a_m, a_{m−1} | select `o`; × digit products; − complement |
| 8 | `a_lo`, `b_hi`, operator-count gathers | paren-drop equalities, `cont`; × ⌊y/10⌋; − answer digits |
| 9 | frame copy gather | `drop`, `cont`, end of frame copy; × carry |
| 10 | — | rewrite region flags; × digit ramps |
| 11 | — | rewrite source position |
| 12 | rewrite copy gather | all rewriter outputs |

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

### 9.2 Choosing and rewriting the sub-expression (`expr.py`, layers 4–14)

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
| layers | 14 |
| heads | 73 |
| MLP units | 462 |
| d_model | 263 |
| nonzero parameters | 3,774 |

The dense size is much larger, mostly zeros in the per-head `d×d` value/output
matrices. A factored per-head value/output layout would shrink it without
changing the nonzero count.

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
- Run all: `../.venv/bin/python -m unittest discover -s tests -t .`
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
2. **Decimals** for + and −: align on `.` with marker lookups. For ×, place the
   point at frac(a) + frac(b).
3. **Division** (`/`) by long division: repeated subtraction lines reusing the
   borrow and complement circuits, with precision from `~N` in the prompt.
4. **Integer powers** `a^n`: repeated multiplication frames (a negative n becomes division).
5. **Square root**, digit by digit, by the subtraction-only (Jarvis) method.
6. **Test tooling:** teacher-forced checks still run the full n² pass; long × / ÷ cases
   should decode with the cached path.

See `ANSWER_FIRST.md` for the analysis of emitting the answer with no working
(fixed-depth limits, parallel evaluation, looped transformers). Deferred.

## 18. Visualizer (`handformer/viz`)

`python -m handformer.viz` (port 8768) serves a step-through view, modelled on `gpt1viz`.
One run is the model's own cached greedy decode plus one traced forward pass
(`Transformer.residual(..., trace=...)`, the same computation the tests check);
step q is the computation at position q that wrote byte q+1.

* sequence strip coloured by phase/operator, with the step's pointers (p, c, o, a_lo, b_hi, F, S, Rb) as tags
* plain-language reading of the step, top logits, every head's read position and what it wrote,
  its hand-written score/value formula, the named residual stream, active MLP units with formulas
* heatmaps: residual stream (dims × positions, per-dim scaled, any snapshot), this step
  (dims × sublayers), full attention pattern of the selected head, MLP activations (units × positions)
* "focus active circuit" hides circuits that are inactive at the current step; future positions are dimmed

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
