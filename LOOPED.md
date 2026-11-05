# The recurrent-depth calculator: the answer is the first byte it writes

`handformer/loop` is the calculator as a recurrent-depth decoder. The model is autoregressive
and causal: one position at a time, a KV cache, no position reading a later one. What loops is
depth: each position runs the same block of layers again and again, as often as its own router
asks. It takes the same expressions as the frame-writing calculator (`+ - * / ^`, unary minus,
brackets, precedence, signed decimals, `~N` precision, spaces, any length) and writes nothing
but the answer:

```
\n2*(3+4)-1=13\n
\n355/113~6=3.141592\n
\n2^64=18446744073709551616\n
```

Every weight is written by hand from a named formula, as in the rest of the repository.

    python -m handformer.loop "2*(3+4)-1"              # 13
    python -m handformer.loop --steps "12*34-400"      # what the thoughts held, step by step
    python -m handformer.viz                           # the visualizer: pick it in the menu

`REWRITE.md` describes an earlier looped model that rewrote the prompt in place. Its block had
heads that read later positions, so it was not a decoder. It is kept, under the name `rewrite`,
but is not offered on the site.

## 1. The model

| part | what it is | here |
|---|---|---|
| prelude | the input embedding `e` of a position | a byte's row, or a latent input (below) |
| recurrent block | one stack of causal layers, applied again and again to the same position | 18 layers |
| adapter | a fixed linear map from the block's last residual to the next iteration's state | carries a position's loop registers |
| router | a dim the block computes: 1 sends the position through the block again | `route` |
| KV cache | a later position reads an earlier one through the earlier one's last iteration | per layer |
| coda | the unembedding of the last residual | the next byte |
| latent map T | when `think` is 1, the next position's input is `T(last residual)`, not a byte | one linear map |

Iteration k of a position reads `e + s_k + positional features`, with `s_0 = 0` and
`s_{k+1} = x_k · A`. The input is injected at every iteration. Attention is causal: the earlier
positions through their cached last iterations, and the position itself at this iteration.

The host is an ordinary decoding loop. It feeds the prompt, takes the argmax and stops at the
end byte. It also follows two bits the weights compute: `route` (run this position through the
block again) and `think` (make the next position a thought). It never reads or changes what
the positions hold.

## 2. How this matches current models

- **Recurrent depth with input injection.** Huginn (Geiping et al. 2025) iterates one block per
  token, re-injecting the embedding each time, and adapts its state between iterations with a
  linear map. This model does the same.
- **A per-token router.** Mixture-of-Recursions (Bae et al. 2025) and Ouro's exit gates (Zhu et
  al. 2025) let each token choose its own depth. Here a router dim does that, and it is a real
  per-token choice: a copy cell exits after one iteration, a product column's predictor runs
  once per term, a quotient digit's once per column.
- **KV sharing across depths.** A token that exited early is read at its last iteration, as in
  those models' cache sharing.
- **Latent thoughts.** Coconut (Hao et al. 2024) feeds the last hidden state back as the next
  input instead of a token. Here T maps the last residual to the next input, and those
  positions are never written out.

Why thoughts and not depth alone: under a causal mask, a position only ever knows the positions
up to itself. The root `=` sees the whole expression, but it is one position with a fixed number
of dims, so it cannot hold `2^200` or a long remainder exactly. Exact, arbitrary-length,
answer-first evaluation therefore needs extra positions before the answer. The model makes them
itself, as latent positions.

## 3. What the thoughts hold

After the prompt the model thinks in steps. A step reads the latest copy of the expression (the
prompt's, at first) and writes the next one:

| position | what it holds | who writes it |
|---|---|---|
| root `=` | the end of a copy; it parses it and plans the step, or halts | the copy's last cell |
| start | the plan: the `C_` dims (operands, columns, signs, where the copy is replaced) | the root |
| working | reduce steps only: one position per column of the result (§4) | the start, then each column |
| copy cells | the next expression, one byte per position (a byte's embedding plus registers) | the position before |

Each position predicts the next one, as in any decoder. The difference is that the prediction is
a vector: hold dims (exact 0/1, "the next byte is b", which T turns into that byte's embedding)
and registers (`n_<reg>`, which T moves to `<reg>`). The plan rides along unchanged: T maps every
`C_` dim to itself, and only a root writes a new plan (new value minus the one it was handed).

When the root finds a number, it halts: `think` is 0, and it writes the copy's first byte. Each
written byte then copies the next cell, until the root, which ends the answer with `\n`.

```
python -m handformer.loop --steps "2*(3+4)-1"
  step   1: '2*7-1'  (9 positions, 9 iterations)
  step   2: '14-1'  (8 positions, 8 iterations)
  step   3: '13'  (7 positions, 7 iterations)
2*(3+4)-1=13
```

### 3.1 The root's parse (parse.py)

The frame-writing calculator's rules: spaces first (one step removes them all), then a group
holding one number loses its brackets (`-(-v)` loses both signs), then the redex: innermost
group, the latest `^`, else the first `*` `/`, else the first `+` `-`.

A copy is the run of cells before its root, after `fz`, the latest position that is not a cell.
Every "latest X" lookup could land in an older copy, so each is clamped to `fz`. The group's end
`c` is the first `)` after `p`, or the copy's end: the root, or the prompt's `~`. Precision `N`
is read from the prompt every time.

The plan dims are written only for the step kind that uses them. A spaces step's parse reads
garbage, and none of it reaches the thoughts.

### 3.2 Writing a copy (walk.py)

A copy is a walk over the old copy, carried from cell to cell in registers: the zone (before or
inside the replacement; after it is neither), `ptr` (the old position copied) and `mk` (the index
in the replacement). Each step of the walk:

- `ptr1` is the first non-space after `ptr`, found with one next-match lookup;
- if it reaches the first replaced position, the replacement starts;
- when the replacement's last byte is written, the walk jumps to the old position after the
  replaced ones;
- when it reaches the old copy's end, the next position is the new root.

The replacement is the group's number (brackets), a power's template (§4.5), or the working's
result. A result's shape comes from the working's last column, the position before the copy:
its top nonzero column, its lowest nonzero column right of the `.`, and the final borrow. The
writer formats it from that shape: the sign, then the integer digits, then `.`, then the
fraction without trailing zeros.

## 4. The working

### 4.1 `+` and `-` (columns.py)

There is one position per column, least significant first, `W = max(ia, ib) + F + 1` of them.
The position before a column computes it in one iteration:

- it gathers a's and b's digits for that column (a's at `C_PA − j`, one further right past the `.`);
- it takes the carry or borrow from its own registers;
- it writes the digit and the carry out.

A subtraction keeps both `|a|−|b|` and `|b|−|a|`. The final borrow picks one when the copy is written.

### 4.2 `*` (columns.py)

Column `c` of `|a|·|b|` is `cm + Σ_i A_i·B_(c−i)`, where `cm` is the carry in. The position that
computes it iterates once per term. Iteration k adds term `i0 + k` into a two-part accumulator:
`slo`, a digit, and `shi`, its tens. The adapter carries both. The router sends the position
round again while terms are left.

The last iteration writes the digit and the carry out, `floor(cm/10) + shi + [cm mod 10 + slo ≥ 10]`.
`floor(cm/10)` comes from a ruler head. The carry is at most 9 per term, so it stays below the
position, which the ruler head needs.

### 4.3 The router at work

| expression | latent positions | iterations | deepest position |
|---|---|---|---|
| `2*(3+4)-1` | 24 | 26 | 1 |
| `12*34-400` | 20 | 22 | 2 |
| `123456789*987654321` | 38 | 120 | 9 |
| `355/113~6` | 55 | 90 | 4 |
| `1/7~30` | 127 | 190 | 2 |
| `2^64` | 221 | 343 | 10 |

### 4.4 `/` (division.py)

Long division, with A = |a|'s digits followed by fb + N zeros and B = |b|'s digits followed by fa
zeros. For each digit t of A, the model writes two things:

- **A quotient digit q_t.** The position before it iterates once per column of `x = 10r + A_t`.
  It compares x with `d·B` for every d = 1..9 at once, keeping a carry and a borrow per d as loop
  registers. After the top column, q is the number of d with no borrow out.
- **The remainder `x − q_t·B`.** This takes one position per digit, each with one iteration.

Then a quotient row of L columns is written, and the copy is written from it as for the other
operators. A division by zero is not evaluated: the root halts, and the answer is the expression
that is left.

### 4.5 `^` (power.py)

There is no working. The copy replaces `a^b` with one step of binary exponentiation, as the
frame-writing calculator does: `(1/a^n)`, `1`, `a`, `(a*a^(b−1))` or `(a*a)^(b/2)`. The template
has eleven segments. The root plans where each segment starts and where each copied segment
reads, and each cell of the walk writes one fixed byte, one copied byte, or one computed digit
of `b−1` or `b/2`.

## 5. Exactness

Every flag, hold and register is a robust step or an exact sum of integers. A gathered digit is
re-quantized before any arithmetic, because a gather leaks about 1e-13 of its neighbours. So
every latent position's input is integers to within 1e-9, however long the run is; the tests
check this on every position of every run.

Two places needed care:

- **The group's end.** It was first a minimum with a BIG-scaled "found" flag. That flag leaked
  1e-6 into the plan, so the end is now a forward search in which the copy's end also counts as
  a match.
- **The plan.** It is written only for the step kind it belongs to.

## 6. Size and speed

| | |
|---|---|
| residual | 543 dims |
| block | 18 layers, 74 heads, 1,509 MLP units |
| nonzero weights | 8,534 (embedding, block, unembedding, adapter, T) |
| one iteration | about 2 ms (numpy, a few hundred positions) |
| `2^64` | 221 latent positions, 0.6 s |

## 7. Tests and the visualizer

`tests/test_loop.py` decodes every position of a run and compares it with
`tests/loop_reference.py`, a separate plain-Python account of the steps. It checks:

- every copy of the expression;
- every working digit, quotient digit and remainder;
- the router's depth at every position;
- the answer, against the reference, the frame-writing calculator and exact `Fraction` arithmetic;
- that nothing is written before the answer;
- that no latent input drifts from integers.

The cases cover hand-picked expressions, spaces, decimals, powers, division, the claims corpus,
random trees, 80-digit operands and deep nesting.

In the visualizer a run is an ordinary decoder run: every position at its last iteration through
the block. That is exactly the computation later positions read, and `test_web.py` checks that
the in-page engine gives the same run.

- Positions are coloured by what they are: prompt, root, start, column, quotient, remainder,
  copy, answer.
- The Depth bar lists the positions the router sent round more than once. Clicking one shows it
  at each of its iterations.

## 8. Limits

- Every step writes a whole new copy, so long expressions cost positions. The cost is linear
  per step, like the frame calculator's working, but none of it is shown.
- Precision `N` is at most 9999.
- Division by zero halts with the expression unevaluated rather than an error byte.
