# Answer-first: can the model emit nothing until it has the answer?

Status: analysis only, for later review (2026-11-03). Nothing here is built.
The current model (see `DESIGN.md`) writes its working into the token stream
before the answer. This note asks what it would take for the **first emitted
byte to be the first digit of the answer**, and what stops us.

## 1. Only one position sees the whole problem

Attention is causal: a position sees only what came before it. In the prompt
`\n123+4567=`, the digit `4` cannot know whether it is the thousands or the
hundreds digit, because b's length arrives later. The only position that
sees the whole problem before the first answer byte is the final `=`.

With no emitted working, the whole computation for the first answer digit
happens in **one forward pass at one position**: a fixed-size residual
stream, a fixed number of layers, a fixed number of attention reads per
layer. Every earlier position can only have computed things about its own
prefix. The working bytes we emit today are exactly what gives the model
more positions that see everything.

## 2. Digits of the same significance can't be paired in the prompt

Carry and borrow recovery need, at some position, both digits of equal
significance together (a_j + b_j). No prompt position knows which digit of a
lines up with it, because that depends on lengths not yet seen.

With aligned pairs available, + and − fit in fixed depth: the carry into
any digit is one next-match lookup ("nearest lower pair whose sum is not 9"),
the same primitive the parser already uses. So for + and − the obstacle is
*alignment*, not depth. Ways around it:

- Change the prompt contract: equal-width zero-padded operands, operands
  written LSB-first, or lengths given up front.
- Give the model more depth after the input is complete (§6).

## 3. × and ÷ need pairings no position holds

Each product digit is a sum of Θ(n) products a_i·b_{k−i} plus a multi-digit
carry. Attention can sum (averages, counts), but every term multiplies digits
at two different positions, paired in a way that depends on the query. A head
scores one key at a time, and the prompt has only O(n) positions to hold n²
pairs.

Theory doesn't rule it out. Multiplication, and even division (Hesse,
Allender, Barrington 2002), are in TC⁰, and constant-depth, log-precision
transformers sit roughly in uniform TC⁰ (Merrill and Sabharwal). But I know of
no construction inside a causal prompt of n positions. In practice the
written working lines are where the n² pairs live.

## 4. Nested expressions in fixed depth

Evaluating a formula with arbitrary nesting is NC¹-complete, even for
Boolean formulas (Buss 1987). Under the standard conjecture TC⁰ ≠ NC¹, **no
fixed-depth transformer evaluates arbitrarily nested expressions in one
pass.** Each extra level needs either another written step or more depth.

How fast the reachable size grows with layers L depends on the algorithm
built into the layers:

- **Innermost-first, in place:** one block of layers per nesting level (ours
  is about 10: group, operator, operand bounds, compute, rewrite). Nesting
  ≈ L/10. A linear ceiling: 120 layers for about 12 levels.
- **Parallel tree evaluation** (Brent restructuring / tree contraction:
  reduce many independent parts of the tree at once, using pointer jumping):
  formula evaluation is in NC¹, i.e. O(log n) sequential stages. The size
  reachable grows **exponentially** in depth, about 2^(L/c). A few hundred
  layers would cover any expression a person would type.

Two caveats on the parallel route:

1. The arithmetic inside each step must also fit in fixed depth (§3). That
   is within reach for + and −; open for × and ÷.
2. Hand-writing tree contraction in attention is a research project: many
   positions rewriting at once, pointer jumping (our previous-match pointers
   are a start), and every position knowing its role in the tree.

## 5. Precision

Packing a whole number into one value (for example softmax weights
proportional to 10^position, which build the number as a real value) holds
about 15 digits in float64. Beyond that, a value can hold a position or a
count, never an unbounded number. That is why every circuit here works with
digits and pointers.

## 6. Summary

| design | expression size limited by | ceiling as layers L grow |
|---|---|---|
| current (written steps) | nothing (length bound ~10⁶) | none |
| no emission, innermost-first | depth ≈ L/10 | linear |
| no emission, parallel tree evaluation | size ≈ 2^(L/c) | exponential |
| looped transformer (repeat one block until a halt flag) | nothing | none (repeats instead) |

**The looped transformer** applies one hand-written block over and over, with
a halt flag in the weights, and emits only the answer. It removes every
ceiling above: unbounded sequential work, positions rewriting their own state
once lengths are known, nesting as the number of passes. It was set aside
(2026-11-03) because the working would no longer be legible tokens. It would
be residual state that changes between passes, which is much harder to read.
A visualizer could show it pass by pass, but that is a weaker form of
interpretability than reading the model's own written working.

**Realistic answer-first targets with fixed depth, if we come back to this:**

1. + and −, answer first, with an aligned prompt format (§2): achievable
   with existing primitives.
2. Shallow expressions of + and − with parallel evaluation (§4): a research
   step.
3. × and ÷ answer first: open (§3).

## 7. How this relates to production language models

Frontier LLMs are, as far as is publicly described, fixed-depth decoder
transformers. Their internal architectures (layer counts, any recurrence)
are not public, so this is general, not specific to any one model. The same
limits apply to them:

- **Within one forward pass** a model does real internal computation before
  each token. Interpretability work on Claude models (Anthropic, "On the
  Biology of a Large Language Model", 2025) traced addition done inside the
  weights through parallel paths: one estimates the rough magnitude, another
  works out the last digit from lookup-table-like features. These are
  heuristics learned in training, not exact carry algorithms, and they fail
  as operands get longer, as fixed depth predicts.
- **Beyond that, they write their working:** chain-of-thought, or
  "extended thinking" tokens before the answer, which is the same move as
  our working lines. Or they call a tool (a calculator, code execution).
- So "answer directly" works for what fits in one pass, and anything that
  needs unbounded sequential work goes through emitted tokens or tools.

This project's version of that line is exact and inspectable: we know
precisely which work happens inside one pass (§4 of `DESIGN.md`: addressing,
products, division, pointer chasing) and which must be written down.
