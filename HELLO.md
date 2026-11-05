# Hello world: a transformer that adds, written by hand

Read this before the full calculator. `handformer/hello.py` is one file and
five layers: 9 attention heads, 66 MLP units and 761 nonzero weights. Every
weight comes from a formula you can read, and the same primitives build the
full calculator (8,915 weights).

```
$ python -m handformer.hello 47+38 99999+1
47+38=58;85
99999+1=000001;100000
```

The prompt is `\n47+38=`. The model writes the sum's digits least-significant
first (`58`, the order a carry travels), then `;`, then the answer read back
most-significant first (`85`), then `\n`. The host feeds bytes, takes the
argmax, and stops at `\n`. That is all it does.

## The residual stream

Each position carries a vector of 72 named numbers. Each one means one thing:
a 0/1 flag, a digit, or a position in the sequence. The byte embedding sets the
inputs: `val` (the digit's value), `oh0`–`oh9` (one-hot digit), `is_plus`,
`is_eq`, `is_semi` and so on. `pos` and `pos²` give the position. Everything
else is written by a layer.

## Layer by layer

**1. Tag the landmarks** [hello.py:55](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/hello.py#L55 "src:hello.features"). An MLP unit writes
`f_P = 2·(pos+1)` on a `+` byte and 0 elsewhere (`f_E` for `=` and `f_S` for
`;` likewise). The start byte gets 1, so "not found yet" means position 0. The
factor 2 keeps the start byte from tying with a match at position 1.

**2. Find them** [hello.py:64](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/hello.py#L64 "src:hello.locate"). A *latest* head has query `40·one` and
key `f_P`, so its score is 40·`f_P`, which grows with position. Softmax picks
the most recent `+`, and the head copies that key's `pos` into `P`. That is
attention as a lookup, not as similarity
([primitives.py:220](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/primitives.py#L220 "src:prim.latest_head_def")). The MLP then sets two flags:
- `seen_eq = [E ≥ 1]` (we are past the prompt);
- `working_done = [S > E]` (`;` has been written) [hello.py:70](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/hello.py#L70 "src:hello.phase").

Both are robust steps: the slope is 4 around ½, so noise of up to ¼ can never
push a flag off 0 or 1. Finally it computes where digit i of each operand
sits, with i = pos − E: the next working digit is digit i
[hello.py:74](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/hello.py#L74 "src:hello.address").

**3. Read the digits** [hello.py:91](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/hello.py#L91 "src:hello.gather"). A *gather* head scores keys by
−30·(k − t)², written as a query of [60t, −30] against keys [pos, pos²]. The
score peaks at exactly one position t, and the head reads that byte's `val`.
An operand that has run out of digits points at its fence (`+` or the start
byte), whose `val` is 0.

**The carry is recovered, never stored** [hello.py:98](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/hello.py#L98 "src:hello.carry"). With y = the
sum of the digits one column lower:

- carry = [y ≥ 10] + [y = 9 and the previous output digit is `0`]
  [hello.py:103](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/hello.py#L103 "src:hello.carry9").

When y = 9, the lower column carried exactly when it wrote a 0. So the model
reads its own previous output and needs no memory beyond the text.

**4. Write a working digit** [hello.py:121](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/hello.py#L121 "src:hello.digit"). Shared ReLU ramps turn
`A + B + carry` (0–19) into a one-hot vote for its last digit `D0`–`D9`
([common.py:121](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/common.py#L121 "src:prim.ramp")). A separate unit writes `;` once both operands are used up
and no carry is left [hello.py:125](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/hello.py#L125 "src:hello.semi"). A latest head finds the top nonzero
working digit [hello.py:115](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/hello.py#L115 "src:hello.top"), which gives the answer's length.

**5. Read it back** [hello.py:132](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/hello.py#L132 "src:hello.answer"). A gather head reads the working
digits from the top down. When the copy is complete, `done` writes `\n`
[hello.py:137](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/hello.py#L137 "src:hello.done").

The unembedding [hello.py:143](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/hello.py#L143 "src:hello.logits") maps `D0`–`D9` to the digit bytes with
weight 10, and `;` and `\n` with weight 20. When a control byte and a digit
fire at the same step, the control byte wins.

## Check the explanation by breaking it

If the carry units really carry, removing them should break exactly the sums
that carry. Knockouts zero a part's output in the weights
([ablate.py:53](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/ablate.py#L53 "src:ablate.apply")):

```python
from handformer import hello, ablate
b = hello.builder(); m = b.build()
carry = [(2, u) for u, s in enumerate(b.layers[2].units) if "carry" in s.write]
no_carry = ablate.apply(m, ablate.Knockout(units=frozenset(carry)))
hello.solve(no_carry, "47+38")    # '57;75'   (was 85: the carry from 7+8 is gone)
hello.solve(no_carry, "12+34")    # '64;46'   (unchanged: nothing carries)
only_y_ge_10 = ablate.apply(m, ablate.Knockout(units=frozenset(carry[2:])))
hello.solve(only_y_ge_10, "99999+1")   # '00999;99900'
```

The last line removes only the "y = 9 and wrote 0" units. The first carry
(9+1 = 10) and the second (9+0+1 = 10) still happen. The chain breaks at the
third digit, where y = 9 + 0 = 9 and the rule that passes the carry on is
gone. That is exactly what the explanation predicts.

## Where to go next

- The full calculator uses the same moves at a larger scale: README.md and
  DESIGN.md, and "Read the circuits" for links into each one.
- `tests/test_hello.py` checks every pair below 100 and long random operands,
  byte for byte.
- The visualizer (`python -m handformer.viz`, or the hosted page) steps
  through the full model with the same kinds of heads and units, and lets you
  knock any of them out.
