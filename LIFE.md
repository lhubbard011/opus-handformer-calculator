# Game of Life, written by hand into a transformer

`handformer/life/` is a second model built with the same engine and primitives as the
calculator: 5 layers, 16 attention heads, 41 MLP units, 641 nonzero weights, every one a
named formula. It plays Conway's Game of Life on any grid size, for up to 9 generations.

```
$ python -m handformer.life "3:.#./.#./.#."
3:.#./.#./.#.=2:.../###/...=1:.#./.#./.#.=0:.../###/...=
```

The prompt is a count, `:`, and the grid: `.` dead, `#` alive, `/` ends a row. The model writes
each next generation as a block `k:<grid>=`, counting down to 0, then `\n`. The count lives in
the text: each block starts with the previous block's count minus one. The host feeds bytes,
takes the argmax and stops at `\n`. That is all it does.

**Try it in the visualizer:** run `python -m handformer.viz` and pick "game of life" in the model
menu (the public site shows only the calculator for now: `handformer/viz/models.yaml`). The **Board**
panel shows the generation one back and the one being written, with the cell being written and
the nine cells its heads read outlined.

## The one idea: every block has the same length

The prompt block is positions 1..L, where L is the position of its `=`. Every block the model
writes has the same length, so the cell written at position p sits at **p − L** in the previous
generation, and its 3×3 neighbourhood there is at fixed offsets: ±1 along the row and ±W1
between rows (W1 = a row's length including its `/`). Nine exact lookups read it. Nothing is
counted, searched or stored.

## Layer by layer

**1. Tag the landmarks** [circuit.py:78](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/life/circuit.py#L78 "src:life.tags"). The latest `:` (this block's start), the earliest
`=` (the prompt's, which gives L) and the first row end (which gives the width). A head also
checks whether the previous byte is `=` or the start byte [circuit.py:84](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/life/circuit.py#L84 "src:life.prev"): a digit there is a
block's count.

**2. Find them** [circuit.py:89](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/life/circuit.py#L89 "src:life.landmarks"), then decide what comes next
[circuit.py:93](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/life/circuit.py#L93 "src:life.count_flag"): after the count digit comes `:`, after `=` the next count, otherwise a
cell or a row end. The top and bottom edges are two exact yes/no steps on the cell's offset o in
its block [circuit.py:101](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/life/circuit.py#L101 "src:life.edges"): it has a row above if o ≥ W1, and a row below if o + W1 ≤ B − 1.

**3. Read the previous generation** [circuit.py:108](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/life/circuit.py#L108 "src:life.neighbours"). The cell itself and its eight
neighbours, L bytes back, by exact lookups (score −30·(k − t)²). The left and right edges cost
nothing: a neighbour off the row lands on a `/`, `:` or `=` byte, which reads as dead. A ninth
lookup reads whether the same offset one generation back is a row end [circuit.py:115](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/life/circuit.py#L115 "src:life.rowend"): then
this byte is one too, so the model copies it. That is how rows end without any arithmetic. When a
block ends, a head reads its count digit [circuit.py:119](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/life/circuit.py#L119 "src:life.count_read").

Then the MLP counts live neighbours [circuit.py:124](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/life/circuit.py#L124 "src:life.count_n"): left and right always count; the
three above and the three below count only when that row exists, an AND of two 0/1 values
(ReLU(a + ok − 1)).

**4. The rule.** next = [n = 3] [circuit.py:132](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/life/circuit.py#L132 "src:life.rule3") + [n = 2 and alive] [circuit.py:134](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/life/circuit.py#L134 "src:life.rule2").
Both are equality tests made of exact steps; the second adds 10·(alive − 1) so it can only fire
for a live cell (the same shift trick as the calculator's carry). The two cases never overlap,
so `next` is exactly 0 or 1.

**5. Vote for the next byte** [circuit.py:140](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/life/circuit.py#L140 "src:life.votes"): `#` or `.` for a cell, a copied `/` or `=`,
`:` after the count, and the next count, this block's minus one [circuit.py:152](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/life/circuit.py#L152 "src:life.next_count"), or
`\n` after the block counted 0.

## Check it by breaking it

Knock out the two units of the "has a row above" test (layer 2) and the cells stop seeing the row
above them. A horizontal blinker `1:...../.###./.....` should become a vertical one,
`..#../..#../..#..`. Without that test the bottom cell is never born, because it can no
longer see the three live cells above it: the model writes `..#../..#../.....`. Try it in the
visualizer: step to a cell, open its units, and press "knock out".

## Tests

`tests/test_life.py` checks, byte for byte against a plain-Python reference
(`tests/life_reference.py`):
- every 3×3 grid (all 512);
- the blinker, block, beehive and a glider over 9 generations;
- edge shapes (1×1, one row, one column, count 0) and 60 random grids up to 8×8;
- a 24×30 grid.

It also checks the same interpretability claims as the calculator: every head reads one exact
value, and every yes/no step stays off its ramp, at every position that writes output (5,170
checks, no exceptions). `tests/test_web.py` holds the in-browser engine to the Python model for
Life as well.
