# Language models from hand-written mechanisms: design

Goal: find out how far hand-written mechanisms take a byte-level language model. The
statistics are counted from text (data is king for those); the mechanisms are built by hand,
following what interpretability research has found small transformers compute at each depth;
and every stage is checked against a trained transformer of the same depth on the same text.
Code: `handformer/lm/`; CLI: `python -m handformer.lm`.

## What changes from the other models

The calculator, Life and the C models write one certain byte: their logits are huge and only
the argmax matters. A language model outputs a probability for every next byte, so here the
logits are **calibrated**: each is a log-probability (stage 0) or a sum of log-odds votes
(later stages), and the softmax of the logits is the prediction. A rule that adds +2 to `u`
says "`u` is e² ≈ 7× more likely than it would be otherwise".

Every weight still has a named source, of one of two kinds, recorded per rule:

- **counted**: a number computed from the training text (a bigram log-probability);
- **written**: a mechanism or a rule written by hand (an induction head, "close an open quote").

That split is the point: the interesting number is how much the written rules add on top of
the counted statistics.

## Measurement

**Bits per byte** on held-out text: the mean of −log₂ P(the byte that came next). Uniform over
256 bytes is 8; a perfect predictor is 0. The corpus is Tiny Shakespeare (1,115,394 bytes,
fetched by `tools/fetch_corpus.py`, checked by SHA-256, kept out of git), split 90/10 as nanoGPT
and char-rnn do. Counts come from the training 90%; any choice about them (smoothing) is made
on the training text's last 10%, never on the test text. The text is scored in windows (512
bytes; the first byte of each window is unscored).

**Reference models** (`tools/train_reference.py`): standard attention-only transformers of the
same depth (learned token and position embeddings, pre-LayerNorm, AdamW), trained on the same
training split and scored the same way. Compared on score, on their tables or attention
patterns, and by knockouts.

## The ladder

| Stage | Depth | What that depth computes (Transformer Circuits) | Hand-built version |
|---|---|---|---|
| 0 | no layers | a bigram: embedding → unembedding | the unembedding row of byte a = log P(next \| a), counted |
| 1 | 1 attention layer | skip-trigrams: "A earlier, B now → C" | heads that find trigger bytes (an open quote, bracket, the line's speaker); votes counted per (trigger, now, next) |
| 2 | 2 attention layers | induction heads: "…A B … A → B" | a previous-byte head, then a head that finds the earlier position whose previous byte matches the current byte and copies the byte after it: written, no counts |
| 3 | mixing | when to trust what | gated votes: the copy is trusted more after a longer match |

## Stage 0 result

| model | bits per byte (held-out) |
|---|---|
| uniform over 256 bytes | 8.0000 |
| byte frequencies | 4.8291 |
| **stage 0: the bigram as weights** (68 dims, 16,961 weights) | **3.5866** |
| trained 0-layer transformer (d 128, 98,560 params, 2,000 steps) | 3.5894 |

The weights reproduce the counted table exactly (`tests/test_lm.py`, to 1e-12). The trained
0-layer model lands on the same table: its next-byte distributions differ from the counted ones
by 0.012 bits on average (KL, weighted by frequency), with the same most likely next byte for
98% of the bytes. Where they differ is where training generalises: after `q`, both put almost
everything on `u`, but the counted table spreads the rest evenly over unseen bytes while the
trained model's shared embeddings guess letters.

## Stage 1 so far: what one attention layer buys

Measured on the test text, all heads attention-only, votes counted as pointwise mutual
information (how much more likely C becomes after A, against how likely C is anyway), one
strength per head chosen on the training tail:

| model | bits per byte |
|---|---|
| bigram (stage 0) | 3.587 |
| + a head on the byte 2 back | 3.380 |
| + heads on the bytes 2 and 3 back | **3.335** |
| + more fixed offsets (4–8) | worse: the separately counted votes double-count |
| + content heads (latest uppercase / vowel / punctuation / consonant) | no gain |
| the same additive structure, tables fitted by optimisation (best: offsets 2, 3) | 3.196 |
| the true trigram, counted (needs two bytes jointly: not additive) | 2.94 |
| **stage 1: 65 conditional heads** (one per current byte; the exact trigram, attention only) | **2.945** |
| + conditional skip votes: the byte 2, 3, 4 before the current one, given it (counted, measured outside the weights so far) | **2.726** |
| **trained 1-layer attention-only reference** (d 128, 4 heads, 3,000 steps) | **2.904** |
| **trained 1-layer, the engine's rules** (no LayerNorm, no biases, positions only as p and p²) | **2.902** |

The trained one-layer model beats even the counted trigram, so the textbook picture (one
layer = additive skip-trigram votes) undersells it. Fitting the additive votes directly only
reaches 3.196: the gap is structure, not estimation. The structure is a **conditional head**: one
job or nothing. "The byte before, when the current byte is `e`" attends to the byte before when
the current byte is `e`, and otherwise to a sink byte at the start of the window whose value
writes nothing; its dims (`p_e_t`: "current `e`, previous `t`") vote log P(c | t, e) − log P(c | e),
so the logits are the trigram's. Every dim keeps one meaning; the price is width (1,450 dims and
65 heads against the trained model's 128 and 4). The skip votes on top of it reach 2.726, below
the trained model (a small one: larger references are the fair test, still to run). Built as
conditional heads they would need ~10,000 pair dims; the same function with an MLP layer of AND
units ("current `e` and 2 back `t`") needs ~600, so that is the build planned, with trained
one-layer references both with and without an MLP.

**A fair reference.** `tools/train_reference.py --arch engine` trains under the hand-built
models' constraints: attention only, no LayerNorm, no biases, positions only as the fixed features
p and p². It scores 2.902, the same as the standard one: LayerNorm and biases were not where its
power came from. `handformer/lm/trained.py` loads it into the engine (two exact rewrites: the
trainer's p/ctx scale and attention's 1/√d_head folded into the weights; logits within 2·10⁻⁴ of
PyTorch, the same score), so it runs in the visualizer next to the hand-built model: same views,
same knockouts, 4 heads and 128 learned dims with no assigned meanings against 65 heads and 1,450
named ones.

In the visualizer (local only: they need the corpus and the trained weights):
`python -m handformer.viz --models calc,lm,lmt`. Inside it, the four heads change what they
attend to with the current byte: inside a word, heads read the byte 1 and 2 back; after a
space or a newline, two of them look far back instead (90–94% of their attention beyond 5 bytes).
Its byte embeddings share directions between similar bytes (`.` with `?` 0.47 and `!` 0.34, `Q`
with `J` 0.53: shared features such as "the sentence ends"), and bytes and positions partly share
dims (mild superposition: 43 + 68 of 128 dims carry 90% of their variance, with overlap up to
0.35). The hand-built model is monosemantic by construction: one meaning per dim, one job per head.

## Planned experiments

Each one isolates a single question; the result goes in the table of its stage.

1. **Estimation or structure?** Fit the additive vote tables (bigram + one table per offset)
   directly by optimisation, same structure as the counted votes. If it nears 2.9 bits, the gap
   to the trained model is in the counting (naive votes double-count); if it stays near 3.3, the
   trained model does something an additive vote cannot.
2. **Context-dependent heads, written.** A head whose query depends on the current byte (inside a
   word: look 1–2 back; after a space or newline: look far back), as the trained heads do, but
   written as separate monosemantic heads (there is no head budget to pack them into).
3. **Shared features, monosemantic.** Dims that each mean one thing shared by several bytes
   ("the current byte ends a sentence": `.` `?` `!`; "a rare capital"), with their own votes
   next to each byte's own: the written analogue of the trained model's shared directions, so
   evidence transfers between similar bytes.
4. **What does monosemanticity cost?** Shrink the trained reference (d 128 → 32 → 16) to force it
   into superposition, and find where it stops beating the hand-built model.
5. **Train from the hand-built weights.** Start gradient descent at the monosemantic model and
   watch what it does: which weights move first, whether named dims stay clean or smear into
   shared directions, whether a one-job head becomes several. A direct look at how polysemanticity
   appears.
6. **Freeze the hand-built parts, train around them.** The written circuits fixed, a small trained
   part learns only the remainder: what it learns is what the rules miss, and the bits it adds
   measure it.
7. **Decompile the trained model into rules.** Read its heads and directions, write them back as
   named monosemantic rules, and count the bits that survive the translation.
8. **Stage 2, induction heads** (the ladder above), with the repeated-random-sequence test on the
   hand-built model, the trained 2-layer reference and GPT-2.

## Limits

- Byte level, so GPT-2 (word-piece tokens) is compared qualitatively (attention patterns,
  the repeated-sequence induction test), not by score.
- Smoothing is add-α. Better counted smoothing (Kneser-Ney) would lower the counted baselines;
  it is a choice about the data side, kept simple until the mechanisms need it.
