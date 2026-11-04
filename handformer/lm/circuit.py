"""Hand-built language models: the mechanisms are written by hand, the numbers are counted.

Every model here outputs a probability for every next byte, so unlike the other models its
logits are calibrated: each one is a log-probability (or a sum of log-odds votes, from stage 1
on), and the softmax of the logits is the model's prediction. Stages follow what small
transformers are known to compute at each depth (docs/LM_DESIGN.md).

Stage 0, no layers: the only path is embedding -> unembedding, so the prediction can depend only
on the current byte, and the best such model is a bigram table. One dim per byte seen in the
training text (``now_e``: "the current byte is 'e'"); the unembedding entry from ``now_a`` to byte
b is log P(b | a), counted and smoothed (ngram.py). The logits are therefore exactly the log
probabilities: logit(b) = log((count(a, b) + alpha) / (count(a) + 256 alpha)). A byte never seen
in training has no dim, so all its logits are 0: the uniform 1/256, the same formula's limit.
"""
from __future__ import annotations

import logging

import numpy as np

from ..model import Layout, ModelBuilder, Transformer
from .ngram import V, Bigram

log = logging.getLogger(__name__)

NAMED = {ord(" "): "space", ord("\n"): "nl", ord("\t"): "tab", ord("!"): "bang", ord("'"): "apos", ord('"'): "quote",
         ord(","): "comma", ord("."): "dot", ord(":"): "colon", ord(";"): "semi", ord("?"): "qmark", ord("-"): "dash",
         ord("&"): "amp", ord("$"): "dollar", ord("("): "lpar", ord(")"): "rpar"}


def byte_name(b: int) -> str:
    """A dim-safe name for a byte: the letter or digit itself, a word for punctuation, else hex."""
    ch = chr(b)
    if ch.isascii() and ch.isalnum():
        return ch
    return NAMED.get(b, f"x{b:02x}")


def now(b: int) -> str:
    return f"now_{byte_name(b)}"


def show(b: int) -> str:
    return repr(chr(b)) if 32 <= b < 127 else {10: "'\\n'", 9: "'\\t'"}.get(b, f"byte {b}")


def bigram_builder(bg: Bigram) -> ModelBuilder:
    """Stage 0: the bigram table as an embedding -> unembedding path (no layers)."""
    seen = [a for a in range(V) if bg.counts[a].sum() > 0]
    lay = Layout()
    lay.add("one", "pos", "pos2", *(now(a) for a in seen))
    b = ModelBuilder(lay)
    for byte in range(V):
        b.embed_byte(byte, {"one": 1.0, **({now(byte): 1.0} if byte in seen else {})})
    logp = bg.logp
    for a in seen:
        for nxt in range(V):
            b.logit(now(a), nxt, float(logp[a, nxt]))
    log.info("stage 0 (bigram): %d byte dims, %d unembedding entries, alpha %s", len(seen), len(seen) * V, bg.alpha)
    return b


def docs(bg: Bigram) -> dict[str, str]:
    """What each dim means (the visualizer shows these on hover)."""
    d = {"one": "1 for every byte (the model's only bias).", "pos": "The position.", "pos2": "The position squared."}
    for a in range(V):
        n = int(bg.counts[a].sum())
        if n:
            top = np.argsort(-bg.counts[a])[:3]
            likely = ", ".join(f"{show(int(t))} {bg.counts[a, t] / n:.0%}" for t in top if bg.counts[a, t])
            d[now(a)] = (f"1 if the current byte is {show(a)}. Its unembedding row is log P(next | {show(a)}), "
                         f"counted from {n} occurrences: most often {likely}.")
    return d


def build(bg: Bigram) -> Transformer:
    m = bigram_builder(bg).build()
    log.info("LM stage 0: d_model %d, %d nonzero weights", len(m.layout), m.nonzero_params())
    return m


# ---------------------------------------------------------------- stage 1: conditional heads
# Each window of text starts with SINK, a byte the text never has. A conditional head either
# does its one job or attends to the sink, whose value writes nothing (an "attention sink").
SINK = 2
SINK_BETA = 30.0  # = primitives.GATHER_BETA: the exact lookup's sharpness per position
MAX_CTX = 1024  # longest window the sink switch is exact for (its score must beat every distance)
SINK_SCORE = SINK_BETA * MAX_CTX ** 2 + 1000.0  # beats the lookup's distance penalty anywhere in a window
ON_SCORE = 2 * SINK_SCORE  # "my condition holds": real positions beat the sink


def pair_dim(a: int, b: int) -> str:
    return f"p_{byte_name(b)}_{byte_name(a)}"  # "the current byte is b and the one before it was a"


def trigram_builder(tri: "Trigram") -> ModelBuilder:  # noqa: F821
    """Stage 1: the bigram (direct path) plus one conditional head per current byte b: when the
    current byte is b, it reads the byte before (a) into the dim p_b_a; otherwise it attends to the
    sink and writes nothing. The unembedding of p_b_a votes log P(c | a, b) - log P(c | b), so the
    logits are log P(c | a, b): the trigram, in one attention layer, one meaning per dim."""
    from ..model import HeadSpec

    bg = tri.bigram
    seen = [a for a in range(V) if bg.counts[a].sum() > 0]
    by_current: dict[int, list[int]] = {}
    for a, b in tri.pairs:
        by_current.setdefault(b, []).append(a)
    lay = Layout()
    lay.add("one", "pos", "pos2", "is_sink", "real", *(now(a) for a in seen),
            *(pair_dim(a, b) for a, b in tri.pairs))
    b_ = ModelBuilder(lay)
    for byte in range(V):
        if byte == SINK:
            b_.embed_byte(byte, {"one": 1.0, "is_sink": 1.0})
        else:
            b_.embed_byte(byte, {"one": 1.0, "real": 1.0, **({now(byte): 1.0} if byte in seen else {})})
    l1 = b_.at(1)
    for b, prevs in sorted(by_current.items()):
        # score(key k) = -beta (k - (t-1))^2 + SINK_SCORE * sink_k + ON_SCORE * [current is b] * real_k
        l1.heads.append(HeadSpec(
            name=f"previous byte, when the current byte is {show(b)}",
            query=[{"pos": 2 * SINK_BETA, "one": -2 * SINK_BETA}, {"one": -SINK_BETA},
                   {"one": SINK_SCORE}, {now(b): ON_SCORE}],
            key=[{"pos": 1.0}, {"pos2": 1.0}, {"is_sink": 1.0}, {"real": 1.0}],
            value={now(a): {pair_dim(a, b): 1.0} for a in prevs},
        ))
    lp = bg.logp
    for a in seen:  # the direct path: the bigram row
        for nxt in range(V):
            b_.logit(now(a), nxt, float(lp[a, nxt]))
    votes = tri.votes()
    for i, (a, b) in enumerate(tri.pairs):
        for nxt in range(V):
            b_.logit(pair_dim(a, b), nxt, float(votes[i, nxt]))
    log.info("stage 1 (trigram): %d conditional heads, %d pair dims, kappa %s", len(by_current), len(tri.pairs), tri.kappa)
    return b_


def build_trigram(tri) -> Transformer:
    m = trigram_builder(tri).build()
    log.info("LM stage 1: d_model %d, %d heads, %d nonzero weights", len(m.layout), len(m.layers[0].heads), m.nonzero_params())
    return m
