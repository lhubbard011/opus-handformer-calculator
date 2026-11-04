"""Counted statistics: the data side of the language models (the mechanisms are hand-written).

Byte counts and byte-pair counts from the training text, turned into probabilities with
add-alpha smoothing: P(b | a) = (count(a, b) + alpha) / (count(a) + 256 * alpha). Every byte
keeps a nonzero probability, so a byte never seen after ``a`` costs bits but never infinitely
many. ``alpha`` is chosen on the last part of the training text (counted from the rest), never
on the test text.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

V = 256  # bytes
ALPHAS = (1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0)
HOLDOUT = 0.1  # the part of the training text alpha is chosen on


def unigram_counts(data: bytes) -> np.ndarray:
    return np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=V).astype(np.float64)


def bigram_counts(data: bytes) -> np.ndarray:
    """(256, 256): row a, column b = how often b followed a."""
    x = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
    c = np.zeros((V, V))
    np.add.at(c, (x[:-1], x[1:]), 1.0)
    return c


def smoothed_log(counts: np.ndarray, alpha: float) -> np.ndarray:
    """log P from counts, add-alpha smoothed along the last axis (natural log)."""
    num = counts + alpha
    return np.log(num) - np.log(num.sum(axis=-1, keepdims=True))


def bits_bigram(logp: np.ndarray, data: bytes) -> float:
    """Bits per byte of ``data`` under the bigram table (every byte after the first is scored)."""
    x = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
    return float(-logp[x[:-1], x[1:]].mean() / np.log(2))


def bits_unigram(logp: np.ndarray, data: bytes) -> float:
    x = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
    return float(-logp[x[1:]].mean() / np.log(2))


@dataclass(frozen=True)
class Bigram:
    """The bigram statistics a stage-0 model is built from."""
    counts: np.ndarray  # (256, 256) from the whole training text
    alpha: float

    @property
    def logp(self) -> np.ndarray:
        return smoothed_log(self.counts, self.alpha)

    def count_of(self, a: int, b: int) -> int:
        return int(self.counts[a, b])


def choose_alpha(train: bytes, alphas=ALPHAS, counter=bigram_counts, scorer=bits_bigram) -> float:
    """The alpha that scores best on the training text's held-out tail, counting the rest."""
    k = int(len(train) * (1 - HOLDOUT))
    fit, held = train[:k], train[k:]
    c = counter(fit)
    scores = {a: scorer(smoothed_log(c, a), held) for a in alphas}
    best = min(scores, key=lambda a: scores[a])
    log.info("alpha on the training tail: %s -> %s", {a: round(s, 4) for a, s in scores.items()}, best)
    return best


def fit_bigram(train: bytes) -> Bigram:
    alpha = choose_alpha(train)
    b = Bigram(bigram_counts(train), alpha)
    log.info("bigram: %d byte pairs counted, %d distinct, alpha %s", len(train) - 1, int((b.counts > 0).sum()), alpha)
    return b


# ---------------------------------------------------------------- trigrams (stage 1)
KAPPAS = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0)


@dataclass(frozen=True)
class Trigram:
    """Trigram statistics for the pairs (previous a, current b) seen in training, smoothed toward
    the bigram: P(c | a, b) = (count(a, b, c) + kappa P(c | b)) / (count(a, b) + kappa). A rare
    pair leans on the bigram; an unseen pair is not here at all (the model falls back to it)."""
    pairs: tuple[tuple[int, int], ...]  # (a, b), in a fixed order
    counts: np.ndarray  # (len(pairs), 256): count(a, b, c)
    kappa: float
    bigram: Bigram

    def logp(self) -> np.ndarray:
        """(len(pairs), 256): log P(c | a, b)."""
        p2 = np.exp(self.bigram.logp[[b for _, b in self.pairs]])
        n = self.counts.sum(1, keepdims=True)
        return np.log((self.counts + self.kappa * p2) / (n + self.kappa))

    def votes(self) -> np.ndarray:
        """(len(pairs), 256): log P(c | a, b) - log P(c | b): what the pair's head adds to the bigram row."""
        return self.logp() - self.bigram.logp[[b for _, b in self.pairs]]


def trigram_counts(data: bytes) -> tuple[tuple[tuple[int, int], ...], np.ndarray]:
    x = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
    key = x[:-2] * V + x[1:-1]
    pairs = np.unique(key)
    row = np.searchsorted(pairs, key)
    c = np.zeros((len(pairs), V))
    np.add.at(c, (row, x[2:]), 1.0)
    return tuple((int(p // V), int(p % V)) for p in pairs), c


def bits_trigram(tri: Trigram, data: bytes) -> float:
    """Bits per byte of ``data`` (bytes 3 onward): the trigram where its pair was seen, else the bigram."""
    x = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
    index = {p: i for i, p in enumerate(tri.pairs)}
    lp3, lp2 = tri.logp(), tri.bigram.logp
    cost = [-(lp3[index[(int(x[t - 2]), int(x[t - 1]))], x[t]] if (int(x[t - 2]), int(x[t - 1])) in index
              else lp2[x[t - 1], x[t]]) for t in range(2, len(x))]
    return float(np.mean(cost) / np.log(2))


def fit_trigram(train: bytes, bigram: Bigram) -> Trigram:
    k = int(len(train) * (1 - HOLDOUT))
    pairs, c = trigram_counts(train[:k])
    bg_fit = Bigram(bigram_counts(train[:k]), bigram.alpha)
    scores = {kp: bits_trigram(Trigram(pairs, c, kp, bg_fit), train[k:]) for kp in KAPPAS}
    kappa = min(scores, key=lambda kp: scores[kp])
    log.info("kappa on the training tail: %s -> %s", {kp: round(s, 4) for kp, s in scores.items()}, kappa)
    pairs, c = trigram_counts(train)
    tri = Trigram(pairs, c, kappa, bigram)
    log.info("trigram: %d (previous, current) pairs seen, kappa %s", len(pairs), kappa)
    return tri
