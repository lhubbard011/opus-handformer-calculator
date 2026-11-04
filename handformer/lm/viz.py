"""The language model in the visualizer (handformer/viz/models.py: the "lm" entry).

Shows the stage-1 model: the bigram path plus one conditional head per current byte (the exact
trigram). Its statistics are counted from the training text the first time anything asks for
them, so the visualizer starts without the corpus; only choosing this model needs it
(``python tools/fetch_corpus.py``).
"""
from __future__ import annotations

import functools
import logging
from collections.abc import Iterator, Mapping

from ..model import ModelBuilder
from . import circuit, corpus, ngram

log = logging.getLogger(__name__)

MAX_INPUT = 300
MAX_NEW = 120  # bytes it may write (greedy decoding loops: "the the the ..."), so a run stays light
SAMPLE = "ROMEO:\nWhat light"
HINT = "any text: the model continues it, one byte at a time, until it writes a newline"


@functools.cache
def stats(name: str = "shakespeare") -> ngram.Trigram:
    train, _test = corpus.split(corpus.load(name))
    bg = ngram.fit_bigram(train)
    tri = ngram.fit_trigram(train, bg)
    log.info("language model statistics counted from %d training bytes of %s", len(train), name)
    return tri


def builder() -> ModelBuilder:
    return circuit.trigram_builder(stats())


def prompt(text: str) -> list[int]:
    """The text as its UTF-8 bytes, after the sink byte (every window starts with it)."""
    if not text:
        raise ValueError("empty text")
    data = text.encode("utf-8")
    if circuit.SINK in data:
        raise ValueError("the text may not contain byte 0x02 (the sink)")
    if len(data) > MAX_INPUT:
        raise ValueError(f"text longer than {MAX_INPUT} bytes")
    return [circuit.SINK, *data]


def category(name: str) -> str:
    """How the visualizer groups the dims: the bigram path, the conditional heads, the inputs."""
    if name.startswith("now_"):
        return "bigram"
    if name.startswith("p_"):
        return "heads"
    return "input"


class LazyDocs(Mapping):
    """The dims' explanations, built from the counts on first use (they need the corpus)."""

    @functools.cached_property
    def _d(self) -> dict[str, str]:
        tri = stats()
        d = circuit.docs(tri.bigram)
        d["is_sink"] = "1 on the sink byte that starts every window: conditional heads attend to it when their condition fails."
        d["real"] = "1 on every real byte (not the sink): the key a conditional head's switch reads."
        n = tri.counts.sum(1)
        for i, (a, b) in enumerate(tri.pairs):
            d[circuit.pair_dim(a, b)] = (f"1 when the current byte is {circuit.show(b)} and the one before it was "
                                         f"{circuit.show(a)} (seen {int(n[i])} times in training). Its unembedding row "
                                         f"turns the bigram row of {circuit.show(b)} into the trigram: "
                                         f"log P(next | {circuit.show(a)}, {circuit.show(b)}) - log P(next | {circuit.show(b)}).")
        return d

    def __getitem__(self, k: str) -> str:
        return self._d[k]

    def __iter__(self) -> Iterator[str]:
        return iter(self._d)

    def __len__(self) -> int:
        return len(self._d)


BYTE_NAMES = {chr(b): circuit.byte_name(b) for b in range(32, 127)} | {"\n": circuit.byte_name(10), "\t": circuit.byte_name(9)}


# ---------------------------------------------------------------- the trained reference, side by side
TRAINED = corpus.ROOT / "data" / "reference" / "engine1.pt"  # tools/train_reference.py --arch engine --layers 1
TRAINED_CTX = 256  # the positions it was trained on (it never saw a later one)
TRAINED_HINT = "any text: the trained model continues it, one byte at a time (it has no sink: windows start at your text)"


def trained_builder() -> ModelBuilder:
    from . import trained

    if not TRAINED.with_suffix(".npz").exists() and not TRAINED.exists():
        raise FileNotFoundError(f"{TRAINED} is missing: python tools/train_reference.py {TRAINED.relative_to(corpus.ROOT)} "
                                "--layers 1 --arch engine")
    return trained.builder(TRAINED)


def trained_prompt(text: str) -> list[int]:
    """The text as its UTF-8 bytes (no sink: the trained model's windows start at the text)."""
    if not text:
        raise ValueError("empty text")
    data = text.encode("utf-8")
    if len(data) >= TRAINED_CTX:
        raise ValueError(f"text of {len(data)} bytes: the trained model knows positions up to {TRAINED_CTX - 1} only")
    return list(data)


def trained_category(name: str) -> str:
    return "input" if name in ("pos", "pos2") else "learned"


TRAINED_DOCS = {
    **{f"d{i}": "A learned dim: no meaning was assigned to it. It is whatever training made it, usually a blend "
                "of several features (superposition)." for i in range(1024)},
    "pos": "The position p (fixed, not learned): the only positional signal, as in the hand-built models.",
    "pos2": "The position squared (fixed, not learned).",
}
