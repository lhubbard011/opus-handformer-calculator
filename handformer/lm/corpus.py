"""The texts the language models count from and are scored on (kept out of git: data/).

Each corpus is a byte string with a fixed train/test split: the first ``TRAIN_FRACTION`` is
for counting statistics (and choosing their smoothing), the rest is held out, so a score is
always on text the counts never saw. ``tools/fetch_corpus.py`` downloads a corpus and checks it.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
TRAIN_FRACTION = 0.9  # the split nanoGPT and char-rnn use for this corpus


@dataclass(frozen=True)
class Corpus:
    name: str
    path: str  # relative to the repository root
    url: str
    sha256: str
    about: str


CORPORA = {
    "shakespeare": Corpus(
        "shakespeare", "data/tinyshakespeare/input.txt",
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed",
        "Tiny Shakespeare (karpathy/char-rnn): 1,115,394 bytes of plays, 65 distinct characters"),
}


def load(name: str = "shakespeare") -> bytes:
    """The corpus's bytes, checked against its recorded SHA-256."""
    c = CORPORA[name]
    path = ROOT / c.path
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run  python tools/fetch_corpus.py {name}")
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != c.sha256:
        raise ValueError(f"{path}: SHA-256 {digest} is not the recorded {c.sha256} (a different file?)")
    log.info("corpus %s: %d bytes from %s", name, len(data), path)
    return data


def split(data: bytes) -> tuple[bytes, bytes]:
    """(train, test): the first TRAIN_FRACTION, then the rest."""
    k = int(len(data) * TRAIN_FRACTION)
    return data[:k], data[k:]
