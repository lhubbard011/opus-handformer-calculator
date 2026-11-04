"""Teacher-forced verification against the reference sequence.

One forward pass over prompt + expected output; the argmax at every output
position must equal the next expected byte. If every step is right under
teacher forcing, greedy decoding reproduces exactly that sequence.
"""

import random

import numpy as np

from handformer import arith
from tests import reference

_MODEL = None


def model():
    global _MODEL
    if _MODEL is None:
        _MODEL = arith.build()
    return _MODEL


def check(expression: str) -> None:
    p = arith.prompt(expression)
    seq = p + list(reference.output(expression).encode("ascii"))
    pred = np.argmax(model().logits(seq[:-1]), axis=1)
    start = len(p) - 1
    got = bytes(int(t) for t in pred[start:]).decode("ascii", "replace")
    want = bytes(seq[start + 1:]).decode("ascii")
    if got != want:
        k = next(i for i, (g, w) in enumerate(zip(got, want)) if g != w)
        raise AssertionError(f"{expression}: first diff at {k}\n want {want!r}\n  got {got!r}")


def rand_digits(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("0123456789") for _ in range(n))
