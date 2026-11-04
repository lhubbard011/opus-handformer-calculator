"""Seeded sampling of a next byte, the same in the Python engine and the page's (engine.js).

For the language models the host may draw the next byte from the model's distribution instead of
taking its argmax: the model still computes every probability; the host only chooses. Both
engines must draw the same bytes for the same seed (the parity tests check it), so the random
numbers come from mulberry32 (a 32-bit generator that is exact in JavaScript integer arithmetic)
and the draw is an inverse-CDF walk in byte order:

    z = logits / temperature;  e_b = exp(z_b - max z);  pick the first b with e_0 + ... + e_b > u * sum(e)

with u in [0, 1) from the generator, one draw per written byte.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MASK = 0xFFFFFFFF


class Mulberry32:
    """The mulberry32 generator (same sequence as engine.js Mulberry32 for the same seed)."""

    def __init__(self, seed: int) -> None:
        self.a = seed & MASK

    def next(self) -> float:
        self.a = (self.a + 0x6D2B79F5) & MASK
        t = self.a
        t = ((t ^ (t >> 15)) * (t | 1)) & MASK
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & MASK)) & MASK
        return ((t ^ (t >> 14)) & MASK) / 4294967296.0


@dataclass(frozen=True)
class Sampling:
    temperature: float
    seed: int

    def __post_init__(self) -> None:
        if not 0.05 <= self.temperature <= 5.0:
            raise ValueError(f"temperature {self.temperature} outside 0.05..5")
        if not 0 <= self.seed <= MASK:
            raise ValueError(f"seed {self.seed} outside 0..{MASK}")


def draw(logits: np.ndarray, temperature: float, rng: Mulberry32) -> int:
    z = np.asarray(logits, dtype=np.float64) / temperature
    cum = np.cumsum(np.exp(z - z.max()))  # sequential, in byte order (as engine.js)
    u = rng.next() * cum[-1]
    return int(np.searchsorted(cum, u, side="right"))
