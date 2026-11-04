"""Language models with hand-written mechanisms and counted statistics (docs/LM_DESIGN.md).

Each stage adds one mechanism small transformers are known to use, scored in bits per byte on
held-out text: stage 0 is the bigram table (embedding -> unembedding)."""
from __future__ import annotations

from . import circuit, corpus, evaluate, ngram

__all__ = ["circuit", "corpus", "evaluate", "ngram"]
