"""python -m handformer.lm [--corpus shakespeare] [--sample N]

Counts the statistics on the training text, builds the stage models, and scores each in bits per
byte on the held-out text, next to the baselines (uniform, byte frequencies)."""
from __future__ import annotations

import argparse
import logging

import numpy as np

from . import circuit, corpus, evaluate, ngram

log = logging.getLogger("handformer.lm")


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m handformer.lm", description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", default="shakespeare", choices=sorted(corpus.CORPORA))
    ap.add_argument("--sample", type=int, default=0, help="also draw this many bytes from the model (seeded)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    train, test = corpus.split(corpus.load(args.corpus))
    log.info("train %d bytes, test %d bytes", len(train), len(test))

    rows = [("uniform over 256 bytes", 8.0)]
    ua = ngram.choose_alpha(train, counter=ngram.unigram_counts, scorer=ngram.bits_unigram)
    rows.append(("byte frequencies (unigram)", ngram.bits_unigram(ngram.smoothed_log(ngram.unigram_counts(train), ua), test)))
    bg = ngram.fit_bigram(train)
    rows.append(("bigram table, counted (reference)", ngram.bits_bigram(bg.logp, test)))
    m = circuit.build(bg)
    rows.append(("stage 0: the bigram as weights", evaluate.bits_per_byte(m, test)))

    width = max(len(r[0]) for r in rows)
    log.info("bits per byte on the held-out %d bytes:\n%s", len(test),
             "\n".join(f"  {name:<{width}}  {bpb:.4f}" for name, bpb in rows))
    if args.sample:
        out = evaluate.sample(m, b"\n", args.sample, seed=args.seed)
        log.info("a sample (the host draws from the model's distribution, seed %d):\n%s", args.seed,
                 out.decode("latin-1"))


if __name__ == "__main__":
    np.seterr(all="raise")
    main()
