"""Command line: ``python -m handformer.loop "2*(3+4)-1"`` (or no argument for a prompt loop).

Builds the recurrent-depth calculator, feeds ``\\n<expression>=`` and decodes greedily until
``\\n``: the model thinks in latent positions, then writes the answer and nothing else.
``--steps`` also prints what the thoughts held: every copy of the expression, and how many
positions and iterations each step took; ``-v`` logs the build and the run.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import numpy as np

from . import build, run as loop_run

log = logging.getLogger("handformer.loop")


def copies(model, res) -> list[tuple[str, int, int]]:
    """(copy, positions, iterations) per step: the latent cells between roots, read back as
    bytes (display only: the nearest embedding row of each thought's byte features)."""
    lay = model.layout
    feat = [j for j in np.flatnonzero(np.abs(model.embed).sum(axis=0)) if j != lay["one"]]
    rows = model.embed[:, feat]
    roles = [lay[r] for r in ("k_s", "k_w", "k_q", "k_r")]
    out, text, n, iters = [], [], 0, 0
    for t in res.tokens[len(res.prompt):]:
        if not t.latent:
            break
        e = res.inputs[t.index]
        n, iters = n + 1, iters + t.iterations
        if any(e[j] > 0.5 for j in roles):
            continue
        ch = chr(int(((e[feat][None] - rows) ** 2).sum(axis=1).argmin()))
        if ch == "=":
            out.append(("".join(text), n, iters))
            text, n, iters = [], 0, 0
        else:
            text.append(ch)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m handformer.loop")
    ap.add_argument("expression", nargs="*", help="e.g. 2*(3+4)-1 or 355/113~6; omit for an interactive loop")
    ap.add_argument("--steps", action="store_true", help="print every copy of the expression the model thought")
    ap.add_argument("-v", "--verbose", action="store_true", help="log the model build and the run")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    t0 = time.perf_counter()
    model = build()
    log.info("recurrent-depth model built in %.2fs", time.perf_counter() - t0)

    def one(expression: str) -> None:
        expression = expression.strip("\r\n")  # only the line break is framing
        if not expression.strip():
            return
        t = time.perf_counter()
        res = loop_run(model, expression)
        n_lat = sum(tk.latent for tk in res.tokens)
        log.info("%d latent positions, %d iterations, %d bytes written in %.2fs", n_lat,
                 sum(tk.iterations for tk in res.tokens), len(res.answer), time.perf_counter() - t)
        if args.steps:
            for k, (text, n, iters) in enumerate(copies(model, res), 1):
                sys.stdout.write(f"  step {k:>3}: {text!r}  ({n} positions, {iters} iterations)\n")
        sys.stdout.write(f"{expression}={bytes(res.answer).decode(errors='replace')}")
        if not res.finished:
            sys.stdout.write("  (a budget ran out before the end byte)\n")
        sys.stdout.flush()

    if args.expression:
        for e in args.expression:
            one(e)
        return 0
    try:
        while True:
            one(input("> "))
    except (EOFError, KeyboardInterrupt):
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
