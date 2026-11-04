"""Command line: ``python -m handformer "2+3*(4-1)"`` (or no argument for a prompt loop).

Builds the hand-written model, feeds ``\\n<expression>=`` and greedy-decodes
until ``\\n``. The model's bytes are written to stdout exactly as produced;
the answer is the text after the last ``]``. Diagnostics go to the log
(``-v`` for per-step detail).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from . import arith

log = logging.getLogger("handformer")


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m handformer")
    ap.add_argument("expression", nargs="*", help="e.g. 2+3*(4-1); omit for an interactive loop")
    ap.add_argument("-v", "--verbose", action="store_true", help="log model build and decoding")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    t0 = time.perf_counter()
    model = arith.build()
    log.info("model built in %.2fs", time.perf_counter() - t0)

    def run(expression: str) -> None:
        expression = expression.strip().replace(" ", "")
        if not expression:
            return
        t = time.perf_counter()
        out = arith.solve(model, expression)
        log.info("decoded %d bytes in %.2fs", len(out), time.perf_counter() - t)
        sys.stdout.write(f"{expression}={out}")
        sys.stdout.write(f"  -> {out.rsplit(']', 1)[-1]}")
        sys.stdout.flush()

    if args.expression:
        for e in args.expression:
            run(e)
        return 0
    try:
        while True:
            run(input("> "))
    except (EOFError, KeyboardInterrupt):
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
