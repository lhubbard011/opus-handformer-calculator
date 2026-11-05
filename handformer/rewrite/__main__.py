"""Command line: ``python -m handformer.rewrite "2*(3+4)-1"`` (or no argument for a prompt loop).

Builds the looped calculator, feeds ``\\n<expression>=``, runs the recurrent block until its
router stops it, then greedy-decodes the coda until ``\\n``. What the model writes is the
answer and nothing else. ``--passes`` also prints the tape after every pass (the state the
block left in the prompt's positions); ``-v`` logs the build and the loop.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from . import builder, run as loop_run

log = logging.getLogger("handformer.rewrite")


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m handformer.rewrite")
    ap.add_argument("expression", nargs="*", help="e.g. 2*(3+4)-1 or 355/113~6; omit for an interactive loop")
    ap.add_argument("--passes", action="store_true", help="print the tape going into every pass")
    ap.add_argument("-v", "--verbose", action="store_true", help="log the model build and the loop")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    t0 = time.perf_counter()
    model = builder().build()
    log.info("looped model built in %.2fs", time.perf_counter() - t0)

    def one(expression: str) -> None:
        expression = expression.strip("\r\n")  # only the line break is framing
        if not expression.strip():
            return
        t = time.perf_counter()
        res = loop_run(model, expression)
        log.info("%d passes and %d answer bytes in %.2fs", len(res.passes), len(res.answer), time.perf_counter() - t)
        if args.passes:
            for p in res.passes:
                sys.stdout.write(f"  pass {p.index + 1:>4}: {bytes(p.tape[1:-1]).decode(errors='replace')!r}"
                                 f"{'  (room: a blank is added)' if p.grew else ''}\n")
        sys.stdout.write(f"{expression}={bytes(res.answer).decode(errors='replace')}")
        if not res.halted:
            sys.stdout.write("  (the pass budget ran out before the router stopped the loop)\n")
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
