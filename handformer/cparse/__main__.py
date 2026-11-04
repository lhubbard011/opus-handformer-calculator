"""python -m handformer.cparse "if (a) { b + 1; }"   -> the model's letters, parse records and the tree."""
import argparse
import logging
import sys

from . import build, circuit, grammar, solve, tokens

log = logging.getLogger("handformer.cparse")


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m handformer.cparse",
                                 description="Lex and parse C with the hand-written parser (stage 2: the toy grammar)")
    ap.add_argument("source", nargs="*", help="programs (default: stdin), e.g. 'if (a) { b + 1; } else ;'")
    ap.add_argument("--records", action="store_true", help="also show the letters and every record the model writes")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    m = build()
    for src in args.source or [sys.stdin.read()]:
        letters, records = circuit.split(solve(m, src))
        if args.records:
            log.info("  %s\n  %s", src.replace("\n", " "), letters)
            for rec in records.rstrip("\n").split(grammar.SEP):
                if rec:
                    log.info("  %s", rec)
        log.info("%s\n  %s", src, grammar.tree(records, tokens.texts(src, letters)))


if __name__ == "__main__":
    main()
