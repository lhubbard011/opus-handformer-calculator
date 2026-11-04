"""python -m handformer.cparse "i(x){x+1;}"   -> the model's parse records and the tree they spell."""
import argparse
import logging
import sys

from . import build, grammar, solve

log = logging.getLogger("handformer.cparse")


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m handformer.cparse",
                                 description="Parse with the hand-written parser (stage 1: the toy grammar)")
    ap.add_argument("source", nargs="*", help="programs (default: stdin), e.g. 'i(x){x+1;} e ;'")
    ap.add_argument("--records", action="store_true", help="also show every record the model writes")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    m = build()
    for src in args.source or [sys.stdin.read()]:
        out = solve(m, src)
        if args.records:
            for rec in out.rstrip("\n").split(grammar.SEP):
                if rec:
                    log.info("  %s", rec)
        log.info("%s\n  %s", src, grammar.tree(out))


if __name__ == "__main__":
    main()
