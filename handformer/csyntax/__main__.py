"""python -m handformer.csyntax file.c   -> the source with one state letter per byte, aligned below it."""
import argparse
import logging
import sys

from . import build, solve

log = logging.getLogger("handformer.csyntax")


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m handformer.csyntax", description="Lex C source with the hand-written model")
    ap.add_argument("file", nargs="?", help="a C file (default: stdin)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    src = open(args.file).read() if args.file else sys.stdin.read()
    letters = solve(build(), src)
    for line_src, line_out in zip(src.split("\n"), _split_like(letters, src)):
        log.info("%s\n%s", line_src, line_out)


def _split_like(letters: str, src: str) -> list[str]:
    out, i = [], 0
    for line in src.split("\n"):
        out.append(letters[i:i + len(line)])
        i += len(line) + 1
    return out


if __name__ == "__main__":
    main()
