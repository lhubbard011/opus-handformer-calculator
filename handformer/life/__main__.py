"""python -m handformer.life "3:.#./.#./.#."   -> the model writes 3 generations."""
import argparse
import logging

from . import build, solve

log = logging.getLogger("handformer.life")


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m handformer.life",
                                 description="Run Game of Life generations: N:<rows of . and # separated by />")
    ap.add_argument("grids", nargs="+", help='e.g. "3:.#./.#./.#." (a blinker, 3 generations)')
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    m = build()
    for g in args.grids:
        out = solve(m, g)
        log.info("%s=%s", g, out)
        for block in out.split("="):
            if ":" in block:
                k, grid = block.split(":", 1)
                log.info("  generation (count %s):", k)
                for row in grid.split("/"):
                    log.info("    %s", row)


if __name__ == "__main__":
    main()
