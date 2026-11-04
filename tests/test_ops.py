"""Single operations through the full expression model (one frame each)."""

import logging
import random
import unittest

from tests.helpers import check, model, rand_digits

log = logging.getLogger(__name__)


class TestSingleOps(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model()

    def test_exhaustive_small(self):
        for op in "+-*":
            for a in range(40):
                for b in range(40):
                    check(f"{a}{op}{b}")
            log.info("%s: all a,b < 40 ok", op)

    def test_signed_operands(self):
        for op in "+-*":
            for a in range(-12, 13):
                for b in range(-12, 13):
                    check(f"{a}{op}{b}")
            log.info("%s: all signed a,b in -12..12 ok", op)

    def test_edges(self):
        for e in ["0+0", "0-0", "0*0", "999+1", "1-1000", "1000-1", "007+5", "5-007",
                  "0*123", "9999*9999", "10*100", "1-10000000"]:
            check(e)

    def test_random_long(self):
        rng = random.Random(3)
        for op, length in (("+", 300), ("-", 300), ("*", 25)):
            for _ in range(6):
                check(f"{rand_digits(rng, rng.randint(1, length))}{op}"
                      f"{rand_digits(rng, rng.randint(1, length))}")
            log.info("%s random length<=%d ok", op, length)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    unittest.main()
