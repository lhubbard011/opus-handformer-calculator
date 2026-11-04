"""The hello-world adder (handformer/hello.py): exact on every pair below 100, long and random operands."""

import logging
import random
import unittest

import numpy as np

from handformer import common as cm, hello

log = logging.getLogger(__name__)


def expected(a: int, b: int, sa: str, sb: str) -> str:
    s = a + b
    width = max(len(sa), len(sb)) + (len(str(s)) > max(len(sa), len(sb)))
    return f"{str(s).zfill(width)[::-1]};{s}"


class TestHello(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = hello.build()

    def check(self, sa: str, sb: str) -> None:
        """Teacher-forced: the argmax at every output position is the next expected byte."""
        p = hello.prompt(f"{sa}+{sb}")
        want = expected(int(sa), int(sb), sa, sb) + "\n"
        seq = p + list(want.encode())
        pred = np.argmax(self.m.logits(seq[:-1]), axis=1)[len(p) - 1:]
        got = bytes(int(t) for t in pred).decode("ascii", "replace")
        self.assertEqual(got, want, f"{sa}+{sb}")

    def test_all_pairs_below_100(self):
        for a in range(100):
            for b in range(100):
                self.check(str(a), str(b))
        log.info("10000 pairs exact")

    def test_long_and_random(self):
        rng = random.Random(7)
        cases = [("9" * 40, "1"), ("1", "9" * 40), ("5" * 30, "5" * 30)]
        cases += [(str(rng.randrange(10 ** rng.randint(1, 30))), str(rng.randrange(10 ** rng.randint(1, 30))))
                  for _ in range(200)]
        for sa, sb in cases:
            self.check(sa, sb)

    def test_greedy_decoding(self):
        for s, want in (("47+38", "58;85"), ("99999+1", "000001;100000"), ("0+0", "0;0")):
            self.assertEqual(hello.solve(self.m, s), want)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
