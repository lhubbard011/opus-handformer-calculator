"""The hand-built language models (handformer/lm): their logits are exactly the counted
log-probabilities, and their scores are exactly the counted scores.

These tests count from a fixed in-repo text, so they need no download; the Shakespeare corpus
(data/, fetched by tools/fetch_corpus.py) is checked too when it is present."""

import logging
import unittest

import numpy as np

from handformer.lm import circuit, corpus, evaluate, ngram

log = logging.getLogger(__name__)

TEXT = (b"To be, or not to be, that is the question:\nWhether 'tis nobler in the mind to suffer\n"
        b"The slings and arrows of outrageous fortune,\nOr to take arms against a sea of troubles\n") * 3


class TestBigram(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.test = TEXT[:300], TEXT[300:]
        cls.bg = ngram.fit_bigram(cls.train)
        cls.m = circuit.build(cls.bg)

    def test_logits_are_the_counted_log_probabilities(self):
        """For every byte: softmax of the model's logits == the smoothed count table's row."""
        lp = evaluate.log_softmax(self.m.logits(list(range(256))))
        np.testing.assert_allclose(lp, self.bg.logp, atol=1e-12)
        np.testing.assert_allclose(np.exp(self.bg.logp).sum(1), 1.0, atol=1e-12)

    def test_a_count_by_hand(self):
        """After 't' in the training text: P('o' | 't') = (count + alpha) / (count('t') + 256 alpha)."""
        t, o = ord("t"), ord("o")
        n_to = sum(1 for i in range(len(self.train) - 1) if self.train[i] == t and self.train[i + 1] == o)
        n_t = sum(1 for i in range(len(self.train) - 1) if self.train[i] == t)
        want = np.log((n_to + self.bg.alpha) / (n_t + 256 * self.bg.alpha))
        self.assertAlmostEqual(float(self.m.logits([t])[0, o]), want, places=12)

    def test_unseen_byte_is_uniform(self):
        lp = evaluate.log_softmax(self.m.logits([ord("Z")]))[0]  # 'Z' never occurs in the text
        np.testing.assert_allclose(lp, np.log(1 / 256), atol=1e-12)

    def test_score_matches_the_table(self):
        """bits per byte through the weights == straight from the counts (one window: same bytes scored)."""
        self.assertAlmostEqual(evaluate.bits_per_byte(self.m, self.test, ctx=len(self.test)),
                               ngram.bits_bigram(self.bg.logp, self.test[1:]), places=10)  # bytes 3 onward
        self.assertLess(ngram.bits_bigram(self.bg.logp, self.test), 8.0)

    def test_alpha_is_chosen_without_the_test_text(self):
        self.assertIn(self.bg.alpha, ngram.ALPHAS)
        self.assertEqual(ngram.choose_alpha(self.train), self.bg.alpha)

    def test_every_byte_has_its_own_dim_name(self):
        names = [circuit.now(b) for b in range(256)]
        self.assertEqual(len(set(names)), 256)


@unittest.skipUnless((corpus.ROOT / corpus.CORPORA["shakespeare"].path).exists(),
                     "Tiny Shakespeare not fetched (python tools/fetch_corpus.py)")
class TestShakespeare(unittest.TestCase):
    def test_stage0_scores(self):
        train, test = corpus.split(corpus.load("shakespeare"))
        bg = ngram.fit_bigram(train)
        bpb = evaluate.bits_per_byte(circuit.build(bg), test)
        log.info("Tiny Shakespeare stage 0: %.4f bits per byte", bpb)
        self.assertAlmostEqual(bpb, ngram.bits_bigram(bg.logp, test), delta=0.002)  # windows drop 1 byte in 512
        self.assertLess(bpb, 3.65)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
