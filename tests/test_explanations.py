"""Every part of the model has a real explanation and a source line.

A reader of the visualizer or the docs should never meet an unexplained part:
every residual dim has a specific doc (not the generic "private working value"
fallback), every MLP unit a hand-written label, every head a description, and
every head and unit the source line that wrote it.
"""

import logging
import unittest

from handformer import arith
from handformer.viz.glossary import describe_dim, describe_head

log = logging.getLogger(__name__)


class TestExplanations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b = arith.builder()

    def test_every_dim_has_a_specific_doc(self):
        names = self.b.layout.names
        missing = [n for n in names if not describe_dim(n) or "private working value" in describe_dim(n)]
        log.info("%d dims, %d without a specific doc", len(names), len(missing))
        self.assertEqual(missing, [], "add a doc in handformer/viz/glossary.py or op_dims.json")

    def test_every_unit_has_a_label_and_source(self):
        units = [(li, ui, u) for li, l in enumerate(self.b.layers) for ui, u in enumerate(l.units)]
        unlabeled = [f"L{li + 1} #{ui} {u.where}" for li, ui, u in units if not u.label.strip()]
        no_src = [f"L{li + 1} #{ui}" for li, ui, u in units if not u.where]
        log.info("%d units, %d unlabeled, %d without source", len(units), len(unlabeled), len(no_src))
        self.assertEqual(unlabeled, [], "pass label= where the unit is built")
        self.assertEqual(no_src, [])

    def test_every_head_has_a_description_and_source(self):
        heads = [h for l in self.b.layers for h in l.heads]
        bad = [h.name for h in heads if not describe_head(h) or not h.where]
        log.info("%d heads, %d without description or source", len(heads), len(bad))
        self.assertEqual(bad, [])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
