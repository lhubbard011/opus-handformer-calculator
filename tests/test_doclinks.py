"""tools/doclinks.py: references resolve to the right line, stale links are refreshed,
code spans are left alone, bad keys are reported; and the repo's own docs are current."""

import logging
import sys
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import doclinks  # noqa: E402

log = logging.getLogger(__name__)


def resolver(**named) -> doclinks.Resolver:
    with open(doclinks.MAP, "rb") as f:
        table = tomllib.load(f)
    table["src"] = named
    return doclinks.Resolver(table)


def line_of(path: str, text: str) -> int:
    return next(i + 1 for i, line in enumerate((ROOT / path).read_text().splitlines()) if text in line)


class TestDocLinks(unittest.TestCase):
    def test_named_reference_points_at_the_matching_line(self):
        res = resolver(carry={"file": "handformer/addition.py", "match": '{"carry": 1.0})  # [y >= 10]'})
        errors: list[str] = []
        out, changed, found = doclinks.rewrite("see [](src:carry).", res, "t.md", errors)
        n = line_of("handformer/addition.py", '{"carry": 1.0})  # [y >= 10]')
        self.assertEqual((errors, changed, found), ([], 1, 1))
        self.assertIn(f"[addition.py:{n}](", out)
        self.assertIn(f'/blob/{res.ref}/handformer/addition.py#L{n} "src:carry")', out)

    def test_stale_link_is_refreshed_and_current_link_is_kept(self):
        res = resolver(carry={"file": "handformer/addition.py", "match": '{"carry": 1.0})  # [y >= 10]'})
        good = res.link("carry")
        stale = good.replace("#L", "#L9").replace("addition.py:", "addition.py:9")
        errors: list[str] = []
        out, changed, _ = doclinks.rewrite(f"{stale} and {good}", res, "t.md", errors)
        self.assertEqual((errors, changed, out), ([], 1, f"{good} and {good}"))

    def test_code_spans_are_left_alone(self):
        errors: list[str] = []
        text = "write `[](src:KEY)` like this\n```\n[](src:nope)\n```\n"
        out, changed, found = doclinks.rewrite(text, resolver(), "t.md", errors)
        self.assertEqual((out, changed, found, errors), (text, 0, 0, []))

    def test_dim_reference_points_at_a_line_that_writes_it(self):
        res = resolver()
        path, line = res.where("dim:inB").rsplit(":", 1)
        self.assertIn("inB", (ROOT / path).read_text().splitlines()[int(line) - 1])

    def test_bad_references_are_reported(self):
        res = resolver(twice={"file": "handformer/addition.py", "match": "carry"})
        errors: list[str] = []
        doclinks.rewrite("[](src:missing) [](src:twice) [](src:dim:no_such_dim)", res, "t.md", errors)
        self.assertEqual(len(errors), 3, errors)
        self.assertIn("unknown key 'missing'", errors[0])
        self.assertIn("need exactly 1", errors[1])
        self.assertIn("no residual dim named 'no_such_dim'", errors[2])

    def test_repo_docs_are_current(self):
        with open(doclinks.MAP, "rb") as f:
            res = doclinks.Resolver(tomllib.load(f))
        for name in doclinks.DOCS:
            errors: list[str] = []
            _, changed, found = doclinks.rewrite((ROOT / name).read_text(), res, name, errors)
            log.info("%s: %d links, %d stale", name, found, changed)
            self.assertEqual(errors, [])
            self.assertEqual(changed, 0, f"{name} has stale links: run python tools/doclinks.py")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
