"""Keep the docs' links to source lines correct.

In any Markdown file, write a reference with an empty or any text:

    [](src:add.carry)        a named location from docs/links.toml
    [](src:dim:inB)          where a residual dim is written (from the model itself)

and run

    python tools/doclinks.py            rewrite README.md and DESIGN.md in place
    python tools/doclinks.py --check    exit 1 if any link is stale or broken (CI)

Each reference becomes an ordinary link to the current line on GitHub, keeping
its key in the link title so it can be found and refreshed later:

    [addition.py:65](https://github.com/…/blob/main/handformer/addition.py#L65 "src:add.carry")

Named locations store a file and a piece of text that must occur on exactly one
line of it, never a line number, so edits to the code move the links with it.
Dim locations come from the source line every head and unit records when it is
built (``model.authored_at``). A dim written in several places links to the
first; ``src:dim:NAME`` therefore needs the model to build (numpy).
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAP = ROOT / "docs" / "links.toml"
DOCS = ("README.md", "DESIGN.md", "HELLO.md")

log = logging.getLogger("doclinks")

# Code spans and fenced blocks are left alone, so docs can show the syntax itself.
CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.S)
# [text](src:key) or [text](url "src:key")
REF = re.compile(r'\[(?P<text>[^\]]*)\]\((?:src:(?P<k1>[\w.:-]+)|[^)\s]+ "src:(?P<k2>[\w.:-]+)")\)')


class Resolver:
    def __init__(self, table: dict) -> None:
        self.repo = table["repo"]["url"].rstrip("/")
        self.ref = table["repo"]["ref"]
        self.named = table.get("src", {})
        self._dims: dict[str, list[str]] | None = None
        self.used: set[str] = set()

    def _dim_sites(self) -> dict[str, list[str]]:
        if self._dims is None:
            sys.path.insert(0, str(ROOT))
            from handformer import arith
            from handformer.viz.sources import dim_sites
            sites = dim_sites(arith.builder())
            log.info("built the model for dim locations: %d dims", len(sites))
            self._dims = sites
        return self._dims

    def where(self, key: str) -> str:
        """``path:line`` for a key, or raise ValueError saying why not."""
        self.used.add(key)
        if key.startswith("dim:"):
            name = key[4:]
            sites = self._dim_sites().get(name)
            if not sites:
                raise ValueError(f"no residual dim named {name!r}")
            return sites[0]
        entry = self.named.get(key)
        if entry is None:
            raise ValueError(f"unknown key {key!r} (add it to {MAP.relative_to(ROOT)})")
        path = ROOT / entry["file"]
        lines = path.read_text().splitlines()
        hits = [i + 1 for i, line in enumerate(lines) if entry["match"] in line]
        if len(hits) != 1:
            raise ValueError(f"{key}: {entry['match']!r} is on {len(hits)} lines of {entry['file']} "
                             f"{hits[:5]} (need exactly 1)")
        return f"{entry['file']}:{hits[0]}"

    def link(self, key: str) -> str:
        w = self.where(key)
        path, line = w.rsplit(":", 1)
        return f'[{Path(path).name}:{line}]({self.repo}/blob/{self.ref}/{path}#L{line} "src:{key}")'


def rewrite(text: str, res: Resolver, name: str, errors: list[str]) -> tuple[str, int, int]:
    """(new text, links changed, links found), outside code spans only."""
    count = found = 0
    offset = 0

    def one(m: re.Match) -> str:
        nonlocal count, found
        found += 1
        key = m["k1"] or m["k2"]
        try:
            new = res.link(key)
        except ValueError as e:
            line = text.count("\n", 0, offset + m.start()) + 1
            errors.append(f"{name}:{line}: {e}")
            return m[0]
        count += new != m[0]
        return new

    out = []
    for i, part in enumerate(CODE.split(text)):
        out.append(part if i % 2 else REF.sub(one, part))
        offset += len(part)
    return "".join(out), count, found


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("files", nargs="*", default=list(DOCS), help="Markdown files (default: README.md DESIGN.md)")
    ap.add_argument("--check", action="store_true", help="change nothing; exit 1 if any link would change")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    with open(MAP, "rb") as f:
        res = Resolver(tomllib.load(f))
    errors: list[str] = []
    stale = 0
    for name in args.files:
        path = ROOT / name
        text = path.read_text()
        new, changed, refs = rewrite(text, res, name, errors)
        log.info("%s: %d source links, %d %s", name, refs, changed, "stale" if args.check else "updated")
        stale += changed
        if changed and not args.check:
            path.write_text(new)
    unused = sorted(set(res.named) - res.used)
    if unused:
        log.warning("keys in %s that no doc uses: %s", MAP.name, ", ".join(unused))
    for e in errors:
        log.error(e)
    if errors or (args.check and stale):
        if stale:
            log.error("%d link(s) out of date: run python tools/doclinks.py", stale)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
