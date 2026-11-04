"""Download a language-model corpus into data/ and check its SHA-256 (handformer/lm/corpus.py).

    python tools/fetch_corpus.py [shakespeare]
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from handformer.lm.corpus import CORPORA, ROOT  # noqa: E402

log = logging.getLogger("fetch_corpus")


def fetch(name: str) -> Path:
    c = CORPORA[name]
    path = ROOT / c.path
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == c.sha256:
        log.info("%s already present and verified: %s", name, path)
        return path
    log.info("downloading %s from %s", name, c.url)
    with urllib.request.urlopen(c.url, timeout=60) as r:  # noqa: S310  (a fixed https URL)
        data = r.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != c.sha256:
        raise ValueError(f"downloaded {len(data)} bytes with SHA-256 {digest}, expected {c.sha256}: not saved")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    log.info("saved %d bytes to %s (verified)", len(data), path)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", default="shakespeare", choices=sorted(CORPORA))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    fetch(args.name)


if __name__ == "__main__":
    main()
