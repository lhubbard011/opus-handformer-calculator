"""Build the static site (GitHub Pages): the visualizer with the model running in the page.

    python tools/build_site.py [OUT_DIR]        (default: site/)

OUT_DIR/index.html          the visualizer page, flagged to run the model locally
OUT_DIR/static/*            the same frontend files the Python server serves
OUT_DIR/data/<model>/       model.json (every nonzero weight, tools/export_web.py) + info.json (= server
                            /api/info), for each model in handformer/viz/models.yaml
OUT_DIR/data/models.json    the drop-down (= server /api/models)
OUT_DIR/handformer-weights.npz   raw weights download (handformer/export.py)
OUT_DIR/handformer-weights.safetensors   the same weights as one safetensors file

No build step, no bundler: the browser loads plain ES modules. Serve OUT_DIR
with any static file server (`python -m http.server -d site`).
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import export_web  # noqa: E402
from handformer import export as npz_export  # noqa: E402
from handformer.viz.models import MODELS, shown  # noqa: E402

log = logging.getLogger("build_site")
STATIC = ROOT / "handformer" / "viz" / "static"
FLAG = '<script>window.HANDFORMER_STATIC = "data/";</script>\n'
FRONTEND = ("app.js", "engine.js", "local-api.js", "heatmap.js", "render.js", "graph.js", "style.css")


def build(out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    (out / "static").mkdir(parents=True)
    for name in FRONTEND:
        shutil.copy2(STATIC / name, out / "static" / name)
    html = (STATIC / "index.html").read_text()
    anchor = '<script type="module" src="static/app.js"></script>'
    if anchor not in html:
        raise RuntimeError(f"index.html no longer contains {anchor!r}; update build_site.py")
    (out / "index.html").write_text(html.replace(anchor, FLAG + "  " + anchor))
    (out / ".nojekyll").write_text("")  # serve files as-is
    keys = shown()  # models.yaml: only these are offered, so only these are exported
    for key in keys:  # data/calc/, data/life/, ...: model.json + info.json each
        export_web.export(out / "data" / key, key)
    menu = {"models": [[k, MODELS[k].title] for k in keys]}  # = the server's /api/models
    (out / "data" / "models.json").write_text(json.dumps(menu))
    npz_export.export(out / "handformer-weights.npz")
    npz_export.export(out / "handformer-weights.safetensors")
    files = sorted(p for p in out.rglob("*") if p.is_file())
    log.info("site built in %s: %d files, %d bytes", out, len(files), sum(p.stat().st_size for p in files))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path, nargs="?", default=ROOT / "site")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    build(args.out)


if __name__ == "__main__":
    main()
