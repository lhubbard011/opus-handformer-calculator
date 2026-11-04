"""Where each part of the model is written in the source, as GitHub links.

Every ``HeadSpec`` / ``UnitSpec`` records the circuit line that wrote it
(``model.authored_at``). From that, every residual dim gets the lines of its
writers. The repository URL and branch come from ``docs/links.toml`` (the same
file the docs link generator uses), so there is one place to change them.
Documentation only: nothing here touches a weight.
"""
from __future__ import annotations

import inspect
import logging
import tomllib
from pathlib import Path

from .. import common as cm
from ..model import ModelBuilder, Transformer

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
LINKS = ROOT / "docs" / "links.toml"


def repo() -> dict | None:
    """{"url", "ref"} from docs/links.toml, or None when it is missing (e.g. an installed copy)."""
    try:
        with open(LINKS, "rb") as f:
            r = tomllib.load(f)["repo"]
        return {"url": r["url"].rstrip("/"), "ref": r["ref"]}
    except (OSError, KeyError, tomllib.TOMLDecodeError) as e:
        log.warning("no repository links (%s): %s", LINKS, e)
        return None


def _def_site(obj) -> str:
    src = inspect.getsourcefile(obj)
    if src is None:
        return ""
    path = Path(src).resolve().relative_to(ROOT).as_posix()
    return f"{path}:{inspect.getsourcelines(obj)[1]}"


def dim_sites(builder: ModelBuilder) -> dict[str, list[str]]:
    """dim -> source lines of every head and unit that writes it (input dims: the embedding)."""
    sites: dict[str, list[str]] = {n: [] for n in builder.layout.names}

    def note(d: str, where: str) -> None:
        if where and where not in sites[d]:
            sites[d].append(where)

    embed, positional = _def_site(cm.embed), _def_site(Transformer.positional)
    for n in cm.INPUT_DIMS:
        note(n, positional if n in ("pos", "pos2") else embed)
    for spec in builder.layers:
        for h in spec.heads:
            for dst in h.value.values():
                for d in dst:
                    note(d, h.where)
        for u in spec.units:
            for d in u.write:
                note(d, u.where)
    missing = [n for n, s in sites.items() if not s]
    if missing:
        log.warning("%d dims have no source line: %s", len(missing), missing[:8])
    return sites


def url(where: str, r: dict) -> str:
    path, line = where.rsplit(":", 1)
    return f"{r['url']}/blob/{r['ref']}/{path}#L{line}"
