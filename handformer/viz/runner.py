"""Runs the hand-written model and records everything needed to step through it.

One run = greedy decoding of an expression, then one traced forward pass over
the whole generated sequence. Because the model is causal, position q of that
single pass is exactly the computation that produced byte q+1, so every step
of the algorithm is available from one trace.

The trace is the model's own forward pass (``Transformer.residual`` with a
``Tracer``), not a reimplementation.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from .. import arith, common as cm, expr, frame
from ..model import Form, Tracer
from .glossary import describe_dim, describe_head

log = logging.getLogger(__name__)

MAX_TOKENS = 3000
MAX_RUNS = 4
TOP_KEYS = 3
TOP_LOGITS = 5

# Shared dims shown on the sequence strip as pointers for the selected step.
MARK_DIMS = ("Ls", "F", "p", "c", "o", "a_lo", "b_hi", "E", "S", "Rb", "P", "R")
SUMMARY_DIMS = (*MARK_DIMS, "phase1", "phase2", "phase4", "act_add", "act_sub", "act_mul", "act_div", "act_pow",
                "open", "seen_eq", "seen_semi", "seen_bar", "drop", "cont", "end2", "end4",
                "in1", "in12", "in123", "src4", "neg", "use_mul")


def _num(c: float) -> str:
    if c == int(c) and abs(c) < 1e5:
        return str(int(c))
    return f"{c:.4g}"


def form_str(form: Form) -> str:
    """Human-readable linear form: ``{"P": 1, "F": -1, "one": 1}`` -> ``P - F + 1``."""
    terms, const = [], 0.0
    for d, c in form.items():
        if c == 0:
            continue
        if d == "one":
            const += c
            continue
        mag = "" if abs(c) == 1 else f"{_num(abs(c))}·"
        terms.append(("-" if c < 0 else "+", f"{mag}{d}"))
    if const:
        terms.append(("-" if const < 0 else "+", _num(abs(const))))
    if not terms:
        return "0"
    s = ("-" if terms[0][0] == "-" else "") + terms[0][1]
    for sign, t in terms[1:]:
        s += f" {sign} {t}"
    return s


def _category(name: str) -> str:
    if name in cm.INPUT_DIMS:
        return "input"
    if name in frame.DIMS:
        return "frame"
    if name in expr.DIMS:
        return "parser"
    if name in expr.OUT_DIMS or name in arith.OP_OUTPUTS:
        return "output"
    return name.split(".", 1)[0] if "." in name else "other"


def _circuit(write_dims: list[str], read_dims: list[str]) -> str:
    """Which circuit a head/unit belongs to: the category of the dims it writes."""
    cats = [_category(d) for d in write_dims]
    owner = next((c for c in cats if c not in ("output", "input")), None)
    if owner:
        return owner
    ns = next((d.split(".", 1)[0] for d in read_dims if "." in d), None)
    return ns or "parser"


@dataclass
class _Recorder(Tracer):
    """Keeps residual snapshots, top attention keys per head, MLP pre-activations."""

    snapshots: list[tuple[str, np.ndarray]] = field(default_factory=list)
    top_idx: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    top_w: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    pre: dict[int, np.ndarray] = field(default_factory=dict)

    def residual(self, label: str, x: np.ndarray) -> None:
        self.snapshots.append((label, x.copy()))

    def attention(self, layer: int, head: int, attn: np.ndarray) -> None:
        k = min(TOP_KEYS, attn.shape[1])
        idx = np.argsort(-attn, axis=1, kind="stable")[:, :k]  # ties: earlier key first
        self.top_idx[(layer, head)] = idx
        self.top_w[(layer, head)] = np.take_along_axis(attn, idx, axis=1)

    def mlp(self, layer: int, pre: np.ndarray) -> None:
        self.pre[layer] = pre.copy()


@dataclass
class Run:
    run_id: str
    expression: str
    tokens: list[int]
    prompt_len: int
    rec: _Recorder
    logits: np.ndarray


class Runner:
    def __init__(self) -> None:
        t0 = time.perf_counter()
        self.builder = arith.builder()
        self.model = self.builder.build()
        self.layout = self.model.layout
        self.names = self.layout.names
        self.runs: OrderedDict[str, Run] = OrderedDict()
        # Identifies this exact model, so a page loaded against an older server can tell.
        self.model_id = (f"d{len(self.layout)}-L{len(self.model.layers)}-h{sum(len(l.heads) for l in self.model.layers)}"
                         f"-u{sum(l.w_in.shape[1] for l in self.model.layers)}-nz{self.model.nonzero_params()}")
        log.info("model built in %.2fs: %d layers, d_model=%d, %d nonzero params",
                 time.perf_counter() - t0, len(self.model.layers), len(self.layout),
                 self.model.nonzero_params())

    # ------------------------------------------------------------- static info

    def _wiring(self) -> dict[str, dict[str, list[str]]]:
        """For every dim: which sublayers write it and which read it (from the weight specs)."""
        wires = {n: {"writers": [], "readers": []} for n in self.names}
        def note(dim, role, where):
            lst = wires[dim][role]
            if where not in lst:
                lst.append(where)
        for n in cm.INPUT_DIMS:
            note(n, "writers", "the position features" if n in ("pos", "pos2") else "the byte embedding")
        for li, spec in enumerate(self.builder.layers):
            for h in spec.heads:
                where = f"L{li + 1} head '{h.name}'"
                for f in [*h.query, *h.key]:
                    for d in f:
                        note(d, "readers", where)
                for src, dst in h.value.items():
                    note(src, "readers", where)
                    for d in dst:
                        note(d, "writers", where)
            for u in spec.units:
                where = f"L{li + 1} MLP"
                for d in u.read:
                    note(d, "readers", where)
                for d in u.write:
                    note(d, "writers", where)
        return wires

    @staticmethod
    def _summ(items: list[str], limit: int = 4) -> str:
        if not items:
            return "nothing"
        more = len(items) - limit
        return ", ".join(items[:limit]) + (f" and {more} more" if more > 0 else "")

    def info(self) -> dict:
        wires = self._wiring()
        layers = []
        for li, spec in enumerate(self.builder.layers):
            heads = [{
                "name": h.name,
                "doc": describe_head(h),
                "circuit": _circuit([d for f in h.value.values() for d in f],
                                    [d for f in h.query for d in f]),
                "query": [form_str(f) for f in h.query],
                "key": [form_str(f) for f in h.key],
                "value": {src: form_str(dst) for src, dst in h.value.items()},
            } for h in spec.heads]
            units = [{"read": form_str(u.read), "write": form_str(u.write), "label": u.label,
                      "doc": u.label or ("Writes " + "; ".join(f"{d}: {describe_dim(d) or 'working value'}"
                                                              for d in u.write)),
                      "circuit": _circuit(list(u.write), list(u.read))} for u in spec.units]
            layers.append({"index": li, "name": spec.name, "heads": heads, "units": units})
        return {
            "name": "handformer",
            "model_id": self.model_id,
            "d_model": len(self.layout),
            "nonzero_params": self.model.nonzero_params(),
            "dims": [{"name": n, "cat": _category(n), "doc": describe_dim(n),
                      "written_by": self._summ(wires[n]["writers"]), "read_by": self._summ(wires[n]["readers"])}
                     for n in self.names],
            "layers": layers,
        }

    # ------------------------------------------------------------- runs

    def run(self, expression: str) -> dict:
        expression = expression.replace(" ", "").removesuffix("=")  # the page adds the '=' itself
        if not expression:
            raise ValueError("empty expression")
        bad = sorted({ch for ch in expression if ord(ch) not in cm.ALPHABET and ch != "~"})
        if bad:
            raise ValueError(f"unsupported characters: {''.join(bad)!r} (use digits . + - * / ( ) and ~N)")
        prompt = arith.prompt(expression)
        t0 = time.perf_counter()
        budget = MAX_TOKENS - len(prompt)  # the model stops itself at '\n' 
        tokens = self.model.generate(prompt, stop=cm.EOS, max_new=budget)
        t1 = time.perf_counter()
        rec = _Recorder()
        logits = self.model.residual(tokens, trace=rec) @ self.model.unembed
        t2 = time.perf_counter()
        run = Run(uuid.uuid4().hex[:12], expression, tokens, len(prompt), rec, logits)
        self.runs[run.run_id] = run
        while len(self.runs) > MAX_RUNS:
            evicted, _ = self.runs.popitem(last=False)
            log.info("evicted run %s", evicted)
        finished = tokens[-1] == cm.EOS
        log.info("run %s %r: %d bytes generated in %.2fs, traced in %.2fs, finished=%s",
                 run.run_id, expression, len(tokens) - len(prompt), t1 - t0, t2 - t1, finished)
        return {
            "model_id": self.model_id,
            "run_id": run.run_id,
            "expression": expression,
            "chars": [chr(t) for t in tokens],
            "prompt_len": run.prompt_len,
            "finished": finished,
            "steps": [self._step(run, q) for q in range(len(tokens))],
            "snapshots": [label for label, _ in rec.snapshots],
        }

    def _get(self, run_id: str) -> Run:
        run = self.runs.get(run_id)
        if run is None:
            raise ValueError(f"unknown or evicted run {run_id}")
        return run

    def _val(self, run: Run, q: int, dim: str) -> float:
        return float(run.rec.snapshots[-1][1][q, self.layout[dim]])

    def _step(self, run: Run, q: int) -> dict:
        """Raw per-step values; the page turns them into phase, explanation and pointer tags."""
        x = run.rec.snapshots[-1][1][q]
        return {"pred": chr(int(np.argmax(run.logits[q]))),
                "v": {d: round(float(x[self.layout[d]]), 6) for d in SUMMARY_DIMS}}

    def position(self, run_id: str, q: int) -> dict:
        run = self._get(run_id)
        if not 0 <= q < len(run.tokens):
            raise ValueError(f"position {q} out of range 0..{len(run.tokens) - 1}")
        rec = run.rec
        values = np.stack([x[q] for _, x in rec.snapshots], axis=1)  # (d, n_snapshots)

        heads = []
        for li, layer in enumerate(self.model.layers):
            x_in = rec.snapshots[2 * li][1][: q + 1]  # residual entering layer li, keys 0..q
            for hi, h in enumerate(layer.heads):
                idx = rec.top_idx[(li, hi)][q]
                w = rec.top_w[(li, hi)][q]
                # Exactly what this head added at q: its full attention row, not just the top keys.
                scores = (x_in @ h.w_k) @ (x_in[q] @ h.w_q)
                attn = np.exp(scores - scores.max())
                attn /= attn.sum()
                wrote = ((attn @ x_in) @ h.w_v) @ h.w_o
                nz = np.flatnonzero(np.abs(wrote) > 1e-9)
                heads.append({
                    "layer": li, "head": hi,
                    "top": [[int(k), float(wt)] for k, wt in zip(idx, w)],
                    "wrote": {self.names[d]: float(wrote[d]) for d in nz},
                })

        units = []
        for li, pre in rec.pre.items():
            active = np.flatnonzero(pre[q] > 1e-9)
            units += [{"layer": li, "unit": int(u), "pre": float(pre[q, u])} for u in active]

        top = np.argsort(-run.logits[q])[:TOP_LOGITS]
        log.info("position %s q=%d: %d heads, %d active units", run_id, q, len(heads), len(units))
        return {
            "model_id": self.model_id,
            "q": q,
            "values": values.round(9).tolist(),
            "heads": heads,
            "units": units,
            "logits": [[chr(int(t)), float(run.logits[q, t])] for t in top],
        }

    # ------------------------------------------------------------- heatmaps

    def attention_map(self, run_id: str, layer: int, head: int) -> tuple[np.ndarray, dict]:
        """Full causal attention pattern (n, n) of one head, recomputed by the model's own method."""
        run = self._get(run_id)
        if not 0 <= layer < len(self.model.layers) or not 0 <= head < len(self.model.layers[layer].heads):
            raise ValueError(f"no head {head} in layer {layer}")
        x_in = run.rec.snapshots[2 * layer][1]
        n = len(x_in)
        causal = np.triu(np.ones((n, n), dtype=bool), k=1)
        attn = self.model.head_attention(x_in, self.model.layers[layer].heads[head], causal)
        return attn, {"layer": layer, "head": head, "name": self.builder.layers[layer].heads[head].name}

    def residual_map(self, run_id: str, snapshot: int) -> tuple[np.ndarray, dict]:
        """Residual stream (d, n) at one snapshot, each dim scaled by its max |value| over positions.

        Positions, flags and big gated features differ by orders of magnitude, so a
        shared colour scale would hide everything but the largest dims.
        """
        run = self._get(run_id)
        if not 0 <= snapshot < len(run.rec.snapshots):
            raise ValueError(f"snapshot {snapshot} out of range")
        label, x = run.rec.snapshots[snapshot]
        mat = x.T
        scale = np.abs(mat).max(axis=1)
        safe = np.where(scale > 1e-9, scale, 1.0)
        return mat / safe[:, None], {"snapshot": snapshot, "label": label, "scale": scale.round(6).tolist()}

    def mlp_map(self, run_id: str, layer: int) -> tuple[np.ndarray, dict]:
        """Post-ReLU MLP activations (units, n) of one layer, each unit scaled by its max."""
        run = self._get(run_id)
        if layer not in run.rec.pre:
            raise ValueError(f"layer {layer} has no MLP units")
        act = np.maximum(run.rec.pre[layer], 0.0).T
        scale = act.max(axis=1)
        safe = np.where(scale > 1e-9, scale, 1.0)
        return act / safe[:, None], {"layer": layer, "scale": scale.round(6).tolist()}

    # ------------------------------------------------------------- raw weights

    def weights(self, kind: str, layer: int | None, head: int | None, trim: bool) -> tuple[np.ndarray, dict]:
        """One raw weight matrix with named rows/columns (exact float values).

        kinds: embed (bytes x dims), unembed (dims x bytes), wq / wk (dims x head
        cols), wov (dims x dims, fused value-output), w_in (dims x units),
        w_out (units x dims). ``trim`` drops rows/columns that are all zero.
        """
        m = self.model
        names = self.names
        byte_labels = [repr(chr(i)) for i in range(256)]
        if kind in ("embed", "unembed"):
            mat = m.embed if kind == "embed" else m.unembed
            rows, cols = (byte_labels, names) if kind == "embed" else (names, byte_labels)
            title = "token embedding (byte -> residual dims)" if kind == "embed" else "unembedding (residual dims -> byte logits)"
        else:
            if layer is None or not 0 <= layer < len(m.layers):
                raise ValueError(f"layer {layer} out of range")
            lyr, spec = m.layers[layer], self.builder.layers[layer]
            if kind in ("wq", "wk", "wov"):
                if head is None or not 0 <= head < len(lyr.heads):
                    raise ValueError(f"head {head} out of range in layer {layer}")
                h = lyr.heads[head]
                mat = {"wq": h.w_q, "wk": h.w_k, "wov": h.w_ov}[kind]
                cols = names if kind == "wov" else [f"col{c}" for c in range(mat.shape[1])]
                rows = names
                title = f"L{layer + 1} {spec.heads[head].name}: {kind}"
            elif kind in ("w_in", "w_out"):
                units = [f"#{u} -> {form_str(us.write)}" for u, us in enumerate(spec.units)]
                mat = lyr.w_in if kind == "w_in" else lyr.w_out
                rows, cols = (names, units) if kind == "w_in" else (units, names)
                title = f"L{layer + 1} MLP {kind}"
            else:
                raise ValueError(f"unknown weight kind {kind!r}")
        r_idx = np.arange(mat.shape[0])
        c_idx = np.arange(mat.shape[1])
        if trim:
            r_idx = np.flatnonzero(np.abs(mat).sum(axis=1) > 0)
            c_idx = np.flatnonzero(np.abs(mat).sum(axis=0) > 0)
        sub = mat[np.ix_(r_idx, c_idx)] if len(r_idx) and len(c_idx) else np.zeros((0, 0))
        log.info("weights %s layer=%s head=%s: %s -> %s (trim=%s), nonzero=%d",
                 kind, layer, head, mat.shape, sub.shape, trim, int(np.count_nonzero(mat)))
        return sub, {
            "title": title,
            "rows": [rows[i] for i in r_idx],
            "cols": [cols[i] for i in c_idx],
            "full_shape": list(mat.shape),
            "nonzero": int(np.count_nonzero(mat)),
            "max_abs": float(np.abs(mat).max()) if mat.size else 0.0,
        }
