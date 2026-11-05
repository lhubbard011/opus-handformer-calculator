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

from .. import ablate, common as cm
from ..model import Form, Tracer
from . import sources
from ..sampling import Sampling
from .models import MODELS, ModelDef
from ..rewrite import viz as loopviz
from ..loop import viz as depthviz
from .glossary import describe_dim, describe_head

log = logging.getLogger(__name__)

MAX_RUNS = 4
MAX_RUN_BYTES = 1 << 30  # stored traces of all kept runs together; oldest runs go first
# A knocked-out model may never write its final '\n'. It gets this many bytes, so a broken
# run costs about what a normal one does (same rule in viz/static/local-api.js).
def knockout_budget(normal_new: int, budget: int) -> int:
    return min(budget, 2 * normal_new + 32)


# Input with bytes the model has no features for (lenient models) often never finishes either;
# with no normal run to compare, it gets a fixed multiple of its prompt (same rule in local-api.js).
def unknown_budget(prompt_len: int, budget: int) -> int:
    return min(budget, 16 * prompt_len + 64)

MAX_KNOCKOUT_MODELS = 4
TOP_KEYS = 3
TOP_LOGITS = 5
TOP_DLA = 3  # bytes whose score is broken into direct contributions (same in local-api.js)



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


def _circuit(category, default: str, write_dims: list[str], read_dims: list[str]) -> str:
    """Which circuit a head/unit belongs to: the category of the dims it writes (output writers:
    the calculator namespace they read, else the model's main circuit ``default``)."""
    cats = [category(d) for d in write_dims]
    owner = next((c for c in cats if c not in ("output", "input")), None)
    if owner:
        return owner
    ns = next((d.split(".", 1)[0] for d in read_dims if "." in d), None)
    return ns or default


@dataclass
class _Recorder(Tracer):
    """Keeps residual snapshots, top attention keys per head, MLP pre-activations.

    A sublayer writes only a few dims, so each snapshot after the first is stored as
    the columns that changed (exact float64): about 20x less memory than a full copy
    per sublayer, which matters for long runs.
    """

    labels: list[str] = field(default_factory=list)
    base: np.ndarray | None = None  # the embedding snapshot (n, d)
    deltas: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)  # (cols, values (n, k))
    top_idx: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    top_w: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    pre: dict[int, np.ndarray] = field(default_factory=dict)
    _prev: np.ndarray | None = None
    _cache: dict[int, np.ndarray] = field(default_factory=dict)

    def residual(self, label: str, x: np.ndarray) -> None:
        self.labels.append(label)
        if self.base is None:
            self.base = x.copy()
        else:
            cols = np.flatnonzero((x != self._prev).any(axis=0))
            self.deltas.append((cols, x[:, cols].copy()))
        self._prev = x.copy()

    def snapshot(self, s: int) -> np.ndarray:
        """Residual (n, d) after snapshot s (0 = embedding), rebuilt exactly from the stored columns."""
        if s not in self._cache:
            x = self.base.copy()
            for cols, vals in self.deltas[:s]:
                x[:, cols] = vals
            if len(self._cache) >= 4:  # a few recent snapshots, for paging through heatmaps
                self._cache.pop(next(iter(self._cache)))
            self._cache[s] = x
        return self._cache[s]

    def row_history(self, q: int) -> np.ndarray:
        """Position q's residual after every snapshot, (d, n_snapshots), without rebuilding whole snapshots."""
        x = self.base[q].copy()
        out = [x.copy()]
        for cols, vals in self.deltas:
            x[cols] = vals[q]
            out.append(x.copy())
        return np.stack(out, axis=1)

    def sweep(self):
        """Yield (s, residual after snapshot s) in order, reusing one working copy (do not keep it)."""
        x = self.base.copy()
        yield 0, x
        for s, (cols, vals) in enumerate(self.deltas, start=1):
            x[:, cols] = vals
            yield s, x

    def finish(self) -> None:
        self._prev = None

    def nbytes(self) -> int:
        arrays = [self.base, *(v for _, v in self.deltas), *self.top_idx.values(), *self.top_w.values(),
                  *self.pre.values(), *self._cache.values()]
        return int(sum(a.nbytes for a in arrays if a is not None))

    def attention(self, layer: int, head: int, attn: np.ndarray) -> None:
        k = min(TOP_KEYS, attn.shape[1])
        idx = np.argsort(-attn, axis=1, kind="stable")[:, :k]  # ties: earlier key first
        self.top_idx[(layer, head)] = idx
        self.top_w[(layer, head)] = np.take_along_axis(attn, idx, axis=1)

    def mlp(self, layer: int, pre: np.ndarray) -> None:
        self.pre[layer] = pre.copy()


@dataclass
class _LoopRecord:
    """A looped run, kept so the page can switch between its passes without running it again."""

    res: object  # loop.engine.LoopResult
    tapes: list[list[int]]  # the bytes going into each pass
    held: list[np.ndarray]  # the carried dims going into each pass (n, len(CARRIED))
    acts: list[dict]  # the root's decision dims in each pass
    cap: int  # the pass budget it ran with


@dataclass
class Run:
    run_id: str
    expression: str
    tokens: list[int]
    prompt_len: int
    rec: _Recorder
    logits: np.ndarray
    model: object = None  # the (possibly knocked-out) Transformer that produced this run
    knockout: dict = field(default_factory=dict)


class Runner:
    def __init__(self, model: str = "calc") -> None:
        t0 = time.perf_counter()
        if model not in MODELS:
            raise ValueError(f"unknown model {model!r} (one of {', '.join(MODELS)})")
        self.spec: ModelDef = MODELS[model]
        # circuit of units that only write outputs
        self._main = "parser" if model == "calc" else "block" if model == "rewrite" else model
        self.builder = self.spec.builder()
        self.model = self.builder.build()
        self.layout = self.model.layout
        self.names = self.layout.names
        self.runs: OrderedDict[str, Run] = OrderedDict()
        self.ko_models: OrderedDict[str, object] = OrderedDict()  # knockout key -> model
        self.normal_len: dict[str, int] = {}  # expression -> bytes the intact model writes
        self.looped = hasattr(self.model, "loop")  # the looped rewriter (handformer/rewrite/viz.py)
        self.loops: OrderedDict[tuple[str, str], _LoopRecord] = OrderedDict()  # (expression, knockout) -> run
        self.depth = hasattr(self.model, "depth")  # the recurrent-depth calculator (handformer/loop/viz.py)
        self.depth_runs: OrderedDict[tuple[str, str], depthviz.Record] = OrderedDict()
        # Identifies this exact model, so a page loaded against an older server can tell.
        self.model_id = (f"{self.spec.key}-d{len(self.layout)}-L{len(self.model.layers)}-h{sum(len(l.heads) for l in self.model.layers)}"
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
        for n in self.names:
            if self.spec.category(n) == "input":
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

    def _doc(self, name: str) -> str:
        return self.spec.docs.get(name) or describe_dim(name)

    def info(self) -> dict:
        wires = self._wiring()
        sites = sources.dim_sites(self.builder)
        layers = []
        for li, spec in enumerate(self.builder.layers):
            heads = [{
                "name": h.name,
                "doc": describe_head(h),
                "circuit": _circuit(self.spec.category, self._main, [d for f in h.value.values() for d in f],
                                    [d for f in h.query for d in f]),
                "query": [form_str(f) for f in h.query],
                "key": [form_str(f) for f in h.key],
                "value": {src: form_str(dst) for src, dst in h.value.items()},
                "src": h.where,
                "reads": sorted({d for f in [*h.query, *h.key] for d in f} | set(h.value)),
                "writes": sorted({d for dst in h.value.values() for d in dst}),
            } for h in spec.heads]
            units = [{"read": form_str(u.read), "write": form_str(u.write), "label": u.label,
                      "doc": u.label or ("Writes " + "; ".join(f"{d}: {describe_dim(d) or 'working value'}"
                                                              for d in u.write)),
                      "circuit": _circuit(self.spec.category, self._main, list(u.write), list(u.read)), "src": u.where,
                      "reads": sorted(u.read), "writes": sorted(u.write)} for u in spec.units]
            layers.append({"index": li, "name": spec.name, "heads": heads, "units": units})
        return {
            "name": "handformer",
            "model": self.spec.key, "title": self.spec.title, "marks": list(self.spec.marks),
            "default_input": self.spec.default_input, "input_hint": self.spec.input_hint,
            "multiline": self.spec.multiline, "node_names": dict(self.spec.node_names), **self.spec.extra_info,
            "sampling": self.spec.sampling,
            "model_id": self.model_id,
            "d_model": len(self.layout),
            # the dims the unembedding reads: the only ones a logit depends on directly
            "readout": [self.names[j] for j in self._readout_rows()],
            "unembed": self._unembed_block(),
            "nonzero_params": self.model.nonzero_params(),
            "source": sources.repo(),  # GitHub repo + branch for "view source" links
            "dims": [{"name": n, "cat": self.spec.category(n), "doc": self._doc(n),
                      "written_by": self._summ(wires[n]["writers"]), "read_by": self._summ(wires[n]["readers"]),
                      "src": sites[n]}
                     for n in self.names],
            "layers": layers,
        }

    def _readout_rows(self) -> np.ndarray:
        return np.flatnonzero(np.abs(self.model.unembed).sum(axis=1))

    def _unembed_block(self) -> dict:
        """W_U cut to its non-zero rows (the readout dims, in the order of "readout") and columns
        (the bytes that can score at all), as [row, col, weight] triples: the whole matrix the
        logits come from, minus rows and columns that are entirely zero. Knock-outs never change
        it (ablate leaves the unembedding as is), so it belongs to the model, not a run."""
        U = self.model.unembed
        rows, cols = self._readout_rows(), np.flatnonzero(np.abs(U).sum(axis=0))
        block = U[np.ix_(rows, cols)]
        w = [[int(i), int(k), float(block[i, k])] for i, k in zip(*np.nonzero(block))]
        log.info("unembedding block: %d of %d dims x %d of %d bytes, %d weights",
                 len(rows), U.shape[0], len(cols), U.shape[1], len(w))
        return {"bytes": [chr(int(c)) for c in cols], "w": w, "d_model": int(U.shape[0]), "vocab": int(U.shape[1])}

    # ------------------------------------------------------------- runs

    def _model_for(self, ko: "ablate.Knockout"):
        if not ko:
            return self.model
        key = ko.key()
        if key not in self.ko_models:
            self.ko_models[key] = (loopviz.knock_out(self.model, ko) if self.looped else
                                   depthviz.knock_out(self.model, ko) if self.depth else ablate.apply(self.model, ko))
            while len(self.ko_models) > MAX_KNOCKOUT_MODELS:
                self.ko_models.popitem(last=False)
        self.ko_models.move_to_end(key)
        return self.ko_models[key]

    def run(self, expression: str, knockout: dict | None = None, max_new: int | None = None,
            sample: dict | None = None, loop_pass: int | None = None, iteration: list[int] | None = None) -> dict:
        """``max_new``: how many bytes the model may write ("run longer"), up to the model's limit;
        by default a knocked-out run or one with unknown bytes is cut off sooner (see the budgets).
        ``sample`` ({"temperature", "seed"}, language models only): seeded sampling instead of the
        argmax (handformer/sampling.py). ``loop_pass`` (the looped calculator only): which pass to
        show; None shows the answer (the coda's decode). ``iteration`` (the recurrent-depth
        calculator only): [position, iteration] to show that position at that iteration."""
        if self.looped:
            return self._run_loop(expression, knockout, max_new, loop_pass)
        if self.depth:
            return self._run_depth(expression, knockout, max_new, iteration)
        if loop_pass is not None:
            raise ValueError(f"the {self.spec.title} has no passes: it is not a looped model")
        if iteration is not None:
            raise ValueError(f"the {self.spec.title} has no iterations: its depth is fixed")
        sampling = None
        if sample is not None:
            if not self.spec.sampling:
                raise ValueError(f"the {self.spec.title} decodes greedily only (one right byte at each step)")
            sampling = Sampling(float(sample["temperature"]), int(sample["seed"]))
        prompt = self.spec.prompt(expression)  # checks the input and its framing, or ValueError
        pre, end = len(self.spec.prompt_prefix.encode()), len(self.spec.prompt_end.encode())
        expression = bytes(prompt[pre:len(prompt) - end]).decode("utf-8")
        ko = ablate.Knockout.from_request(knockout)
        model = self._model_for(ko)
        t0 = time.perf_counter()
        budget = max_budget = min(self.spec.max_tokens - len(prompt), self.spec.max_new or self.spec.max_tokens)
        if ko:
            normal = self._normal_len(expression, prompt, budget)
            budget = knockout_budget(normal, budget)
            log.info("knockout run capped at %d new bytes (normal answer: %d)", budget, normal)
        unknown = "".join(sorted({ch for ch in expression if ch not in self.spec.input_chars})) if self.spec.lenient else ""
        if unknown:
            budget = unknown_budget(len(prompt), budget)
            log.info("input has bytes the model has no features for (%r): run capped at %d new bytes", unknown, budget)
        if max_new is not None:  # asked for: the viewer's choice beats both caps
            budget = max(1, min(max_budget, int(max_new)))
            log.info("run longer: %d new bytes allowed", budget)
        tokens = model.generate(prompt, stop=cm.EOS, max_new=budget, sample=sampling)
        if sampling:
            log.info("sampled at temperature %g, seed %d", sampling.temperature, sampling.seed)
        t1 = time.perf_counter()
        rec = _Recorder()
        logits = model.residual(tokens, trace=rec) @ model.unembed
        rec.finish()
        t2 = time.perf_counter()
        run = Run(uuid.uuid4().hex[:12], expression, tokens, len(prompt), rec, logits, model, ko.as_request())
        self.runs[run.run_id] = run
        while len(self.runs) > MAX_RUNS or (len(self.runs) > 1 and self._stored_bytes() > MAX_RUN_BYTES):
            evicted, _ = self.runs.popitem(last=False)
            log.info("evicted run %s (kept runs now %d, %.0f MB)", evicted, len(self.runs), self._stored_bytes() / 1e6)
        finished = tokens[-1] == cm.EOS
        log.info("run %s %r: %d bytes generated in %.2fs, traced in %.2fs, finished=%s, knockout=%s",
                 run.run_id, expression, len(tokens) - len(prompt), t1 - t0, t2 - t1, finished, ko.key() if ko else "-")
        return {
            "model_id": self.model_id,
            "run_id": run.run_id,
            "expression": expression,
            "chars": [chr(t) for t in tokens],
            "prompt_len": run.prompt_len,
            "finished": finished,
            # bytes the model has no input features for (lenient models let them in): see what it does
            "unknown_chars": unknown,
            "budget": budget, "max_budget": max_budget,  # bytes the model could write this run / at most
            "sample": {"temperature": sampling.temperature, "seed": sampling.seed} if sampling else None,
            "knockout": run.knockout,
            "steps": [self._step(run, q) for q in range(len(tokens))],
            "snapshots": list(rec.labels),
        }

    def _loop_record(self, expression: str, prompt: list[int], ko, stack, max_new: int | None) -> "_LoopRecord":
        """The looped run itself (cached): every pass's tape, carried dims and root decisions."""
        key = (expression, ko.key() if ko else "")
        if key in self.loops and (max_new is None or self.loops[key].cap >= max_new):
            self.loops.move_to_end(key)
            return self.loops[key]
        cap = int(max_new) if max_new else loopviz.MAX_PASSES
        lay = stack.layout
        carried = [lay[d] for d in loopviz.circuit.CARRIED]
        tapes, held, acts = [], [], []

        def on_pass(r, state, x):
            tapes.append(stack.loop.tape_of(state))
            held.append(state[:, carried].copy())
            acts.append({k: round(float(x[-1, lay[k]]), 6) for k in loopviz.ACTIONS})

        t0 = time.perf_counter()
        res = stack.loop.run(prompt, stop=cm.EOS, max_passes=cap, on_pass=on_pass)
        rec = _LoopRecord(res, tapes, held, acts, cap)
        self.loops[key] = rec
        while len(self.loops) > MAX_RUNS * 2:
            self.loops.popitem(last=False)
        log.info("looped %r: %d passes in %.2fs, halted=%s, answer %r", expression, len(res.passes),
                 time.perf_counter() - t0, res.halted, bytes(res.answer).decode(errors="replace"))
        return rec

    def _run_loop(self, expression: str, knockout: dict | None, max_new: int | None, loop_pass: int | None) -> dict:
        """One view of a looped run, as an ordinary run: a pass (the tape through the block's
        layers; each position's next byte is the byte it holds next) or the answer (the final
        tape and the answer through the coda's layers). The page steps through either the same
        way; ``loop`` lists every pass so it can switch views."""
        prompt = self.spec.prompt(expression)
        expression = bytes(prompt[1:-1]).decode("utf-8")
        ko = ablate.Knockout.from_request(knockout)
        stack = self._model_for(ko)
        lr = self._loop_record(expression, prompt, ko, stack, max_new)
        res, n_pass = lr.res, len(lr.res.passes)
        kind = "pass" if loop_pass is not None and loop_pass < n_pass else "answer"
        lay = stack.layout
        if kind == "pass":
            tokens = lr.tapes[loop_pass]
            x0 = stack.embed[np.asarray(tokens, dtype=np.int64)].copy()
            x0[:, [lay[d] for d in loopviz.circuit.CARRIED]] = lr.held[loop_pass]
            prompt_len = len(tokens)
        else:
            tokens = [*res.final_tape, *res.answer]
            x0 = stack.embed[np.asarray(tokens, dtype=np.int64)]
            prompt_len = len(res.final_tape)
        view = stack.view("pass" if kind == "pass" else "answer")
        t0 = time.perf_counter()
        rec = _Recorder()
        logits = view.residual_from(x0, trace=rec) @ view.unembed
        rec.finish()
        run = Run(uuid.uuid4().hex[:12], expression, tokens, prompt_len, rec, logits, view, ko.as_request())
        self.runs[run.run_id] = run
        while len(self.runs) > MAX_RUNS or (len(self.runs) > 1 and self._stored_bytes() > MAX_RUN_BYTES):
            self.runs.popitem(last=False)
        answer = bytes(res.answer).decode(errors="replace")
        log.info("loop view %s %r: %s %s, %d positions traced in %.2fs", run.run_id, expression, kind,
                 loop_pass if kind == "pass" else "", len(tokens), time.perf_counter() - t0)
        return {
            "model_id": self.model_id, "run_id": run.run_id, "expression": expression,
            "chars": [chr(t) for t in tokens], "prompt_len": prompt_len,
            "finished": res.halted and answer.endswith("\n"), "unknown_chars": "",
            "budget": lr.cap, "max_budget": max(lr.cap, 10 * loopviz.MAX_PASSES),
            "sample": None, "knockout": run.knockout,
            "steps": [self._step(run, q) for q in range(len(tokens))],
            "snapshots": list(rec.labels),
            "loop": {"kind": kind, "pass": loop_pass if kind == "pass" else None, "halted": res.halted,
                     "answer": answer.rstrip("\n"),
                     "passes": [{"tape": bytes(t[1:-1]).decode(errors="replace"), "routed": p.routed, "grew": p.grew,
                                 "action": loopviz.action(a)} for t, p, a in zip(lr.tapes, res.passes, lr.acts)]},
        }

    def _depth_record(self, expression: str, prompt: list[int], ko, stack, max_new: int | None) -> depthviz.Record:
        """The recurrent-depth run itself (cached): every iteration's input, every position's kind."""
        key = (expression, ko.key() if ko else "")
        if key in self.depth_runs and (max_new is None or self.depth_runs[key].cap >= max_new):
            self.depth_runs.move_to_end(key)
            return self.depth_runs[key]
        cap = int(max_new) if max_new else depthviz.MAX_TOKENS
        t0 = time.perf_counter()
        rec = depthviz.record(stack, prompt, cap)
        self.depth_runs[key] = rec
        while len(self.depth_runs) > MAX_RUNS * 2:
            self.depth_runs.popitem(last=False)
        res = rec.res
        log.info("recurrent-depth %r: %d positions, %d iterations in %.2fs, finished=%s, answer %r", expression,
                 len(res.tokens), sum(t.iterations for t in res.tokens), time.perf_counter() - t0, res.finished,
                 bytes(res.answer).decode(errors="replace"))
        return rec

    def _run_depth(self, expression: str, knockout: dict | None, max_new: int | None,
                   iteration: list[int] | None) -> dict:
        """A recurrent-depth run as an ordinary run: every position at its last iteration through
        the block (exactly the cached computation), or the positions up to q with q at iteration
        k. ``depth`` lists every position's kind and depth so the page can switch views."""
        prompt = self.spec.prompt(expression)
        expression = bytes(prompt[1:-1]).decode("utf-8")
        ko = ablate.Knockout.from_request(knockout)
        stack = self._model_for(ko)
        dr = self._depth_record(expression, prompt, ko, stack, max_new)
        res, n = dr.res, len(dr.res.tokens)
        iters = [t.iterations for t in res.tokens]
        if iteration is not None:
            q, k = int(iteration[0]), int(iteration[1])
            if not 0 <= q < n:
                raise ValueError(f"position {q} out of range 0..{n - 1}")
            if not 0 <= k < iters[q]:
                raise ValueError(f"position {q} iterated {iters[q]} times: no iteration {k + 1}")
            rows = dr.last_in[: q + 1].copy()
            if k < iters[q] - 1:
                rows[q] = dr.earlier[(q, k)]
        else:
            rows = dr.last_in
        t0 = time.perf_counter()
        rec = _Recorder()
        logits = stack.residual_from(rows, trace=rec) @ stack.unembed
        rec.finish()
        chars = dr.chars[: len(rows)]
        run = Run(uuid.uuid4().hex[:12], expression, [ord(c) for c in chars], len(prompt), rec, logits, stack,
                  ko.as_request())
        self.runs[run.run_id] = run
        while len(self.runs) > MAX_RUNS or (len(self.runs) > 1 and self._stored_bytes() > MAX_RUN_BYTES):
            self.runs.popitem(last=False)
        answer = bytes(res.answer).decode(errors="replace")
        log.info("depth view %s %r: %s, %d positions traced in %.2fs", run.run_id, expression,
                 f"position {iteration[0]} iteration {iteration[1]}" if iteration else "last iterations",
                 len(rows), time.perf_counter() - t0)
        return {
            "model_id": self.model_id, "run_id": run.run_id, "expression": expression,
            "chars": chars, "prompt_len": len(prompt),
            "finished": res.finished and answer.endswith("\n"), "unknown_chars": "",
            "budget": dr.cap, "max_budget": max(dr.cap, 10 * depthviz.MAX_TOKENS),
            "sample": None, "knockout": run.knockout,
            "steps": [self._step(run, q) for q in range(len(rows))],
            "snapshots": list(rec.labels),
            "depth": {"kinds": dr.kinds, "iters": iters, "chars": dr.chars, "answer": answer.rstrip("\n"),
                      "finished": res.finished,
                      "view": [int(iteration[0]), int(iteration[1])] if iteration else None},
        }

    def _stored_bytes(self) -> int:
        return sum(r.rec.nbytes() + r.logits.nbytes for r in self.runs.values())

    def _normal_len(self, expression: str, prompt: list[int], budget: int) -> int:
        """Bytes the intact model writes for this expression (cached; generation only, no trace)."""
        if expression not in self.normal_len:
            out = self.model.generate(prompt, stop=cm.EOS, max_new=budget)
            self.normal_len[expression] = len(out) - len(prompt)
            while len(self.normal_len) > 256:
                self.normal_len.pop(next(iter(self.normal_len)))
        return self.normal_len[expression]

    def _get(self, run_id: str) -> Run:
        run = self.runs.get(run_id)
        if run is None:
            raise ValueError(f"unknown or evicted run {run_id}")
        return run

    def _val(self, run: Run, q: int, dim: str) -> float:
        return float(run.rec.snapshot(len(run.rec.labels) - 1)[q, self.layout[dim]])

    def _step(self, run: Run, q: int) -> dict:
        """Raw per-step values; the page turns them into phase, explanation and pointer tags."""
        x = run.rec.snapshot(len(run.rec.labels) - 1)[q]
        return {"pred": chr(int(np.argmax(run.logits[q]))),
                "v": {d: round(float(x[self.layout[d]]), 6) for d in self.spec.summary}}

    def position(self, run_id: str, q: int) -> dict:
        run = self._get(run_id)
        if not 0 <= q < len(run.tokens):
            raise ValueError(f"position {q} out of range 0..{len(run.tokens) - 1}")
        rec = run.rec
        values = rec.row_history(q)  # (d, n_snapshots)

        heads, head_out = [], {}
        # layer inputs: keys 0..q, or every key when a bidirectional head reads them (looped model)
        bidir = any(h.full for layer in run.model.layers for h in layer.heads)
        inputs = {s: x.copy() if bidir else x[: q + 1].copy() for s, x in rec.sweep() if s % 2 == 0}
        for li, layer in enumerate(run.model.layers):
            x_in = inputs[2 * li]  # residual entering layer li
            for hi, h in enumerate(layer.heads):
                idx = rec.top_idx[(li, hi)][q]
                w = rec.top_w[(li, hi)][q]
                # Exactly what this head added at q: its full attention row, not just the top keys.
                keys = x_in if h.full else x_in[: q + 1]
                scores = (keys @ h.w_k) @ (x_in[q] @ h.w_q)
                attn = np.exp(scores - scores.max())
                attn /= attn.sum()
                wrote = ((attn @ keys) @ h.w_v) @ h.w_o
                nz = np.flatnonzero(np.abs(wrote) > 1e-9)
                head_out[(li, hi)] = wrote
                heads.append({
                    "layer": li, "head": hi,
                    "top": [[int(k), float(wt)] for k, wt in zip(idx, w)],
                    "wrote": {self.names[d]: float(wrote[d]) for d in nz},
                })

        units = []
        for li, pre in rec.pre.items():
            active = np.flatnonzero(pre[q] > 1e-9)
            units += [{"layer": li, "unit": int(u), "pre": float(pre[q, u])} for u in active]

        top = np.argsort(-run.logits[q], kind="stable")[:TOP_LOGITS]  # ties by byte value, as local-api.js
        dla = self._dla(run, q, values, head_out, units, top[:TOP_DLA])
        log.info("position %s q=%d: %d heads, %d active units", run_id, q, len(heads), len(units))
        return {
            "model_id": self.model_id,
            "q": q,
            "values": values.round(9).tolist(),
            "heads": heads,
            "units": units,
            "logits": [[chr(int(t)), float(run.logits[q, t])] for t in top],
            "dla": dla,
        }

    def _dla(self, run: Run, q: int, values: np.ndarray, head_out: dict, units: list[dict], tops) -> list[dict]:
        """Direct logit attribution: each top byte's score as an exact sum of direct contributions.

        logit(b) = x_final · W_U[:, b], and x_final = (embedding + position) + every head's output
        + every fired unit's output (no LayerNorm, so the split is exact). by_dim: final value ×
        weight for each dim the readout uses; by_part: each component's output · W_U[:, b]."""
        U = run.model.unembed
        x_final, x_embed = values[:, -1], values[:, 0]
        out = []
        for t in tops:
            col = U[:, t]
            by_dim = [[self.names[j], float(x_final[j]), float(col[j]), float(x_final[j] * col[j])]
                      for j in np.flatnonzero(col) if abs(x_final[j] * col[j]) > 1e-9]
            parts = [["embed", -1, -1, float(x_embed @ col)]]
            parts += [["head", li, hi, float(v @ col)] for (li, hi), v in head_out.items()]
            parts += [["unit", u["layer"], u["unit"], float(u["pre"] * (run.model.layers[u["layer"]].w_out[u["unit"]] @ col))]
                      for u in units]
            kind = {"embed": 0, "head": 1, "unit": 2}
            parts = sorted((x for x in parts if abs(x[3]) > 1e-9), key=lambda x: (-round(abs(x[3]), 6), x[1], kind[x[0]], x[2]))
            by_dim.sort(key=lambda x: (-round(abs(x[3]), 6), x[0]))
            out.append({"byte": chr(int(t)), "logit": float(run.logits[q, t]), "by_dim": by_dim, "by_part": parts})
        return out

    # ------------------------------------------------------------- heatmaps

    def attention_map(self, run_id: str, layer: int, head: int) -> tuple[np.ndarray, dict]:
        """Full causal attention pattern (n, n) of one head, recomputed by the model's own method."""
        run = self._get(run_id)
        if not 0 <= layer < len(self.model.layers) or not 0 <= head < len(self.model.layers[layer].heads):
            raise ValueError(f"no head {head} in layer {layer}")
        if layer in getattr(run.model, "skip", ()):
            raise ValueError(f"layer {layer + 1} does not run in this view of the looped model "
                             "(the coda runs once, for the answer; answer bytes skip the block)")
        x_in = run.rec.snapshot(2 * layer)
        n = len(x_in)
        causal = np.triu(np.ones((n, n), dtype=bool), k=1)
        attn = run.model.head_attention(x_in, run.model.layers[layer].heads[head], causal)
        return attn, {"layer": layer, "head": head, "name": self.builder.layers[layer].heads[head].name}

    def residual_map(self, run_id: str, snapshot: int) -> tuple[np.ndarray, dict]:
        """Residual stream (d, n) at one snapshot, each dim scaled by its max |value| over positions.

        Positions, flags and big gated features differ by orders of magnitude, so a
        shared colour scale would hide everything but the largest dims.
        """
        run = self._get(run_id)
        if not 0 <= snapshot < len(run.rec.labels):
            raise ValueError(f"snapshot {snapshot} out of range")
        label, x = run.rec.labels[snapshot], run.rec.snapshot(snapshot)
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
