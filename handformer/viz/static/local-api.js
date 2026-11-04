// The visualizer's /api/* endpoints, answered in the page from engine.js.
//
// Mirrors handformer/viz/runner.py response for response, so app.js is the same
// page whether it talks to the Python server or runs fully static (GitHub Pages).
// tests/test_web.py checks this file against the server's answers.

import { Model } from "./engine.js";

const log = (...a) => console.info("[local-api]", ...a);
const MAX_RUNS = 4;

const round = (v, p) => { const s = 10 ** p; const r = Math.round(v * s) / s; return r === 0 ? 0 : r; };

export class LocalApi {
  constructor(modelJson, info) {
    this.model = new Model(modelJson);
    this.cfg = modelJson;
    this.info = info;
    this.runs = new Map();
    this.labels = this.model.snapshotLabels();
    this.layerStart = []; // global head index of each layer's first head
    let g = 0;
    for (const L of this.model.layers) { this.layerStart.push(g); g += L.heads.length; }
    this.index = new Map(this.model.names.map((n, i) => [n, i]));
  }

  static async load(base) {
    const t0 = performance.now();
    const [m, i] = await Promise.all([fetch(`${base}model.json`), fetch(`${base}info.json`)]);
    if (!m.ok || !i.ok) throw new Error(`could not load model files from ${base} (${m.status}/${i.status})`);
    const api = new LocalApi(await m.json(), await i.json());
    log(`loaded model ${api.cfg.model_id} from ${base} in ${(performance.now() - t0).toFixed(0)}ms`);
    return api;
  }

  /** Same contract as app.js api(): JSON object, or {meta, data} when binary. */
  async call(path, body) {
    const t0 = performance.now();
    const route = {
      "/api/info": () => this.info,
      "/api/run": () => this.run(body.expression),
      "/api/position": () => this.position(body.run_id, body.q),
      "/api/attention": () => this.attentionMap(body.run_id, body.layer, body.head),
      "/api/residual": () => this.residualMap(body.run_id, body.snapshot),
      "/api/mlp": () => this.mlpMap(body.run_id, body.layer),
      "/api/weights": () => this.weights(body.kind, body.layer ?? null, body.head ?? null, body.trim ?? true),
    }[path];
    if (!route) throw new Error(`${path}: no such endpoint`);
    try {
      const out = route();
      log(`${path} ok in ${(performance.now() - t0).toFixed(0)}ms`);
      return out;
    } catch (e) {
      console.error("[local-api]", path, "failed:", e.message);
      throw new Error(`${path}: ${e.message}`);
    }
  }

  // ------------------------------------------------------------- runs

  run(expression) {
    expression = String(expression).replaceAll(" ", "");
    if (expression.endsWith("=")) expression = expression.slice(0, -1);
    if (!expression) throw new Error("empty expression");
    if (expression.length > 400) throw new Error("expression longer than 400 characters");
    const allowed = new Set(this.cfg.input_chars);
    const bad = [...new Set([...expression].filter((ch) => !allowed.has(ch)))].sort();
    if (bad.length) throw new Error(`unsupported characters: '${bad.join("")}' (use digits . + - * / ( ) and ~N)`);
    const prompt = [...`\n${expression}=`].map((c) => c.charCodeAt(0));
    const tr = this.model.generate(prompt, this.cfg.max_tokens - prompt.length, this.cfg.top_keys);
    const runId = Math.random().toString(16).slice(2, 14).padEnd(12, "0");
    this.runs.set(runId, { expression, tr, promptLen: prompt.length });
    while (this.runs.size > MAX_RUNS) {
      const old = this.runs.keys().next().value;
      this.runs.delete(old);
      log("evicted run", old);
    }
    const tokens = tr.tokens;
    const finished = tokens[tokens.length - 1] === this.model.eos;
    log(`run ${runId} ${JSON.stringify(expression)}: ${tokens.length - prompt.length} bytes, finished=${finished}`);
    return {
      model_id: this.cfg.model_id,
      run_id: runId,
      expression,
      chars: tokens.map((t) => String.fromCharCode(t)),
      prompt_len: prompt.length,
      finished,
      steps: tokens.map((_, q) => this._step(tr, q)),
      snapshots: this.labels,
    };
  }

  _get(runId) {
    const r = this.runs.get(runId);
    if (!r) throw new Error(`unknown or evicted run ${runId}`);
    return r;
  }

  _step(tr, q) {
    const x = tr.final[q], lg = tr.logits[q];
    let best = 0;
    for (let b = 1; b < lg.length; b++) if (lg[b] > lg[best]) best = b;
    const v = {};
    for (const d of this.cfg.summary_dims) v[d] = round(x[this.index.get(d)], 6);
    return { pred: String.fromCharCode(best), v };
  }

  position(runId, q) {
    const { tr } = this._get(runId);
    if (!(q >= 0 && q < tr.tokens.length)) throw new Error(`position ${q} out of range 0..${tr.tokens.length - 1}`);
    const d = this.model.d, S = this.labels.length;
    const values = Array.from({ length: d }, () => new Array(S));
    for (let s = 0; s < S; s++) {
      const x = tr.residualAt(q, s);
      for (let j = 0; j < d; j++) values[j][s] = round(x[j], 9);
    }
    const heads = [];
    this.model.layers.forEach((L, li) => L.heads.forEach((_, hi) => {
      const gh = this.layerStart[li] + hi;
      const wrote = tr.headWrote(li, hi, q);
      const w = {};
      for (let j = 0; j < d; j++) if (Math.abs(wrote[j]) > 1e-9) w[this.model.names[j]] = wrote[j];
      heads.push({ layer: li, head: hi, top: tr.top[q][gh].map(([k, wt]) => [k, wt]), wrote: w });
    }));
    const units = [];
    tr.pre[q].forEach((p, li) => {
      if (!p) return;
      for (let i = 0; i < p.units.length; i++) if (p.pre[i] > 1e-9) units.push({ layer: li, unit: p.units[i], pre: p.pre[i] });
    });
    const lg = tr.logits[q];
    const order = [...lg.keys()].sort((a, b) => lg[b] - lg[a] || a - b).slice(0, this.cfg.top_logits);
    log(`position ${runId} q=${q}: ${heads.length} heads, ${units.length} active units`);
    return {
      model_id: this.cfg.model_id, q, values, heads, units,
      logits: order.map((t) => [String.fromCharCode(t), lg[t]]),
    };
  }

  // ------------------------------------------------------------- heatmaps

  attentionMap(runId, layer, head) {
    const { tr } = this._get(runId);
    const L = this.model.layers[layer];
    if (!L || !(head >= 0 && head < L.heads.length)) throw new Error(`no head ${head} in layer ${layer}`);
    const n = tr.tokens.length, data = new Float32Array(n * n);
    for (let q = 0; q < n; q++) data.set(tr.attentionRow(layer, head, q), q * n);
    const name = this.info.layers[layer].heads[head].name;
    return { meta: { layer, head, name, shape: [n, n], dtype: "float32" }, data };
  }

  residualMap(runId, snapshot) {
    const { tr } = this._get(runId);
    if (!(snapshot >= 0 && snapshot < this.labels.length)) throw new Error(`snapshot ${snapshot} out of range`);
    const n = tr.tokens.length, d = this.model.d;
    const mat = new Float64Array(d * n);
    for (let q = 0; q < n; q++) {
      const x = tr.residualAt(q, snapshot);
      for (let j = 0; j < d; j++) mat[j * n + q] = x[j];
    }
    return scaled(mat, d, n, true, { snapshot, label: this.labels[snapshot] });
  }

  mlpMap(runId, layer) {
    const { tr } = this._get(runId);
    const L = this.model.layers[layer];
    if (!L || !L.units) throw new Error(`layer ${layer} has no MLP units`);
    const n = tr.tokens.length, u = L.units;
    const mat = new Float64Array(u * n);
    for (let q = 0; q < n; q++) {
      const p = tr.pre[q][layer];
      for (let i = 0; i < p.units.length; i++) mat[p.units[i] * n + q] = p.pre[i];
    }
    return scaled(mat, u, n, false, { layer });
  }

  // ------------------------------------------------------------- raw weights

  weights(kind, layer, head, trim) {
    const m = this.model, names = m.names, J = this.cfg;
    let triples, shape, rows, cols, title;
    if (kind === "embed" || kind === "unembed") {
      triples = kind === "embed" ? J.embed : J.unembed;
      shape = kind === "embed" ? [256, m.d] : [m.d, 256];
      [rows, cols] = kind === "embed" ? [J.byte_labels, names] : [names, J.byte_labels];
      title = kind === "embed" ? "token embedding (byte -> residual dims)" : "unembedding (residual dims -> byte logits)";
    } else {
      if (layer === null || !(layer >= 0 && layer < m.layers.length)) throw new Error(`layer ${layer} out of range`);
      const LJ = J.layers[layer], spec = this.info.layers[layer];
      if (kind === "wq" || kind === "wk" || kind === "wov") {
        if (head === null || !(head >= 0 && head < LJ.heads.length)) throw new Error(`head ${head} out of range in layer ${layer}`);
        const h = LJ.heads[head];
        if (kind === "wov") {
          triples = fuse(h.v, h.o);
          shape = [m.d, m.d];
          cols = names;
        } else {
          triples = kind === "wq" ? h.q : h.k;
          shape = [m.d, h.dk];
          cols = Array.from({ length: h.dk }, (_, c) => `col${c}`);
        }
        rows = names;
        title = `L${layer + 1} ${spec.heads[head].name}: ${kind}`;
      } else if (kind === "w_in" || kind === "w_out") {
        const units = spec.units.map((u, i) => `#${i} -> ${u.write}`);
        triples = kind === "w_in" ? LJ.win : LJ.wout;
        shape = kind === "w_in" ? [m.d, LJ.units] : [LJ.units, m.d];
        [rows, cols] = kind === "w_in" ? [names, units] : [units, names];
        title = `L${layer + 1} MLP ${kind}`;
      } else {
        throw new Error(`unknown weight kind '${kind}'`);
      }
    }
    let rIdx = [...Array(shape[0]).keys()], cIdx = [...Array(shape[1]).keys()];
    if (trim) {
      const rs = new Set(), cs = new Set();
      for (const [r, c, v] of triples) if (v !== 0) { rs.add(r); cs.add(c); }
      rIdx = [...rs].sort((a, b) => a - b);
      cIdx = [...cs].sort((a, b) => a - b);
    }
    const rPos = new Map(rIdx.map((r, i) => [r, i])), cPos = new Map(cIdx.map((c, i) => [c, i]));
    const nr = rIdx.length && cIdx.length ? rIdx.length : 0, nc = nr ? cIdx.length : 0;
    const data = new Float32Array(nr * nc);
    let maxAbs = 0, nonzero = 0;
    for (const [r, c, v] of triples) {
      if (v === 0) continue;
      nonzero++;
      maxAbs = Math.max(maxAbs, Math.abs(v));
      if (nr && rPos.has(r) && cPos.has(c)) data[rPos.get(r) * nc + cPos.get(c)] = v;
    }
    log(`weights ${kind} layer=${layer} head=${head}: [${shape}] -> [${nr},${nc}] (trim=${trim}), nonzero=${nonzero}`);
    return {
      meta: {
        title, rows: nr ? rIdx.map((i) => rows[i]) : [], cols: nr ? cIdx.map((i) => cols[i]) : [],
        full_shape: shape, nonzero, max_abs: maxAbs, shape: [nr, nc], dtype: "float32",
      },
      data,
    };
  }
}

/** Divide each row by its max |value| (relu: values already ≥ 0), as the server does. */
function scaled(mat, rows, n, abs, meta) {
  const scale = new Array(rows), data = new Float32Array(rows * n);
  for (let r = 0; r < rows; r++) {
    let s = 0;
    for (let q = 0; q < n; q++) s = Math.max(s, abs ? Math.abs(mat[r * n + q]) : mat[r * n + q]);
    scale[r] = round(s, 6);
    const div = s > 1e-9 ? s : 1;
    for (let q = 0; q < n; q++) data[r * n + q] = mat[r * n + q] / div;
  }
  return { meta: { ...meta, scale, shape: [rows, n], dtype: "float32" }, data };
}

/** Sparse product W_V·W_O as [row, col, value] triples (the fused value-output map). */
function fuse(v, o) {
  const oRows = new Map();
  for (const [r, c, w] of o) {
    if (!oRows.has(r)) oRows.set(r, []);
    oRows.get(r).push([c, w]);
  }
  const acc = new Map();
  for (const [r, mid, w] of v) {
    for (const [c, w2] of oRows.get(mid) || []) {
      const key = r * 1e6 + c;
      acc.set(key, (acc.get(key) || 0) + w * w2);
    }
  }
  return [...acc.entries()].map(([key, val]) => [Math.floor(key / 1e6), key % 1e6, val]);
}
