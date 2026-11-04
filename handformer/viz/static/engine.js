// In-browser forward pass of the hand-written transformer (float64, sparse weights).
//
// Same computation as handformer/model.py `_DecodeCache.step`: byte embedding +
// pos/pos2 features; per layer every head reads the same input, causal softmax
// of unscaled q·k, outputs summed into the stream; then the ReLU MLP. No
// LayerNorm, no biases. Weights come from tools/export_web.py (model.json), a
// list of every nonzero entry, so nothing here knows anything about arithmetic.
//
// Values are cached per head (v = x·W_V) as well as keys, so attention costs
// O(n·d_head) per token; (attn·X)·W_V == attn·(X·W_V) up to float rounding.
//
// While decoding, a compact trace is kept: for every position and sublayer the
// residual entries that changed (exact new values), the top attention keys,
// queries, and positive MLP pre-activations. That is enough to answer every
// visualizer request without a second forward pass.

const log = (...a) => console.info("[engine]", ...a);

/** Sparse [row, col, value] triples grouped by row: rows[r] = {cols, vals}. */
function byRow(triples) {
  const rows = new Map();
  for (const [r, c, v] of triples) {
    if (!rows.has(r)) rows.set(r, { cols: [], vals: [] });
    rows.get(r).cols.push(c);
    rows.get(r).vals.push(v);
  }
  return [...rows.entries()].map(([r, e]) => ({ r, cols: Int32Array.from(e.cols), vals: Float64Array.from(e.vals) }));
}

/** out (zeroed, length = n cols) = x · W for sparse W given by rows. */
function matvec(x, rows, out) {
  out.fill(0);
  for (const { r, cols, vals } of rows) {
    const xr = x[r];
    if (xr === 0) continue;
    for (let i = 0; i < cols.length; i++) out[cols[i]] += xr * vals[i];
  }
  return out;
}

export class Model {
  constructor(json) {
    this.json = json;
    this.d = json.d;
    this.names = json.names;
    this.pos = json.pos;
    this.pos2 = json.pos2;
    this.eos = json.eos;
    this.embed = byRow(json.embed); // rows = bytes
    this.embedRow = new Map(this.embed.map((e) => [e.r, e]));
    this.unembed = byRow(json.unembed); // rows = dims
    this.layers = json.layers.map((L) => ({
      name: L.name,
      units: L.units,
      win: byRow(L.win),
      wout: byRow(L.wout),
      heads: L.heads.map((h) => ({ dk: h.dk, dv: h.dv, q: byRow(h.q), k: byRow(h.k), v: byRow(h.v), o: byRow(h.o) })),
    }));
    this.nHeads = this.layers.reduce((a, L) => a + L.heads.length, 0);
    log(`model ${json.model_id}: d=${this.d}, ${this.layers.length} layers, ${this.nHeads} heads`);
  }

  /** Snapshot labels, same order as the Python tracer: embed, then attn/mlp per layer. */
  snapshotLabels() {
    return ["embed", ...this.layers.flatMap((L) => [`${L.name} attn`, `${L.name} mlp`])];
  }

  /** Greedy decode from prompt bytes until EOS or maxNew bytes; returns a Trace. */
  generate(prompt, maxNew, topKeys) {
    const t0 = performance.now();
    const tr = new Trace(this, prompt.length + maxNew, topKeys);
    let logits = null;
    for (const t of prompt) logits = tr.step(t);
    const out = [...prompt];
    let finished = false;
    for (let s = 0; s < maxNew; s++) {
      let best = 0;
      for (let b = 1; b < logits.length; b++) if (logits[b] > logits[best]) best = b;
      out.push(best);
      if (best === this.eos) { finished = true; break; }
      logits = tr.step(best);
    }
    // The model's prediction after the last byte is still recorded (the step strip shows it).
    if (finished) tr.step(this.eos);
    tr.tokens = out;
    log(`generated ${out.length - prompt.length} bytes in ${(performance.now() - t0).toFixed(0)}ms, finished=${finished}`);
    if (!finished) console.warn("[engine] hit max_new without EOS");
    return tr;
  }
}

/** Decode cache + compact trace of every position. */
export class Trace {
  constructor(model, capacity, topKeys) {
    this.m = model;
    this.topKeys = topKeys;
    this.cap = Math.max(16, capacity);
    this.n = 0;
    const L = model.layers;
    this.keys = L.map((Ly) => Ly.heads.map((h) => new Float64Array(this.cap * h.dk)));
    this.vals = L.map((Ly) => Ly.heads.map((h) => new Float64Array(this.cap * h.dv)));
    this.qs = L.map((Ly) => Ly.heads.map((h) => new Float64Array(this.cap * h.dk)));
    this.x0 = []; // per position: [[dim, value]] nonzero entries after embedding
    this.changes = []; // per position: per sublayer (2 per layer): {dims: Int32Array, vals: Float64Array}
    this.top = []; // per position: per global head index: [[key, weight], ...]
    this.pre = []; // per position: per layer: {units: Int32Array, pre: Float64Array} with pre > 0
    this.logits = []; // per position: Float64Array(256)
    this.final = []; // per position: Float64Array(d) final residual
    this.tokens = [];
    this._q = new Float64Array(64);
    this._k = new Float64Array(64);
    this._v = new Float64Array(64);
    this._o = null;
  }

  _grow() {
    const nc = this.cap * 2;
    const g = (arr, w) => { const a = new Float64Array(nc * w); a.set(arr); return a; };
    this.keys = this.keys.map((hs, li) => hs.map((k, hi) => g(k, this.m.layers[li].heads[hi].dk)));
    this.vals = this.vals.map((hs, li) => hs.map((v, hi) => g(v, this.m.layers[li].heads[hi].dv)));
    this.qs = this.qs.map((hs, li) => hs.map((q, hi) => g(q, this.m.layers[li].heads[hi].dk)));
    this.cap = nc;
    log(`decode cache grown to ${nc} positions`);
  }

  /** Append one token; return its next-byte logits (Float64Array(256)). */
  step(token) {
    if (this.n === this.cap) this._grow();
    const m = this.m, d = m.d, q = this.n;
    const x = new Float64Array(d);
    const e = m.embedRow.get(token);
    if (e) for (let i = 0; i < e.cols.length; i++) x[e.cols[i]] = e.vals[i];
    x[m.pos] += q;
    x[m.pos2] += q * q;
    const x0 = [];
    for (let j = 0; j < d; j++) if (x[j] !== 0) x0.push([j, x[j]]);
    this.x0.push(x0);
    const changes = [], tops = [], pres = [];
    const delta = new Float64Array(d);
    let attn = new Float64Array(q + 1);
    let gh = 0;
    for (let li = 0; li < m.layers.length; li++) {
      const Ly = m.layers[li];
      delta.fill(0);
      for (let hi = 0; hi < Ly.heads.length; hi++, gh++) {
        const h = Ly.heads[hi], dk = h.dk, dv = h.dv;
        const K = this.keys[li][hi], V = this.vals[li][hi], Q = this.qs[li][hi];
        const kq = matvec(x, h.k, this._k.subarray(0, dk));
        const qq = matvec(x, h.q, this._q.subarray(0, dk));
        const vq = matvec(x, h.v, this._v.subarray(0, dv));
        K.set(kq, q * dk);
        Q.set(qq, q * dk);
        V.set(vq, q * dv);
        let mx = -Infinity;
        for (let t = 0; t <= q; t++) {
          let s = 0;
          for (let c = 0; c < dk; c++) s += K[t * dk + c] * qq[c];
          attn[t] = s;
          if (s > mx) mx = s;
        }
        let z = 0;
        for (let t = 0; t <= q; t++) { attn[t] = Math.exp(attn[t] - mx); z += attn[t]; }
        const mixed = new Float64Array(dv);
        for (let t = 0; t <= q; t++) {
          const a = (attn[t] /= z);
          if (a === 0) continue;
          for (let c = 0; c < dv; c++) mixed[c] += a * V[t * dv + c];
        }
        for (const { r, cols, vals } of h.o) {
          const mr = mixed[r];
          if (mr === 0) continue;
          for (let i = 0; i < cols.length; i++) delta[cols[i]] += mr * vals[i];
        }
        tops.push(topK(attn, q + 1, this.topKeys));
      }
      changes.push(applyDelta(x, delta));
      if (Ly.units) {
        const pre = matvec(x, Ly.win, new Float64Array(Ly.units));
        const act = [], actPre = [];
        delta.fill(0);
        for (let u = 0; u < pre.length; u++) {
          if (pre[u] > 0) { act.push(u); actPre.push(pre[u]); }
        }
        const wout = Ly.woutRow || (Ly.woutRow = new Map(Ly.wout.map((w) => [w.r, w])));
        for (let i = 0; i < act.length; i++) {
          const w = wout.get(act[i]);
          if (!w) continue;
          for (let j = 0; j < w.cols.length; j++) delta[w.cols[j]] += actPre[i] * w.vals[j];
        }
        pres.push({ units: Int32Array.from(act), pre: Float64Array.from(actPre) });
        changes.push(applyDelta(x, delta));
      } else {
        pres.push(null);
        changes.push({ dims: new Int32Array(0), vals: new Float64Array(0) });
      }
    }
    this.changes.push(changes);
    this.top.push(tops);
    this.pre.push(pres);
    this.final.push(x);
    const logits = matvec(x, m.unembed, new Float64Array(256));
    this.logits.push(logits);
    this.n++;
    return logits;
  }

  /** Residual (Float64Array(d)) of position q after snapshot s (0 = embed). */
  residualAt(q, s) {
    const x = new Float64Array(this.m.d);
    for (const [j, v] of this.x0[q]) x[j] = v;
    const ch = this.changes[q];
    for (let k = 0; k < s; k++) {
      const { dims, vals } = ch[k];
      for (let i = 0; i < dims.length; i++) x[dims[i]] = vals[i];
    }
    return x;
  }

  /** Full causal attention row of one head at query position q (from cached keys and queries). */
  attentionRow(li, hi, q) {
    const h = this.m.layers[li].heads[hi], dk = h.dk;
    const K = this.keys[li][hi], Q = this.qs[li][hi];
    const row = new Float64Array(q + 1);
    let mx = -Infinity;
    for (let t = 0; t <= q; t++) {
      let s = 0;
      for (let c = 0; c < dk; c++) s += K[t * dk + c] * Q[q * dk + c];
      row[t] = s;
      if (s > mx) mx = s;
    }
    let z = 0;
    for (let t = 0; t <= q; t++) { row[t] = Math.exp(row[t] - mx); z += row[t]; }
    for (let t = 0; t <= q; t++) row[t] /= z;
    return row;
  }

  /** Exactly what head (li, hi) added at q: (Σ_t attn_t·v_t)·W_O over its full attention row. */
  headWrote(li, hi, q) {
    const h = this.m.layers[li].heads[hi], dv = h.dv, V = this.vals[li][hi];
    const row = this.attentionRow(li, hi, q);
    const mixed = new Float64Array(dv);
    for (let t = 0; t <= q; t++) if (row[t] !== 0) for (let c = 0; c < dv; c++) mixed[c] += row[t] * V[t * dv + c];
    const out = new Float64Array(this.m.d);
    for (const { r, cols, vals } of h.o) for (let i = 0; i < cols.length; i++) out[cols[i]] += mixed[r] * vals[i];
    return out;
  }
}

function applyDelta(x, delta) {
  const dims = [], vals = [];
  for (let j = 0; j < delta.length; j++) {
    if (delta[j] !== 0) {
      x[j] += delta[j];
      dims.push(j);
      vals.push(x[j]);
    }
  }
  return { dims: Int32Array.from(dims), vals: Float64Array.from(vals) };
}

/** Indices of the k largest weights (ties: lower index first), as [[index, weight]]. */
function topK(a, n, k) {
  const best = [];
  for (let t = 0; t < n; t++) {
    const w = a[t];
    if (best.length < k) {
      best.push([t, w]);
      best.sort((p, r) => r[1] - p[1] || p[0] - r[0]);
    } else if (w > best[k - 1][1]) {
      best[k - 1] = [t, w];
      best.sort((p, r) => r[1] - p[1] || p[0] - r[0]);
    }
  }
  return best;
}
