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
      // full: a bidirectional head (the looped calculator's block), which reads every position
      heads: L.heads.map((h) => ({ dk: h.dk, dv: h.dv, q: byRow(h.q), k: byRow(h.k), v: byRow(h.v), o: byRow(h.o), full: !!h.full })),
    }));
    this.nHeads = this.layers.reduce((a, L) => a + L.heads.length, 0);
    log(`model ${json.model_id}: d=${this.d}, ${this.layers.length} layers, ${this.nHeads} heads`);
  }

  /** Snapshot labels, same order as the Python tracer: embed, then attn/mlp per layer. */
  snapshotLabels() {
    return ["embed", ...this.layers.flatMap((L) => [`${L.name} attn`, `${L.name} mlp`])];
  }

  /** Greedy decode from prompt bytes until EOS or maxNew bytes; returns a Trace. */
  generate(prompt, maxNew, topKeys, sample = null) {
    const t0 = performance.now();
    const tr = new Trace(this, prompt.length + maxNew, topKeys);
    let logits = null;
    for (const t of prompt) logits = tr.step(t);
    const out = [...prompt];
    let finished = false;
    const rng = sample ? mulberry32(sample.seed) : null; // the language models: a seeded draw (handformer/sampling.py)
    for (let s = 0; s < maxNew; s++) {
      let best = 0;
      if (rng) best = draw(logits, sample.temperature, rng);
      else for (let b = 1; b < logits.length; b++) if (logits[b] > logits[best]) best = b;
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

/** mulberry32: the same 32-bit generator as handformer/sampling.py (same seed, same numbers). */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) | 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** The first byte whose running total of exp(z - max z), z = logits / T, passes u * total (sampling.py draw). */
export function draw(logits, temperature, rng) {
  let mx = -Infinity;
  for (let b = 0; b < logits.length; b++) mx = Math.max(mx, logits[b] / temperature);
  const cum = new Float64Array(logits.length);
  let acc = 0;
  for (let b = 0; b < logits.length; b++) { acc += Math.exp(logits[b] / temperature - mx); cum[b] = acc; }
  const u = rng() * acc;
  for (let b = 0; b < logits.length; b++) if (cum[b] > u) return b;
  return logits.length - 1;
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
    const x = new Float64Array(this.m.d);
    const e = this.m.embedRow.get(token);
    if (e) for (let i = 0; i < e.cols.length; i++) x[e.cols[i]] = e.vals[i];
    return this.stepVec(x);
  }

  /** Append one position whose input is the residual x (positions are added here): the looped
   *  calculator's coda starts from the tape's final state (handformer/model.py _DecodeCache.step_vec). */
  stepVec(input) {
    if (this.n === this.cap) this._grow();
    const m = this.m, d = m.d, q = this.n;
    const x = Float64Array.from(input);
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

  /** Full attention row of one head at query position q (from cached keys and queries): keys
   *  0..q, or every key for a bidirectional head. */
  attentionRow(li, hi, q) {
    const h = this.m.layers[li].heads[hi], dk = h.dk;
    const K = this.keys[li][hi], Q = this.qs[li][hi];
    const len = h.full ? this.n : q + 1;
    const row = new Float64Array(len);
    let mx = -Infinity;
    for (let t = 0; t < len; t++) {
      let s = 0;
      for (let c = 0; c < dk; c++) s += K[t * dk + c] * Q[q * dk + c];
      row[t] = s;
      if (s > mx) mx = s;
    }
    let z = 0;
    for (let t = 0; t < len; t++) { row[t] = Math.exp(row[t] - mx); z += row[t]; }
    for (let t = 0; t < len; t++) row[t] /= z;
    return row;
  }

  /** Exactly what head (li, hi) added at q: (Σ_t attn_t·v_t)·W_O over its full attention row. */
  headWrote(li, hi, q) {
    const h = this.m.layers[li].heads[hi], dv = h.dv, V = this.vals[li][hi];
    const row = this.attentionRow(li, hi, q);
    const mixed = new Float64Array(dv);
    for (let t = 0; t < row.length; t++) if (row[t] !== 0) for (let c = 0; c < dv; c++) mixed[c] += row[t] * V[t * dv + c];
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

// ------------------------------------------------------------------ the looped calculator
// handformer/rewrite/engine.py: the block runs over every position at once (some heads are
// bidirectional), the adapter re-embeds each position's chosen byte and keeps the carried dims,
// the router stops the loop, and the coda decodes the answer causally.

/** Every position through layers [0, upto) at once; x0 rows are inputs without positions. With
 *  ``trace`` (a Trace), the same record a decode keeps is filled, for the visualizer. */
export function forwardRows(model, x0Rows, upto = model.layers.length, trace = null) {
  const n = x0Rows.length, d = model.d;
  const X = x0Rows.map((r, q) => { const x = Float64Array.from(r); x[model.pos] += q; x[model.pos2] += q * q; return x; });
  if (trace) {
    trace.x0 = X.map((x) => { const nz = []; for (let j = 0; j < d; j++) if (x[j] !== 0) nz.push([j, x[j]]); return nz; });
    trace.changes = X.map(() => []); trace.top = X.map(() => []); trace.pre = X.map(() => []);
  }
  const kbuf = new Float64Array(64), qbuf = new Float64Array(64), vbuf = new Float64Array(64);
  for (let li = 0; li < upto; li++) {
    const Ly = model.layers[li];
    const deltas = X.map(() => new Float64Array(d));
    for (let hi = 0; hi < Ly.heads.length; hi++) {
      const h = Ly.heads[hi], dk = h.dk, dv = h.dv;
      const K = trace ? trace.keys[li][hi] : new Float64Array(n * dk);
      const Q = trace ? trace.qs[li][hi] : new Float64Array(n * dk);
      const V = trace ? trace.vals[li][hi] : new Float64Array(n * dv);
      for (let q = 0; q < n; q++) {
        K.set(matvec(X[q], h.k, kbuf.subarray(0, dk)), q * dk);
        Q.set(matvec(X[q], h.q, qbuf.subarray(0, dk)), q * dk);
        V.set(matvec(X[q], h.v, vbuf.subarray(0, dv)), q * dv);
      }
      const attn = new Float64Array(n);
      for (let q = 0; q < n; q++) {
        const len = h.full ? n : q + 1;
        let mx = -Infinity;
        for (let t = 0; t < len; t++) {
          let s = 0;
          for (let c = 0; c < dk; c++) s += K[t * dk + c] * Q[q * dk + c];
          attn[t] = s;
          if (s > mx) mx = s;
        }
        let z = 0;
        for (let t = 0; t < len; t++) { attn[t] = Math.exp(attn[t] - mx); z += attn[t]; }
        const mixed = new Float64Array(dv);
        for (let t = 0; t < len; t++) {
          const a = (attn[t] /= z);
          if (a === 0) continue;
          for (let c = 0; c < dv; c++) mixed[c] += a * V[t * dv + c];
        }
        const delta = deltas[q];
        for (const { r, cols, vals } of h.o) {
          const mr = mixed[r];
          if (mr === 0) continue;
          for (let i = 0; i < cols.length; i++) delta[cols[i]] += mr * vals[i];
        }
        if (trace) trace.top[q].push(topK(attn, len, trace.topKeys));
      }
    }
    for (let q = 0; q < n; q++) {
      const ch = applyDelta(X[q], deltas[q]);
      if (trace) trace.changes[q].push(ch);
    }
    const wout = Ly.woutRow || (Ly.woutRow = new Map(Ly.wout.map((w) => [w.r, w])));
    for (let q = 0; q < n; q++) {
      if (!Ly.units) {
        if (trace) { trace.pre[q].push(null); trace.changes[q].push({ dims: new Int32Array(0), vals: new Float64Array(0) }); }
        continue;
      }
      const pre = matvec(X[q], Ly.win, new Float64Array(Ly.units));
      const delta = new Float64Array(d), act = [], actPre = [];
      for (let u = 0; u < pre.length; u++) if (pre[u] > 0) { act.push(u); actPre.push(pre[u]); }
      for (let i = 0; i < act.length; i++) {
        const w = wout.get(act[i]);
        if (!w) continue;
        for (let j = 0; j < w.cols.length; j++) delta[w.cols[j]] += actPre[i] * w.vals[j];
      }
      const ch = applyDelta(X[q], delta);
      if (trace) { trace.pre[q].push({ units: Int32Array.from(act), pre: Float64Array.from(actPre) }); trace.changes[q].push(ch); }
    }
  }
  if (trace) {
    trace.final = X;
    trace.logits = X.map((x) => matvec(x, model.unembed, new Float64Array(256)));
    trace.n = n;
  }
  return X;
}

/** A trace of a looped view: every position at once (see forwardRows). */
export function rowsTrace(model, x0Rows, tokens, topKeys) {
  const tr = new Trace(model, x0Rows.length, topKeys);
  forwardRows(model, x0Rows, model.layers.length, tr);
  tr.tokens = tokens;
  return tr;
}

/** Run the looped model (handformer/rewrite/engine.py LoopedTransformer.run): the block until no
 *  position routes in, then the coda greedily until EOS. ``loop`` is model.json's "loop" entry;
 *  ``coda`` is a Model of the coda's layers alone. Returns the passes (tape bytes, carried dims,
 *  the root's decision values), the final tape and the answer bytes. */
export function runLoop(model, coda, loop, prompt, maxPasses, maxNew) {
  const t0 = performance.now(), d = model.d, nb = loop.n_block;
  const adapter = byRow(loop.adapter);
  const holdByte = new Map(loop.holds.map(([dim, byte]) => [dim, byte]));
  const embedDense = (b) => { const x = new Float64Array(d); const e = model.embedRow.get(b); if (e) for (let i = 0; i < e.cols.length; i++) x[e.cols[i]] = e.vals[i]; return x; };
  let state = prompt.map(embedDense), tape = [...prompt];
  const passes = [];
  let halted = false;
  for (let r = 0; r < maxPasses; r++) {
    const X = forwardRows(model, state, nb);
    const root = X[X.length - 1];
    const routed = X.map((x) => x[loop.router] > 0.5);
    const grew = root[loop.grow] > 0.5;
    passes.push({ tape: [...tape], held: state.map((x) => loop.carried.map((j) => x[j])), routed: routed.filter(Boolean).length,
                  grew, acts: Object.fromEntries(loop.actions.map(([name, j]) => [name, Math.round(root[j] * 1e6) / 1e6])) });
    if (!routed.some(Boolean)) { halted = true; break; }
    const next = [], nextTape = [];
    for (let q = 0; q < X.length; q++) {
      if (!routed[q]) { next.push(state[q]); nextTape.push(tape[q]); continue; }
      next.push(matvec(X[q], adapter, new Float64Array(d)));
      let best = null, bv = -Infinity;
      for (const [dim, byte] of holdByte) if (X[q][dim] > bv) { bv = X[q][dim]; best = byte; }
      nextTape.push(best);
    }
    state = next; tape = nextTape;
    if (grew) { // room: a blank position just before the root
      state.splice(state.length - 1, 0, embedDense(loop.blank));
      tape.splice(tape.length - 1, 0, loop.blank);
    }
  }
  if (!halted) console.warn(`[engine] loop: no halt within ${maxPasses} passes`);
  const tr = new Trace(coda, state.length + 64, 1);
  let logits = null;
  for (const row of state) logits = tr.stepVec(row);
  const answer = [];
  for (let s = 0; s < maxNew; s++) {
    let best = 0;
    for (let b = 1; b < logits.length; b++) if (logits[b] > logits[best]) best = b;
    answer.push(best);
    if (best === model.eos) break;
    logits = tr.step(best);
  }
  log(`looped run: ${prompt.length} prompt bytes, ${passes.length} passes, ${answer.length} answer bytes in ${(performance.now() - t0).toFixed(0)}ms`);
  return { passes, finalTape: tape, finalState: state, answer, halted };
}

// ------------------------------------------------------------------ the recurrent-depth calculator
// handformer/loop/engine.py RecurrentDepthLM.run: an ordinary decoding loop over positions, where
// each position iterates the block (input e + loop state s + its position) until its router
// dim is 0, the adapter giving the next iteration's state. Earlier positions are read through
// their last iteration's keys and values (the cache). When the last residual's think dim is 1
// the next position's input is T of it (a latent position), else the argmax byte's embedding.

/** Run it: every position's kind of input, depth, last-iteration input row (without positions),
 *  the earlier iterations' rows of positions that iterated ("q,k" -> row), and the answer. */
export function runDepth(model, depth, prompt, maxTokens, maxIters) {
  const t0 = performance.now(), d = model.d, L = model.layers;
  const adapter = byRow(depth.adapter), thought = byRow(depth.thought);
  let cap = Math.max(64, 2 * prompt.length);
  let keys = L.map((Ly) => Ly.heads.map((h) => new Float64Array(cap * h.dk)));
  let vals = L.map((Ly) => Ly.heads.map((h) => new Float64Array(cap * h.dv)));
  const grow = () => {
    const g = (a) => { const b = new Float64Array(a.length * 2); b.set(a); return b; };
    keys = keys.map((hs) => hs.map(g)); vals = vals.map((hs) => hs.map(g)); cap *= 2;
  };
  const kb = new Float64Array(64), qb = new Float64Array(64), vb = new Float64Array(64);
  // one iteration at position q: this iteration's keys and values go into slot q (the last stays)
  const forward = (x, q) => {
    const delta = new Float64Array(d);
    let attn = new Float64Array(q + 1);
    for (let li = 0; li < L.length; li++) {
      const Ly = L[li];
      delta.fill(0);
      for (let hi = 0; hi < Ly.heads.length; hi++) {
        const h = Ly.heads[hi], dk = h.dk, dv = h.dv, K = keys[li][hi], V = vals[li][hi];
        K.set(matvec(x, h.k, kb.subarray(0, dk)), q * dk);
        V.set(matvec(x, h.v, vb.subarray(0, dv)), q * dv);
        const qq = matvec(x, h.q, qb.subarray(0, dk));
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
          const a = attn[t] / z;
          if (a === 0) continue;
          for (let c = 0; c < dv; c++) mixed[c] += a * V[t * dv + c];
        }
        for (const { r, cols, vals: w } of h.o) {
          const mr = mixed[r];
          if (mr === 0) continue;
          for (let i = 0; i < cols.length; i++) delta[cols[i]] += mr * w[i];
        }
      }
      for (let j = 0; j < d; j++) x[j] += delta[j];
      if (Ly.units) {
        const pre = matvec(x, Ly.win, new Float64Array(Ly.units));
        const wout = Ly.woutRow || (Ly.woutRow = new Map(Ly.wout.map((w) => [w.r, w])));
        delta.fill(0);
        for (let u = 0; u < pre.length; u++) {
          if (!(pre[u] > 0)) continue;
          const w = wout.get(u);
          if (!w) continue;
          for (let j = 0; j < w.cols.length; j++) delta[w.cols[j]] += pre[u] * w.vals[j];
        }
        for (let j = 0; j < d; j++) x[j] += delta[j];
      }
    }
    return x;
  };
  const tokens = [], lastIn = [], earlier = new Map(), inputs = [];
  const step = (e) => {
    if (tokens.length === cap) grow();
    const q = tokens.length;
    let s = new Float64Array(d), row = null, x = null, k = 0;
    for (k = 0; k < maxIters; k++) {
      const r = new Float64Array(d);
      for (let j = 0; j < d; j++) r[j] = e[j] + s[j];
      if (k) earlier.set(`${q},${k - 1}`, row);
      row = r;
      x = Float64Array.from(r);
      x[model.pos] += q;
      x[model.pos2] += q * q;
      forward(x, q);
      if (!(x[depth.router] > 0.5)) break;
      s = matvec(x, adapter, new Float64Array(d));
    }
    lastIn.push(row);
    inputs.push(e);
    return [x, Math.min(k + 1, maxIters)];
  };
  const embedDense = (b) => { const x = new Float64Array(d); const e = model.embedRow.get(b); if (e) for (let i = 0; i < e.cols.length; i++) x[e.cols[i]] = e.vals[i]; return x; };
  let x = null, it = 0;
  for (const b of prompt) { [x, it] = step(embedDense(b)); tokens.push({ latent: false, byte: b, iters: it }); }
  const answer = [];
  let finished = false;
  for (let n = 0; n < maxTokens; n++) {
    let e, byte = null;
    if (x[depth.think] > 0.5) e = matvec(x, thought, new Float64Array(d));
    else {
      const lg = matvec(x, model.unembed, new Float64Array(256));
      byte = 0;
      for (let b = 1; b < 256; b++) if (lg[b] > lg[byte]) byte = b;
      answer.push(byte);
      if (byte === model.eos) { finished = true; break; }
      e = embedDense(byte);
    }
    [x, it] = step(e);
    tokens.push({ latent: byte === null, byte, iters: it });
  }
  if (!finished) console.warn(`[engine] recurrent depth: no end byte within ${maxTokens} positions`);
  if (tokens.some((t) => t.iters >= maxIters)) { finished = false; console.warn(`[engine] a position reached ${maxIters} iterations`); }
  log(`recurrent-depth run: ${prompt.length} prompt bytes, ${tokens.filter((t) => t.latent).length} latent positions, ` +
      `${tokens.reduce((a, t) => a + t.iters, 0)} iterations, ${answer.length} bytes written in ${(performance.now() - t0).toFixed(0)}ms`);
  return { tokens, lastIn, earlier, inputs, answer, finished };
}
