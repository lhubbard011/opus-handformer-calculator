// Step-through visualizer for the hand-written arithmetic transformer.
// One run = the model's own greedy output plus one traced forward pass; step q
// is the computation at position q that produced byte q+1.

import { decodeF16, SEQ, DIV } from "./render.js";
import { drawHeatmap, attachHeatmapEvents } from "./heatmap.js";

const $ = (id) => document.getElementById(id);
const log = (...args) => console.info("[handviz]", ...args);
const logErr = (...args) => console.error("[handviz]", ...args);
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const MARK_LABELS = { Ls: "Ls", F: "[", p: "p", c: "c", o: "o", a_lo: "a_lo", b_hi: "b_hi", E: "=", S: ";", Rb: "]" };

const state = {
  info: null,
  run: null, // /api/run response
  q: 0,
  pos: new Map(), // q -> /api/position response
  head: null, // [layer, head]
  openDim: null,
  playing: false,
  gen: 0,
  map: "residual",
  maps: new Map(), // cache key -> { meta, data } for binary heatmap payloads
};

// ------------------------------------------------------------------ API

async function api(path, body, binary = false) {
  const t0 = performance.now();
  const r = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: body === undefined ? {} : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!r.ok) {
    let detail = await r.text();
    try {
      const j = JSON.parse(detail);
      detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch { /* keep raw text */ }
    logErr(`${path} failed ${r.status}: ${detail}`);
    throw new Error(`${path}: ${detail}`);
  }
  if (binary) {
    const buf = await r.arrayBuffer();
    const meta = JSON.parse(r.headers.get("X-Meta"));
    const data = decodeF16(buf);
    const expected = meta.shape.reduce((a, b) => a * b, 1);
    if (data.length !== expected) throw new Error(`${path}: got ${data.length} values, expected ${expected}`);
    log(`${path} ${buf.byteLength} bytes shape=[${meta.shape}] in ${(performance.now() - t0).toFixed(0)}ms`);
    return { meta, data };
  }
  const j = await r.json();
  log(`${path} ok in ${(performance.now() - t0).toFixed(0)}ms`);
  return j;
}

// Circuits that matter at this step (used by "focus active circuit").
function focusSet() {
  const s = state.run?.steps[state.q];
  const set = new Set(["input", "output", "frame"]);
  if (!s) return set;
  if (s.phase === "copy" || s.phase === "rewrite") set.add("parser");
  if (s.phase === "op" && s.op) set.add(s.op);
  return set;
}
const focused = (circuit) => !$("focus").checked || focusSet().has(circuit);

function setStatus(msg, isError = false) {
  $("status").textContent = msg;
  $("status").classList.toggle("error", isError);
}

const showChar = (c) => (c === "\n" ? "⏎" : c);
const fmt = (v) => (Math.abs(v - Math.round(v)) < 1e-6 ? String(Math.round(v)) : v.toFixed(4));
const phaseClass = (s) => (s.phase === "op" ? `ph-${s.op || "prompt"}` : `ph-${s.phase}`);

// ------------------------------------------------------------------ runs

async function runExpression(expression) {
  stopPlay();
  setStatus("running…");
  try {
    const run = await api("/api/run", { expression });
    state.run = run;
    state.pos.clear();
    state.head = null;
    state.openDim = null;
    state.maps.clear();
    const snap = $("snapshot");
    snap.replaceChildren(...run.snapshots.map((l, i) => new Option(l, i)));
    snap.value = run.snapshots.length - 1;
    $("scrubber").min = run.prompt_len - 1;
    $("scrubber").max = run.chars.length - 2;
    const answer = run.chars.join("").split("]").pop().trim();
    setStatus(`${run.chars.length - run.prompt_len} bytes written by the model · answer ${answer}` +
      (run.finished ? "" : " · (stopped before EOS)"));
    log("run", run.run_id, run.expression, run.chars.length, "bytes");
    await goto(run.prompt_len - 1);
  } catch (e) {
    setStatus(e.message, true);
  }
}

async function position(q) {
  if (!state.pos.has(q)) {
    state.pos.set(q, await api("/api/position", { run_id: state.run.run_id, q }));
  }
  return state.pos.get(q);
}

async function goto(q) {
  if (!state.run) return;
  const run = state.run;
  q = clamp(q, run.prompt_len - 1, run.chars.length - 2);
  state.q = q;
  const gen = ++state.gen;
  $("scrubber").value = q;
  renderStrip();
  renderStep();
  try {
    const p = await position(q);
    if (gen !== state.gen) return; // a newer step was requested meanwhile
    renderLogits(p);
    renderHeads(p);
    renderDims(p);
    renderUnits(p);
    renderHeadDetail(p);
    await renderMap(p);
  } catch (e) {
    setStatus(e.message, true);
  }
}

// ------------------------------------------------------------------ rendering

function renderStrip() {
  const run = state.run;
  const q = state.q;
  const step = run.steps[q];
  const marks = {};
  for (const [dim, label] of Object.entries(MARK_LABELS)) {
    const at = step.marks[dim];
    if (step.phase === "prompt" || step.phase === "open") break;
    if (at >= 0 && at < run.chars.length) (marks[at] ||= []).push(label);
  }
  const srcs = new Set();
  const p = state.pos.get(q);
  if (p && state.head) {
    const h = p.heads.find((x) => x.layer === state.head[0] && x.head === state.head[1]);
    if (h) h.top.filter(([, w]) => w > 0.05).forEach(([k]) => srcs.add(k));
  }
  const strip = $("tokens");
  strip.replaceChildren();
  run.chars.forEach((c, i) => {
    const el = document.createElement("span");
    const producer = i >= run.prompt_len ? run.steps[i - 1] : run.steps[0];
    el.className = "chip " + (i >= run.prompt_len ? phaseClass(producer) : "ph-prompt");
    if (i === q) el.classList.add("query");
    if (i === q + 1) el.classList.add("output");
    if (i > q + 1) el.classList.add("future");
    if (srcs.has(i)) el.classList.add("src");
    el.textContent = showChar(c);
    el.title = `pos ${i}` + (i >= run.prompt_len ? ` · written at step ${i - 1} (${producer.phase}${producer.op ? " " + producer.op : ""})` : " · prompt");
    if (marks[i]) {
      const tags = document.createElement("span");
      tags.className = "tags";
      marks[i].forEach((m) => {
        const t = document.createElement("span");
        t.className = "tag";
        t.textContent = m;
        tags.appendChild(t);
      });
      el.appendChild(tags);
    }
    const idx = document.createElement("span");
    idx.className = "idx";
    idx.textContent = i % 5 === 0 ? i : "";
    el.appendChild(idx);
    el.addEventListener("click", () => goto(i));
    strip.appendChild(el);
  });
  strip.querySelector(".chip.output")?.scrollIntoView({ block: "nearest", inline: "nearest" });
}

function renderStep() {
  const run = state.run;
  const s = run.steps[state.q];
  const n = run.chars.length - 1 - (run.prompt_len - 1);
  $("step-label").textContent = `step ${state.q - run.prompt_len + 2} / ${n} · query pos ${state.q} ${JSON.stringify(run.chars[state.q])} → writes ${JSON.stringify(run.chars[state.q + 1])}`;
  $("explain").textContent = s.explain;
  $("explain").style.borderLeftColor = getComputedStyle(document.documentElement)
    .getPropertyValue(`--${phaseClass(s)}`) || "var(--accent)";
  const m = $("marks");
  m.replaceChildren();
  for (const [dim, v] of Object.entries(s.marks)) {
    const el = document.createElement("span");
    const ch = v >= 0 && v < run.chars.length ? showChar(run.chars[v]) : "";
    el.innerHTML = `${dim} <b>${v}</b> <span class="muted">${ch}</span>`;
    m.appendChild(el);
  }
}

function renderLogits(p) {
  const el = $("logits");
  el.replaceChildren();
  p.logits.forEach(([c, v], i) => {
    const s = document.createElement("span");
    s.className = "logit" + (i === 0 ? " win" : "");
    s.textContent = `${showChar(c)} ${v.toFixed(1)}`;
    el.appendChild(s);
  });
}

function headSpec(layer, head) {
  return state.info.layers[layer].heads[head];
}

function renderHeads(p) {
  const filter = $("head-filter").value.trim().toLowerCase();
  const el = $("heads");
  el.replaceChildren();
  let lastLayer = -1;
  for (const h of p.heads) {
    const spec = headSpec(h.layer, h.head);
    if (filter && !spec.name.toLowerCase().includes(filter)) continue;
    if (!focused(spec.circuit)) continue;
    if (h.layer !== lastLayer) {
      const lh = document.createElement("div");
      lh.className = "layer-head";
      lh.textContent = `layer ${h.layer + 1}`;
      el.appendChild(lh);
      lastLayer = h.layer;
    }
    const [k, w] = h.top[0];
    const row = document.createElement("div");
    row.className = "row head-row";
    if (state.head && state.head[0] === h.layer && state.head[1] === h.head) row.classList.add("sel");
    const wrote = Object.entries(h.wrote).map(([d, v]) => `${d}=${fmt(v)}`).join(" ");
    row.innerHTML = `<span>${spec.name}</span><span class="num">← ${k} ${showChar(state.run.chars[k])} ${w < 0.999 ? (w * 100).toFixed(0) + "%" : ""}</span><span class="wrote" title="${wrote}">${wrote}</span>`;
    row.addEventListener("click", () => {
      state.head = [h.layer, h.head];
      renderHeads(p);
      renderHeadDetail(p);
      renderStrip();
      if (state.map === "attention") renderMap(p);
    });
    el.appendChild(row);
  }
}

function renderHeadDetail(p) {
  const el = $("head-detail");
  if (!state.head) {
    el.textContent = "click a head";
    el.classList.add("muted");
    return;
  }
  el.classList.remove("muted");
  const [layer, head] = state.head;
  const spec = headSpec(layer, head);
  const h = p.heads.find((x) => x.layer === layer && x.head === head);
  const score = spec.query.map((qf, i) => `(${qf})·key[${spec.key[i]}]`).join("\n     + ");
  const value = Object.entries(spec.value).map(([src, dst]) => `${src} → ${dst}`).join("\n       ");
  const top = h.top.map(([k, w]) => `${k} ${JSON.stringify(state.run.chars[k])} ${(w * 100).toFixed(2)}%`).join(", ");
  const wrote = Object.entries(h.wrote).map(([d, v]) => `${d} += ${fmt(v)}`).join("\n       ") || "(nothing)";
  el.textContent = `L${layer + 1} ${spec.name}\nscore = ${score}\nvalue: ${value}\nreads: ${top}\nwrote: ${wrote}`;
}

function renderDims(p) {
  const filter = $("dim-filter").value.trim().toLowerCase();
  const showZero = $("show-zero").checked;
  const labels = state.run.snapshots;
  const el = $("dims");
  el.replaceChildren();
  let lastCat = null;
  state.info.dims.forEach((d, i) => {
    const vals = p.values[i];
    const final = vals[vals.length - 1];
    if (!showZero && vals.every((v) => Math.abs(v) < 1e-9)) return;
    if (filter && !d.name.toLowerCase().includes(filter)) return;
    if (!focused(d.cat)) return;
    if (d.cat !== lastCat) {
      const ch = document.createElement("div");
      ch.className = "cat-head";
      ch.textContent = d.cat;
      el.appendChild(ch);
      lastCat = d.cat;
    }
    let written = 0;
    for (let s = 1; s < vals.length; s++) if (Math.abs(vals[s] - vals[s - 1]) > 1e-9) { written = s; break; }
    const row = document.createElement("div");
    row.className = "row dim-row" + (state.openDim === d.name ? " sel" : "");
    row.innerHTML = `<span>${d.name}</span><span class="num">${fmt(final)}</span><span class="muted small">${written ? labels[written] : "embed"}</span>`;
    row.addEventListener("click", () => {
      state.openDim = state.openDim === d.name ? null : d.name;
      renderDims(p);
    });
    el.appendChild(row);
    if (state.openDim === d.name) {
      const t = document.createElement("div");
      t.className = "traj";
      t.textContent = labels.map((l, s) => `${l.padEnd(9)} ${fmt(vals[s])}`)
        .filter((_, s) => s === 0 || Math.abs(vals[s] - vals[s - 1]) > 1e-9).join("\n");
      row.appendChild(t);
    }
  });
}

function renderUnits(p) {
  const filter = $("unit-filter").value.trim().toLowerCase();
  const el = $("units");
  el.replaceChildren();
  let lastLayer = -1;
  for (const u of p.units) {
    const spec = state.info.layers[u.layer].units[u.unit];
    const text = `${spec.read} ${spec.write}`;
    if (filter && !text.toLowerCase().includes(filter)) continue;
    if (!focused(spec.circuit)) continue;
    if (u.layer !== lastLayer) {
      const lh = document.createElement("div");
      lh.className = "layer-head";
      lh.textContent = `layer ${u.layer + 1} mlp`;
      el.appendChild(lh);
      lastLayer = u.layer;
    }
    const row = document.createElement("div");
    row.className = "row unit-row";
    row.innerHTML = `<span class="muted">#${u.unit}</span><span class="num">${fmt(u.pre)}</span><span class="f">ReLU(<b>${spec.read}</b>) → ${spec.write}</span>`;
    el.appendChild(row);
  }
  if (!el.children.length) el.textContent = "no active units match";
}

// ------------------------------------------------------------------ heatmaps

async function mapData(path, body) {
  const key = path + JSON.stringify(body);
  if (!state.maps.has(key)) state.maps.set(key, await api(path, body, true));
  return { key, ...state.maps.get(key) };
}

function groupsOf(cats) {
  const groups = [];
  cats.forEach((c, r) => { if (r === 0 || c !== cats[r - 1]) groups.push({ row: r, label: c }); });
  return groups;
}

const charAt = (c) => JSON.stringify(state.run.chars[c] ?? "");

async function renderMap(p) {
  if (!state.run) return;
  document.querySelectorAll("#map-controls label").forEach((l) => l.classList.toggle("on", l.dataset.for === state.map));
  const canvas = $("map");
  const run = state.run;
  const n = run.chars.length;
  const q = state.q;
  const cap = $("map-caption");

  if (state.map === "residual") {
    const snap = Number($("snapshot").value);
    const { key, meta, data } = await mapData("/api/residual", { run_id: run.run_id, snapshot: snap });
    const rows = state.info.dims.map((d, i) => ({ ...d, i }))
      .filter((d) => focused(d.cat) && meta.scale[d.i] > 1e-9);
    cap.textContent = `every residual dim (rows) at every position (columns) after ${meta.label}; ` +
      "each row scaled by its own max |value| (blue negative, orange positive); positions after the current step are dimmed; click a cell to jump there";
    drawHeatmap(canvas, {
      key: key + rows.map((r) => r.i).join(","),
      rows: rows.length, cols: n,
      get: (r, c) => data[rows[r].i * n + c], norm: (v) => (v + 1) / 2, lut: DIV,
      rowLabels: rows.map((r) => r.name), groups: groupsOf(rows.map((r) => r.cat)),
      valueText: (r, c) => `${rows[r].name} @ ${c} ${charAt(c)}\n= ${fmt(data[rows[r].i * n + c] * meta.scale[rows[r].i])}`,
      onClick: (r, c) => { state.openDim = rows[r].name; goto(c); },
    }, { q });
  } else if (state.map === "step") {
    const labels = run.snapshots;
    const rows = state.info.dims.map((d, i) => ({ ...d, i, vals: p.values[i] }))
      .filter((d) => focused(d.cat) && d.vals.some((v) => Math.abs(v) > 1e-9));
    const scale = rows.map((r) => Math.max(...r.vals.map(Math.abs)));
    cap.textContent = `position ${q} ${charAt(q)}: every dim (rows) after each sublayer (columns: embed, L1 attn, L1 mlp, …); ` +
      "a colour change marks the sublayer that wrote the dim";
    drawHeatmap(canvas, {
      key: `step${run.run_id}:${q}:${rows.map((r) => r.i).join(",")}`,
      rows: rows.length, cols: labels.length,
      get: (r, c) => rows[r].vals[c] / scale[r], norm: (v) => (v + 1) / 2, lut: DIV,
      rowLabels: rows.map((r) => r.name), groups: groupsOf(rows.map((r) => r.cat)),
      valueText: (r, c) => `${rows[r].name} after ${labels[c]}\n= ${fmt(rows[r].vals[c])}`,
      onClick: (r) => { state.openDim = rows[r].name; renderDims(p); },
    }, {});
  } else if (state.map === "attention") {
    if (!state.head) {
      cap.textContent = "select a head in the Attention heads panel";
      drawHeatmap(canvas, { rows: 0, cols: 0 }, {});
      return;
    }
    const [layer, head] = state.head;
    const { key, meta, data } = await mapData("/api/attention", { run_id: run.run_id, layer, head });
    const g = Number($("gamma").value);
    cap.textContent = `L${layer + 1} ${meta.name}: query positions (rows) × key positions (columns); ` +
      "the outlined row is the current step; each row is a near one-hot pointer";
    drawHeatmap(canvas, {
      key, rows: n, cols: n, square: true,
      get: (r, c) => data[r * n + c], norm: (v) => Math.pow(v, g), lut: SEQ,
      valueText: (r, c) => `query ${r} ${charAt(r)} → key ${c} ${charAt(c)}\nweight ${(data[r * n + c] * 100).toFixed(3)}%`,
      onClick: (r) => goto(r),
    }, { q });
  } else {
    const sel = $("mlp-layer");
    const layer = Number(sel.value);
    const { key, meta, data } = await mapData("/api/mlp", { run_id: run.run_id, layer });
    const specs = state.info.layers[layer].units;
    const rows = specs.map((u, i) => ({ ...u, i })).filter((u) => focused(u.circuit) && meta.scale[u.i] > 1e-9);
    cap.textContent = `layer ${layer + 1} MLP: post-ReLU activation of each unit (rows) at every position (columns), ` +
      "each unit scaled by its max; hover for the unit's hand-written formula";
    drawHeatmap(canvas, {
      key: key + rows.map((r) => r.i).join(","),
      rows: rows.length, cols: n,
      get: (r, c) => data[rows[r].i * n + c], norm: (v) => v, lut: SEQ,
      rowLabels: rows.map((r) => `#${r.i} ${r.write}`), groups: groupsOf(rows.map((r) => r.circuit)),
      valueText: (r, c) => `#${rows[r].i} @ ${c} ${charAt(c)}: ReLU(${rows[r].read})\n= ${fmt(data[rows[r].i * n + c] * meta.scale[rows[r].i])} → ${rows[r].write}`,
      onClick: (r, c) => goto(c),
    }, { q });
  }
}

// ------------------------------------------------------------------ controls

function stopPlay() {
  state.playing = false;
  $("btn-play").textContent = "Play";
}

async function play() {
  if (state.playing) return stopPlay();
  state.playing = true;
  $("btn-play").textContent = "Pause";
  while (state.playing && state.run && state.q < state.run.chars.length - 2) {
    await goto(state.q + 1);
    await sleep(Number($("speed").value));
  }
  stopPlay();
}

function wire() {
  $("expr-form").addEventListener("submit", (e) => {
    e.preventDefault();
    runExpression($("expr-input").value);
  });
  $("btn-first").addEventListener("click", () => state.run && goto(state.run.prompt_len - 1));
  $("btn-back").addEventListener("click", () => goto(state.q - 1));
  $("btn-step").addEventListener("click", () => goto(state.q + 1));
  $("btn-play").addEventListener("click", play);
  $("scrubber").addEventListener("input", (e) => goto(Number(e.target.value)));
  for (const [id, fn] of [["head-filter", renderHeads], ["dim-filter", renderDims], ["unit-filter", renderUnits]]) {
    $(id).addEventListener("input", () => state.pos.has(state.q) && fn(state.pos.get(state.q)));
  }
  document.querySelectorAll("[data-map]").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll("[data-map]").forEach((x) => x.classList.toggle("active", x === b));
    state.map = b.dataset.map;
    if (state.pos.has(state.q)) renderMap(state.pos.get(state.q));
  }));
  for (const id of ["snapshot", "mlp-layer", "gamma"]) {
    $(id).addEventListener("input", () => state.pos.has(state.q) && renderMap(state.pos.get(state.q)));
  }
  $("focus").addEventListener("change", () => state.run && goto(state.q));
  window.addEventListener("resize", () => state.pos.has(state.q) && renderMap(state.pos.get(state.q)));
  attachHeatmapEvents($("map"), $("tooltip"));
  $("show-zero").addEventListener("change", () => state.pos.has(state.q) && renderDims(state.pos.get(state.q)));
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" && e.target.type === "text") return;
    if (e.key === "ArrowRight") { e.preventDefault(); goto(state.q + 1); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); goto(state.q - 1); }
    else if (e.key === "Home" && state.run) { e.preventDefault(); goto(state.run.prompt_len - 1); }
    else if (e.key === "End" && state.run) { e.preventDefault(); goto(state.run.chars.length - 2); }
    else if (e.key === " ") { e.preventDefault(); play(); }
  });
}

async function main() {
  wire();
  try {
    state.info = await api("/api/info");
    $("mlp-layer").replaceChildren(...state.info.layers
      .filter((l) => l.units.length).map((l) => new Option(`${l.index + 1}`, l.index)));
    log("model", state.info.d_model, "dims,", state.info.layers.length, "layers,", state.info.nonzero_params, "params");
    setStatus(`${state.info.layers.length} layers · d_model ${state.info.d_model} · ${state.info.nonzero_params} hand-written params`);
    await runExpression($("expr-input").value);
  } catch (e) {
    setStatus(e.message, true);
  }
}

main();
