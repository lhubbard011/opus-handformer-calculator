// Step-through visualizer for the hand-written arithmetic transformer.
// One run = the model's own greedy output plus one traced forward pass; step q
// is the computation at position q that produced byte q+1.

import { decodeF16, SEQ, DIV } from "./render.js";
import { drawHeatmap, attachHeatmapEvents, isTouch, showSheet, hideSheet } from "./heatmap.js";
import { LocalApi } from "./local-api.js";

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

// Static hosting (GitHub Pages): window.HANDFORMER_STATIC is the folder holding
// model.json + info.json, and the model runs in this page (local-api.js answers
// the same endpoints the Python server does). Otherwise: the Python server.
let local = null;

async function api(path, body, binary = false) {
  const t0 = performance.now();
  if (window.HANDFORMER_STATIC !== undefined) {
    if (!local) local = LocalApi.load(window.HANDFORMER_STATIC).catch((e) => { local = null; throw e; });
    const out = await (await local).call(path, body);
    if (binary) log(`${path} (in page) shape=[${out.meta.shape}] in ${(performance.now() - t0).toFixed(0)}ms`);
    return out;
  }
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
    const data = meta.dtype === "float32" ? new Float32Array(buf) : decodeF16(buf);
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

// One tooltip for every element carrying data-tip (list rows, tags, bytes, controls).
function wireTips() {
  const tip = $("tooltip");
  // Touch: tapping anything with an explanation shows it (and still does the tap's
  // normal action); tapping the explanation or anywhere else closes it.
  let touchAt = 0;
  document.addEventListener("pointerdown", (e) => { if (e.pointerType !== "mouse") touchAt = Date.now(); }, true);
  const touching = () => Date.now() - touchAt < 1000;
  document.addEventListener("click", (e) => {
    if (!touching() || e.target.closest("canvas")) return;
    const el = e.target.closest("[data-tip]");
    if (el && !e.target.closest("#tooltip")) showSheet(tip, el.dataset.tip);
    else hideSheet(tip);
  });
  document.addEventListener("mouseover", (e) => {
    if (touching()) return;
    const el = e.target.closest("[data-tip]");
    if (!el || e.target.closest("canvas")) return;
    tip.textContent = el.dataset.tip;
    tip.hidden = false;
  });
  document.addEventListener("mousemove", (e) => {
    if (touching() || tip.hidden || e.target.closest("canvas")) return;
    const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
    tip.style.left = `${Math.min(e.clientX + pad, innerWidth - w - 8)}px`;
    tip.style.top = `${Math.min(e.clientY + pad, innerHeight - h - 8)}px`;
  });
  document.addEventListener("mouseout", (e) => {
    if (touching()) return;
    if (e.target.closest("[data-tip]") && !e.relatedTarget?.closest?.("[data-tip]")) tip.hidden = true;
  });
}
const dimDoc = (name) => state.info?.dims.find((d) => d.name === name);

function setStatus(msg, isError = false) {
  $("status").textContent = msg;
  $("status").classList.toggle("error", isError);
}

const showChar = (c) => (c === "\n" ? "⏎" : c);
const fmt = (v) => (Math.abs(v - Math.round(v)) < 1e-6 ? String(Math.round(v)) : v.toFixed(4));
// Turn a step's raw values into its phase, calculator, pointer tags and a plain-language
// reading. Display only; both backends (server and in-browser) send the same raw values.
const MARK_DIMS = ["Ls", "F", "p", "c", "o", "a_lo", "b_hi", "E", "S", "Rb", "P", "R"];
const OPS = ["add", "sub", "mul", "div", "pow"];
function summarize(run, q, raw) {
  const v = raw.v, r = Math.round;
  const ch = (pos) => { const i = r(pos); return i >= 0 && i < run.chars.length ? JSON.stringify(run.chars[i]) : "?"; };
  let phase, op = null;
  if (q < run.prompt_len - 1) phase = "prompt";
  else if (v.phase1 > 0.5) phase = "open";
  else if (v.phase2 > 0.5) phase = "copy";
  else if (v.open > 0.5) { phase = "op"; op = OPS.find((o) => v[`act_${o}`] > 0.5) || null; }
  else phase = "rewrite";
  const redex = `the chosen operator is ${ch(v.o)} at ${r(v.o)}, its numbers sit between ${r(v.a_lo)} and ${r(v.b_hi)}` +
    (v.drop > 0.5 ? ", and its brackets will disappear" : "");
  let explain;
  if (phase === "prompt") explain = "Part of your prompt: given, not predicted.";
  else if (phase === "open") explain = "The line ends with '=' and no step is open, so the model opens one: it writes '['.";
  else if (phase === "copy") explain = v.end2 > 0.5 ? `The chosen piece is fully copied: write the step's '='. (${redex}.)`
    : `Copying the chosen piece into the step: byte ${r(v.a_lo + 1 + q - v.F)} ${ch(v.a_lo + 1 + q - v.F)}. (${redex}.)`;
  else if (phase === "op") {
    const sub = v.seen_semi > 0.5 ? "writing the answer" : op === "mul" && v.seen_bar > 0.5 ? "gathering the product" : "doing the working";
    explain = `The ${op || "?"} calculator is ${sub}, inside the step that starts at ${r(v.F)}.`;
  } else {
    const rr = r(q - v.Rb);
    if (v.end4 > 0.5) explain = `The new line is complete: ${v.cont > 0.5 ? "another operator remains, so write '='" : "no operator is left, so write the final newline"}.`;
    else explain = `Writing the simpler line, byte ${rr}: ${v.in1 > 0.5 ? "the part before the solved piece" : v.in12 > 0.5 ? "the solved piece's result" : "the part after it"} (copied from ${r(v.src4)} ${ch(v.src4)}).`;
  }
  return { q, phase, op, pred: raw.pred, explain, marks: Object.fromEntries(MARK_DIMS.map((d) => [d, r(v[d])])) };
}

const phaseClass = (s) => (s.phase === "op" ? `ph-${s.op || "prompt"}` : `ph-${s.phase}`);

// ------------------------------------------------------------------ runs

async function loadInfo() {
  state.info = await api("/api/info");
  log("model", state.info.model_id, state.info.d_model, "dims,", state.info.layers.length, "layers");
  $("mlp-layer").replaceChildren(...state.info.layers
    .filter((l) => l.units.length).map((l) => new Option(`${l.index + 1}`, l.index)));
  $("w-layer").replaceChildren(...state.info.layers.map((l) => new Option(`${l.index + 1}`, l.index)));
  $("w-layer").dispatchEvent(new Event("input"));
}

async function runExpression(expression) {
  stopPlay();
  setStatus("running…");
  try {
    const run = await api("/api/run", { expression });
    run.steps = run.steps.map((raw, q) => summarize(run, q, raw));
    if (run.model_id !== state.info?.model_id) {
      log("server model changed", state.info?.model_id, "->", run.model_id, "; reloading model description");
      await loadInfo();
    }
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
    try {
      state.pos.set(q, await api("/api/position", { run_id: state.run.run_id, q }));
    } catch (e) {
      if (!/unknown or evicted run/.test(e.message)) throw e;
      log("run", state.run.run_id, "no longer on the server (restarted?); running again");
      const keep = q;
      await runExpression(state.run.expression);
      await goto(keep);
      return null;  // the re-run's own goto has rendered this step
    }
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
    if (!p || gen !== state.gen) return; // a newer step (or a re-run) has rendered meanwhile
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
    el.dataset.tip = `position ${i}: "${showChar(c)}"\n` + (i >= run.prompt_len
      ? `Written by the model at step ${i - 1} (${producer.phase}${producer.op ? ", " + producer.op + " calculator" : ""}).`
      : "Part of the prompt you typed.") + (marks[i] ? `\nPointers to it right now: ${marks[i].join(", ")}` : "") +
      "\n(click to step to this position)";
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
    el.dataset.tip = `${dim} = ${v}${ch ? ` (the "${ch}" at position ${v})` : ""}\n${dimDoc(dim)?.doc || ""}`;
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
    s.dataset.tip = i === 0 ? `The model writes "${showChar(c)}" next: it has the highest score (${v.toFixed(1)}).`
      : `Score for "${showChar(c)}": ${v.toFixed(1)}. Lower than the winner, so it isn't written.`;
    el.appendChild(s);
  });
}

function headSpec(layer, head) {
  return state.info.layers[layer]?.heads[head];
}

function renderHeads(p) {
  const filter = $("head-filter").value.trim().toLowerCase();
  const el = $("heads");
  el.replaceChildren();
  let lastLayer = -1;
  for (const h of p.heads) {
    const spec = headSpec(h.layer, h.head);
    if (!spec) { logErr("head", h.layer, h.head, "missing from the model description; reload the page"); continue; }
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
    row.dataset.tip = `layer ${h.layer + 1} · ${spec.name}\n${spec.doc}\n\nAt this step it reads position ${k} ("${showChar(state.run.chars[k])}") with ${(w * 100).toFixed(1)}% of its attention.` +
      (wrote ? `\nIt adds: ${wrote}` : "") + "\n(click for its formulas and its full attention pattern)";
    row.innerHTML = `<span>${spec.name}</span><span class="num">← ${k} ${showChar(state.run.chars[k])} ${w < 0.999 ? (w * 100).toFixed(0) + "%" : ""}</span><span class="wrote">${wrote}</span>`;
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
    row.dataset.tip = `${d.name} = ${fmt(final)}\n${d.doc}\n\nwritten by: ${d.written_by}\nread by: ${d.read_by}\n(click to see how it changes layer by layer)`;
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
    const spec = state.info.layers[u.layer]?.units[u.unit];
    if (!spec) { logErr("unit", u.layer, u.unit, "missing from the model description; reload the page"); continue; }
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
    row.dataset.tip = `layer ${u.layer + 1} · unit #${u.unit}\n${spec.doc}\n\nReLU(${spec.read})\n= ${fmt(u.pre)} at this step → adds to ${spec.write}`;
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

// Per-map "how to use" text: what you see, how to read it, things to try.
const HOWTO = {
  residual: {
    see: ["Every note the model keeps (<b>rows</b>) at every character (<b>columns</b>), after the layer picked in <b>after</b>.",
          "Rows are grouped by part of the model: input, frame, parser, the calculators, output."],
    read: ["<b>Black</b> = 0. <b>Orange</b> = positive, <b>blue</b> = negative; brighter = bigger.",
           "Compare brightness <b>along a row</b>, not between rows: each row is scaled to its own maximum.",
           "<b>Long horizontal stripes</b> are pointers: they hold a position until a newer landmark appears.",
           "<b>Single bright cells</b> are facts about one character, such as which byte it is.",
           "The <b>outlined column</b> is the current step; dimmed columns haven't happened yet."],
    try: ["Step with ← → and watch which rows change at the outline.",
          "Set <b>after</b> to an early layer (L2 attn), then later ones, to watch notes appear layer by layer.",
          "Untick <b>focus active circuit</b> to see every part of the model at once.",
          "Click a cell to jump to that character and open that note in the Residual stream panel."],
  },
  step: {
    see: ["One character's notes (<b>rows</b>) after each layer (<b>columns</b>: embed, L1 attn, L1 mlp, …).",
          "The character is the current step (the outlined one in the byte strip)."],
    read: ["A row stays <b>black until the layer that writes it</b>: where its colour first appears is where it was computed.",
           "Most notes are written once and then just carried along to the right.",
           "Each row is scaled on its own, like the residual map."],
    try: ["Step to an interesting character (the '=' that starts a step, or a digit with a carry) to see the order the model works things out.",
          "Click a row to list its value after every layer in the Residual stream panel.",
          "Hover a cell for the note's meaning and exact value."],
  },
  attention: {
    see: ["For the head selected in the Attention heads panel: how much each character (<b>row</b>) looks at each earlier character (<b>column</b>).",
          "Only the lower-left half can light up: a character can't look ahead."],
    read: ["Each row is almost <b>one bright dot</b>: the head reads exactly one position.",
           "A <b>diagonal line</b> means 'read the byte at a fixed offset' (for example the previous byte).",
           "A <b>vertical stripe</b> means many characters read the same landmark.",
           "A <b>staircase</b> means a pointer jumping to newer landmarks as they appear."],
    try: ["Pick a head first, for example <b>latest F</b>, <b>first ) after p</b> or an <b>answer digit</b> head.",
          "Raise <b>gamma</b> to brighten weak attention, lower it to see only the strongest.",
          "Click a row to jump to that character."],
  },
  mlp: {
    see: ["For one layer (choose it with <b>layer</b>): how strongly each calculating unit (<b>row</b>) fired at each character (<b>column</b>).",
          "Rows are labelled with what each unit writes; hover for its formula."],
    read: ["Units only light up when their condition holds; <b>black means 'off'</b>.",
           "A unit that lights up in just a few columns is a test that is true only there (for example 'this digit equals 9').",
           "<b>Digit-ramp</b> units light up where a calculator is writing digits."],
    try: ["Choose a layer where a calculator writes its answer (L5 for add, L10 for mul or div) and step through a step [ … ].",
          "Leave <b>focus active circuit</b> on so only the working calculator's units are listed.",
          "Hover a cell to read the unit's hand-written formula and value."],
  },
  weights: {
    see: ["The <b>hand-written numbers themselves</b> for one matrix, not values during a run.",
          "Rows and columns are labelled by what they connect: notes, bytes, or calculating units."],
    read: ["<b>Black</b> = 0, which is most of every matrix. <b>Orange</b> positive, <b>blue</b> negative.",
           "Colour uses sign·log(1+|w|), so a weight of 1 and a gating weight of 10⁶ are both visible.",
           "<b>hide all-zero rows/cols</b> shows only the part that is actually wired."],
    try: ["<b>embedding</b>: which notes each byte sets (e.g. '7' sets val = 7 and is_digit = 1).",
          "A head's <b>W_OV</b>: which note it copies into which. <b>W_Q / W_K</b>: what it aims at.",
          "<b>MLP W_in</b>: what each unit reads. Hover any cell for the exact number."],
  },
};

const TOUCH_HINT = "On a touch screen, <b>tap</b> a cell to read it and tap the same cell again to do its click action; swipe sideways to scroll the map.";

function renderHowto() {
  const h = HOWTO[state.map];
  const list = (items) => `<ul>${items.map((t) => `<li>${t}</li>`).join("")}</ul>`;
  $("map-howto-body").innerHTML = h ? `<div class="howto-grid">
      <div><h4>What you see</h4>${list(h.see)}</div>
      <div><h4>How to read it</h4>${list(h.read)}</div>
      <div><h4>Try this</h4>${list(isTouch() ? [...h.try, TOUCH_HINT] : h.try)}</div></div>` : "";
}

async function renderMap(p) {
  if (!state.run) return;
  document.querySelectorAll("#map-controls label").forEach((l) => l.classList.toggle("on", l.dataset.for === state.map));
  const canvas = $("map");
  const run = state.run;
  const n = run.chars.length;
  const q = state.q;
  const cap = $("map-caption");
  renderHowto();

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
      valueText: (r, c) => `${rows[r].name} at position ${c} ${charAt(c)}\n= ${fmt(data[rows[r].i * n + c] * meta.scale[rows[r].i])}\n${rows[r].doc}`,
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
      valueText: (r, c) => `${rows[r].name} after ${labels[c]}\n= ${fmt(rows[r].vals[c])}\n${rows[r].doc}`,
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
  } else if (state.map === "weights") {
    const kind = $("w-kind").value;
    const body = { kind, trim: $("w-trim").checked };
    if (!["embed", "unembed"].includes(kind)) body.layer = Number($("w-layer").value);
    if (["wq", "wk", "wov"].includes(kind)) body.head = Number($("w-head").value);
    const { key, meta, data } = await mapData("/api/weights", body);
    const [rows, cols] = meta.shape;
    const lmax = Math.log10(1 + meta.max_abs);
    cap.textContent = `${meta.title}: ${meta.full_shape.join(" × ")} stored, ${meta.nonzero} nonzero` +
      (body.trim ? `, showing ${rows} × ${cols} non-empty rows/cols` : "") +
      " · colour = sign·log(1+|w|) (blue negative, orange positive); hover for exact values";
    drawHeatmap(canvas, {
      key, rows, cols,
      get: (r, c) => data[r * cols + c],
      norm: (v) => 0.5 + 0.5 * Math.sign(v) * Math.log10(1 + Math.abs(v)) / (lmax || 1), lut: DIV,
      rowLabels: meta.rows,
      valueText: (r, c) => `row ${meta.rows[r]}\ncol ${meta.cols[c]}\nw = ${data[r * cols + c]}`,
    }, {});
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
      valueText: (r, c) => `unit #${rows[r].i} at position ${c} ${charAt(c)}\n${rows[r].doc}\nReLU(${rows[r].read})\n= ${fmt(data[rows[r].i * n + c] * meta.scale[rows[r].i])} → ${rows[r].write}`,
      onClick: (_r, c) => goto(c),
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
  wireTips();
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
  const fillHeads = () => {
    const layer = state.info.layers[Number($("w-layer").value)];
    $("w-head").replaceChildren(...layer.heads.map((h, i) => new Option(`${i}: ${h.name}`, i)));
  };
  $("w-layer").addEventListener("input", fillHeads);
  for (const id of ["snapshot", "mlp-layer", "gamma", "w-kind", "w-layer", "w-head", "w-trim"]) {
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

/** Phones: the step controls are a fixed bottom bar; keep its height in --bar-h for padding. */
function trackBar() {
  const bar = document.querySelector(".controls");
  const set = () => {
    const fixed = getComputedStyle(bar).position === "fixed";
    document.documentElement.style.setProperty("--bar-h", fixed ? `${bar.offsetHeight}px` : "0px");
  };
  new ResizeObserver(set).observe(bar);
  window.addEventListener("resize", set);
  set();
}

async function main() {
  wire();
  trackBar();
  if (matchMedia("(max-width: 700px)").matches) $("map-howto").open = false; // long on a phone; one tap opens it
  try {
    await loadInfo();
    setStatus(`${state.info.layers.length} layers · d_model ${state.info.d_model} · ${state.info.nonzero_params} hand-written params`);
    await runExpression($("expr-input").value);
  } catch (e) {
    setStatus(e.message, true);
  }
}

main();
