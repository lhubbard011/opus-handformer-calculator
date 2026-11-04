// Step-through visualizer for the hand-written arithmetic transformer.
// One run = the model's own greedy output plus one traced forward pass; step q
// is the computation at position q that produced byte q+1.

import { decodeF16, SEQ, DIV } from "./render.js";
import { drawHeatmap, attachHeatmapEvents, isTouch, showSheet, hideSheet, setTip } from "./heatmap.js";
import { LocalApi } from "./local-api.js";
import { drawGraph, drawOverview } from "./graph.js";

const $ = (id) => document.getElementById(id);
// The input is a one-line box (calculator, Life) or a code box (C): read and write it through these.
const inputBox = () => (state.info?.multiline ? $("code-input") : $("expr-input"));
const getInput = () => inputBox().value;
const setInput = (v) => { inputBox().value = v; };
const log = (...args) => console.info("[handviz]", ...args);
const logErr = (...args) => console.error("[handviz]", ...args);
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Tags that are positions computed for display, not dims of their own.
const PTR_DOCS = {
  cell: "The same cell one generation back (L bytes back): the 'self' head reads it.",
  src: "The source byte being classified at this step (position = this position - E + 1).",
  nb: "A neighbour of that cell, read by one of the eight neighbour heads ('/' ':' '=' read as dead).",
};
// Pointer tags drawn above the sequence, per model (dim -> label).
const MARK_LABELS_BY = {
  calc: { Ls: "Ls", F: "[", p: "p", c: "c", o: "o", a_lo: "a_lo", b_hi: "b_hi", E: "=", S: ";", Rb: "]" },
  life: { C: ":", cell: "cell", nb: "nb" },
  c: { E: "end", src: "src" },
  parse: { E: "end", M: "lex end", Lsep: "rec", la_pos: "next", mpos: "pop" },
};

const state = {
  model: "calc", // which hand-written model the page shows (handformer/viz/models.py)
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
  ko: { heads: [], units: [], dims: [] }, // knocked-out parts (see handformer/ablate.py)
  runSeq: 0, // latest run request; an older response that arrives late is dropped
  running: false,
  baseline: new Map(), // expression -> answer with nothing knocked out
};

// ------------------------------------------------------------------ API

// Static hosting (GitHub Pages): window.HANDFORMER_STATIC is the folder holding
// model.json + info.json, and the model runs in this page (local-api.js answers
// the same endpoints the Python server does). Otherwise: the Python server.
// One page, several models (calculator, Game of Life): every request names state.model.
const locals = {}; // model -> LocalApi (static site: each model's files live in data/<model>/)

async function api(path, body, binary = false) {
  const t0 = performance.now();
  if (body !== undefined) body = { ...body, model: state.model };
  if (window.HANDFORMER_STATIC !== undefined) {
    const m = state.model;
    if (!locals[m]) locals[m] = LocalApi.load(`${window.HANDFORMER_STATIC}${m}/`).catch((e) => { delete locals[m]; throw e; });
    const out = await (await locals[m]).call(path, body);
    if (binary) log(`${path} (in page) shape=[${out.meta.shape}] in ${(performance.now() - t0).toFixed(0)}ms`);
    return out;
  }
  const r = await fetch(body === undefined ? `${path}?model=${encodeURIComponent(state.model)}` : path, {
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
  if (state.model === "life") return new Set(["input", "output", "life"]); // one circuit, always on
  const set = new Set(["input", "output", "frame"]);
  if (!s) return set;
  if (s.phase === "copy" || s.phase === "rewrite") set.add("parser");
  if (s.phase === "op" && s.op) set.add(s.op);
  return set;
}
const focused = (circuit) => !$("focus").checked || ["c", "parse"].includes(state.model) || focusSet().has(circuit);

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
    if (el && !e.target.closest("#tooltip")) showSheet(tip, el.dataset.tip, el.dataset.tipPtr || "");
    else hideSheet(tip);
  });
  document.addEventListener("mouseover", (e) => {
    if (touching()) return;
    const el = e.target.closest("[data-tip]");
    if (!el || e.target.closest("canvas")) return;
    setTip(tip, el.dataset.tip, el.dataset.tipPtr || "");
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
// "View source": the circuit line that wrote a head, unit or dim (from the model
// description; the repository and branch come from docs/links.toml).
function srcUrl(where) {
  const r = state.info?.source;
  if (!r || !where) return null;
  const i = where.lastIndexOf(":");
  return `${r.url}/blob/${r.ref}/${where.slice(0, i)}#L${where.slice(i + 1)}`;
}
function srcLinks(wheres) {
  const frag = document.createDocumentFragment();
  for (const w of [].concat(wheres || [])) {
    const u = srcUrl(w);
    if (!u) continue;
    const a = document.createElement("a");
    a.className = "src";
    a.href = u;
    a.target = "_blank";
    a.rel = "noopener";
    a.textContent = `${w.split("/").pop()} ↗`;
    a.dataset.tip = `Open the Python line that wrote this: ${w}`;
    a.addEventListener("click", (e) => e.stopPropagation()); // don't toggle the row
    frag.append(a, " ");
  }
  return frag;
}
const srcText = (wheres) => ([].concat(wheres || []).length ? `\nsource: ${[].concat(wheres).join(", ")}` : "");
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

// Game of Life: what the byte written at this step is, and why (from the model's own values:
// C latest ':', L block length, F first row end, n live neighbours, self, nxt). Display only.
const NEIGH = [["ul", -1, -1], ["u", -1, 0], ["ur", -1, 1], ["l", 0, -1], ["r", 0, 1], ["dl", 1, -1], ["d", 1, 0], ["dr", 1, 1]];
function summarizeLife(run, q, raw) {
  const v = raw.v, r = Math.round, p = raw.pred;
  const W1 = r(v.F) - 2, L = r(v.L), o = q - r(v.C);
  const gen = run.chars.slice(0, q + 1).filter((c) => c === "=").length; // generation being written
  const cell = q + 1 - L; // the same cell one generation back
  const marks = { C: r(v.C), L, F: r(v.F) };
  let phase, explain, nbs = [];
  if (q < run.prompt_len - 1) { phase = "prompt"; explain = "Part of your prompt: the count, then the starting grid."; }
  else if (p === "#" || p === ".") {
    phase = "cell";
    const row = Math.floor(o / W1), col = o % W1, n = r(v.n), was = v.self > 0.5, now = v.nxt > 0.5;
    const nb = `${n} live neighbour${n === 1 ? "" : "s"}`;
    const why = now ? (was ? `survives with ${nb} (2 or 3)` : "is born with exactly 3 live neighbours")
      : (was ? `dies with ${nb} (it needs 2 or 3)` : `stays dead with ${nb} (it needs exactly 3)`);
    explain = `Generation ${gen}, row ${row}, column ${col}: one generation back it was ${was ? "alive" : "dead"}, and it ${why}, ` +
      `so the model writes '${p}'. It read its 3×3 neighbourhood ${L} bytes back (tags "cell" and "nb").`;
    marks.cell = cell;
    nbs = NEIGH.filter(([, dr]) => (dr < 0 ? v.up_ok > 0.5 : dr > 0 ? v.down_ok > 0.5 : true))
      .map(([, dr, dc]) => cell + dr * W1 + dc);
  } else if (p === "/" || p === "=") {
    phase = "rowend";
    explain = p === "/" ? `One generation back has '/' at this offset (${L} bytes back), so this row ends here: copy it.`
      : `One generation back has '=' here: the grid of generation ${gen} is complete.`;
    marks.cell = cell;
  } else if (p === ":") { phase = "count"; explain = "After the count digit comes ':', then the grid."; }
  else {
    phase = "count";
    const k = [...Array(10).keys()].find((d) => v[`K${d}`] > 0.5);
    explain = p === "\n" ? `This block's count was ${k ?? "?"}: the last generation is written, so the model writes the final newline.`
      : `Generation ${gen} is complete and its count was ${k ?? "?"}, so the next block starts with ${p} (one less): the count lives in the text.`;
  }
  return { q, phase, op: null, pred: p, explain, marks, nbs };
}

// C highlighter: the state letter written, and the one rule dim that is 1 (the model's own values).
const C_CLASS = { B: "comment", D: "comment", C: "comment", E: "comment", L: "comment", c: "comment", s: "string",
  S: "string", t: "string", z: "string", P: "string", G: "string", H: "string",
  a: "char", A: "char", b: "char", y: "char", g: "char", h: "char", p: "preproc", n: "number",
  k: "keyword", i: "ident", o: "punct", _: "space" };
function summarizeC(run, q, raw) {
  const v = raw.v, E = Math.round(v.E), src = q - E + 1, p = raw.pred;
  if (q < run.prompt_len - 1) return { q, phase: "prompt", op: null, pred: p, explain: "Part of the C source you gave: input, not predicted.", marks: {} };
  const fired = Object.keys(v).filter((k) => k.startsWith("r_") && v[k] > 0.5);
  const doc = (k) => dimDoc(k)?.doc || k;
  const ch = run.chars[src] ?? "";
  const prev = run.chars[q];
  const was = q === E ? "the start" : `state '${prev}'`;
  const explain = p === "\n" ? "The source is used up, so the model writes the final newline (rule end)."
    : p === LEND ? "The source is used up, so the model writes '\\x04', the end of the letters: the parse starts after it (rule end)."
    : `Source byte ${src} ${JSON.stringify(ch)}, after ${was}: ${fired.length === 1 ? doc(fired[0]) : `rules ${fired.join(", ")}`}`;
  return { q, phase: C_CLASS[p] || "punct", op: null, pred: p, explain, marks: { E, src }, rule: fired[0] };
}

// Parser: which field of which record is being written, and the one rule that fired. A record is
// 20 bytes at fixed offsets (handformer/cparse/circuit.py); Lsep is the '|' that ends the previous one.
const REC = 20;
const REC_FIELD = ["the action", "the symbol", "a space", "the depth", "the depth", "the depth", "a space", "the pointer",
  "the pointer", "the pointer", "the pointer", "a space", "the item", "the item", "the item", "a space",
  "the return item", "the return item", "the return item", "the record's end '|'"];
const REC_ACT = { O: "open", T: "token", C: "close", "!": "error" };
const LEND = "\x04"; // the C parser writes it after the letters (handformer/cparse/circuit.py)
// A token starts where its letter can start one and does not continue the token before it
// (handformer/cparse/tokens.py starts(), the same reading of the letters; display only).
const PUNCT_LEN = { o: 1, q: 2, r: 3, v: 4 };
function tokenStarts(letters) {
  return [...letters].map((cur, i) => {
    const prev = i ? letters[i - 1] : "_";
    if (cur in PUNCT_LEN) return true;
    if ("ikn".includes(cur)) return prev !== cur;
    if (cur === "P") return prev !== "P";
    if (cur === "s") return !"sStGHP".includes(prev);
    if (cur === "a") return !"aAbghP".includes(prev);
    return false;
  });
}
/** The source text of the token starting at source position ``at`` (1-based), from the letters. */
function tokenText(chars, E, at) {
  const letters = chars.slice(E + 1, chars.indexOf(LEND, E + 1) < 0 ? chars.length : chars.indexOf(LEND, E + 1)).join("");
  const st = tokenStarts(letters), i = at - 1, L = letters[i];
  if (L === undefined || !st[i]) return chars[at] ?? "";
  if (L in PUNCT_LEN) return chars.slice(at, at + PUNCT_LEN[L]).join("");
  let j = i + 1;
  while (j < letters.length && !st[j] && "ikn".includes(L) && letters[j] === L) j++;
  if ("sPa".includes(L)) while (j < letters.length && !st[j] && !"_BLp".includes(letters[j])) j++;
  return chars.slice(at, j + 1).join("");
}
function summarizeParse(run, q, raw) {
  const v = raw.v, r = Math.round, p = raw.pred;
  if (q < run.prompt_len - 1) return { q, phase: "prompt", op: null, pred: p, explain: "Part of the source you gave: input, not predicted.", marks: {} };
  if (v.after_M < 0.5) { // first the letters: the highlighter's own lexer, inside this model
    const lex = summarizeC(run, q, raw);
    return { ...lex, explain: `Lexing (the letters come first, then the records). ${lex.explain}` };
  }
  const E = r(v.E), M = r(v.M), L = r(v.Lsep), o = q - L, n = (L - M) / REC + 1, la = r(v.la_pos);
  const fired = Object.keys(v).filter((k) => k.startsWith("r_") && v[k] > 0.5);
  const doc = (k) => (dimDoc(k)?.doc || k).replace(/\.$/, "");
  const act = o === 0 ? p : run.chars[L + 1];
  const halt = o === 0 && p === "\n";
  const phase = halt ? "halt" : REC_ACT[act] || "token";
  const rule = fired.length === 1 ? doc(fired[0]) : `rules ${fired.join(", ") || "(none)"}`;
  const tok = la < E ? JSON.stringify(tokenText(run.chars, E, la)) : la === E ? "the end ($)" : "nothing";
  const marks = { E, M, Lsep: L, la_pos: la };
  let explain;
  if (halt) explain = `${rule}.`;
  else {
    explain = `Record ${n}, offset ${o}: writing ${REC_FIELD[o]} (${JSON.stringify(p)}). ${rule}. Lookahead: ${tok}` +
      (la < E ? ` at ${la}.` : ".");
    if (v.isclose > 0.5) {
      marks.mpos = r(v.mpos);
      explain += ` The stack head found the open record ending at ${r(v.mpos)} (the latest open at this depth): its return item is the parent's next item.`;
    }
  }
  return { q, phase, op: null, pred: p, explain, marks, rule: fired[0] };
}

const summarizeStep = (run, q, raw) => (state.model === "life" ? summarizeLife(run, q, raw)
  : state.model === "c" ? summarizeC(run, q, raw) : state.model === "parse" ? summarizeParse(run, q, raw)
  : summarize(run, q, raw));
const phaseClass = (s) => (s.phase === "op" ? `ph-${s.op || "prompt"}` : `ph-${s.phase}`);

// ------------------------------------------------------------------ runs

const GRAPH_CIRCUITS = {
  calc: [["all", "all circuits together"], ["frame", "frame locator"], ["parser", "parser / rewriter"],
         ["add", "add"], ["sub", "sub"], ["mul", "mul"], ["div", "div"], ["pow", "pow"]],
  life: [["life", "game of life"]],
  c: [["rules", "the rules"], ["lookups", "lookups"], ["keywords", "keywords"], ["landmarks", "landmarks"]],
  parse: [["rules", "the rules"], ["stack", "the stack search"], ["lookahead", "the lookahead"],
          ["record", "record fields"], ["landmarks", "landmarks"]],
};

/** Show the parts of the page that belong to the current model (legend, guide, graph menu, input hint). */
function applyModelChrome() {
  const m = state.model, info = state.info;
  $("model-sel").value = m;
  document.querySelectorAll(".legend[data-model]").forEach((el) => { el.hidden = el.dataset.model !== m; });
  $("guide").hidden = m !== "calc"; // the guide's examples are the calculator's
  const keep = $("g-circuit").value;
  $("g-circuit").replaceChildren(...GRAPH_CIRCUITS[m].map(([v, t]) => new Option(t, v)));
  if (GRAPH_CIRCUITS[m].some(([v]) => v === keep)) $("g-circuit").value = keep;
  $("expr-input").hidden = !!info.multiline;
  $("code-input").hidden = !info.multiline;
  inputBox().placeholder = info.input_hint;
  inputBox().dataset.tip = `Input for the ${info.title}: ${info.input_hint}.`;
  document.title = `handformer · ${info.title}`;
}

/** Switch the page to another model and run its default (or given) input. */
async function switchModel(key, input = null) {
  if (key === state.model && state.info) return;
  log("switching model", state.model, "->", key);
  stopPlay();
  Object.assign(state, { model: key, run: null, head: null, openDim: null, base: null, q: 0 });
  state.pos.clear(); state.maps.clear(); state.baseline.clear();
  state.ko = { heads: [], units: [], dims: [] };
  await loadInfo();
  applyModelChrome();
  setInput(input ?? state.info.default_input);
  await runExpression(getInput());
}

/** Fill the model drop-down from models.yaml (server /api/models; static site data/models.json); returns the keys. */
async function loadModels() {
  const url = window.HANDFORMER_STATIC !== undefined ? `${window.HANDFORMER_STATIC}models.json` : "/api/models";
  const r = await fetch(url);
  if (!r.ok) throw new Error(`could not load the model list from ${url} (${r.status})`);
  const { models } = await r.json();
  $("model-sel").replaceChildren(...models.map(([key, title]) => new Option(title, key)));
  log("models offered:", models.map(([key]) => key).join(", "));
  return models.map(([key]) => key);
}

async function loadInfo() {
  state.info = await api("/api/info");
  log("model", state.info.model_id, state.info.d_model, "dims,", state.info.layers.length, "layers");
  $("mlp-layer").replaceChildren(...state.info.layers
    .filter((l) => l.units.length).map((l) => new Option(`${l.index + 1}`, l.index)));
  $("w-layer").replaceChildren(...state.info.layers.map((l) => new Option(`${l.index + 1}`, l.index)));
  $("a-layer").replaceChildren(...state.info.layers.filter((l) => l.heads.length)
    .map((l) => new Option(`${l.index + 1}`, l.index)));
  $("w-layer").dispatchEvent(new Event("input"));
  renderGraph();
}

/** The final line, made safe to show: unprintable bytes as '·', capped, and flagged if the model never stopped. */
function answerOf(run) {
  const text = run.chars.join("");
  // Life: the last generation's grid; the calculator: the line after the last step
  if (state.model === "c") {
    const written = run.chars.length - run.prompt_len - (run.finished ? 1 : 0), source = run.prompt_len - 2;
    return `${written} of ${source} source bytes classified` + (run.finished ? "" : " (did not finish)");
  }
  if (state.model === "parse") {
    const recs = parseRecords(run.chars, run.chars.length - 1);
    const last = recs[recs.length - 1];
    const verdict = !run.finished ? "did not finish" : last?.act === "!" ? `syntax error at ${last.ptr}`
      : last?.act === "C" && last.depth === 0 ? "parsed" : "stopped early";
    return `${recs.length} records · ${verdict}`;
  }
  const raw = state.model === "life" ? (text.trim().split("=").filter(Boolean).pop() || "").split(":").pop()
    : text.split("]").pop().trim();
  let a = [...raw].map((c) => (c >= " " && c <= "~" ? c : "·")).join("");
  if (a.length > 40) a = `${a.slice(0, 40)}… (${raw.length} bytes)`;
  return run.finished ? a : `${a || "—"} (did not finish)`;
}
const koCount = () => state.ko.heads.length + state.ko.units.length + state.ko.dims.length;
const koHas = (kind, item) => state.ko[kind].some((x) => JSON.stringify(x) === JSON.stringify(item));

/**
 * Run ``fn`` (a re-run) without moving what you are looking at: the element nearest the middle
 * of the screen keeps its on-screen position, and the panels' own scroll positions are restored.
 * Without this, the knockout bar appearing and the lists re-rendering shift the page.
 */
function holdView() {
  const panels = [...document.querySelectorAll(".panel")];
  const mid = innerHeight / 2;
  const anchor = panels.map((p) => [p, p.getBoundingClientRect()]).filter(([, r]) => r.bottom > 0 && r.top < innerHeight)
    .sort((a, b) => Math.abs((a[1].top + a[1].bottom) / 2 - mid) - Math.abs((b[1].top + b[1].bottom) / 2 - mid))[0]?.[0];
  const top0 = anchor?.getBoundingClientRect().top;
  const inner = [...document.querySelectorAll(".scroll, .graph-wrap, .map-wrap")].map((e) => [e.id || e.className, e.scrollTop, e.scrollLeft]);
  return () => {
    if (anchor) window.scrollBy(0, anchor.getBoundingClientRect().top - top0);
    for (const [key, t, l] of inner) {
      if (key === "tokens") continue; // the strip follows the current byte itself
      const e = document.getElementById(key) || document.getElementsByClassName(key.split(" ")[0])[0];
      if (e) { e.scrollTop = t; e.scrollLeft = l; }
    }
  };
}

async function keepView(fn) {
  const restore = holdView();
  try { await fn(restore); } finally { restore(); }
}

/** Switch a head / unit / dim off or back on, then re-run the expression at the same step. */
async function toggleKnockout(kind, item) {
  state.ko[kind] = koHas(kind, item) ? state.ko[kind].filter((x) => JSON.stringify(x) !== JSON.stringify(item))
    : [...state.ko[kind], item];
  log("knockout", kind, JSON.stringify(item), "now", koCount(), "parts off");
  const keep = state.q;
  if (!state.run) return;
  await keepView(async (restore) => {
    state.running = true;
    renderKnockouts(); // show the new set at once; the answer follows when the re-run returns
    restore();         // the bar may have grown: hold the view still right away
    await runExpression(state.run.expression, keep);
  });
}

function renderKnockouts() {
  const bar = $("ko-bar");
  if (!koCount() || !state.run) { bar.hidden = true; bar.replaceChildren(); return; }
  bar.hidden = false;
  bar.replaceChildren();
  const add = (text, kind, item, tip) => {
    const b = document.createElement("button");
    b.className = "ko-chip";
    b.textContent = `${text} ✕`;
    b.dataset.tip = `${tip}\n(click to switch it back on)`;
    b.addEventListener("click", () => toggleKnockout(kind, item));
    bar.append(b);
  };
  const lbl = document.createElement("span");
  lbl.className = "ko-label";
  lbl.textContent = "Knocked out:";
  bar.append(lbl);
  // tooltip = the part's own description, then what the knockout does to it
  for (const [l, h] of state.ko.heads) add(`L${l + 1} ${headSpec(l, h)?.name}`, "heads", [l, h], `${headSpec(l, h)?.doc || ""}\n\n${KO_NOTE.heads}`);
  for (const [l, u] of state.ko.units) add(`L${l + 1} unit #${u}`, "units", [l, u], `${state.info.layers[l].units[u]?.doc || ""}\n\n${KO_NOTE.units}`);
  for (const d of state.ko.dims) add(d, "dims", d, `${dimDoc(d)?.doc || ""}\n\n${KO_NOTE.dims}`);
  const got = answerOf(state.run), want = state.baseline.get(state.run.expression);
  const res = document.createElement("span");
  res.className = "ko-result" + (!state.running && want !== undefined && got !== want ? " changed" : "");
  res.textContent = state.running ? "re-running…" : want === undefined ? `answer now ${got}` : got === want
    ? `answer unchanged: ${got} (this part isn't needed here)` : `answer now ${got} · normally ${want}`;
  const clear = document.createElement("button");
  clear.textContent = "restore all";
  clear.addEventListener("click", async () => {
    state.ko = { heads: [], units: [], dims: [] };
    log("knockout: restore all");
    await keepView(() => runExpression(state.run.expression, state.q));
  });
  bar.append(res, clear);
}

// What a knockout does to each kind of part, appended under the part's own description.
const KO_NOTE = {
  heads: "⊘ Knocked out in this run: it still attends, but its output is zeroed, so it writes nothing.",
  units: "⊘ Knocked out in this run: it still fires, but its output is zeroed, so it adds nothing.",
  dims: "⊘ Knocked out in this run: every write into it is zeroed, so it stays 0 at every position.",
};
const withKo = (kind, item, tip) => (koHas(kind, item) ? `${tip}\n\n${KO_NOTE[kind]}` : tip);

/** A small "knock out / restore" button for the detail panels. */
function koButton(kind, item) {
  const b = document.createElement("button");
  b.className = "ko-btn";
  const on = koHas(kind, item);
  b.textContent = on ? "restore" : "knock out";
  b.dataset.tip = on ? "Switch this part back on and re-run." :
    "Switch this part off (its writes are zeroed in the weights) and re-run, to see what it is needed for.";
  b.addEventListener("click", (e) => { e.stopPropagation(); toggleKnockout(kind, item); });
  return b;
}

async function runExpression(expression, keepStep = null) {
  stopPlay();
  setStatus("running…");
  const seq = ++state.runSeq;
  state.running = true;
  try {
    const ko = koCount() ? JSON.parse(JSON.stringify(state.ko)) : null; // the set this request was made with
    const body = ko ? { expression, knockout: ko } : { expression };
    const run = await api("/api/run", body);
    if (ko && !state.baseline.has(run.expression)) {
      const normal = await api("/api/run", { expression }); // also the difference view's comparison run
      state.base = normal;
      state.baseline.set(run.expression, answerOf(normal));
    } else if (!ko) {
      state.baseline.set(run.expression, answerOf(run));
      state.base = run; // the normal run itself
    }
    if (seq !== state.runSeq) { log("dropping a stale run response", run.run_id); return; }
    state.running = false;
    run.steps = run.steps.map((raw, q) => summarizeStep(run, q, raw));
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
    const answer = answerOf(run);
    setStatus(`${run.chars.length - run.prompt_len} bytes written by the model · answer ${answer}` +
      (koCount() ? ` · ${koCount()} part(s) knocked out` : ""));
    log("run", run.run_id, run.expression, run.chars.length, "bytes");
    renderKnockouts();
    const last = run.chars.length - 2;
    await goto(keepStep === null ? run.prompt_len - 1 : Math.max(run.prompt_len - 1, Math.min(keepStep, last)));
  } catch (e) {
    if (seq !== state.runSeq) return; // a newer request owns the page
    state.running = false;
    renderKnockouts();
    setStatus(window.HANDFORMER_STATIC === undefined ? `${e.message} (is the server running?)` : e.message, true);
    logErr("run failed:", e.message);
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
  const restoreView = holdView(); // stepping must not shift the panel you are watching
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
    if (gen === state.gen) restoreView();
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
  const labels = MARK_LABELS_BY[state.model] || {};
  for (const [dim, label] of Object.entries(labels)) {
    const at = step.marks[dim];
    if (step.phase === "prompt" || step.phase === "open") break;
    if (at >= 0 && at < run.chars.length) (marks[at] ||= []).push({ dim, label });
  }
  // Life: the 3x3 neighbourhood the cell's heads read, one generation back
  for (const at of step.nbs || []) {
    if (at >= 0 && at < run.chars.length) (marks[at] ||= []).push({ dim: "nb", label: labels.nb || "nb" });
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
      : "Part of the prompt you typed.") + "\n(click to step to this position)";
    if (marks[i]) {  // the pointers drawn above this byte, spelled out (orange, like the tags)
      el.dataset.tipPtr = "Pointers here at this step:\n" + marks[i].map(({ dim, label }) =>
        `▲ ${label}${label !== dim ? ` (${dim})` : ""}: ${dimDoc(dim)?.doc || PTR_DOCS[dim] || ""}`).join("\n");
    }
    if (marks[i]) {
      const tags = document.createElement("span");
      tags.className = "tags";
      marks[i].forEach(({ label }) => {
        const t = document.createElement("span");
        t.className = "tag";
        t.textContent = label;
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
  // Keep the next byte visible inside the strip only: scrollIntoView would also scroll the
  // window, yanking the page up to the strip while you look at a heatmap further down.
  const out = strip.querySelector(".chip.output");
  if (out) {
    const r = out.getBoundingClientRect(), rs = strip.getBoundingClientRect();
    const top = r.top - rs.top + strip.scrollTop, bottom = top + r.height;
    if (top < strip.scrollTop) strip.scrollTop = top - 18;
    else if (bottom > strip.scrollTop + strip.clientHeight) strip.scrollTop = bottom - strip.clientHeight + 18;
  }
}

// ------------------------------------------------------------------ Life board
// The grid one generation back and the one being written, from the written text (display only);
// the outlined cells are the model's own pointers: the cell (q+1-L) and its neighbours.
function lifeBlocks(chars) {
  const blocks = [];
  let start = 1;
  for (let i = 1; i < chars.length; i++) {
    if (chars[i] === "=" || i === chars.length - 1) {
      const colon = chars.indexOf(":", start);
      if (colon >= start && colon < i) blocks.push({ start, colon, end: i }); // the final '\n' is not a block
      start = i + 1;
    }
  }
  return blocks;
}
function drawBoard(el, chars, block, upto, marks) {
  el.replaceChildren();
  if (!block) return;
  let row = document.createElement("div");
  row.className = "board-line";
  for (let p = block.colon + 1; p < block.end || (p === block.end && chars[p] !== "="); p++) {
    const c = chars[p];
    if (c === "/") { el.append(row); row = document.createElement("div"); row.className = "board-line"; continue; }
    if (c !== "#" && c !== ".") continue;
    const sq = document.createElement("span");
    sq.className = "bcell" + (p > upto ? " later" : c === "#" ? " alive" : "") + (marks[p] ? ` ${marks[p]}` : "");
    sq.dataset.tip = `position ${p}: '${c}'` + (marks[p] === "is-cell" ? "\nthe cell being written, one generation back" :
      marks[p] === "is-nb" ? "\na neighbour its heads read" : marks[p] === "is-next" ? "\nwritten at this step" : "");
    sq.addEventListener("click", () => goto(p - 1));
    row.append(sq);
  }
  el.append(row);
}
function renderBoard() {
  const panel = $("p-board");
  panel.hidden = state.model !== "life";
  if (panel.hidden || !state.run) return;
  const run = state.run, q = state.q, s = run.steps[q], blocks = lifeBlocks(run.chars);
  const i = blocks.findIndex((b) => q + 1 >= b.start && q + 1 <= b.end);
  const cur = blocks[i], prev = blocks[i - 1];
  const prevMarks = {}, curMarks = { [q + 1]: "is-next" };
  for (const p of s.nbs || []) prevMarks[p] = "is-nb";
  if (s.marks.cell !== undefined) prevMarks[s.marks.cell] = "is-cell";
  drawBoard($("board-prev"), run.chars, prev, Infinity, prevMarks);
  drawBoard($("board-cur"), run.chars, cur, q + 1, curMarks);
  $("board-prev-title").textContent = prev ? `one generation back (count ${run.chars[prev.start]})` : "the prompt is being read";
  $("board-cur-title").textContent = cur ? `being written (count ${run.chars[cur.start]})` : "";
  $("board-caption").textContent = s.phase === "cell" ? `· ${s.explain.split(":")[0]}` : "";
}

// ------------------------------------------------------------------ C code view
// The source coloured by the letters written so far (display only); the outlined byte is the
// one being classified at this step.
function renderCode() {
  const panel = $("p-code");
  panel.hidden = state.model !== "c";
  if (panel.hidden || !state.run) return;
  const run = state.run, q = state.q, E = run.prompt_len - 1, view = $("code-view");
  view.replaceChildren();
  for (let s = 1; s < E; s++) {
    const out = E + s; // the letter for source byte s is written at position E + s
    const sp = document.createElement("span");
    sp.textContent = run.chars[s];
    const letter = out <= q ? run.chars[out] : null;
    sp.className = letter ? `t-${C_CLASS[letter] || "punct"}` : "later";
    if (s === q - E + 1) sp.classList.add("now");
    sp.dataset.tip = `source byte ${s} ${JSON.stringify(run.chars[s])}` + (letter ? ` → state '${letter}' (${C_CLASS[letter]})` : " (not classified yet)") +
      "\n(click to step to it)";
    sp.addEventListener("click", () => goto(E + s - 1));
    view.append(sp);
  }
  const st = run.steps[q];
  $("code-caption").textContent = st.rule ? `· rule ${st.rule.slice(2)}` : "";
  // keep the current byte in view by scrolling this box only (scrollIntoView would move the page)
  const now = view.querySelector(".now");
  if (now) {
    const r = now.getBoundingClientRect(), rv = view.getBoundingClientRect();
    if (r.top < rv.top || r.bottom > rv.bottom) view.scrollTop += r.top - rv.top - rv.height / 2;
    if (r.left < rv.left || r.right > rv.right) view.scrollLeft += r.left - rv.left - rv.width / 2;
  }
}

// ------------------------------------------------------------------ parse tree view
/** The complete records in chars whose '|' is at or before ``upto`` (they start after the letters' end). */
function parseRecords(chars, upto) {
  const out = [];
  const M = chars.indexOf(LEND);
  if (M < 0) return out;
  for (let s = M + 1; s + REC - 1 <= upto && chars[s + REC - 1] === "|"; s += REC) {
    const t = chars.slice(s, s + REC).join("");
    out.push({ start: s, act: t[0], sym: t[1], depth: +t.slice(3, 6), ptr: +t.slice(7, 11), item: +t.slice(12, 15),
      ret: t.slice(16, 19) === "..." ? null : +t.slice(16, 19) });
  }
  return out;
}
// The tree spelled by the records written so far (display only: the model's own text). The node
// being opened or closed now is outlined; the source shows consumed tokens and the lookahead.
function renderTree() {
  const panel = $("p-tree");
  panel.hidden = state.model !== "parse";
  if (panel.hidden || !state.run) return;
  const run = state.run, q = state.q, E = run.prompt_len - 1, s = run.steps[q];
  const names = state.info.node_names || {};
  const recs = parseRecords(run.chars, q + 1);
  const last = recs[recs.length - 1];
  // source: consumed (up to the last record's pointer), the lookahead outlined
  const src = $("tree-src");
  src.replaceChildren();
  const done = last && last.act !== "!" ? last.ptr : 0;
  for (let i = 1; i < E; i++) {
    const sp = document.createElement("span");
    sp.textContent = run.chars[i];
    sp.className = i <= done ? "t-done" : "later";
    if (i === s.marks?.la_pos) sp.classList.add("now");
    if (last?.act === "!" && i === last.ptr) sp.classList.add("t-bad");
    sp.dataset.tip = `source byte ${i} ${JSON.stringify(run.chars[i])}` + (i <= done ? " (consumed)" : "") +
      (i === s.marks?.la_pos ? "\nthe lookahead: the first token after the pointer" : "");
    src.append(sp);
  }
  // tree
  const view = $("tree-view");
  view.replaceChildren();
  const stack = [view];
  const nowStart = s.marks?.Lsep !== undefined ? s.marks.Lsep + 1 : -1; // the record being written starts here
  for (const rc of recs) {
    const parent = stack[stack.length - 1];
    if (rc.act === "O") {
      const node = document.createElement("div");
      node.className = "tnode";
      const head = document.createElement("span");
      head.className = "thead";
      head.textContent = `${rc.sym} ${names[rc.sym] || ""}`;
      head.dataset.tip = `node ${rc.sym} (${names[rc.sym] || "?"}), opened by the record at ${rc.start} (depth ${rc.depth}, ` +
        `continues at item ${rc.ret} when it closes)\n(click to step to that record)`;
      head.addEventListener("click", () => goto(rc.start - 1));
      node.append(head);
      parent.append(node);
      stack.push(node);
    } else if (rc.act === "C") {
      const node = stack.pop();
      if (node && node !== view) node.classList.add("closed");
    } else if (rc.act === "T") {
      const t = document.createElement("span");
      t.className = "ttok";
      t.textContent = rc.sym === "$" ? "$" : tokenText(run.chars, E, rc.ptr);
      t.dataset.tip = `token ${JSON.stringify(t.textContent)} (code ${JSON.stringify(rc.sym)}) at source byte ${rc.ptr}, ` +
        `consumed by the record at ${rc.start}`;
      t.addEventListener("click", () => goto(rc.start - 1));
      parent.append(t);
    } else {
      const t = document.createElement("span");
      t.className = "ttok t-bad";
      t.textContent = `syntax error at ${rc.ptr}`;
      parent.append(t);
    }
  }
  // the innermost open node is where the next record acts
  const cur = stack[stack.length - 1];
  if (cur !== view) cur.classList.add("open-now");
  const wrote = run.chars.slice(nowStart, q + 2).join("");
  $("tree-caption").textContent = s.phase === "prompt" ? "" : `· record so far: ${JSON.stringify(wrote)}` + (s.rule ? ` · rule ${s.rule.slice(2)}` : "");
  const now = view.querySelector(".open-now > .thead") || view.lastElementChild;
  if (now) {
    const r = now.getBoundingClientRect(), rv = view.getBoundingClientRect();
    if (r.top < rv.top || r.bottom > rv.bottom) view.scrollTop += r.top - rv.top - rv.height / 2;
  }
}

function renderStep() {
  const run = state.run;
  const s = run.steps[state.q];
  const n = run.chars.length - 1 - (run.prompt_len - 1);
  $("step-label").textContent = `step ${state.q - run.prompt_len + 2} / ${n} · query pos ${state.q} ${JSON.stringify(run.chars[state.q])} → writes ${JSON.stringify(run.chars[state.q + 1])}`;
  $("explain").textContent = s.explain;
  renderBoard();
  renderCode();
  renderTree();
  $("explain").style.borderLeftColor = getComputedStyle(document.documentElement)
    .getPropertyValue(`--${phaseClass(s)}`) || "var(--accent)";
  const m = $("marks");
  m.replaceChildren();
  for (const [dim, v] of Object.entries(s.marks)) {
    const el = document.createElement("span");
    const ch = v >= 0 && v < run.chars.length ? showChar(run.chars[v]) : "";
    el.innerHTML = `${dim} <b>${v}</b> <span class="muted">${ch}</span>`;
    el.dataset.tip = `${dim} = ${v}${ch ? ` (the "${ch}" at position ${v})` : ""}\n${dimDoc(dim)?.doc || PTR_DOCS[dim] || ""}`;
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
  let lastLayer = -1, hidden = 0;
  for (const h of p.heads) {
    const spec = headSpec(h.layer, h.head);
    if (!spec) { logErr("head", h.layer, h.head, "missing from the model description; reload the page"); continue; }
    if (filter && !spec.name.toLowerCase().includes(filter)) continue;
    if (!focused(spec.circuit)) { hidden++; continue; }
    if (h.layer !== lastLayer) {
      const lh = document.createElement("div");
      lh.className = "layer-head";
      lh.textContent = `layer ${h.layer + 1}`;
      el.appendChild(lh);
      lastLayer = h.layer;
    }
    const [k, w] = h.top[0];
    const row = document.createElement("div");
    row.className = "row head-row" + (koHas("heads", [h.layer, h.head]) ? " ko" : "");
    if (state.head && state.head[0] === h.layer && state.head[1] === h.head) row.classList.add("sel");
    const wrote = Object.entries(h.wrote).map(([d, v]) => `${d}=${fmt(v)}`).join(" ");
    row.dataset.tip = `layer ${h.layer + 1} · ${spec.name}\n${spec.doc}\n\nAt this step it reads position ${k} ("${showChar(state.run.chars[k])}") with ${(w * 100).toFixed(1)}% of its attention.` +
      (wrote ? `\nIt adds: ${wrote}` : "") + srcText(spec.src) + "\n(click for its formulas, source line and full attention pattern)";
    row.dataset.tip = withKo("heads", [h.layer, h.head], row.dataset.tip);
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
  if (hidden) {
    const note = document.createElement("div");
    note.className = "muted small hidden-note";
    note.textContent = `${hidden} more heads belong to calculators that are switched off at this step: they still run, ` +
      "but their output is gated to zero. Untick “focus active circuit” to list them.";
    el.appendChild(note);
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
  el.textContent = `L${layer + 1} ${spec.name}\nscore = ${score}\nvalue: ${value}\nreads: ${top}\nwrote: ${wrote}\nsource: `;
  el.append(srcLinks(spec.src), document.createElement("br"), koButton("heads", [layer, head]));
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
    row.className = "row dim-row" + (state.openDim === d.name ? " sel" : "") + (koHas("dims", d.name) ? " ko" : "");
    row.dataset.tip = `${d.name} = ${fmt(final)}\n${d.doc}\n\nwritten by: ${d.written_by}\nread by: ${d.read_by}${srcText(d.src)}\n(click to see how it changes layer by layer)`;
    row.dataset.tip = withKo("dims", d.name, row.dataset.tip);
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
        .filter((_, s) => s === 0 || Math.abs(vals[s] - vals[s - 1]) > 1e-9).join("\n") + "\nwritten at: ";
      t.append(srcLinks(d.src));
      if (!["pos", "pos2"].includes(d.name)) t.append(" ", koButton("dims", d.name));
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
    row.className = "row unit-row" + (koHas("units", [u.layer, u.unit]) ? " ko" : "");
    row.dataset.tip = `layer ${u.layer + 1} · unit #${u.unit}\n${spec.doc}\n\nReLU(${spec.read})\n= ${fmt(u.pre)} at this step → adds to ${spec.write}${srcText(spec.src)}`;
    row.dataset.tip = withKo("units", [u.layer, u.unit], row.dataset.tip);
    row.innerHTML = `<span class="muted">#${u.unit}</span><span class="num">${fmt(u.pre)}</span><span class="f">ReLU(<b>${spec.read}</b>) → ${spec.write} </span>`;
    row.querySelector(".f").append(srcLinks(spec.src), koButton("units", [u.layer, u.unit]));
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

/** Explain every residual dim named in a linear form such as "2·f_R - P + 1". */
function formDims(form) {
  const names = [...new Set(form.match(/[A-Za-z_][\w.]*/g) || [])];
  return names.map(dimDoc).filter(Boolean).map((d) => `${d.name}: ${d.doc}`).join("\n");
}
const dimTip = (name) => { const d = dimDoc(name); return d ? `${d.name}\n${d.doc}${srcText(d.src)}` : name; };
const unitTip = (u, i) => `unit #${i} · ${u.circuit}\n${u.doc}\n\nReLU(${u.read}) → ${u.write}\n\n${formDims(u.write)}${srcText(u.src)}`;

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
    see: ["For one head: how much each character (<b>row</b>) looks at each earlier character (<b>column</b>). The characters are written along both edges when there is room.",
          "Only the lower-left half can light up: a character can't look ahead."],
    read: ["Each row is almost <b>one bright dot</b>: the head reads exactly one position.",
           "A <b>diagonal line</b> means 'read the byte at a fixed offset' (for example the previous byte).",
           "A <b>vertical stripe</b> means many characters read the same landmark.",
           "A <b>staircase</b> means a pointer jumping to newer landmarks as they appear."],
    try: ["Pick a head with the <b>layer / head</b> menus (or click one in the Attention heads panel), for example <b>latest F</b> or <b>first ) after p</b>.",
          "Follow the <b>blue line</b>: hover a row name to read 'this character reads that one'.",
          "Lower <b>gamma</b> to brighten weak attention, raise it to see only the strongest.",
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
          "Hover a cell to read the unit's hand-written formula and value.", "Hover a <b>row name</b> (left edge) for what each abbreviation in it means, e.g. f_R or f_Lg."],
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

/** A head worth showing at this step: the first one in a focused circuit that wrote something. */
function defaultHead(p) {
  const h = p.heads.find((x) => Object.keys(x.wrote).length && focused(headSpec(x.layer, x.head).circuit))
    || p.heads.find((x) => Object.keys(x.wrote).length);
  return h ? [h.layer, h.head] : null;
}

function syncHeadPicker() {
  if (!state.head) return;
  const [layer, head] = state.head;
  if ($("a-layer").value !== String(layer) || $("a-head").options.length !== state.info.layers[layer].heads.length) {
    $("a-layer").value = layer;
    $("a-head").replaceChildren(...state.info.layers[layer].heads.map((h, i) => new Option(`${i}: ${h.name}`, i)));
  }
  $("a-head").value = head;
}

// ------------------------------------------------------------------ knockout difference view
// With parts knocked out, a heatmap can show knocked-out minus normal: the knockout's causal
// footprint. Both runs share their first bytes (the prompt and whatever the knockout left
// unchanged); after the first byte where their text differs, the inputs differ too, which the
// map marks with a dashed line.

/** The normal (nothing knocked out) run of the current expression, for comparison. */
async function baseRun() {
  const e = state.run.expression;
  if (!state.base || state.base.expression !== e) {
    state.base = await api("/api/run", { expression: e });
    log("difference view: normal run", state.base.run_id, "for", e);
  }
  return state.base;
}

/** mapData for the normal run, re-running it once if the server dropped it. */
async function baseMap(path, body) {
  try {
    return await mapData(path, { ...body, run_id: (await baseRun()).run_id });
  } catch (e) {
    if (!/unknown or evicted run/.test(e.message)) throw e;
    state.base = null;
    return mapData(path, { ...body, run_id: (await baseRun()).run_id });
  }
}

const diffOn = () => koCount() > 0 && $("diff").checked && state.map !== "weights";

async function renderDiffMap(p) {
  const canvas = $("map"), cap = $("map-caption"), run = state.run, q = state.q;
  const base = await baseRun();
  const nK = run.chars.length, nB = base.chars.length, nc = Math.min(nK, nB);
  let split = 0;
  while (split < nc && run.chars[split] === base.chars[split]) split++;
  const where = split >= nc ? "Both runs write the same text, so every change is the knockout's own effect."
    : `Both runs write the same text up to position ${split - 1}; from position ${split} (dashed line) they write ` +
      "different bytes, so later changes also reflect different input.";
  // f16 transport (server) quantises each value to ~1e-3 of its row's scale: ignore changes below that.
  const tolOf = (meta) => (meta.dtype === "float16" ? 2e-3 : 1e-6);
  const pairs = async (path, body) => {
    const [K, B] = await Promise.all([mapData(path, { ...body, run_id: run.run_id }), baseMap(path, body)]);
    return { K, B };
  };
  const drawRows = (title, names, valK, valB, nRows, docs, clickRow) => {
    // valK(r, c) / valB(r, c): raw values; keep rows with any real change, scale each by its biggest change
    const rows = [];
    for (let r = 0; r < nRows; r++) {
      let mx = 0, ref = 0;
      for (let c = 0; c < nc; c++) {
        mx = Math.max(mx, Math.abs(valK(r, c) - valB(r, c)));
        ref = Math.max(ref, Math.abs(valB(r, c)), Math.abs(valK(r, c)));
      }
      if (mx > title.tol * Math.max(1, ref)) rows.push({ r, mx });
    }
    cap.textContent = `${title.text}: knocked-out run minus normal run (orange = higher, blue = lower), each row ` +
      `scaled by its biggest change. ${rows.length} of ${nRows} rows change. ${where} ` +
      "Private values of switched-off calculators (e.g. div.*) can move too: they compute at every byte, " +
      "but their outputs are gated to zero, so they cannot change the answer." +
      (rows.length ? "" : " Nothing changes in this view: try a later layer or step, or the residual stream map.");
    drawHeatmap(canvas, {
      key: `diff:${title.key}:${run.run_id}:${base.run_id}`,
      rows: rows.length, cols: nc,
      get: (i, c) => (valK(rows[i].r, c) - valB(rows[i].r, c)) / rows[i].mx, norm: (v) => (v + 1) / 2, lut: DIV,
      rowLabels: rows.map((x) => names(x.r)), markCol: split < nc ? split : undefined,
      valueText: (i, c) => { const r = rows[i].r;
        return `${names(r)} at position ${c} ${charAt(c)}
normal ${fmt(valB(r, c))} → knocked out ${fmt(valK(r, c))}` +
          ` (change ${fmt(valK(r, c) - valB(r, c))})
${docs(r)}`; },
      rowTip: (i) => docs(rows[i].r),
      onClick: (i, c) => clickRow(rows[i].r, c),
    }, { q, noDim: true });
  };
  if (state.map === "residual") {
    const snap = Number($("snapshot").value);
    const { K, B } = await pairs("/api/residual", { snapshot: snap });
    const dims = state.info.dims;
    drawRows({ text: `residual stream after ${K.meta.label}`, key: `res${snap}`, tol: tolOf(K.meta) },
      (r) => dims[r].name, (r, c) => K.data[r * nK + c] * K.meta.scale[r], (r, c) => B.data[r * nB + c] * B.meta.scale[r],
      dims.length, (r) => dimTip(dims[r].name), (r, c) => { state.openDim = dims[r].name; goto(c); });
  } else if (state.map === "mlp") {
    const layer = Number($("mlp-layer").value);
    const { K, B } = await pairs("/api/mlp", { layer });
    const units = state.info.layers[layer].units;
    drawRows({ text: `layer ${layer + 1} MLP activations`, key: `mlp${layer}`, tol: tolOf(K.meta) },
      (r) => `#${r} ${units[r].write}`, (r, c) => K.data[r * nK + c] * K.meta.scale[r], (r, c) => B.data[r * nB + c] * B.meta.scale[r],
      units.length, (r) => unitTip(units[r], r), (_r, c) => goto(c));
  } else if (state.map === "step") {
    if (q >= nc) { cap.textContent = `position ${q} exists only in one of the two runs. ${where}`; drawHeatmap(canvas, { rows: 0, cols: 0 }, {}); return; }
    const pb = await api("/api/position", { run_id: base.run_id, q });
    const labels = run.snapshots, dims = state.info.dims;
    cap.textContent = "";
    const S = labels.length;
    const changed = dims.map((d, i) => i).filter((i) => p.values[i].some((v, s) => Math.abs(v - pb.values[i][s]) > 1e-6 * Math.max(1, Math.abs(v))));
    cap.textContent = `position ${q} ${charAt(q)}: each dim (rows) after each sublayer (columns), knocked-out minus normal, ` +
      `each row scaled by its biggest change. ${changed.length} of ${dims.length} dims change here. ${where}` +
      (changed.length ? "" : " Nothing changes in this view: try a later layer or step, or the residual stream map.");
    const mx = changed.map((i) => Math.max(...p.values[i].map((v, s) => Math.abs(v - pb.values[i][s]))));
    drawHeatmap(canvas, {
      key: `diffstep:${run.run_id}:${base.run_id}:${q}`, rows: changed.length, cols: S,
      get: (r, s) => (p.values[changed[r]][s] - pb.values[changed[r]][s]) / mx[r], norm: (v) => (v + 1) / 2, lut: DIV,
      rowLabels: changed.map((i) => dims[i].name),
      valueText: (r, s) => { const i = changed[r];
        return `${dims[i].name} after ${labels[s]}
normal ${fmt(pb.values[i][s])} → knocked out ${fmt(p.values[i][s])}
${dims[i].doc}`; },
      rowTip: (r) => dimTip(dims[changed[r]].name),
    }, {});
  } else if (state.map === "attention") {
    if (!state.head) state.head = defaultHead(p);
    if (!state.head) { cap.textContent = "pick a head to compare"; drawHeatmap(canvas, { rows: 0, cols: 0 }, {}); return; }
    const [layer, head] = state.head;
    syncHeadPicker();
    const { K, B } = await pairs("/api/attention", { layer, head });
    let mx = 0;
    for (let r = 0; r < nc; r++) for (let c = 0; c <= r; c++) mx = Math.max(mx, Math.abs(K.data[r * nK + c] - B.data[r * nB + c]));
    const changedRows = new Set();
    for (let r = 0; r < nc; r++) for (let c = 0; c <= r; c++) if (Math.abs(K.data[r * nK + c] - B.data[r * nB + c]) > 2e-3) changedRows.add(r);
    cap.textContent = `L${layer + 1} '${K.meta.name}': attention of the knocked-out run minus the normal run ` +
      `(orange = reads more, blue = reads less). ${changedRows.size} of ${nc} rows read differently. ${where}` +
      (changedRows.size ? "" : " This head reads the same everywhere: try a later layer's head.");
    drawHeatmap(canvas, {
      key: `diffattn:${run.run_id}:${base.run_id}:${layer}:${head}`, rows: nc, cols: nc, square: true,
      get: (r, c) => (mx ? (K.data[r * nK + c] - B.data[r * nB + c]) / mx : 0), norm: (v) => (v + 1) / 2, lut: DIV,
      colChars: run.chars.slice(0, nc).map(showChar), markCol: split < nc ? split : undefined,
      valueText: (r, c) => `query ${r} ${charAt(r)} → key ${c} ${charAt(c)}
normal ${(B.data[r * nB + c] * 100).toFixed(2)}%` +
        ` → knocked out ${(K.data[r * nK + c] * 100).toFixed(2)}%`,
      onClick: (r) => goto(r),
    }, { q, noDim: true });
  }
}

async function renderMap(p) {
  if (!state.run) return;
  scheduleHash();
  $("diff-ctl").hidden = !(koCount() > 0 && state.map !== "weights");
  if (diffOn()) {
    document.querySelectorAll("#map-controls label").forEach((l) => l.classList.toggle("on", l.dataset.for === state.map));
    renderHowto();
    return renderDiffMap(p);
  }
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
      rowTip: (r) => dimTip(rows[r].name),
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
      rowTip: (r) => dimTip(rows[r].name),
      onClick: (r) => { state.openDim = rows[r].name; renderDims(p); },
    }, {});
  } else if (state.map === "attention") {
    if (!state.head) state.head = defaultHead(p);
    if (!state.head) {
      cap.textContent = "pick a head with the layer / head menus above, or click one in the Attention heads panel";
      drawHeatmap(canvas, { rows: 0, cols: 0 }, {});
      return;
    }
    const [layer, head] = state.head;
    syncHeadPicker();
    const { key, meta, data } = await mapData("/api/attention", { run_id: run.run_id, layer, head });
    const g = Number($("gamma").value);
    const spec = headSpec(layer, head);
    const pointers = Array.from({ length: n }, (_, r) => {
      let best = 0;
      for (let c = 1; c <= r; c++) if (data[r * n + c] > data[r * n + best]) best = c;
      return best;
    });
    const at = p.heads.find((h) => h.layer === layer && h.head === head);
    const [k, w] = at.top[0];
    const wrote = Object.entries(at.wrote).map(([d, v]) => `${d}=${fmt(v)}`).join(", ") || "nothing (its value is zero here)";
    cap.textContent = `L${layer + 1} '${meta.name}' — ${spec.doc}\n` +
      `At this step (row ${q} ${charAt(q)}) it reads position ${k} ${charAt(k)}${w < 0.999 ? ` (${(w * 100).toFixed(0)}%)` : ""} and writes ${wrote}.\n` +
      "Rows = the character doing the reading, columns = the character read. The blue line traces where each row reads, " +
      "so a vertical run is a pointer that holds and a sideways step is a jump to a newer landmark.";
    drawHeatmap(canvas, {
      key: `${key}:g${g}`, rows: n, cols: n, square: true, // gamma changes the colours: part of the cache key
      get: (r, c) => data[r * n + c], norm: (v) => Math.pow(v, g), lut: SEQ,
      pointers, colChars: run.chars.map(showChar),
      rowLabels: run.chars.map((ch, r) => `${r} ${showChar(ch)} → ${pointers[r]} ${showChar(run.chars[pointers[r]])}`),
      rowTip: (r) => `row ${r} ${charAt(r)} reads position ${pointers[r]} ${charAt(pointers[r])} ` +
        `(${(data[r * n + pointers[r]] * 100).toFixed(1)}%)\n${spec.doc}`,
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
      rowTip: (r) => {
        const m = /^#(\d+) -> /.exec(meta.rows[r]);
        return m ? unitTip(state.info.layers[body.layer].units[Number(m[1])], Number(m[1])) : dimTip(meta.rows[r]);
      },
      valueText: (r, c) => `row ${meta.rows[r]}\ncol ${meta.cols[c]}\nw = ${data[r * cols + c]}`,
    }, {});
  } else {
    const sel = $("mlp-layer");
    const layer = Number(sel.value);
    const { key, meta, data } = await mapData("/api/mlp", { run_id: run.run_id, layer });
    const specs = state.info.layers[layer].units;
    const rows = specs.map((u, i) => ({ ...u, i })).filter((u) => focused(u.circuit) && meta.scale[u.i] > 1e-9);
    cap.textContent = `layer ${layer + 1} MLP: post-ReLU activation of each unit (rows) at every position (columns), ` +
      "each unit scaled by its max; hover a cell for the unit's formula, a row name for what its abbreviations mean";
    drawHeatmap(canvas, {
      key: key + rows.map((r) => r.i).join(","),
      rows: rows.length, cols: n,
      get: (r, c) => data[rows[r].i * n + c], norm: (v) => v, lut: SEQ,
      rowLabels: rows.map((r) => `#${r.i} ${r.write}`), groups: groupsOf(rows.map((r) => r.circuit)),
      valueText: (r, c) => `unit #${rows[r].i} at position ${c} ${charAt(c)}\n${rows[r].doc}\nReLU(${rows[r].read})\n= ${fmt(data[rows[r].i * n + c] * meta.scale[rows[r].i])} → ${rows[r].write}`,
      rowTip: (r) => unitTip(rows[r], rows[r].i),
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
    runExpression(getInput());
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
  $("g-circuit").addEventListener("input", () => { renderGraph(); scheduleHash(); });
  $("btn-link").addEventListener("click", copyLink);
  $("model-sel").addEventListener("change", (e) => switchModel(e.target.value).catch((err) => setStatus(err.message, true)));
  $("diff").addEventListener("change", () => state.pos.has(state.q) && renderMap(state.pos.get(state.q)));
  document.querySelectorAll("#guide .try").forEach((b) => {
    b.dataset.tip = "Set the visualizer up to show this (runs an example and opens the right view).";
    b.addEventListener("click", () => tryIt(JSON.parse(b.dataset.try)).catch((e) => setStatus(e.message, true)));
  });
  const pickHead = (layer, head) => {
    state.head = [layer, head];
    if (!state.pos.has(state.q)) return;
    const p = state.pos.get(state.q);
    renderHeads(p);
    renderHeadDetail(p);
    renderMap(p);
  };
  $("a-layer").addEventListener("input", () => pickHead(Number($("a-layer").value), 0));
  $("a-head").addEventListener("input", () => pickHead(Number($("a-layer").value), Number($("a-head").value)));
  for (const id of ["snapshot", "mlp-layer", "gamma", "w-kind", "w-layer", "w-head", "w-trim"]) {
    $(id).addEventListener("input", () => state.pos.has(state.q) && renderMap(state.pos.get(state.q)));
  }
  $("focus").addEventListener("change", () => state.run && goto(state.q));
  window.addEventListener("resize", () => state.pos.has(state.q) && renderMap(state.pos.get(state.q)));
  // The overview fits the panel width (stages spread, or stacked on narrow screens): redraw it on resize.
  let graphW = 0, graphTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(graphTimer);
    graphTimer = setTimeout(() => {
      const w = $("graph").clientWidth;
      if (w !== graphW && $("g-circuit").value === "all") { graphW = w; renderGraph(); }
    }, 150);
  });
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

function renderGraph() {
  if (!state.info) return;
  const circuit = $("g-circuit").value;
  const pick = $("g-pick");
  if (circuit === "all") {
    const { edges } = drawOverview($("graph"), state.info);
    pick.classList.add("muted");
    pick.textContent = "hover an arrow to list the values it carries, a box for what it reads and feeds";
    $("g-caption").textContent = `How the ${state.info.layers.length} layers' circuits work together: the frame locator ` +
      "finds the current step's landmarks and switches one calculator on (act_*); the parser picks the sub-expression, " +
      "copies it into the frame and rewrites the line; each calculator reads the frame and writes its votes for the next " +
      `byte. ${edges.length} circuit-to-circuit flows; one, pos and pos² are left out.`;
    return;
  }
  const out = drawGraph($("graph"), state.info, circuit, {
    onPick: (n) => {
      pick.classList.remove("muted");
      pick.textContent = `${n.name}: ${n.doc}\nwritten at: `;
      pick.append(srcLinks(n.src));
    },
  });
  $("g-caption").textContent = `${out.nodes} values and ${out.edges} dependencies of the ${circuit} circuit, ` +
    "laid out by the layer that first writes each value (blue: an attention head moves it, orange: an MLP unit " +
    "computes it). Faded names come from other circuits. one, pos and pos² feed almost everything and are left out. " +
    "Hover an edge to see which heads or units it stands for.";
}

/** "Try it" buttons in the guide: set up an expression, step, map, head or knockout to show one idea. */
async function tryIt(cfg) {
  log("guide: try", JSON.stringify(cfg));
  if (cfg.circuit) { $("g-circuit").value = cfg.circuit; renderGraph(); }
  if (cfg.expr) {
    $("expr-input").value = cfg.expr;
    state.ko = { heads: [], units: [], dims: [] };
    if (cfg.knockOutWriters) { // every unit of that layer that writes the named value
      state.ko.units = state.info.layers[cfg.layer].units.map((u, i) => (u.writes.includes(cfg.knockOutWriters) ? [cfg.layer, i] : null))
        .filter(Boolean);
    }
    await runExpression(cfg.expr);
    if (cfg.phase) {
      const q = state.run.steps.findIndex((s) => s.phase === cfg.phase);
      if (q >= 0) await goto(q);
    }
  }
  if (cfg.head) {
    for (const L of state.info.layers) {
      const h = L.heads.findIndex((x) => x.name === cfg.head);
      if (h >= 0) { state.head = [L.index, h]; break; }
    }
  }
  if (cfg.mlpLayer !== undefined) $("mlp-layer").value = cfg.mlpLayer;
  if (cfg.map) {
    document.querySelectorAll("[data-map]").forEach((x) => x.classList.toggle("active", x.dataset.map === cfg.map));
    state.map = cfg.map;
  }
  if (state.pos.has(state.q)) {
    const p = state.pos.get(state.q);
    renderHeads(p); renderHeadDetail(p); await renderMap(p);
  }
  const target = cfg.scroll ? $(cfg.scroll) : cfg.map ? $("map") : $("tokens");
  target?.scrollIntoView({ behavior: "smooth", block: "center" });
}

// ------------------------------------------------------------------ pinning
// Up to three panels can be pinned directly under the sequence slider, so their views sit next
// to the controls that change the step: e.g. a heatmap with the Step panel and the heads beside
// it. Big panels (heatmaps, circuit graph) get their own column; the smaller ones stack in a
// column next to them, or sit side by side when no big panel is pinned (wide screens; narrow
// screens stack everything). Pinning a fourth sends the oldest back. The choice is a per-viewer
// convenience, remembered in this browser only.
const PIN_KEY = "handviz.pinned";
const MAX_PINS = 3;
const BIG_PANELS = new Set(["p-maps", "p-graph"]);
const pinHome = new Map(); // panel id -> placeholder comment marking its usual place
let pins = []; // pinned panel ids, oldest first

function unpinOne(id) {
  const panel = $(id);
  pinHome.get(id).replaceWith(panel);
  pinHome.delete(id);
  panel.classList.remove("is-pinned");
  panel.querySelector(".pin").textContent = "📌";
  pins = pins.filter((x) => x !== id);
}

function pinOne(id) {
  if (pins.length >= MAX_PINS) unpinOne(pins[0]);
  const panel = $(id);
  const home = document.createComment(`home of ${id}`);
  panel.replaceWith(home);
  pinHome.set(id, home);
  $(BIG_PANELS.has(id) ? "pinned-big" : "pinned-small").append(panel);
  panel.classList.add("is-pinned");
  panel.querySelector(".pin").textContent = "✕ unpin";
  pins.push(id);
}

/** Pin ``id`` (or unpin it if it is pinned); ``ids`` restores a saved set. */
function togglePin(id, { ids = null, scroll = true } = {}) {
  if (ids) ids.filter((x) => $(x)?.dataset.pin).slice(-MAX_PINS).forEach(pinOne);
  else if (pins.includes(id)) unpinOne(id);
  else pinOne(id);
  const slot = $("pinned");
  const nBig = pins.filter((x) => BIG_PANELS.has(x)).length, nSmall = pins.length - nBig;
  slot.hidden = !pins.length;
  slot.dataset.layout = nBig && nSmall ? "big+small" : nBig ? "big" : "small";
  $("pinned-small").style.setProperty("--n", String(Math.max(1, nSmall)));
  $("pinned-big").hidden = !nBig;
  $("pinned-small").hidden = !nSmall;
  try { pins.length ? localStorage.setItem(PIN_KEY, JSON.stringify(pins)) : localStorage.removeItem(PIN_KEY); }
  catch { /* private mode */ }
  log("pinned", pins.join(", ") || "nothing");
  scheduleHash();
  // widths changed: redraw the canvases and the graph that size themselves to their panel
  if (state.pos.has(state.q)) renderMap(state.pos.get(state.q));
  renderGraph();
  if (scroll && pins.length) slot.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function setupPins() {
  document.querySelectorAll("[data-pin]").forEach((panel) => {
    const b = document.createElement("button");
    b.className = "pin";
    b.textContent = "📌";
    b.dataset.tip = `Pin "${panel.dataset.pin}" right under the slider, so it stays next to the step controls ` +
      `(up to ${MAX_PINS} panels; pinning another sends the oldest back; remembered on this device). ` +
      "Press again to put it back.";
    b.addEventListener("click", (e) => { e.stopPropagation(); togglePin(panel.id); });
    panel.querySelector(".panel-title")?.prepend(b);
  });
  let saved = [];
  try {
    const raw = localStorage.getItem(PIN_KEY);
    saved = raw ? (raw.startsWith("[") ? JSON.parse(raw) : [raw]) : []; // older pages saved one id
  } catch { /* private mode or bad value */ }
  if (saved.length) togglePin(null, { ids: saved, scroll: false });
}

// ------------------------------------------------------------------ shareable links
// The view lives in the URL hash (expression, step, heatmap, head, layers, pins, knockouts), so
// a link reopens exactly what you are looking at. replaceState: no history entry per step.
let hashTimer = null;
function viewParams() {
  const h = new URLSearchParams();
  if (!state.run) return h;
  if (state.model !== "calc") h.set("m", state.model);
  h.set("e", state.run.expression);
  h.set("q", String(state.q));
  h.set("map", state.map);
  if (state.head) h.set("head", state.head.join("."));
  if (state.map === "mlp") h.set("ml", $("mlp-layer").value);
  if (state.map === "residual") h.set("snap", $("snapshot").value);
  if ($("g-circuit").value !== "all") h.set("g", $("g-circuit").value);
  if (pins.length) h.set("pins", pins.join(","));
  const ko = [...state.ko.heads.map(([l, x]) => `h${l}.${x}`), ...state.ko.units.map(([l, x]) => `u${l}.${x}`),
              ...state.ko.dims.map((d) => `d${d}`)];
  if (ko.length) h.set("ko", ko.join(","));
  if (!$("diff").checked) h.set("diff", "0");
  return h;
}
function scheduleHash() {
  clearTimeout(hashTimer);
  hashTimer = setTimeout(() => {
    const p = viewParams().toString();
    if (p && location.hash.slice(1) !== p) history.replaceState(null, "", `#${p}`);
  }, 250);
}
function readHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  const num = (k) => (h.has(k) && /^-?\d+$/.test(h.get(k)) ? Number(h.get(k)) : null);
  const ko = { heads: [], units: [], dims: [] };
  for (const t of (h.get("ko") || "").split(",").filter(Boolean)) {
    const m = /^([hu])(\d+)\.(\d+)$/.exec(t);
    if (m) ko[m[1] === "h" ? "heads" : "units"].push([Number(m[2]), Number(m[3])]);
    else if (t[0] === "d") ko.dims.push(t.slice(1));
  }
  const head = /^(\d+)\.(\d+)$/.exec(h.get("head") || "");
  return { m: h.get("m"), e: h.get("e"), q: num("q"), map: h.get("map"), head: head ? [Number(head[1]), Number(head[2])] : null,
           ml: num("ml"), snap: num("snap"), g: h.get("g"), pins: (h.get("pins") || "").split(",").filter(Boolean),
           ko, diff: h.get("diff") !== "0", any: h.toString().length > 0 };
}
async function copyLink() {
  scheduleHash();
  clearTimeout(hashTimer);
  history.replaceState(null, "", `#${viewParams().toString()}`);
  try {
    await navigator.clipboard.writeText(location.href);
    setStatus("link to this exact view copied");
  } catch {
    prompt("Copy this link:", location.href); // clipboard blocked (http, old browser)
  }
  log("copied view link", location.href.length, "chars");
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
  setupPins();
  trackBar();
  if (matchMedia("(max-width: 700px)").matches) $("map-howto").open = false; // long on a phone; one tap opens it
  try {
    const v = readHash();
    const offered = await loadModels();
    const want = v.m || "calc"; // links leave out m for the calculator
    state.model = offered.includes(want) ? want : offered[0]; // a model that is not offered opens the default
    if (v.m && v.m !== state.model) log("link names model", v.m, "which is not offered; showing", state.model);
    await loadInfo();
    applyModelChrome();
    if (!v.e) setInput(state.info.default_input);
    setStatus(`${state.info.layers.length} layers · d_model ${state.info.d_model} · ${state.info.nonzero_params} hand-written params`);
    if (v.any) log("opening the view from the link", location.hash.length, "chars");
    if (v.e) setInput(v.e);
    state.ko = v.ko;
    if (v.map && document.querySelector(`[data-map="${v.map}"]`)) {
      state.map = v.map;
      document.querySelectorAll("[data-map]").forEach((x) => x.classList.toggle("active", x.dataset.map === v.map));
    }
    if (v.head && state.info.layers[v.head[0]]?.heads[v.head[1]]) state.head = v.head;
    if (v.ml !== null && state.info.layers[v.ml]?.units.length) $("mlp-layer").value = v.ml;
    if (v.g) $("g-circuit").value = v.g;
    $("diff").checked = v.diff;
    if (v.pins.length) { [...pins].forEach(unpinOne); togglePin(null, { ids: v.pins, scroll: false }); }
    renderGraph();
    await runExpression(getInput());
    if (v.snap !== null && v.snap < state.run.snapshots.length) $("snapshot").value = v.snap;
    if (v.q !== null) await goto(v.q);
    else if (v.snap !== null && state.pos.has(state.q)) await renderMap(state.pos.get(state.q));
  } catch (e) {
    setStatus(e.message, true);
  }
}

main();
