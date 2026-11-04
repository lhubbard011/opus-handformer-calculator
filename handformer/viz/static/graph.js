// Circuit graph: the dataflow of one circuit, generated from the model description.
//
// Nodes are residual dims, placed in the column of the first layer that writes
// them (inputs in column 0). An edge r -> w means some head (blue) or MLP unit
// (orange) of the circuit reads r and writes w. Nothing here is drawn by hand:
// it is the weight specs, laid out by layer. `one`, `pos` and `pos2` feed almost
// everything, so their edges are left out (said in the caption).

const SVGNS = "http://www.w3.org/2000/svg";
const COL_W = 132, ROW_H = 15, PAD = 18, NODE_W = 112;
const HIDDEN = new Set(["one", "pos", "pos2"]);
const log = (...a) => console.info("[graph]", ...a);

function el(tag, attrs = {}, text = null) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text !== null) e.textContent = text;
  return e;
}

/** {nodes, edges} for one circuit: specs whose circuit matches, their dims, and the dims they read. */
export function circuitGraph(info, circuit) {
  const dims = new Map(info.dims.map((d) => [d.name, d]));
  const firstWrite = new Map(); // dim -> earliest layer (1-based) that writes it; inputs 0
  for (const d of info.dims) if (d.cat === "input") firstWrite.set(d.name, 0);
  const edges = new Map(); // "r>w" -> {r, w, kind, specs: [...]}
  for (const L of info.layers) {
    const parts = [...L.heads.map((h, i) => ({ ...h, kind: "head", id: `L${L.index + 1} head '${h.name}'`, i })),
                   ...L.units.map((u, i) => ({ ...u, kind: "unit", id: `L${L.index + 1} unit #${i}`, i }))];
    for (const s of parts) {
      for (const w of s.writes) if (!firstWrite.has(w)) firstWrite.set(w, L.index + 1);
      if (s.circuit !== circuit) continue;
      for (const r of s.reads) {
        if (HIDDEN.has(r)) continue;
        for (const w of s.writes) {
          if (r === w) continue;
          const k = `${r}>${w}`;
          if (!edges.has(k)) edges.set(k, { r, w, kinds: new Set(), specs: [] });
          edges.get(k).kinds.add(s.kind);
          if (edges.get(k).specs.length < 6) edges.get(k).specs.push({ id: s.id, src: s.src });
        }
      }
    }
  }
  const names = new Set();
  for (const e of edges.values()) { names.add(e.r); names.add(e.w); }
  const nodes = [...names].map((n) => ({
    name: n, doc: dims.get(n)?.doc || "", src: dims.get(n)?.src || [],
    col: firstWrite.get(n) ?? 0, own: (dims.get(n)?.cat || "") === circuit,
  }));
  return { nodes, edges: [...edges.values()] };
}

/**
 * Draw the graph into `container` (an element that scrolls). `srcUrl(where)` makes source links;
 * `onPick(node)` is called when a node is clicked.
 */
export function drawGraph(container, info, circuit, { onPick } = {}) {
  const t0 = performance.now();
  const { nodes, edges } = circuitGraph(info, circuit);
  container.replaceChildren();
  if (!nodes.length) { container.textContent = "this circuit has no dataflow edges"; return { nodes: 0, edges: 0 }; }
  const cols = Math.max(...nodes.map((n) => n.col)) + 1;
  const perCol = Array.from({ length: cols }, () => []);
  // own dims first in each column, then by name, so the circuit's own values sit together
  nodes.sort((a, b) => a.col - b.col || Number(b.own) - Number(a.own) || a.name.localeCompare(b.name));
  for (const n of nodes) { n.row = perCol[n.col].length; perCol[n.col].push(n); }
  const height = PAD * 2 + 16 + Math.max(...perCol.map((c) => c.length)) * ROW_H;
  const width = PAD * 2 + cols * COL_W;
  const svg = el("svg", { width, height, viewBox: `0 0 ${width} ${height}`, class: "graph" });
  const pos = new Map(nodes.map((n) => [n.name, { x: PAD + n.col * COL_W, y: PAD + 16 + n.row * ROW_H }]));

  for (let c = 0; c < cols; c++) {
    if (!perCol[c].length) continue;
    svg.append(el("text", { x: PAD + c * COL_W, y: PAD, class: "g-col" }, c === 0 ? "input" : `L${c}`));
  }
  const edgeEls = [];
  const gEdges = el("g");
  for (const e of edges) {
    const a = pos.get(e.r), b = pos.get(e.w);
    const x1 = a.x + NODE_W, y1 = a.y - 4, x2 = b.x, y2 = b.y - 4;
    // a same-column edge (a dim written and read in one layer's chain) loops out and back
    const dx = x2 <= x1 ? 40 : Math.max(30, (x2 - x1) / 2);
    const d = `M${x1},${y1} C${x1 + dx},${y1} ${x2 - dx},${y2} ${x2},${y2}`;
    const kind = e.kinds.size > 1 ? "both" : [...e.kinds][0];
    const p = el("path", { d, class: `g-edge g-${kind}` });
    p.dataset.tip = `${e.r} → ${e.w}\nvia ${e.specs.map((s) => s.id).join(", ")}${e.specs.length >= 6 ? ", …" : ""}`;
    p._e = e;
    edgeEls.push(p);
    gEdges.append(p);
  }
  svg.append(gEdges);
  const nodeEls = new Map();
  for (const n of nodes) {
    const { x, y } = pos.get(n.name);
    const g = el("g", { class: `g-node${n.own ? "" : " g-ext"}`, transform: `translate(${x},${y})` });
    g.append(el("rect", { x: -3, y: -11, width: NODE_W, height: ROW_H - 2, rx: 3 }));
    g.append(el("text", { x: 2, y: 0 }, n.name.length > 18 ? `${n.name.slice(0, 17)}…` : n.name));
    g.dataset.tip = `${n.name}${n.own ? "" : ` (from outside the ${circuit} circuit)`}\n${n.doc}` +
      (n.src.length ? `\nsource: ${n.src.join(", ")}` : "") + "\n(click to highlight what it reads and feeds)";
    g.addEventListener("click", () => { highlight(n.name); onPick?.(n); });
    nodeEls.set(n.name, g);
    svg.append(g);
  }
  let picked = null;
  function highlight(name) {
    picked = picked === name ? null : name;
    const near = new Set(picked ? [picked] : []);
    for (const p of edgeEls) {
      const on = picked && (p._e.r === picked || p._e.w === picked);
      p.classList.toggle("on", !!on);
      p.classList.toggle("off", !!picked && !on);
      if (on) { near.add(p._e.r); near.add(p._e.w); }
    }
    for (const [nm, g] of nodeEls) {
      g.classList.toggle("on", nm === picked);
      g.classList.toggle("off", !!picked && !near.has(nm));
    }
  }
  container.append(svg);
  log(`${circuit}: ${nodes.length} dims, ${edges.length} edges, ${cols} columns in ${(performance.now() - t0).toFixed(0)}ms`);
  return { nodes: nodes.length, edges: edges.length, highlight };
}

// ---------------------------------------------------------------- overview

const STAGES = [ // column, and the order circuits stack in it
  ["input"], ["frame"], ["parser"], ["add", "sub", "mul", "div", "pow"], ["output"],
];
const CIRCUIT_NAME = { input: "byte embedding + position", frame: "frame locator", parser: "parser / rewriter",
  add: "add", sub: "sub", mul: "mul", div: "div", pow: "pow", output: "output votes → logits" };
const CALCS = ["frame", "parser", "add", "sub", "mul", "div", "pow"];

/** Circuit-level dataflow: {nodes: {c: {heads, units, dims}}, edges: [{from, to, dims:[...]}], layers}. */
export function overviewGraph(info) {
  const cat = new Map(info.dims.map((d) => [d.name, d.cat]));
  const writers = new Map(); // dim -> Set of circuits whose specs write it
  const parts = info.layers.flatMap((L) => [...L.heads.map((h) => ({ ...h, kind: "head", layer: L.index })),
                                           ...L.units.map((u) => ({ ...u, kind: "unit", layer: L.index }))]);
  for (const d of info.dims) if (d.cat === "input") writers.set(d.name, new Set(["input"]));
  for (const s of parts) for (const w of s.writes) {
    if (!writers.has(w)) writers.set(w, new Set());
    writers.get(w).add(s.circuit);
  }
  const nodes = Object.fromEntries(STAGES.flat().map((c) => [c, { heads: 0, units: 0, dims: 0 }]));
  for (const d of info.dims) if (nodes[d.cat]) nodes[d.cat].dims++;
  const layers = Object.fromEntries(CALCS.map((c) => [c, info.layers.map(() => ({ heads: 0, units: 0 }))]));
  const edges = new Map();
  const edge = (from, to, d) => {
    const k = `${from}>${to}`;
    if (!edges.has(k)) edges.set(k, { from, to, dims: new Set() });
    edges.get(k).dims.add(d);
  };
  for (const s of parts) {
    if (!nodes[s.circuit]) continue;
    nodes[s.circuit][s.kind === "head" ? "heads" : "units"]++;
    if (layers[s.circuit]) layers[s.circuit][s.layer][s.kind === "head" ? "heads" : "units"]++;
    for (const r of s.reads) {
      if (HIDDEN.has(r)) continue;
      for (const w of writers.get(r) || []) if (w !== s.circuit && nodes[w]) edge(w, s.circuit, r);
    }
    for (const w of s.writes) if (cat.get(w) === "output") edge(s.circuit, "output", w);
  }
  return { nodes, layers, edges: [...edges.values()].map((e) => ({ ...e, dims: [...e.dims].sort() })) };
}

/** Draw the overview: circuits as boxes joined by the values they pass, then the layer-sharing strip. */
export function drawOverview(container, info) {
  const t0 = performance.now();
  const { nodes, layers, edges } = overviewGraph(info);
  container.replaceChildren();
  // Wide screens: stages left to right. Narrow (phones): one column of full-width boxes, value
  // arrows curving down the right edge and the text flow down the left, so nothing scrolls sideways.
  // The stages need 5 boxes and 4 gaps of at least 40 px; below that, the stacked layout.
  const avail = container.clientWidth - 18; // leave room for the panel's own scrollbar
  const narrow = avail < PAD * 2 + 5 * 170 + 4 * 40;
  const BOX_H = 46, nL = info.layers.length, ROW = 20;
  const GAP_X = narrow ? 0 : Math.min(160, (avail - PAD * 2 - 5 * 170) / 4); // spread to fill the panel
  const pos = {};
  let BOX_W, TOP, flowH, width, LABEL, CELL, valuePath, textPath;
  if (narrow) {
    const W = Math.max(260, container.clientWidth - 18), ARC = 56, LEFT = 22, GAP_Y = 14; // room for its scrollbar
    TOP = 44; BOX_W = W - PAD * 2 - ARC - LEFT;
    const order = STAGES.flat();
    order.forEach((c, i) => { pos[c] = { x: PAD + LEFT, y: TOP + i * (BOX_H + GAP_Y), i }; });
    flowH = TOP + order.length * (BOX_H + GAP_Y);
    width = W; LABEL = 92; CELL = Math.max(9, (W - PAD * 2 - LABEL) / nL);
    valuePath = (a, b) => {
      const x = a.x + BOX_W, off = Math.min(ARC - 6, 12 + 8 * Math.abs(b.i - a.i));
      return `M${x},${a.y + BOX_H / 2} C${x + off},${a.y + BOX_H / 2} ${x + off},${b.y + BOX_H / 2} ${x},${b.y + BOX_H / 2}`;
    };
    textPath = (a, b) => {
      const x = a.x, off = Math.min(LEFT - 2, 6 + 3 * Math.abs(b.i - a.i));
      return `M${x},${a.y + BOX_H / 2} C${x - off},${a.y + BOX_H / 2} ${x - off},${b.y + 12} ${x},${b.y + 12}`;
    };
  } else {
    const GAP_Y = 16;
    TOP = 30; BOX_W = 170;
    const colH = Math.max(...STAGES.map((col) => col.length * (BOX_H + GAP_Y)));
    STAGES.forEach((col, ci) => {
      const h = col.length * (BOX_H + GAP_Y) - GAP_Y;
      col.forEach((c, ri) => {
        pos[c] = { x: PAD + ci * (BOX_W + GAP_X), y: TOP + (colH - h) / 2 + ri * (BOX_H + GAP_Y) };
      });
    });
    pos.parser.y = TOP - 6; // above the frame → calculator flows: the parser does not feed them values
    flowH = TOP + colH + 10;
    LABEL = 110; CELL = Math.max(9, Math.min(44, (avail - PAD * 2 - LABEL) / nL));
    width = Math.max(avail, PAD * 2 + STAGES.length * BOX_W + 4 * GAP_X);
    valuePath = (a, b) => {
      if (b.x > a.x) {
        const x1 = a.x + BOX_W, y1 = a.y + BOX_H / 2, x2 = b.x, y2 = b.y + BOX_H / 2;
        return `M${x1},${y1} C${x1 + GAP_X / 2},${y1} ${x2 - GAP_X / 2},${y2} ${x2},${y2}`;
      }
      const x1 = a.x + BOX_W / 2, y1 = a.y + BOX_H, x2 = b.x + BOX_W / 2, y2 = b.y + BOX_H; // same column: loop below
      return `M${x1},${y1} C${x1},${y1 + 60} ${x2},${y2 + 60} ${x2},${y2}`;
    };
    textPath = (a, b) => {
      const x1 = a.x + BOX_W, y1 = a.y + BOX_H / 2, x2 = b.x, y2 = b.y + 10;
      return `M${x1},${y1} C${x1 + GAP_X / 2},${y1} ${x2 - GAP_X / 2},${y2} ${x2},${y2}`;
    };
  }
  const stripTop = flowH + (narrow ? 46 : 40);
  const height = stripTop + 22 + CALCS.length * ROW + 30;
  const svg = el("svg", { width, height, viewBox: `0 0 ${width} ${height}`, class: "graph overview" });
  const title = narrow ? ["solid (right): residual values passed", "dashed (left): through the written text"]
    : ["who feeds whom: solid arrows carry residual values (width = how many); dashed ones go through the text"];
  title.forEach((t, i) => svg.append(el("text", { x: PAD, y: 16 + i * 14, class: "g-col" }, t)));

  const maxDims = Math.max(...edges.map((e) => e.dims.length));
  const gEdges = el("g");
  for (const e of edges.sort((a, b) => b.dims.length - a.dims.length)) {
    const d = valuePath(pos[e.from], pos[e.to]);
    const w = 1 + (narrow ? 4 : 7) * Math.sqrt(e.dims.length / maxDims);
    const p = el("path", { d, class: "g-flow", "stroke-width": w.toFixed(1) });
    p.dataset.tip = `${CIRCUIT_NAME[e.from]} → ${CIRCUIT_NAME[e.to]}: ${e.dims.length} value(s)\n` +
      e.dims.slice(0, 40).join(", ") + (e.dims.length > 40 ? ", …" : "");
    gEdges.append(p);
  }
  svg.append(gEdges);
  // The parser and the calculators share no residual values: they cooperate through the text the
  // model writes (the copied sub-expression in the frame; the result bytes the rewrite copies back).
  const gText = el("g");
  for (const c of STAGES[3]) {
    const p = el("path", { d: textPath(pos.parser, pos[c]), class: "g-text" });
    p.dataset.tip = `parser ⇄ ${c}, through the written text (no shared values):\n` +
      `· the parser copies the chosen sub-expression into the frame as bytes; ${c} reads those bytes\n` +
      `· ${c} writes its result as bytes; the parser's rewrite copies them into the next line`;
    gText.append(p);
  }
  svg.append(gText);
  if (!narrow) {
    svg.append(el("text", { x: pos.parser.x + 8, y: pos.parser.y + BOX_H + 14, class: "g-box-sub" },
      "dashed → calculators: through the written text"));
  }
  for (const [c, n] of Object.entries(nodes)) {
    const { x, y } = pos[c];
    const g = el("g", { class: `g-box g-box-${c}`, transform: `translate(${x},${y})` });
    g.append(el("rect", { width: BOX_W, height: BOX_H, rx: 6 }));
    g.append(el("text", { x: 8, y: 18, class: "g-box-title" }, CIRCUIT_NAME[c]));
    const stats = c === "input" ? (narrow ? `${n.dims} values: byte + position` : `${n.dims} values from the byte and its position`)
      : c === "output" ? (narrow ? `${n.dims} votes → unembedding` : `${n.dims} votes, read by the unembedding`)
      : narrow ? `${n.heads} heads · ${n.units} units`
      : `${n.heads} heads · ${n.units} units · ${n.dims} own values`;
    g.append(el("text", { x: 8, y: 36, class: "g-box-sub" }, stats));
    const ins = edges.filter((e) => e.to === c), outs = edges.filter((e) => e.from === c);
    g.dataset.tip = `${CIRCUIT_NAME[c]}\nreads from: ${ins.map((e) => `${e.from} (${e.dims.length})`).join(", ") || "—"}\n` +
      `feeds: ${outs.map((e) => `${e.to} (${e.dims.length})`).join(", ") || "—"}` +
      (CALCS.includes(c) ? "\n(pick it in the circuit menu for its own graph)" : "");
    svg.append(g);
  }

  // layer strip: which circuits own heads (blue) and units (orange) at each layer
  const stripTitle = narrow ? ["heads (blue) and units (orange)", "per layer, L1–L22"]
    : ["working together in depth: heads (blue) and MLP units (orange) per layer; every position runs every layer"];
  stripTitle.forEach((t, i) => svg.append(el("text", { x: PAD, y: stripTop - 12 - (stripTitle.length - 1 - i) * 14, class: "g-col" }, t)));
  const every = CELL >= 26 ? 1 : CELL >= 13 ? 2 : 5; // layer labels that fit
  for (let l = 0; l < nL; l++) {
    if (l % every === 0) svg.append(el("text", { x: PAD + LABEL + l * CELL + (every === 1 ? 4 : 0), y: stripTop + 6, class: "g-col" }, `L${l + 1}`));
  }
  const maxH = Math.max(...CALCS.flatMap((c) => layers[c].map((v) => v.heads)), 1);
  const maxU = Math.max(...CALCS.flatMap((c) => layers[c].map((v) => v.units)), 1);
  CALCS.forEach((c, r) => {
    const y = stripTop + 14 + r * ROW;
    svg.append(el("text", { x: PAD, y: y + 13, class: "g-box-sub" }, CIRCUIT_NAME[c]));
    layers[c].forEach((v, l) => {
      if (!v.heads && !v.units) return;
      const x = PAD + LABEL + l * CELL;
      const g = el("g");
      if (v.heads) g.append(el("rect", { x, y: y + 2, width: Math.max(2, (CELL - 6) * Math.sqrt(v.heads / maxH)), height: 7, class: "g-h" }));
      if (v.units) g.append(el("rect", { x, y: y + 10, width: Math.max(2, (CELL - 6) * Math.sqrt(v.units / maxU)), height: 7, class: "g-u" }));
      g.dataset.tip = `layer ${l + 1} · ${CIRCUIT_NAME[c]}: ${v.heads} heads, ${v.units} units`;
      svg.append(g);
    });
  });
  container.append(svg);
  log(`overview: ${edges.length} circuit-to-circuit flows in ${(performance.now() - t0).toFixed(0)}ms`);
  return { edges };
}
