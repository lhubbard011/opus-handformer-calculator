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
