// The token pass as a picture of the model: the residual stream is a vertical spine; each
// sublayer branches off it and adds back in (⊕), as the forward pass does. Every attention head
// is a dot and every MLP unit a tick, in build order, coloured by the code module that wrote it.
// For the byte being traced, the heads that wrote and the units that fired are lit; the
// computation being played glows and the spine fills up to it. Built from the model description
// (/api/info) and the position data (/api/position), like the rest of the page.

const NS = "http://www.w3.org/2000/svg";
const log = (...a) => console.log("[pass-diagram]", ...a);
const PALETTE = ["#f2a541", "#5ec8e5", "#7bd88f", "#c792ea", "#ff8a65", "#e6db74", "#82aaff", "#f07178", "#89ddff", "#c3e88d"];
const SPINE_X = 86, BOX_X = 112, PAD_R = 12, ROW_GAP = 8, DOT = 7, DOT_GAP = 3, TICK_W = 3, TICK_H = 10, TICK_GAP = 1;

const moduleOf = (where) => (where ? where.split(":")[0].split("/").pop().replace(/\.py$/, "") : "?");

function svg(tag, attrs = {}, parent = null) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.append(e);
  return e;
}

export class PassDiagram {
  /** hooks: onStage(s) (a sublayer was clicked), onItem(kind, layer, index) (a head or unit) */
  constructor(root, hooks) {
    this.root = root;
    this.hooks = hooks;
    this.info = null;
    this.width = 0;
    new ResizeObserver(() => {
      const w = Math.round(this.root.clientWidth);
      if (this.info && Math.abs(w - this.width) > 8) this.draw(); // relayout on width changes
    }).observe(root);
  }

  /** Lay the model out (once per model, and again when the width changes). */
  setModel(info, labels) {
    this.info = info;
    this.labels = labels;
    const mods = [];
    for (const L of info.layers) for (const x of [...L.heads, ...L.units]) { const m = moduleOf(x.src); if (!mods.includes(m)) mods.push(m); }
    this.color = Object.fromEntries(mods.map((m, i) => [m, PALETTE[i % PALETTE.length]]));
    this.draw();
  }

  draw() {
    const info = this.info;
    const W = Math.max(320, Math.round(this.root.clientWidth));
    this.width = W;
    const inner = W - BOX_X - PAD_R - 12;
    const s = svg("svg", { class: "pd", width: W, role: "img", "aria-label": "The forward pass: the residual stream and every sublayer" });
    this.rows = []; // per stage: {g, y, kind, layer, items: [el], mid}
    let y = 10;
    const row = (stage, label, kind, layer, build) => {
      const g = svg("g", { class: "pd-row" }, s);
      const top = y;
      const inner_ = svg("g", { transform: `translate(${BOX_X + 6}, ${top + 18})` }, g);
      const { h, items } = build(inner_);
      const height = Math.max(28, h + 26);
      const box = svg("rect", { x: BOX_X, y: top, width: W - BOX_X - PAD_R, height, rx: 6, class: "pd-box" });
      g.prepend(box);
      svg("text", { x: BOX_X + 6, y: top + 13, class: "pd-label" }, g).textContent = label;
      const mid = top + 14;
      svg("line", { x1: SPINE_X, y1: mid, x2: BOX_X, y2: mid, class: "pd-branch" }, g);
      svg("circle", { cx: SPINE_X, cy: mid, r: 6, class: "pd-add" }, g);
      svg("text", { x: SPINE_X, y: mid + 3.5, class: "pd-plus", "text-anchor": "middle" }, g).textContent = kind === "embed" ? "●" : kind === "logits" ? "▼" : "+";
      svg("text", { x: SPINE_X - 12, y: mid + 4, class: "pd-layer", "text-anchor": "end" }, g).textContent =
        kind === "embed" ? "embed" : kind === "logits" ? "logits" : kind === "attn" ? `L${layer + 1}` : "";
      g.dataset.tip = kind === "embed" ? "The embedding: the byte's input features enter the residual stream." :
        kind === "logits" ? "The unembedding reads the residual stream into a score per byte." :
          `${label}: click to play this computation`;
      g.addEventListener("click", (e) => { if (!e.target.closest(".pd-item")) this.hooks.onStage?.(stage); });
      this.rows[stage] = { g, y: top, h: height, mid, kind, layer, items };
      y = top + height + ROW_GAP;
    };
    // grid of items (dots for heads, ticks for units), wrapping at the box width
    const grid = (parent, list, kind, layer) => {
      const w = kind === "heads" ? DOT : TICK_W, h = kind === "heads" ? DOT : TICK_H, gap = kind === "heads" ? DOT_GAP : TICK_GAP;
      const per = Math.max(1, Math.floor((inner + gap) / (w + gap)));
      const items = list.map((x, i) => {
        const cx = (i % per) * (w + gap), cy = Math.floor(i / per) * (h + gap + 1);
        const e = kind === "heads" ? svg("circle", { cx: cx + w / 2, cy: cy + h / 2, r: w / 2, class: "pd-item" }, parent)
          : svg("rect", { x: cx, y: cy, width: w, height: h, rx: 1, class: "pd-item" }, parent);
        const m = moduleOf(x.src);
        e.style.setProperty("--c", this.color[m]);
        e.dataset.tip = kind === "heads" ? `head ${i} · ${x.name} · ${m}\n${x.doc}` : `unit #${i} → ${x.write} · ${m}\n${x.doc}`;
        e.addEventListener("click", () => this.hooks.onItem?.(kind, layer, i));
        return e;
      });
      const lines = list.length ? Math.ceil(list.length / per) : 0;
      return { h: lines * (h + gap + 1), items };
    };
    row(0, "embedding + position", "embed", null, () => ({ h: 0, items: [] }));
    info.layers.forEach((L, li) => {
      row(1 + 2 * li, `attention · ${L.heads.length} head${L.heads.length === 1 ? "" : "s"}`, "attn", li, (p) => grid(p, L.heads, "heads", li));
      row(2 + 2 * li, `MLP · ${L.units.length} unit${L.units.length === 1 ? "" : "s"}`, "mlp", li, (p) => grid(p, L.units, "units", li));
    });
    const endStage = 1 + 2 * info.layers.length;
    row(endStage, "logits: the next byte", "logits", null, (p) => {
      this.logitG = svg("g", {}, p);
      return { h: 40, items: [] };
    });
    // the spine: grey all the way, filled up to the computation being played
    const top = this.rows[0].mid, bottom = this.rows[endStage].mid;
    const spine = svg("line", { x1: SPINE_X, y1: top, x2: SPINE_X, y2: bottom, class: "pd-spine" });
    this.fill = svg("line", { x1: SPINE_X, y1: top, x2: SPINE_X, y2: top, class: "pd-fill" });
    s.prepend(spine, this.fill);
    s.setAttribute("height", y + 4);
    s.setAttribute("viewBox", `0 0 ${W} ${y + 4}`);
    // the module legend
    const legend = document.createElement("div");
    legend.className = "pd-legend";
    for (const [m, c] of Object.entries(this.color)) {
      const sw = document.createElement("span");
      sw.className = "pd-key";
      sw.style.setProperty("--c", c);
      sw.textContent = m;
      legend.append(sw);
    }
    this.root.replaceChildren(legend, s);
    this.svgEl = s;
    if (this.active) this.setActive(this.active);
    if (this.stage !== undefined) this.setStage(this.stage, false);
    log(`drew ${info.layers.length} layers at ${W}px`);
  }

  /** Which heads wrote and which units fired for the byte being traced, and its top logits. */
  setActive(active) {
    this.active = active;
    if (!this.rows) return;
    this.info.layers.forEach((_L, li) => {
      this.rows[1 + 2 * li].items.forEach((e, h) => e.classList.toggle("on", active.heads.has(`${li}.${h}`)));
      this.rows[2 + 2 * li].items.forEach((e, u) => e.classList.toggle("on", active.units.has(`${li}.${u}`)));
    });
    this.rows.forEach((r, s) => r.g.classList.toggle("quiet", s > 0 && s < this.rows.length - 1 && !active.changed[s]));
    // the logits row: the top bytes as bars
    this.logitG.replaceChildren();
    const top = active.logits || [], mx = Math.max(...top.map(([, v]) => Math.abs(v)), 1e-9);
    const bw = Math.max(40, (this.width - BOX_X - PAD_R - 30) / Math.max(1, top.length));
    top.forEach(([b, v], i) => {
      const x = i * bw, h = Math.max(1, (24 * Math.max(0, v)) / mx);
      svg("rect", { x, y: 26 - h, width: bw - 8, height: h, rx: 2, class: i === 0 ? "pd-logit win" : "pd-logit" }, this.logitG);
      const label = b === "\n" ? "⏎" : b.charCodeAt(0) < 32 ? `\\x${b.charCodeAt(0).toString(16).padStart(2, "0")}` : b;
      svg("text", { x, y: 38, class: "pd-logit-label" }, this.logitG).textContent = `${label} ${Math.abs(v) < 1e-3 ? 0 : v.toFixed(1)}`;
    });
  }

  /** Glow the computation being played and fill the spine down to it; keep it in view. */
  setStage(s, scroll = true) {
    this.stage = s;
    if (!this.rows) return;
    this.rows.forEach((r, k) => { r.g.classList.toggle("now", k === s); r.g.classList.toggle("done", k < s); });
    const r = this.rows[s];
    this.fill.setAttribute("y2", r.mid);
    const box = this.root.closest(".pd-scroll");
    if (scroll && box) { // scroll the diagram's own box (never the page) so the row is visible
      const top = r.y - 40, bottom = r.y + r.h + 40;
      if (top < box.scrollTop) box.scrollTop = top;
      else if (bottom > box.scrollTop + box.clientHeight) box.scrollTop = bottom - box.clientHeight;
    }
  }
}
