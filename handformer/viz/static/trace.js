// Token pass: one byte's trip through the network, played back one computation at a time.
//
// The page already records, for the current position, every residual dim's value after each
// sublayer (embed, L1 attn, L1 mlp, ...), each head's attention and writes, each unit that fired,
// and the top logits (/api/position). The player walks those stages in forward-pass order: the
// residual stream column shows every dim this byte uses, with the ones the current stage changed
// lit up (before -> after), and the side panel says what the stage did. Play, pause, step,
// scrub; "skip quiet" passes over stages that change nothing. The stage is kept when the main
// step slider moves, so the same computation can be compared across bytes. The diagram
// (pass-diagram.js) shows the same pass as a picture of the model, beside the lists or alone.

import { PassDiagram } from "./pass-diagram.js";

const log = (...a) => console.log("[trace]", ...a);
const SPEEDS = { slow: 1200, normal: 500, fast: 150 };
const VIEW_KEY = "handviz.traceview";
const VIEWS = { both: "diagram + lists", diagram: "diagram", lists: "lists" };
const TOL = 1e-9;

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}
// display only: within 1e-4 of an integer shows the integer (the model's gates leave ~1e-6 residue)
const fmt = (v) => (Math.abs(v - Math.round(v)) < 1e-4 ? String(Math.round(v)) : String(Number(v.toFixed(4))));
const showChar = (c) => (c === "\n" ? "⏎" : c === undefined ? "" : c);
/** A byte as text: printable as itself (quoted), control bytes as \\xNN. */
const byteText = (c) => (c === "\n" ? "⏎" : c.charCodeAt(0) < 32 || c.charCodeAt(0) === 127
  ? `\\x${c.charCodeAt(0).toString(16).padStart(2, "0")}` : JSON.stringify(c));

export class TracePlayer {
  /** hooks: srcLinks(where), koButton(kind, item), onDim(name), onHead(layer, head) */
  constructor(root, hooks) {
    this.root = root;
    this.hooks = hooks;
    this.stage = 0;
    this.timer = null;
    this.data = null;
    this.build();
  }

  build() {
    const bar = el("div", "tr-bar");
    const btn = (label, tip, fn) => { const b = el("button", "", label); b.dataset.tip = tip; b.addEventListener("click", fn); bar.append(b); return b; };
    btn("⏮", "The embedding: where the byte enters.", () => this.go(0));
    btn("◀", "The previous computation.", () => this.step(-1));
    this.playBtn = btn("▶ play", "Play the pass from here; press again to pause.", () => this.toggle());
    btn("▶|", "The next computation.", () => this.step(1));
    btn("⏭", "The logits: what the byte's pass decides.", () => this.go(this.last()));
    this.range = el("input");
    Object.assign(this.range, { type: "range", min: 0, max: 0, value: 0 });
    this.range.setAttribute("aria-label", "Computation");
    this.range.addEventListener("input", () => { this.pause(); this.go(Number(this.range.value)); });
    this.speed = el("select");
    for (const k of Object.keys(SPEEDS)) this.speed.append(new Option(k, k));
    this.speed.value = "normal";
    this.speed.addEventListener("change", () => { if (this.timer) { this.pause(); this.play(); } });
    const quiet = el("label", "small");
    this.quiet = el("input");
    this.quiet.type = "checkbox";
    this.quiet.checked = true;
    quiet.append(this.quiet, " skip quiet");
    quiet.dataset.tip = "Step and play over the computations that change nothing for this byte.";
    this.changedOnly = el("input");
    this.changedOnly.type = "checkbox";
    const co = el("label", "small");
    co.append(this.changedOnly, " only what changed");
    co.dataset.tip = "List only the residual dims the current computation changed.";
    this.changedOnly.addEventListener("change", () => this.render());
    this.view = el("select");
    for (const [k, t] of Object.entries(VIEWS)) this.view.append(new Option(t, k));
    try { const v = localStorage.getItem(VIEW_KEY); if (v in VIEWS) this.view.value = v; } catch { /* private mode */ }
    this.view.dataset.tip = "Show the pass as a picture of the model, as lists (residual stream and what each step did), or both side by side.";
    this.view.addEventListener("change", () => {
      try { localStorage.setItem(VIEW_KEY, this.view.value); } catch { /* private mode */ }
      this.applyView();
    });
    bar.append(this.range, this.speed, quiet, co, this.view);
    this.caption = el("div", "tr-caption");
    const body = el("div", "tr-body");
    this.dims = el("div", "tr-dims scroll");
    this.dims.id = "trace-dims";
    this.dims.dataset.resize = "the traced residual stream";
    this.detail = el("div", "tr-detail");
    const left = el("div", "tr-left"); // holds the list and the resize grip setupResizers adds after it
    left.append(this.dims);
    this.pdWrap = el("div", "tr-diagram");
    const pdScroll = el("div", "pd-scroll");
    pdScroll.id = "trace-diagram";
    pdScroll.dataset.resize = "the pass diagram";
    const pdRoot = el("div", "pd-root");
    pdScroll.append(pdRoot);
    this.pdWrap.append(pdScroll);
    this.lists = [left, this.detail];
    body.append(this.pdWrap, left, this.detail);
    this.body = body;
    this.root.replaceChildren(bar, this.caption, body);
    this.diagram = new PassDiagram(pdRoot, {
      onStage: (st) => { this.pause(); this.focus = null; this.go(st); },
      onItem: (kind, layer, i) => { // play that computation and show the part in the details
        this.pause();
        this.focus = { kind, layer, i };
        this.go(kind === "heads" ? 1 + 2 * layer : 2 + 2 * layer);
      },
    });
    this.applyView();
  }

  applyView() {
    const v = this.view.value;
    this.pdWrap.hidden = v === "lists";
    for (const x of this.lists) x.hidden = v === "diagram";
    this.body.dataset.view = v;
  }

  last() { return this.data ? this.data.labels.length : 0; } // the logits stage, after the last sublayer

  /** New position data (the main step slider moved, or a new run). The stage is kept. */
  load(p, run, info) {
    const labels = run.snapshots;
    const used = info.dims.map((d, i) => ({ ...d, i, vals: p.values[i] })).filter((d) => d.vals.some((v) => Math.abs(v) > TOL));
    const scale = used.map((d) => Math.max(...d.vals.map(Math.abs)) || 1);
    this.readout = new Set(info.readout || []);
    this.data = { p, run, info, labels, used, scale, q: p.q };
    if (this.modelId !== info.model_id) { this.modelId = info.model_id; this.diagram.setModel(info, labels); }
    const changed = labels.map((_, st) => st > 0 && used.some((d) => Math.abs(d.vals[st] - d.vals[st - 1]) > TOL));
    this.diagram.setActive({
      heads: new Set(p.heads.filter((h) => Object.keys(h.wrote).length).map((h) => `${h.layer}.${h.head}`)),
      units: new Set(p.units.map((u) => `${u.layer}.${u.unit}`)),
      changed, logits: p.logits,
    });
    this.range.max = String(this.last());
    // rows built once per position; each stage only updates them
    this.dims.replaceChildren();
    this.rows = [];
    let cat = null;
    used.forEach((d, k) => {
      if (d.cat !== cat) { cat = d.cat; this.dims.append(el("div", "tr-group", cat)); }
      const row = el("div", "tr-row");
      const name = el("span", "tr-name", d.name);
      const val = el("span", "tr-val");
      const barWrap = el("span", "tr-barwrap");
      const fill = el("span", "tr-fill");
      barWrap.append(fill);
      row.append(name, val, barWrap);
      if (this.readout.has(d.name)) {
        row.classList.add("readout");
        name.dataset.tip = "The unembedding reads this dim: it is one of the values the next byte's score is computed from.";
      }
      row.dataset.tip = `${d.name}: ${d.doc || ""}\n(click to open it in the Residual stream panel)`;
      row.addEventListener("click", () => this.hooks.onDim?.(d.name));
      this.dims.append(row);
      this.rows.push({ row, val, fill, k });
    });
    this.stage = Math.min(this.stage, this.last());
    this.render();
    log(`position ${p.q}: ${used.length} dims used, ${labels.length} sublayers`);
  }

  /** Did stage s change anything (s >= 1)? */
  changed(s) {
    if (s === 0 || s >= this.last()) return true;
    return this.data.used.some((d) => Math.abs(d.vals[s] - d.vals[s - 1]) > TOL);
  }

  go(s) {
    if (!this.data) return;
    this.stage = Math.max(0, Math.min(this.last(), s));
    this.render();
  }

  step(dir) {
    if (!this.data) return;
    let s = this.stage + dir;
    while (this.quiet.checked && s > 0 && s < this.last() && !this.changed(s)) s += dir;
    this.go(s);
    return this.stage;
  }

  toggle() { if (this.timer) this.pause(); else this.play(); }

  play() {
    if (!this.data) return;
    if (this.stage >= this.last()) this.go(0);
    this.playBtn.textContent = "⏸ pause";
    this.timer = setInterval(() => {
      if (this.stage >= this.last()) { this.pause(); return; }
      this.step(1);
    }, SPEEDS[this.speed.value]);
    log("play from", this.stage);
  }

  pause() {
    clearInterval(this.timer);
    this.timer = null;
    this.playBtn.textContent = "▶ play";
  }

  render() {
    const D = this.data;
    if (!D) return;
    const s = this.stage, endStage = this.last();
    const snap = Math.min(s, endStage - 1); // the logits stage shows the final residual
    this.range.value = String(s);
    this.diagram.setStage(s);
    const dla = D.p.dla?.[this.dlaPick || 0];
    this.diagram.setAttribution(s === this.last() && dla ? dla.by_part : null);
    const label = s === endStage ? "logits" : D.labels[s];
    const ch = D.run.chars;
    this.caption.textContent = `position ${D.q} ${JSON.stringify(showChar(ch[D.q]))} → writes ${JSON.stringify(showChar(ch[D.q + 1]))} · ` +
      `${label} (${s + 1} / ${endStage + 1})`;
    // the residual column
    for (const { row, val, fill, k } of this.rows) {
      const d = D.used[k], v = d.vals[snap], before = s > 0 && s < endStage ? d.vals[s - 1] : v;
      const changed = s > 0 && s < endStage && Math.abs(v - before) > TOL;
      row.classList.toggle("changed", changed);
      row.hidden = this.changedOnly.checked && !changed;
      val.textContent = changed ? `${fmt(before)} → ${fmt(v)}` : fmt(v);
      fill.style.width = `${Math.min(100, (100 * Math.abs(v)) / D.scale[k])}%`;
      fill.classList.toggle("neg", v < 0);
    }
    this.renderDetail(s);
  }

  renderDetail(s) {
    const D = this.data, H = this.hooks, endStage = this.last();
    const box = this.detail;
    box.replaceChildren();
    if (s === 0) {
      const set = D.used.filter((d) => Math.abs(d.vals[0]) > TOL);
      box.append(el("div", "tr-title", "Embedding"),
        el("div", "muted", `The byte ${JSON.stringify(showChar(D.run.chars[D.q]))} enters as its embedding row, plus its position (pos ${D.q}, pos2 ${D.q * D.q}):`),
        el("div", "tr-list", set.map((d) => `${d.name}=${fmt(d.vals[0])}`).join(" · ")));
      return;
    }
    if (s === endStage) {
      box.append(el("div", "tr-title", "Logits: the next byte"),
        el("div", "muted", "The unembedding reads the final residual stream; the host writes the highest-scoring byte."));
      const top = D.p.logits, mx = Math.max(...top.map(([, v]) => Math.abs(v)), 1e-9);
      top.forEach(([b, v], i) => {
        const r = el("div", "tr-logit" + (i === 0 ? " win" : ""));
        const bw = el("span", "tr-barwrap"), f = el("span", "tr-fill");
        f.style.width = `${Math.max(0, (100 * v) / mx)}%`;
        bw.append(f);
        r.append(el("span", "tr-name", byteText(b)), el("span", "tr-val", fmt(v)), bw);
        box.append(r);
      });
      this.renderAttribution(box);
      return;
    }
    const label = D.labels[s];
    const layer = Number(/^L(\d+)/.exec(label)[1]) - 1;
    const spec = D.info.layers[layer];
    const changedDims = D.used.filter((d) => Math.abs(d.vals[s] - d.vals[s - 1]) > TOL);
    if (label.endsWith("attn")) {
      const wrote = D.p.heads.filter((h) => h.layer === layer && Object.keys(h.wrote).length);
      box.append(el("div", "tr-title", `Layer ${layer + 1} attention · ${spec.heads.length} heads, ${wrote.length} wrote something`));
      for (const h of wrote) {
        const hs = spec.heads[h.head], [k, w] = h.top[0];
        const item = el("div", "tr-item");
        item.append(el("span", "tr-item-name", hs.name),
          el("span", "", `← ${k} ${JSON.stringify(showChar(D.run.chars[k]))} ${(w * 100).toFixed(w > 0.999 ? 0 : 1)}%`),
          el("span", "tr-wrote", Object.entries(h.wrote).map(([dd, v]) => `${dd} += ${fmt(v)}`).join(" · ")));
        const links = el("span", "tr-links");
        links.append(H.srcLinks(hs.src));
        if (H.onHead) { const b = el("button", "", "attention"); b.addEventListener("click", () => H.onHead(layer, h.head)); links.append(b); }
        if (H.koButton) links.append(H.koButton("heads", [layer, h.head]));
        item.append(links);
        item.dataset.tip = hs.doc;
        item.dataset.key = `heads.${layer}.${h.head}`;
        box.append(item);
      }
      if (!wrote.length) box.append(el("div", "muted", "No head wrote anything for this byte here."));
      this.focusNote(box, "heads", layer, wrote.map((h) => h.head), (i) => spec.heads[i]);
    } else {
      const fired = D.p.units.filter((u) => u.layer === layer);
      box.append(el("div", "tr-title", `Layer ${layer + 1} MLP · ${spec.units.length} units, ${fired.length} fired`));
      for (const u of fired) {
        const us = spec.units[u.unit];
        const item = el("div", "tr-item");
        item.append(el("span", "tr-item-name", `#${u.unit}`), el("span", "", `ReLU(${us.read}) = ${fmt(u.pre)}`),
          el("span", "tr-wrote", `→ ${us.write}`));
        const links = el("span", "tr-links");
        links.append(H.srcLinks(us.src));
        if (H.koButton) links.append(H.koButton("units", [layer, u.unit]));
        item.append(links);
        item.dataset.tip = us.doc;
        item.dataset.key = `units.${layer}.${u.unit}`;
        box.append(item);
      }
      if (!fired.length) box.append(el("div", "muted", "No unit fired for this byte here."));
      this.focusNote(box, "units", layer, fired.map((u) => u.unit), (i) => spec.units[i]);
    }
    box.append(el("div", "tr-sub", changedDims.length ? `changed: ${changedDims.map((d) => `${d.name} ${fmt(d.vals[s - 1])} → ${fmt(d.vals[s])}`).join(" · ")}`
      : "changed nothing in the residual stream for this byte"));
  }

  /** The part clicked in the diagram: outline it in the list, or say it was silent for this byte. */
  focusNote(box, kind, layer, listed, specOf) {
    const f = this.focus;
    if (!f || f.kind !== kind || f.layer !== layer) return;
    const key = `${kind}.${layer}.${f.i}`;
    let item = box.querySelector(`[data-key="${key}"]`);
    if (!item) {
      const sp = specOf(f.i);
      item = el("div", "tr-item");
      item.append(el("span", "tr-item-name", kind === "heads" ? sp.name : `#${f.i}`),
        el("span", "muted", kind === "heads" ? "wrote nothing for this byte" : `did not fire for this byte: ReLU(${sp.read}) = 0`));
      const links = el("span", "tr-links");
      links.append(this.hooks.srcLinks(sp.src));
      if (this.hooks.koButton) links.append(this.hooks.koButton(kind, [layer, f.i]));
      item.append(links);
      item.dataset.tip = sp.doc;
      box.insertBefore(item, box.children[1] || null);
    }
    item.classList.add("focus");
  }

  /** Direct logit attribution at the logits stage: a top byte's score as an exact sum of what
   *  each part wrote (by part) and of what each read dim holds (by dim). */
  renderAttribution(box) {
    const D = this.data, H = this.hooks, dla = D.p.dla || [];
    if (!dla.length) return;
    const pick = Math.min(this.dlaPick || 0, dla.length - 1);
    const e = dla[pick];
    const head = el("div", "tr-title tr-dla-title", "Why this score: direct logit attribution");
    head.dataset.tip = "The score is the final residual stream times the unembedding, and that stream is the sum of what " +
      "the embedding, every head and every unit wrote (no LayerNorm here, so the split is exact). Each line is one " +
      "part's direct share; they add up to the score. Parts that only feed later parts show up as 0 here: use knock-outs for those.";
    const tabs = el("div", "tr-dla-tabs");
    dla.forEach((x, i) => {
      const b = el("button", i === pick ? "on" : "", `${byteText(x.byte)} ${fmt(x.logit)}`);
      b.addEventListener("click", () => { this.dlaPick = i; this.render(); });
      tabs.append(b);
    });
    box.append(head, tabs);
    const parts = e.by_part, mx = Math.max(...parts.map((x) => Math.abs(x[3])), 1e-9);
    const list = el("div", "tr-dla");
    for (const [kind, layer, idx, c] of parts) {
      const r = el("div", "tr-dla-row " + (c >= 0 ? "pos" : "neg"));
      let name, src = null;
      if (kind === "embed") name = "embedding + position";
      else if (kind === "head") { const h = D.info.layers[layer].heads[idx]; name = `L${layer + 1} head ${h.name}`; src = h.src; }
      else { const u = D.info.layers[layer].units[idx]; name = `L${layer + 1} unit #${idx} → ${u.write}`; src = u.src; }
      const bw = el("span", "tr-barwrap"), f = el("span", "tr-fill");
      f.style.width = `${(100 * Math.abs(c)) / mx}%`;
      bw.append(f);
      r.append(el("span", "tr-name", name), el("span", "tr-val", `${c >= 0 ? "+" : ""}${fmt(c)}`), bw);
      if (src) { const l = el("span", "tr-links"); l.append(H.srcLinks(src)); r.append(l); }
      if (kind !== "embed") {
        r.dataset.tip = "Click to play the computation this part belongs to.";
        r.addEventListener("click", (ev) => {
          if (ev.target.closest("a")) return;
          this.focus = { kind: kind === "head" ? "heads" : "units", layer, i: idx };
          this.go(kind === "head" ? 1 + 2 * layer : 2 + 2 * layer);
        });
      }
      list.append(r);
    }
    const sum = parts.reduce((t, x) => t + x[3], 0);
    list.append(el("div", "tr-dla-sum", `sum ${fmt(sum)} = score ${fmt(e.logit)} (${parts.length} part${parts.length === 1 ? "" : "s"})`));
    box.append(el("div", "tr-sub", "by part"), list);
    const dims = el("div", "tr-dla");
    for (const [name, v, w, c] of e.by_dim) {
      const r = el("div", "tr-dla-row tr-dla-dim " + (c >= 0 ? "pos" : "neg"));
      r.append(el("span", "tr-name", name), el("span", "tr-val", `${fmt(v)} × ${fmt(w)} = ${c >= 0 ? "+" : ""}${fmt(c)}`));
      r.addEventListener("click", () => H.onDim?.(name));
      r.dataset.tip = "Its final value times its unembedding weight for this byte. Click to open the dim.";
      dims.append(r);
    }
    if (!e.by_dim.length) dims.append(el("div", "muted", "no dim the unembedding reads is set for this byte"));
    box.append(el("div", "tr-sub", "by dim (final value × unembedding weight)"), dims);
  }
}
