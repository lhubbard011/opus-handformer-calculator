// The unembedding as the matrix product it is: logits = x_final · W_U. The Token pass shows it
// as its own computation, between the last MLP and the logits: the readout dims' final values,
// the block of W_U that is not zero (from /api/info "unembed"), every product x[d]·W_U[d, b]
// that is not zero, and each byte's score as the sum down its column. Display only: the page
// recomputes nothing the model does not; the sums are checked against the logits it was sent.

const log = (...a) => console.log("[unembed]", ...a);
const TOL = 1e-9;

// display only: within 1e-4 of an integer shows the integer (same rule as trace.js)
const fmt = (v) => (Math.abs(v - Math.round(v)) < 1e-4 ? String(Math.round(v)) : String(Number(v.toFixed(4))));
const byteText = (c) => (c === "\n" ? "⏎" : c.charCodeAt(0) < 32 || c.charCodeAt(0) === 127
  ? `\\x${c.charCodeAt(0).toString(16).padStart(2, "0")}` : c);

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

/** Why this works, for the hover on the diagram row and on the matrix (filled in per model). */
export function unembedTip(info) {
  const ue = info.unembed, n = info.readout.length;
  const single = ue && countPerRow(ue).every((c) => c === 1);
  return "How the next byte's scores are made: logit[b] = x_final · W_U[:, b]. One matrix multiply, nothing " +
    "else: no final LayerNorm, no bias.\n\n" +
    "x_final is a sum: the embedding plus what every head and every MLP unit added. W_U is linear, so it " +
    "distributes over that sum: each part's share of a score is its own output · W_U[:, b], and the shares " +
    "add up exactly (the attribution at the logits step).\n\n" +
    (ue ? `Only the part of W_U that is not zero is drawn: ${n} of its ${ue.d_model} rows (the readout dims) and ` +
      `${ue.bytes.length} of its ${ue.vocab} columns (the bytes that can score at all). ` : "") +
    "A dim with no row here never touches a score directly; it matters only by feeding a later layer." +
    (single ? "\n\nIn this model every readout dim has exactly one weight: W_U is a routing table, each dim votes for one byte." : "");
}

function countPerRow(ue) {
  const c = [];
  for (const [r] of ue.w) c[r] = (c[r] || 0) + 1;
  return c;
}

/** The readout dims' final values for the position in `p` (index i = info.readout[i]). */
export function readoutValues(info, p, nSnap) {
  const ix = new Map(info.dims.map((d, i) => [d.name, i]));
  return info.readout.map((name) => p.values[ix.get(name)][nSnap - 1]);
}

/**
 * Draw the product into `box`. hooks: onDim(name) (a row was clicked), onByte(byte) (a column
 * was clicked). `tops`: the bytes the attribution covers, outlined.
 */
export function renderUnembed(box, info, p, nSnap, tops, hooks) {
  const ue = info.unembed;
  const tip = unembedTip(info);
  const title = el("div", "tr-title ue-title", "Unembedding: logits = x_final · W_U");
  title.dataset.tip = tip;
  box.append(title);
  if (!ue) { box.append(el("div", "muted", "This model description has no unembedding block (an older server?).")); return; }
  const x = readoutValues(info, p, nSnap);
  const rows = info.readout, bytes = ue.bytes;
  const set = x.filter((v) => Math.abs(v) > TOL).length;
  const lead = el("div", "muted ue-lead",
    `${rows.length} of ${ue.d_model} dims have a row in W_U and ${bytes.length} of ${ue.vocab} bytes have a column; ` +
    `${set} of those dims are set for this byte. Each lit cell is x × weight; each score is the sum down its column. ` +
    "Hover for why this is exact.");
  lead.dataset.tip = tip;
  box.append(lead);

  // the block, as a lookup and per-column sums
  const cell = new Map(), sums = new Float64Array(bytes.length);
  for (const [r, k, w] of ue.w) { cell.set(`${r}.${k}`, w); sums[k] += x[r] * w; }
  let best = 0;
  for (let k = 1; k < sums.length; k++) if (sums[k] > sums[best]) best = k;
  // the page was sent the real logits: the column sums must be them (bytes outside the block score 0)
  const sumOf = new Map(bytes.map((b, k) => [b, sums[k]]));
  const bad = (p.logits || []).filter(([b, v]) => Math.abs((sumOf.get(b) || 0) - v) > 1e-6 * Math.max(1, Math.abs(v)));
  if (bad.length) log("column sums differ from the logits at", p.q, JSON.stringify(bad));
  else log(`position ${p.q}: ${set} readout dims set, column sums match the ${(p.logits || []).length} top logits`);

  const wrap = el("div", "ue-wrap");
  wrap.dataset.resize = "the unembedding matrix";
  const t = el("table", "ue-grid");
  const head = el("tr");
  const corner = el("th", "ue-corner", "dim · x");
  corner.dataset.tip = "Each row: a readout dim and its final value x for this byte.";
  head.append(corner);
  const topSet = new Set(tops);
  bytes.forEach((b, k) => {
    const th = el("th", "ue-byte" + (k === best ? " win" : "") + (topSet.has(b) ? " top" : ""), byteText(b));
    th.dataset.tip = `byte ${JSON.stringify(b)}: its column of W_U.` + (topSet.has(b) ? "\n(click for its attribution at the logits step)" : "");
    th.addEventListener("click", () => hooks.onByte?.(b));
    head.append(th);
  });
  const thead = el("thead");
  thead.append(head);
  const tbody = el("tbody");
  rows.forEach((name, r) => {
    const tr = el("tr", Math.abs(x[r]) > TOL ? "set" : "");
    const th = el("th", "ue-dim");
    const line = el("div", "ue-dimrow");
    line.append(el("span", "ue-name", name), el("span", "ue-x", fmt(x[r])));
    th.append(line);
    th.dataset.tip = `${name} = ${fmt(x[r])} at the end of the pass.` + (Math.abs(x[r]) > TOL ? "" : " Zero: its whole row adds nothing.") +
      "\n(click to open it in the Residual stream panel)";
    th.addEventListener("click", () => hooks.onDim?.(name));
    tr.append(th);
    bytes.forEach((b, k) => {
      const w = cell.get(`${r}.${k}`);
      const td = el("td", "ue-cell" + (k === best ? " win" : ""));
      if (w !== undefined) {
        const c = x[r] * w, lit = Math.abs(c) > TOL;
        td.classList.add(lit ? "lit" : "w", ...(lit ? [c > 0 ? "pos" : "neg"] : []));
        td.textContent = lit ? fmt(c) : "·";
        td.dataset.tip = `${name} × W_U[${name}, ${JSON.stringify(b)}] = ${fmt(x[r])} × ${fmt(w)} = ${fmt(c)}\n` +
          (lit ? `adds ${fmt(c)} to the score of ${JSON.stringify(b)}` : `the weight is there, but ${name} is 0 for this byte`);
      }
      tr.append(td);
    });
    tbody.append(tr);
  });
  const foot = el("tr", "ue-sums");
  const ft = el("th", "ue-dim", "logit = Σ");
  ft.dataset.tip = "Each byte's score: the sum down its column, x_final · W_U[:, b]. The highest one is written.";
  foot.append(ft);
  bytes.forEach((b, k) => {
    const td = el("td", "ue-sum" + (k === best ? " win" : "") + (Math.abs(sums[k]) > TOL ? " nz" : ""), fmt(sums[k]));
    td.dataset.tip = `score of ${JSON.stringify(b)} = ${fmt(sums[k])}` + (k === best ? " (the highest: the byte written)" : "");
    foot.append(td);
  });
  const tfoot = el("tfoot");
  tfoot.append(foot);
  t.append(thead, tbody, tfoot);
  wrap.append(t);
  box.append(wrap);
  if (bad.length) box.append(el("div", "tr-sub ue-bad", `these column sums differ from the logits the model produced: ${bad.map(([b]) => JSON.stringify(b)).join(", ")}`));
  // centre the matrix's own box (never the page) on what decides the byte: a lit cell in the
  // winning column, else any lit cell, else the winning column's header
  requestAnimationFrame(() => {
    const target = t.querySelector("td.ue-cell.lit.win") || t.querySelector("td.ue-cell.lit") || t.querySelector("th.ue-byte.win");
    if (!target) return;
    const wr = wrap.getBoundingClientRect(), r = target.getBoundingClientRect();
    wrap.scrollLeft += r.left + r.width / 2 - (wr.left + wr.width / 2);
    if (target.tagName === "TD") wrap.scrollTop += r.top + r.height / 2 - (wr.top + wr.height / 2);
  });
}
