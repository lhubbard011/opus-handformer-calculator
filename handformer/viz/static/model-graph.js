// Model graph: the forward pass as a torchviz-style diagram (weights in blue with their shapes,
// operations in grey, the output in green), true to the model: it is built from the same head
// and unit specs the weights are built from (the /api/info model description), in the order the
// forward pass runs them:
//
//   ids -> embed + p·pos + p²·pos2 -> block 1 ... block L -> unembedding -> logits -> argmax
//   block: x + attention(x) (heads in parallel, summed = concat · W_O), then x + MLP(x)
//
// Every block expands into its attention and MLP sublayers, those into their heads and units
// grouped by the code module that wrote them, and each head or unit into its formula and source
// line. Lists render only when opened, so a 1,000-unit model stays light. Each head and unit has
// the page's own knock-out button (hooks.koButton); refreshKnockouts keeps them in step with the
// current knockout set after every run.

const log = (...a) => console.log("[model-graph]", ...a);

const moduleOf = (where) => (where ? where.split(":")[0].split("/").pop() : "(no source)");
const lineOf = (where) => (where ? Number(where.slice(where.lastIndexOf(":") + 1)) : null);
/** "115–202" (or "184"): the source lines a group of heads/units comes from. */
function lineRange(list) {
  const ls = list.map(([it]) => lineOf(it.src)).filter((x) => x !== null);
  if (!ls.length) return "";
  const lo = Math.min(...ls), hi = Math.max(...ls);
  return lo === hi ? `${lo}` : `${lo}–${hi}`;
}

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

/** A weight (blue, with its shape), an operation (grey) or the output (green). */
function node(kind, label, sub, tip) {
  const n = el("div", `mg-node mg-${kind}`);
  n.append(el("span", "mg-label", label));
  if (sub) n.append(el("span", "mg-sub", sub));
  if (tip) n.dataset.tip = tip;
  return n;
}

/** One step of the flow: weights side by side feeding an operation, with an arrow into it. */
function step(params, op) {
  const row = el("div", "mg-step");
  if (params.length) {
    const ps = el("div", "mg-params");
    ps.append(...params);
    row.append(ps);
  }
  row.append(op);
  return row;
}

/** <details> whose body is built the first time it opens. */
function lazy(cls, summary, build) {
  const d = el("details", cls);
  const s = el("summary");
  s.append(...[].concat(summary));
  d.append(s);
  let built = false;
  d.addEventListener("toggle", () => {
    if (!d.open || built) return;
    built = true;
    d.append(build());
  });
  return d;
}

/** Heads or units grouped by the code module that wrote them, in build order inside a group. */
function byModule(items) {
  const groups = new Map();
  items.forEach((it, i) => {
    const m = moduleOf(it.src);
    if (!groups.has(m)) groups.set(m, []);
    groups.get(m).push([it, i]);
  });
  return [...groups.entries()];
}

function chips(items) {
  const span = el("span", "mg-chips");
  for (const [m, list] of byModule(items)) {
    const c = el("span", "mg-chip", `${m.replace(/\.py$/, "")} ${list.length} · ${lineRange(list)}`);
    c.dataset.tip = `${list.length} from ${m}, lines ${lineRange(list)}`;
    span.append(c);
  }
  return span;
}

export function renderModelGraph(root, info, hooks = {}) {
  const t0 = performance.now();
  const d = info.d_model;
  const srcLinks = hooks.srcLinks || (() => document.createTextNode(""));
  root.replaceChildren();
  const flow = el("div", "mg-flow");

  flow.append(node("op", "ids", "the bytes so far (t)", "The input: one byte per position, the prompt and everything written so far."));
  flow.append(step([
    node("param", "embedding", `(256, ${d})`, "One row per byte value: the byte's input features (every byte sets 'one' = 1, the model's only bias)."),
    node("param", "pos, pos²", `(2, ${d})`, "Position p and p² added into the dims 'pos' and 'pos2': the only positional encoding (p² makes exact lookups possible)."),
  ], node("op", "x = embed(ids) + p·pos + p²·pos2", `residual stream (t, ${d})`)));

  let heads = 0, units = 0;
  for (const L of info.layers) {
    heads += L.heads.length;
    units += L.units.length;
    const H = L.heads.length, U = L.units.length;
    const summary = [el("span", "mg-block-name", `Block ${L.index + 1}`),
      el("span", "mg-muted", [H ? `attention ${H} head${H > 1 ? "s" : ""}` : "", U ? `MLP ${U} unit${U > 1 ? "s" : ""}` : ""]
        .filter(Boolean).join(" · ") || "empty"),
      chips([...L.heads, ...L.units])];
    flow.append(lazy("mg-block", summary, () => block(L, d, srcLinks, hooks)));
  }

  flow.append(step([node("param", "unembedding", `(${d}, 256)`, "Reads named dims into one score per byte value (lm_head).")],
                   node("op", "logits = x · W_U", "(t, 256)")));
  flow.append(node("out", "argmax", "the next byte; stop at '\\n'", "Greedy decoding: the host takes the highest score and feeds it back. That is all the host does."));

  const bar = el("div", "mg-bar");
  const open = el("button", "", "expand blocks"), close = el("button", "", "collapse all");
  open.addEventListener("click", () => flow.querySelectorAll("details.mg-block").forEach((x) => { x.open = true; }));
  close.addEventListener("click", () => flow.querySelectorAll("details").forEach((x) => { x.open = false; }));
  bar.append(el("span", "mg-muted", `${info.layers.length} blocks · ${heads} heads · ${units} MLP units · d_model ${d} · ` +
    `${info.nonzero_params} nonzero weights`), open, close);
  root.append(bar, flow);
  log(`drew ${info.layers.length} blocks (${heads} heads, ${units} units) in ${(performance.now() - t0).toFixed(0)}ms`);
}

function block(L, d, srcLinks, hooks) {
  const body = el("div", "mg-inner");
  const H = L.heads.length, U = L.units.length;
  if (H) {
    const dk = L.heads.map((h) => h.query.length), dv = L.heads.map((h) => Object.keys(h.value).length);
    const sum = (a) => a.reduce((x, y) => x + y, 0);
    const attn = lazy("mg-sub-block", [el("span", "mg-block-name", `Attention · ${H} head${H > 1 ? "s" : ""} in parallel`),
      chips(L.heads)], () => {
      const inner = el("div", "mg-inner");
      inner.append(
        step([node("param", "W_Q", `(${d}, ${sum(dk)})`, `Each head's query projection, side by side (d_k per head: ${[...new Set(dk)].join(", ")}).`),
              node("param", "W_K", `(${d}, ${sum(dk)})`, "Each head's key projection."),
              node("param", "W_V", `(${d}, ${sum(dv)})`, `Each head's value projection (d_v per head: ${[...new Set(dv)].join(", ")}).`)],
             node("op", "scores = Q·Kᵀ per head, causal", "(heads, t, t)")),
        node("op", "softmax over the keys", "almost always all on one key: exact lookups"),
        step([node("param", "W_O", `(${sum(dv)}, ${d})`, "Each head's output projection: writes its named dims.")],
             node("op", "Σ heads (softmax · V) · W_O", "= concat(heads) · W_O")),
      );
      for (const [m, list] of byModule(L.heads)) {
        inner.append(lazy("mg-group", [el("span", "mg-mod", m), el("span", "mg-muted", `${list.length} head${list.length > 1 ? "s" : ""} · lines ${lineRange(list)}`)], () => {
          const g = el("div", "mg-items");
          for (const [h, i] of list) g.append(headItem(L, h, i, srcLinks, hooks));
          return g;
        }));
      }
      return inner;
    });
    body.append(attn, node("op", "x = x + attention(x)", "residual add"));
  }
  if (U) {
    const mlp = lazy("mg-sub-block", [el("span", "mg-block-name", `MLP · ${U} unit${U > 1 ? "s" : ""}`), chips(L.units)], () => {
      const inner = el("div", "mg-inner");
      inner.append(
        step([node("param", "W_in", `(${d}, ${U})`, "Each unit's read: a column of named coefficients (fc1).")],
             node("op", "ReLU(x · W_in)", `(t, ${U})`)),
        step([node("param", "W_out", `(${U}, ${d})`, "Each unit's write: a row of named coefficients (fc2).")],
             node("op", "· W_out", `(t, ${d})`)),
      );
      for (const [m, list] of byModule(L.units)) {
        inner.append(lazy("mg-group", [el("span", "mg-mod", m), el("span", "mg-muted", `${list.length} unit${list.length > 1 ? "s" : ""} · lines ${lineRange(list)}`)], () => {
          const g = el("div", "mg-items");
          for (const [u, i] of list) g.append(unitItem(L, u, i, srcLinks, hooks));
          return g;
        }));
      }
      return inner;
    });
    body.append(mlp, node("op", "x = x + MLP(x)", "residual add"));
  }
  if (!H && !U) body.append(el("div", "mg-muted", "nothing in this block"));
  return body;
}

/** A slot for the page's knock-out button of one head or unit (filled now and after every run). */
function koSlot(hooks, kind, item, row) {
  const slot = el("span", "mg-ko");
  if (!hooks.koButton) return slot;
  slot.dataset.kind = kind;
  slot.dataset.item = JSON.stringify(item);
  slot.append(hooks.koButton(kind, item));
  row.classList.toggle("mg-off", !!hooks.koHas?.(kind, item));
  return slot;
}

/** Re-make every knock-out button and mark what is knocked out (call after the set changes). */
export function refreshKnockouts(root, hooks) {
  for (const slot of root.querySelectorAll(".mg-ko[data-kind]")) {
    const kind = slot.dataset.kind, item = JSON.parse(slot.dataset.item);
    slot.replaceChildren(hooks.koButton(kind, item));
    slot.closest(".mg-item")?.classList.toggle("mg-off", hooks.koHas(kind, item));
  }
}

function headItem(L, h, i, srcLinks, hooks) {
  const it = el("details", "mg-item");
  const s = el("summary");
  s.append(el("span", "mg-item-name", `head ${i} · ${h.name}`), srcLinks(h.src), koSlot(hooks, "heads", [L.index, i], it),
           el("span", "mg-muted", `writes ${h.writes.join(", ") || "nothing"}`));
  it.append(s);
  const body = el("div", "mg-formula");
  const score = h.query.map((q, k) => `(${q})·(${h.key[k]})`).join(" + ");
  body.append(el("div", "", `score(q, k) = ${score}`),
              el("div", "", `value: ${Object.entries(h.value).map(([src, dst]) => `${src} → ${dst}`).join("; ")}`),
              el("div", "mg-muted", h.doc));
  const links = el("div", "mg-links");
  if (hooks.onHead) {
    const b = el("button", "", "show its attention");
    b.addEventListener("click", () => hooks.onHead(L.index, i));
    links.append(b);
  }
  body.append(links);
  it.append(body);
  return it;
}

function unitItem(L, u, i, srcLinks, hooks) {
  const it = el("details", "mg-item");
  const s = el("summary");
  s.append(el("span", "mg-item-name", `unit ${i} → ${u.write}`), srcLinks(u.src), koSlot(hooks, "units", [L.index, i], it),
           el("span", "mg-muted", u.label ? u.label.slice(0, 90) : ""));
  it.append(s);
  const body = el("div", "mg-formula");
  body.append(el("div", "", `ReLU(${u.read}) → ${u.write}`), el("div", "mg-muted", u.doc));
  it.append(body);
  return it;
}
