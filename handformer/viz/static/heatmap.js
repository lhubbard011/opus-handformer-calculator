// Canvas heatmaps with step-aware overlays (current position, causal future dimming),
// row labels grouped by circuit, hover readout and click-to-navigate.
import { matrixBitmap, setupCanvas, lutCss } from "./render.js";

const GUTTER = 128; // px reserved for row labels
// Touch screens: cells at least this big (the map scrolls sideways inside its panel).
const TOUCH_COL = 8, TOUCH_ROW = 5, TOUCH_MAX_W = 4000;
export const isTouch = () => matchMedia("(pointer: coarse)").matches;
// Row names: every row gets at least LABEL_ROW px so its name fits (the map scrolls inside its
// resizable box); off, the rows shrink until the whole map fits FIT_H px and only the circuit
// groups are named.
const LABEL_ROW = 11, LABEL_MIN = 8, FIT_H = 560;
let rowNames = true;
export function setRowNames(on) { rowNames = !!on; }

// spec: { key, rows, cols, get(r,c) -> raw, norm(raw) -> [0,1], lut, rowLabels[], groups[{row,label}],
//         colLabel(c), valueText(r,c), rowTip(r) (hover on a row label), onClick(r,c), square,
//         pointers[r] (column to trace per row: an attention head's argmax), colChars[c] (axis characters),
//         markCol (a dashed vertical line at that column) }
// view: { q } current query position (column, or row for attention)
export function drawHeatmap(canvas, spec, view) {
  const wrap = canvas.parentElement;
  const touch = isTouch();
  let cssW = Math.max(240, wrap.clientWidth - 2);
  if (touch && spec.cols) cssW = Math.max(cssW, GUTTER + Math.min(TOUCH_MAX_W, spec.cols * TOUCH_COL));
  const plotW = cssW - GUTTER;
  const fit = Math.min(12, FIT_H / Math.max(1, spec.rows));
  const rowH = spec.square ? plotW / spec.cols
    : Math.max(touch ? TOUCH_ROW : 3, rowNames && spec.rowLabels ? Math.max(LABEL_ROW, fit) : fit);
  const plotH = spec.square ? plotW : rowH * spec.rows;
  const colW0 = plotW / Math.max(1, spec.cols);
  const charAxis = spec.colChars && colW0 >= 9; // room for one character per column
  const cssH = Math.max(40, plotH + (charAxis ? 32 : 18));
  const ctx = setupCanvas(canvas, cssW, cssH);
  ctx.fillStyle = "#0d0f12";
  ctx.fillRect(0, 0, cssW, cssH);
  if (!spec.rows || !spec.cols) {
    ctx.fillStyle = "#7d8696";
    ctx.fillText("nothing to show for this step", 8, 20);
    return;
  }

  if (!canvas._bitmap || canvas._bitmapKey !== spec.key) {
    canvas._bitmap = matrixBitmap(spec.rows, spec.cols, spec.get, spec.norm, spec.lut);
    canvas._bitmapKey = spec.key;
  }
  const colW = plotW / spec.cols;
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(canvas._bitmap, GUTTER, 0, plotW, plotH);

  // pointer trace: where each row's attention peaks, drawn as a staircase so jumps stand out
  if (spec.pointers) {
    ctx.strokeStyle = "#5ec8e5";
    ctx.lineWidth = Math.max(1, Math.min(2, colW / 4));
    ctx.beginPath();
    spec.pointers.forEach((k, r) => {
      const x = GUTTER + (k + 0.5) * colW;
      if (r === 0) ctx.moveTo(x, r * rowH); else ctx.lineTo(x, r * rowH);
      ctx.lineTo(x, (r + 1) * rowH);
    });
    ctx.stroke();
  }

  // causal view: everything after the current step has not been computed yet
  const q = view.q;
  if (q !== undefined) {
    // view.noDim: keep the current-step outline but don't dim later positions (difference view)
    ctx.fillStyle = view.noDim ? "rgba(0,0,0,0)" : spec.square ? "rgba(13,15,18,0.55)" : "rgba(13,15,18,0.72)";
    if (spec.square) {
      ctx.fillRect(GUTTER, (q + 1) * rowH, plotW, plotH - (q + 1) * rowH);
      ctx.strokeStyle = "#f2a541";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(GUTTER, q * rowH, plotW, rowH);
    } else {
      const x0 = GUTTER + (q + 1) * colW;
      ctx.fillRect(x0, 0, cssW - x0, plotH);
      ctx.strokeStyle = "#f2a541";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(GUTTER + q * colW, 0, Math.max(colW, 1.5), plotH);
    }
  }

  // a marked column (the knockout difference view: where the two runs' text first differs)
  if (spec.markCol !== undefined && spec.markCol >= 0 && spec.markCol < spec.cols) {
    ctx.save();
    ctx.strokeStyle = "#e07bd0";
    ctx.setLineDash([4, 3]);
    ctx.lineWidth = 1.5;
    const x = GUTTER + spec.markCol * colW;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, plotH); ctx.stroke();
    ctx.restore();
  }

  // row labels and circuit group separators
  ctx.font = "10px ui-monospace, Menlo, monospace";
  ctx.textBaseline = "middle";
  if (rowH >= LABEL_MIN && spec.rowLabels) {
    ctx.fillStyle = "#aab2c0";
    spec.rowLabels.forEach((l, r) => ctx.fillText(l.slice(0, 19), 4, r * rowH + rowH / 2));
  }
  (spec.groups || []).forEach((g) => {
    ctx.strokeStyle = "#3a4150";
    ctx.beginPath();
    ctx.moveTo(0, g.row * rowH + 0.5);
    ctx.lineTo(cssW, g.row * rowH + 0.5);
    ctx.stroke();
    if (rowH < LABEL_MIN) {
      ctx.fillStyle = "#f2a541";
      ctx.fillText(g.label, 4, g.row * rowH + 7);
    }
  });
  // column ticks
  ctx.fillStyle = "#7d8696";
  ctx.textBaseline = "top";
  const step = Math.max(1, Math.ceil(30 / colW));
  if (charAxis) {
    ctx.textAlign = "center";
    ctx.fillStyle = "#d8dde6";
    ctx.font = "11px ui-monospace, Menlo, monospace";
    spec.colChars.forEach((ch, c) => ctx.fillText(ch, GUTTER + (c + 0.5) * colW, plotH + 3));
    ctx.textAlign = "start";
    ctx.fillStyle = "#7d8696";
    ctx.font = "10px ui-monospace, Menlo, monospace";
  }
  for (let c = 0; c < spec.cols; c += step) ctx.fillText(String(c), GUTTER + c * colW, plotH + (charAxis ? 17 : 3));

  canvas._geom = { spec, rowH, colW, plotW, plotH };
}

export function attachHeatmapEvents(canvas, tooltip) {
  const cell = (e) => {
    const g = canvas._geom;
    if (!g) return null;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left - GUTTER;
    const y = e.clientY - rect.top;
    if (y < 0 || y >= g.plotH || x >= g.plotW) return null;
    if (x < 0) return g.spec.rowTip ? [Math.floor(y / g.rowH), -1] : null; // row label gutter
    return [Math.floor(y / g.rowH), Math.floor(x / g.colW)];
  };
  const text = (spec, r, c) => (c < 0 ? spec.rowTip(r) : spec.valueText(r, c));
  // Touch: the first tap on a cell shows its readout, a second tap on the same cell
  // does what a mouse click does. Compatibility mouse events after a touch are ignored.
  let touchAt = 0, tapped = null;
  canvas.addEventListener("pointerdown", (e) => { if (e.pointerType !== "mouse") touchAt = Date.now(); });
  const touching = () => Date.now() - touchAt < 1000;
  canvas.addEventListener("mousemove", (e) => {
    if (touching()) return;
    const rc = cell(e);
    if (!rc) { tooltip.hidden = true; return; }
    const { spec } = canvas._geom;
    const [r, c] = rc;
    tooltip.hidden = false;
    tooltip.textContent = text(spec, r, c);
    canvas.style.cursor = c < 0 ? "help" : "crosshair";
    tooltip.style.left = `${e.clientX + 14}px`;
    tooltip.style.top = `${e.clientY + 10}px`;
  });
  canvas.addEventListener("mouseleave", () => { if (!touching()) tooltip.hidden = true; });
  canvas.addEventListener("click", (e) => {
    const rc = cell(e);
    if (touching()) {
      if (!rc) { tapped = null; hideSheet(tooltip); return; }
      const key = `${canvas._geom.spec.key}:${rc}`;
      if (rc[1] < 0) { tapped = null; showSheet(tooltip, text(canvas._geom.spec, ...rc)); return; }
      if (tapped !== key) {
        tapped = key;
        showSheet(tooltip, text(canvas._geom.spec, ...rc) + (canvas._geom.spec.onClick ? "\n\n(tap the same cell again to jump there)" : ""));
        return;
      }
      tapped = null;
      hideSheet(tooltip);
    }
    if (rc && rc[1] >= 0 && canvas._geom.spec.onClick) canvas._geom.spec.onClick(...rc);
  });
}

/** Tooltip text plus an optional orange section (the pointers on a byte, as drawn above it). */
export function setTip(tooltip, text, extra = "") {
  tooltip.textContent = text;
  if (extra) {
    const s = document.createElement("span");
    s.className = "tip-ptr";
    s.textContent = `\n\n${extra}`;
    tooltip.append(s);
  }
}

/** Touch tooltip: docked above the bottom step bar instead of following a cursor. */
export function showSheet(tooltip, text, extra = "") {
  setTip(tooltip, text, extra);
  tooltip.classList.add("sheet");
  tooltip.style.left = tooltip.style.top = "";
  tooltip.hidden = false;
}

export function hideSheet(tooltip) {
  tooltip.hidden = true;
  tooltip.classList.remove("sheet");
}

export { lutCss };
