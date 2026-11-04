// Canvas heatmaps with step-aware overlays (current position, causal future dimming),
// row labels grouped by circuit, hover readout and click-to-navigate.
import { matrixBitmap, setupCanvas, lutCss } from "./render.js";

const GUTTER = 128; // px reserved for row labels
// Touch screens: cells at least this big (the map scrolls sideways inside its panel).
const TOUCH_COL = 8, TOUCH_ROW = 5, TOUCH_MAX_W = 4000;
export const isTouch = () => matchMedia("(pointer: coarse)").matches;

// spec: { key, rows, cols, get(r,c) -> raw, norm(raw) -> [0,1], lut, rowLabels[], groups[{row,label}],
//         colLabel(c), valueText(r,c), onClick(r,c), square }
// view: { q } current query position (column, or row for attention)
export function drawHeatmap(canvas, spec, view) {
  const wrap = canvas.parentElement;
  const touch = isTouch();
  let cssW = Math.max(240, wrap.clientWidth - 2);
  if (touch && spec.cols) cssW = Math.max(cssW, GUTTER + Math.min(TOUCH_MAX_W, spec.cols * TOUCH_COL));
  const plotW = cssW - GUTTER;
  const rowH = spec.square ? plotW / spec.cols
    : Math.max(touch ? TOUCH_ROW : 3, Math.min(12, 560 / Math.max(1, spec.rows)));
  const plotH = spec.square ? plotW : rowH * spec.rows;
  const cssH = Math.max(40, plotH + 18);
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

  // causal view: everything after the current step has not been computed yet
  const q = view.q;
  if (q !== undefined) {
    ctx.fillStyle = "rgba(13,15,18,0.72)";
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

  // row labels and circuit group separators
  ctx.font = "10px ui-monospace, Menlo, monospace";
  ctx.textBaseline = "middle";
  if (rowH >= 8 && spec.rowLabels) {
    ctx.fillStyle = "#aab2c0";
    spec.rowLabels.forEach((l, r) => ctx.fillText(l.slice(0, 19), 4, r * rowH + rowH / 2));
  }
  (spec.groups || []).forEach((g) => {
    ctx.strokeStyle = "#3a4150";
    ctx.beginPath();
    ctx.moveTo(0, g.row * rowH + 0.5);
    ctx.lineTo(cssW, g.row * rowH + 0.5);
    ctx.stroke();
    if (rowH < 8) {
      ctx.fillStyle = "#f2a541";
      ctx.fillText(g.label, 4, g.row * rowH + 7);
    }
  });
  // column ticks
  ctx.fillStyle = "#7d8696";
  ctx.textBaseline = "top";
  const step = Math.max(1, Math.ceil(30 / colW));
  for (let c = 0; c < spec.cols; c += step) ctx.fillText(String(c), GUTTER + c * colW, plotH + 3);

  canvas._geom = { spec, rowH, colW, plotW, plotH };
}

export function attachHeatmapEvents(canvas, tooltip) {
  const cell = (e) => {
    const g = canvas._geom;
    if (!g) return null;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left - GUTTER;
    const y = e.clientY - rect.top;
    if (x < 0 || y < 0 || x >= g.plotW || y >= g.plotH) return null;
    return [Math.floor(y / g.rowH), Math.floor(x / g.colW)];
  };
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
    tooltip.textContent = spec.valueText(r, c);
    tooltip.style.left = `${e.clientX + 14}px`;
    tooltip.style.top = `${e.clientY + 10}px`;
  });
  canvas.addEventListener("mouseleave", () => { if (!touching()) tooltip.hidden = true; });
  canvas.addEventListener("click", (e) => {
    const rc = cell(e);
    if (touching()) {
      if (!rc) { tapped = null; hideSheet(tooltip); return; }
      const key = `${canvas._geom.spec.key}:${rc}`;
      if (tapped !== key) {
        tapped = key;
        showSheet(tooltip, canvas._geom.spec.valueText(...rc) + (canvas._geom.spec.onClick ? "\n\n(tap the same cell again to jump there)" : ""));
        return;
      }
      tapped = null;
      hideSheet(tooltip);
    }
    if (rc && canvas._geom.spec.onClick) canvas._geom.spec.onClick(...rc);
  });
}

/** Touch tooltip: docked above the bottom step bar instead of following a cursor. */
export function showSheet(tooltip, text) {
  tooltip.textContent = text;
  tooltip.classList.add("sheet");
  tooltip.style.left = tooltip.style.top = "";
  tooltip.hidden = false;
}

export function hideSheet(tooltip) {
  tooltip.hidden = true;
  tooltip.classList.remove("sheet");
}

export { lutCss };
