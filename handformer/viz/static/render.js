// Rendering helpers: float16 decoding, colormaps, and matrix → canvas.

// ---- float16 (IEEE 754 half, little-endian) → Float32 lookup table
const F16 = (() => {
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1;
    const e = (h >> 10) & 0x1f;
    const f = h & 0x3ff;
    let v;
    if (e === 0) v = f * 2 ** -24;
    else if (e === 31) v = f ? NaN : Infinity;
    else v = (1 + f / 1024) * 2 ** (e - 15);
    t[h] = s * v;
  }
  return t;
})();

export function decodeF16(buffer) {
  const u = new Uint16Array(buffer);
  const out = new Float32Array(u.length);
  for (let i = 0; i < u.length; i++) out[i] = F16[u[i]];
  return out;
}

// ---- colormaps: 256-entry RGB LUTs built by piecewise-linear interpolation of control points
function buildLut(stops) {
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const x = i / 255;
    let k = 0;
    while (k < stops.length - 2 && x > stops[k + 1][0]) k++;
    const [x0, c0] = stops[k];
    const [x1, c1] = stops[k + 1];
    const u = x1 > x0 ? (x - x0) / (x1 - x0) : 0;
    for (let c = 0; c < 3; c++) lut[i * 3 + c] = c0[c] + (c1[c] - c0[c]) * u;
  }
  return lut;
}

// Magma-like sequential map (black → purple → orange → pale yellow)
export const SEQ = buildLut([
  [0.0, [0, 0, 4]], [0.15, [28, 16, 68]], [0.3, [79, 18, 123]], [0.45, [129, 37, 129]],
  [0.6, [181, 54, 122]], [0.75, [229, 80, 100]], [0.87, [251, 135, 97]], [1.0, [252, 253, 191]],
]);

// Dark-centred diverging map (blue ← near-black → orange) for signed weights
export const DIV = buildLut([
  [0.0, [94, 200, 229]], [0.25, [40, 100, 160]], [0.5, [16, 18, 22]], [0.75, [170, 80, 30]], [1.0, [242, 165, 65]],
]);

export function lutCss(lut, x) {
  const i = Math.max(0, Math.min(255, Math.round(x * 255))) * 3;
  return `rgb(${lut[i]},${lut[i + 1]},${lut[i + 2]})`;
}

// Paint a rows×cols matrix (row-major, values via `get(r, c)` mapped to [0,1] by `norm`) into an
// offscreen canvas at 1 px per cell. Callers then scale it onto a display canvas.
export function matrixBitmap(rows, cols, get, norm, lut) {
  const off = document.createElement("canvas");
  off.width = cols;
  off.height = rows;
  const ctx = off.getContext("2d");
  const img = ctx.createImageData(cols, rows);
  const px = img.data;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x = norm(get(r, c));
      const li = Math.max(0, Math.min(255, (x * 255) | 0)) * 3;
      const o = (r * cols + c) * 4;
      px[o] = lut[li];
      px[o + 1] = lut[li + 1];
      px[o + 2] = lut[li + 2];
      px[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return off;
}

// Size a canvas for CSS pixels × devicePixelRatio and return a context in CSS-pixel units.
export function setupCanvas(canvas, cssW, cssH) {
  const dpr = window.devicePixelRatio || 1;
  canvas.style.width = `${cssW}px`;
  canvas.style.height = `${cssH}px`;
  canvas.width = Math.max(1, Math.round(cssW * dpr));
  canvas.height = Math.max(1, Math.round(cssH * dpr));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

// Token display. The server sends each token's text with word spacing made explicit as spaces
// (GPT-2: leading space, e.g. " room"; GPT-1: trailing space after a word-final piece, e.g. "room ").
//   text: label with surrounding spaces stripped and control characters made visible
//   vis:  like text, but spaces shown as "·" so " room" and "room" stay distinguishable
//   lead / trail: whether the token starts / ends with a space (used for chip spacing)
const CONTROL = { "\n": "↵", "\r": "␍", "\t": "⇥" };
const showControls = (s) => s.replace(/[\n\r\t]/g, (c) => CONTROL[c]);

export function fmtToken(display) {
  const lead = /^ /.test(display);
  const trail = / $/.test(display);
  const core = showControls(display.replace(/^ +| +$/g, ""));
  const text = core || (display.length ? "␣".repeat(display.length) : "∅");
  const vis = showControls(display).replace(/ /g, "·") || "∅";
  return { text, vis, lead, trail };
}

export function fmtPct(p) {
  if (p >= 0.1) return `${(p * 100).toFixed(1)}%`;
  if (p >= 0.001) return `${(p * 100).toFixed(2)}%`;
  return p.toExponential(1);
}

// Max-pool a t×t (lower-triangular) attention matrix down to at most `target` cells per side so that
// thin structures (diagonals, single-token spikes) survive downsampling. Returns a bitmap canvas.
export function attentionBitmap(t, get, target, norm, lut) {
  const s = Math.max(1, Math.min(t, Math.round(target)));
  if (s === t) return matrixBitmap(t, t, (r, c) => (c <= r ? get(r, c) : 0), norm, lut);
  const pooled = new Float32Array(s * s);
  const scale = s / t;
  for (let r = 0; r < t; r++) {
    const pr = Math.min(s - 1, (r * scale) | 0) * s;
    for (let c = 0; c <= r; c++) {
      const i = pr + Math.min(s - 1, (c * scale) | 0);
      const v = get(r, c);
      if (v > pooled[i]) pooled[i] = v;
    }
  }
  return matrixBitmap(s, s, (r, c) => pooled[r * s + c], norm, lut);
}
