// Minimal SVG helpers: element factory, linear scales, tick generation.

const NS = "http://www.w3.org/2000/svg";

type Attrs = Record<string, string | number | undefined>;

export function el<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Attrs = {}, parent?: Element): SVGElementTagNameMap[K] {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== undefined) node.setAttribute(k, String(v));
  if (parent) parent.appendChild(node);
  return node;
}

export function text(parent: Element, x: number, y: number, content: string, attrs: Attrs = {}): SVGTextElement {
  const t = el("text", { x, y, ...attrs }, parent);
  t.textContent = content;
  return t;
}

export interface Scale {
  (v: number): number;
  domain: [number, number];
  range: [number, number];
}

export function linear(domain: [number, number], range: [number, number]): Scale {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const f = ((v: number) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0)) as Scale;
  f.domain = domain;
  f.range = range;
  return f;
}

export function logScale(domain: [number, number], range: [number, number]): Scale {
  const l = linear([Math.log(domain[0]), Math.log(domain[1])], range);
  const f = ((v: number) => l(Math.log(v))) as Scale;
  f.domain = domain;
  f.range = range;
  return f;
}

/** Round tick values covering [lo, hi], roughly `count` of them. */
export function ticks(lo: number, hi: number, count = 5): number[] {
  const span = hi - lo;
  if (span <= 0) return [lo];
  const raw = span / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= count) ?? 10 * mag;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

export function svgRoot(container: HTMLElement, width: number, height: number): SVGSVGElement {
  container.querySelector("svg")?.remove();
  const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, width, height, preserveAspectRatio: "xMinYMin meet" });
  svg.style.height = "auto";
  container.appendChild(svg);
  return svg;
}

export const fmt = {
  pct: (v: number, d = 1) => `${(v * 100).toFixed(d)}%`,
  num: (v: number, d = 2) => v.toFixed(d),
  signed: (v: number, d = 2) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(d)}`,
};

/** Width available to a chart container, clamped so charts stay legible. */
export function widthOf(container: HTMLElement, min = 300, max = 1000): number {
  return Math.max(min, Math.min(max, Math.floor(container.clientWidth || 640)));
}
