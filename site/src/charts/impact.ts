// Illustration only: the same net flow moves price by lambda x flow in two stylized markets.
import { el, linear, svgRoot, text, ticks, widthOf } from "../lib/svg";

const DEEP = 20;
const THIN = 80;

export function mountImpact(container: HTMLElement, input: HTMLInputElement): void {
  const out = document.getElementById("flow-out");
  const deep = document.getElementById("deep-move");
  const thin = document.getElementById("thin-move");

  const draw = () => {
    const flow = Number(input.value);
    const W = widthOf(container, 260, 560);
    const H = 190;
    const m = { l: 40, r: 48, t: 14, b: 26 };
    const svg = svgRoot(container, W, H);
    const x = linear([0, 10], [m.l, W - m.r]);
    const y = linear([-90, 90], [H - m.b, m.t]);
    for (const v of ticks(-80, 80, 4)) {
      el("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), stroke: v === 0 ? "var(--rule-strong)" : "var(--rule)" }, svg);
      text(svg, m.l - 6, y(v) + 4, `${v > 0 ? "+" : ""}${v}`, { "text-anchor": "end" });
    }
    text(svg, m.l, H - 6, "before", {});
    text(svg, x(4) + 4, H - 6, "net flow arrives", {});
    el("line", { x1: x(4), x2: x(4), y1: m.t, y2: H - m.b, stroke: "var(--rule-strong)", "stroke-dasharray": "3 3" }, svg);
    const series: [number, string, string][] = [[DEEP, "var(--accent)", "deep"], [THIN, "var(--series-2)", "thin"]];
    for (const [lam, color, name] of series) {
      const move = lam * flow;
      el("path", { d: `M${x(0)},${y(0)} H${x(4)} V${y(move)} H${x(10)}`, fill: "none", stroke: color, "stroke-width": 2.5, "stroke-linejoin": "round" }, svg);
      el("circle", { cx: x(10), cy: y(move), r: 4, fill: color }, svg);
      text(svg, x(10) + 8, y(move) + 4, name, { fill: color, "font-weight": 600 });
    }
    text(svg, m.l - 6, m.t - 3, "bps", { "text-anchor": "end" });
    const f = (v: number) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(1)} bps`;
    if (out) out.textContent = `${flow >= 0 ? "+" : "−"}$${Math.abs(flow).toFixed(2)}M`;
    if (deep) deep.textContent = f(DEEP * flow);
    if (thin) thin.textContent = f(THIN * flow);
  };
  input.addEventListener("input", draw);
  new ResizeObserver(draw).observe(container);
  draw();
}
