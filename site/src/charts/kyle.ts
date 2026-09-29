// Kyle (1985) equilibrium lambda_n for N = 4, 20, 200 auctions, from the repo's solver.
import type { SiteData } from "../data/types";
import { el, linear, svgRoot, text, widthOf } from "../lib/svg";

export function mountKyle(container: HTMLElement, data: SiteData): void {
  const draw = () => {
    const W = widthOf(container, 280, 560);
    const H = 220;
    const m = { l: 36, r: 14, t: 14, b: 36 };
    const svg = svgRoot(container, W, H);
    const x = linear([0, 1], [m.l, W - m.r]);
    const y = linear([0.55, 1.08], [H - m.b, m.t]);
    for (const v of [0.6, 0.7, 0.8, 0.9, 1.0]) {
      el("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), stroke: "var(--rule)" }, svg);
      text(svg, m.l - 6, y(v) + 4, v.toFixed(1), { "text-anchor": "end" });
    }
    text(svg, x(0), H - m.b + 16, "open", { "text-anchor": "start" });
    text(svg, x(1), H - m.b + 16, "close", { "text-anchor": "end" });
    text(svg, (m.l + W - m.r) / 2, H - 4, "time within the trading period", { "text-anchor": "middle" });
    el("line", { x1: m.l, x2: W - m.r, y1: y(1), y2: y(1), stroke: "var(--ink)", "stroke-dasharray": "5 3", "stroke-width": 1.2 }, svg);
    text(svg, m.l + 6, y(1) - 7, "continuous limit: λ = σᵥ/σᵤ", { class: "label-mid" });
    const colors: Record<string, string> = { "4": "var(--series-3)", "20": "var(--series-2)", "200": "var(--accent)" };
    for (const N of ["4", "20", "200"] as const) {
      const lam = data.kyle_equilibrium[N];
      const n = lam.length;
      let d = `M${x(0)},${y(lam[0])}`;
      lam.forEach((_, i) => {
        d += ` H${x((i + 1) / n)}`;
        if (i + 1 < n) d += ` V${y(lam[i + 1])}`;
      });
      el("path", { d, fill: "none", stroke: colors[N], "stroke-width": 2 }, svg);
    }
  };
  new ResizeObserver(draw).observe(container);
  draw();
}
