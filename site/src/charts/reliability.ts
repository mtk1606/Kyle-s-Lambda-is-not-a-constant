// Phase 1: split-half reliability of a rolling lambda by window length (BTCUSDT, 1 s bars).
import type { SiteData } from "../data/types";
import { el, linear, svgRoot, text, widthOf } from "../lib/svg";
import { bindTip, tooltipFor } from "../lib/tooltip";

const WINDOWS: [string, string][] = [["60", "1 min"], ["300", "5 min"], ["900", "15 min"], ["3600", "1 h"], ["14400", "4 h"], ["86400", "1 day"]];

export function mountReliability(container: HTMLElement, data: SiteData): (est: "kyle_usd" | "sqrt_usd") => void {
  const tip = tooltipFor(container);
  let current: "kyle_usd" | "sqrt_usd" = "kyle_usd";

  const draw = () => {
    const rel = data.reliability_btc[current];
    const W = widthOf(container, 300, 560);
    const H = 250;
    const m = { l: 34, r: 8, t: 24, b: 34 };
    const svg = svgRoot(container, W, H);
    const y = linear([0, 1], [H - m.b, m.t]);
    const band = (W - m.l - m.r) / WINDOWS.length;
    const bw = Math.min(46, band * 0.62);
    for (const v of [0, 0.25, 0.5, 0.75, 1]) {
      el("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), stroke: v === 0 ? "var(--rule-strong)" : "var(--rule)" }, svg);
      text(svg, m.l - 6, y(v) + 4, `${Math.round(v * 100)}%`, { "text-anchor": "end" });
    }
    text(svg, m.l - 6, m.t - 12, "real share of variation", { class: "label-mid" });
    WINDOWS.forEach(([key, label], i) => {
      const v = rel[key];
      const cx = m.l + band * (i + 0.5);
      const hl = key === "300";
      // bar anchored to the baseline, with a small rounded data end
      const top = y(v);
      const bar = el("path", {
        d: `M${cx - bw / 2},${y(0)} V${top + 3} Q${cx - bw / 2},${top} ${cx - bw / 2 + 3},${top} H${cx + bw / 2 - 3} Q${cx + bw / 2},${top} ${cx + bw / 2},${top + 3} V${y(0)} Z`,
        fill: hl ? "var(--accent)" : "var(--series-muted)",
      }, svg);
      text(svg, cx, top - 6, v.toFixed(2), { "text-anchor": "middle", class: hl ? "label-strong" : "label-mid" });
      text(svg, cx, H - m.b + 16, label, { "text-anchor": "middle", class: hl ? "label-strong" : "" });
      bindTip(bar, tip, () => [cx, top - 4], [[`${label} windows`], ["reliability", v.toFixed(2)], ["noise share", `${Math.round((1 - v) * 100)}%`]],
        `${label} windows: reliability ${v.toFixed(2)}, ${Math.round((1 - v) * 100)}% noise`);
    });
  };
  new ResizeObserver(() => draw()).observe(container);
  draw();
  return (est) => { current = est; draw(); };
}
