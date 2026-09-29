// Phase 2: out-of-sample R^2 against Kyle's constant, by forecast horizon (development pairs).
import type { SiteData } from "../data/types";
import { el, linear, logScale, svgRoot, text, ticks, widthOf } from "../lib/svg";
import { bindTip, tooltipFor } from "../lib/tooltip";

type Dev = "BTCUSDT" | "BNBUSDT";
const SERIES: ["har" | "trailing_cal" | "seasonal_cal", string, string][] = [
  ["har", "var(--accent)", "Conditional (HAR)"],
  ["trailing_cal", "var(--series-2)", "Trailing λ, calibrated"],
  ["seasonal_cal", "var(--series-3)", "Time of day, calibrated"],
];

export function mountHorizons(container: HTMLElement, data: SiteData): (pair: Dev) => void {
  const tip = tooltipFor(container);
  let pair: Dev = "BTCUSDT";
  const yMax = Math.max(...Object.values(data.horizons).flatMap((h) => h.har)) * 100;
  const top = Math.ceil(yMax / 5) * 5;

  const draw = () => {
    const h = data.horizons[pair];
    const W = widthOf(container, 300, 560);
    const H = 250;
    const m = { l: 34, r: 12, t: 24, b: 34 };
    const svg = svgRoot(container, W, H);
    const x = logScale([h.minutes[0], h.minutes[h.minutes.length - 1]], [m.l + 8, W - m.r - 8]);
    const y = linear([0, top], [H - m.b, m.t]);
    for (const v of ticks(0, top, 4)) {
      el("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), stroke: v === 0 ? "var(--rule-strong)" : "var(--rule)" }, svg);
      text(svg, m.l - 6, y(v) + 4, `${v}%`, { "text-anchor": "end" });
    }
    text(svg, m.l - 6, m.t - 12, "out-of-sample R²", { class: "label-mid" });
    const lbl = (mins: number) => (mins >= 60 ? `${mins / 60} h` : `${mins} min`);
    h.minutes.forEach((mins) => text(svg, x(mins), H - m.b + 16, lbl(mins), { "text-anchor": "middle" }));
    for (const [key, color, name] of SERIES) {
      const vals = h[key].map((v) => v * 100);
      let d = "";
      vals.forEach((v, i) => { d += `${i ? "L" : "M"}${x(h.minutes[i])},${y(v)}`; });
      el("path", { d, fill: "none", stroke: color, "stroke-width": 2 }, svg);
      vals.forEach((v, i) => {
        const cx = x(h.minutes[i]);
        const cy = y(v);
        const dot = el("circle", { cx, cy, r: 4.5, fill: color, stroke: "var(--surface)", "stroke-width": 2 }, svg);
        bindTip(dot, tip, () => [cx, cy], [[`${name}, ${lbl(h.minutes[i])} ahead`], ["R²", `${v.toFixed(1)}%`]],
          `${pair}, ${name}, ${lbl(h.minutes[i])} ahead: R squared ${v.toFixed(1)} percent`);
      });
    }
  };
  new ResizeObserver(() => draw()).observe(container);
  draw();
  return (p) => { pair = p; draw(); };
}
