// Per-pair charts: information coefficients (Phase 3), quote gating (Phase 4a), execution (Phase 4b).
import type { Pair, SiteData } from "../data/types";
import { el, linear, svgRoot, text, ticks, widthOf } from "../lib/svg";
import { bindTip, tooltipFor } from "../lib/tooltip";
import { rowLayout } from "./rows";

function finish(svg: SVGSVGElement, W: number, H: number): void {
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("height", String(H));
}

function xAxis(svg: SVGSVGElement, x: (v: number) => number, vals: number[], top: number, bottom: number, fmt: (v: number) => string, zero = 0): void {
  for (const v of vals) {
    el("line", { x1: x(v), x2: x(v), y1: top, y2: bottom, stroke: v === zero ? "var(--ink-3)" : "var(--rule)", "stroke-width": v === zero ? 1.2 : 1 }, svg);
    text(svg, x(v), bottom + 14, fmt(v), { "text-anchor": "middle" });
  }
}

// ---------------------------------------------------------------- Phase 3: IC dot plot
const IC_SERIES: [keyof SiteData["pairs"][Pair], string, string, number][] = [
  ["ic_forecast_lambda", "var(--accent)", "Forecast λ → realized λ", -9],
  ["ic_trailing_lambda", "var(--series-muted)", "Trailing λ → realized λ", -3],
  ["ic_forecast_as", "var(--series-2)", "Forecast λ → maker adverse selection", 3],
  ["ic_vol_as", "var(--series-3)", "Trailing volatility → maker adverse selection", 9],
];

export function mountIC(container: HTMLElement, data: SiteData): void {
  const tip = tooltipFor(container);
  const draw = () => {
    const W = widthOf(container, 320, 1000);
    const left = W < 520 ? 84 : 110;
    const svg = svgRoot(container, W, 400);
    const x = linear([-0.1, 0.3], [left + 10, W - 14]);
    const rows = rowLayout(svg, data, W, 0, left, 36);
    xAxis(svg, x, ticks(-0.1, 0.3, 4), 12, rows.height - 6, (v) => v.toFixed(2).replace("-", "−"));
    for (const p of [...data.meta.development, ...data.meta.holdout]) {
      const s = data.pairs[p];
      for (const [key, color, name, dy] of IC_SERIES) {
        const v = s[key] as number;
        const cx = x(v);
        const cy = rows.y(p) + dy;
        const dot = el("circle", { cx, cy, r: 5, fill: color, stroke: "var(--surface)", "stroke-width": 1.5 }, svg);
        bindTip(dot, tip, () => [cx, cy], [[`${p} · ${name}`], ["IC", v.toFixed(3)]], `${p}, ${name}: IC ${v.toFixed(3)}`);
      }
    }
    text(svg, x(0) + 4, rows.height + 22, "mean daily rank IC →", { class: "label-mid" });
    finish(svg, W, rows.height + 28);
  };
  new ResizeObserver(() => draw()).observe(container);
  draw();
}

// ---------------------------------------------------------------- Phase 4a: quote gating vs always quoting
const Q_SERIES: ["trailing" | "har" | "oracle", string, string, number][] = [
  ["trailing", "var(--series-2)", "Gate on trailing λ", -7],
  ["har", "var(--accent)", "Gate on forecast λ", 0],
  ["oracle", "var(--series-muted)", "Gate on realized λ (oracle)", 7],
];

export function mountQuoting(container: HTMLElement, data: SiteData): void {
  const tip = tooltipFor(container);
  const draw = () => {
    const W = widthOf(container, 320, 560);
    const left = 84;
    const svg = svgRoot(container, W, 400);
    const x = linear([-2.0, 0.5], [left + 10, W - 14]);
    const rows = rowLayout(svg, data, W, 0, left, 32);
    xAxis(svg, x, [-2, -1.5, -1, -0.5, 0, 0.5], 12, rows.height - 6, (v) => (v === 0 ? "0" : v.toFixed(1).replace("-", "−")));
    for (const p of [...data.meta.development, ...data.meta.holdout]) {
      const q = data.pairs[p].quote_bps;
      for (const [key, color, name, dy] of Q_SERIES) {
        const v = q[key] - q.always;
        const cx = x(Math.max(-2, Math.min(0.5, v)));
        const cy = rows.y(p) + dy;
        const dot = el("circle", { cx, cy, r: 5, fill: color, stroke: "var(--surface)", "stroke-width": 1.5 }, svg);
        const sign = v >= 0 ? "+" : "−";
        bindTip(dot, tip, () => [cx, cy], [[`${p} · ${name}`], ["vs always quoting", `${sign}${Math.abs(v).toFixed(3)} bps`]],
          `${p}, ${name}: ${sign}${Math.abs(v).toFixed(3)} bps versus always quoting`);
      }
    }
    text(svg, x(0) - 4, rows.height + 22, "← worse than always quoting", { class: "label-mid", "text-anchor": "end" });
    text(svg, x(0) + 4, rows.height + 22, "better →", { class: "label-mid" });
    finish(svg, W, rows.height + 28);
  };
  new ResizeObserver(() => draw()).observe(container);
  draw();
}

// ---------------------------------------------------------------- Phase 4b: execution savings
export function mountExecution(container: HTMLElement, data: SiteData): void {
  const tip = tooltipFor(container);
  const draw = () => {
    const W = widthOf(container, 320, 1000);
    const left = W < 520 ? 84 : 110;
    const svg = svgRoot(container, W, 400);
    const x = linear([-2, 12], [left + 10, W - 14]);
    const rows = rowLayout(svg, data, W, 0, left, 34);
    xAxis(svg, x, [-2, 0, 2, 4, 6, 8, 10, 12], 12, rows.height - 6, (v) => `${v}%`.replace("-", "−"));
    for (const p of [...data.meta.development, ...data.meta.holdout]) {
      const e = data.pairs[p].execution.trailing;
      const cy = rows.y(p);
      if (e.saving_pct_hi - e.saving_pct_lo > 40) {
        const note = W < 560 ? "too thin to estimate" : `too thin to estimate (95% CI ${Math.round(e.saving_pct_lo)}% to ${Math.round(e.saving_pct_hi)}%)`;
        text(svg, x(0) + 8, cy + 4, note.replace("-", "−"), { class: "label-mid" });
        continue;
      }
      const v = e.saving_pct;
      const x0 = x(0);
      const x1 = x(v);
      const bar = el("rect", {
        x: Math.min(x0, x1), y: cy - 8, width: Math.max(2, Math.abs(x1 - x0)), height: 16, rx: 3,
        fill: e.significant ? "var(--accent)" : "var(--surface)", stroke: e.significant ? "none" : "var(--ink-3)", "stroke-width": 1.5,
      }, svg);
      el("line", { x1: x(e.saving_pct_lo), x2: x(e.saving_pct_hi), y1: cy, y2: cy, stroke: "var(--ink)", "stroke-width": 1.5 }, svg);
      for (const b of [e.saving_pct_lo, e.saving_pct_hi]) el("line", { x1: x(b), x2: x(b), y1: cy - 5, y2: cy + 5, stroke: "var(--ink)", "stroke-width": 1.5 }, svg);
      const lab = `${v.toFixed(1)}%${e.significant ? "" : " · CI crosses 0"}`;
      text(svg, x(e.saving_pct_hi) + 6, cy + 4, lab, { class: e.significant ? "label-strong" : "label-mid" });
      bindTip(bar, tip, () => [x1, cy - 8], [[p], ["saving", `${v.toFixed(2)}%`], ["95% CI", `${e.saving_pct_lo.toFixed(2)}% to ${e.saving_pct_hi.toFixed(2)}%`]],
        `${p}: saving ${v.toFixed(2)} percent, 95 percent interval ${e.saving_pct_lo.toFixed(2)} to ${e.saving_pct_hi.toFixed(2)}${e.significant ? ", significant" : ", not significant"}`);
    }
    text(svg, x(0) + 4, rows.height + 22, "cost saved vs trailing-λ schedule →", { class: "label-mid" });
    finish(svg, W, rows.height + 28);
  };
  new ResizeObserver(() => draw()).observe(container);
  draw();
}
