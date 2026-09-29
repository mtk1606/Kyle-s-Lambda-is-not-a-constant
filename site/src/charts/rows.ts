// Shared layout for the per-pair dot plots and bar charts: one row per pair, development
// pairs above held-out pairs, with a labelled divider between the two groups.
import type { Pair, SiteData } from "../data/types";
import { el, text } from "../lib/svg";

export interface RowLayout {
  y: (pair: Pair) => number;
  height: number;
  rowH: number;
}

export function rowLayout(svg: SVGSVGElement, data: SiteData, W: number, top: number, left: number, rowH = 34): RowLayout {
  const order: Pair[] = [...data.meta.development, ...data.meta.holdout];
  const gap = 26;
  const pos = new Map<Pair, number>();
  let yy = top + 18;
  text(svg, 0, top + 10, "DEVELOPMENT", { class: "label-mid", "font-size": 10, "letter-spacing": "0.08em" });
  order.forEach((p, i) => {
    if (i === data.meta.development.length) {
      yy += gap - rowH / 2;
      el("line", { x1: 0, x2: W, y1: yy - 12, y2: yy - 12, stroke: "var(--rule-strong)" }, svg);
      text(svg, 0, yy + 2, "HELD OUT", { class: "label-mid", "font-size": 10, "letter-spacing": "0.08em" });
      yy += 16;
    }
    pos.set(p, yy + rowH / 2);
    text(svg, 0, yy + rowH / 2 + 4, p, { class: "label-strong", "font-size": 12 });
    el("line", { x1: left, x2: W, y1: yy + rowH / 2, y2: yy + rowH / 2, stroke: "var(--rule)", "stroke-dasharray": "2 3" }, svg);
    yy += rowH;
  });
  return { y: (p) => pos.get(p) ?? 0, height: yy + 8, rowH };
}
