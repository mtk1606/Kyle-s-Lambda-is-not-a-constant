// One BTCUSDT day: noisy 5-minute estimates, the constant, and the out-of-sample forecast.
import type { SiteData } from "../data/types";
import { el, linear, svgRoot, text, widthOf } from "../lib/svg";
import { tooltipFor } from "../lib/tooltip";

export interface DayLayers { forecast: boolean; band: boolean }

export function mountDay(container: HTMLElement, data: SiteData, layers: DayLayers): () => void {
  const day = data.headline_day;
  const tip = tooltipFor(container);
  const est = day.estimate;
  const fc = day.forecast;
  const finite = est.filter((v): v is number => v !== null).sort((a, b) => a - b);
  const p98 = finite[Math.floor(finite.length * 0.98)];
  const yMax = Math.ceil((p98 * 1.05) / 10) * 10;
  let idx = 144;

  const draw = () => {
    const W = widthOf(container, 300, 1000);
    const H = W < 520 ? 260 : 320;
    const m = { l: 40, r: 12, t: 22, b: 30 };
    const svg = svgRoot(container, W, H);
    const x = linear([0, 1440], [m.l, W - m.r]);
    const y = linear([0, yMax], [H - m.b, m.t]);
    for (let v = 0; v <= yMax; v += 20) {
      el("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), stroke: v === 0 ? "var(--rule-strong)" : "var(--rule)" }, svg);
      text(svg, m.l - 6, y(v) + 4, String(v), { "text-anchor": "end" });
    }
    const step = W < 520 ? 360 : 180;
    for (let t = 0; t <= 1440; t += step) {
      const hh = String(Math.floor(t / 60) % 24).padStart(2, "0");
      text(svg, x(t), H - m.b + 18, `${hh}:00`, { "text-anchor": t === 0 ? "start" : t === 1440 ? "end" : "middle" });
    }
    text(svg, m.l - 6, m.t - 10, "λ, bps per $1M of net flow", { class: "label-mid" });

    const minutes = day.minutes;
    if (layers.band) {
      let top = "";
      let bot = "";
      minutes.forEach((t, i) => {
        const f = fc[i];
        if (f !== null) top += `${top ? "L" : "M"}${x(t)},${y(Math.min(f, yMax))} `;
      });
      for (let i = minutes.length - 1; i >= 0; i--) if (fc[i] !== null) bot += `L${x(minutes[i])},${y(day.constant)} `;
      el("path", { d: `${top}${bot}Z`, fill: "var(--accent-soft)", stroke: "none" }, svg);
    }
    const dots = el("g", { "aria-hidden": "true" }, svg);
    minutes.forEach((t, i) => {
      const v = est[i];
      if (v === null) return;
      const clipped = v > yMax || v < 0;
      el("circle", {
        cx: x(t), cy: y(Math.max(0, Math.min(v, yMax))), r: 2.4,
        fill: clipped ? "none" : "var(--series-muted)", stroke: clipped ? "var(--series-muted)" : "none",
      }, dots);
    });
    el("line", { x1: m.l, x2: W - m.r, y1: y(day.constant), y2: y(day.constant), stroke: "var(--ink)", "stroke-width": 1.5, "stroke-dasharray": "6 4" }, svg);
    if (layers.forecast) {
      let d = "";
      minutes.forEach((t, i) => {
        const f = fc[i];
        if (f !== null) d += `${d ? "L" : "M"}${x(t)},${y(Math.min(f, yMax))}`;
      });
      el("path", { d, fill: "none", stroke: "var(--accent)", "stroke-width": 2, "stroke-linejoin": "round" }, svg);
    }

    // crosshair, for pointer and keyboard users
    const cross = el("line", { y1: m.t, y2: H - m.b, stroke: "var(--ink-3)", "stroke-width": 1, visibility: "hidden" }, svg);
    const hit = el("rect", {
      x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent", tabindex: 0, role: "slider",
      "aria-label": "Inspect a 5-minute window. Use the left and right arrow keys; hold Shift to move an hour.",
      "aria-valuemin": 0, "aria-valuemax": minutes.length - 1,
    }, svg);
    const show = (i: number) => {
      idx = Math.max(0, Math.min(minutes.length - 1, i));
      const t = minutes[idx];
      const hh = String(Math.floor(t / 60)).padStart(2, "0");
      const mm = String(t % 60).padStart(2, "0");
      cross.setAttribute("x1", String(x(t)));
      cross.setAttribute("x2", String(x(t)));
      cross.setAttribute("visibility", "visible");
      const e = est[idx];
      const f = fc[idx];
      hit.setAttribute("aria-valuenow", String(idx));
      hit.setAttribute("aria-valuetext", `${hh}:${mm} UTC: estimate ${e === null ? "missing" : e.toFixed(1)}, forecast ${f === null ? "missing" : f.toFixed(1)}`);
      tip.show(x(t), m.t + 8, [
        [`${hh}:${mm} UTC`],
        ["estimate", e === null ? "n/a" : e.toFixed(1)],
        ["forecast", f === null ? "n/a" : f.toFixed(1)],
        ["constant", day.constant.toFixed(1)],
      ]);
    };
    const hide = () => {
      tip.hide();
      cross.setAttribute("visibility", "hidden");
    };
    hit.addEventListener("pointermove", (ev) => {
      const r = svg.getBoundingClientRect();
      const px = ((ev.clientX - r.left) / r.width) * W;
      show(Math.round((((px - m.l) / (W - m.l - m.r)) * 1440) / 5));
    });
    hit.addEventListener("pointerleave", hide);
    hit.addEventListener("focus", () => show(idx));
    hit.addEventListener("blur", hide);
    hit.addEventListener("keydown", (ev) => {
      if (ev.key !== "ArrowRight" && ev.key !== "ArrowLeft") return;
      ev.preventDefault();
      show(idx + (ev.key === "ArrowRight" ? 1 : -1) * (ev.shiftKey ? 12 : 1));
    });
  };
  new ResizeObserver(() => draw()).observe(container);
  draw();
  return draw;
}
