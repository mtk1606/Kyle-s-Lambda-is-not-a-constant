import "./styles.css";
import raw from "./data/site-data.json";
import type { SiteData } from "./data/types";
import { mountImpact } from "./charts/impact";
import { mountKyle } from "./charts/kyle";
import { mountDay, type DayLayers } from "./charts/day";
import { mountReliability } from "./charts/reliability";
import { mountHorizons } from "./charts/horizons";
import { mountExecution, mountIC, mountQuoting } from "./charts/pairs";
import { bugTable, constancyTable, executionTable, scorecardTable, venueTable } from "./tables";

const data = raw as unknown as SiteData;

const $ = <T extends Element = HTMLElement>(sel: string) => document.querySelector<T>(sel);

/** Toggle group: exactly one button pressed; calls back with its data value. */
function segmented(groupSel: string, attr: string, onChange: (value: string) => void): void {
  const group = $(groupSel);
  if (!group) return;
  const buttons = Array.from(group.querySelectorAll<HTMLButtonElement>(`button[${attr}]`));
  buttons.forEach((b) =>
    b.addEventListener("click", () => {
      buttons.forEach((o) => o.setAttribute("aria-pressed", String(o === b)));
      onChange(b.getAttribute(attr) ?? "");
    }),
  );
}

function mountAll(): void {
  const impact = $("[data-chart='impact']");
  const flow = $<HTMLInputElement>("#flow");
  if (impact && flow) mountImpact(impact, flow);

  const kyle = $("[data-chart='kyle']");
  if (kyle) mountKyle(kyle, data);

  const day = $("[data-chart='day']");
  if (day) {
    const layers: DayLayers = { forecast: true, band: true };
    const redraw = mountDay(day, data, layers);
    // independent on/off toggles for the two layers
    day.closest("figure")?.querySelectorAll<HTMLButtonElement>("button[data-layer]").forEach((b) =>
      b.addEventListener("click", () => {
        const key = b.dataset.layer as keyof DayLayers;
        layers[key] = !layers[key];
        b.setAttribute("aria-pressed", String(layers[key]));
        redraw();
      }),
    );
  }

  const rel = $("[data-chart='reliability']");
  if (rel) {
    const set = mountReliability(rel, data);
    segmented("#audit .seg", "data-est", (v) => set(v as "kyle_usd" | "sqrt_usd"));
  }

  const hz = $("[data-chart='horizons']");
  if (hz) {
    const set = mountHorizons(hz, data);
    segmented("#forecast .seg", "data-pair", (v) => set(v as "BTCUSDT" | "BNBUSDT"));
  }

  const ic = $("[data-chart='ic']");
  if (ic) mountIC(ic, data);
  const q = $("[data-chart='quoting']");
  if (q) mountQuoting(q, data);
  const ex = $("[data-chart='execution']");
  if (ex) mountExecution(ex, data);

  const tables: [string, (el: HTMLElement, d: SiteData) => void][] = [
    ["constancy", constancyTable], ["execution", executionTable], ["scorecard", scorecardTable],
    ["bugdiff", bugTable], ["venues", venueTable],
  ];
  for (const [key, fill] of tables) {
    const body = $(`[data-table='${key}']`);
    if (body) fill(body, data);
  }
  document.querySelectorAll<HTMLElement>("[data-bind='tests']").forEach((n) => (n.textContent = String(data.meta.tests)));
}

/** Mark the nav link of the section currently in view. */
function trackSections(): void {
  const links = Array.from(document.querySelectorAll<HTMLAnchorElement>(".topnav a"));
  const byId = new Map(links.map((a) => [a.hash.slice(1), a]));
  const sections = Array.from(byId.keys()).map((id) => document.getElementById(id)).filter((s): s is HTMLElement => !!s);
  if (!("IntersectionObserver" in window) || sections.length === 0) return;
  const io = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (!e.isIntersecting) continue;
        links.forEach((a) => a.removeAttribute("aria-current"));
        const link = byId.get(e.target.id);
        if (link) {
          link.setAttribute("aria-current", "true");
          // keep the active link visible in the horizontally scrolling nav without moving the page
          const nav = link.parentElement;
          if (nav && nav.scrollWidth > nav.clientWidth) {
            nav.scrollLeft = link.offsetLeft - nav.clientWidth / 2 + link.clientWidth / 2;
          }
        }
      }
    },
    { rootMargin: "-45% 0px -50% 0px" },
  );
  sections.forEach((s) => io.observe(s));
}

mountAll();
trackSections();
