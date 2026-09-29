// Tables filled from site-data.json. Each is also the accessible "table view" for a chart.
import type { Pair, SiteData } from "./data/types";

function cell(tag: "td" | "th", content: string | Node, cls = ""): HTMLTableCellElement {
  const c = document.createElement(tag);
  if (cls) c.className = cls;
  if (tag === "th") c.scope = "row";
  c.append(content);
  return c;
}

function status(kind: "supported" | "not" | "partial" | "insufficient" | "dev", label: string): HTMLSpanElement {
  const s = document.createElement("span");
  s.className = `status ${kind}`;
  s.textContent = label;
  return s;
}

function groupRow(label: string, span: number): HTMLTableRowElement {
  const tr = document.createElement("tr");
  tr.className = "group";
  const th = document.createElement("th");
  th.colSpan = span;
  th.scope = "colgroup";
  th.textContent = label;
  tr.append(th);
  return tr;
}

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
const minus = (s: string) => s.replace(/-/g, "−");

export function constancyTable(body: HTMLElement, data: SiteData): void {
  body.replaceChildren();
  for (const [label, set] of [["Development", data.meta.development], ["Held out", data.meta.holdout]] as const) {
    body.append(groupRow(label, 4));
    for (const p of set) {
      const s = data.pairs[p];
      const tr = document.createElement("tr");
      tr.append(cell("th", p), cell("td", label === "Development" ? "dev" : "holdout"),
        cell("td", `${pct(s.constancy_reject_1pct)} of ${s.days}`, "r"), cell("td", s.reliability_5m.toFixed(2), "r"));
      body.append(tr);
    }
  }
}

export function executionTable(body: HTMLElement, data: SiteData): void {
  body.replaceChildren();
  for (const p of [...data.meta.development, ...data.meta.holdout]) {
    const s = data.pairs[p];
    const e = s.execution.trailing;
    const holdout = s.set === "holdout";
    const thin = e.saving_pct_hi - e.saving_pct_lo > 40;
    const tr = document.createElement("tr");
    tr.append(
      cell("th", p),
      cell("td", holdout ? "holdout" : "dev"),
      cell("td", e.cost_har.toFixed(3), "r"),
      cell("td", e.cost_base.toFixed(3), "r"),
      cell("td", minus(`${e.diff_bps.toFixed(3)} [${e.ci_lo.toFixed(3)}, ${e.ci_hi.toFixed(3)}]`), "r"),
      cell("td", thin ? "n/a" : minus(`${e.saving_pct.toFixed(1)}%`), "r"),
      cell("td", !holdout ? status("dev", "dev only") : e.significant ? status("supported", "yes") : status(thin ? "insufficient" : "not", thin ? "too thin" : "no")),
    );
    body.append(tr);
  }
}

const PREDICTIONS: [string, string, (p: Pair, d: SiteData) => string][] = [
  ["P1", "Constant λ rejected on ≥10% of days", (p, d) => pct(d.pairs[p].constancy_reject_1pct)],
  ["P2", "5-min reliability below 0.6", (p, d) => d.pairs[p].reliability_5m.toFixed(2)],
  ["P3", "Beats both baselines, DM t > 2", (p, d) => `t ${Math.min(d.pairs[p].dm_vs_trailing, d.pairs[p].dm_vs_seasonal).toFixed(1)}`],
  ["P4", "Forecast λ predicts adverse selection", (p, d) => minus(`IC ${d.pairs[p].ic_forecast_as.toFixed(3)}`)],
  ["P5", "Forecast beats trailing λ at predicting λ", (p, d) => minus(`${d.pairs[p].ic_forecast_lambda.toFixed(2)} vs ${d.pairs[p].ic_trailing_lambda.toFixed(2)}`)],
  ["P6", "Forecast quote gate beats trailing gate", (p, d) => minus(`${d.scorecard[p].P6_diff_bps.toFixed(3)} bps`)],
  ["P7", "Forecast schedule cheaper than trailing", (p, d) => minus(`${d.scorecard[p].P7_diff_bps.toFixed(3)} bps`)],
];

export function scorecardTable(body: HTMLElement, data: SiteData): void {
  body.replaceChildren();
  for (const [key, label, value] of PREDICTIONS) {
    const tr = document.createElement("tr");
    const th = cell("th", `${key} `);
    const span = document.createElement("span");
    span.className = "muted";
    span.textContent = label;
    th.append(span);
    tr.append(th);
    for (const p of data.meta.holdout) {
      const pass = data.scorecard[p][key as "P1"];
      const td = cell("td", "", "r");
      const mark = document.createElement("span");
      mark.className = pass ? "pass" : "fail";
      mark.textContent = pass ? "✓ " : "✕ ";
      mark.setAttribute("aria-label", pass ? "pass" : "fail");
      td.append(mark, value(p, data));
      tr.append(td);
    }
    const v = data.verdict[key];
    const kind = v.holds ? "supported" : key === "P7" ? "partial" : "not";
    tr.append(cell("td", status(kind, `${v.holds ? "Holds" : "Fails"} ${v.holdout_pass}/4`)));
    body.append(tr);
  }
}

export function bugTable(body: HTMLElement, data: SiteData): void {
  body.replaceChildren();
  for (let k = 1; k <= 7; k++) {
    const v = data.verdict[`P${k}`];
    const changed = v.holdout_pass !== v.as_run_pass;
    const tr = document.createElement("tr");
    tr.append(
      cell("th", `P${k}`),
      cell("td", `${v.as_run_pass}/4`, "r"),
      cell("td", `${v.holdout_pass}/4${changed ? " (changed)" : ""}`, "r"),
      cell("td", status(v.holds ? "supported" : k === 7 ? "partial" : "not", v.holds === v.as_run_holds ? `${v.holds ? "Holds" : "Fails"} · unchanged` : "Verdict changed")),
    );
    body.append(tr);
  }
}

export function venueTable(body: HTMLElement, data: SiteData): void {
  body.replaceChildren();
  const rows: [keyof SiteData["venues"], string, string][] = [
    ["AMZN", "AMZN · Nasdaq · 2012-06-21", "one day"],
    ["CB_BTC", "Coinbase BTC-USDT L3", "insufficient"],
    ["CB_ETH", "Coinbase ETH-USDT L3", "insufficient"],
  ];
  for (const [key, name, st] of rows) {
    const v = data.venues[key];
    const tr = document.createElement("tr");
    const rel = `${v.reliability_5m.kyle_usd.toFixed(2)} / ${v.reliability_5m.ofi.toFixed(2)}`;
    const p = v.constancy_p.kyle_usd;
    tr.append(
      cell("th", `${name} (${v.hours} h)`),
      cell("td", v.trades.toLocaleString("en-US"), "r"),
      cell("td", minus(rel), "r"),
      cell("td", p <= 0.01 ? `rejected, p = ${p.toFixed(3)}` : `not rejected, p = ${p.toFixed(2)}`),
      cell("td", st === "one day" ? status("dev", "One day only") : status("insufficient", "Insufficient data")),
    );
    body.append(tr);
  }
}
