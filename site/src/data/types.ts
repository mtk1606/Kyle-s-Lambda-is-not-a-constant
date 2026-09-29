// Shape of site-data.json, written by scripts/export_site_data.py from results/*.csv.

export type Pair = "BTCUSDT" | "BNBUSDT" | "ETHBTC" | "LTCBTC" | "NEOUSDT" | "QTUMUSDT";

export interface ExecDiff {
  diff_bps: number; ci_lo: number; ci_hi: number;
  cost_har: number; cost_base: number;
  saving_pct: number; saving_pct_lo: number; saving_pct_hi: number;
  significant: boolean;
}

export interface PairStats {
  set: "development" | "holdout";
  reliability_5m: number;
  constancy_reject_1pct: number;
  constancy_reject_1pct_sqrt: number;
  days: number;
  r2: { last: number; trailing_cal: number; seasonal_cal: number; har: number };
  dm_vs_trailing: number; dm_vs_seasonal: number;
  ic_forecast_lambda: number; ic_trailing_lambda: number;
  ic_forecast_as: number; ic_forecast_as_t: number; ic_vol_as: number;
  quote_diff_bps: number; quote_ci: [number, number];
  quote_bps: { always: number; trailing: number; har: number; oracle: number };
  execution: { trailing: ExecDiff; twap: ExecDiff };
}

export interface ScoreRow {
  P1: boolean; P2: boolean; P3: boolean; P4: boolean; P5: boolean; P6: boolean; P7: boolean;
  P6_diff_bps: number; P7_diff_bps: number; P6_ci: string; P7_ci: string;
}

export interface Venue {
  reliability_5m: { kyle_usd: number; sqrt_usd: number; ofi: number };
  trades: number; hours: number;
  constancy_p: { kyle_usd: number; sqrt_usd: number; ofi: number };
  dispersion: number;
}

export interface SiteData {
  kyle_equilibrium: Record<"4" | "20" | "200", number[]>;
  meta: {
    trades: number; pairs: Pair[]; development: Pair[]; holdout: Pair[];
    period: [string, string]; tests: number;
    commits: { hash: string; time: string; subject: string }[];
  };
  btc_5m: { reliability: number; noise_share: number; rel_noise: number; frac_negative: number; median_lambda: number };
  reliability_btc: Record<"kyle_usd" | "sqrt_usd", Record<string, number>>;
  horizons: Record<"BTCUSDT" | "BNBUSDT", { minutes: number[]; har: number[]; trailing_cal: number[]; seasonal_cal: number[]; ceiling: number }>;
  pairs: Record<Pair, PairStats>;
  scorecard: Record<Pair, ScoreRow>;
  scorecard_as_run: Record<Pair, ScoreRow>;
  verdict: Record<string, { holdout_pass: number; as_run_pass: number; holds: boolean; as_run_holds: boolean }>;
  venues: Record<"AMZN" | "CB_BTC" | "CB_ETH", Venue>;
  headline_day: { date: string; window_sec: number; constant: number; minutes: number[]; estimate: (number | null)[]; forecast: (number | null)[] };
}
