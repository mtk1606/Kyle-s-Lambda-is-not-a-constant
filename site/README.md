# Project website

A single-page research explainer for *Kyle's Lambda Is Not a Constant*: static HTML,
TypeScript and hand-drawn SVG charts. There is no framework and no chart library; the
only runtime dependencies are Google Fonts (with system fallbacks).

```
site/
  index.html                 all copy, in reading order; charts mount into [data-chart] slots
  src/main.ts                mounts charts and tables, nav highlighting
  src/charts/*.ts            one module per figure (impact demo, Kyle solver, one-day path,
                             reliability, horizons, IC, quoting, execution)
  src/tables.ts              scorecard, execution, constancy, venue and bug-comparison tables
  src/data/site-data.json    every number the charts and tables show (generated)
  src/styles.css             design tokens (light and dark) and layout
  scripts/export_site_data.py  results/*.csv  ->  src/data/site-data.json
  scripts/verify_claims.py     checks each number in the prose against results/, writes ACCURACY_AUDIT.md
  scripts/render_og.mjs        og/og.html  ->  public/og.png (1200x630)
  ACCURACY_AUDIT.md          every public claim mapped to its evidence
```

## Data flow

Nothing on the page is typed in by hand except prose. Chart and table values come from
`src/data/site-data.json`, which `scripts/export_site_data.py` builds from the committed
`results/*.csv`, git history and `pytest --collect-only`. The one-day chart also reads
`data/windows` and `data/preds`, which the research pipeline rebuilds (they are not
committed). Numbers that appear in the prose are checked by `scripts/verify_claims.py`,
which fails if a phrase is missing from the page or no longer matches the results.

## Commands

From `site/`:

```bash
npm install
npm run data        # regenerate src/data/site-data.json (needs the research pipeline's data/ for the day chart)
npm run verify      # recompute every numeric claim and rewrite ACCURACY_AUDIT.md; exits 1 on a mismatch
npm run dev         # local server with hot reload
npm run build       # typecheck + production build into dist/
npm run preview     # serve dist/
NODE_PATH=$(npm root -g) node scripts/render_og.mjs   # re-render the social preview (needs Playwright)
```

`site-data.json` is committed, so `npm run build` works on a fresh clone without the
research data.

## Deployment

`.github/workflows/pages.yml` builds `site/` and publishes `site/dist` to GitHub Pages on
every push to the default branch that touches `site/`. To turn it on: repository
Settings → Pages → Source: **GitHub Actions**.

The build uses a relative base (`./`), so `dist/` also works unchanged on Netlify,
Vercel, Cloudflare Pages or any static host.

**Canonical URL.** `index.html` uses
`https://mtk1606.github.io/Kyle-s-Lambda-is-not-a-constant/` for the canonical link and
the Open Graph image. That is a placeholder until the site is live: if you deploy
elsewhere or add a custom domain, update the four URLs in the `<head>`. LinkedIn needs
the absolute `og:image` URL to resolve.

## Accessibility and performance

* Semantic landmarks, one `h1`, a skip link, and a heading order that follows the argument.
* Every chart has a text alternative: an `aria-label` summary, focusable marks with the
  values, and, for the scorecard and execution results, a real table.
* The one-day chart is keyboard-operable (arrow keys; Shift for an hour).
* Status is never carried by color alone; each label has an icon and text.
* Light and dark themes are both designed, and both follow `prefers-color-scheme`.
* The production build is about 13 KB of gzipped JS and 5 KB of gzipped CSS.
