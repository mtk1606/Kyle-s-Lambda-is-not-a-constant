// Renders og/og.html to public/og.png (1200x630) with Playwright's Chromium.
//   NODE_PATH=$(npm root -g) node scripts/render_og.mjs
import { createRequire } from "module";
import { fileURLToPath } from "url";
import path from "path";
const require = createRequire(import.meta.url);
const { chromium } = require("playwright");
const here = path.dirname(fileURLToPath(import.meta.url));
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1200, height: 630 }, deviceScaleFactor: 1 });
await page.goto("file://" + path.join(here, "..", "og", "og.html"));
await page.screenshot({ path: path.join(here, "..", "public", "og.png") });
await browser.close();
console.log("wrote public/og.png");
