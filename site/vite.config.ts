import { defineConfig } from "vite";

// Relative base so the built site works on GitHub Pages (project path), Netlify,
// Vercel or any static host without reconfiguration.
export default defineConfig({
  base: "./",
  build: { target: "es2020", outDir: "dist", assetsInlineLimit: 0, sourcemap: false },
});
