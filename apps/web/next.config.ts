import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Static export (API-03): `next build` writes apps/web/out/, a plain folder of HTML/JS/CSS plus
  // everything in public/ (the data bundles and the terrain tiles). Any static file server hosts it:
  // Vercel on every `main` merge, `scripts/serve-offline.sh` on stage with Wi-Fi off. The providers
  // fetch site-relative `/data/<mode>/...` at runtime, so no server-side code is involved.
  output: "export",
  // The app is one route; `/` -> `out/index.html`. Extra routes would export as `<route>/index.html`,
  // which plain file servers (python http.server, nginx) resolve without rewrite rules.
  trailingSlash: true,
  // The default image loader needs a server; nothing uses next/image today, but keep the export honest.
  images: { unoptimized: true },
  // Workspace packages ship .ts source, not built JS.
  transpilePackages: ["@hq/visualization", "@hq/contracts"],
};

export default nextConfig;
