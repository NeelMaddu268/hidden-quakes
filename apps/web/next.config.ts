import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Workspace packages ship .ts source, not built JS.
  transpilePackages: ["@hq/visualization", "@hq/contracts"],
};

export default nextConfig;
